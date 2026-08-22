"""
Now that we have prepared our data for our ml models there are some important information to take into account before we start training our models.

In standard machine learning (like predicting if a picture is a cat or a dog), you randomly shuffle your dataset before splitting it into Training and Testing sets.

If you do this in quantitative finance, your model is immediately invalid. Because your data is a time-series, randomly shuffling the rows means you might 
accidentally place data from 12:05 PM in the Training set, and data from 12:01 PM in the Test set. Your model would learn how to predict the past by looking 
at the future. This is called "look-ahead leakage". 

In a live market, you never have access to future data, so a model trained this way will show a 99% win rate on your computer and immediately lose all your money in real life.

To prevent this, we execute a strict chronological slice:

   - Training Set (First 70%): The model learns the historical market behaviors and target correlations.

   - Validation Set (Middle 15%): We use this strictly to tune our hyperparameters (like tree depth or learning rate) without touching the final test data.

   - Test Set (Final 15%): The absolute out-of-sample "blind" test. The model has never seen this timeframe, simulating a true live trading deployment.
"""

import os
import polars as pl
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.preprocessing import StandardScaler

# Silence false Apple Accelerate BLAS CPU flags in NumPy 2.x
np.seterr(all='ignore')

def load_and_split_data():
    print("Loading serialized feature matrix...")
    
    # 1. Define the path to the Parquet file
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/engineered_features.parquet"))
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Parquet file not found at {data_path}. Did you run generate_signals.py?")
        
    # Load the high-speed parquet file
    df = pl.read_parquet(data_path)
    
    # 2. Calculate the exact chronological index boundaries
    total_rows = df.height
    train_end_idx = int(total_rows * 0.70)
    val_end_idx = int(total_rows * 0.85)
    
    # 3. Execute strict chronological slices (No randomized shuffling)
    train_df = df.slice(0, train_end_idx)
    val_df = df.slice(train_end_idx, val_end_idx - train_end_idx)
    test_df = df.slice(val_end_idx, total_rows - val_end_idx)
    
    return train_df, val_df, test_df

# building and training baseline model - logistic regression model since this is the most simplest and basic model to start with
def train_baseline_model(train_df: pl.DataFrame, test_df: pl.DataFrame):
    print("\n--- INITIATING BASELINE MODEL TRAINING ---")
    
    # using the level 1 metrics only and only for the baseline model 
    features = ["obi_0.2"]
    target = "target_label"
    
    print(f"Features mapped: {features}")
    print(f"Target mapped: {target}")
    
    # Scikit-learn requires Numpy arrays, so we extract them from Polars
    # .ravel() flattens the target array into the correct 1D shape
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select(target).to_numpy().ravel()
    
    X_test = test_df.select(features).to_numpy()
    y_test = test_df.select(target).to_numpy().ravel()
    
    # intializing the logistic regression model 
    # the multi_class wil be multinomial since we have 3 classes to predict (buy, sell, hold)
    
    baseline_model = LogisticRegression(solver='lbfgs', random_state=42) 
    
    # fit(train) the model using only the hstorical training data 
    print("Fitting Logistic Regression to Training data")
    baseline_model.fit(X_train, y_train)
    
    # predicitng 30s into the future on the unseen test data
    print("Predicting on unseen out-of-sample test data")
    y_pred = baseline_model.predict(X_test)
    
    # accuracy check
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- BASELINE MODEL PERFORMANCE (OUT-OF-SAMPLE) ---")
    print(f"Absolute Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    # zero_division=0 prevents warnings if the simple model fails to predict a certain class
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--------------------------------------------------\n")
    

# building the more advanced model XGBoost 
def train_xgboost_model(train_df: pl.DataFrame, val_df: pl.DataFrame,test_df: pl.DataFrame):
    print("\n--- INITIATING ADVANCED XGBOOST MODEL CLASSIFIER ---")
    
    features = [
        "obi_0.2", "obi_1.0", "obi_2.0", "obi_3.0", "obi_5.0",
        "obi_global_weighted", "liquidity_width",
        "obi_velocity_5s", "obi_velocity_10s", "obi_velocity_30s",
        "obi_volatility_10s"
    ]
    
    target = "target_label"
    print(f"Feeding {len(features)} engineered features to XGBoost")
    
    # converting the polars dataframes to numpy arrays
    X_train = train_df.select(features).to_numpy()
    y_train_raw = train_df.select(target).to_numpy().ravel()
    
    X_val = val_df.select(features).to_numpy()
    y_val_raw = val_df.select(target).to_numpy().ravel()
    
    X_test = test_df.select(features).to_numpy()
    y_test_raw = test_df.select(target).to_numpy().ravel()

    # XGBoost strictly requires classes to be mapped to [0, 1, 2] instead of [-1, 0, 1]
    # we use LabelEncoder to handle this mapping cleanly
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    
    # initialize the XGBoost classifier with hyperparameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,   #max number of trees to build
        learning_rate=0.05, #step size shrinkage to prevent overfitting
        max_depth=4,        #maximum depth of a tree (for complexity limit)
        objective='multi:softmax', #multi-class classification
        num_class=3,        #number of target classes 
        eval_metric ='mlogloss', #evaluation metric
        early_stopping_rounds=10, #stop validation see no improvement in 10 rounds
        random_state=42
    )
    
    # fit the model passing the validation set to monitor for early stopping
    print("Trainign XGBoost and monitoring validation set to prevent overfitting...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False # set to True if you want see round-by-round progress 
    )
    
    # predicting on out-of smaple unseen data
    print("Predicting on unseen out-of-sample test data")
    y_pred_encoded = xgb_model.predict(X_test)
    
    # translate the encoded predictions back to original labels
    y_pred = le.inverse_transform(y_pred_encoded)
    
    # accuracy check
    accuracy = accuracy_score(y_test_raw, y_pred)
    
    print("\n--- XGBOOST MODEL PERFORMANCE (OUT-OF-SAMPLE) ---")
    print(f"Absolute Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test_raw, y_pred, zero_division=0))
    print("-------------------------------------------------\n")

    # --- Feature Importance Extraction ---
    print("Extracting Feature Importance Matrix...")
    importances = xgb_model.feature_importances_
    
    # create a visual chart
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='steelblue')
    plt.xlabel("Relative Importance Weight")
    plt.ylabel("Microstructure Features")
    plt.title("XGBoost Feature Importance: Which signals drive the Alpha?")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    
    # save chart to the data folder
    chart_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/feature_importance.png"))
    plt.savefig(chart_path)
    print(f"Saved feature importance chart to: {chart_path}")

def evaluate_naive_benchmark(test_df: pl.DataFrame):
    print("\n--- INITIATING NAIVE BENCHMARK EVALUATION ---")

    # extract the raw level 1 OBI and true labels 
    obi_l1 = test_df.select("obi_0.2").to_numpy().ravel()
    y_test = test_df.select("target_label").to_numpy().ravel()

    # apply the rule-based hueristic thresholding 
    # if OBI > 0.5 -> 1, If OBI < -0.5 -> -1, Else -> 0
    y_pred = [] 
    for val in obi_l1:
        if val > 0.5:
            y_pred.append(1)
        elif val < -0.5:
            y_pred.append(-1)
        else:
            y_pred.append(0)

    # evaluat the performance against the true market outcomes 
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- NAIVE BENCHMARK PERFORMANCE (OUT-OF-SAMPLE) ---")
    print(f"Absolute Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--------------------------------------------------\n")

# during the training of the baseline model we trained the multinomial logistic regression on 1 feature (obi_0.2)
# but now that we have trained an advanced xgboost model (xgbosst categorical classifier) which is a complex model, 
# it takes more CPU/GPU microseconds to evaluate than a linear model. So the objective for this task is to evaluate 
# the performance vs latency trade-off. 
# for this we will now train the multinomial logistic regression on all the 11 features. 

def train_full_logistic_model(train_df: pl.DataFrame, test_df: pl.DataFrame):
    print("\n--- INITIATING STATISTICAL BENCHMARK (FULL FEATURE LOGISTIC REGRESSION) ---")

    #1. Mapping the complete feature matrix (all 11 of them)
    features = [
        "obi_0.2", "obi_1.0", "obi_2.0", "obi_3.0", "obi_5.0",
        "obi_global_weighted", "liquidity_width",
        "obi_velocity_5s", "obi_velocity_10s", "obi_velocity_30s",
        "obi_volatility_10s"
    ]

    target = "target_label"

    print(f"Feeding all {len(features)} engineered features to Multinomial Logistic Regression")

    #2. Extracting arrays from Polars 
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select(target).to_numpy().ravel()

    X_test = test_df.select(features).to_numpy()
    y_test = test_df.select(target).to_numpy().ravel()

    #3. Convert any lingering NaN, +inf, or -inf into 0.0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    #4. Standarize feature scales (Zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) #Note - fit on train, transform test to avoid data leakage 

    #5. Clipping extreme Z-score outliers to [-5.0, 5.0]

    X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
    X_test_scaled = np.clip(X_test_scaled, -5.0, 5.0)

    #4. Fit the Multinomial Linear Regression 
    full_log_model = LogisticRegression(solver='liblinear', C=0.1 ,max_iter=1000, random_state=42) 
    
    # the C=0.1 is to penalize multicollinearity and stop weight explosion
    # before when i was running the function without the C=0.1 parameter i was getting this error : 
    # RuntimeWarning: divide by zero encountered in matmul
    # this happened because of the multicollinearity between the 11 features, in other words 6 out of 11 features 
    # has a high collinearity 
    # so in order to avoid getting this error i introduced the C=0.1 parameter which penalize multicollinearity 
    # and stop weight explosion. 

    """
    the following is what gemini has to give as an explanation for the above mentioned RuntimeWarning: 
    
    1. Collinear Features: 6 of your 11 features (obi_0.2, obi_1.0, obi_2.0, obi_3.0, obi_5.0, obi_global_weighted) 
    measure the exact same market force—buying vs. selling pressure—at slightly different depth layers. 
    They are highly correlated.
    
    2. Weight Explosion: When LogisticRegression with the default setting (C=1.0) attempts to fit highly correlated 
    features, the un-regularized optimizer (L-BFGS) pits the features against each other. 
    It assigns massive positive weights to one level (+10,000) and massive negative weights to another (−10,000).

    3. Softmax Overflow: During matrix multiplication (X @ weights.T), multiplying inputs by these exploded weight 
    vectors creates numbers like +1,000. When scikit-learn exponentiates these in the softmax function (e^1000), it 
    exceeds 64-bit floating-point memory limits, triggering the overflow and divide by zero warnings.
    """
    # so after adding this C=0.1 parameter the absolute accuracy changed from 29.55% to 30.50% for this particular model
    # but the runtimewarning for the matmul still presists. 
    #asking gemini this is what i can understand 

    """
    in high-frequency order book data, market spikes occur(eg. liquidity_width or obi_velocity suddenly jumping during a 
    volatility burst)

    when standardscaler() standardizes this sudden spike, it creates massive z-scores (eg. +500 or +1000) 
    standard deviations 

    finally whne LogisticRegression evaluates X @ weights.T, multiplying a +500 z-score by weight vectors casues 
    sk-learn's inital internal matrix math (e^500xW) to overflow the 64-bit float limit.

    so the suggested solution to this fix is to prevent extreme market spikes from exploding matrix multiplication, 
    where clipping the sclaed z-scores to a clean range like [-5.0,5.0] (capping the features within 5 
    standard deviations) and then apply np.nan_to_num after scaling
    """

    # so after doing the change it didnt prevent the warning but what i could see as an improvement was the fact 
    # that the precision score has changed in the -1.0 and weighted avg from the last time i ran the code.
    # again asking gemini this is what it said : 

    """ 
    You are running Python 3.13 with NumPy 2.x on macOS.

    Intermediate Trial Steps: During full_log_model.fit(), the L-BFGS optimization algorithm tests candidate weight 
    vectors to find the mathematical minimum.

    NumPy 2.0 BLAS Flag: During some of these temporary trial steps, intermediate dot products (X @ weights.T) 
    briefly produce large values. NumPy 2.0 on macOS (using Apple Accelerate BLAS) detects these trial values 
    and emits a RuntimeWarning.

    Normal Convergence: L-BFGS immediately rejects those trial steps, corrects course, and converges cleanly to 
    the final model (which is why your precision and accuracy updated successfully to 30.50%!).

    These are harmless internal solver trial warnings. In Python/scikit-learn, it is standard practice to suppress 
    intermediate optimizer warnings around .fit().
    """

    # when i asked i want to try and fix these warning this is what it said : 

    """
    Why lbfgs Keeps Warning and How to Fix It 100%

    solver='lbfgs' in scikit-learn 1.5+ routes through an internal Python module named: 
    sklearn.linear_model._linear_loss.py. 
    
    During optimization, it uses NumPy's @ (matmul) matrix operator in pure Python, which triggers RuntimeWarnings 
    on Python 3.13 / NumPy 2.0 during line-search steps.

    The Permanent Solution: Switch to a C-Compiled Solver (liblinear or saga)
    By changing the solver to solver='liblinear' (or solver='saga'):

    1. Optimization is handled by C/C++ compiled binaries (LIBLINEAR / SAGA).
    2. It completely bypasses _linear_loss.py and pure Python NumPy matrix operations.
    3. Result: 100% warning-free execution with high numerical precision and speed!
    """

    # so i got the _linear_loss.py warning fixed but then the following warning persisted which came from the extmath.py
    """
    /opt/anaconda3/envs/conda-env/lib/python3.13/site-packages/sklearn/utils/extmath.py:205: 
    RuntimeWarning: divide by zero encountered in matmul
    ret = a @ b
    /opt/anaconda3/envs/conda-env/lib/python3.13/site-packages/sklearn/utils/extmath.py:205: 
    RuntimeWarning: overflow encountered in matmul
    ret = a @ b
    /opt/anaconda3/envs/conda-env/lib/python3.13/site-packages/sklearn/utils/extmath.py:205: 
    RuntimeWarning: invalid value encountered in matmul
    ret = a @ b
    """

    # and from what i could understand from gemini explanation this error comes from the exthmath.py whihc is called 
    # by StandardScaler.fit_transform(X_train)

    """
    Why extmath.py Warned:

    StandardScaler calculates dot products (a @ b) on X_train. Because np.nan_to_num(X_train) was placed after 
    scaler.fit_transform(), StandardScaler was calculating matrix statistics on the uncleaned raw X_train array 
    containing raw inf values.

    The Fix: Clean X_train BEFORE StandardScaler
    Place np.nan_to_num BEFORE scaler.fit_transform(X_train).
    """

    # the warning did not go away after all changes i made and the conclusion that gemini came was due to : 
    # running Python 3.13 with NumPy 2.x on an Apple Silicon Mac (M1/M2/M3/M4).

    # The Technical Cause:
    # NumPy 2.0 + Apple Accelerate BLAS: On macOS arm64, NumPy 2.x links against Apple's Accelerate.framework 
    # BLAS library for matrix multiplication (a @ b).
    # False BLAS Floating-Point Exception: Apple Accelerate's low-level C/assembly GEMM matrix multiplication 
    # routine sets internal CPU floating-point status flags during matrix dot products. NumPy 2.0's error-checking 
    # layer catches these CPU hardware flags and emits a false RuntimeWarning: divide by zero encountered 
    # in matmul—even when all numbers in the matrix are 100% clean and finite!
    # Location in Code: It happens inside extmath.py:205 when predict() computes a @ b (X_test_scaled @ coef_.T).
    # Because this is a hardware-level BLAS CPU flag in NumPy 2.x on macOS, no amount of data cleaning will 
    # prevent Apple's Accelerate BLAS from setting that CPU flag during a @ b.

    # I also cross-checked this error via the old method stackoverflow and github forum and people with similar problem
    # said the same as gemini its due to macOS M4 https://github.com/numpy/numpy/issues/28687
    # so temp solution is to turn off all the warning related to this Apple Accelerate BLAS CPU flags by doign this :
    #np.seterr(all='ignore')


    print("Fitting Full-feature Logistic Regression to Trianing data")
    full_log_model.fit(X_train_scaled, y_train)

    #5. Predicting on unseen data 
    y_pred = full_log_model.predict(X_test_scaled)

    #6. Accuracy and Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- FULL FEATURE LOGISTIC REGRESSION PERFORMANCE (OUT-OF-SAMPLE) ---")
    print(f"Absolute Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--------------------------------------------------\n")
    
    

if __name__ == "__main__":
    # Execute the split
    train_data, val_data, test_data = load_and_split_data()
    
    # Verification Printout
    print("\n--- CHRONOLOGICAL PARTITION VERIFICATION ---")
    print(f"Total Dataset Rows: {train_data.height + val_data.height + test_data.height}")
    print(f"Training Set (70%): {train_data.height} rows")
    print(f"Validation Set (15%): {val_data.height} rows")
    print(f"Testing Set  (15%): {test_data.height} rows")
    print("--------------------------------------------")
    
    # Verify no time leakage by checking the exact start and end timestamps of each set
    print("\n--- TIME BOUNDARY INTEGRITY CHECK ---")
    print(f"Train End: {train_data.select('timestamp').tail(1).item()}")
    print(f"Val Start: {val_data.select('timestamp').head(1).item()}")
    print(f"Val End:   {val_data.select('timestamp').tail(1).item()}")
    print(f"Test Start:{test_data.select('timestamp').head(1).item()}")
    print("--------------------------------------------\n")
    
    # Train the baseline model first
    train_baseline_model(train_data, test_data)
    
    # Train and evaluate the advanced XGBoost model
    train_xgboost_model(train_data, val_data, test_data)

    # Naive Benchmark Heuristic
    evaluate_naive_benchmark(test_data)

    # Full feature Logistic regression
    train_full_logistic_model(train_data, test_data)