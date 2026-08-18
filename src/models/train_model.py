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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score 

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
    
    baseline_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42) 
    
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