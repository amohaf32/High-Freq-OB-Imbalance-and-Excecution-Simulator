import sys
import os
import polars as pl

# discover the project root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_processing.ingest_data import run_ingestion_pipeline

def engineer_OrderBook_features(df: pl.DataFrame) -> pl.DataFrame:
    print("Start of feature engineering on the wide layout orderbook matrix")
    
    # mapping out the paired levels based on the available percentage buckets 
    # For Binance bookDepth, these represent distance layers from the market center
    levels = ["0.2", "1.0", "2.0", "3.0", "5.0"]
    
    # calcualte the vectorised local order book imbalance for each level
    # Formula: (Bid_Depth - Ask_Depth) / (Bid_Depth + Ask_Depth)
    
    obi_expressions = [
        (
            (pl.col(f"bid_depth_{lvl}") - pl.col(f"ask_depth_{lvl}")) /
            (pl.col(f"bid_depth_{lvl}") + pl.col(f"ask_depth_{lvl}"))
        ).alias(f"obi_{lvl}") 
        for lvl in levels if f"bid_depth_{lvl}" in df.columns and f"ask_depth_{lvl}" in df.columns
    ]
    
    # calculate the gloabl weighted order book imbalance across whole depth profile
    # the logic that will be used here will be distance decay logic
    # this means closer levels like 0.2% will have a higher weight and more significance than deep level like 5%
    """
    In high-frequency trading, order books have memory degradation. 
    Orders sitting right near the current price (0.2%) are highly urgent and about to trade immediately. 
    Orders sitting far away (5.0%) might just be institutional walls or passive orders that won't trade for hours.
    
    The general rule ("closer orders matter more than further orders") applies to almost all order book data, 
    how you choose to decay those weights depends entirely on the asset class, the specific dataset structure, and your trading horizon
    
    Therefore we assign linear distance weights: 0.2% -> 1.0, 1.0% -> 0.8, 2.0% -> 0.6, etc.
    """
    weights = {"0.2": 1.0, "1.0": 0.8, "2.0": 0.6, "3.0": 0.4, "5.0": 0.2}
    
    # calculate the wrighted for both bid and ask sides 
    weighted_bid = sum(weights[lvl] * pl.col(f"bid_depth_{lvl}") for lvl in levels)
    weighted_ask = sum(weights[lvl] * pl.col(f"ask_depth_{lvl}") for lvl in levels)
    
    # calculate the global weighted order book imbalance
    global_weighted_obi_exp = (((weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)).alias("obi_global_weighted"))
    
    # apply the expressions to the dataframe
    feature_df = df.with_columns(obi_expressions + [global_weighted_obi_exp])
    
    
    """
    in the lines of code above to the feature_df we calculated the order book imbalance and weights so that we can measure 
    the immediate pressure between the buyers and sellers at differnet depth levels of the book at a single point in time aka a forzen snapshot. 
    
    - What it tells the model: "At exactly 12:00:05, do buyers outnumber sellers at the 0.20% layer?"

    - The Math: It's purely structural and localized to that specific row. It takes the depth of the bid and the depth of the ask simultaneously 
    and outputs a ratio between -1 and 1.

    - The Limitation: It has zero memory. If the imbalance is 0.8 (strongly bullish), Task 2.1 doesn't know if it has been 0.8 for the 
    last ten minutes or if it just spiked from -0.9 one second ago.
    
    but from the code below onwards we will be introducing time and movement (history and volatility), in other words we will be looking across
    multiple consectuive rows(seconds) to see how the market state is shifting instead of measuring the pressure at one moment, it measures the
    speed, acceleration and instability of the pressure. 
    
    Here is what each feature in Task 2.2 achieves that Task 2.1 cannot:

    OBI Velocity (5s, 10s, 30s): 
    
        Concept: This tracks how fast the imbalance itself is changing.

        Why it matters: If Task 2.1 says the imbalance is 0.5 (moderately bullish), but the 5s Velocity is +0.7, it means the book is 
        aggressively flipping into a buy-heavy state. If the velocity is -0.4, it means the buying pressure is dying out. This 
        captures momentum.

    OBI Volatility Profile (Rolling Standard Deviation):

        Concept: This tracks how chaotic or stable the front-line order book is over a window of time.

        Why it matters: High volatility in the imbalance indicates a high-frequency battleground—traders are aggressively 
        cancelling and replacing orders. Low volatility means a steady, calm institutional queuing environment. Machine learning models use 
        this to identify regime shifts.

    Liquidity Width:

        Concept: A proxy for a bid-ask spread environment when absolute prices are missing. It measures the net divergence of volume 
        right at the market frontline.
    """
    # starting for 2.2 where we will appply the feature_df since it contains obi_0.2
    feature_df = feature_df.with_columns([
        # calculate the liquidity width proxy (baiscally the frontline depth difference between the bid and ask)
        (pl.col("ask_depth_0.2") - pl.col("bid_depth_0.2")).alias("liquidity_width"),
        
        # calculate the price velocity & momentum (rolling changes in the frontline OBI over 5s,10s,30s windows)
        (pl.col("obi_0.2") - pl.col("obi_0.2").shift(5)).alias("obi_velocity_5s"),
        (pl.col("obi_0.2") - pl.col("obi_0.2").shift(10)).alias("obi_velocity_10s"),
        (pl.col("obi_0.2") - pl.col("obi_0.2").shift(30)).alias("obi_velocity_30s"),
        
        # high-frequency historical volatility profile
        (pl.col("obi_0.2").rolling_std(window_size=10)).alias("obi_volatility_10s")
    ])
    
    """
    in task 2.2 we were esentially using the past windows of data to analyze the current state momentum (eg. trailing 5s,10s,30s) in other words historical/backward-looking
    but when coming to task 2.3 we will now be peeking into the future to see what happens next (eg. see what happens in the 30s) in other words future/forward-looking.
    
    a great analogy is think of it like a sports telemetry system. Task 2.2 is the car's live speedometer and acceleration forces. Task 2.3 is the finish line position 30 seconds later.
    
    so when you run your strategy live, your feature pipeline (Task 2.2) can calculate metrics instantaneously because it only uses data that has already occurred conversely, you can never 
    calculate Task 2.3 in a live trading script until those 30 seconds have actually expired.
    
    """
    # calculate the future order book imbalance (forward-looking) / target discrete label generation (forward horizon)
    TAU = 30 # forward lookahead horizon in 30 seconds
    ALPHA = 0.015 
    
    """
    the alpha is the eduacted baseline guess (a heuristic) chosen based on the obi_global_weighted which is contstrained between -1.0 and 1.0 and if we were to look between the row 3 and row 4 
    we can see that the obi_global_weighted has shifter roughly around 0.012 in about 30 seconds, so when setting the alpha we can see the in a 30s window the fluctuations happens somewhere between 
    0.0005 and 0.012 so setting the alpha threshold to 0.015 slighlty above the normal noise now tells the model the following : 
    
    Only classify this as a true 'Up' (1) or 'Down' (-1) movement if the global order book balance shifts by more than 1.5%.
    Anything less than 1.5% is just normal, random market vibration, so classify it as 'Stationary' (0)
    """   
    
    feature_df = feature_df.with_columns([
        # calculate forward looking proxy return (using continous forward to the book state)
        (pl.col("obi_global_weighted").shift(-TAU) - pl.col("obi_global_weighted")).alias("forward_book_return")
    ])
    
    feature_df = feature_df.with_columns([
        # map the continous return to disscrete 3-class system (-1,0,1) for the model to classify
        pl.when(pl.col("forward_book_return") > ALPHA).then(1)
         .when(pl.col("forward_book_return") < -ALPHA).then(-1)
         .otherwise(0)
         .alias("target_label")
    ])
    
    """
    till now we have calculted both ob_velocity_30s (looking 30s into the past) and forward_book_return (looking 30s into the future), the first 30 rows and the last 30 ros of the dataset contains 
    incomplete mathematical calculations (Null) and if we feed these blind rows into the ml model it will learn the curropted patterns, so we need slice them cleanly 
    
    now in polars library there is a method called parquet upgrade method and what this does is compared to csv files are slow, massive and terrible at remembering fata types, so by saving the 
    final matrix as a .parquet file compresses the data heavily and reads 10x to 50x faster into ml models permantly locking the column data types. 
    """
    
    # 2.4 optiimizing the dataset serialization (filtration)
    # drop the incomplete edge rows before filling nulls
    # we explcitly drops the rows where out maximum look-back or look-forward generated a null. 
    feature_df = feature_df.drop_nulls(subset=["obi_velocity_30s", "forward_book_return"])
    
    # fill empty entries with a neutral score of 0.0
    # NOTE: This will also clean up the initial nulls created by .shift() and .rolling_std()
    feature_df = feature_df.fill_null(0.0)
    
    print("\n---- Feature engineering completed ----")
    return feature_df

if __name__ == "__main__":
    # generate the ingested base dataframe from Module 1
    base_df = run_ingestion_pipeline()
    
    # Calculate our alpha signals
    final_features = engineer_OrderBook_features(base_df)
    
    # Isolate key signal columns to verify the calculations
    sample_cols = [
        "timestamp", 
        "obi_0.2", 
        "obi_global_weighted", 
        "liquidity_width", 
        "obi_velocity_5s", 
        "obi_volatility_10s"
    ]
    print("\n--- SIGNAL MATRIX VERIFICATION SUCCESSFUL ---")
    print(final_features.select(sample_cols).slice(40,5))
    print("------------------------------------------------\n")
    
    # inspection of the balance distribution of our new ML labels
    print("--- TARGET CLASS DISTRIBUTION ---")
    print(final_features["target_label"].value_counts())
    print("---------------------------------\n")
    
    # 2.4 optimizing the dataset serialization (filtration)
    # defining the exact path to the data folder 
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/engineered_features.parquet"))
    
    # write the dataframe into the disk using polars high-speed parquet engine
    final_features.write_parquet(output_path)
    
    print(f"--- DATASET SERIALIZED SUCCESSFULLY ---")
    print(f"Saved optimized .parquet file to: {output_path}")
    print("---------------------------------------\n")