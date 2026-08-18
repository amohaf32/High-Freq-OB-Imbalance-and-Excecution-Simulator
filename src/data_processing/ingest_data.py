import os 
import polars as pl

def run_ingestion_pipeline():
    
    #to define and initi
    raw_data_path = "data/BTCUSDT-bookDepth-2026-06-24.csv"
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    
    print (f"Reading raw data from {raw_data_path}...")
    
    actual_header = ["timestamp", "percentage", "depth", "notional"]
    
    # force Polars lib to read columns as numbers, avoiding string inference errors (except for timestamp which is a string)
    explicit_schema = {
        "timestamp": pl.String,
        "percentage": pl.Float64,
        "depth": pl.Float64,
        "notional": pl.Float64
    }
    
    # read the raw data with schema overrides
    lazy_df = pl.scan_csv(
        raw_data_path, 
        has_header = True, 
        new_columns = actual_header,
        schema_overrides = explicit_schema  # Fixes the string arithmetic error
    )
    
    # clean and then isolate the bid and the ask based on the +ve/-ve percentage
    processed_df = (
        lazy_df
        #converting the timestamps format to a micorsecond format
        .with_columns(pl.col("timestamp").str.to_datetime())
        
        # normally any standard order book dataset would be in a wide format(one single line per timestamp containing all prices and quantities horizontally)
        # but since binance dataset is in a long format (one single line per price level per timestamp) i would need to pivot the percentage column
        # so when the percentage would be -ve then that would be the bid_depth and when the percentage would be +ve then that would be the ask_depth
        .with_columns([
            pl.when(pl.col("percentage") < 0 )
            .then(pl.lit("bid_depth_") + pl.col("percentage").abs().cast(pl.String))
            .otherwise(pl.lit("ask_depth_") + pl.col("percentage").cast(pl.String))
            .alias("side_level")
        ])
    )
    
    # pivot the df from rows to side-by-side horizontal simulator header 
    # converts multiple rows into singular wide row per timestamp
    wide_df = (
        processed_df.collect() # bring into memory to pivot
        .pivot(
            on = "side_level",
            index = "timestamp",
            values = "depth",
            aggregate_function = "first" 
        )
        # enforce chronological ordering across the index
        .sort("timestamp")
    )
    
    print("\n---- New wide simulator layout created ----")
    print(wide_df.head(5))
    print("--------------------------------------------\n")
    
    return wide_df

if __name__ == "__main__":
    run_ingestion_pipeline()
