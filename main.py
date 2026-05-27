import argparse
import logging
from pathlib import Path
from src import config
from src.utils import set_seed, setup_logging, reduce_mem_usage
from src.data_preprocessing import DataPreprocessor

logger = logging.getLogger("main_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Gridlock Traffic Demand Preprocessing Pipeline")
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=config.RANDOM_SEED, 
        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED})"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--save-format", 
        type=str, 
        default=config.SAVE_FORMAT, 
        choices=["csv", "parquet"],
        help=f"Format to save processed datasets (default: {config.SAVE_FORMAT})"
    )
    parser.add_argument(
        "--train-path", 
        type=str, 
        default=str(config.TRAIN_PATH), 
        help="Path to raw train.csv"
    )
    parser.add_argument(
        "--test-path", 
        type=str, 
        default=str(config.TEST_PATH), 
        help="Path to raw test.csv"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR), 
        help="Directory to save preprocessed datasets"
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run comprehensive Exploratory Data Analysis (EDA) on raw train data"
    )
    
    return parser.parse_args()

def main():
    # 1. Parse CLI arguments
    args = parse_args()
    
    # 2. Setup logging and random seeds
    setup_logging(log_dir=str(config.LOG_DIR), log_level=args.log_level)
    logger.info("Initializing Gridlock Traffic Demand Preprocessing Pipeline...")
    
    logger.info(f"Setting random seed to {args.seed}")
    set_seed(args.seed)
    
    # Resolve Paths
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    output_dir = Path(args.output_dir)
    
    # Run EDA if requested
    if args.run_eda:
        logger.info("--- Running Exploratory Data Analysis (EDA) ---")
        from src.eda import TrafficEDA
        try:
            eda_runner = TrafficEDA(data_path=str(train_path), output_dir="reports")
            eda_runner.run_all_eda()
            logger.info("EDA completed successfully! Report generated at reports/eda_report.md")
        except Exception as e:
            logger.error(f"Failed to execute EDA pipeline: {e}", exc_info=True)
    
    # 3. Instantiate DataPreprocessor
    preprocessor = DataPreprocessor(
        target_col=config.TARGET_COL,
        index_col=config.INDEX_COL,
        geohash_col=config.GEOHASH_COL
    )
    
    # 4. Load Data
    try:
        train_raw, test_raw = preprocessor.load_data(train_path, test_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading raw datasets: {e}")
        logger.error("Please verify that raw data is placed in data/raw/ or specify the paths via CLI arguments.")
        return
        
    # 5. Fit Preprocessor on Training Data
    preprocessor.fit(train_raw)
    
    # 6. Preprocess Train Dataset
    logger.info("--- Processing TRAIN dataset ---")
    train_cleaned = preprocessor.basic_cleaning(train_raw)
    train_features = preprocessor.time_feature_extraction(train_cleaned)
    train_final = reduce_mem_usage(train_features)
    
    # 7. Preprocess Test Dataset
    logger.info("--- Processing TEST dataset ---")
    test_cleaned = preprocessor.basic_cleaning(test_raw)
    test_features = preprocessor.time_feature_extraction(test_cleaned)
    test_final = reduce_mem_usage(test_features)
    
    # 8. Save Processed Datasets
    logger.info("Saving preprocessed datasets...")
    suffix = f".{args.save_format}"
    processed_train_file = output_dir / f"train_processed{suffix}"
    processed_test_file = output_dir / f"test_processed{suffix}"
    
    preprocessor.save_processed_data(train_final, processed_train_file, save_format=args.save_format)
    preprocessor.save_processed_data(test_final, processed_test_file, save_format=args.save_format)
    
    # 9. Verify Data integrity and Log Summary Stats
    logger.info("=========================================")
    logger.info("Preprocessing Pipeline Completed Successfully!")
    logger.info(f"Processed Train Shape: {train_final.shape}")
    logger.info(f"Processed Test Shape:  {test_final.shape}")
    
    logger.info("Processed Train Columns:")
    logger.info(list(train_final.columns))
    logger.info("Processed Test Columns:")
    logger.info(list(test_final.columns))
    
    # Print a tiny preview of the preprocessed data
    logger.info("Train dataset preview:")
    logger.info("\n" + str(train_final.head(3)))
    logger.info("=========================================")

if __name__ == "__main__":
    main()
