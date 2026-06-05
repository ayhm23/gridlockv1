import os
from pathlib import Path

# ==============================================================================
# Path Management
# ==============================================================================
# Base directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = ROOT_DIR / "dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Log directory
LOG_DIR = ROOT_DIR / "logs"

# File paths
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

# Preprocessed file paths (can save as csv or parquet)
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.parquet"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test_processed.parquet"

# Save format ("parquet" or "csv")
SAVE_FORMAT = "parquet"

# ==============================================================================
# Global settings
# ==============================================================================
RANDOM_SEED = 42

# ==============================================================================
# Column Definitions
# ==============================================================================
INDEX_COL = "Index"
GEOHASH_COL = "geohash"
DAY_COL = "day"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "demand"

CATEGORICAL_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUMERICAL_COLS = ["NumberofLanes", "Temperature"]

# ==============================================================================
# Model Hyperparameters (For future steps)
# ==============================================================================
XGB_PARAMS = {
    "random_state": RANDOM_SEED,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}

LGBM_PARAMS = {
    "random_state": RANDOM_SEED,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "random_seed": RANDOM_SEED,
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "verbose": 100,
}
