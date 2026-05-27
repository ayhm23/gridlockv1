import logging
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import decode_geohash, reduce_mem_usage

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    DataPreprocessor handles the loading, cleaning, and time-series feature extraction
    for the traffic demand prediction datasets.
    
    It maintains state (fitted imputation values) to avoid data leakage between 
    the train and test datasets.
    """
    def __init__(self, target_col="demand", index_col="Index", geohash_col="geohash"):
        self.target_col = target_col
        self.index_col = index_col
        self.geohash_col = geohash_col
        
        # State for imputation values
        self.is_fitted = False
        self.categorical_imputation_val = "Unknown"
        self.numerical_imputations = {}
        
    def load_data(self, train_path: Path, test_path: Path) -> tuple:
        """
        Loads the train and test CSV files.
        """
        logger.info(f"Loading raw train data from {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"Train data loaded. Shape: {train_df.shape}")
        
        logger.info(f"Loading raw test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data loaded. Shape: {test_df.shape}")
        
        return train_df, test_df
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Computes training set statistics for consistent imputation on test set.
        """
        logger.info("Fitting data preprocessor on training data...")
        
        # Identify columns to impute
        # Categorical columns are imputed with "Unknown" by default
        # Numerical columns are imputed with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in [self.index_col, self.target_col, "day"]:
                median_val = df[col].median()
                self.numerical_imputations[col] = median_val
                logger.debug(f"Imputation value for {col}: {median_val}")
                
        self.is_fitted = True
        logger.info("DataPreprocessor successfully fitted.")
        
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning:
        - Imputes missing values using fitted statistics.
        - Standardizes string representations.
        - Decodes geohash to latitude/longitude coordinates (with mapping optimization).
        """
        logger.info("Performing basic cleaning...")
        df = df.copy()
        
        # 1. Clean and Standardize string categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if self.geohash_col in categorical_cols:
            categorical_cols.remove(self.geohash_col)  # Handled separately
            
        for col in categorical_cols:
            # Strip whitespace and capitalize/normalize casing
            df[col] = df[col].astype(str).str.strip().str.title()
            # Replace 'Nan' strings resulting from casting nulls to title
            df[col] = df[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan})
            # Impute categorical nulls
            df[col] = df[col].fillna(self.categorical_imputation_val)
            
        # 2. Impute numerical columns
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted on training data before calling transform/cleaning.")
            
        for col, fill_val in self.numerical_imputations.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
                
        # 3. Geohash Decoding (Optimized)
        logger.info("Decoding geohashes to Latitude/Longitude...")
        unique_geohashes = df[self.geohash_col].dropna().unique()
        logger.info(f"Decoding {len(unique_geohashes)} unique geohashes...")
        
        geohash_map = {}
        for gh in unique_geohashes:
            try:
                lat, lon = decode_geohash(gh)
                geohash_map[gh] = (lat, lon)
            except Exception as e:
                logger.error(f"Error decoding geohash {gh}: {e}")
                geohash_map[gh] = (np.nan, np.nan)
                
        df["latitude"] = df[self.geohash_col].map(lambda x: geohash_map.get(x, (np.nan, np.nan))[0])
        df["longitude"] = df[self.geohash_col].map(lambda x: geohash_map.get(x, (np.nan, np.nan))[1])
        
        # Check if latitude or longitude has missing values (e.g. from failed decode)
        # Impute with overall median coordinates if necessary
        if df["latitude"].isnull().any():
            median_lat = df["latitude"].median()
            df["latitude"] = df["latitude"].fillna(median_lat if not pd.isnull(median_lat) else 0.0)
        if df["longitude"].isnull().any():
            median_lon = df["longitude"].median()
            df["longitude"] = df["longitude"].fillna(median_lon if not pd.isnull(median_lon) else 0.0)
            
        return df
        
    def time_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal and cyclic features from timestamps:
        - `hour` and `minute` integers.
        - `time_slot` (0-95 slots per day for 15-minute intervals).
        - `sin_time` and `cos_time` (cyclical representation of time of day).
        - `day_of_week` (based on raw day value).
        """
        logger.info("Extracting temporal features...")
        df = df.copy()
        
        # Check for timestamp presence
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column found. Skipping temporal feature extraction.")
            return df
            
        # Standardize and split timestamp (format: "H:M" e.g., "2:15" or "0:0")
        df["timestamp"] = df["timestamp"].fillna("0:0").astype(str)
        time_split = df["timestamp"].str.split(":", expand=True)
        
        df["hour"] = time_split[0].astype(int)
        df["minute"] = time_split[1].astype(int)
        
        # Time slot index (0 to 95 for 15-minute aggregates)
        df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
        
        # Cyclical encoding of the time slot
        df["sin_time"] = np.sin(2 * np.pi * df["time_slot"] / 96.0)
        df["cos_time"] = np.cos(2 * np.pi * df["time_slot"] / 96.0)
        
        # Calculate day of week (assuming day starts at 0 or any integer index)
        if "day" in df.columns:
            df["day_of_week"] = df["day"] % 7
            
        return df

    def save_processed_data(self, df: pd.DataFrame, file_path: Path, save_format: str = "parquet") -> None:
        """
        Saves the processed dataset as Parquet or CSV.
        Falls back to CSV if Parquet libraries are not available.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format.lower() == "parquet":
            try:
                import pyarrow
                df.to_parquet(file_path, index=False)
                logger.info(f"Successfully saved processed dataset to {file_path} in PARQUET format.")
                return
            except ImportError:
                logger.warning("pyarrow or fastparquet is not installed. Falling back to CSV.")
                # Update file path extension to CSV
                file_path = file_path.with_suffix(".csv")
                
        # Save as CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved processed dataset to {file_path} in CSV format.")
