import os
import random
import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ==============================================================================
# Seeding for reproducibility
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across different packages.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try setting PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Try setting TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ==============================================================================
# Logging Setup
# ==============================================================================
def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Sets up python standard logging. Logs will be written to stdout and a file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pipeline_{timestamp}.log"
    
    # Reset existing handlers
    logging.root.handlers = []
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info(f"Logging configured. Saving log output to {log_file}")


# ==============================================================================
# Geohash decoding
# ==============================================================================
def decode_geohash(geohash: str) -> tuple:
    """
    Decodes a geohash string into latitude and longitude coordinates.
    Tries to use pygeohash if installed, otherwise falls back to a custom, 
    efficient, pure-python manual implementation.
    
    Args:
        geohash: string representation of the geohash (e.g. 'qp02z1')
    Returns:
        tuple (latitude, longitude) of floats representing the centroid.
    """
    try:
        import pygeohash as pgh
        return pgh.decode(geohash)
    except ImportError:
        pass

    # Pure Python Fallback Implementation
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_map = {char: i for i, char in enumerate(base32)}
    
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    
    is_even = True
    for char in geohash:
        if char not in char_map:
            raise ValueError(f"Invalid character in geohash: {char}")
        val = char_map[char]
        for mask in [16, 8, 4, 2, 1]:
            bit = 1 if (val & mask) else 0
            if is_even:
                # Even bits: longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit == 1:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                # Odd bits: latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit == 1:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon


# ==============================================================================
# Memory optimization
# ==============================================================================
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all columns of a dataframe and downcasts numeric types to save memory.
    """
    logger = logging.getLogger(__name__)
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(f"Initial memory usage of dataframe: {start_mem:.2f} MB")
        
    for col in df.columns:
        col_type = df[col].dtype
        
        # Check if the column is a category already
        if isinstance(col_type, pd.CategoricalDtype):
            continue
            
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Check if converting to category makes sense (low cardinality string column)
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                df[col] = df[col].astype('category')
                
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Memory decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
        
    return df
