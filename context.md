# Gridlock Traffic Demand Prediction Project Context

This document outlines the architecture, data structures, preprocessing pipeline, feature engineering, and model training strategies implemented in the Gridlock Hackathon "Traffic Demand Prediction" codebase.

---

## 1. Project Objective & Task
The objective is to predict the traffic **demand** (a bounded float value in `[0.0, 1.0]`) at specific location geohashes and 15-minute time slots. The problem is framed as a supervised regression task evaluated using Root Mean Squared Error (RMSE).

---

## 2. Dataset Properties & Statistics

- **Raw Data Files** (`dataset/`):
  - `train.csv`: Training data containing 77,299 rows and 11 columns, spanning two days (days 48 and 49).
  - `test.csv`: Test data containing 41,778 rows and 10 columns (excluding the target `demand` column).
  - `sample_submission.csv`: Benchmark format showing required submission format.
  
- **Key Columns**:
  - `Index`: Unique row identifier.
  - `geohash`: Spatial region encoded as a geohash string (e.g., `qp02z1`).
  - `day`: Consecutive day index.
  - `timestamp`: Time of day string in 24-hour format (e.g., `2:15` or `23:45`).
  - `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`: Categorical factors.
  - `NumberofLanes`, `Temperature`: Numerical factors.
  - `demand` (Target): Heavily right-skewed float value (mean: `0.0939`, median: `0.0478`, skewness: `3.73`, kurtosis: `17.33`).

- **Data Anomalies & Risks**:
  - **Temporal Lags Leakage**: Rows are not sorted chronologically by default. Any lag feature computation requires sorting by `(day, time_slot)` grouped by `geohash` to prevent leakage.
  - **Zero-Variance Columns**: The `month` variable is constant (spanning only days 48 and 49) and is excluded from features to prevent zero-variance issues.
  - **Shifted Coordinates**: Decoded geohash coordinates map to the Indian Ocean (synthetic/anonymized coordinates). Overlaying external maps (e.g., real road networks of Bengaluru) is not viable.

---

## 3. Codebase Architecture

```
gridlockv1/
├── venv/                       # Python virtual environment containing packages
├── dataset/                    # Directory containing raw train, test, and sample submission csvs
├── data/
│   └── processed/              # Preprocessed train and test parquet datasets
├── logs/                       # Running logs directory
├── submissions/                # Generated predictions, blend CV metadata, and submissions
├── src/
│   ├── __init__.py
│   ├── config.py              # Central hyperparameters, paths, and column definition configuration
│   ├── utils.py               # Downcasting, reproducibility seeding, and geohash decoding utilities
│   ├── data_preprocessing.py  # Loading, standardization, median/unknown imputation, & geohash decoding
│   ├── feature_engineering.py # Lag, rolling, Target Encoding, geohash-hour means, and interactions
│   └── cross_validation.py    # GroupKFold / manual fold splits and training logic
├── main.py                    # Preprocessing driver script
├── train.py                   # Model training and ensembling driver script
└── context.md                  # This file
```

---

## 4. Pipeline Execution Flow

### Step 1: Preprocessing (`main.py`)
- Reads raw files from `dataset/`.
- Cleans and standardizes categories (e.g. normalizing string casing).
- Imputes missing categories with `"Unknown"` and missing numericals with the training median to avoid data leakage.
- Decodes geohash to latitude and longitude coordinates.
- Extracts base temporal columns: `hour`, `minute`, `time_slot` (0-95 indices representing 15-minute intervals), cyclical time components (`sin_time`, `cos_time`), and `day_of_week`.
- Downcasts datatypes (`reduce_mem_usage`) to decrease memory footprint by ~78%.
- Saves clean data to `data/processed/` as Parquet files.

### Step 2: Feature Engineering (`src/feature_engineering.py`)
Features are generated statefully (fit on train, apply on test) under these categories:
- **Extended Lags**: Previous demand values for lag indices `[1, 2, 3, 4, 6, 8, 12, 96, 192]`. `t-192` captures the demand exactly 2 days ago (same time slot).
- **Rolling Features**: Trailing rolling mean (`3`, `6`, and `12` windows), standard deviation (`6` window), maximum (`6` window), minimum (`6` window), and Exponentially Weighted Moving Average (`ewm` with alpha `0.3`) of demand per geohash.
- **Geohash × Hour/Slot Group Means**: Target-encoded representation of geohash averages per hour (`gh_hour_mean`) and per time slot (`gh_slot_mean`) built OOF-safe on training and mapped as global lookup dicts to test.
- **Out-of-Fold Target Encoding**: OOF target encoding for geohash and categoricals (`RoadType`, `Weather`, `LargeVehicles`, `Landmarks`) to safely map categories to historical average demand without leakages.
- **Spatial Clusters**: KMeans coordinates clustering (6 clusters) and computing distance to closest centroid.
- **Cyclical Features**: Cycled representations of `hour` (`hour_sin`, `hour_cos`) and `time_slot` (`sin_time`, `cos_time`).
- **Interaction Flags**: `lanes_x_hour`, `cluster_x_dow`, `cluster_x_slot`, `gh_x_slot`, `lanes_x_cluster`, and `is_rush_hour`.

### Step 3: CV Training & Ensembling (`train.py`)
- Fits three high-performance gradient-boosting regressor models: **LightGBM**, **XGBoost**, and **CatBoost**.
- **Log1p Target Transform**: Target `demand` values are transformed using $y_{log} = \ln(1 + \text{demand})$ before training to normalize the residuals and handle outliers. Predictions are mapped back using the exponential inverse ($e^x - 1$).
- **Cross-Validation**: A 2-fold cross-validation scheme split by `day` (one fold trains on day 48 and validates on day 49; the second fold reverses this split).
- **Blending**: Performs optimal Dirichlet convex weight blending (2000 trials random search) to minimize OOF RMSE.
- **Output Submission**: Clips predictions to `[0.0, 1.0]` and aligns predictions to index, saving results to `submissions/submission_latest.csv`.

---

## 5. Current Performance & Blended Benchmarks

Training results from the optimized pipeline:

- **LightGBM Log CV RMSE**: `0.030223`
- **XGBoost Log CV RMSE**: `0.025291`
- **CatBoost Log CV RMSE**: `0.031376`
- **Optimal Blended Weights**:
  - XGBoost: `96.48%`
  - CatBoost: `2.51%`
  - LightGBM: `1.01%`
- **Final Blend OOF RMSE**:
  - Log space: **`0.025349`**
  - Original space: **`0.031440`**

---

## 6. How to Run the Pipeline

Ensure you have your environment configured:
```bash
# 1. Activate the environment
source venv/bin/activate

# 2. Run preprocessing
python main.py

# 3. Run training
python train.py --n-folds 2
```
Final predictions will be saved under `submissions/submission_latest.csv`.
