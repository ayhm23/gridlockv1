# Gridlock Traffic Demand Prediction

This is a modular machine learning codebase for the Gridlock Hackathon "Traffic Demand Prediction" challenge. It implements high-performance preprocessing, stateful cleaning/imputation, spatial geohash decoding, and time-series temporal feature extraction.

## Directory Structure

```
d:/FlipKartGridlock/
├── data/
│   ├── raw/                # train.csv, test.csv, sample_submission.csv
│   └── processed/          # Preprocessed datasets (train_processed.parquet, test_processed.parquet)
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters, column names, paths, and configurations
│   ├── utils.py            # Helper functions (seeding, logging, memory downcasting, geohash decode)
│   └── data_preprocessing.py # DataPreprocessor class (imputation, cleaning, time features)
│
├── main.py                 # Pipeline execution entry point
├── requirements.txt        # Package dependencies
├── README.md               # Codebase documentation
└── logs/                   # Log directory for experiments and pipelines
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Preprocessing Pipeline:**
   Execute the driver script to load raw datasets, run cleaning & geohash decoding, optimize memory usage, and save preprocessed datasets under `data/processed/`:
   ```bash
   python main.py
   ```

### Command Line Arguments

You can customize execution using command line overrides:
- `--seed <int>`: Set a custom random seed (default: `42`).
- `--log-level <level>`: Define logging output detail: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`).
- `--save-format <format>`: Format to save the datasets: `csv` or `parquet` (default: `parquet`).
- `--train-path <path>`: Override the default path to raw train dataset.
- `--test-path <path>`: Override the default path to raw test dataset.
- `--output-dir <dir>`: Override output directory for processed files.

Example of running with overrides:
```bash
python main.py --save-format csv --log-level DEBUG
```

## Features Preprocessed

- **Spatial Decoding**: Geohashes are parsed into exact `latitude` and `longitude` coordinates using a highly efficient mapping cache (only decodes unique geohashes to speed up processing).
- **Temporal Extraction**: Split timestamps (`timestamp`) into numeric `hour` and `minute`.
- **Cyclical Features**: Convert 15-minute time slots (0-95 index) into cyclic sin/cos components (`sin_time`, `cos_time`) to help models capture diurnal traffic periodicities.
- **Calendar Signals**: Extract weekday index (`day_of_week`) from day indexes.
- **Stateful Imputation**: Fits numerical medians on training data and applies them to both training and test sets to prevent target/data leakage. Categorical nulls are filled with `"Unknown"`.
- **Memory Downcasting**: Performs column-wise datatype casting to minimize RAM footprint.
