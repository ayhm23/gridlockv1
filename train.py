"""
train.py
=========
Main training entrypoint for the Gridlock Traffic Demand Prediction pipeline.

Usage:
    python train.py                        # Default: LightGBM + XGBoost + CatBoost
    python train.py --models lgbm xgb     # Only LightGBM + XGBoost
    python train.py --n-folds 2           # 2-fold CV (matches 2-day dataset)
    python train.py --seed 42 --log-level DEBUG

Pipeline steps:
  1. Load preprocessed train + test (from data/processed/)
  2. Feature engineering (lags, rolling, target encoding, spatial)
  3. CV training for each model (OOF predictions + test predictions)
  4. Ensemble / blend predictions
  5. Save submission CSV to submissions/
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error

from src import config
from src.utils import set_seed, setup_logging
from src.feature_engineering import FeatureEngineer, get_feature_cols
from src.cross_validation import run_cv

logger = logging.getLogger("train_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Model configurations — tuned for demand ∈ [0,1], small dataset, fast runs
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "n_estimators":       2000,
    "learning_rate":      0.03,
    "num_leaves":         63,
    "max_depth":          -1,
    "subsample":          0.85,
    "colsample_bytree":   0.85,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "min_child_samples":  20,
    "n_jobs":             -1,
    "verbose":            -1,
}

XGB_PARAMS = {
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "max_depth":        7,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_weight": 5,
    "tree_method":      "hist",
    "device":           "cpu",
    "n_jobs":           -1,
    "verbosity":        0,
}

CATBOOST_PARAMS = {
    "iterations":       2000,
    "learning_rate":    0.03,
    "depth":            8,
    "l2_leaf_reg":      3.0,
    "bootstrap_type":   "Bernoulli",
    "subsample":        0.85,
    "verbose":          200,
    "task_type":        "CPU",
    "allow_writing_files": False,
}

MODEL_PARAMS = {
    "lgbm":     LGBM_PARAMS,
    "xgb":      XGB_PARAMS,
    "catboost": CATBOOST_PARAMS,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gridlock Traffic Demand — Training Pipeline"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["lgbm", "xgb", "catboost"],
        choices=["lgbm", "xgb", "catboost"],
        help="Models to train and ensemble (default: all three)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=2,
        help="Number of CV folds (default: 2, matching 2-day dataset)"
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed (default: {config.RANDOM_SEED})"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--train-path", type=str,
        default=str(config.PROCESSED_TRAIN_PATH),
        help="Path to processed train parquet"
    )
    parser.add_argument(
        "--test-path", type=str,
        default=str(config.PROCESSED_TEST_PATH),
        help="Path to processed test parquet"
    )
    parser.add_argument(
        "--submission-path", type=str,
        default=str(config.SUBMISSION_PATH),
        help="Path to sample_submission.csv for Index alignment"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(log_dir=str(config.LOG_DIR), log_level=args.log_level)
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Gridlock Traffic Demand — Training Pipeline")
    logger.info(f"Models:  {args.models}")
    logger.info(f"N-Folds: {args.n_folds}")
    logger.info(f"Seed:    {args.seed}")
    logger.info("=" * 60)

    # ── 1. Load processed datasets ───────────────────────────────────────────
    logger.info("Loading preprocessed datasets...")
    train_path = Path(args.train_path)
    test_path  = Path(args.test_path)

    if not train_path.exists():
        logger.error(f"Processed train not found at {train_path}. Run main.py first.")
        return
    if not test_path.exists():
        logger.error(f"Processed test not found at {test_path}. Run main.py first.")
        return

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    # ── 2. Feature Engineering ───────────────────────────────────────────────
    logger.info("Running feature engineering...")
    fe = FeatureEngineer(target_col=config.TARGET_COL, n_folds=args.n_folds)
    train_fe = fe.fit_transform(train_df)
    test_fe  = fe.transform(test_df, train_df=train_fe)

    feature_cols = get_feature_cols(train_fe, target_col=config.TARGET_COL)
    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    X_train = train_fe[feature_cols]
    y_train = train_fe[config.TARGET_COL]
    X_test  = test_fe[feature_cols]
    groups  = train_fe["day"] if "day" in train_fe.columns else None

    logger.info(f"X_train: {X_train.shape} | y_train mean: {y_train.mean():.5f}")

    # ── 3. Cross-Validation Training ─────────────────────────────────────────
    oof_results  : dict[str, np.ndarray] = {}
    test_results : dict[str, np.ndarray] = {}
    cv_scores    : dict[str, float]       = {}

    for model_name in args.models:
        logger.info(f"\n{'---' * 14}")
        logger.info(f"Training: {model_name.upper()}")
        logger.info(f"{'---' * 14}")
        params = MODEL_PARAMS[model_name]

        oof, test_pred, rmse = run_cv(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            groups=groups,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        oof_results[model_name]  = oof
        test_results[model_name] = test_pred
        cv_scores[model_name]    = rmse

    # ── 4. Ensemble / Blend ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Blending model predictions...")

    # Compute optimal ensemble weights by minimizing OOF RMSE
    oof_matrix  = np.column_stack([oof_results[m]  for m in args.models])
    test_matrix = np.column_stack([test_results[m] for m in args.models])

    best_weights, best_rmse = _optimize_blend_weights(oof_matrix, y_train.values, n_trials=500)

    logger.info(f"Optimal blend weights: {dict(zip(args.models, best_weights.round(4)))}")
    logger.info(f"Ensemble OOF RMSE:     {best_rmse:.6f}")
    logger.info(f"Individual CV RMSEs:   {cv_scores}")

    final_test_preds = test_matrix @ best_weights
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    # ── 5. Generate Submission ───────────────────────────────────────────────
    logger.info("Generating submission CSV...")
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    # Load sample submission to align Index column
    sample_sub = pd.read_csv(args.submission_path)
    logger.info(f"Sample submission shape: {sample_sub.shape}")

    # Build submission DataFrame aligned to test Index
    test_index_col = test_fe["Index"] if "Index" in test_fe.columns else pd.Series(range(len(test_fe)))

    submission = pd.DataFrame({
        "Index":  test_index_col.values,
        "demand": final_test_preds,
    })

    # Verify row count matches sample submission
    if len(submission) != len(sample_sub):
        logger.warning(
            f"Submission row count ({len(submission)}) does not match "
            f"sample submission ({len(sample_sub)})! Check alignment."
        )

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path   = submissions_dir / f"submission_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to: {sub_path}")

    # Also save a 'latest' symlink-style copy for convenience
    latest_path = submissions_dir / "submission_latest.csv"
    submission.to_csv(latest_path, index=False)

    # Save CV metadata
    meta = {
        "timestamp":      timestamp,
        "models":         args.models,
        "n_folds":        args.n_folds,
        "seed":           args.seed,
        "feature_count":  len(feature_cols),
        "cv_scores":      cv_scores,
        "blend_weights":  dict(zip(args.models, best_weights.tolist())),
        "ensemble_rmse":  best_rmse,
    }
    meta_path = submissions_dir / f"cv_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"CV metadata saved to: {meta_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Submission: {sub_path}")
    logger.info(f"Ensemble RMSE (OOF): {best_rmse:.6f}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Blend weight optimization
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_blend_weights(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 500,
    seed: int = 42
) -> tuple[np.ndarray, float]:
    """
    Randomly search for the best convex combination of OOF predictions
    that minimizes RMSE. Returns (weights, best_rmse).
    """
    rng         = np.random.default_rng(seed)
    n_models    = oof_matrix.shape[1]
    best_rmse   = float("inf")
    best_weights = np.ones(n_models) / n_models  # equal weights as default

    for _ in range(n_trials):
        # Sample from Dirichlet distribution → weights sum to 1
        w     = rng.dirichlet(np.ones(n_models))
        preds = np.clip(oof_matrix @ w, 0.0, 1.0)
        rmse  = float(np.sqrt(mean_squared_error(y_true, preds)))
        if rmse < best_rmse:
            best_rmse    = rmse
            best_weights = w

    return best_weights, best_rmse


if __name__ == "__main__":
    main()
