"""
train_tuned.py
==============
Tuned training pipeline incorporating:
  1. Optuna-tuned hyperparameters for XGBoost.
  2. Fixed target-leakage-free KNN spatial neighborhood lag features.
  3. Weighted ensemble of LightGBM, tuned XGBoost, and CatBoost (optimized for R2).
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score

from src import config
from src.utils import set_seed, setup_logging
from src.feature_engineering import FeatureEngineer, get_feature_cols
from src.cross_validation import run_cv, train_on_full_data

logger = logging.getLogger("train_tuned")

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

# Hyperparameters tuned via Optuna study
XGB_PARAMS_TUNED = {
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
    "eval_metric":      "R2",
    "allow_writing_files": False,
}

MODEL_PARAMS = {
    "lgbm":     LGBM_PARAMS,
    "xgb":      XGB_PARAMS_TUNED,
    "catboost": CATBOOST_PARAMS,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gridlock — Tuned Training Pipeline")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def optimize_blend_weights(oof_matrix: np.ndarray, y_true: np.ndarray, n_trials: int = 100, seed: int = 42):
    from scipy.optimize import minimize
    
    n_models = oof_matrix.shape[1]
    
    def objective(w):
        if np.sum(w) == 0:
            return 0.0
        w_norm = w / np.sum(w)
        preds = np.clip(oof_matrix @ w_norm, 0, 1)
        return -r2_score(y_true, preds)
        
    best_r2 = -float("inf")
    best_w = np.ones(n_models) / n_models
    
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        w0 = rng.dirichlet(np.ones(n_models))
        res = minimize(
            objective, 
            w0, 
            method="SLSQP", 
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        )
        if res.success:
            w_opt = res.x / np.sum(res.x)
            preds = np.clip(oof_matrix @ w_opt, 0, 1)
            r2 = float(r2_score(y_true, preds))
            if r2 > best_r2:
                best_r2 = r2
                best_w = w_opt
                
    return best_w, best_r2



def main():
    args = parse_args()
    setup_logging(log_dir=str(config.LOG_DIR), log_level=args.log_level)
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Gridlock Traffic Demand — Tuned Training Pipeline")
    logger.info(f"Seed:    {args.seed}")
    logger.info("=" * 60)

    # Load data
    train_df = pd.read_parquet(config.PROCESSED_TRAIN_PATH)
    test_df  = pd.read_parquet(config.PROCESSED_TEST_PATH)
    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    # Feature Engineering
    fe = FeatureEngineer(target_col=config.TARGET_COL, n_folds=2)
    train_fe = fe.fit_transform(train_df)
    test_fe  = fe.transform(test_df, train_df=train_fe)

    feature_cols = get_feature_cols(train_fe, target_col=config.TARGET_COL)
    feature_cols = [c for c in feature_cols if c not in ["knn_lag96_mean", "knn_lag96_std"]]
    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    X_train = train_fe[feature_cols]
    y_train = train_fe[config.TARGET_COL]
    X_test  = test_fe[feature_cols]
    groups  = train_fe["day"]

    # Compute sample weights based on yesterday's same-slot demand (lag 96)
    sample_weight = 1.0 + X_train["demand_lag_96"] ** 2

    # Model training with CV
    model_names = ["lgbm", "xgb", "catboost"]
    oof_results  = {}
    test_results = {}
    cv_scores    = {}
    best_iters   = {}

    for model_name in model_names:
        logger.info(f"\n{'---' * 14}")
        logger.info(f"Validating: {model_name.upper()}")
        logger.info(f"{'---' * 14}")
        params = MODEL_PARAMS[model_name]

        oof, _, rmse, best_iter = run_cv(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            groups=groups,
            n_folds=2,
            sample_weight=sample_weight,
            seed=args.seed,
        )
        oof_results[model_name] = oof
        cv_scores[model_name]   = rmse
        best_iters[model_name]  = best_iter

        logger.info(f"\nRetraining {model_name.upper()} on all data...")
        test_pred_full = train_on_full_data(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            best_iter=best_iter,
            sample_weight=sample_weight,
            seed=args.seed,
        )
        test_results[model_name] = test_pred_full

    # Blending
    logger.info("\n" + "=" * 60)
    logger.info("Blending model predictions to maximize R2...")
    val_idx = np.where(train_fe["day"] == 49)[0]
    oof_matrix_val = np.column_stack([oof_results[m][val_idx] for m in model_names])
    y_val_true     = y_train.values[val_idx]

    best_weights, best_r2 = optimize_blend_weights(oof_matrix_val, y_val_true, n_trials=1000, seed=args.seed)
    
    # Calculate RMSE for this best blend
    blended_val_preds = np.clip(oof_matrix_val @ best_weights, 0, 1)
    ensemble_rmse = float(np.sqrt(mean_squared_error(y_val_true, blended_val_preds)))

    logger.info(f"Optimal blend weights: {dict(zip(model_names, best_weights.round(4)))}")
    logger.info(f"Ensemble OOF R2 Score:  {best_r2 * 100:.4f}%")
    logger.info(f"Ensemble OOF RMSE:      {ensemble_rmse:.6f}")
    logger.info(f"Individual CV RMSEs:   {cv_scores}")

    # Log individual validation R2s
    for m in model_names:
        r2_val = r2_score(y_val_true, oof_results[m][val_idx]) * 100
        logger.info(f"Model {m.upper()} Val R2: {r2_val:.4f}%")

    final_test_preds = np.column_stack([test_results[m] for m in model_names]) @ best_weights
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    # Submission
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    test_index_col = test_fe["Index"] if "Index" in test_fe.columns else pd.Series(range(len(test_fe)))
    submission = pd.DataFrame({
        "Index":  test_index_col.values,
        "demand": final_test_preds,
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path  = submissions_dir / f"submission_r2optimized_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to: {sub_path}")

    # Save a separate XGBoost-only submission for testing the hypothesis
    xgb_submission = pd.DataFrame({
        "Index":  test_index_col.values,
        "demand": test_results["xgb"],
    })
    xgb_sub_path = submissions_dir / f"submission_r2optimized_xgb_only_{timestamp}.csv"
    xgb_submission.to_csv(xgb_sub_path, index=False)
    logger.info(f"XGB-only submission saved to: {xgb_sub_path}")

    latest_path = submissions_dir / "submission_latest.csv"
    submission.to_csv(latest_path, index=False)

    meta = {
        "timestamp":         timestamp,
        "models":            model_names,
        "seed":              args.seed,
        "feature_count":     len(feature_cols),
        "cv_scores":         cv_scores,
        "best_iters":        best_iters,
        "blend_weights":     dict(zip(model_names, best_weights.tolist())),
        "ensemble_r2":       best_r2 * 100,
        "ensemble_rmse":     ensemble_rmse,
        "demand_min":        float(submission["demand"].min()),
        "demand_max":        float(submission["demand"].max()),
    }
    meta_path = submissions_dir / f"cv_meta_tuned_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"CV metadata saved to: {meta_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TUNED TRAINING PIPELINE COMPLETE")
    logger.info(f"Submission file: {sub_path}")
    logger.info(f"Latest copy:     {latest_path}")
    logger.info(f"Ensemble R2 (x100): {best_r2 * 100:.4f}")
    logger.info(f"Ensemble RMSE:   {ensemble_rmse:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
