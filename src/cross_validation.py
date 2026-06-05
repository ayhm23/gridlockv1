"""
cross_validation.py
====================
Time-series aware cross-validation for tree-based models.

Strategy: GroupKFold split by `day` column.
  - With only 2 days in training data, we use a stratified-time approach:
    Fold 1 → train on day 48, validate on day 49
    Fold 2 → train on day 49, validate on day 48  (reversed)
    (or standard KFold if more days appear in future data)
  - Each fold saves test predictions → final test pred = mean of all folds.

Usage:
    oof_preds, test_preds, cv_score = run_cv(
        model_name="lgbm",
        params=LGBM_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        groups=groups,      # day column values
        n_folds=5,
        seed=42
    )
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def run_cv(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    groups: pd.Series | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs k-fold cross-validation for the specified model.

    Returns
    -------
    oof_preds  : np.ndarray — OOF predictions on training set (shape: len(X_train),)
    test_preds : np.ndarray — Averaged test predictions      (shape: len(X_test),)
    cv_rmse    : float       — Overall CV RMSE across all folds
    """
    oof_preds   = np.zeros(len(X_train), dtype=np.float64)
    test_preds  = np.zeros(len(X_test),  dtype=np.float64)
    fold_scores = []

    # Choose split strategy
    if groups is not None and groups.nunique() >= n_folds:
        splitter = GroupKFold(n_splits=n_folds)
        splits   = list(splitter.split(X_train, y_train, groups=groups))
    elif groups is not None and groups.nunique() == 2:
        # Only 2 unique days → manual day-wise splits
        splits = _two_day_splits(groups)
        n_folds = len(splits)
        logger.info(f"Only 2 unique day groups detected — using {n_folds} day-wise splits")
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits   = list(splitter.split(X_train))

    for fold_idx, (trn_idx, val_idx) in enumerate(splits):
        logger.info(f"[{model_name.upper()}] Fold {fold_idx + 1}/{n_folds}")

        X_trn, y_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # ── Train model ─────────────────────────────────────────────────────
        model = _build_model(model_name, params, seed=seed)
        model = _fit_model(model, model_name, X_trn, y_trn, X_val, y_val)

        # ── Predict ─────────────────────────────────────────────────────────
        val_pred  = _predict(model, model_name, X_val)
        test_fold = _predict(model, model_name, X_test)

        oof_preds[val_idx]  = val_pred
        test_preds         += test_fold / n_folds

        fold_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        fold_scores.append(fold_rmse)
        logger.info(f"[{model_name.upper()}] Fold {fold_idx + 1} RMSE: {fold_rmse:.6f}")

    cv_rmse = float(np.sqrt(mean_squared_error(y_train, oof_preds)))
    logger.info(
        f"[{model_name.upper()}] CV Complete — "
        f"OOF RMSE: {cv_rmse:.6f} | "
        f"Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}"
    )
    return oof_preds, test_preds, cv_rmse


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _two_day_splits(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Creates two folds: each day used once as the validation set."""
    unique_days = sorted(groups.unique())
    splits = []
    for val_day in unique_days:
        trn_idx = np.where(groups.values != val_day)[0]
        val_idx = np.where(groups.values == val_day)[0]
        splits.append((trn_idx, val_idx))
    return splits


def _build_model(name: str, params: dict, seed: int = 42):
    """Instantiates the model object from its name."""
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        p = {**params, "random_state": seed}
        return LGBMRegressor(**p)

    elif name == "xgb":
        from xgboost import XGBRegressor
        p = {**params, "random_state": seed, "eval_metric": "rmse"}
        return XGBRegressor(**p)

    elif name == "catboost":
        from catboost import CatBoostRegressor
        p = {**params, "random_seed": seed}
        return CatBoostRegressor(**p)

    else:
        raise ValueError(f"Unknown model name: '{name}'. Choose from: lgbm, xgb, catboost")


def _fit_model(model, name: str, X_trn, y_trn, X_val, y_val):
    """Fits model with early stopping when supported."""
    if name == "lgbm":
        from lightgbm import early_stopping, log_evaluation
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(200)]
        )

    elif name == "xgb":
        # XGBoost 3.x: early_stopping_rounds is a constructor/set_params arg, not fit() arg
        model.set_params(early_stopping_rounds=100)
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            verbose=200
        )

    elif name == "catboost":
        model.fit(
            X_trn, y_trn,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=200
        )

    return model


def _predict(model, name: str, X: pd.DataFrame) -> np.ndarray:
    """Unified prediction interface."""
    preds = model.predict(X)
    # Clip to valid demand range [0, 1]
    preds = np.clip(preds, 0.0, 1.0)
    return preds.astype(np.float64)
