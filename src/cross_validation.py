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
    n_folds: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Runs custom time-series cross-validation (Train: Day 48, Validate: Day 49).
    Also returns the optimal iterations (early stopping rounds) found.

    Returns
    -------
    oof_preds  : np.ndarray — OOF predictions on validation set (Day 49), zero elsewhere.
    test_preds : np.ndarray — Predictions on test set (Day 49 slots 9-55) from the validation model.
    cv_rmse    : float       — Validation RMSE on Day 49
    best_iter  : int         — The best iteration/trees number found during validation early stopping
    """
    oof_preds   = np.zeros(len(X_train), dtype=np.float64)
    test_preds  = np.zeros(len(X_test),  dtype=np.float64)

    # Split: Train on Day 48, Validate on Day 49
    trn_idx = np.where(X_train["day"] == 48)[0]
    val_idx = np.where(X_train["day"] == 49)[0]

    logger.info(f"[{model_name.upper()}] Time-series Split: Train rows={len(trn_idx)}, Val rows={len(val_idx)}")

    X_trn, y_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # Build and fit model
    model = _build_model(model_name, params, seed=seed)
    model = _fit_model(model, model_name, X_trn, y_trn, X_val, y_val)

    # Get best iteration/n_estimators
    best_iter = 1000  # fallback
    if model_name == "lgbm":
        best_iter = int(model.best_iteration_)
    elif model_name == "xgb":
        best_iter = int(model.best_iteration)
    elif model_name == "catboost":
        best_iter = int(model.get_best_iteration())
        
    logger.info(f"[{model_name.upper()}] Best early-stopping iteration: {best_iter}")

    # Predict
    val_pred  = _predict(model, model_name, X_val)
    test_preds = _predict(model, model_name, X_test)

    oof_preds[val_idx] = val_pred

    cv_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    logger.info(f"[{model_name.upper()}] Validation RMSE: {cv_rmse:.6f}")

    return oof_preds, test_preds, cv_rmse, best_iter


def train_on_full_data(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    best_iter: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Trains the model on the full training set (Day 48 + Day 49 slots 0-8)
    for a fixed number of iterations (best_iter) and predicts on X_test.
    """
    logger.info(f"[{model_name.upper()}] Retraining on full training set (rows={len(X_train)}) for {best_iter} iterations...")
    
    # Copy params and update iterations
    p = {**params}
    if model_name == "lgbm":
        p["n_estimators"] = max(10, best_iter)
        # Disable early stopping/callbacks
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**p, random_state=seed)
        model.fit(X_train, y_train)
        
    elif model_name == "xgb":
        p["n_estimators"] = max(10, best_iter)
        from xgboost import XGBRegressor
        model = XGBRegressor(**p, random_state=seed, eval_metric="rmse")
        model.fit(X_train, y_train)
        
    elif model_name == "catboost":
        p["iterations"] = max(10, best_iter)
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**p, random_seed=seed)
        model.fit(X_train, y_train, verbose=200)
        
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
        
    test_preds = _predict(model, model_name, X_test)
    return test_preds

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
