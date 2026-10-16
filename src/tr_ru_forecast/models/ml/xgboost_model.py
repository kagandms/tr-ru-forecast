"""XGBoost Direct model builder, label embargo enforcer, and trainer.

Implements Section 6 & 7:
- Direct forecasting semantics (separate models for h1, h3, h6).
- Strict label-availability embargo: training_period + horizon <= forecast_origin.
- Whitelist feature filtering (strictly 21 predictors, zero metadata/future columns).
- Infinite value safeguards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from tr_ru_forecast.models.ml.constants import (
    FIXED_XGB_PARAMS,
    FORBIDDEN_COLUMNS,
    M9_PREDICTORS,
)
from tr_ru_forecast.models.ml.records import JsonObject


def validate_predictor_features(df: pd.DataFrame) -> None:
    """Validate that the DataFrame contains all 21 whitelist predictors and no forbidden columns as predictors."""
    missing = [c for c in M9_PREDICTORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required M9 predictors: {missing}")

    # Check for forbidden columns if df is passed as X
    overlap = [c for c in FORBIDDEN_COLUMNS if c in df.columns and c in M9_PREDICTORS]
    if overlap:
        raise ValueError(f"CRITICAL: Forbidden columns found in predictor set: {overlap}")


def get_training_subset(
    series_df: pd.DataFrame,
    forecast_origin: str,
    horizon: int,
) -> pd.DataFrame:
    """Return origin-safe direct training rows. Args: series, origin, horizon. Raises: ValueError."""
    origin_p = pd.Period(forecast_origin.strip(), freq="M")
    cutoff_p = origin_p - horizon
    cutoff_str = str(cutoff_p)
    label_col = f"target_h{horizon}"

    if label_col not in series_df.columns:
        raise ValueError(f"Target label column '{label_col}' missing from series DataFrame")

    # Strictly filter rows by period <= cutoff_str AND non-null label
    eligible_mask = (series_df["period"] <= cutoff_str) & series_df[label_col].notnull()
    eligible = series_df[eligible_mask].copy()

    if eligible.empty:
        return eligible

    _validate_training_embargo(eligible, str(origin_p), horizon)
    return eligible


def build_xgb_regressor(params: JsonObject) -> xgb.XGBRegressor:
    """Build an XGBRegressor combining fixed hyperparameters with candidate hyperparameters."""
    # Filter candidate params to exclude candidate_id or non-XGBoost keys
    clean_params = {
        k: v
        for k, v in params.items()
        if k in ["max_depth", "min_child_weight", "learning_rate", "n_estimators"]
    }
    full_params = {**FIXED_XGB_PARAMS, **clean_params}
    return xgb.XGBRegressor(**full_params)


def prepare_training_xy(
    train_df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (21 whitelist predictors) and y (target_h{horizon}) from training subset."""
    label_col = f"target_h{horizon}"
    validate_predictor_features(train_df)

    X = train_df[M9_PREDICTORS].copy()
    y = train_df[label_col].copy()

    # Fail closed on +inf / -inf
    if np.isinf(X.to_numpy(dtype=float, na_value=0.0)).any():
        raise ValueError("Infinite values detected in training features X")
    if np.isinf(y.to_numpy(dtype=float, na_value=0.0)).any():
        raise ValueError("Infinite values detected in training labels y")

    if y.isnull().any():
        raise ValueError("Missing label detected in eligible training rows")

    return X, y


def extract_origin_features(
    series_df: pd.DataFrame,
    forecast_origin: str,
) -> pd.DataFrame:
    """Extract exactly one feature row corresponding to the forecast origin t."""
    validate_predictor_features(series_df)
    origin_rows = series_df[series_df["period"] == forecast_origin]
    if len(origin_rows) == 0:
        raise ValueError(f"No feature row found for origin period '{forecast_origin}'")
    if len(origin_rows) > 1:
        raise ValueError(f"Duplicate feature rows found for origin period '{forecast_origin}'")

    X_origin = origin_rows[M9_PREDICTORS].copy()
    if np.isinf(X_origin.to_numpy(dtype=float, na_value=0.0)).any():
        raise ValueError(f"Infinite values detected in origin features for '{forecast_origin}'")

    return X_origin


def _validate_training_embargo(eligible: pd.DataFrame, origin: str, horizon: int) -> None:
    label_periods = pd.PeriodIndex(eligible["period"], freq="M") + horizon
    if (label_periods > pd.Period(origin, freq="M")).any():
        raise ValueError("CRITICAL LABEL LEAKAGE: training label beyond origin")
