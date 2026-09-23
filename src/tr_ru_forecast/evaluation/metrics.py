"""Deterministic time-series evaluation metrics for forecasting models.

Implements primary, secondary, and diagnostic metrics:
- sMAPE (Symmetric Mean Absolute Percentage Error) with strict both-zero safeguard.
- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def compute_smape_single(actual: float, prediction: float) -> float:
    """Compute sMAPE component for a single observation pair.

    Rule:
    If actual == 0 and prediction == 0:
        return 0.0
    Else:
        return 2.0 * abs(actual - prediction) / (abs(actual) + abs(prediction)) * 100.0
    """
    if pd.isnull(actual) or pd.isnull(prediction):
        return np.nan

    actual_f = float(actual)
    pred_f = float(prediction)

    if actual_f == 0.0 and pred_f == 0.0:
        return 0.0

    denom = abs(actual_f) + abs(pred_f)
    if denom == 0.0:
        return 0.0

    return (2.0 * abs(actual_f - pred_f) / denom) * 100.0


def calculate_smape(actuals: np.ndarray | pd.Series, predictions: np.ndarray | pd.Series) -> float:
    """Compute sMAPE across a vector of observations."""
    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)

    valid_mask = ~np.isnan(act) & ~np.isnan(pred)
    if not np.any(valid_mask):
        return np.nan

    act_v = act[valid_mask]
    pred_v = pred[valid_mask]

    components = []
    for a, p in zip(act_v, pred_v):
        components.append(compute_smape_single(a, p))

    return float(np.mean(components))


def calculate_mae(actuals: np.ndarray | pd.Series, predictions: np.ndarray | pd.Series) -> float:
    """Compute Mean Absolute Error across a vector of observations."""
    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)

    valid_mask = ~np.isnan(act) & ~np.isnan(pred)
    if not np.any(valid_mask):
        return np.nan

    return float(np.mean(np.abs(act[valid_mask] - pred[valid_mask])))


def calculate_rmse(actuals: np.ndarray | pd.Series, predictions: np.ndarray | pd.Series) -> float:
    """Compute Root Mean Squared Error across a vector of observations."""
    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)

    valid_mask = ~np.isnan(act) & ~np.isnan(pred)
    if not np.any(valid_mask):
        return np.nan

    return float(np.sqrt(np.mean((act[valid_mask] - pred[valid_mask]) ** 2)))


def compute_prediction_error_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute_error, squared_error, and smape_component to prediction DataFrame.

    Requires columns 'actual_value' and 'prediction'.
    """
    res = df.copy()

    abs_err = np.abs(res["actual_value"] - res["prediction"])
    sq_err = (res["actual_value"] - res["prediction"]) ** 2

    # Compute vectorized smape components
    smapes = []
    for a, p in zip(res["actual_value"], res["prediction"]):
        smapes.append(compute_smape_single(a, p))

    res["absolute_error"] = abs_err
    res["squared_error"] = sq_err
    res["smape_component"] = smapes

    return res


def aggregate_group_metrics(
    prediction_df: pd.DataFrame,
    run_id: str,
    model_type: str = "seasonal_naive",
) -> pd.DataFrame:
    """Aggregate row-level predictions into the canonical 42 series x horizon metrics table.

    Groups by [direction, hs2, is_total, horizon].
    """
    group_records: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()

    group_cols = ["direction", "hs2", "is_total", "horizon"]
    for (dir_name, hs2_code, is_tot, h), group in prediction_df.groupby(group_cols):
        valid_group = group[group["prediction"].notnull() & group["actual_value"].notnull()]

        smape_val = float(valid_group["smape_component"].mean()) if not valid_group.empty else np.nan
        mae_val = float(valid_group["absolute_error"].mean()) if not valid_group.empty else np.nan
        rmse_val = float(np.sqrt(valid_group["squared_error"].mean())) if not valid_group.empty else np.nan

        first_orig = str(valid_group["forecast_origin"].min()) if not valid_group.empty else None
        last_orig = str(valid_group["forecast_origin"].max()) if not valid_group.empty else None

        group_records.append({
            "run_id": run_id,
            "direction": str(dir_name),
            "hs2": str(hs2_code),
            "is_total": bool(is_tot),
            "horizon": int(h),
            "model_type": str(model_type),
            "n_validation_origins": int(len(valid_group)),
            "first_origin": first_orig,
            "last_origin": last_orig,
            "smape": smape_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "created_at": now_iso,
        })

    metrics_df = pd.DataFrame(group_records)
    metrics_df = metrics_df.sort_values(["direction", "hs2", "horizon"]).reset_index(drop=True)
    return metrics_df
