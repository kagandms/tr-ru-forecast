"""TreeSHAP explainer and feature attribution aggregator for M9 XGBoost.

Implements Section 20, 21, 22, 23:
- Fold-local TreeExplainer with tree_path_dependent feature perturbation.
- Additive consistency verification (base_value + sum(shap_values) == prediction).
- Local SHAP contribution persistence across all 3,934 validation predictions (82,614 rows).
- Group-specific global feature importance (mean absolute SHAP per 42 series/horizon groups, 882 rows).
- Strictly non-causal terminology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.models.ml.constants import M9_PREDICTORS

logger = get_logger(__name__)

ADDITIVITY_RTOL: float = 1e-4
ADDITIVITY_ATOL: float = 1.0


def explain_fold_prediction(
    model: xgb.XGBRegressor,
    X_origin: pd.DataFrame,
    point_prediction: float,
    direction: str,
    hs2: str,
    horizon: int,
    forecast_origin: str,
    forecast_period: str,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    """Compute TreeSHAP attributions for a single fold prediction and verify additive consistency.

    Returns:
        base_value: Expected value of the model.
        feature_shaps: Dict of feature -> shap_value.
        local_records: List of 21 dicts for the long-form local SHAP artifact.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    explanation = explainer(X_origin)

    base_val = float(explanation.base_values[0])
    shap_vals = np.array(explanation.values[0], dtype=float)

    # Verify additive consistency: base_value + sum(shap_values) ≈ point_prediction
    # Accounts for float32 library precision in tree output for large monetary scales
    shap_sum = float(np.sum(shap_vals))
    reconstructed_pred = base_val + shap_sum
    discrepancy = abs(reconstructed_pred - point_prediction)
    threshold = max(ADDITIVITY_ATOL, ADDITIVITY_RTOL * abs(point_prediction))

    if discrepancy > threshold:
        raise ValueError(
            f"CRITICAL SHAP ADDITIVITY FAILURE for ({direction}, {hs2}, h={horizon}) at {forecast_origin}: "
            f"base_val={base_val}, shap_sum={shap_sum}, reconstructed={reconstructed_pred}, "
            f"point_prediction={point_prediction}, diff={discrepancy} > threshold={threshold}"
        )


    feature_shaps: dict[str, float] = {}
    local_records: list[dict[str, Any]] = []

    for idx, feat_name in enumerate(M9_PREDICTORS):
        val_shap = float(shap_vals[idx])
        feat_val = float(X_origin[feat_name].iloc[0])
        feature_shaps[feat_name] = val_shap

        local_records.append({
            "direction": str(direction),
            "hs2": str(hs2),
            "horizon": int(horizon),
            "forecast_origin": str(forecast_origin),
            "forecast_period": str(forecast_period),
            "feature": str(feat_name),
            "feature_value": feat_val,
            "shap_value": val_shap,
            "base_value": base_val,
            "point_prediction": float(point_prediction),
        })

    return base_val, feature_shaps, local_records


def compute_global_shap_importance(
    local_shap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute global feature importance as mean absolute SHAP for each of the 42 series/horizon groups.

    Formula:
        global_importance = mean(abs(shap_value))
    """
    records: list[dict[str, Any]] = []
    group_cols = ["direction", "hs2", "horizon"]

    for (dir_name, hs2_code, h), group in local_shap_df.groupby(group_cols):
        for feat_name, feat_group in group.groupby("feature"):
            mean_abs = float(np.mean(np.abs(feat_group["shap_value"])))
            records.append({
                "direction": str(dir_name),
                "hs2": str(hs2_code),
                "horizon": int(h),
                "feature": str(feat_name),
                "global_importance": mean_abs,
                "n_observations": len(feat_group),
            })

    global_df = pd.DataFrame(records)
    # Sort within each group by global_importance descending
    global_df = global_df.sort_values(
        by=["direction", "hs2", "horizon", "global_importance"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return global_df
