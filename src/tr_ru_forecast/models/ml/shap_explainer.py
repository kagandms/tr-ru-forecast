"""Fold-local TreeSHAP and scale-specific aggregation; predictions remain unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from tr_ru_forecast.models.ml.constants import M9_PREDICTORS
from tr_ru_forecast.models.ml.records import ForecastIdentity, JsonObject

ADDITIVITY_RTOL = 1e-4
ADDITIVITY_ATOL = 1.0


def _verify_additivity(base_value: float, values: np.ndarray, prediction: float) -> None:
    reconstructed = base_value + float(np.sum(values))
    threshold = max(ADDITIVITY_ATOL, ADDITIVITY_RTOL * abs(prediction))
    if abs(reconstructed - prediction) > threshold:
        raise ValueError(
            f"CRITICAL SHAP ADDITIVITY FAILURE: reconstructed={reconstructed}, point_prediction={prediction}"
        )


def _build_local_records(
    identity: ForecastIdentity, features: pd.DataFrame, explanation: tuple[float, np.ndarray, float]
) -> list[JsonObject]:
    base_value, values, prediction = explanation
    return [
        {
            "direction": identity.direction,
            "hs2": identity.hs2,
            "horizon": identity.horizon,
            "forecast_origin": identity.forecast_origin,
            "forecast_period": identity.forecast_period,
            "feature": name,
            "feature_value": float(features[name].iloc[0]),
            "shap_value": float(values[index]),
            "base_value": base_value,
            "point_prediction": float(prediction),
        }
        for index, name in enumerate(M9_PREDICTORS)
    ]


def explain_fold_prediction(
    model: xgb.XGBRegressor,
    X_origin: pd.DataFrame,
    point_prediction: float,
    direction: str,
    hs2: str,
    horizon: int,
    forecast_origin: str,
    forecast_period: str,
) -> tuple[float, dict[str, float], list[JsonObject]]:
    """Explain frozen fold model. Args: model, row, prediction, identity. Returns: base, values, records.

    Raises: ValueError for failed additivity. Legacy argument shape retained for existing callers.
    """
    explanation = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw"
    )(X_origin)
    base_value = float(explanation.base_values[0])
    values = np.array(explanation.values[0], dtype=float)
    _verify_additivity(base_value, values, point_prediction)
    identity = ForecastIdentity(direction, hs2, horizon, forecast_origin, forecast_period)
    records = _build_local_records(identity, X_origin, (base_value, values, point_prediction))
    return base_value, dict(zip(M9_PREDICTORS, map(float, values), strict=True)), records


def compute_global_shap_importance(local_shap_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-task mean absolute SHAP. Args: local rows. Raises: KeyError for invalid schema."""
    records = []
    for (direction, code, horizon), group in local_shap_df.groupby(["direction", "hs2", "horizon"]):
        for feature, feature_group in group.groupby("feature"):
            records.append(
                {
                    "direction": str(direction),
                    "hs2": str(code),
                    "horizon": int(horizon),
                    "feature": str(feature),
                    "global_importance": float(np.mean(np.abs(feature_group["shap_value"]))),
                    "n_observations": len(feature_group),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(
            ["direction", "hs2", "horizon", "global_importance"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
