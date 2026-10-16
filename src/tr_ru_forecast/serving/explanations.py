"""Serialize actual frozen XGBoost explanations, preserving their registered role."""

from __future__ import annotations

import pandas as pd

from tr_ru_forecast.serving.catalog_schemas import TaskComparison
from tr_ru_forecast.serving.explain_schemas import (
    Attribution,
    AvailableExplanation,
    ExplanationResponse,
    FeatureCategory,
    FeatureImportance,
    LocalExplanation,
    UnavailableExplanation,
)
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE


def classify_feature(feature: str) -> FeatureCategory:
    """Classify an approved predictor. Args: feature. Returns: category. Raises: ValueError."""
    if feature.startswith("lag_") or feature in {"mom_growth", "yoy_growth"}:
        return "trade_lag"
    if feature.startswith("rolling_"):
        return "rolling_stat"
    if feature in {"month", "quarter", "year"}:
        return "seasonality"
    if feature in {"brent_lagged", "ru_ip_lagged", "rub_usd_lagged", "tr_ip_lagged", "usd_try_lagged"}:
        return "macro_indicator"
    raise ValueError("Unregistered SHAP feature")


def build_explanation(
    task: TaskComparison, local: pd.DataFrame, summary: pd.DataFrame
) -> ExplanationResponse:
    """Project a task's XGBoost role. Args: task, local, summary. Returns: explanation. Raises: schema errors."""
    if "xgboost" not in (task.champion_model, task.fallback_model):
        return UnavailableExplanation(sector_id=task.sector_id, horizon=task.horizon,
                                      evaluation_scope=EVALUATION_SCOPE)
    return AvailableExplanation(
        sector_id=task.sector_id, horizon=task.horizon, evaluation_scope=EVALUATION_SCOPE,
        role="champion" if task.champion_model == "xgboost" else "fallback",
        global_importance=tuple(_build_importance(record) for _, record in summary.sort_values(
            ["mean_abs_shap", "feature"], ascending=[False, True]).iterrows()),
        local=tuple(_build_local(group) for _, group in local.groupby("forecast_period", sort=True)),
    )


def _build_importance(record: pd.Series) -> FeatureImportance:
    return FeatureImportance(
        feature=record.feature, mean_abs_shap=record.mean_abs_shap,
        mean_signed_shap=record.mean_signed_shap, category=classify_feature(record.feature),
        n_origins=record.n_origins,
    )


def _build_local(frame: pd.DataFrame) -> LocalExplanation:
    first = frame.iloc[0]
    contributions = tuple(Attribution(
        feature=record.feature, feature_value=record.feature_value, shap_value=record.shap_value,
        category=classify_feature(record.feature),
    ) for record in frame.sort_values("feature").itertuples(index=False))
    positive = tuple(sorted((part for part in contributions if (part.shap_value or 0) > 0),
                            key=_sort_shap, reverse=True)[:5])
    negative = tuple(sorted((part for part in contributions if (part.shap_value or 0) < 0),
                            key=_sort_shap)[:5])
    return LocalExplanation(
        origin_month=first.forecast_origin, target_month=first.forecast_period,
        base_value=first.base_value, prediction=first.point_prediction,
        contributions=contributions, top_positive=positive, top_negative=negative,
    )


def _sort_shap(attribution: Attribution) -> float:
    return attribution.shap_value or 0.0
