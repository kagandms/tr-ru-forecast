"""XGBoost-only explanations; missing model roles are explicitly unavailable."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tr_ru_forecast.serving.schemas import FrozenModel, Horizon, Month, Number

FeatureCategory = Literal["trade_lag", "rolling_stat", "seasonality", "macro_indicator"]


class FeatureImportance(FrozenModel):
    """Immutable feature importance contract."""

    feature: str
    mean_abs_shap: Number
    mean_signed_shap: Number
    category: FeatureCategory
    n_origins: int


class Attribution(FrozenModel):
    """Immutable attribution contract."""

    feature: str
    feature_value: Number
    shap_value: Number
    category: FeatureCategory


class LocalExplanation(FrozenModel):
    """Immutable local explanation contract."""

    origin_month: Month
    target_month: Month
    base_value: Number
    prediction: Number
    contributions: tuple[Attribution, ...]
    top_positive: tuple[Attribution, ...]
    top_negative: tuple[Attribution, ...]


class AvailableExplanation(FrozenModel):
    """Immutable available explanation contract."""

    status: Literal["available"] = "available"
    sector_id: str
    horizon: Horizon
    model_family: Literal["xgboost"] = "xgboost"
    role: Literal["champion", "fallback"]
    evaluation_scope: str
    global_importance: tuple[FeatureImportance, ...]
    local: tuple[LocalExplanation, ...]
    explanation_scope: Literal["predictive_attribution_not_causality"] = "predictive_attribution_not_causality"


class UnavailableExplanation(FrozenModel):
    """Immutable unavailable explanation contract."""

    status: Literal["unavailable"] = "unavailable"
    sector_id: str
    horizon: Horizon
    evaluation_scope: str
    reason: Literal["no_xgboost_role"] = "no_xgboost_role"


ExplanationResponse = Annotated[
    AvailableExplanation | UnavailableExplanation, Field(discriminator="status")
]
