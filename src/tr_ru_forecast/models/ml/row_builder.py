"""M9 row serialization with stable column order."""

from __future__ import annotations

from dataclasses import dataclass

from tr_ru_forecast.models.ml.constants import MODEL_TYPE
from tr_ru_forecast.models.ml.records import ForecastIdentity, JsonObject


@dataclass(frozen=True)
class FoldValues:
    """Immutable scalar forecast results consumed by serialization."""

    identity: ForecastIdentity
    is_total: bool
    train_start: str
    train_feature_end: str
    train_label_end: str
    training_row_count: int
    parameter_set_id: str
    actual: float
    prediction: float
    absolute_error: float
    squared_error: float
    smape_component: float
    calibration: JsonObject


def build_prediction_row(fold: FoldValues) -> JsonObject:
    """Return ordered artifact row. Args: fold. Raises: none."""
    return {
        "direction": fold.identity.direction,
        "hs2": fold.identity.hs2,
        "is_total": fold.is_total,
        "horizon": fold.identity.horizon,
        "forecast_origin": fold.identity.forecast_origin,
        "forecast_period": fold.identity.forecast_period,
        "train_start": fold.train_start,
        "train_feature_end": fold.train_feature_end,
        "train_label_end": fold.train_label_end,
        "training_row_count": fold.training_row_count,
        "model_type": MODEL_TYPE,
        "parameter_set_id": fold.parameter_set_id,
        "actual": fold.actual,
        "prediction": fold.prediction,
        **_build_error_fields(fold),
        "interval_available": fold.calibration["interval_available"],
        "calibration_count": fold.calibration["calibration_count"],
        "lower_80": fold.calibration["lower_80"],
        "upper_80": fold.calibration["upper_80"],
        "lower_95": fold.calibration["lower_95"],
        "upper_95": fold.calibration["upper_95"],
        "interval_width_80": fold.calibration["interval_width_80"],
        "interval_width_95": fold.calibration["interval_width_95"],
        "covered_80": fold.calibration["covered_80"],
        "covered_95": fold.calibration["covered_95"],
    }


def build_interval_row(fold: FoldValues) -> JsonObject:
    """Return ordered artifact row. Args: fold. Raises: none."""
    return {
        "direction": fold.identity.direction,
        "hs2": fold.identity.hs2,
        "horizon": fold.identity.horizon,
        "forecast_origin": fold.identity.forecast_origin,
        "forecast_period": fold.identity.forecast_period,
        "interval_available": fold.calibration["interval_available"],
        "calibration_count": fold.calibration["calibration_count"],
        "max_calibration_origin": fold.calibration["max_calibration_origin"],
        "max_calibration_target_period": fold.calibration["max_calibration_target_period"],
        "q80": fold.calibration["q80"],
        "q95": fold.calibration["q95"],
        "lower_80": fold.calibration["lower_80"],
        "upper_80": fold.calibration["upper_80"],
        "lower_95": fold.calibration["lower_95"],
        "upper_95": fold.calibration["upper_95"],
        "actual": fold.actual,
        "prediction": fold.prediction,
        "covered_80": fold.calibration["covered_80"],
        "covered_95": fold.calibration["covered_95"],
    }


def build_fit_row(fold: FoldValues) -> JsonObject:
    """Return ordered artifact row. Args: fold. Raises: none."""
    return {
        "stage": "outer_validation",
        "direction": fold.identity.direction,
        "hs2": fold.identity.hs2,
        "horizon": fold.identity.horizon,
        "origin": fold.identity.forecast_origin,
        "candidate_id": fold.parameter_set_id,
        "status": "SUCCESS",
        "exception_type": None,
        "exception_message": None,
    }


def _build_error_fields(fold: FoldValues) -> JsonObject:
    return {
        "absolute_error": fold.absolute_error,
        "squared_error": fold.squared_error,
        "smape_component": fold.smape_component,
    }
