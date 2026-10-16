"""Bounded, pre-validation M9 tuning with explicit typed fit outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.metrics import calculate_smape
from tr_ru_forecast.models.ml.constants import (
    FORECAST_HORIZONS,
    TUNING_ORIGINS,
    generate_candidate_grid,
)
from tr_ru_forecast.models.ml.records import JsonObject
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class TuningSeriesOutcome:
    """Complete origin block with explicit failed-fit evidence."""

    actuals: tuple[float, ...]
    predictions: tuple[float, ...]
    failures: tuple[str, ...]


def _fit_tuning_origin(
    series: pd.DataFrame, origin: str, task: tuple[int, JsonObject]
) -> tuple[float, float]:
    horizon, candidate = task
    training = get_training_subset(series, origin, horizon)
    if training.empty:
        raise ValueError(f"No eligible training rows at {origin}")
    predictors, labels = prepare_training_xy(training, horizon)
    model = build_xgb_regressor(candidate)
    model.fit(predictors, labels)
    prediction = float(model.predict(extract_origin_features(series, origin))[0])
    actual = float(series.loc[series["period"] == origin, f"target_h{horizon}"].iloc[0])
    return actual, prediction


def _evaluate_series(
    series: pd.DataFrame, identity: JsonObject, task: tuple[int, JsonObject]
) -> TuningSeriesOutcome:
    horizon, candidate = task
    actuals, predictions, failures = [], [], []
    for origin in TUNING_ORIGINS[horizon]:
        try:
            actual, prediction = _fit_tuning_origin(series, origin, task)
            actuals.append(actual)
            predictions.append(prediction)
        except Exception as error:
            failures.append(f"{identity['direction']}-{identity['hs2']}-{origin}: {error}")
            logger.warning(
                "Tuning fit failed for %s at %s: %s", candidate["candidate_id"], origin, error
            )
    return TuningSeriesOutcome(tuple(actuals), tuple(predictions), tuple(failures))


def evaluate_tuning_candidate(
    dev_features_df: pd.DataFrame,
    series_list: list[JsonObject],
    horizon: int,
    candidate: JsonObject,
) -> JsonObject:
    """Return unchanged M9 candidate scores. Args: features, series, horizon, params. Raises: none."""
    smapes: list[float] = []
    failures: list[str] = []
    for identity in series_list:
        series = dev_features_df.loc[
            dev_features_df.direction.eq(identity["direction"])
            & dev_features_df.hs2.eq(identity["hs2"])
        ].copy()
        outcome = _evaluate_series(series, identity, (horizon, candidate))
        failures.extend(outcome.failures)
        smapes.append(
            calculate_smape(np.array(outcome.actuals), np.array(outcome.predictions))
            if len(outcome.actuals) == len(TUNING_ORIGINS[horizon]) and not failures
            else np.nan
        )
    has_valid_result = (
        not failures
        and len(smapes) == len(series_list)
        and not any(np.isnan(value) for value in smapes)
    )
    return _build_candidate_result(candidate, (horizon, smapes, has_valid_result), failures)


def _build_candidate_result(
    candidate: JsonObject, scores: tuple[int, list[float], bool], failures: list[str]
) -> JsonObject:
    horizon, smapes, has_valid_result = scores
    return {
        "horizon": int(horizon),
        "candidate_id": str(candidate["candidate_id"]),
        "max_depth": int(candidate["max_depth"]),
        "min_child_weight": int(candidate["min_child_weight"]),
        "learning_rate": float(candidate["learning_rate"]),
        "n_estimators": int(candidate["n_estimators"]),
        "status": "VALID" if has_valid_result else "INVALID",
        "mean_series_smape": float(np.mean(smapes)) if has_valid_result else np.nan,
        "median_series_smape": float(np.median(smapes)) if has_valid_result else np.nan,
        "fit_count": len(smapes) * len(TUNING_ORIGINS[horizon]) - len(failures),
        "failure_count": len(failures),
        "failure_details": "; ".join(failures[:5]) if failures else "",
    }


def run_pre_validation_tuning(
    dev_features_df: pd.DataFrame, series_list: list[JsonObject], horizons: list[int] | None = None
) -> tuple[pd.DataFrame, dict[int, JsonObject]]:
    """Return tuning table and frozen parameters. Args: features, series, horizons. Raises: RuntimeError."""
    candidates = generate_candidate_grid()
    records: list[JsonObject] = []
    selected: dict[int, JsonObject] = {}
    for horizon in horizons if horizons is not None else FORECAST_HORIZONS:
        results = [
            evaluate_tuning_candidate(dev_features_df, series_list, horizon, candidate)
            for candidate in candidates
        ]
        valid = pd.DataFrame(results).query("status == 'VALID'")
        if valid.empty:
            raise RuntimeError(
                f"CRITICAL: All 16 tuning candidates failed for horizon h={horizon}!"
            )
        winner = valid.sort_values(
            ["mean_series_smape", "median_series_smape", "candidate_id"]
        ).iloc[0]["candidate_id"]
        records.extend(
            {**result, "selected": bool(result["candidate_id"] == winner)} for result in results
        )
        selected[horizon] = next(
            candidate for candidate in candidates if candidate["candidate_id"] == winner
        )
    return pd.DataFrame(records).sort_values(["horizon", "candidate_id"]).reset_index(
        drop=True
    ), selected
