"""Frozen rolling-origin holdout execution; no model or feature selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
import pandas as pd

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.champion_io import ChampionEntry
from tr_ru_forecast.evaluation.champion_selector import compute_skill
from tr_ru_forecast.evaluation.holdout_io import HoldoutInputs
from tr_ru_forecast.features.builder import (
    build_complete_calendar_panel,
    compute_autoregressive_lags,
    compute_backward_rolling_features,
    compute_calendar_features,
    compute_growth_features,
    compute_target_labels,
)
from tr_ru_forecast.features.macro_aligner import (
    MACRO_FEATURE_COLUMN_MAP,
    align_publication_safe_macro,
)
from tr_ru_forecast.models.classical.sarima import SARIMAPredictor
from tr_ru_forecast.models.classical.sarimax import SARIMAXPredictor
from tr_ru_forecast.models.ml.conformal import PrequentialResidualCalibrator
from tr_ru_forecast.models.ml.records import ForecastIdentity, JsonObject
from tr_ru_forecast.models.ml.shap_explainer import explain_fold_prediction
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
)

logger = get_logger(__name__)
TaskRows: TypeAlias = tuple[list[JsonObject], list[JsonObject]]


@dataclass(frozen=True)
class HoldoutTask:
    """Frozen registry assignment and scoring calendar."""

    entry: ChampionEntry
    identity: ForecastIdentity
    model_family: str


@dataclass(frozen=True)
class ForecastOutcome:
    """Raw point forecast, uncertainty, validity and fit audit."""

    prediction: float
    lower_80: float = np.nan
    upper_80: float = np.nan
    lower_95: float = np.nan
    upper_95: float = np.nan
    fit_status: str = "SUCCESS"
    converged: bool = True
    failure_message: str = ""
    calibration_count: int = 0
    max_calibration_target_period: str | None = None
    train_end: str = ""
    shap_rows: tuple[JsonObject, ...] = ()
    feature_count: int = 0
    optimizer_used: str = "not_applicable"
    retry_count: int = 0
    warning_type: str = "NONE"


def generate_holdout_origins(entry: ChampionEntry) -> tuple[ForecastIdentity, ...]:
    """Return all twelve target-minus-horizon origins. Args: registry entry. Raises: none."""
    return tuple(
        ForecastIdentity(
            entry.direction, entry.hs2, entry.horizon, str(target - entry.horizon), str(target)
        )
        for target in pd.period_range("2025-01", "2025-12", freq="M")
    )


def build_origin_features(inputs: HoldoutInputs, origin: str) -> pd.DataFrame:
    """Construct snapshot only from origin-safe targets. Args: inputs, origin. Raises: ValueError."""
    visible = inputs.targets.loc[inputs.targets.period.le(origin)].copy()
    panel = build_complete_calendar_panel(
        visible, inputs.sectors, start_period="2012-01", end_period=origin
    )
    for transform in (
        lambda frame: compute_target_labels(frame, max_dev_period=origin),
        compute_autoregressive_lags,
        compute_backward_rolling_features,
        compute_growth_features,
        compute_calendar_features,
    ):
        panel = transform(panel)
    aligned, audit = align_publication_safe_macro(panel, inputs.macro)
    if any(
        row["is_available"]
        and row["selected_macro_availability_date"] > row["feature_snapshot_date"]
        for row in audit
    ):
        raise ValueError("Macro availability beyond feature snapshot")
    if aligned.period.gt(origin).any():
        raise ValueError("Future target entered feature construction")
    return aligned


def initialize_calibrator(
    cv: pd.DataFrame, identity: ForecastIdentity
) -> PrequentialResidualCalibrator:
    """Seed task-local OOS residuals. Args: CV evidence, identity. Raises: ValueError on invalid geometry."""
    calibrator = PrequentialResidualCalibrator(identity.direction, identity.hs2, identity.horizon)
    rows = cv.loc[
        cv.direction.eq(identity.direction)
        & cv.hs2.eq(identity.hs2)
        & cv.horizon.eq(identity.horizon)
        & cv.forecast_period.le(identity.forecast_origin)
    ].sort_values("forecast_origin")
    for row in rows.to_dict("records"):
        if row["forecast_origin"] >= row["forecast_period"]:
            raise ValueError("Invalid CV residual geometry")
        if not np.isfinite([row["actual"], row["prediction"]]).all() or row["prediction"] < 0:
            continue
        calibrator.record_historical_prediction(
            row["forecast_origin"], row["forecast_period"], row["actual"], row["prediction"]
        )
    return calibrator


def _get_series(features: pd.DataFrame, task: HoldoutTask) -> pd.DataFrame:
    return (
        features.loc[
            features.direction.eq(task.identity.direction) & features.hs2.eq(task.identity.hs2)
        ]
        .sort_values("period")
        .reset_index(drop=True)
    )


def _fit_xgboost(
    series: pd.DataFrame,
    task: HoldoutTask,
    context: tuple[HoldoutInputs, PrequentialResidualCalibrator],
) -> ForecastOutcome:
    inputs, calibrator = context
    horizon, origin = task.identity.horizon, task.identity.forecast_origin
    training = get_training_subset(series, origin, horizon)
    predictors, labels = prepare_training_xy(training, horizon)
    origin_features = extract_origin_features(series, origin)
    parameters = inputs.parameters["tuning"]["selected_parameters"][f"h{horizon}"]
    model = build_xgb_regressor(parameters)
    model.set_params(**inputs.parameters["fixed_parameters"])
    model.fit(predictors, labels)
    prediction = float(model.predict(origin_features)[0])
    interval = calibrator.calibrate(origin, prediction)
    _, _, shap_rows = explain_fold_prediction(
        model,
        origin_features,
        prediction,
        task.identity.direction,
        task.identity.hs2,
        horizon,
        origin,
        task.identity.forecast_period,
    )
    return _build_xgb_outcome(training, (prediction, interval, shap_rows), horizon)


def _fit_classical(
    series: pd.DataFrame, task: HoldoutTask, inputs: HoldoutInputs
) -> ForecastOutcome:
    order = next(
        row
        for row in inputs.orders["series_orders"]
        if row["direction"] == task.entry.direction and str(row["hs2"]).zfill(2) == task.entry.hs2
    )
    y_train = series.target.to_numpy(dtype=float)
    if not np.isfinite(y_train).all():
        raise ValueError("Missing/nonfinite classical training target")
    if task.model_family == "sarima":
        result = SARIMAPredictor(
            order["order"], order["seasonal_order"], order["trend"]
        ).fit_and_forecast(y_train, [task.identity.horizon])
    else:
        result = _fit_sarimax(series, y_train, (task, order))
    return _build_classical_outcome(
        result, series, (task.identity.horizon, 5 if task.model_family == "sarimax" else 0)
    )


def _fit_sarimax(
    series: pd.DataFrame, targets: np.ndarray, context: tuple[HoldoutTask, JsonObject]
) -> JsonObject:
    task, order = context
    exogenous = series[list(MACRO_FEATURE_COLUMN_MAP.values())].to_numpy(dtype=float)
    if not np.isfinite(exogenous).all():
        raise ValueError("Missing origin-safe SARIMAX regressors; fallback fails closed")
    future = np.tile(exogenous[-1], (task.identity.horizon, 1))
    return SARIMAXPredictor(
        order["order"], order["seasonal_order"], order["trend"]
    ).fit_and_forecast(targets, exogenous, future, [task.identity.horizon])


def _fit_baseline(series: pd.DataFrame, task: HoldoutTask) -> ForecastOutcome:
    source = str(pd.Period(task.identity.forecast_period, freq="M") - 12)
    if source > task.identity.forecast_origin:
        raise ValueError("Seasonal source beyond origin")
    rows = series.loc[series.period.eq(source)]
    if len(rows) != 1:
        raise ValueError("Missing seasonal baseline source")
    return ForecastOutcome(float(rows.target.iloc[0]), train_end=task.identity.forecast_origin)


def fit_holdout_model(
    features: pd.DataFrame,
    task: HoldoutTask,
    context: tuple[HoldoutInputs, PrequentialResidualCalibrator],
) -> ForecastOutcome:
    """Fit frozen parameters at one origin. Args: features, task, context. Returns: outcome. Raises: none."""
    series = _get_series(features, task)
    try:
        if task.model_family == "seasonal_naive":
            outcome = _fit_baseline(series, task)
        elif task.model_family == "xgboost":
            outcome = _fit_xgboost(series, task, context)
        else:
            outcome = _fit_classical(series, task, context[0])
    except Exception as error:
        logger.exception("Holdout fit failed: %s/%s", task.identity, task.model_family)
        return ForecastOutcome(
            np.nan,
            fit_status="FAILED",
            converged=False,
            failure_message=f"{type(error).__name__}: {error}",
        )
    return _validate_outcome(outcome, task.identity.forecast_origin)


def _get_actual(inputs: HoldoutInputs, identity: ForecastIdentity) -> float:
    rows = inputs.targets.loc[
        inputs.targets.direction.eq(identity.direction)
        & inputs.targets.hs2.eq(identity.hs2)
        & inputs.targets.period.eq(identity.forecast_period)
    ]
    if len(rows) != 1 or not np.isfinite(rows.trade_value_usd.iloc[0]):
        raise ValueError("Missing/nonfinite holdout outcome")
    return float(rows.trade_value_usd.iloc[0])


def _build_prediction_row(
    task: HoldoutTask, outcome: ForecastOutcome, scoring: tuple[str, float, ForecastOutcome]
) -> JsonObject:
    role, actual, fallback = scoring
    is_valid = outcome.fit_status == "SUCCESS" and outcome.converged
    has_fallback = _can_use_fallback(role, outcome, fallback)
    row = {
        **_build_forecast_identity_fields(task, role, actual),
        "prediction": outcome.prediction,
        **_build_fit_audit_fields(outcome, is_valid),
        "has_operational_fallback": has_fallback,
        "operational_prediction": fallback.prediction
        if has_fallback
        else outcome.prediction
        if is_valid
        else np.nan,
        "pre_freeze_origin": task.identity.forecast_origin < "2024-12",
        "feature_count": outcome.feature_count,
    }
    row.update(
        {name: getattr(outcome, name) for name in ("lower_80", "upper_80", "lower_95", "upper_95")}
    )
    row.update(_build_operational_bounds(task, (outcome, fallback), (is_valid, has_fallback)))
    row.update(_build_errors_and_coverage(actual, outcome))
    return row


def _build_errors_and_coverage(actual: float, outcome: ForecastOutcome) -> JsonObject:
    absolute = abs(actual - outcome.prediction)
    denominator = abs(actual) + abs(outcome.prediction)
    has_interval = np.isfinite(
        [outcome.lower_80, outcome.upper_80, outcome.lower_95, outcome.upper_95]
    ).all()
    return {
        "residual": actual - outcome.prediction,
        "absolute_error": absolute,
        "squared_error": absolute**2,
        "smape_component": 200 * absolute / denominator if denominator else 0.0,
        "interval_available": bool(has_interval),
        "covered_80": bool(outcome.lower_80 <= actual <= outcome.upper_80)
        if has_interval
        else None,
        "covered_95": bool(outcome.lower_95 <= actual <= outcome.upper_95)
        if has_interval
        else None,
    }


def _run_task(
    inputs: HoldoutInputs, entry: ChampionEntry, features: dict[str, pd.DataFrame]
) -> TaskRows:
    origins = generate_holdout_origins(entry)
    models = _build_role_assignments(entry)
    calibrators = {
        family: initialize_calibrator(inputs.cv_predictions, origins[0])
        for family in sorted(set(models.values()))
    }
    records, shap_rows = [], []
    for identity in origins:
        for calibrator in calibrators.values():
            _admit_realized_cv(inputs.cv_predictions, identity, calibrator)
        outcomes = {
            family: fit_holdout_model(
                features[identity.forecast_origin],
                HoldoutTask(entry, identity, family),
                (inputs, calibrators[family]),
            )
            for family in sorted(set(models.values()))
        }
        rows, attributions = _record_origin_results(
            inputs, (entry, identity, models), (outcomes, calibrators)
        )
        records.extend(rows)
        shap_rows.extend(attributions)
    return records, shap_rows


def run_holdout_predictions(inputs: HoldoutInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every frozen task/role. Args: inputs. Returns: predictions, SHAP. Raises: validation errors."""
    origins = sorted(
        {
            identity.forecast_origin
            for entry in inputs.registry.champions
            for identity in generate_holdout_origins(entry)
        }
    )
    features = {origin: build_origin_features(inputs, origin) for origin in origins}
    records, shap_rows = [], []
    for entry in inputs.registry.champions:
        rows, attributions = _run_task(inputs, entry, features)
        records.extend(rows)
        shap_rows.extend(attributions)
    predictions = (
        pd.DataFrame(records)
        .sort_values(["sector_id", "horizon", "forecast_origin", "role"])
        .reset_index(drop=True)
    )
    if (
        len(predictions) != 1512
        or predictions.duplicated(["sector_id", "horizon", "forecast_origin", "role"]).any()
    ):
        raise ValueError("Incomplete holdout role matrix")
    return predictions, pd.DataFrame(shap_rows).sort_values(
        ["sector_id", "horizon", "forecast_origin", "feature"]
    ).reset_index(drop=True)


def compute_holdout_metrics(group: pd.DataFrame) -> JsonObject:
    """Summarize raw primary-model performance, with denominators. Args: rows. Raises: schema errors."""
    valid = group.loc[group.is_valid_prediction]
    intervals = valid.loc[valid.interval_available]
    scores = {
        "n_expected_origins": len(group),
        "n_valid_origins": len(valid),
        "n_failed_origins": len(group) - len(valid),
        "mean_smape": float(valid.smape_component.mean()),
        "std_smape": float(valid.smape_component.std(ddof=0)),
        "median_smape": float(valid.smape_component.median()),
        "max_fold_smape": float(valid.smape_component.max()),
        "mean_mae": float(valid.absolute_error.mean()),
        "mean_rmse": float(np.sqrt(valid.squared_error.mean())),
        **_compute_raw_and_served_metrics(group),
        "n_interval_origins": len(intervals),
        "empirical_coverage_80": float(intervals.covered_80.mean()) if len(intervals) else np.nan,
        "empirical_coverage_95": float(intervals.covered_95.mean()) if len(intervals) else np.nan,
        "mean_interval_width_80": float((intervals.upper_80 - intervals.lower_80).mean())
        if len(intervals)
        else np.nan,
        "mean_interval_width_95": float((intervals.upper_95 - intervals.lower_95).mean())
        if len(intervals)
        else np.nan,
    }
    return scores


def build_holdout_leaderboard(
    predictions: pd.DataFrame, entries: tuple[ChampionEntry, ...]
) -> pd.DataFrame:
    """Return 42 task rows, including fallback and baseline metrics. Args: predictions, entries. Raises: errors."""
    rows = []
    for entry in entries:
        task = predictions.loc[
            predictions.sector_id.eq(entry.sector_id) & predictions.horizon.eq(entry.horizon)
        ]
        metrics = {
            role: compute_holdout_metrics(task.loc[task.role.eq(role)])
            for role in ("champion", "fallback", "baseline")
        }
        row = _build_leaderboard_row(entry, task, metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sector_id", "horizon"]).reset_index(drop=True)


def _build_xgb_outcome(
    training: pd.DataFrame, result: tuple[float, JsonObject, list[JsonObject]], horizon: int
) -> ForecastOutcome:
    prediction, interval, shap_rows = result
    return ForecastOutcome(
        prediction,
        **{
            name: float(interval[name]) if interval[name] is not None else np.nan
            for name in ("lower_80", "upper_80", "lower_95", "upper_95")
        },
        calibration_count=int(interval["calibration_count"]),
        max_calibration_target_period=interval["max_calibration_target_period"],
        train_end=str(pd.Period(training.period.max(), freq="M") + horizon),
        shap_rows=tuple(shap_rows),
        feature_count=21,
    )


def _validate_outcome(outcome: ForecastOutcome, origin: str) -> ForecastOutcome:
    if outcome.fit_status != "SUCCESS":
        return outcome
    if not outcome.converged:
        return replace(outcome, fit_status="NON_CONVERGED")
    if not np.isfinite(outcome.prediction) or outcome.prediction < 0:
        return replace(
            outcome,
            fit_status="INVALID_PREDICTION",
            failure_message="Raw point forecast is negative or nonfinite",
        )
    if outcome.train_end > origin:
        raise ValueError("Holdout training beyond origin")
    return outcome


def _build_forecast_identity_fields(task: HoldoutTask, role: str, actual: float) -> JsonObject:
    return {
        "sector_id": task.entry.sector_id,
        "direction": task.entry.direction,
        "hs2": task.entry.hs2,
        "horizon": task.entry.horizon,
        "forecast_origin": task.identity.forecast_origin,
        "origin_date": pd.Period(task.identity.forecast_origin, freq="M")
        .end_time.date()
        .isoformat(),
        "forecast_period": task.identity.forecast_period,
        "role": role,
        "model_family": task.model_family,
        "actual": actual,
    }


def _record_origin_results(
    inputs: HoldoutInputs,
    task: tuple[ChampionEntry, ForecastIdentity, dict[str, str]],
    context: tuple[dict[str, ForecastOutcome], dict[str, PrequentialResidualCalibrator]],
) -> TaskRows:
    entry, identity, models = task
    outcomes, calibrators = context
    records, shap_rows = [], []
    actual = _get_actual(inputs, identity)
    for role, family in models.items():
        task = HoldoutTask(entry, identity, family)
        records.append(
            _build_prediction_row(
                task, outcomes[family], (role, actual, outcomes[entry.fallback_model])
            )
        )
    for family, outcome in outcomes.items():
        shap_rows.extend(
            {**row, "model_family": family, "sector_id": entry.sector_id}
            for row in outcome.shap_rows
        )
        if family == "xgboost" and outcome.fit_status == "SUCCESS":
            calibrators[family].record_historical_prediction(
                identity.forecast_origin, identity.forecast_period, actual, outcome.prediction
            )
    return records, shap_rows


def _admit_realized_cv(
    cv: pd.DataFrame, identity: ForecastIdentity, calibrator: PrequentialResidualCalibrator
) -> None:
    known = {(row["forecast_origin"], row["forecast_period"]) for row in calibrator.history}
    eligible = initialize_calibrator(cv, identity)
    for row in eligible.history:
        if (row["forecast_origin"], row["forecast_period"]) not in known:
            calibrator.record_historical_prediction(
                row["forecast_origin"], row["forecast_period"], row["actual"], row["prediction"]
            )


def _build_leaderboard_row(
    entry: ChampionEntry, task: pd.DataFrame, metrics: dict[str, JsonObject]
) -> JsonObject:
    champion = metrics["champion"]
    paired_baseline = _compute_paired_baseline(task)
    skill = (
        compute_skill(paired_baseline, float(champion["mean_smape"]))[1]
        if champion["n_valid_origins"]
        else np.nan
    )
    row = {
        "sector_id": entry.sector_id,
        "direction": entry.direction,
        "hs2": entry.hs2,
        "horizon": entry.horizon,
        "champion_model": entry.champion_model,
        "fallback_model": entry.fallback_model,
        **champion,
        "skill_vs_naive_pct": skill,
        "paired_baseline_mean_smape": paired_baseline,
        "cv_mean_smape": entry.mean_smape,
        "cv_vs_holdout_smape_drift": champion["mean_smape"] - entry.mean_smape,
        "n_operational_fallbacks": int(
            task.loc[task.role.eq("champion"), "has_operational_fallback"].sum()
        ),
    }
    for role in ("fallback", "baseline"):
        row.update({f"{role}_{key}": value for key, value in metrics[role].items()})
    return row


def _build_fit_audit_fields(outcome: ForecastOutcome, is_valid: bool) -> JsonObject:
    return {
        "fit_status": outcome.fit_status,
        "converged": outcome.converged,
        "is_valid_prediction": is_valid,
        "failure_message": outcome.failure_message,
        "optimizer_used": outcome.optimizer_used,
        "retry_count": outcome.retry_count,
        "warning_type": outcome.warning_type,
        "train_end": outcome.train_end,
        "train_start": "2012-01",
        "calibration_count": outcome.calibration_count,
        "max_calibration_target_period": outcome.max_calibration_target_period,
    }


def _compute_paired_baseline(task: pd.DataFrame) -> float:
    eligible = task.loc[task.role.eq("champion") & task.is_valid_prediction, "forecast_origin"]
    return float(
        task.loc[
            task.role.eq("baseline") & task.forecast_origin.isin(eligible), "smape_component"
        ].mean()
    )


def _build_role_assignments(entry: ChampionEntry) -> dict[str, str]:
    return {
        "champion": entry.champion_model,
        "fallback": entry.fallback_model,
        "baseline": "seasonal_naive",
    }


def _compute_raw_and_served_metrics(group: pd.DataFrame) -> JsonObject:
    raw = group.loc[np.isfinite(group.prediction)]
    served = (
        group.loc[np.isfinite(group.operational_prediction) & group.operational_prediction.ge(0)]
        if "operational_prediction" in group
        else group.iloc[:0]
    )
    errors = (
        served.actual.sub(served.operational_prediction).abs()
        if len(served)
        else pd.Series(dtype=float)
    )
    denominators = (
        served.actual.abs().add(served.operational_prediction.abs())
        if len(served)
        else pd.Series(dtype=float)
    )
    return {
        "n_finite_raw_origins": len(raw),
        "raw_mean_smape": float(raw.smape_component.mean()),
        "n_valid_operational_origins": len(served),
        "operational_mean_smape": float(
            errors.div(denominators.where(denominators.ne(0), 1)).mul(200).mean()
        ),
        "operational_mean_mae": float(errors.mean()),
        "operational_mean_rmse": float(np.sqrt(errors.pow(2).mean())),
    }


def _build_classical_outcome(
    result: JsonObject, series: pd.DataFrame, context: tuple[int, int]
) -> ForecastOutcome:
    horizon, feature_count = context
    forecast = result["forecasts"].get(horizon, {})
    return ForecastOutcome(
        float(forecast.get("prediction", np.nan)),
        **{
            name: float(forecast.get(name, np.nan))
            for name in ("lower_80", "upper_80", "lower_95", "upper_95")
        },
        fit_status=str(result["fit_status"]),
        converged=bool(result["converged"]),
        failure_message=str(result.get("failure_message", "")),
        optimizer_used=str(result.get("optimizer_used", "unknown")),
        retry_count=int(result.get("retry_count", 0)),
        warning_type=str(result.get("warning_type", "NONE")),
        train_end=str(series.period.max()),
        feature_count=feature_count,
    )


def _can_use_fallback(role: str, outcome: ForecastOutcome, fallback: ForecastOutcome) -> bool:
    return (
        role == "champion"
        and not (outcome.fit_status == "SUCCESS" and outcome.converged)
        and fallback.fit_status == "SUCCESS"
        and fallback.converged
    )


def _build_operational_bounds(
    task: HoldoutTask, outcomes: tuple[ForecastOutcome, ForecastOutcome], flags: tuple[bool, bool]
) -> JsonObject:
    outcome, fallback = outcomes
    is_valid, has_fallback = flags
    served = fallback if has_fallback else outcome
    bounds = dict(
        {
            f"operational_{name}": getattr(served, name) if has_fallback or is_valid else np.nan
            for name in ("lower_80", "upper_80", "lower_95", "upper_95")
        }
    )
    bounds["operational_model_family"] = (
        task.entry.fallback_model if has_fallback else task.model_family
    )
    return bounds
