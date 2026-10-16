"""Temporal isolation, frozen gate, coverage, fallback and immutable-output contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.evaluation.holdout_cli import run_m11_holdout
from tr_ru_forecast.evaluation.holdout_gate import (
    audit_m9_structure,
    validate_m8_calendar_equivalence,
)
from tr_ru_forecast.evaluation.holdout_io import (
    HoldoutInputs,
    build_holdout_provenance,
    load_holdout_inputs,
    load_holdout_spec,
    publish_holdout_outputs,
    validate_pre_m11_gate,
)
from tr_ru_forecast.evaluation.holdout_report import (
    build_shap_summary,
    render_table,
)
from tr_ru_forecast.evaluation.holdout_runner import (
    ForecastOutcome,
    HoldoutTask,
    _admit_realized_cv,
    _build_prediction_row,
    build_origin_features,
    compute_holdout_metrics,
    fit_holdout_model,
    generate_holdout_origins,
    initialize_calibrator,
)
from tr_ru_forecast.models.ml.records import ForecastIdentity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inputs() -> HoldoutInputs:
    return load_holdout_inputs(ROOT)


def make_cv() -> pd.DataFrame:
    months = pd.period_range("2020-01", "2021-12", freq="M")
    return pd.DataFrame(
        {
            "direction": "TR_RU",
            "hs2": "TOTAL",
            "horizon": 1,
            "forecast_origin": months.astype(str),
            "forecast_period": (months + 1).astype(str),
            "actual": 100.0,
            "prediction": 90.0,
        }
    )


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_should_cover_twelve_targets_with_exact_monthly_origins(
    horizon: int, inputs: HoldoutInputs
) -> None:
    entry = next(row for row in inputs.registry.champions if row.horizon == horizon)

    origins = generate_holdout_origins(entry)

    assert len(origins) == 12
    assert origins[0].forecast_period == "2025-01"
    assert origins[-1].forecast_period == "2025-12"
    assert all(
        str(pd.Period(row.forecast_origin, freq="M") + horizon) == row.forecast_period
        for row in origins
    )


def test_should_fail_gate_before_reading_target_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_gate(root: Path, spec: object) -> None:
        raise ValueError("test: closed gate")

    def forbid_read(*arguments: object, **options: object) -> pd.DataFrame:
        raise AssertionError("No target parquet read is allowed")

    monkeypatch.setattr("tr_ru_forecast.evaluation.holdout_io.validate_pre_m11_gate", reject_gate)
    monkeypatch.setattr(pd, "read_parquet", forbid_read)

    with pytest.raises(ValueError, match="closed gate"):
        load_holdout_inputs(ROOT)


def test_should_verify_gate_hashes_and_unchanged_assignments(inputs: HoldoutInputs) -> None:
    spec = load_holdout_spec(ROOT / "configs/holdout_spec_v1_1.yaml")

    registry = validate_pre_m11_gate(ROOT, spec)
    equivalence = validate_m8_calendar_equivalence(ROOT, registry)
    audit_m9_structure(ROOT)

    assert registry.status == "CV_SELECTION_FROZEN_M11_READY"
    assert registry.champions == inputs.registry.champions
    assert equivalence["recorded_yaml_sha256"] != equivalence["current_yaml_sha256"]
    assert "NON_CHAMPION" in equivalence["rule"]


def test_should_reject_frozen_config_checksum_drift(
    monkeypatch: pytest.MonkeyPatch, inputs: HoldoutInputs
) -> None:
    from tr_ru_forecast.evaluation import holdout_io

    original = holdout_io.hash_artifact

    def corrupted_hash(path: Path) -> str:
        return "0" * 64 if path.name == "sarima_orders_v1_1.yaml" else original(path)

    monkeypatch.setattr(holdout_io, "hash_artifact", corrupted_hash)

    with pytest.raises(ValueError, match="Frozen checksum mismatch"):
        validate_pre_m11_gate(ROOT, inputs.spec)


@pytest.mark.parametrize("field", ["holdout_start", "trade_lag_months", "calibration_policy"])
def test_should_reject_unfrozen_protocol(field: str, tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "configs/holdout_spec_v1_1.yaml").read_text())
    payload[field] = 1 if field == "trade_lag_months" else "altered"
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError):
        load_holdout_spec(path)


def test_should_ignore_all_post_origin_target_values(inputs: HoldoutInputs) -> None:
    origin = "2024-07"
    altered = inputs.targets.copy()
    altered.loc[altered.period.gt(origin), "trade_value_usd"] = 1e15

    original_features = build_origin_features(inputs, origin)
    changed_features = build_origin_features(replace(inputs, targets=altered), origin)

    pd.testing.assert_frame_equal(original_features, changed_features, check_exact=True)
    assert original_features.period.max() == origin
    assert original_features.loc[original_features.period.eq(origin), "target_h1"].isna().all()


def test_should_ignore_macro_values_not_available_at_origin(inputs: HoldoutInputs) -> None:
    origin = "2024-07"
    altered = inputs.macro.copy()
    altered.loc[altered.availability_date_or_approximation.gt("2024-07-31"), "value"] = 1e15

    original_features = build_origin_features(inputs, origin)
    changed_features = build_origin_features(replace(inputs, macro=altered), origin)

    pd.testing.assert_frame_equal(original_features, changed_features, check_exact=True)


def test_should_seed_only_realized_residuals_and_admit_later_cv_residuals() -> None:
    cv = make_cv()
    identity = ForecastIdentity("TR_RU", "TOTAL", 1, "2021-08", "2021-09")

    calibrator = initialize_calibrator(cv, identity)
    early = calibrator.calibrate(identity.forecast_origin, 100.0)
    _admit_realized_cv(cv, replace(identity, forecast_origin="2021-10"), calibrator)
    later = calibrator.calibrate("2021-10", 100.0)

    assert early["calibration_count"] == 19
    assert early["max_calibration_target_period"] == "2021-08"
    assert later["calibration_count"] == 21
    assert later["max_calibration_target_period"] == "2021-10"


def test_should_embargo_future_holdout_residual_and_never_mix_series() -> None:
    cv = make_cv()
    foreign = cv.assign(hs2="08", actual=1e15)
    identity = ForecastIdentity("TR_RU", "TOTAL", 1, "2021-12", "2022-01")
    calibrator = initialize_calibrator(pd.concat([cv, foreign]), identity)
    calibrator.record_historical_prediction("2021-12", "2022-01", 1e15, 0.0)

    interval = calibrator.calibrate("2021-12", 100.0)

    assert interval["q95"] == 10.0
    assert interval["max_calibration_target_period"] == "2021-12"
    assert interval["calibration_count"] == 23


def test_should_report_unavailable_intervals_when_historical_pool_is_small() -> None:
    identity = ForecastIdentity("TR_RU", "TOTAL", 1, "2020-10", "2020-11")

    interval = initialize_calibrator(make_cv(), identity).calibrate("2020-10", 100.0)

    assert interval["calibration_count"] == 9
    assert interval["interval_available"] is False


@pytest.mark.parametrize(
    ("primary", "fallback", "expected", "has_fallback"),
    [(-5.0, 100.0, 100.0, True), (80.0, 100.0, 80.0, False), (-5.0, -2.0, np.nan, False)],
)
def test_should_preserve_raw_invalid_primary_and_govern_registered_fallback(
    primary: float, fallback: float, expected: float, has_fallback: bool, inputs: HoldoutInputs
) -> None:
    entry = inputs.registry.champions[0]
    identity = generate_holdout_origins(entry)[0]
    task = HoldoutTask(entry, identity, entry.champion_model)
    outcome = ForecastOutcome(
        primary, fit_status="SUCCESS" if primary >= 0 else "INVALID_PREDICTION"
    )
    alternate = ForecastOutcome(
        fallback, fit_status="SUCCESS" if fallback >= 0 else "INVALID_PREDICTION"
    )

    row = _build_prediction_row(task, outcome, ("champion", 90.0, alternate))

    assert row["prediction"] == primary
    assert row["has_operational_fallback"] == has_fallback
    assert (
        np.isnan(row["operational_prediction"])
        if np.isnan(expected)
        else row["operational_prediction"] == expected
    )


def test_should_log_fitting_failure_without_silent_clipping(
    monkeypatch: pytest.MonkeyPatch, inputs: HoldoutInputs, caplog: pytest.LogCaptureFixture
) -> None:
    entry = inputs.registry.champions[0]
    identity = generate_holdout_origins(entry)[0]

    def failed_baseline(series: pd.DataFrame, task: HoldoutTask) -> ForecastOutcome:
        raise ValueError("missing source")

    monkeypatch.setattr("tr_ru_forecast.evaluation.holdout_runner._fit_baseline", failed_baseline)
    features = pd.DataFrame(
        {
            "direction": [entry.direction],
            "hs2": [entry.hs2],
            "period": ["2024-01"],
            "target": [100.0],
        }
    )

    result = fit_holdout_model(
        features,
        HoldoutTask(entry, identity, "seasonal_naive"),
        (inputs, initialize_calibrator(make_cv(), identity)),
    )

    assert not result.converged
    assert result.fit_status == "FAILED"
    assert np.isnan(result.prediction)
    assert "Holdout fit failed" in caplog.text


def test_should_compute_interval_coverage_only_on_available_valid_origins() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [10.0, 20.0, -30.0],
            "operational_prediction": [10.0, 20.0, np.nan],
            "actual": [11.0, 23.0, 70.0],
            "is_valid_prediction": [True, True, False],
            "interval_available": [True, True, True],
            "smape_component": [10.0, 20.0, 200.0],
            "absolute_error": [1.0, 3.0, 100.0],
            "squared_error": [1.0, 9.0, 10000.0],
            "covered_80": [True, False, False],
            "covered_95": [True, True, False],
            "lower_80": [0.0, 0.0, 0.0],
            "upper_80": [2.0, 4.0, 5.0],
            "lower_95": [-1.0, -1.0, -1.0],
            "upper_95": [3.0, 5.0, 6.0],
        }
    )

    summary = compute_holdout_metrics(frame)

    assert summary["mean_smape"] == 15.0
    assert summary["std_smape"] == 5.0
    assert summary["mean_rmse"] == pytest.approx(np.sqrt(5))
    assert summary["empirical_coverage_80"] == 0.5
    assert summary["empirical_coverage_95"] == 1.0
    assert summary["n_valid_origins"] == 2
    assert summary["n_failed_origins"] == 1
    assert summary["mean_interval_width_80"] == 3.0


def test_should_never_overwrite_published_artifacts(tmp_path: Path) -> None:
    table = pd.DataFrame({"prediction": [10.0]})
    hashes = publish_holdout_outputs(
        tmp_path, {"m11_predictions.parquet": table}, {"report.md": "verified"}
    )

    with pytest.raises(FileExistsError):
        publish_holdout_outputs(
            tmp_path, {"m11_predictions.parquet": table}, {"report.md": "changed"}
        )

    assert (tmp_path / "report.md").read_text() == "verified"
    assert len(hashes["report.md"]) == 64


def test_should_fail_fast_before_refitting_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "m11_holdout_run.json").write_text("{}")

    def prohibit_load(root: Path) -> HoldoutInputs:
        raise AssertionError("No refit/data loading on overwrite")

    monkeypatch.setattr("tr_ru_forecast.evaluation.holdout_cli.load_holdout_inputs", prohibit_load)

    with pytest.raises(FileExistsError):
        run_m11_holdout(ROOT, tmp_path)


def test_should_aggregate_signed_shap_without_pooling_sector_scales() -> None:
    local = pd.DataFrame(
        {
            "sector_id": ["TR_RU:08"] * 2,
            "horizon": [1] * 2,
            "model_family": ["xgboost"] * 2,
            "feature": ["lag_1"] * 2,
            "shap_value": [-10.0, 30.0],
        }
    )

    result = build_shap_summary(local)

    assert result.mean_abs_shap.iloc[0] == 20.0
    assert result.mean_signed_shap.iloc[0] == 10.0
    assert result.n_origins.iloc[0] == 2
    assert "TR_RU:08" in render_table(result)


def test_should_record_explicit_retrospective_scope_and_zero_retuning(
    inputs: HoldoutInputs,
) -> None:
    metadata = build_holdout_provenance(inputs)

    assert metadata["zero_retuning"] is True
    assert metadata["historical_vintage_fidelity_proven"] is False
    assert "retrospective" in metadata["scope"]
    assert len(metadata["source_fingerprint"]) == 64


def test_should_retain_native_failure_reason_and_optimizer_retry_audit() -> None:
    from tr_ru_forecast.evaluation.holdout_runner import _validate_outcome

    failed = ForecastOutcome(
        np.nan,
        fit_status="FIT_FAILED",
        converged=False,
        failure_message="optimizer exhausted",
        optimizer_used="powell",
        retry_count=1,
        warning_type="ConvergenceWarning",
    )

    result = _validate_outcome(failed, "2025-01")

    assert result == failed
    assert result.failure_message == "optimizer exhausted"
    assert result.retry_count == 1


def test_should_preserve_complete_final_holdout_matrix_and_audit_metadata(
    inputs: HoldoutInputs,
) -> None:
    directory = ROOT / "data/processed"
    predictions = pd.read_parquet(directory / "m11_holdout_predictions.parquet")
    leaderboard = pd.read_parquet(directory / "m11_holdout_leaderboard.parquet")
    manifest = json.loads((directory / "m11_holdout_run.json").read_text())

    assert len(predictions) == 1512
    assert len(leaderboard) == 42
    assert predictions.groupby(["sector_id", "horizon", "role"]).size().eq(12).all()
    assert predictions.train_end.le(predictions.forecast_origin).all()
    calibrated = predictions.loc[predictions.model_family.eq("xgboost")]
    assert calibrated.calibration_count.ge(19).all()
    assert calibrated.max_calibration_target_period.le(calibrated.forecast_origin).all()
    assert {"optimizer_used", "retry_count", "warning_type"}.issubset(predictions.columns)
    assert manifest["zero_retuning"] is True
    for name, expected in manifest["output_hashes"].items():
        from tr_ru_forecast.evaluation.champion_io import hash_artifact

        assert hash_artifact(directory / name) == expected


def test_should_reproduce_immutable_data_packages_in_independent_directories(
    tmp_path: Path,
) -> None:
    directory = ROOT / "data/processed"
    tables = {
        name: pd.read_parquet(directory / name)
        for name in (
            "m11_holdout_predictions.parquet",
            "m11_holdout_leaderboard.parquet",
            "m11_shap_summary.parquet",
        )
    }

    first = publish_holdout_outputs(tmp_path / "first", tables, {"report.md": "frozen"})
    second = publish_holdout_outputs(tmp_path / "second", tables, {"report.md": "frozen"})

    assert first == second
