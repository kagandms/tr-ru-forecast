"""Behavioral and CV-boundary tests for frozen M10 selection."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from tr_ru_forecast.evaluation.champion_io import (
    ChampionRegistry,
    build_model_reference,
    build_registry_entry,
    collect_provenance,
    hash_artifact,
    load_cv_predictions,
    load_frozen_folds,
    load_selection_spec,
    load_task_grid,
    normalize_cv_predictions,
    resolve_cv_input,
    validate_registry,
)
from tr_ru_forecast.evaluation.champion_report import (
    build_m10_report,
    build_matrix_table,
    build_win_share_table,
)
from tr_ru_forecast.evaluation.champion_selector import (
    MODEL_FAMILIES,
    Candidate,
    ModelFamily,
    SelectionSpec,
    aggregate_candidates,
    build_champion_matrix,
    compute_skill,
    select_champion,
    validate_selection_spec,
)
from tr_ru_forecast.evaluation.cli import main, run_m10_selection
from tr_ru_forecast.evaluation.splits import generate_validation_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selection_spec() -> SelectionSpec:
    return load_selection_spec(PROJECT_ROOT / "configs/champion_selection_spec_v1_1.yaml")


def make_candidate(family: ModelFamily, smape: float, **overrides: object) -> Candidate:
    values: dict[str, object] = {
        "direction": "TR_RU",
        "hs2": "08",
        "is_total": False,
        "horizon": 1,
        "model_family": family,
        "complexity_tier": MODEL_FAMILIES.index(family),
        "mean_smape": smape,
        "std_smape": 5.0,
        "median_smape": smape,
        "max_fold_smape": 60.0,
        "mean_mae": 10.0,
        "mean_rmse": 12.0,
        "n_expected_folds": 3,
        "n_observed_folds": 3,
        "n_valid_folds": 3,
        "n_negative_predictions": 0,
        "convergence_rate": 1.0,
        "eligibility_status": "ELIGIBLE",
    }
    return Candidate.model_validate({**values, **overrides})


def make_predictions(family: ModelFamily) -> pd.DataFrame:
    calendar = generate_validation_folds().query("horizon == 1").head(3)
    predictions = calendar.assign(
        direction="TR_RU",
        hs2="08",
        is_total=False,
        model_type=family,
        actual_value=[100.0, 0.0, 300.0],
        prediction=[90.0, 0.0, 330.0],
        converged=True,
        fit_status="SUCCESS",
    )
    return predictions


def make_task_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "direction": "TR_RU",
                "hs2": "08",
                "is_total": False,
                "horizon": 1,
                "n_expected_folds": 3,
            }
        ]
    )


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected"),
    [
        (40, 30, (0.25, 25.0)),
        (10, 20, (-1.0, -100.0)),
        (0, 0, (0.0, 0.0)),
        (0, 10, (-1.0, -100.0)),
        (20, 20, (0.0, 0.0)),
    ],
)
def test_should_compute_relative_skill_with_explicit_zero_policy(
    baseline: float, candidate: float, expected: tuple[float, float]
) -> None:
    result = compute_skill(baseline, candidate)

    assert result == expected


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_should_reject_invalid_skill_inputs(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compute_skill(20, invalid)


def test_should_aggregate_all_origin_errors_and_preserve_pooled_rmse(
    selection_spec: SelectionSpec,
) -> None:
    calendar = generate_validation_folds()
    frames = [
        normalize_cv_predictions(make_predictions(family), calendar, family)
        for family in ("seasonal_naive", "sarima", "sarimax")
    ]

    candidates = aggregate_candidates(pd.concat(frames), make_task_grid(), selection_spec)

    baseline = candidates[0]
    assert baseline.mean_smape == pytest.approx(np.mean([200 * 10 / 190, 0, 200 * 30 / 630]))
    assert baseline.std_smape == pytest.approx(np.std([200 * 10 / 190, 0, 200 * 30 / 630]))
    assert baseline.mean_mae == pytest.approx(40 / 3)
    assert baseline.mean_rmse == pytest.approx(np.sqrt(1000 / 3))
    assert baseline.n_valid_folds == 3
    assert baseline.skill_score_smape == 0
    assert len(candidates) == 4
    assert candidates[3].eligibility_status == "INCOMPLETE"
    assert candidates[3].mean_smape is None


@pytest.mark.parametrize(
    ("prediction", "is_converged", "expected"),
    [
        (-1.0, True, "INVALID_PREDICTION"),
        (np.inf, True, "INVALID_PREDICTION"),
        (np.nan, True, "INVALID_PREDICTION"),
        (90.0, False, "FIT_FAILURE"),
    ],
)
def test_should_exclude_invalid_or_failed_candidates(
    prediction: float, is_converged: bool, expected: str, selection_spec: SelectionSpec
) -> None:
    calendar = generate_validation_folds()
    baseline = normalize_cv_predictions(
        make_predictions("seasonal_naive"), calendar, "seasonal_naive"
    )
    altered = make_predictions("sarima")
    altered.loc[altered.index[0], ["prediction", "converged"]] = [prediction, is_converged]
    candidate_rows = normalize_cv_predictions(altered, calendar, "sarima")

    candidates = aggregate_candidates(
        pd.concat([baseline, candidate_rows]), make_task_grid(), selection_spec
    )
    decision = select_champion(candidates, selection_spec)

    assert candidates[1].eligibility_status == expected
    assert candidates[1].n_valid_folds == 2
    assert decision.champion.model_family == "seasonal_naive"
    assert decision.fallback.model_family == "seasonal_naive"


def test_should_fail_closed_on_incomplete_baseline(selection_spec: SelectionSpec) -> None:
    normalized = normalize_cv_predictions(
        make_predictions("seasonal_naive").head(2), generate_validation_folds(), "seasonal_naive"
    )

    with pytest.raises(ValueError, match="complete, valid baseline"):
        aggregate_candidates(normalized, make_task_grid(), selection_spec)


@pytest.mark.parametrize(
    ("complex_smape", "winner", "reason"),
    [
        (49.1, "sarima", "PARSIMONY_PREFERENCE"),
        (49.0, "xgboost", "CLEAR_SMAPE_WINNER"),
        (48.9, "xgboost", "CLEAR_SMAPE_WINNER"),
    ],
)
def test_should_apply_strict_two_percent_relative_parsimony(
    complex_smape: float, winner: str, reason: str, selection_spec: SelectionSpec
) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 50),
        make_candidate("xgboost", complex_smape),
    )

    decision = select_champion(candidates, selection_spec)

    assert decision.champion.model_family == winner
    assert decision.raw_best_model == "xgboost"
    assert decision.selection_reason == reason


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"mean_mae": 9}, "TIE_BREAK_MAE"),
        ({"mean_rmse": 11}, "TIE_BREAK_RMSE"),
        ({"std_smape": 4}, "TIE_BREAK_STABILITY"),
        ({}, "TIE_BREAK_COMPLEXITY"),
    ],
)
def test_should_use_protocol_mae_before_rmse_and_deterministic_extensions(
    overrides: dict[str, float], reason: str, selection_spec: SelectionSpec
) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 50, **overrides),
        make_candidate("xgboost", 50),
    )

    decision = select_champion(candidates, selection_spec)

    assert decision.champion.model_family == "sarima"
    assert decision.selection_reason == reason


def test_should_record_raw_mae_winner_when_parsimony_prefers_simpler(
    selection_spec: SelectionSpec,
) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 50),
        make_candidate("xgboost", 50, mean_mae=9, mean_rmse=100),
    )

    decision = select_champion(candidates, selection_spec)

    assert decision.raw_best_model == "xgboost"
    assert decision.champion.model_family == "sarima"
    assert decision.selection_reason == "PARSIMONY_PREFERENCE"


def test_should_flag_worst_origin_without_changing_frozen_selection(
    selection_spec: SelectionSpec,
) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 50, max_fold_smape=140),
        make_candidate("xgboost", 49.5, max_fold_smape=70),
    )

    decision = select_champion(candidates, selection_spec)

    assert decision.champion.model_family == "sarima"
    assert decision.has_stability_flag


def test_should_not_compare_stability_against_ineligible_models(
    selection_spec: SelectionSpec,
) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 50, max_fold_smape=140),
        make_candidate("xgboost", 49.5, max_fold_smape=1, eligibility_status="FIT_FAILURE"),
    )

    decision = select_champion(candidates, selection_spec)

    assert not decision.has_stability_flag


def test_should_use_best_lower_complexity_fallback(selection_spec: SelectionSpec) -> None:
    candidates = (
        make_candidate("seasonal_naive", 100),
        make_candidate("sarima", 60),
        make_candidate("sarimax", 55),
        make_candidate("xgboost", 40),
    )

    decision = select_champion(candidates, selection_spec)

    assert decision.champion.model_family == "xgboost"
    assert decision.fallback.model_family == "sarimax"
    assert decision.baseline_reference.model_family == "seasonal_naive"


def test_should_be_invariant_to_candidate_order(selection_spec: SelectionSpec) -> None:
    candidates = tuple(
        make_candidate(family, smape)
        for family, smape in zip(MODEL_FAMILIES, (100, 50, 70, 49.5), strict=True)
    )
    expected = select_champion(candidates, selection_spec)

    decisions = [
        select_champion(order, selection_spec) for order in itertools.permutations(candidates)
    ]

    assert all(decision == expected for decision in decisions)
    assert build_champion_matrix(candidates, selection_spec) == (expected,)


@pytest.mark.parametrize("change", ["mixed", "duplicate", "no_baseline"])
def test_should_reject_ambiguous_task_selection(change: str, selection_spec: SelectionSpec) -> None:
    baseline = make_candidate("seasonal_naive", 100)
    candidate = make_candidate("sarima", 50)
    candidates = {
        "mixed": (baseline, candidate.model_copy(update={"horizon": 3})),
        "duplicate": (baseline, baseline),
        "no_baseline": (candidate,),
    }[change]

    with pytest.raises(ValueError):
        select_champion(candidates, selection_spec)


@pytest.mark.parametrize("change", ["post_cv", "duplicate", "future_training", "wrong_family"])
def test_should_reject_leakage_and_mismatched_prediction_keys(change: str) -> None:
    predictions = make_predictions("sarima")
    if change == "post_cv":
        predictions.loc[predictions.index[0], "forecast_period"] = "2025-01"
    if change == "duplicate":
        predictions = pd.concat([predictions, predictions.head(1)])
    if change == "future_training":
        predictions.loc[predictions.index[0], "train_end"] = "2025-01"
    if change == "wrong_family":
        predictions.loc[predictions.index[0], "model_type"] = "sarimax"

    with pytest.raises(ValueError):
        normalize_cv_predictions(predictions, generate_validation_folds(), "sarima")


@pytest.mark.parametrize(
    "relative",
    [
        "data/processed/forecast_target_monthly.parquet",
        "../m6_seasonal_naive_predictions.parquet",
        "data/raw/m6_seasonal_naive_predictions.parquet",
    ],
)
def test_should_reject_non_cv_paths(relative: str) -> None:
    with pytest.raises(ValueError, match="Forbidden CV input"):
        resolve_cv_input(PROJECT_ROOT, relative, "m6_seasonal_naive_predictions.parquet")


def test_should_reject_protocol_drift(selection_spec: SelectionSpec) -> None:
    altered = selection_spec.model_copy(update={"tie_breaks": ("mean_smape", "mean_rmse")})

    with pytest.raises(ValueError, match="MAE before"):
        validate_selection_spec(altered)
    with pytest.raises(ValidationError):
        SelectionSpec.model_validate(
            {**selection_spec.model_dump(), "parsimony_relative_threshold": 0.03}
        )


def test_should_reject_missing_hash_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_artifact(tmp_path / "missing.parquet")


@pytest.fixture(scope="module")
def generated_registry(tmp_path_factory: pytest.TempPathFactory) -> tuple[ChampionRegistry, Path]:
    directory = tmp_path_factory.mktemp("m10-output")
    return run_m10_selection(PROJECT_ROOT, directory), directory


def test_should_write_full_registry_and_artifact_checksums(
    generated_registry: tuple[ChampionRegistry, Path],
) -> None:
    registry, directory = generated_registry
    loaded = ChampionRegistry.model_validate(
        yaml.safe_load((directory / "champion_registry_v1_1.yaml").read_text())
    )

    validate_registry(loaded, load_task_grid(PROJECT_ROOT))

    assert loaded == registry
    assert len(pd.read_parquet(directory / "m10_cv_leaderboard.parquet")) == 168
    assert len(pd.read_parquet(directory / "m10_cv_fold_details.parquet")) == 15736
    assert len(pd.read_parquet(directory / "m10_champion_matrix.parquet")) == 42
    assert not registry.provenance.holdout_accessed
    assert registry.provenance.legacy_metadata_discrepancies
    manifest = json.loads((directory / "m10_selection_run.json").read_text())
    for name, expected in manifest["output_hashes"].items():
        assert hash_artifact(directory / name) == expected


@pytest.mark.parametrize("change", ["duplicate", "missing", "nan", "wrong_count"])
def test_should_reject_invalid_registry_schema(
    change: str, generated_registry: tuple[ChampionRegistry, Path]
) -> None:
    registry, _ = generated_registry
    payload = registry.model_dump(mode="json")
    if change == "duplicate":
        payload["champions"][0] = payload["champions"][1]
    if change == "missing":
        payload["champions"].pop()
    if change == "nan":
        payload["champions"][0]["mean_smape"] = np.nan
    if change == "wrong_count":
        payload["champions"][0]["n_valid_folds"] = 5

    with pytest.raises((ValueError, ValidationError)):
        validate_registry(ChampionRegistry.model_validate(payload), load_task_grid(PROJECT_ROOT))


def test_should_reproduce_decisions_without_reading_raw_or_holdout_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_registry: tuple[ChampionRegistry, Path],
) -> None:
    expected, original_directory = generated_registry
    original_read = pd.read_parquet
    accessed: list[Path] = []
    allowed = {"validation_origins.parquet", "m9_xgboost_fit_log.parquet"} | {
        "m6_seasonal_naive_predictions.parquet",
        "m7_sarima_predictions.parquet",
        "m8_sarimax_predictions.parquet",
        "m9_xgboost_predictions.parquet",
    }

    def guarded_read(path: Path, **options: object) -> pd.DataFrame:
        assert Path(path).name in allowed
        accessed.append(Path(path))
        return original_read(path, **options)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)

    actual = run_m10_selection(PROJECT_ROOT, tmp_path / "reproduction")

    assert actual.champions == expected.champions
    assert actual.provenance.input_hashes == expected.provenance.input_hashes
    assert actual.provenance.output_hashes == expected.provenance.output_hashes
    assert accessed
    assert hash_artifact(original_directory / "m10_champion_matrix.parquet") == hash_artifact(
        tmp_path / "reproduction/m10_champion_matrix.parquet"
    )


def test_should_validate_calendar_before_loading_prediction_values(
    monkeypatch: pytest.MonkeyPatch, selection_spec: SelectionSpec
) -> None:
    calendar = generate_validation_folds()
    original_read = pd.read_parquet
    projections: list[object] = []

    def poisoned_calendar(path: Path, **options: object) -> pd.DataFrame:
        projections.append(options.get("columns"))
        frame = original_read(path, **options)
        frame.loc[0, "forecast_period"] = "2025-01"
        return frame

    monkeypatch.setattr(pd, "read_parquet", poisoned_calendar)

    with pytest.raises(ValueError, match="holdout/post-CV"):
        load_cv_predictions(PROJECT_ROOT, selection_spec, (calendar, load_task_grid(PROJECT_ROOT)))

    assert len(projections) == 1
    assert "actual_value" not in projections[0]


def test_should_refuse_overwriting_frozen_outputs(
    generated_registry: tuple[ChampionRegistry, Path],
) -> None:
    _, directory = generated_registry
    original_hash = hash_artifact(directory / "champion_registry_v1_1.yaml")

    with pytest.raises(FileExistsError, match="already exist"):
        run_m10_selection(PROJECT_ROOT, directory)

    assert hash_artifact(directory / "champion_registry_v1_1.yaml") == original_hash


def test_should_address_frozen_config_and_baseline_interval_unavailability(
    selection_spec: SelectionSpec,
) -> None:
    baseline = make_candidate("seasonal_naive", 40)
    decision = select_champion((baseline,), selection_spec)

    reference = build_model_reference(PROJECT_ROOT, baseline, selection_spec)
    entry = build_registry_entry(PROJECT_ROOT, decision, selection_spec)

    assert reference.interval_policy == "unavailable_no_interval_method_approved"
    assert entry.champion_reference == reference
    assert entry.sector_id == "TR_RU:08"


def test_should_fail_on_m9_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, selection_spec: SelectionSpec
) -> None:
    original_hash = hash_artifact

    def altered_hash(path: Path) -> str:
        return "0" * 64 if path.name == "m9_xgboost_predictions.parquet" else original_hash(path)

    monkeypatch.setattr("tr_ru_forecast.evaluation.champion_io.hash_artifact", altered_hash)

    with pytest.raises(ValueError, match="M9 checksum mismatch"):
        collect_provenance(
            PROJECT_ROOT, selection_spec, PROJECT_ROOT / "configs/champion_selection_spec_v1_1.yaml"
        )


def test_should_report_all_decisions_and_actual_win_counts(
    generated_registry: tuple[ChampionRegistry, Path], selection_spec: SelectionSpec
) -> None:
    registry, directory = generated_registry
    candidates = tuple(
        Candidate.model_validate(row)
        for row in pd.read_parquet(directory / "m10_cv_leaderboard.parquet").to_dict("records")
    )

    report = build_m10_report(registry, candidates)

    assert "| sarima |" in build_win_share_table(registry)
    assert build_matrix_table(registry).count("\n") == 43
    assert "not five folds" in report
    assert "independent 2025 performance remains unknown" in report
    assert "M9 A2" in report
    assert "Legacy discrepancies" in report


def test_should_log_cli_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("sys.argv", ["m10-select", "--project-root", str(tmp_path)])

    with pytest.raises(FileNotFoundError):
        main()

    assert "M10 selection failed" in caplog.text


def test_should_verify_frozen_cv_calendar() -> None:
    result = load_frozen_folds(PROJECT_ROOT)

    assert len(result) == 281
    assert result.forecast_period.max() == "2024-12"
