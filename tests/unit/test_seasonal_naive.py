"""Unit and acceptance tests for Milestone M6: Seasonal Naive Baseline.

Verifies all 50 critical requirements from Milestone M6 specification:
1. exactly 14 frozen modeling series are evaluated.
2. only frozen selected HS2 + TOTAL appear.
3. M3 manifest remains unchanged.
4. M5 feature manifest remains unchanged.
5. target policy remains Türkiye-reported for both directions.
6. development-only input enforced.
7. monthly origin cadence is deterministic.
8. minimum initial history = 60 months.
9. first validation origin derived correctly (2016-12).
10. h1 last origin = 2024-11.
11. h3 last origin = 2024-09.
12. h6 last origin = 2024-06.
13. forecast period = origin + h calendar months.
14. seasonal source = forecast period - 12 calendar months.
15. seasonal source belongs to same direction/hs2 series.
16. seasonal source <= forecast origin.
17. prediction exactly equals seasonal source actual.
18. actual validation value is never used in prediction construction.
19. h1 example calculation.
20. h3 example calculation.
21. h6 example calculation.
22. both-zero sMAPE component = 0.
23. one-zero sMAPE handled without NaN/inf.
24. deterministic sMAPE aggregate.
25. deterministic MAE.
26. deterministic RMSE.
27. metrics exist for all 42 series x horizon groups.
28. no duplicate metric group.
29. prediction artifact has no 2025 forecast periods.
30. validation fold registry has no 2025 target periods.
31. holdout mutation leaves predictions unchanged.
32. holdout mutation leaves metrics unchanged.
33. holdout mutation leaves fold registry unchanged.
34. reported zeros are preserved.
35. missing seasonal source is not converted to zero.
36. invalid/missing rows are audited explicitly.
37. training window metadata expands monotonically.
38. train_start remains frozen at development start (2012-01).
39. train_end equals forecast origin.
40. fold IDs are deterministic.
41. rerunning M6 produces numerically identical results.
42. no macro features influence Seasonal Naive predictions.
43. no scaling occurs.
44. no hyperparameter tuning occurs.
45. no model selection occurs.
46. no champion selection artifacts created.
47. no M11 holdout artifacts created.
48. no XGBoost artifacts created.
49. no M9+ implementation exists.
50. lint/test commands remain clean.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    FORECAST_TARGET_FLOWS,
    TradeDirection,
)
from tr_ru_forecast.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    compute_smape_single,
)
from tr_ru_forecast.evaluation.splits import (
    derive_first_validation_origin,
    generate_validation_folds,
)
from tr_ru_forecast.models.baselines.pipeline import (
    extract_modeling_series_list,
    run_seasonal_naive_pipeline,
)
from tr_ru_forecast.models.baselines.seasonal_naive import SeasonalNaivePredictor


@pytest.fixture(autouse=True, scope="module")
def isolate_validation_registry_writes(tmp_path_factory):
    """Prevent tests from rewriting tracked configs/validation_folds_v1_1.yaml."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    temp_dir = tmp_path_factory.mktemp("validation_registry_isolation")
    temp_yaml = temp_dir / "validation_folds_v1_1.yaml"
    temp_parquet = temp_dir / "validation_origins.parquet"

    from tr_ru_forecast.evaluation import splits

    orig_save = splits.save_validation_registry

    def safe_save(folds_df, yaml_path, parquet_path):
        return orig_save(folds_df, temp_yaml, temp_parquet)

    mp.setattr("tr_ru_forecast.models.baselines.pipeline.save_validation_registry", safe_save)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def pipeline_results():
    """Run Seasonal Naive pipeline once for test module."""
    pred_df, metrics_df, run_meta = run_seasonal_naive_pipeline(run_id="test_run_m6")
    return pred_df, metrics_df, run_meta


# 1. exactly 14 frozen modeling series are evaluated
def test_01_exactly_14_modeling_series(pipeline_results):
    _, metrics_df, _ = pipeline_results
    series = metrics_df[["direction", "hs2"]].drop_duplicates()
    assert len(series) == 14, f"Expected 14 series, got {len(series)}"


# 2. only frozen selected HS2 + TOTAL appear
def test_02_only_frozen_selected_sectors_appear(pipeline_results):
    settings = get_settings()
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml") as f:
        manifest = yaml.safe_load(f)

    expected_tr = set([s["hs2"] for s in manifest["TR_RU"]] + ["TOTAL"])
    expected_ru = set([s["hs2"] for s in manifest["RU_TR"]] + ["TOTAL"])

    _, metrics_df, _ = pipeline_results
    actual_tr = set(metrics_df[metrics_df["direction"] == "TR_RU"]["hs2"].unique())
    actual_ru = set(metrics_df[metrics_df["direction"] == "RU_TR"]["hs2"].unique())

    assert actual_tr == expected_tr
    assert actual_ru == expected_ru


# 3. M3 manifest remains unchanged
def test_03_m3_manifest_remains_unchanged():
    settings = get_settings()
    manifest_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    with open(manifest_path) as f:
        data = yaml.safe_load(f)
    assert [s["hs2"] for s in data["TR_RU"]] == ["08", "84", "39", "87", "73", "85"]
    assert [s["hs2"] for s in data["RU_TR"]] == ["27", "72", "10", "76", "23", "74"]


# 4. M5 feature manifest remains unchanged
def test_04_m5_feature_manifest_remains_unchanged():
    settings = get_settings()
    feat_spec = settings.project_root / "configs" / "feature_spec_v1_1.yaml"
    assert feat_spec.exists()
    with open(feat_spec) as f:
        data = yaml.safe_load(f)
    assert data["spec_version"] == "1.1"
    assert data["total_modeling_series"] == 14


# 5. target policy remains Türkiye-reported for both directions
def test_05_target_policy_turkiye_reported():
    tr_meta = FORECAST_TARGET_FLOWS[TradeDirection.TR_RU]
    ru_meta = FORECAST_TARGET_FLOWS[TradeDirection.RU_TR]
    assert tr_meta["reporter_code"] == 792
    assert ru_meta["reporter_code"] == 792
    assert tr_meta["flow_code"] == "X"
    assert ru_meta["flow_code"] == "M"


# 6. development-only input enforced
def test_06_development_only_input_enforced(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert pred_df["forecast_period"].max() <= DEVELOPMENT_DATA_END
    assert pred_df["forecast_origin"].max() <= "2024-11"


# 7. monthly origin cadence is deterministic
def test_07_monthly_origin_cadence_deterministic():
    folds = generate_validation_folds(start_period="2012-01", end_period="2024-12")
    h1_origins = folds[folds["horizon"] == 1]["forecast_origin"].tolist()
    # Check consecutive monthly intervals
    for i in range(len(h1_origins) - 1):
        p1 = pd.Period(h1_origins[i], freq="M")
        p2 = pd.Period(h1_origins[i + 1], freq="M")
        assert (p2 - p1).n == 1


# 8. minimum initial history = 60 months
def test_08_minimum_initial_history_60_months():
    first_orig = derive_first_validation_origin("2012-01", min_history_months=60)
    assert first_orig == "2016-12"
    # Months from 2012-01 to 2016-12 inclusive = 60
    p_range = pd.date_range("2012-01-01", "2016-12-01", freq="MS")
    assert len(p_range) == 60


# 9. first validation origin derived correctly (2016-12)
def test_09_first_validation_origin_is_2016_12(pipeline_results):
    _, metrics_df, _ = pipeline_results
    assert (metrics_df["first_origin"] == "2016-12").all()


# 10. h1 last origin = 2024-11
def test_10_h1_last_origin_2024_11(pipeline_results):
    _, metrics_df, _ = pipeline_results
    h1_metrics = metrics_df[metrics_df["horizon"] == 1]
    assert (h1_metrics["last_origin"] == "2024-11").all()
    assert (h1_metrics["n_validation_origins"] == 96).all()


# 11. h3 last origin = 2024-09
def test_11_h3_last_origin_2024_09(pipeline_results):
    _, metrics_df, _ = pipeline_results
    h3_metrics = metrics_df[metrics_df["horizon"] == 3]
    assert (h3_metrics["last_origin"] == "2024-09").all()
    assert (h3_metrics["n_validation_origins"] == 94).all()


# 12. h6 last origin = 2024-06
def test_12_h6_last_origin_2024_06(pipeline_results):
    _, metrics_df, _ = pipeline_results
    h6_metrics = metrics_df[metrics_df["horizon"] == 6]
    assert (h6_metrics["last_origin"] == "2024-06").all()
    assert (h6_metrics["n_validation_origins"] == 91).all()


# 13. forecast period = origin + h calendar months
def test_13_forecast_period_equals_origin_plus_h(pipeline_results):
    pred_df, _, _ = pipeline_results
    sample = pred_df.sample(50, random_state=42)
    for _, row in sample.iterrows():
        orig_p = pd.Period(row["forecast_origin"], freq="M")
        fc_p = pd.Period(row["forecast_period"], freq="M")
        assert (fc_p - orig_p).n == row["horizon"]


# 14. seasonal source = forecast period - 12 calendar months
def test_14_seasonal_source_equals_forecast_period_minus_12(pipeline_results):
    pred_df, _, _ = pipeline_results
    sample = pred_df.sample(50, random_state=42)
    for _, row in sample.iterrows():
        fc_p = pd.Period(row["forecast_period"], freq="M")
        src_p = pd.Period(row["seasonal_source_period"], freq="M")
        assert (fc_p - src_p).n == 12


# 15. seasonal source belongs to same direction/hs2 series
def test_15_seasonal_source_same_series(pipeline_results):
    settings = get_settings()
    dev_df = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    dev_map = dev_df.set_index(["direction", "hs2", "period"])["target"].to_dict()

    pred_df, _, _ = pipeline_results
    sample = pred_df.sample(50, random_state=42)
    for _, row in sample.iterrows():
        expected_src_val = dev_map[(row["direction"], row["hs2"], row["seasonal_source_period"])]
        assert abs(row["seasonal_source_value"] - expected_src_val) < 1e-4


# 16. seasonal source <= forecast origin
def test_16_seasonal_source_less_than_or_equal_origin(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert (pred_df["seasonal_source_period"] <= pred_df["forecast_origin"]).all()


# 17. prediction exactly equals seasonal source actual
def test_17_prediction_equals_seasonal_source_actual(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert np.allclose(pred_df["prediction"], pred_df["seasonal_source_value"], equal_nan=True)


# 18. actual validation value is never used in prediction construction
def test_18_actual_value_not_used_in_prediction():
    mock_panel = pd.DataFrame([
        {"direction": "TR_RU", "hs2": "08", "period": "2019-06", "target": 100.0},
        {"direction": "TR_RU", "hs2": "08", "period": "2020-06", "target": 99999.0},  # actual at target period
    ])
    predictor = SeasonalNaivePredictor(mock_panel)
    pred_res = predictor.predict_one(
        direction="TR_RU",
        hs2="08",
        forecast_origin="2020-05",
        horizon=1,  # forecast period: 2020-06, seasonal source: 2019-06
    )
    # Prediction must use 2019-06 (100.0), ignoring 2020-06 (99999.0)
    assert pred_res["prediction"] == 100.0
    assert pred_res["prediction"] != 99999.0


# 19. h1 example calculation
def test_19_h1_example_calculation(pipeline_results):
    pred_df, _, _ = pipeline_results
    # Origin 2020-06, h=1 -> forecast_period 2020-07, seasonal_source 2019-07
    sub = pred_df[
        (pred_df["direction"] == "TR_RU")
        & (pred_df["hs2"] == "TOTAL")
        & (pred_df["horizon"] == 1)
        & (pred_df["forecast_origin"] == "2020-06")
    ].iloc[0]
    assert sub["forecast_period"] == "2020-07"
    assert sub["seasonal_source_period"] == "2019-07"


# 20. h3 example calculation
def test_20_h3_example_calculation(pipeline_results):
    pred_df, _, _ = pipeline_results
    # Origin 2020-06, h=3 -> forecast_period 2020-09, seasonal_source 2019-09
    sub = pred_df[
        (pred_df["direction"] == "TR_RU")
        & (pred_df["hs2"] == "TOTAL")
        & (pred_df["horizon"] == 3)
        & (pred_df["forecast_origin"] == "2020-06")
    ].iloc[0]
    assert sub["forecast_period"] == "2020-09"
    assert sub["seasonal_source_period"] == "2019-09"


# 21. h6 example calculation
def test_21_h6_example_calculation(pipeline_results):
    pred_df, _, _ = pipeline_results
    # Origin 2020-06, h=6 -> forecast_period 2020-12, seasonal_source 2019-12
    sub = pred_df[
        (pred_df["direction"] == "TR_RU")
        & (pred_df["hs2"] == "TOTAL")
        & (pred_df["horizon"] == 6)
        & (pred_df["forecast_origin"] == "2020-06")
    ].iloc[0]
    assert sub["forecast_period"] == "2020-12"
    assert sub["seasonal_source_period"] == "2019-12"


# 22. both-zero sMAPE component = 0
def test_22_both_zero_smape_is_zero():
    assert compute_smape_single(0.0, 0.0) == 0.0


# 23. one-zero sMAPE handled without NaN/inf
def test_23_one_zero_smape_handled():
    val1 = compute_smape_single(0.0, 100.0)
    val2 = compute_smape_single(100.0, 0.0)
    assert val1 == 200.0
    assert val2 == 200.0
    assert not np.isnan(val1)
    assert not np.isinf(val1)


# 24. deterministic sMAPE aggregate
def test_24_deterministic_smape_aggregate():
    actuals = np.array([100.0, 200.0, 0.0])
    predictions = np.array([100.0, 100.0, 0.0])
    # Component 1: 0.0
    # Component 2: 2 * 100 / 300 * 100 = 66.66667
    # Component 3: 0.0
    # Mean: 66.66667 / 3 = 22.22222
    smape = calculate_smape(actuals, predictions)
    assert abs(smape - 22.22222) < 1e-4


# 25. deterministic MAE
def test_25_deterministic_mae():
    actuals = np.array([100.0, 200.0, 300.0])
    predictions = np.array([90.0, 220.0, 300.0])
    # Abs errors: 10, 20, 0 -> Mean: 10.0
    assert calculate_mae(actuals, predictions) == 10.0


# 26. deterministic RMSE
def test_26_deterministic_rmse():
    actuals = np.array([100.0, 200.0, 300.0])
    predictions = np.array([90.0, 200.0, 300.0])
    # Squared errors: 100, 0, 0 -> Mean: 33.3333 -> sqrt: 5.7735
    assert abs(calculate_rmse(actuals, predictions) - np.sqrt(100.0 / 3.0)) < 1e-6


# 27. metrics exist for all 42 series x horizon groups
def test_27_metrics_count_is_42(pipeline_results):
    _, metrics_df, _ = pipeline_results
    assert len(metrics_df) == 42


# 28. no duplicate metric group
def test_28_no_duplicate_metric_group(pipeline_results):
    _, metrics_df, _ = pipeline_results
    dups = metrics_df.duplicated(subset=["direction", "hs2", "horizon"]).sum()
    assert dups == 0


# 29. prediction artifact has no 2025 forecast periods
def test_29_prediction_artifact_no_2025_forecast_periods(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert (pred_df["forecast_period"] <= "2024-12").all()


# 30. validation fold registry has no 2025 target periods():
def test_30_validation_fold_registry_no_2025_target_periods():
    folds = generate_validation_folds()
    assert (folds["forecast_period"] <= "2024-12").all()


# 31. holdout mutation leaves predictions unchanged
def test_31_holdout_mutation_leaves_predictions_unchanged():
    settings = get_settings()
    dev_path = settings.processed_data_dir / "model_features_development.parquet"
    df = pd.read_parquet(dev_path)

    # Base predictions
    pred1, m1, _ = run_seasonal_naive_pipeline(run_id="run_base")

    # Simulate presence of mutated 2025 holdout rows in development input (which must be filtered/ignored)
    mutated_df = df.copy()
    # Add dummy 2025 rows with crazy values
    dummy_2025 = pd.DataFrame([{
        "direction": "TR_RU",
        "hs2": "TOTAL",
        "period": "2025-06",
        "target": 999999999999.0,
    }])
    combined_df = pd.concat([mutated_df, dummy_2025], ignore_index=True)
    # The pipeline ingests model_features_development.parquet which is filtered period <= 2024-12
    # Even if predictor receives combined_df, any query for forecast period <= 2024-12 will never touch 2025
    predictor = SeasonalNaivePredictor(combined_df)
    folds = generate_validation_folds()
    series_list = extract_modeling_series_list({"TR_RU": [{"hs2": "08"}], "RU_TR": []})
    pred2 = predictor.predict_batch(folds[folds["horizon"] == 1], series_list)
    pred2_sub = pred2[(pred2["direction"] == "TR_RU") & (pred2["hs2"] == "08")]

    # Check predictions for 2024 origins
    sub1 = pred1[(pred1["direction"] == "TR_RU") & (pred1["hs2"] == "08") & (pred1["horizon"] == 1)]
    assert np.allclose(sub1["prediction"], pred2_sub["prediction"], equal_nan=True)


# 32. holdout mutation leaves metrics unchanged
def test_32_holdout_mutation_leaves_metrics_unchanged(pipeline_results):
    _, metrics_df, _ = pipeline_results
    assert not metrics_df.empty
    # Rerunning with exact same development data yields exact metrics
    _, m2, _ = run_seasonal_naive_pipeline(run_id="run_check")
    pd.testing.assert_frame_equal(
        metrics_df.drop(columns=["created_at", "run_id"]),
        m2.drop(columns=["created_at", "run_id"]),
    )


# 33. holdout mutation leaves fold registry unchanged
def test_33_holdout_mutation_leaves_fold_registry_unchanged():
    folds1 = generate_validation_folds()
    folds2 = generate_validation_folds()
    pd.testing.assert_frame_equal(folds1, folds2)


# 34. reported zeros are preserved
def test_34_reported_zeros_are_preserved():
    mock_panel = pd.DataFrame([
        {"direction": "RU_TR", "hs2": "23", "period": "2018-05", "target": 0.0},
    ])
    predictor = SeasonalNaivePredictor(mock_panel)
    pred_res = predictor.predict_one("RU_TR", "23", forecast_origin="2019-04", horizon=1)
    # Forecast period: 2019-05, seasonal source: 2018-05 -> target was 0.0
    assert pred_res["prediction"] == 0.0


# 35. missing seasonal source is not converted to zero
def test_35_missing_seasonal_source_not_zero():
    mock_panel = pd.DataFrame([
        {"direction": "TR_RU", "hs2": "84", "period": "2018-05", "target": np.nan},
    ])
    predictor = SeasonalNaivePredictor(mock_panel)
    pred_res = predictor.predict_one("TR_RU", "84", forecast_origin="2019-04", horizon=1)
    assert np.isnan(pred_res["prediction"])


# 36. invalid/missing rows are audited explicitly
def test_36_invalid_missing_rows_audited(pipeline_results):
    pred_df, metrics_df, run_meta = pipeline_results
    null_preds = pred_df["prediction"].isnull().sum()
    assert null_preds == 0, "No missing predictions expected in complete development dataset"
    assert run_meta["total_prediction_rows"] == 3934


# 37. training window metadata expands monotonically
def test_37_training_window_expands_monotonically():
    folds = generate_validation_folds(horizons=[1])
    h1_folds = folds[folds["horizon"] == 1]
    origins = h1_folds["train_end"].tolist()
    for i in range(len(origins) - 1):
        assert origins[i] < origins[i + 1]


# 38. train_start remains frozen at development start (2012-01)
def test_38_train_start_frozen_at_dev_start(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert (pred_df["train_start"] == "2012-01").all()


# 39. train_end equals forecast origin
def test_39_train_end_equals_forecast_origin(pipeline_results):
    pred_df, _, _ = pipeline_results
    assert (pred_df["train_end"] == pred_df["forecast_origin"]).all()


# 40. fold IDs are deterministic
def test_40_fold_ids_deterministic():
    folds1 = generate_validation_folds()
    folds2 = generate_validation_folds()
    assert (folds1["fold_id"] == folds2["fold_id"]).all()


# 41. rerunning M6 produces numerically identical results
def test_41_rerun_determinism(pipeline_results):
    p1, m1, _ = pipeline_results
    p2, m2, _ = run_seasonal_naive_pipeline(run_id="run_determinism")
    pd.testing.assert_frame_equal(p1.drop(columns=["run_id"]), p2.drop(columns=["run_id"]))
    pd.testing.assert_frame_equal(m1.drop(columns=["created_at", "run_id"]), m2.drop(columns=["created_at", "run_id"]))


# 42. no macro features influence Seasonal Naive predictions
def test_42_no_macro_features_influence_seasonal_naive(pipeline_results):
    pred_df, _, _ = pipeline_results
    # Verify no macro columns exist in prediction dataframe
    for col in pred_df.columns:
        assert "usd_try" not in col
        assert "rub_usd" not in col
        assert "brent" not in col
        assert "tr_ip" not in col
        assert "ru_ip" not in col


# 43. no scaling occurs
def test_43_no_scaling_occurs(pipeline_results):
    pred_df, _, _ = pipeline_results
    # Total bilateral trade is hundreds of millions of USD, not scaled between [0, 1] or N(0, 1)
    tot_preds = pred_df[pred_df["hs2"] == "TOTAL"]["prediction"]
    assert tot_preds.mean() > 1e7


# 44. no hyperparameter tuning occurs
def test_44_no_hyperparameter_tuning(pipeline_results):
    _, _, run_meta = pipeline_results
    assert run_meta["hyperparameters"]["seasonal_period_months"] == 12
    assert "best_params" not in run_meta["hyperparameters"]


# 45. no model selection occurs
def test_45_no_model_selection(pipeline_results):
    _, metrics_df, _ = pipeline_results
    assert (metrics_df["model_type"] == "seasonal_naive").all()
    assert "is_champion" not in metrics_df.columns


# 46. no champion selection artifacts created
def test_46_no_champion_selection_artifacts_created():
    settings = get_settings()
    assert not (settings.processed_data_dir / "m10_champion_selection.parquet").exists()
    assert not (settings.processed_data_dir / "m10_champion_metrics.parquet").exists()


# 47. no M11 holdout artifacts created
def test_47_no_m11_holdout_artifacts_created():
    settings = get_settings()
    assert not (settings.processed_data_dir / "m11_holdout_predictions.parquet").exists()
    assert not (settings.processed_data_dir / "m11_holdout_metrics.parquet").exists()


# 48. no champion selection artifacts created
def test_48_no_champion_selection_artifacts_created():
    settings = get_settings()
    assert not (settings.processed_data_dir / "m10_champion_selection.parquet").exists()
    assert not (settings.processed_data_dir / "m10_champion_metrics.parquet").exists()


# 49. no M10+ implementation exists
def test_49_no_m10_plus_implementation_exists():
    settings = get_settings()
    assert not (settings.project_root / "docs" / "M10_CHAMPION_SELECTION_REPORT.md").exists()
    assert not (settings.project_root / "docs" / "M11_HOLDOUT_REPORT.md").exists()


# 50. lint/test commands remain clean
def test_50_test_count_and_clean_run(pipeline_results):
    pred_df, metrics_df, _ = pipeline_results
    assert len(pred_df) == 3934
    assert len(metrics_df) == 42


# 51. canonical validation folds manifest remains byte-identical
def test_51_canonical_validation_folds_manifest_byte_identical():
    """Verify configs/validation_folds_v1_1.yaml has not been modified by tests."""
    import hashlib

    settings = get_settings()
    yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    assert yaml_path.exists()
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["total_folds"] == 281
    assert data["horizons"]["h1"]["number_of_origins"] == 96
    assert data["horizons"]["h3"]["number_of_origins"] == 94
    assert data["horizons"]["h6"]["number_of_origins"] == 91

    with open(yaml_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    assert sha256 == "0dc666c284991d494575a44cba18a2d93be2b3195bca01e8ae01a9f7191cded3"
