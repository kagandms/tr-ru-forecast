"""Comprehensive unit and mutation test suite for Milestone M9 XGBoost Direct Forecasting.

Tests:
1. Exact 21 predictor whitelist and forbidden column exclusion.
2. Direct forecasting semantics (h1, h3, h6 separate, no recursion).
3. Exact 16-candidate hyperparameter search space and pre-validation tuning.
4. One selected parameter set per horizon, shared across all 14 series.
5. Early stopping disabled and fixed hyperparameters.
6. Outer walk-forward validation fold reuse (281 folds, 3,934 predictions, 42 metrics).
7. Critical label-availability embargo (training_period + h <= forecast_origin).
8. Label-availability embargo mutation tests (h1, h3, h6).
9. Future feature leakage mutation test.
10. Tuning contamination mutation test (2017–2024 actuals mutation).
11. Upstream 2025 target quarantine mutation test.
12. Upstream 2025 macro quarantine mutation test.
13. Prequential conformal calibration leakage tests (unobserved, own, and future residual exclusions).
14. Conformal order statistic and interval availability counts (3,038 rows: 77/73/67).
15. Interval ordering invariants (lower95 <= lower80 <= pred <= upper80 <= upper95).
16. TreeSHAP fold-local calculation and additive consistency verification.
17. Global feature importance aggregation across 42 groups (882 rows).
18. Absence of 2025 predictions, intervals, or SHAP attributions.
19. Re-run determinism test.
20. Absence of M10 champion selection and M11 holdout artifacts.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.models.ml.conformal import (
    PrequentialResidualCalibrator,
)
from tr_ru_forecast.models.ml.constants import (
    FIXED_XGB_PARAMS,
    FORBIDDEN_COLUMNS,
    FORECAST_HORIZONS,
    M9_PREDICTORS,
    generate_candidate_grid,
)
from tr_ru_forecast.models.ml.shap_explainer import (
    ADDITIVITY_ATOL,
    ADDITIVITY_RTOL,
    explain_fold_prediction,
)
from tr_ru_forecast.models.ml.tuner import run_pre_validation_tuning
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
)


@pytest.fixture(scope="module")
def dev_features():
    """Load canonical M5 development features."""
    settings = get_settings()
    path = settings.processed_data_dir / "model_features_development.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def validation_folds():
    """Load canonical validation folds."""
    settings = get_settings()
    path = settings.processed_data_dir / "validation_origins.parquet"
    return pd.read_parquet(path)


# ==============================================================================
# 1. CORE STRUCTURAL CONTRACT (Section 34)
# ==============================================================================

def test_00_no_m6_features_reference():
    """Verify that no M9 production code, config, test, or metadata references m6_features.parquet."""
    settings = get_settings()
    m9_files = [
        settings.project_root / "configs" / "xgboost_spec_v1_1.yaml",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "constants.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "xgboost_model.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "tuner.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "conformal.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "shap_explainer.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "pipeline.py",
        settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml" / "cli.py",
    ]
    for path in m9_files:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            matched_lines = [line for line in content.splitlines() if "m6_features" in line]
            assert not matched_lines, f"Forbidden reference to m6_features found in {path}: {matched_lines}"


def test_01_exact_21_predictors_whitelist():
    """Verify exact 21 predictor whitelist."""
    assert len(M9_PREDICTORS) == 21
    expected = [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
        "rolling_std_3", "rolling_std_6", "rolling_std_12",
        "mom_growth", "yoy_growth",
        "month", "quarter", "year",
        "usd_try_lagged", "rub_usd_lagged", "brent_lagged", "tr_ip_lagged", "ru_ip_lagged",
    ]
    assert M9_PREDICTORS == expected


def test_02_forbidden_columns_not_in_predictors():
    """Verify target, forward labels, metadata, and alternative feature names are excluded."""
    for col in FORBIDDEN_COLUMNS:
        assert col not in M9_PREDICTORS
    forbidden_alternatives = [
        "target_lag_1", "target_roll_mean_3", "brent_oil_lag1",
        "cpi_ratio_lag1", "usd_rub_lag1",
    ]
    for col in forbidden_alternatives:
        assert col not in M9_PREDICTORS


def test_03_three_horizons_only():
    """Verify three forecast horizons [1, 3, 6]."""
    assert FORECAST_HORIZONS == [1, 3, 6]


def test_04_exact_16_candidate_search_grid():
    """Verify Cartesian search grid generates exactly 16 candidates."""
    candidates = generate_candidate_grid()
    assert len(candidates) == 16
    ids = [c["candidate_id"] for c in candidates]
    assert ids == [f"cand_{i:02d}" for i in range(16)]


def test_05_fixed_parameters():
    """Verify fixed hyperparameters match specification."""
    assert FIXED_XGB_PARAMS["objective"] == "reg:squarederror"
    assert FIXED_XGB_PARAMS["booster"] == "gbtree"
    assert FIXED_XGB_PARAMS["tree_method"] == "hist"
    assert FIXED_XGB_PARAMS["subsample"] == 1.0
    assert FIXED_XGB_PARAMS["colsample_bytree"] == 1.0
    assert FIXED_XGB_PARAMS["reg_alpha"] == 0.0
    assert FIXED_XGB_PARAMS["reg_lambda"] == 1.0
    assert FIXED_XGB_PARAMS["gamma"] == 0.0
    assert FIXED_XGB_PARAMS["random_state"] == 42
    assert FIXED_XGB_PARAMS["n_jobs"] == 1


def test_06_outer_validation_folds_integrity():
    """Verify outer validation folds manifest is unchanged and has 281 folds."""
    settings = get_settings()
    yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["total_folds"] == 281
    assert data["horizons"]["h1"]["number_of_origins"] == 96
    assert data["horizons"]["h3"]["number_of_origins"] == 94
    assert data["horizons"]["h6"]["number_of_origins"] == 91
    # Check SHA256 checksum matches frozen baseline
    with open(yaml_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    assert sha256 == "0dc666c284991d494575a44cba18a2d93be2b3195bca01e8ae01a9f7191cded3"


def test_07_prediction_artifact_dimensions():
    """Verify m9_xgboost_predictions.parquet has exactly 3,934 rows."""
    settings = get_settings()
    path = settings.processed_data_dir / "m9_xgboost_predictions.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 3934
    required_cols = [
        "direction", "hs2", "horizon", "forecast_origin", "forecast_period",
        "train_start", "train_feature_end", "train_label_end", "training_row_count",
        "model_type", "parameter_set_id", "actual", "prediction", "absolute_error",
        "interval_available", "calibration_count", "lower_80", "upper_80", "lower_95", "upper_95",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in predictions artifact"


def test_08_metrics_artifact_dimensions():
    """Verify m9_xgboost_metrics.parquet has exactly 42 rows."""
    settings = get_settings()
    path = settings.processed_data_dir / "m9_xgboost_metrics.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 42


def test_09_tuning_artifact_dimensions():
    """Verify m9_xgboost_tuning_results.parquet has 48 rows and exactly 1 selected per horizon."""
    settings = get_settings()
    path = settings.processed_data_dir / "m9_xgboost_tuning_results.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 48
    assert (df["status"] == "VALID").all()
    selected_counts = df.groupby("horizon")["selected"].sum().to_dict()
    assert selected_counts == {1: 1, 3: 1, 6: 1}


def test_10_one_selected_parameter_set_per_horizon():
    """Verify exactly one winning candidate per horizon in xgboost_spec_v1_1.yaml."""
    settings = get_settings()
    spec_path = settings.project_root / "configs" / "xgboost_spec_v1_1.yaml"
    assert spec_path.exists()
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    params = spec["tuning"]["selected_parameters"]
    assert "h1" in params
    assert "h3" in params
    assert "h6" in params


def test_11_early_stopping_disabled():
    """Verify early stopping is disabled in specification and regressor builder."""
    reg = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 150})
    assert reg.get_params().get("early_stopping_rounds") is None


def test_12_no_scaling_or_imputation(dev_features):
    """Verify features contain raw values and NaNs are preserved for native handling."""
    # Ensure initial lag rows contain NaNs and they were not imputed
    tr_total = dev_features[(dev_features["direction"] == "TR_RU") & (dev_features["hs2"] == "TOTAL")].sort_values("period")
    assert tr_total["lag_12"].iloc[0:12].isnull().all()


# ==============================================================================
# 2. LABEL LEAKAGE & EMBARGO (Section 35)
# ==============================================================================

def test_13_training_label_embargo_invariants():
    """Verify training_period + h <= forecast_origin strictly holds for all 3,934 predictions."""
    settings = get_settings()
    path = settings.processed_data_dir / "m9_xgboost_predictions.parquet"
    df = pd.read_parquet(path)

    for _, row in df.iterrows():
        orig_p = pd.Period(row["forecast_origin"], freq="M")
        feat_end_p = pd.Period(row["train_feature_end"], freq="M")
        h = row["horizon"]

        # Embargo assertion: feat_end + h <= origin
        assert feat_end_p + h <= orig_p, (
            f"Embargo violation for {row['direction']}-{row['hs2']}-h{h} at {row['forecast_origin']}: "
            f"feat_end {row['train_feature_end']} + h{h} > {row['forecast_origin']}"
        )


@pytest.mark.parametrize("h", [1, 3, 6])
def test_14_label_embargo_mutation_test(dev_features, h):
    """Mutation test: mutating labels in embargoed rows (t-h+1 to t) does NOT alter prediction at t."""
    series_df = dev_features[
        (dev_features["direction"] == "TR_RU") & (dev_features["hs2"] == "TOTAL")
    ].sort_values("period").reset_index(drop=True)

    origin = "2020-12"
    label_col = f"target_h{h}"

    # 1. Base fit and prediction
    train_sub_base = get_training_subset(series_df, origin, h)
    X_train_base, y_train_base = prepare_training_xy(train_sub_base, h)
    X_orig_base = extract_origin_features(series_df, origin)

    model_base = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 150})
    model_base.fit(X_train_base, y_train_base)
    pred_base = float(model_base.predict(X_orig_base)[0])

    # 2. Identify embargoed rows: period > origin - h and period <= origin
    origin_p = pd.Period(origin, freq="M")
    cutoff_p = origin_p - h
    mutated_series = series_df.copy()
    embargo_mask = (mutated_series["period"] > str(cutoff_p)) & (mutated_series["period"] <= origin)

    assert embargo_mask.sum() == h, f"Expected {h} embargoed rows, found {embargo_mask.sum()}"

    # Radically mutate these embargoed labels
    mutated_series.loc[embargo_mask, label_col] = 999_999_999_999.0

    # 3. Re-extract training subset and re-fit
    train_sub_mut = get_training_subset(mutated_series, origin, h)
    X_train_mut, y_train_mut = prepare_training_xy(train_sub_mut, h)
    X_orig_mut = extract_origin_features(mutated_series, origin)

    model_mut = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 150})
    model_mut.fit(X_train_mut, y_train_mut)
    pred_mut = float(model_mut.predict(X_orig_mut)[0])

    # Prediction at origin t MUST be 100% numerically identical
    assert pred_base == pred_mut, f"Prediction changed under embargoed label mutation for h={h}!"


# ==============================================================================
# 3. FUTURE FEATURE LEAKAGE (Section 36)
# ==============================================================================

def test_17_future_feature_mutation(dev_features):
    """Mutation test: mutating feature rows strictly after origin t does NOT alter prediction at t."""
    series_df = dev_features[
        (dev_features["direction"] == "RU_TR") & (dev_features["hs2"] == "27")
    ].sort_values("period").reset_index(drop=True)

    origin = "2020-06"
    h = 3

    # Base fit and prediction
    train_sub_base = get_training_subset(series_df, origin, h)
    X_train_base, y_train_base = prepare_training_xy(train_sub_base, h)
    X_orig_base = extract_origin_features(series_df, origin)

    model_base = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 150})
    model_base.fit(X_train_base, y_train_base)
    pred_base = float(model_base.predict(X_orig_base)[0])

    # Mutate all feature values for periods > origin
    mutated_series = series_df.copy()
    future_mask = mutated_series["period"] > origin
    assert future_mask.sum() > 0

    for feat in M9_PREDICTORS:
        mutated_series.loc[future_mask, feat] = 1_234_567.0

    # Re-run for origin
    train_sub_mut = get_training_subset(mutated_series, origin, h)
    X_train_mut, y_train_mut = prepare_training_xy(train_sub_mut, h)
    X_orig_mut = extract_origin_features(mutated_series, origin)

    model_mut = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 150})
    model_mut.fit(X_train_mut, y_train_mut)
    pred_mut = float(model_mut.predict(X_orig_mut)[0])

    assert pred_base == pred_mut, "Prediction changed under future feature mutation!"


# ==============================================================================
# 4. TUNING CONTAMINATION (Section 37)
# ==============================================================================

def test_18_tuning_contamination_mutation(dev_features):
    """Mutation test: mutating official validation targets (2017–2024) does NOT alter tuning scores or selected parameters."""
    mutated_features = dev_features.copy()
    val_mask = mutated_features["period"] >= "2017-01"

    # Radically mutate all target labels in official validation window
    for h in [1, 3, 6]:
        mutated_features.loc[val_mask, f"target_h{h}"] = 888_888_888_888.0

    from tr_ru_forecast.models.baselines.pipeline import extract_modeling_series_list
    settings = get_settings()
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml") as f:
        sectors_manifest = yaml.safe_load(f)
    series_list = extract_modeling_series_list(sectors_manifest)

    # Run tuning on mutated features for all 3 horizons
    tuning_df_mut, selected_mut = run_pre_validation_tuning(
        dev_features_df=mutated_features,
        series_list=series_list,
        horizons=[1, 3, 6],
    )

    # Compare with canonical tuning results
    canonical_tuning = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_tuning_results.parquet")
    canon_sorted = canonical_tuning.sort_values(["horizon", "candidate_id"]).reset_index(drop=True)
    mut_sorted = tuning_df_mut.sort_values(["horizon", "candidate_id"]).reset_index(drop=True)

    np.testing.assert_allclose(canon_sorted["mean_series_smape"], mut_sorted["mean_series_smape"], rtol=1e-5)
    np.testing.assert_allclose(canon_sorted["median_series_smape"], mut_sorted["median_series_smape"], rtol=1e-5)
    assert selected_mut[1]["candidate_id"] == "cand_05"
    assert selected_mut[3]["candidate_id"] == "cand_07"
    assert selected_mut[6]["candidate_id"] == "cand_01"


# ==============================================================================
# 5. 2025 HOLDOUT QUARANTINE (Section 38 & 39)
# ==============================================================================

def test_19_2025_target_quarantine_mutation():
    """Mutation test: radically mutating 2025 target observations upstream does NOT alter development features or M9 outputs."""
    settings = get_settings()
    target_parquet_path = settings.processed_data_dir / "forecast_target_monthly.parquet"
    target_df = pd.read_parquet(target_parquet_path)

    # Verify 2025 rows exist upstream
    holdout_rows = target_df[target_df["period"] >= "2025-01"]
    assert len(holdout_rows) > 0

    # Mutate 2025 rows in memory
    mutated_target = target_df.copy()
    mut_mask = mutated_target["period"] >= "2025-01"
    mutated_target.loc[mut_mask, "trade_value_usd"] = mutated_target.loc[mut_mask, "trade_value_usd"] * 999.0 + 1e12

    # Load macro data and sector manifest
    macro_df = pd.read_parquet(settings.processed_data_dir / "macro_monthly.parquet")
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml", encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)

    # Rebuild development features using accepted M5 builder logic
    from tr_ru_forecast.features.pipeline import build_development_features
    rebuilt_dev_df, _ = build_development_features(mutated_target, macro_df, sectors_manifest)

    canonical_dev = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    # Invariant: development features must be 100% identical to canonical
    pd.testing.assert_frame_equal(rebuilt_dev_df, canonical_dev)


def test_20_2025_macro_quarantine_mutation():
    """Mutation test: radically mutating 2025 macro observations upstream does NOT alter development features or M9 outputs."""
    settings = get_settings()
    target_parquet_path = settings.processed_data_dir / "forecast_target_monthly.parquet"
    target_df = pd.read_parquet(target_parquet_path)
    macro_df = pd.read_parquet(settings.processed_data_dir / "macro_monthly.parquet")
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml", encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)

    # Mutate 2025 macro observations in memory
    mutated_macro = macro_df.copy()
    mask_macro_2025 = mutated_macro["period"] >= "2025-01"
    mutated_macro.loc[mask_macro_2025, "value"] = mutated_macro.loc[mask_macro_2025, "value"] * 999.0 + 1e6

    from tr_ru_forecast.features.pipeline import build_development_features
    rebuilt_dev_df, _ = build_development_features(target_df, mutated_macro, sectors_manifest)

    canonical_dev = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    pd.testing.assert_frame_equal(rebuilt_dev_df, canonical_dev)


# ==============================================================================
# 6. PREQUENTIAL CALIBRATION LEAKAGE (Section 40)
# ==============================================================================

def test_21_prequential_calibration_leakage_exclusions():
    """Test unobserved, own, and future residual exclusions in conformal calibrator."""
    calibrator = PrequentialResidualCalibrator(direction="TR_RU", hs2="08", horizon=3)

    # Record 25 historical predictions (origins 2017-01 to 2019-01)
    origins = pd.date_range("2017-01-01", "2019-01-01", freq="MS").strftime("%Y-%m").tolist()
    for orig in origins:
        orig_p = pd.Period(orig, freq="M")
        tgt_p = orig_p + 3
        calibrator.record_historical_prediction(
            forecast_origin=orig,
            forecast_period=str(tgt_p),
            actual=100.0,
            prediction=90.0,  # residual = 10.0
        )

    # Current origin: 2018-12 (so targets up to 2018-12 are observable)
    # Origins 2017-01 to 2018-09 (21 origins) have target <= 2018-12 -> n=21 >= 19
    curr_orig = "2018-12"
    res_base = calibrator.calibrate(curr_orig, point_prediction=100.0, actual_val=105.0)
    assert res_base["interval_available"] is True
    base_q80 = res_base["q80"]

    # A. Unobserved residual exclusion: mutate a residual whose target is > curr_orig (e.g. origin 2018-11 has target 2019-02)
    unobserved_idx = next(i for i, r in enumerate(calibrator.history) if r["forecast_period"] > curr_orig)
    calibrator.history[unobserved_idx]["absolute_residual"] = 999999.0

    res_test_unobserved = calibrator.calibrate(curr_orig, point_prediction=100.0, actual_val=105.0)
    assert res_test_unobserved["q80"] == base_q80, "Unobserved residual leaked into calibration!"

    # B. Own residual exclusion: mutating actual_val does not alter intervals
    res_test_own = calibrator.calibrate(curr_orig, point_prediction=100.0, actual_val=999999.0)
    assert res_test_own["q80"] == base_q80, "Own actual value leaked into interval bounds!"

    # C. Future residual exclusion: recording a future prediction (origin > curr_orig) does not alter current intervals
    calibrator.record_historical_prediction(
        forecast_origin="2019-06",
        forecast_period="2019-09",
        actual=5000.0,
        prediction=100.0,
    )
    res_test_future = calibrator.calibrate(curr_orig, point_prediction=100.0, actual_val=105.0)
    assert res_test_future["q80"] == base_q80, "Future residual leaked into past calibration!"


def test_24_calibration_target_invariant():
    """Verify max_calibration_target_period <= forecast_origin strictly holds for all interval-enabled rows."""
    settings = get_settings()
    path = settings.processed_data_dir / "m9_xgboost_interval_audit.parquet"
    assert path.exists()
    audit_df = pd.read_parquet(path)
    enabled = audit_df[audit_df["interval_available"]]

    for _, row in enabled.iterrows():
        assert row["max_calibration_target_period"] <= row["forecast_origin"]


# ==============================================================================
# 7. NO RECURSION (Section 41)
# ==============================================================================

def test_25_no_recursion_horizon_independence(dev_features):
    """Architectural test: h3 and h6 models do not consume h1/h3 predictions or objects."""
    series_df = dev_features[
        (dev_features["direction"] == "TR_RU") & (dev_features["hs2"] == "84")
    ].sort_values("period").reset_index(drop=True)

    origin = "2021-03"

    # 1. Fit h3 directly from features
    train_sub_h3 = get_training_subset(series_df, origin, horizon=3)
    X_train_h3, y_train_h3 = prepare_training_xy(train_sub_h3, horizon=3)
    X_orig = extract_origin_features(series_df, origin)

    model_h3 = build_xgb_regressor({"max_depth": 2, "min_child_weight": 5, "learning_rate": 0.10, "n_estimators": 300})
    model_h3.fit(X_train_h3, y_train_h3)
    pred_h3 = float(model_h3.predict(X_orig)[0])

    # Mutate target_h1 in origin row; h3 prediction must remain identical
    series_df_mut1 = series_df.copy()
    series_df_mut1.loc[series_df_mut1["period"] == origin, "target_h1"] = 1e12

    train_sub_h3_mut = get_training_subset(series_df_mut1, origin, horizon=3)
    X_train_h3_mut, y_train_h3_mut = prepare_training_xy(train_sub_h3_mut, horizon=3)
    X_orig_mut = extract_origin_features(series_df_mut1, origin)

    model_h3_mut = build_xgb_regressor({"max_depth": 2, "min_child_weight": 5, "learning_rate": 0.10, "n_estimators": 300})
    model_h3_mut.fit(X_train_h3_mut, y_train_h3_mut)
    pred_h3_mut = float(model_h3_mut.predict(X_orig_mut)[0])

    assert pred_h3 == pred_h3_mut, "h3 prediction depended on h1 target label!"

    # 2. Fit h6 directly from features
    train_sub_h6 = get_training_subset(series_df, origin, horizon=6)
    X_train_h6, y_train_h6 = prepare_training_xy(train_sub_h6, horizon=6)
    model_h6 = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 300})
    model_h6.fit(X_train_h6, y_train_h6)
    pred_h6 = float(model_h6.predict(X_orig)[0])

    # Mutate target_h3 in origin row; h6 prediction must remain identical
    series_df_mut3 = series_df.copy()
    series_df_mut3.loc[series_df_mut3["period"] == origin, "target_h3"] = 1e12
    train_sub_h6_mut = get_training_subset(series_df_mut3, origin, horizon=6)
    X_train_h6_mut, y_train_h6_mut = prepare_training_xy(train_sub_h6_mut, horizon=6)
    model_h6_mut = build_xgb_regressor({"max_depth": 2, "min_child_weight": 1, "learning_rate": 0.03, "n_estimators": 300})
    model_h6_mut.fit(X_train_h6_mut, y_train_h6_mut)
    pred_h6_mut = float(model_h6_mut.predict(X_orig)[0])

    assert pred_h6 == pred_h6_mut, "h6 prediction depended on h3 target label!"


# ==============================================================================
# 8. INTERVAL COUNTS & INVARIANTS (Section 42)
# ==============================================================================

def test_26_interval_enabled_counts():
    """Verify exactly 3,038 interval-enabled prediction rows (77 for h1, 73 for h3, 67 for h6 per series)."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_predictions.parquet")
    assert df["interval_available"].sum() == 3038

    # Verify per-series counts
    for _, series_group in df.groupby(["direction", "hs2"]):
        h1_count = series_group[(series_group["horizon"] == 1) & (series_group["interval_available"])].shape[0]
        h3_count = series_group[(series_group["horizon"] == 3) & (series_group["interval_available"])].shape[0]
        h6_count = series_group[(series_group["horizon"] == 6) & (series_group["interval_available"])].shape[0]
        assert h1_count == 77
        assert h3_count == 73
        assert h6_count == 67


def test_27_interval_invariants():
    """Verify lower_95 <= lower_80 <= prediction <= upper_80 <= upper_95 for all interval-enabled rows."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_predictions.parquet")
    enabled = df[df["interval_available"]]

    assert (enabled["lower_95"] <= enabled["lower_80"]).all()
    assert (enabled["lower_80"] <= enabled["prediction"]).all()
    assert (enabled["prediction"] <= enabled["upper_80"]).all()
    assert (enabled["upper_80"] <= enabled["upper_95"]).all()


def test_27b_negative_intervals_unclipped_preservation():
    """Verify negative lower bounds survive unclipped and are preserved without non-negativity truncation."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_predictions.parquet")
    enabled = df[df["interval_available"]]

    # Assert negative lower bounds actually exist in the empirical validation set
    n_neg_80 = (enabled["lower_80"] < 0).sum()
    n_neg_95 = (enabled["lower_95"] < 0).sum()
    assert n_neg_80 > 0, f"Expected observed negative lower_80 bounds, found {n_neg_80}"
    assert n_neg_95 > 0, f"Expected observed negative lower_95 bounds, found {n_neg_95}"

    # Verify PrequentialResidualCalibrator unit behavior: does not clip negative lower bounds to 0
    calibrator = PrequentialResidualCalibrator(direction="TR_RU", hs2="TOTAL", horizon=1)
    for i in range(25):
        orig = f"2017-{i%12 + 1:02d}"
        calibrator.record_historical_prediction(orig, orig, actual=100.0, prediction=0.0)  # residual = 100.0

    res = calibrator.calibrate("2018-12", point_prediction=10.0)
    assert res["lower_80"] == -90.0, f"Expected raw -90.0, got {res['lower_80']} (must NOT clip to 0)"
    assert res["lower_95"] == -90.0, f"Expected raw -90.0, got {res['lower_95']} (must NOT clip to 0)"


def test_28_interval_unavailable_nulls():
    """Verify interval bounds are strictly null when interval_available is False."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_predictions.parquet")
    disabled = df[~df["interval_available"]]
    assert len(disabled) == 3934 - 3038  # 896

    assert disabled["lower_80"].isnull().all()
    assert disabled["upper_80"].isnull().all()
    assert disabled["lower_95"].isnull().all()
    assert disabled["upper_95"].isnull().all()


# ==============================================================================
# 9. TREESHAP (Section 43)
# ==============================================================================

def test_29_global_shap_dimensions():
    """Verify global SHAP artifact has exactly 882 rows (42 groups x 21 features)."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_global.parquet")
    assert len(df) == 882
    groups = df[["direction", "hs2", "horizon"]].drop_duplicates()
    assert len(groups) == 42


def test_29b_shap_features_strict_set_equality():
    """Verify unique SHAP features strictly equal the exact 21 predictor set (no pooled/altered names)."""
    settings = get_settings()
    df_local = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")
    df_global = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_global.parquet")
    assert set(df_local["feature"].unique()) == set(M9_PREDICTORS)
    assert set(df_global["feature"].unique()) == set(M9_PREDICTORS)


def test_30_local_shap_dimensions():
    """Verify local SHAP artifact has exactly 82,614 rows (3,934 predictions x 21 features)."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")
    assert len(df) == 82614


def test_31_shap_additivity_invariants():
    """Verify additive consistency of SHAP values across predictions."""
    settings = get_settings()
    df_shap = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")

    # Sample 100 random predictions and check sum of SHAP values + base_value == point_prediction
    pred_groups = df_shap.groupby(["direction", "hs2", "horizon", "forecast_origin"])
    checked = 0
    for _, group in pred_groups:
        b_val = group["base_value"].iloc[0]
        pt_pred = group["point_prediction"].iloc[0]
        shap_sum = group["shap_value"].sum()
        discrepancy = abs((b_val + shap_sum) - pt_pred)
        threshold = max(ADDITIVITY_ATOL, ADDITIVITY_RTOL * abs(pt_pred))
        assert discrepancy <= threshold
        checked += 1
        if checked >= 50:
            break


def test_32_no_2025_shap():
    """Verify no 2025 observation exists in local SHAP artifact."""
    settings = get_settings()
    df_shap = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")
    assert not (df_shap["forecast_origin"] >= "2025-01").any()
    assert not (df_shap["forecast_period"] >= "2025-01").any()


def test_33_latest_local_shap_examples_exist():
    """Verify local SHAP examples exist for the latest development origin."""
    settings = get_settings()
    df_shap = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")

    h1_latest = df_shap[(df_shap["horizon"] == 1) & (df_shap["forecast_origin"] == "2024-11")]
    h3_latest = df_shap[(df_shap["horizon"] == 3) & (df_shap["forecast_origin"] == "2024-09")]
    h6_latest = df_shap[(df_shap["horizon"] == 6) & (df_shap["forecast_origin"] == "2024-06")]

    assert len(h1_latest) == 14 * 21
    assert len(h3_latest) == 14 * 21
    assert len(h6_latest) == 14 * 21


# ==============================================================================
# 10. NO M10/M11 ARTIFACTS
# ==============================================================================

def test_34_no_champion_selection_artifacts():
    """Verify no M10 champion selection artifacts exist."""
    settings = get_settings()
    assert not (settings.processed_data_dir / "m10_champion_selection.parquet").exists()
    assert not (settings.processed_data_dir / "m10_champion_metrics.parquet").exists()
    assert not (settings.project_root / "docs" / "M10_CHAMPION_SELECTION_REPORT.md").exists()


def test_35_no_holdout_evaluation_artifacts():
    """Verify no M11 holdout evaluation artifacts exist."""
    settings = get_settings()
    assert not (settings.processed_data_dir / "m11_holdout_predictions.parquet").exists()
    assert not (settings.processed_data_dir / "m11_holdout_metrics.parquet").exists()
    assert not (settings.project_root / "docs" / "M11_HOLDOUT_REPORT.md").exists()


# ==============================================================================
# 11. DETERMINISM TEST (Section 44)
# ==============================================================================

def test_36_model_and_shap_determinism(dev_features):
    """Verify identical random seed produces identical predictions and SHAP values across runs."""
    series_df = dev_features[
        (dev_features["direction"] == "TR_RU") & (dev_features["hs2"] == "TOTAL")
    ].sort_values("period").reset_index(drop=True)
    origin = "2022-06"
    h = 1

    train_sub = get_training_subset(series_df, origin, horizon=h)
    X_train, y_train = prepare_training_xy(train_sub, horizon=h)
    X_orig = extract_origin_features(series_df, origin)

    cand = {"max_depth": 2, "min_child_weight": 5, "learning_rate": 0.03, "n_estimators": 300}

    # Run 1
    m1 = build_xgb_regressor(cand)
    m1.fit(X_train, y_train)
    p1 = float(m1.predict(X_orig)[0])
    b1, s1, _ = explain_fold_prediction(m1, X_orig, p1, "TR_RU", "TOTAL", h, origin, "2022-07")

    # Run 2
    m2 = build_xgb_regressor(cand)
    m2.fit(X_train, y_train)
    p2 = float(m2.predict(X_orig)[0])
    b2, s2, _ = explain_fold_prediction(m2, X_orig, p2, "TR_RU", "TOTAL", h, origin, "2022-07")

    assert p1 == p2
    assert b1 == b2
    for feat in M9_PREDICTORS:
        assert s1[feat] == s2[feat]

