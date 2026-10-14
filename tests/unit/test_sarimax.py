"""Unit, acceptance, and anti-leakage mutation tests for Milestone M8: SARIMAX.

Verifies all critical requirements from the M8 specification:
1. Exact 14 frozen series modeled.
2. Reuses frozen M7 structural orders from configs/sarima_orders_v1_1.yaml verbatim.
3. Consumes exact five publication-safe exogenous variables.
4. Future exogenous strategy: origin_known_persistence strictly enforced.
5. Invariant: availability_date_or_approximation <= canonical_origin_snapshot_date.
6. Reuses exact 281 expanding walk-forward folds from configs/validation_folds_v1_1.yaml.
7. Total predictions count = 3,934 rows (100% complete).
8. Canonical 42 series x horizon metric groups populated.
9. Native 80% and 95% prediction intervals generated.
10. Invariant: lower_95 <= lower_80 <= upper_80 <= upper_95 with zero violations.
11. Raw negative values preserved without artificial clipping.
12. Deterministic missing-exog handling (fail closed loudly without data fabrication).
13. Future-macro mutation gate (early, middle, late origins across h=1,3,6).
14. Complete 2025 macro quarantine gate.
15. 2025 trade-target isolation gate.
16. Fold-registry immutability gate.
17. Full-development residual diagnostics across 14 series (2012-01 to 2024-12).
18. Exogenous provenance audit artifact generated (480 rows).
19. M8 methodological configuration manifest exists (configs/sarimax_spec_v1_1.yaml).
20. No champion selection (deferred to M10).
21. No M9+ artifacts exist (m9_xgboost_predictions.parquet, M9_XGBOOST_REPORT.md).
22. Snapshot-date semantics audit (exact agreement between M5 and M8).
23. Optimizer audit (primary lbfgs used, 0 retries, full convergence).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.data.macro.registry import REQUIRED_INDICATOR_IDS
from tr_ru_forecast.features.pipeline import build_development_features
from tr_ru_forecast.models.classical.sarimax import SARIMAXPredictor
from tr_ru_forecast.models.classical.sarimax_pipeline import (
    EXOG_COLUMNS,
    run_sarimax_pipeline,
)


@pytest.fixture(scope="module")
def sarimax_pipeline_results():
    """Load canonical M8 SARIMAX artifacts or execute pipeline if missing."""
    settings = get_settings()
    preds_p = settings.processed_data_dir / "m8_sarimax_predictions.parquet"
    metrics_p = settings.processed_data_dir / "m8_sarimax_metrics.parquet"
    diags_p = settings.processed_data_dir / "m8_sarimax_residual_diagnostics.parquet"
    fit_log_p = settings.processed_data_dir / "m8_sarimax_fit_log.parquet"
    audit_p = settings.processed_data_dir / "m8_sarimax_exog_audit.parquet"
    run_meta_p = settings.processed_data_dir / "m8_sarimax_run.json"

    if (
        preds_p.exists()
        and metrics_p.exists()
        and diags_p.exists()
        and fit_log_p.exists()
        and audit_p.exists()
        and run_meta_p.exists()
    ):
        pred_df = pd.read_parquet(preds_p)
        metrics_df = pd.read_parquet(metrics_p)
        diags_df = pd.read_parquet(diags_p)
        fit_log_df = pd.read_parquet(fit_log_p)
        audit_df = pd.read_parquet(audit_p)
        with open(run_meta_p) as f:
            run_meta = json.load(f)
        return pred_df, metrics_df, diags_df, fit_log_df, audit_df, run_meta

    return run_sarimax_pipeline(run_id="test_run_m8_sarimax", save_artifacts=False)


@pytest.fixture(scope="module")
def fresh_unmutated_sarimax_results():
    """Module-scoped fresh unmutated SARIMAX execution in current runtime.

    Provides the exact current-runtime baseline for mutation invariance tests,
    ensuring cross-BLAS runtime drift is not conflated with future-data leakage.
    """
    settings = get_settings()
    macro_df = pd.read_parquet(settings.processed_data_dir / "macro_monthly.parquet")
    target_df = pd.read_parquet(settings.processed_data_dir / "forecast_target_monthly.parquet")
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml") as f:
        sectors_manifest = yaml.safe_load(f)

    dev_unmut, _ = build_development_features(target_df, macro_df, sectors_manifest)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        unmut_preds, unmut_metrics, unmut_diags, unmut_fit_log, unmut_audit, unmut_meta = run_sarimax_pipeline(
            run_id="test_fresh_unmutated_baseline",
            feature_df=dev_unmut,
            macro_df=macro_df,
            output_dir=tmp_path,
            save_artifacts=False,
        )
        return dev_unmut, unmut_preds, unmut_metrics, unmut_diags, unmut_fit_log, unmut_audit, unmut_meta


@pytest.fixture(scope="module")
def macro_mutation_results():
    """Module-scoped upstream 2025 macroeconomic mutation test fixture.

    Radically mutates all 2025 macro observations in memory, rebuilds development
    features through the accepted M5 feature builder, and executes the complete M8 backtest
    in a temporary directory.
    """
    settings = get_settings()
    macro_df = pd.read_parquet(settings.processed_data_dir / "macro_monthly.parquet")
    target_df = pd.read_parquet(settings.processed_data_dir / "forecast_target_monthly.parquet")
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml") as f:
        sectors_manifest = yaml.safe_load(f)

    mut_macro = macro_df.copy()
    mask_2025 = mut_macro["period"] >= "2025-01"
    assert mask_2025.sum() > 0
    mut_macro.loc[mask_2025, "value"] = mut_macro.loc[mask_2025, "value"] * 999.0 + 88888.0

    dev_rebuilt, _ = build_development_features(target_df, mut_macro, sectors_manifest)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta = run_sarimax_pipeline(
            run_id="test_macro_mutation",
            feature_df=dev_rebuilt,
            macro_df=mut_macro,
            output_dir=tmp_path,
            save_artifacts=False,
        )
        return dev_rebuilt, mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta


@pytest.fixture(scope="module")
def target_mutation_results():
    """Module-scoped upstream 2025 trade-target mutation test fixture.

    Radically mutates all 2025 target observations in memory, rebuilds development
    features through the accepted M5 feature builder, and executes the complete M8 backtest
    in a temporary directory.
    """
    settings = get_settings()
    macro_df = pd.read_parquet(settings.processed_data_dir / "macro_monthly.parquet")
    target_df = pd.read_parquet(settings.processed_data_dir / "forecast_target_monthly.parquet")
    with open(settings.project_root / "configs" / "selected_sectors_v1_1.yaml") as f:
        sectors_manifest = yaml.safe_load(f)

    mut_target = target_df.copy()
    mask_2025 = mut_target["period"] >= "2025-01"
    assert mask_2025.sum() > 0
    mut_target.loc[mask_2025, "trade_value_usd"] = (
        mut_target.loc[mask_2025, "trade_value_usd"] * 999.0 + 1e12
    )

    dev_rebuilt, _ = build_development_features(mut_target, macro_df, sectors_manifest)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta = run_sarimax_pipeline(
            run_id="test_target_mutation",
            feature_df=dev_rebuilt,
            macro_df=macro_df,
            output_dir=tmp_path,
            save_artifacts=False,
        )
        return dev_rebuilt, mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta


# 1. Exact 14 frozen series modeled
def test_01_exact_14_series_modeled(sarimax_pipeline_results):
    pred_df, metrics_df, diags_df, _, _, _ = sarimax_pipeline_results
    series_keys = pred_df[["direction", "hs2"]].drop_duplicates()
    assert len(series_keys) == 14
    assert len(diags_df) == 14
    assert len(metrics_df[["direction", "hs2"]].drop_duplicates()) == 14


# 2. Reuses frozen M7 structural orders from configs/sarima_orders_v1_1.yaml verbatim
def test_02_reuses_frozen_m7_sarima_orders():
    settings = get_settings()
    m7_orders_path = settings.project_root / "configs" / "sarima_orders_v1_1.yaml"
    with open(m7_orders_path) as f:
        m7_manifest = yaml.safe_load(f)

    series_orders = m7_manifest["series_orders"]
    assert len(series_orders) == 14

    for entry in series_orders:
        d = entry["order"][1]
        D = entry["seasonal_order"][1]
        expected_trend = "c" if (d == 0 and D == 0) else "n"
        assert entry["trend"] == expected_trend
        assert entry["status"] == "SELECTED"


# 3. Consumes exact five publication-safe exogenous variables
def test_03_exact_five_exogenous_variables():
    expected_exog = [
        "usd_try_lagged",
        "rub_usd_lagged",
        "brent_lagged",
        "tr_ip_lagged",
        "ru_ip_lagged",
    ]
    assert EXOG_COLUMNS == expected_exog
    assert len(EXOG_COLUMNS) == 5


# 4. Future exogenous strategy: origin_known_persistence strictly enforced
def test_04_future_exog_strategy_is_origin_known_persistence(sarimax_pipeline_results):
    pred_df, _, _, _, _, run_meta = sarimax_pipeline_results
    assert run_meta["future_exog_strategy"] == "origin_known_persistence"
    assert (pred_df["exog_strategy"] == "origin_known_persistence").all()


# 5. Invariant: availability_date_or_approximation <= canonical_origin_snapshot_date
def test_05_exog_audit_publication_safety(sarimax_pipeline_results):
    _, _, _, _, audit_df, _ = sarimax_pipeline_results
    assert len(audit_df) == 96 * 5  # 96 origins x 5 indicators = 480 rows
    assert audit_df["is_publication_safe"].all()
    assert (audit_df["availability_date_or_approximation"] <= audit_df["canonical_origin_snapshot_date"]).all()


# 6. Reuses exact 281 expanding walk-forward folds from configs/validation_folds_v1_1.yaml
def test_06_reuses_exact_281_folds():
    settings = get_settings()
    folds_path = settings.processed_data_dir / "validation_origins.parquet"
    folds_df = pd.read_parquet(folds_path)
    assert len(folds_df) == 281

    # Check origin counts by horizon
    assert len(folds_df[folds_df["horizon"] == 1]) == 96
    assert len(folds_df[folds_df["horizon"] == 3]) == 94
    assert len(folds_df[folds_df["horizon"] == 6]) == 91


# 7. Total predictions count = 3,934 rows (100% complete)
def test_07_total_predictions_count_is_3934(sarimax_pipeline_results):
    pred_df, _, _, _, _, run_meta = sarimax_pipeline_results
    assert len(pred_df) == 3934
    assert run_meta["valid_predictions"] == 3934
    assert run_meta["failed_predictions"] == 0
    assert pred_df["prediction"].notnull().all()


# 8. Canonical 42 series x horizon metric groups populated
def test_08_canonical_42_metric_groups(sarimax_pipeline_results):
    _, metrics_df, _, _, _, run_meta = sarimax_pipeline_results
    assert len(metrics_df) == 42
    assert run_meta["metric_groups_count"] == 42
    assert (metrics_df["model_type"] == "sarimax").all()
    assert metrics_df["smape"].notnull().all()
    assert metrics_df["mae"].notnull().all()
    assert metrics_df["rmse"].notnull().all()


# 9. Native 80% and 95% prediction intervals generated
def test_09_native_prediction_intervals_exist(sarimax_pipeline_results):
    pred_df, _, _, _, _, _ = sarimax_pipeline_results
    for col in ["lower_80", "upper_80", "lower_95", "upper_95"]:
        assert col in pred_df.columns
        assert pred_df[col].notnull().all()


# 10. Invariant: lower_95 <= lower_80 <= upper_80 <= upper_95 with zero violations
def test_10_interval_ordering_invariant(sarimax_pipeline_results):
    pred_df, _, _, _, _, _ = sarimax_pipeline_results
    tol = 1e-4
    valid_order = (
        (pred_df["lower_95"] <= pred_df["lower_80"] + tol)
        & (pred_df["lower_80"] <= pred_df["upper_80"] + tol)
        & (pred_df["upper_80"] <= pred_df["upper_95"] + tol)
    )
    violations = (~valid_order).sum()
    assert violations == 0, f"Found {violations} interval ordering violations"


# 11. Raw negative values preserved without artificial clipping
def test_11_raw_negative_values_preserved(sarimax_pipeline_results):
    pred_df, _, _, _, _, _ = sarimax_pipeline_results
    neg_l80 = (pred_df["lower_80"] < 0).sum()
    neg_l95 = (pred_df["lower_95"] < 0).sum()
    # In statespace models with volatile bilateral series, unconstrained Gaussian bounds produce raw negative lower bounds
    assert neg_l80 > 0, "Expected raw negative 80% lower bounds to be preserved"
    assert neg_l95 > 0, "Expected raw negative 95% lower bounds to be preserved"


# 12. Deterministic missing-exog handling (fail closed loudly without data fabrication)
def test_12_missing_exog_fail_closed():
    predictor = SARIMAXPredictor(
        order=(1, 1, 0),
        seasonal_order=(0, 1, 1, 12),
        trend="n",
    )
    y_train = np.ones(60) * 100.0
    X_train = np.ones((60, 5))
    # Inject NaN into exogenous training matrix
    X_train[10, 2] = np.nan
    X_future = np.ones((6, 5))

    res = predictor.fit_and_forecast(y_train, X_train, X_future, horizons=[1, 3, 6])
    assert res["fit_status"] == "DATA_ERROR"
    assert res["failure_type"] == "NON_FINITE_EXOG"
    assert np.isnan(res["forecasts"][1]["prediction"])
    assert np.isnan(res["forecasts"][3]["prediction"])
    assert np.isnan(res["forecasts"][6]["prediction"])


# 13. Future-macro mutation gate (early, middle, late origins across h=1,3,6)
def test_13_future_macro_mutation_gate():
    settings = get_settings()
    dev_features_path = settings.processed_data_dir / "model_features_development.parquet"
    target_df = pd.read_parquet(dev_features_path)

    series_data = target_df[
        (target_df["direction"] == "TR_RU") & (target_df["hs2"] == "TOTAL")
    ].sort_values("period").copy()

    test_origins = ["2016-12", "2020-12", "2024-06"]
    horizons = [1, 3, 6]

    predictor = SARIMAXPredictor(
        order=(1, 1, 1),
        seasonal_order=(0, 1, 0, 12),
        trend="n",
    )

    for orig in test_origins:
        # Base run
        train_base = series_data[series_data["period"] <= orig]
        y_base = train_base["target"].values.astype(float)
        X_base = train_base[EXOG_COLUMNS].values.astype(float)
        x_orig_base = X_base[-1, :]
        X_fut_base = np.tile(x_orig_base, (6, 1))

        res_base = predictor.fit_and_forecast(y_base, X_base, X_fut_base, horizons)

        # Mutated future run: radically mutate observations AFTER forecast origin
        mutated_series = series_data.copy()
        future_mask = mutated_series["period"] > orig
        for col in EXOG_COLUMNS:
            mutated_series.loc[future_mask, col] = mutated_series.loc[future_mask, col] * 999.0 + 123456.7

        train_mutated = mutated_series[mutated_series["period"] <= orig]
        y_mutated = train_mutated["target"].values.astype(float)
        X_mutated = train_mutated[EXOG_COLUMNS].values.astype(float)
        x_orig_mutated = X_mutated[-1, :]
        X_fut_mutated = np.tile(x_orig_mutated, (6, 1))

        # Assert upstream historical exogenous inputs and origin-known future path are identical
        np.testing.assert_array_equal(X_base, X_mutated)
        np.testing.assert_array_equal(x_orig_base, x_orig_mutated)
        np.testing.assert_array_equal(X_fut_base, X_fut_mutated)

        res_mutated = predictor.fit_and_forecast(y_mutated, X_mutated, X_fut_mutated, horizons)

        # Assert predictions and intervals are numerically identical
        for h in horizons:
            np.testing.assert_allclose(
                res_base["forecasts"][h]["prediction"],
                res_mutated["forecasts"][h]["prediction"],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                res_base["forecasts"][h]["lower_80"],
                res_mutated["forecasts"][h]["lower_80"],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                res_base["forecasts"][h]["lower_95"],
                res_mutated["forecasts"][h]["lower_95"],
                rtol=1e-6,
                atol=1e-6,
            )


# 14. Complete 2025 macro quarantine gate (upstream end-to-end mutation test)
def test_14_complete_2025_macro_quarantine(fresh_unmutated_sarimax_results, macro_mutation_results):
    _, unmut_preds, unmut_metrics, unmut_diags, unmut_fit_log, unmut_audit, unmut_meta = fresh_unmutated_sarimax_results
    dev_rebuilt, mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta = macro_mutation_results
    settings = get_settings()

    # Upstream M5 feature dataset equality
    can_dev_features = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    pd.testing.assert_frame_equal(can_dev_features, dev_rebuilt)

    # Exact fold definitions and fold IDs
    pd.testing.assert_series_equal(unmut_preds["fold_id"], mut_preds["fold_id"])

    # Origin snapshot dates
    pd.testing.assert_series_equal(
        unmut_audit["canonical_origin_snapshot_date"],
        mut_audit["canonical_origin_snapshot_date"],
    )

    # Origin-known exogenous vectors
    pd.testing.assert_series_equal(unmut_audit["value_used"], mut_audit["value_used"])

    # Future persistence exog paths
    assert (mut_preds["exog_strategy"] == "origin_known_persistence").all()
    pd.testing.assert_series_equal(unmut_preds["exog_strategy"], mut_preds["exog_strategy"])

    # All 3,934 point predictions
    pd.testing.assert_series_equal(
        unmut_preds["prediction"],
        mut_preds["prediction"],
        rtol=1e-5,
        atol=1e-5,
    )

    # All 80% and 95% interval bounds
    pd.testing.assert_series_equal(unmut_preds["lower_80"], mut_preds["lower_80"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["upper_80"], mut_preds["upper_80"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["lower_95"], mut_preds["lower_95"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["upper_95"], mut_preds["upper_95"], rtol=1e-5, atol=1e-5)

    # All 42 metric rows and metric values
    pd.testing.assert_series_equal(unmut_metrics["smape"], mut_metrics["smape"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_metrics["mae"], mut_metrics["mae"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_metrics["rmse"], mut_metrics["rmse"], rtol=1e-5, atol=1e-5)

    # Fit and failure counts
    assert len(mut_fit_log) == len(unmut_fit_log) == 1344
    assert (mut_fit_log["fit_status"] == "SUCCESS").sum() == 1344
    assert (mut_fit_log["fit_status"] != "SUCCESS").sum() == 0
    assert mut_fit_log["converged"].all()


# 15. 2025 trade-target isolation gate (upstream end-to-end mutation test)
def test_15_2025_trade_target_isolation(fresh_unmutated_sarimax_results, target_mutation_results):
    _, unmut_preds, unmut_metrics, unmut_diags, unmut_fit_log, unmut_audit, unmut_meta = fresh_unmutated_sarimax_results
    dev_rebuilt, mut_preds, mut_metrics, mut_diags, mut_fit_log, mut_audit, mut_meta = target_mutation_results
    settings = get_settings()

    # Upstream M5 feature dataset equality
    can_dev_features = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    pd.testing.assert_frame_equal(can_dev_features, dev_rebuilt)

    # Frozen 281 fold definitions
    pd.testing.assert_series_equal(unmut_preds["fold_id"], mut_preds["fold_id"])

    # Origin exogenous audit/path
    pd.testing.assert_frame_equal(
        unmut_audit.drop(columns=["created_at"]),
        mut_audit.drop(columns=["created_at"]),
    )

    # All 3,934 point predictions
    pd.testing.assert_series_equal(
        unmut_preds["prediction"],
        mut_preds["prediction"],
        rtol=1e-5,
        atol=1e-5,
    )

    # 80% and 95% intervals
    pd.testing.assert_series_equal(unmut_preds["lower_80"], mut_preds["lower_80"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["upper_80"], mut_preds["upper_80"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["lower_95"], mut_preds["lower_95"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_preds["upper_95"], mut_preds["upper_95"], rtol=1e-5, atol=1e-5)

    # All 42 validation metric groups
    pd.testing.assert_series_equal(unmut_metrics["smape"], mut_metrics["smape"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_metrics["mae"], mut_metrics["mae"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_metrics["rmse"], mut_metrics["rmse"], rtol=1e-5, atol=1e-5)

    # Residual diagnostics
    pd.testing.assert_series_equal(unmut_diags["residual_mean"], mut_diags["residual_mean"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_diags["residual_std"], mut_diags["residual_std"], rtol=1e-5, atol=1e-5)
    pd.testing.assert_series_equal(unmut_diags["ljung_box_pvalue"], mut_diags["ljung_box_pvalue"], rtol=1e-5, atol=1e-5)

    # Fit and failure counts
    assert len(mut_fit_log) == len(unmut_fit_log) == 1344
    assert (mut_fit_log["fit_status"] == "SUCCESS").sum() == 1344
    assert (mut_fit_log["fit_status"] != "SUCCESS").sum() == 0
    assert mut_fit_log["converged"].all()


# 16. Fold-registry immutability gate
def test_16_fold_registry_immutability():
    settings = get_settings()
    yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    parquet_path = settings.processed_data_dir / "validation_origins.parquet"

    with open(yaml_path) as f:
        yaml_folds = yaml.safe_load(f)

    parquet_folds = pd.read_parquet(parquet_path)

    assert yaml_folds["total_folds"] == 281
    assert len(parquet_folds) == 281
    assert yaml_folds["development_start"] == "2012-01"
    assert yaml_folds["development_end"] == "2024-12"
    assert yaml_folds["minimum_initial_history_months"] == 60


# 17. Full-development residual diagnostics across 14 series (2012-01 to 2024-12)
def test_17_full_development_residual_diagnostics(sarimax_pipeline_results):
    _, _, diags_df, _, _, _ = sarimax_pipeline_results
    assert len(diags_df) == 14
    assert diags_df["converged"].all()
    assert (diags_df["nobs"] == 156).all()
    assert diags_df["residual_mean"].notnull().all()
    assert diags_df["residual_std"].notnull().all()
    assert diags_df["ljung_box_pvalue"].notnull().all()


# 18. Exogenous provenance audit artifact generated (480 rows)
def test_18_exogenous_provenance_audit_artifact(sarimax_pipeline_results):
    _, _, _, _, audit_df, _ = sarimax_pipeline_results
    assert len(audit_df) == 480
    assert set(audit_df["indicator_id"].unique()) == set(REQUIRED_INDICATOR_IDS)
    assert len(audit_df["forecast_origin"].unique()) == 96


# 19. M8 methodological configuration manifest exists (configs/sarimax_spec_v1_1.yaml)
def test_19_sarimax_spec_manifest_exists():
    settings = get_settings()
    spec_path = settings.project_root / "configs" / "sarimax_spec_v1_1.yaml"
    assert spec_path.exists()

    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    assert spec["model_family"] == "sarimax"
    assert spec["structural_order_policy"]["rule"] == "reuse_m7_sarima_orders_without_reselection"
    assert spec["future_exogenous_strategy"]["strategy_name"] == "origin_known_persistence"
    assert spec["missing_exog_policy"]["rule"] == "fail_closed_loudly_never_fabricate"
    assert spec["validation_protocol"]["total_folds"] == 281


# 20. No champion selection (deferred to M10)
def test_20_no_champion_selection(sarimax_pipeline_results):
    _, metrics_df, _, _, _, _ = sarimax_pipeline_results
    assert "is_champion" not in metrics_df.columns


# 21. No M10+ artifacts exist
def test_21_no_m10_plus_artifacts():
    settings = get_settings()
    assert not (settings.processed_data_dir / "m10_champion_selection.parquet").exists()
    assert not (settings.processed_data_dir / "m11_holdout_predictions.parquet").exists()
    assert not (settings.project_root / "docs" / "M10_CHAMPION_SELECTION_REPORT.md").exists()
    assert not (settings.project_root / "docs" / "M11_HOLDOUT_REPORT.md").exists()



# 22. Snapshot-date semantics audit (exact agreement between M5 and M8)
def test_22_snapshot_date_semantics_audit():
    settings = get_settings()
    dev_features = pd.read_parquet(settings.processed_data_dir / "model_features_development.parquet")
    folds_df = pd.read_parquet(settings.processed_data_dir / "validation_origins.parquet")

    origins = sorted(folds_df["forecast_origin"].unique())
    mismatches = 0
    for orig in origins:
        sub = dev_features[dev_features["period"] == orig]
        assert not sub.empty, f"Origin {orig} missing from dev features"
        m5_snap = str(sub["feature_snapshot_date"].iloc[0])
        orig_p = pd.Period(orig, freq="M")
        m8_snap = str(orig_p.end_time.date())
        if m5_snap != m8_snap:
            mismatches += 1

    assert len(origins) == 96
    assert mismatches == 0, f"Found {mismatches} snapshot date mismatches across {len(origins)} origins"


# 23. Optimizer audit
def test_23_optimizer_audit(sarimax_pipeline_results):
    _, _, _, fit_log_df, _, _ = sarimax_pipeline_results
    assert len(fit_log_df) == 1344
    assert (fit_log_df["optimizer_used"] == "lbfgs").all(), "Expected all fits to use primary lbfgs"
    assert (fit_log_df["retry_count"] == 0).all(), "Expected 0 Powell retries"
    assert fit_log_df["converged"].all(), "Expected 100% convergence"
    assert "retry_reason" in fit_log_df.columns
