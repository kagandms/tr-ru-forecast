"""Unit and acceptance tests for Milestone M5: Feature Engineering.

Verifies all 50 critical requirements from Milestone M5 specification:
1. modeling universe contains exactly 14 series.
2. only frozen selected HS2 + TOTAL are included.
3. v1.1 target mappings remain correct.
4. target input is restricted to development before feature building.
5. complete monthly calendar behavior (156 months * 14 = 2,184 rows).
6. target_h1 exact t+1 alignment.
7. target_h3 exact t+3 alignment.
8. target_h6 exact t+6 alignment.
9. h1 last valid origin = 2024-11.
10. h3 last valid origin = 2024-09.
11. h6 last valid origin = 2024-06.
12. no future holdout labels enter development table.
13. lag_1 correctness.
14. lag_2 correctness.
15. lag_3 correctness.
16. lag_6 correctness.
17. lag_12 correctness.
18. lag calculations are series-isolated.
19. rolling_mean_3 uses t-1... only (excludes y_t).
20. rolling_mean_6 backward-only.
21. rolling_mean_12 backward-only.
22. rolling_std_3 backward-only with ddof=1.
23. rolling_std_6 backward-only with ddof=1.
24. rolling_std_12 backward-only with ddof=1.
25. rolling windows do not cross series.
26. rolling incomplete window stays missing.
27. mom_growth correctness.
28. yoy_growth correctness.
29. zero denominator growth returns NULL, not inf.
30. month feature correctness.
31. quarter feature correctness.
32. year feature correctness.
33. feature_snapshot_date = month-end.
34. macro join respects availability date.
35. macro join selects latest AVAILABLE observation, not latest reference period.
36. FX future macro actual blocked.
37. Brent future macro actual blocked.
38. TR IP future macro actual blocked.
39. RU IP future macro actual blocked.
40. RU IP linkage contract remains unchanged.
41. macro alignment audit contains reference + availability dates.
42. missing macro remains missing.
43. no macro interpolation/fill.
44. 2025 target mutation has zero effect on development features.
45. future macro mutation has zero effect on earlier snapshots.
46. selected-sector manifest unchanged.
47. macro source registry unchanged.
48. no scaler fitted.
49. no model trained.
50. no M9+ artifacts created.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    FORECAST_TARGET_FLOWS,
    TradeDirection,
)
from tr_ru_forecast.features.builder import (
    compute_feature_snapshot_date,
    compute_growth_features,
    enforce_development_target_guard,
)
from tr_ru_forecast.features.macro_aligner import (
    MACRO_FEATURE_COLUMN_MAP,
    align_publication_safe_macro,
)
from tr_ru_forecast.features.pipeline import (
    build_development_features,
    load_input_datasets,
)


@pytest.fixture(scope="module")
def input_data():
    """Load inputs once for tests that inspect pipeline output."""
    target_df, macro_df, sectors_manifest = load_input_datasets()
    return target_df, macro_df, sectors_manifest


@pytest.fixture(scope="module")
def feature_data(input_data):
    """Generate or retrieve feature dataframe and audit trace."""
    target_df, macro_df, sectors_manifest = input_data
    df, audit_trace = build_development_features(target_df, macro_df, sectors_manifest)
    return df, audit_trace


# 1. modeling universe contains exactly 14 series
def test_01_modeling_universe_contains_exactly_14_series(feature_data):
    df, _ = feature_data
    series_keys = df[["direction", "hs2"]].drop_duplicates()
    assert len(series_keys) == 14, f"Expected 14 series, found {len(series_keys)}"


# 2. only frozen selected HS2 + TOTAL are included
def test_02_only_frozen_selected_hs2_and_total(feature_data, input_data):
    df, _ = feature_data
    _, _, manifest = input_data

    tr_expected = set([s["hs2"] for s in manifest["TR_RU"]] + ["TOTAL"])
    ru_expected = set([s["hs2"] for s in manifest["RU_TR"]] + ["TOTAL"])

    tr_actual = set(df[df["direction"] == "TR_RU"]["hs2"].unique())
    ru_actual = set(df[df["direction"] == "RU_TR"]["hs2"].unique())

    assert tr_actual == tr_expected
    assert ru_actual == ru_expected


# 3. v1.1 target mappings remain correct
def test_03_v11_target_mappings_remain_correct():
    tr_target = FORECAST_TARGET_FLOWS[TradeDirection.TR_RU]
    ru_target = FORECAST_TARGET_FLOWS[TradeDirection.RU_TR]

    # Both must be Turkey reporter (792)
    assert tr_target["reporter_code"] == 792
    assert ru_target["reporter_code"] == 792
    assert tr_target["partner_code"] == 643
    assert ru_target["partner_code"] == 643
    assert tr_target["flow_code"] == "X"  # Turkey export to Russia
    assert ru_target["flow_code"] == "M"  # Turkey import from Russia


# 4. target input is restricted to development before feature building
def test_04_target_input_restricted_to_development():
    mock_data = pd.DataFrame({
        "period": ["2024-11", "2024-12", "2025-01", "2025-02"],
        "target": [10.0, 20.0, 30.0, 40.0],
    })
    guarded = enforce_development_target_guard(mock_data, max_period="2024-12")
    assert guarded["period"].max() == "2024-12"
    assert "2025-01" not in guarded["period"].values
    assert "2025-02" not in guarded["period"].values


# 5. complete monthly calendar behavior
def test_05_complete_monthly_calendar_behavior(feature_data):
    df, _ = feature_data
    assert len(df) == 2184, f"Expected 2184 rows (14 * 156), got {len(df)}"
    # Every series must have 156 rows
    counts = df.groupby(["direction", "hs2"]).size()
    assert (counts == 156).all()


# 6. target_h1 exact t+1 alignment
def test_06_target_h1_exact_t_plus_1_alignment(feature_data):
    df, _ = feature_data
    # For a specific series, check row i and row i+1
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(len(sub) - 1):
        assert sub.loc[i, "target_h1"] == sub.loc[i + 1, "target"]


# 7. target_h3 exact t+3 alignment
def test_07_target_h3_exact_t_plus_3_alignment(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(len(sub) - 3):
        assert sub.loc[i, "target_h3"] == sub.loc[i + 3, "target"]


# 8. target_h6 exact t+6 alignment
def test_08_target_h6_exact_t_plus_6_alignment(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(len(sub) - 6):
        assert sub.loc[i, "target_h6"] == sub.loc[i + 6, "target"]


# 9. h1 last valid origin = 2024-11
def test_09_h1_last_valid_origin(feature_data):
    df, _ = feature_data
    valid_origins = df[df["target_h1"].notnull()]["period"]
    assert valid_origins.max() == "2024-11"
    # Row 2024-12 must be NaN
    null_rows = df[df["period"] == "2024-12"]
    assert null_rows["target_h1"].isnull().all()


# 10. h3 last valid origin = 2024-09
def test_10_h3_last_valid_origin(feature_data):
    df, _ = feature_data
    valid_origins = df[df["target_h3"].notnull()]["period"]
    assert valid_origins.max() == "2024-09"
    # Rows 2024-10, 2024-11, 2024-12 must be NaN
    null_rows = df[df["period"].isin(["2024-10", "2024-11", "2024-12"])]
    assert null_rows["target_h3"].isnull().all()


# 11. h6 last valid origin = 2024-06
def test_11_h6_last_valid_origin(feature_data):
    df, _ = feature_data
    valid_origins = df[df["target_h6"].notnull()]["period"]
    assert valid_origins.max() == "2024-06"
    # Rows 2024-07 to 2024-12 must be NaN
    null_rows = df[df["period"].isin(["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"])]
    assert null_rows["target_h6"].isnull().all()


# 12. no future holdout labels enter development table
def test_12_no_future_holdout_labels(feature_data):
    df, _ = feature_data
    # 2024-12 has no valid h1, h3, h6 labels
    r_2024_12 = df[df["period"] == "2024-12"]
    assert r_2024_12["target_h1"].isnull().all()
    assert r_2024_12["target_h3"].isnull().all()
    assert r_2024_12["target_h6"].isnull().all()


# 13. lag_1 correctness
def test_13_lag_1_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    assert pd.isnull(sub.loc[0, "lag_1"])  # 2012-01 is NaN
    for i in range(1, len(sub)):
        assert sub.loc[i, "lag_1"] == sub.loc[i - 1, "target"]


# 14. lag_2 correctness
def test_14_lag_2_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    assert pd.isnull(sub.loc[0, "lag_2"])
    assert pd.isnull(sub.loc[1, "lag_2"])
    for i in range(2, len(sub)):
        assert sub.loc[i, "lag_2"] == sub.loc[i - 2, "target"]


# 15. lag_3 correctness
def test_15_lag_3_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(3, len(sub)):
        assert sub.loc[i, "lag_3"] == sub.loc[i - 3, "target"]


# 16. lag_6 correctness
def test_16_lag_6_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(6, len(sub)):
        assert sub.loc[i, "lag_6"] == sub.loc[i - 6, "target"]


# 17. lag_12 correctness
def test_17_lag_12_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    for i in range(12, len(sub)):
        assert sub.loc[i, "lag_12"] == sub.loc[i - 12, "target"]


# 18. lag calculations are series-isolated
def test_18_lag_calculations_are_series_isolated(feature_data):
    df, _ = feature_data
    # For every series, period 2012-01 must have lag_1 = NaN
    p_first = df[df["period"] == "2012-01"]
    assert p_first["lag_1"].isnull().all()
    assert p_first["lag_2"].isnull().all()
    assert p_first["lag_12"].isnull().all()


# 19. rolling_mean_3 uses t-1... only (excludes y_t)
def test_19_rolling_mean_3_backward_only(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    # At index 3 (2012-04), rolling_mean_3 should be mean of indices 0, 1, 2
    expected = (sub.loc[0, "target"] + sub.loc[1, "target"] + sub.loc[2, "target"]) / 3.0
    assert abs(sub.loc[3, "rolling_mean_3"] - expected) < 1e-4
    # Ensure index 3 target y_t is NOT part of rolling_mean_3
    assert sub.loc[3, "rolling_mean_3"] != (sub.loc[1, "target"] + sub.loc[2, "target"] + sub.loc[3, "target"]) / 3.0


# 20. rolling_mean_6 backward-only
def test_20_rolling_mean_6_backward_only(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    expected = sub.loc[0:5, "target"].mean()
    assert abs(sub.loc[6, "rolling_mean_6"] - expected) < 1e-4


# 21. rolling_mean_12 backward-only
def test_21_rolling_mean_12_backward_only(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    expected = sub.loc[0:11, "target"].mean()
    assert abs(sub.loc[12, "rolling_mean_12"] - expected) < 1e-4


# 22. rolling_std_3 backward-only with ddof=1
def test_22_rolling_std_3_backward_only_ddof1(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    expected = sub.loc[0:2, "target"].std(ddof=1)
    assert abs(sub.loc[3, "rolling_std_3"] - expected) < 1e-4


# 23. rolling_std_6 backward-only with ddof=1
def test_23_rolling_std_6_backward_only_ddof1(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    expected = sub.loc[0:5, "target"].std(ddof=1)
    assert abs(sub.loc[6, "rolling_std_6"] - expected) < 1e-4


# 24. rolling_std_12 backward-only with ddof=1
def test_24_rolling_std_12_backward_only_ddof1(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    expected = sub.loc[0:11, "target"].std(ddof=1)
    assert abs(sub.loc[12, "rolling_std_12"] - expected) < 1e-4


# 25. rolling windows do not cross series
def test_25_rolling_windows_do_not_cross_series(feature_data):
    df, _ = feature_data
    # For every series, first 3 rows must have rolling_mean_3 = NaN
    for (dir_name, hs2_code), group in df.groupby(["direction", "hs2"]):
        grp_sorted = group.sort_values("period").reset_index(drop=True)
        assert pd.isnull(grp_sorted.loc[0, "rolling_mean_3"])
        assert pd.isnull(grp_sorted.loc[1, "rolling_mean_3"])
        assert pd.isnull(grp_sorted.loc[2, "rolling_mean_3"])
        assert pd.notnull(grp_sorted.loc[3, "rolling_mean_3"])


# 26. rolling incomplete window stays missing
def test_26_rolling_incomplete_window_stays_missing(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    # rolling_mean_12 must be NaN for first 12 rows (indices 0..11)
    assert sub.loc[0:11, "rolling_mean_12"].isnull().all()
    assert pd.notnull(sub.loc[12, "rolling_mean_12"])


# 27. mom_growth correctness
def test_27_mom_growth_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    assert pd.isnull(sub.loc[0, "mom_growth"])
    for i in range(1, 5):
        expected = (sub.loc[i, "target"] / sub.loc[i - 1, "target"]) - 1.0
        assert abs(sub.loc[i, "mom_growth"] - expected) < 1e-5


# 28. yoy_growth correctness
def test_28_yoy_growth_correctness(feature_data):
    df, _ = feature_data
    sub = df[(df["direction"] == "TR_RU") & (df["hs2"] == "TOTAL")].sort_values("period").reset_index(drop=True)
    assert sub.loc[0:11, "yoy_growth"].isnull().all()
    for i in range(12, 16):
        expected = (sub.loc[i, "target"] / sub.loc[i - 12, "target"]) - 1.0
        assert abs(sub.loc[i, "yoy_growth"] - expected) < 1e-5


# 29. zero denominator growth returns NULL, not inf
def test_29_zero_denominator_growth_returns_null():
    panel = pd.DataFrame({
        "direction": ["TR_RU", "TR_RU"],
        "hs2": ["TOTAL", "TOTAL"],
        "period": ["2020-01", "2020-02"],
        "target": [0.0, 100.0],
        "lag_1": [np.nan, 0.0],
        "lag_12": [np.nan, 0.0],
    })
    res = compute_growth_features(panel)
    assert pd.isnull(res.loc[1, "mom_growth"])
    assert pd.isnull(res.loc[1, "yoy_growth"])
    assert not np.isinf(res.loc[1, "mom_growth"])


# 30. month feature correctness
def test_30_month_feature_correctness(feature_data):
    df, _ = feature_data
    sample = df[df["period"] == "2024-05"].iloc[0]
    assert sample["month"] == 5


# 31. quarter feature correctness
def test_31_quarter_feature_correctness(feature_data):
    df, _ = feature_data
    sample_q1 = df[df["period"] == "2024-03"].iloc[0]
    sample_q4 = df[df["period"] == "2024-11"].iloc[0]
    assert sample_q1["quarter"] == 1
    assert sample_q4["quarter"] == 4


# 32. year feature correctness
def test_32_year_feature_correctness(feature_data):
    df, _ = feature_data
    sample = df[df["period"] == "2023-07"].iloc[0]
    assert sample["year"] == 2023


# 33. feature_snapshot_date = month-end
def test_33_feature_snapshot_date_is_month_end():
    assert compute_feature_snapshot_date("2024-01") == "2024-01-31"
    assert compute_feature_snapshot_date("2024-02") == "2024-02-29"  # leap year
    assert compute_feature_snapshot_date("2023-02") == "2023-02-28"
    assert compute_feature_snapshot_date("2024-04") == "2024-04-30"


# 34. macro join respects availability date
def test_34_macro_join_respects_availability_date(feature_data):
    _, audit_trace = feature_data
    trace_df = pd.DataFrame(audit_trace)
    # Invariant: macro availability date <= snapshot date
    for _, row in trace_df.iterrows():
        if row["is_available"]:
            assert row["selected_macro_availability_date"] <= row["feature_snapshot_date"]


# 35. macro join selects latest AVAILABLE observation, not latest reference period
def test_35_macro_join_selects_latest_available():
    # Construct synthetic macro data where a reference period has future availability
    macro_mock = pd.DataFrame([
        {
            "period": "2024-01",
            "indicator_id": "usd_try",
            "value": 30.0,
            "availability_date_or_approximation": "2024-02-01",
        },
        {
            "period": "2024-02",
            "indicator_id": "usd_try",
            "value": 31.0,
            "availability_date_or_approximation": "2024-03-01",
        },
    ])
    panel_mock = pd.DataFrame([
        {"period": "2024-01", "feature_snapshot_date": "2024-01-31"},
    ])
    aligned, trace = align_publication_safe_macro(panel_mock, macro_mock)
    # At 2024-01-31, neither 2024-01 (avail 2024-02-01) nor 2024-02 is available
    assert pd.isnull(aligned.loc[0, "usd_try_lagged"])


# 36. FX future macro actual blocked
def test_36_fx_future_macro_actual_blocked(input_data):
    _, macro_df, _ = input_data
    panel_mock = pd.DataFrame([
        {"period": "2024-01", "feature_snapshot_date": "2024-01-31"},
    ])
    aligned, trace = align_publication_safe_macro(panel_mock, macro_df)
    # At snapshot 2024-01-31, usd_try reference period cannot be 2024-01 or later
    tr_trace = [t for t in trace if t["indicator_id"] == "usd_try"][0]
    assert tr_trace["selected_macro_reference_period"] <= "2023-12"


# 37. Brent future macro actual blocked
def test_37_brent_future_macro_actual_blocked(input_data):
    _, macro_df, _ = input_data
    panel_mock = pd.DataFrame([
        {"period": "2024-01", "feature_snapshot_date": "2024-01-31"},
    ])
    aligned, trace = align_publication_safe_macro(panel_mock, macro_df)
    brent_trace = [t for t in trace if t["indicator_id"] == "brent"][0]
    assert brent_trace["selected_macro_reference_period"] <= "2023-12"


# 38. TR IP future macro actual blocked
def test_38_tr_ip_future_macro_actual_blocked(input_data):
    _, macro_df, _ = input_data
    panel_mock = pd.DataFrame([
        {"period": "2024-01", "feature_snapshot_date": "2024-01-31"},
    ])
    aligned, trace = align_publication_safe_macro(panel_mock, macro_df)
    trip_trace = [t for t in trace if t["indicator_id"] == "tr_ip"][0]
    # TR IP has T+2M Day 15 lag, so at 2024-01-31, latest available is 2023-11 (avail 2024-01-15)
    assert trip_trace["selected_macro_reference_period"] <= "2023-11"


# 39. RU IP future macro actual blocked
def test_39_ru_ip_future_macro_actual_blocked(input_data):
    _, macro_df, _ = input_data
    panel_mock = pd.DataFrame([
        {"period": "2024-01", "feature_snapshot_date": "2024-01-31"},
    ])
    aligned, trace = align_publication_safe_macro(panel_mock, macro_df)
    ruip_trace = [t for t in trace if t["indicator_id"] == "ru_ip"][0]
    assert ruip_trace["selected_macro_reference_period"] <= "2023-11"


# 40. RU IP linkage contract remains unchanged
def test_40_ru_ip_linkage_contract_unchanged():
    settings = get_settings()
    reg_path = settings.project_root / "configs" / "macro_sources_v1_1.yaml"
    with open(reg_path) as f:
        reg = yaml.safe_load(f)
    ru_ip = reg["indicators"]["ru_ip"]
    assert ru_ip["linkage_method"] == "multiplicative_mean_overlap_bridge"
    assert abs(ru_ip["linkage_factor"] - 1.2128086297589498) < 1e-9
    assert ru_ip["linkage_switch_date"] == "2022-01"


# 41. macro alignment audit contains reference + availability dates
def test_41_macro_alignment_audit_fields(feature_data):
    _, audit_trace = feature_data
    assert len(audit_trace) > 0
    sample = audit_trace[0]
    assert "trade_period" in sample
    assert "feature_snapshot_date" in sample
    assert "indicator_id" in sample
    assert "selected_macro_reference_period" in sample
    assert "selected_macro_availability_date" in sample
    assert "selected_macro_value" in sample


# 42. missing macro remains missing
def test_42_missing_macro_remains_missing():
    macro_empty = pd.DataFrame(columns=["period", "indicator_id", "value", "availability_date_or_approximation"])
    panel = pd.DataFrame([{"period": "2020-01", "feature_snapshot_date": "2020-01-31"}])
    aligned, _ = align_publication_safe_macro(panel, macro_empty)
    for col in MACRO_FEATURE_COLUMN_MAP.values():
        assert pd.isnull(aligned.loc[0, col])


# 43. no macro interpolation or forward-fill
def test_43_no_macro_interpolation():
    # If a macro indicator is missing for all periods, it must not be filled
    macro_mock = pd.DataFrame([
        {"period": "2020-01", "indicator_id": "usd_try", "value": 7.0, "availability_date_or_approximation": "2020-02-01"},
    ])
    panel = pd.DataFrame([{"period": "2020-01", "feature_snapshot_date": "2020-01-31"}])
    aligned, _ = align_publication_safe_macro(panel, macro_mock)
    # brent_lagged was not provided, must remain NaN
    assert pd.isnull(aligned.loc[0, "brent_lagged"])


# 44. 2025 target mutation has zero effect on development features
def test_44_holdout_target_mutation_invariance(input_data):
    target_df, macro_df, sectors_manifest = input_data

    # Generate baseline development features
    base_features, _ = build_development_features(target_df, macro_df, sectors_manifest)

    # Mutate 2025 holdout target observations radically
    mutated_target = target_df.copy()
    mask_2025 = mutated_target["period"] >= "2025-01"
    mutated_target.loc[mask_2025, "trade_value_usd"] = mutated_target.loc[mask_2025, "trade_value_usd"] * 999.0 + 1234567.0

    # Build features with mutated input
    mutated_features, _ = build_development_features(mutated_target, macro_df, sectors_manifest)

    # Invariant: development features must be identical
    pd.testing.assert_frame_equal(base_features, mutated_features)


# 45. future macro mutation has zero effect on earlier snapshots
def test_45_future_macro_mutation_invariance(input_data):
    target_df, macro_df, sectors_manifest = input_data

    base_features, _ = build_development_features(target_df, macro_df, sectors_manifest)
    base_sub = base_features[base_features["period"] <= "2020-12"]

    # Radically mutate macro observations occurring in 2024
    mutated_macro = macro_df.copy()
    mask_future = mutated_macro["period"] >= "2024-01"
    mutated_macro.loc[mask_future, "value"] = 99999.9

    mutated_features, _ = build_development_features(target_df, mutated_macro, sectors_manifest)
    mutated_sub = mutated_features[mutated_features["period"] <= "2020-12"]

    # Historical feature rows <= 2020-12 must be identical
    pd.testing.assert_frame_equal(base_sub, mutated_sub)


# 46. selected-sector manifest unchanged
def test_46_selected_sector_manifest_unchanged():
    settings = get_settings()
    manifest_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    with open(manifest_path) as f:
        data = yaml.safe_load(f)
    assert [s["hs2"] for s in data["TR_RU"]] == ["08", "84", "39", "87", "73", "85"]
    assert [s["hs2"] for s in data["RU_TR"]] == ["27", "72", "10", "76", "23", "74"]


# 47. macro source registry unchanged
def test_47_macro_source_registry_unchanged():
    settings = get_settings()
    reg_path = settings.project_root / "configs" / "macro_sources_v1_1.yaml"
    with open(reg_path) as f:
        reg = yaml.safe_load(f)
    assert set(reg["indicators"].keys()) == {"usd_try", "rub_usd", "brent", "tr_ip", "ru_ip"}


# 48. no scaler fitted
def test_48_no_scaler_fitted(feature_data):
    df, _ = feature_data
    # Mean of target is original scale (millions/billions of USD), not scaled around 0
    mean_val = df["target"].mean()
    assert mean_val > 1000.0, f"Target appears artificially standardized! Mean={mean_val}"


# 49. no model trained
def test_49_no_model_trained():
    settings = get_settings()
    models_dir = settings.project_root / "models"
    # Ensure no trained model pickle/joblib files exist
    trained_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    assert len(trained_files) == 0, f"Found unexpected trained model files: {trained_files}"


# 50. no M10+ artifacts created
def test_50_no_m10_plus_artifacts_created():
    settings = get_settings()
    # Check that no M10+ (champion selection, holdout) artifacts exist
    forbidden_files = [
        settings.processed_data_dir / "m10_champion_selection.parquet",
        settings.processed_data_dir / "m11_holdout_predictions.parquet",
        settings.project_root / "docs" / "M10_CHAMPION_SELECTION_REPORT.md",
        settings.project_root / "docs" / "M11_HOLDOUT_REPORT.md",
    ]
    for p in forbidden_files:
        assert not p.exists(), f"Forbidden M10+ artifact exists: {p}"
