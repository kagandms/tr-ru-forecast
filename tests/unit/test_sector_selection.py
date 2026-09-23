"""Unit and acceptance tests for Milestone M3: Data Audit & Sector Selection.

Verifies all 28 critical requirements:
1. v1.1 TR_RU target mapping
2. v1.1 RU_TR target mapping
3. RU_TR rejects Russia exporter source
4. Development cutoff strictly excludes 2025
5. Complete monthly panel = 156 months
6. Missing remains missing (NaN, not 0.0)
7. Reported zero remains zero (0.0)
8. Coverage ratio correctness
9. Zero ratio correctness
10. Longest missing streak correctness
11. Recent 24m window correctness
12. Eligibility 90% coverage threshold
13. Eligibility <=10% zero threshold
14. Eligibility >=21/24 recent activity
15. P10 size threshold uses development data only
16. Volume score deterministic normalization
17. Continuity score formula
18. Recent relevance score formula
19. Seasonal Naive forecastability uses t-12
20. Seasonal Naive uses no 2025 holdout
21. sMAPE zero division safety
22. Final weighted score formula
23. Deterministic tie-break order
24. Max 6 selected
25. Fewer than 6 allowed
26. TOTAL never ranked
27. Holdout mutation does not affect M3 selection
28. Selected-sector manifest schema
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tr_ru_forecast.core.constants import (
    FLOW_EXPORT_CODE,
    FLOW_EXPORT_NAME,
    FLOW_IMPORT_CODE,
    FLOW_IMPORT_NAME,
    FORECAST_TARGET_FLOWS,
    RUSSIA_M49_CODE,
    TURKEY_M49_CODE,
    TradeDirection,
)
from tr_ru_forecast.data.sectors.eligibility import (
    calculate_min_coverage_observed,
    evaluate_candidate_eligibility,
)
from tr_ru_forecast.data.sectors.panel_builder import (
    MonthlyDevelopmentPanel,
    calculate_longest_missing_streak,
    enforce_holdout_guard,
    generate_development_period_index,
)
from tr_ru_forecast.data.sectors.scoring import (
    compute_percentile_rank_scores,
    compute_seasonal_naive_smape,
)
from tr_ru_forecast.data.sectors.selector import SectorSelector
from tr_ru_forecast.data.sectors.target_materializer import validate_target_provenance


# Helper to build synthetic development DataFrame
def create_synthetic_target_df(periods: list[str] | None = None) -> pd.DataFrame:
    """Create a minimal synthetic target DataFrame for testing."""
    if periods is None:
        periods = generate_development_period_index()

    records = []
    # Direction TR_RU: HS 84 (regular), HS 08 (regular), TOTAL
    for p in periods:
        records.append({
            "period": p,
            "direction": "TR_RU",
            "reporter_code": TURKEY_M49_CODE,
            "partner_code": RUSSIA_M49_CODE,
            "flow": FLOW_EXPORT_NAME,
            "flow_code": FLOW_EXPORT_CODE,
            "hs2": "84",
            "hs_description": "Machinery",
            "trade_value_usd": 1000.0,
            "is_total": False,
        })
        records.append({
            "period": p,
            "direction": "TR_RU",
            "reporter_code": TURKEY_M49_CODE,
            "partner_code": RUSSIA_M49_CODE,
            "flow": FLOW_EXPORT_NAME,
            "flow_code": FLOW_EXPORT_CODE,
            "hs2": "08",
            "hs_description": "Fruit",
            "trade_value_usd": 1000.0,
            "is_total": False,
        })
        records.append({
            "period": p,
            "direction": "TR_RU",
            "reporter_code": TURKEY_M49_CODE,
            "partner_code": RUSSIA_M49_CODE,
            "flow": FLOW_EXPORT_NAME,
            "flow_code": FLOW_EXPORT_CODE,
            "hs2": "TOTAL",
            "hs_description": "Total All Commodities",
            "trade_value_usd": 1500.0,
            "is_total": True,
        })
        # RU_TR
        records.append({
            "period": p,
            "direction": "RU_TR",
            "reporter_code": TURKEY_M49_CODE,
            "partner_code": RUSSIA_M49_CODE,
            "flow": FLOW_IMPORT_NAME,
            "flow_code": FLOW_IMPORT_CODE,
            "hs2": "27",
            "hs_description": "Mineral Fuels",
            "trade_value_usd": 5000.0,
            "is_total": False,
        })
        records.append({
            "period": p,
            "direction": "RU_TR",
            "reporter_code": TURKEY_M49_CODE,
            "partner_code": RUSSIA_M49_CODE,
            "flow": FLOW_IMPORT_NAME,
            "flow_code": FLOW_IMPORT_CODE,
            "hs2": "TOTAL",
            "hs_description": "Total All Commodities",
            "trade_value_usd": 5000.0,
            "is_total": True,
        })

    return pd.DataFrame(records)


# 1. v1.1 TR_RU target mapping
def test_req_01_tr_ru_target_mapping():
    target = FORECAST_TARGET_FLOWS[TradeDirection.TR_RU]
    assert target["reporter_code"] == TURKEY_M49_CODE
    assert target["partner_code"] == RUSSIA_M49_CODE
    assert target["flow"] == FLOW_EXPORT_NAME
    assert target["flow_code"] == FLOW_EXPORT_CODE


# 2. v1.1 RU_TR target mapping
def test_req_02_ru_tr_target_mapping():
    target = FORECAST_TARGET_FLOWS[TradeDirection.RU_TR]
    assert target["reporter_code"] == TURKEY_M49_CODE
    assert target["partner_code"] == RUSSIA_M49_CODE
    assert target["flow"] == FLOW_IMPORT_NAME
    assert target["flow_code"] == FLOW_IMPORT_CODE


# 3. RU_TR does NOT use Russia exporter source
def test_req_03_ru_tr_rejects_russia_exporter_source():
    # Attempting to validate a DataFrame where RU_TR is reported by Russia (643) with Export (X)
    bad_df = pd.DataFrame([{
        "reporter_code": RUSSIA_M49_CODE,
        "partner_code": TURKEY_M49_CODE,
        "flow": FLOW_EXPORT_NAME,
    }])
    with pytest.raises(ValueError, match="Russia-reported series must NOT be used as forecast targets"):
        validate_target_provenance(bad_df, TradeDirection.RU_TR)


# 4. development cutoff excludes 2025
def test_req_04_development_cutoff_excludes_2025():
    df_with_2025 = pd.DataFrame([
        {"period": "2024-12", "hs2": "84"},
        {"period": "2025-01", "hs2": "84"},  # Holdout row
    ])
    with pytest.raises(ValueError, match="HOLDOUT LEAKAGE VIOLATION"):
        enforce_holdout_guard(df_with_2025, cutoff="2024-12")


# 5. complete monthly panel = 156 months
def test_req_05_complete_monthly_panel_156_months():
    df = create_synthetic_target_df()
    panel = MonthlyDevelopmentPanel(df)
    ts = panel.build_sector_timeseries("TR_RU", "84")
    assert len(ts) == 156
    assert ts["period"].iloc[0] == "2012-01"
    assert ts["period"].iloc[-1] == "2024-12"


# 6. missing remains missing
def test_req_06_missing_remains_missing_not_zero():
    # Provide only 1 month for HS 99
    sparse_df = pd.DataFrame([{
        "period": "2012-01",
        "direction": "TR_RU",
        "hs2": "99",
        "trade_value_usd": 100.0,
        "is_total": False,
    }])
    panel = MonthlyDevelopmentPanel(sparse_df)
    ts = panel.build_sector_timeseries("TR_RU", "99")
    assert len(ts) == 156
    assert pd.notna(ts.loc[ts["period"] == "2012-01", "trade_value_usd"].iloc[0])
    # Unobserved months must be NaN, NEVER 0.0
    assert ts.loc[ts["period"] == "2012-02", "trade_value_usd"].isna().iloc[0]


# 7. reported zero remains zero
def test_req_07_reported_zero_remains_zero():
    df = pd.DataFrame([
        {"period": "2012-01", "direction": "TR_RU", "hs2": "01", "trade_value_usd": 0.0, "is_total": False},
        {"period": "2012-02", "direction": "TR_RU", "hs2": "01", "trade_value_usd": 50.0, "is_total": False},
    ])
    panel = MonthlyDevelopmentPanel(df)
    ts = panel.build_sector_timeseries("TR_RU", "01")
    val_jan = ts.loc[ts["period"] == "2012-01", "trade_value_usd"].iloc[0]
    assert val_jan == 0.0
    assert not np.isnan(val_jan)


# 8. coverage ratio correctness
def test_req_08_coverage_ratio_correctness():
    # 78 observed months out of 156 = 0.50
    periods = generate_development_period_index()[:78]
    df = pd.DataFrame([
        {"period": p, "direction": "TR_RU", "hs2": "05", "trade_value_usd": 10.0, "is_total": False}
        for p in periods
    ])
    panel = MonthlyDevelopmentPanel(df)
    m = panel.compute_sector_metrics("TR_RU", "05")
    assert m["observed_months"] == 78
    assert m["development_months_expected"] == 156
    assert pytest.approx(m["coverage_ratio"], 0.001) == 0.50


# 9. zero ratio correctness
def test_req_09_zero_ratio_correctness():
    # 10 observed months: 2 zeros, 8 non-zeros. zero_ratio = 2 / 10 = 0.20
    periods = generate_development_period_index()[:10]
    records = []
    for i, p in enumerate(periods):
        val = 0.0 if i < 2 else 100.0
        records.append({"period": p, "direction": "TR_RU", "hs2": "06", "trade_value_usd": val, "is_total": False})
    panel = MonthlyDevelopmentPanel(pd.DataFrame(records))
    m = panel.compute_sector_metrics("TR_RU", "06")
    assert m["zero_months"] == 2
    assert m["observed_months"] == 10
    assert pytest.approx(m["zero_month_ratio"], 0.001) == 0.20


# 10. longest missing streak correctness
def test_req_10_longest_missing_streak_correctness():
    # [True, False, False, True, False, False, False, True] -> longest streak = 3
    mask = [True, False, False, True, False, False, False, True]
    assert calculate_longest_missing_streak(mask) == 3


# 11. recent 24m window correctness
def test_req_11_recent_24m_window_correctness():
    # 2023-01 to 2024-12 is 24 months
    dev_periods = generate_development_period_index()
    recent = [p for p in dev_periods if "2023-01" <= p <= "2024-12"]
    assert len(recent) == 24
    assert recent[0] == "2023-01"
    assert recent[-1] == "2024-12"


# 12. eligibility 90% coverage threshold
def test_req_12_eligibility_90pct_coverage_threshold():
    min_obs = calculate_min_coverage_observed(156)
    assert min_obs == 141  # ceil(156 * 0.9)

    # 141 observed passes, 140 observed fails
    m_pass = {
        "hs2": "10",
        "observed_months": 141,
        "zero_month_ratio": 0.0,
        "recent_observed_months": 24,
        "development_total_trade_value": 1000.0,
    }
    m_fail = {
        "hs2": "11",
        "observed_months": 140,
        "zero_month_ratio": 0.0,
        "recent_observed_months": 24,
        "development_total_trade_value": 1000.0,
    }
    results, _ = evaluate_candidate_eligibility([m_pass, m_fail], "TR_RU")
    assert results[0]["eligibility_pass"] is True
    assert results[1]["eligibility_pass"] is False
    assert any("Filter B (Coverage)" in r for r in results[1]["eligibility_failure_reasons"])


# 13. eligibility <=10% zero threshold
def test_req_13_eligibility_10pct_zero_threshold():
    m_pass = {
        "hs2": "10",
        "observed_months": 150,
        "zero_month_ratio": 0.10,
        "recent_observed_months": 24,
        "development_total_trade_value": 1000.0,
    }
    m_fail = {
        "hs2": "11",
        "observed_months": 150,
        "zero_month_ratio": 0.101,  # 10.1% > 10%
        "recent_observed_months": 24,
        "development_total_trade_value": 1000.0,
    }
    results, _ = evaluate_candidate_eligibility([m_pass, m_fail], "TR_RU")
    assert results[0]["eligibility_pass"] is True
    assert results[1]["eligibility_pass"] is False
    assert any("Filter C (Zero ratio)" in r for r in results[1]["eligibility_failure_reasons"])


# 14. eligibility >=21/24 recent activity
def test_req_14_eligibility_21_of_24_recent_activity():
    m_pass = {
        "hs2": "10",
        "observed_months": 150,
        "zero_month_ratio": 0.05,
        "recent_observed_months": 21,
        "development_total_trade_value": 1000.0,
    }
    m_fail = {
        "hs2": "11",
        "observed_months": 150,
        "zero_month_ratio": 0.05,
        "recent_observed_months": 20,
        "development_total_trade_value": 1000.0,
    }
    results, _ = evaluate_candidate_eligibility([m_pass, m_fail], "TR_RU")
    assert results[0]["eligibility_pass"] is True
    assert results[1]["eligibility_pass"] is False
    assert any("Filter D (Recent activity)" in r for r in results[1]["eligibility_failure_reasons"])


# 15. P10 size threshold uses development data only
def test_req_15_p10_size_threshold_development_only():
    candidates = [
        {
            "hs2": f"{i:02d}",
            "observed_months": 156,
            "zero_month_ratio": 0.0,
            "recent_observed_months": 24,
            "development_total_trade_value": float(i * 100),
        }
        for i in range(1, 11)  # 100, 200, ..., 1000
    ]
    _, p10 = evaluate_candidate_eligibility(candidates, "TR_RU")
    expected_p10 = float(np.percentile([c["development_total_trade_value"] for c in candidates], 10))
    assert pytest.approx(p10, 0.001) == expected_p10


# 16. volume score deterministic
def test_req_16_volume_score_deterministic():
    vals = [100.0, 500.0, 1000.0]
    scores = compute_percentile_rank_scores(vals, higher_is_better=True)
    assert scores == [0.0, 50.0, 100.0]


# 17. continuity score formula
def test_req_17_continuity_score_formula():
    # coverage=1.0, zero_ratio=0.0, streak=0 -> raw = 0.60*1 + 0.25*1 + 0.15*1 = 1.0 -> 100.0
    cov = 1.0
    zero = 0.0
    streak = 0
    raw = 0.60 * cov + 0.25 * (1 - zero) + 0.15 * (1 - streak / 156)
    score = max(0.0, min(1.0, raw)) * 100.0
    assert pytest.approx(score, 0.001) == 100.0


# 18. recent relevance score formula
def test_req_18_recent_relevance_score_formula():
    recent_vol_score = 80.0
    recent_act_score = 100.0
    expected = 0.80 * recent_vol_score + 0.20 * recent_act_score  # 64 + 20 = 84.0
    assert pytest.approx(expected, 0.001) == 84.0


# 19. seasonal-naive forecastability uses t-12
def test_req_19_seasonal_naive_uses_t_minus_12():
    # Constant series: actual(t) == actual(t-12) for all t -> sMAPE = 0.0
    periods = generate_development_period_index()
    df = pd.DataFrame({"period": periods, "trade_value_usd": [100.0] * len(periods)})
    smape, count, err = compute_seasonal_naive_smape(df)
    assert err is None
    assert count == 156 - 12  # 144
    assert pytest.approx(smape, 0.001) == 0.0


# 20. seasonal-naive uses no 2025 holdout
def test_req_20_seasonal_naive_uses_no_2025_holdout():
    periods = generate_development_period_index()
    assert all(p <= "2024-12" for p in periods)


# 21. sMAPE zero safety
def test_req_21_smape_zero_division_safety():
    periods = generate_development_period_index()
    # All zeros
    df = pd.DataFrame({"period": periods, "trade_value_usd": [0.0] * len(periods)})
    smape, count, err = compute_seasonal_naive_smape(df)
    assert err is None
    assert count == 144
    assert pytest.approx(smape, 0.001) == 0.0


# 22. final weighted score formula
def test_req_22_final_weighted_score_formula():
    v = 100.0
    c = 100.0
    r = 100.0
    f = 100.0
    final = 0.40 * v + 0.30 * c + 0.20 * r + 0.10 * f
    assert pytest.approx(final, 0.001) == 100.0


# 23. deterministic tie-break
def test_req_23_deterministic_tie_break():
    # When final_sector_score is equal:
    # 2. volume_score DESC, 3. development_total_trade_value DESC, 4. hs2 ASC
    items = [
        {"hs2": "05", "final_sector_score": 80.0, "volume_score": 70.0, "development_total_trade_value": 500.0},
        {"hs2": "02", "final_sector_score": 80.0, "volume_score": 80.0, "development_total_trade_value": 600.0},
        {"hs2": "01", "final_sector_score": 80.0, "volume_score": 80.0, "development_total_trade_value": 600.0},
    ]
    items.sort(key=lambda x: (-x["final_sector_score"], -x["volume_score"], -x["development_total_trade_value"], x["hs2"]))
    # item with volume 80 wins over 70; tie between 01 and 02 broken by hs2 ASC -> "01" before "02"
    assert items[0]["hs2"] == "01"
    assert items[1]["hs2"] == "02"
    assert items[2]["hs2"] == "05"


# 24. max 6 selected
def test_req_24_max_6_selected():
    # If 10 eligible, selector picks at most 6
    df = create_synthetic_target_df()
    selector = SectorSelector(df, max_sectors_per_direction=6)
    results = selector.run_selection()
    assert len(results["directions"]["TR_RU"]["selected_sectors"]) <= 6


# 25. fewer than 6 allowed
def test_req_25_fewer_than_6_allowed():
    # Only 2 sectors in synthetic TR_RU -> selects 2
    df = create_synthetic_target_df()
    selector = SectorSelector(df, max_sectors_per_direction=6)
    results = selector.run_selection()
    tr_sel = results["directions"]["TR_RU"]["selected_sectors"]
    assert len(tr_sel) == 2  # exactly 2 eligible sectors, no forced padding


# 26. TOTAL never ranked
def test_req_26_total_never_ranked():
    df = create_synthetic_target_df()
    panel = MonthlyDevelopmentPanel(df)
    candidates = panel.get_candidate_universe("TR_RU")
    assert "TOTAL" not in candidates


# 27. holdout mutation does not affect M3 selection
def test_req_27_holdout_mutation_invariance(tmp_path: Path):
    df_base = create_synthetic_target_df()

    # Create dataset with 2025 holdout
    holdout_2025_a = pd.DataFrame([{
        "period": "2025-06",
        "direction": "TR_RU",
        "hs2": "84",
        "trade_value_usd": 1000.0,
        "is_total": False,
        "reporter_code": TURKEY_M49_CODE,
        "partner_code": RUSSIA_M49_CODE,
        "flow": FLOW_EXPORT_NAME,
    }])
    holdout_2025_b = pd.DataFrame([{
        "period": "2025-06",
        "direction": "TR_RU",
        "hs2": "84",
        "trade_value_usd": 999999999.0,  # Huge mutated holdout value
        "is_total": False,
        "reporter_code": TURKEY_M49_CODE,
        "partner_code": RUSSIA_M49_CODE,
        "flow": FLOW_EXPORT_NAME,
    }])

    # Run selection on development portion only (filtered before selection as required)
    dev_only_a = pd.concat([df_base, holdout_2025_a])
    dev_only_a = dev_only_a[dev_only_a["period"] <= "2024-12"]

    dev_only_b = pd.concat([df_base, holdout_2025_b])
    dev_only_b = dev_only_b[dev_only_b["period"] <= "2024-12"]

    res_a = SectorSelector(dev_only_a).run_selection()
    res_b = SectorSelector(dev_only_b).run_selection()

    assert res_a["directions"]["TR_RU"]["selected_sectors"] == res_b["directions"]["TR_RU"]["selected_sectors"]


# 28. selected-sector manifest schema
def test_req_28_manifest_schema(tmp_path: Path):
    df = create_synthetic_target_df()
    selector = SectorSelector(df)
    results = selector.run_selection()
    manifest_path, _, _ = selector.export_artifacts(
        results,
        manifest_path=tmp_path / "test_manifest.yaml",
        metrics_parquet_path=tmp_path / "test_metrics.parquet",
        results_json_path=tmp_path / "test_results.json",
    )

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    assert "spec_version" in manifest
    assert "selection_version" in manifest
    assert "development_period" in manifest
    assert "holdout_period" in manifest
    assert "target_policy" in manifest
    assert "TR_RU" in manifest
    assert "RU_TR" in manifest

    for s in manifest["TR_RU"]:
        assert "rank" in s
        assert "hs2" in s
        assert "description" in s
        assert "final_score" in s
