"""Deterministic SARIMA order selection on the initial 60-month training window.

Evaluates a controlled 64-candidate grid (p,d,q,P,D,Q in {0,1}, s=12) strictly on
observations from 2012-01 to 2016-12 for each of the 14 frozen target series.
Enforces the anti-leakage policy, deterministic AIC/BIC/complexity tie-breaking,
and full candidate attempt auditing.
"""

import itertools
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from tr_ru_forecast.config.settings import get_settings


def generate_candidate_grid() -> list[tuple[int, int, int, int, int, int]]:
    """Generate the controlled 64-candidate grid: p,d,q,P,D,Q in {0, 1}."""
    return list(itertools.product([0, 1], repeat=6))


def determine_trend(d: int, D: int) -> str:
    """Determine deterministic trend specification.

    If d == 0 and D == 0:
        trend = 'c' (constant/intercept)
    Else:
        trend = 'n' (no constant/intercept)
    """
    if d == 0 and D == 0:
        return "c"
    return "n"


def evaluate_candidate(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    maxiter: int = 200,
) -> dict[str, Any]:
    """Fit a single SARIMA candidate and extract diagnostic and selection metrics."""
    record: dict[str, Any] = {
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "trend": trend,
        "optimizer": "lbfgs",
        "converged": False,
        "aic": np.nan,
        "bic": np.nan,
        "nobs": len(y),
        "warning_count": 0,
        "failure_type": "NONE",
        "failure_message": "",
        "valid": False,
    }

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            res = mod.fit(disp=False, method="lbfgs", maxiter=maxiter)

            # Record warnings
            record["warning_count"] = len(captured_warnings)

            # Check convergence
            converged = False
            if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                converged = bool(res.mle_retvals.get("converged", False))
            record["converged"] = converged

            aic = float(res.aic) if res.aic is not None else np.nan
            bic = float(res.bic) if res.bic is not None else np.nan

            llf = float(res.llf) if res.llf is not None else np.nan
            if not np.isfinite(llf) or llf == 0.0:
                record["failure_type"] = "DEGENERATE_LIKELIHOOD"
                record["failure_message"] = f"Degenerate likelihood llf={llf}"
            elif np.isfinite(aic) and np.isfinite(bic):
                # Verify forecast uncertainty is computable
                try:
                    fc = res.get_forecast(steps=1)
                    ci = fc.conf_int(alpha=0.20)
                    if np.any(np.isnan(ci)) or np.any(np.isinf(ci)):
                        record["failure_type"] = "INVALID_FORECAST_VARIANCE"
                        record["failure_message"] = "Forecast confidence interval contains NaN/inf"
                    else:
                        record["aic"] = aic
                        record["bic"] = bic
                        record["valid"] = True
                except Exception as fc_err:
                    record["failure_type"] = "FORECAST_EXTRACTION_FAILURE"
                    record["failure_message"] = str(fc_err)
            else:
                record["failure_type"] = "NON_FINITE_METRICS"
                record["failure_message"] = f"Non-finite metrics: aic={aic}, bic={bic}"

        except Exception as exc:
            record["warning_count"] = len(captured_warnings)
            record["failure_type"] = type(exc).__name__
            record["failure_message"] = str(exc)

    return record


def select_order_for_series(
    y: np.ndarray,
    direction: str,
    hs2: str,
    grid: list[tuple[int, int, int, int, int, int]],
    float_tol: float = 1e-8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all 64 candidates for a single series and select the winning order."""
    candidate_records: list[dict[str, Any]] = []
    valid_candidates: list[tuple[float, float, int, tuple[int, ...], str, dict[str, Any]]] = []

    for p, d, q, P, D, Q in grid:
        order = (p, d, q)
        seasonal_order = (P, D, Q, 12)
        trend = determine_trend(d, D)

        eval_res = evaluate_candidate(y, order, seasonal_order, trend)
        eval_res["direction"] = direction
        eval_res["hs2"] = hs2
        candidate_records.append(eval_res)

        if eval_res["valid"]:
            complexity = p + d + q + P + D + Q
            # Key: (rounded aic, rounded bic, complexity, lexicographic tuple)
            key = (
                round(eval_res["aic"] / float_tol) * float_tol,
                round(eval_res["bic"] / float_tol) * float_tol,
                complexity,
                (p, d, q, P, D, Q),
            )
            valid_candidates.append((*key, trend, eval_res))

    valid_count = len(valid_candidates)
    failed_count = len(grid) - valid_count

    if valid_count == 0:
        selection_record = {
            "direction": direction,
            "hs2": hs2,
            "is_total": (hs2 == "TOTAL"),
            "order": None,
            "seasonal_order": None,
            "trend": None,
            "selected_aic": None,
            "selected_bic": None,
            "valid_candidate_count": 0,
            "failed_candidate_count": failed_count,
            "status": "ORDER_SELECTION_FAILED",
        }
        return selection_record, candidate_records

    # Sort according to deterministic tie-break sequence:
    # 1. AIC
    # 2. BIC
    # 3. Structural complexity
    # 4. Lexicographic order tuple
    valid_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    winner = valid_candidates[0]
    win_aic, win_bic, win_comp, win_order_tup, win_trend, win_eval = winner

    p, d, q, P, D, Q = win_order_tup
    selection_record = {
        "direction": direction,
        "hs2": hs2,
        "is_total": (hs2 == "TOTAL"),
        "order": [p, d, q],
        "seasonal_order": [P, D, Q, 12],
        "trend": win_trend,
        "selected_aic": float(win_eval["aic"]),
        "selected_bic": float(win_eval["bic"]),
        "valid_candidate_count": valid_count,
        "failed_candidate_count": failed_count,
        "status": "SELECTED",
    }

    return selection_record, candidate_records


def run_order_selection(
    feature_df: pd.DataFrame,
    sectors_manifest_path: Path | None = None,
    output_yaml_path: Path | None = None,
    output_parquet_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Execute SARIMA order selection across the 14 frozen modeling series on 2012-01 to 2016-12."""
    settings = get_settings()
    if sectors_manifest_path is None:
        sectors_manifest_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    if output_yaml_path is None:
        output_yaml_path = settings.project_root / "configs" / "sarima_orders_v1_1.yaml"
    if output_parquet_path is None:
        output_parquet_path = settings.processed_data_dir / "m7_sarima_order_search.parquet"

    with open(sectors_manifest_path) as f:
        sectors_meta = yaml.safe_load(f)

    # 14 series definitions: 6 HS2 + TOTAL per direction
    series_list: list[tuple[str, str]] = []
    # TR_RU
    for s in sectors_meta.get("TR_RU", []):
        series_list.append(("TR_RU", str(s["hs2"]).zfill(2)))
    series_list.append(("TR_RU", "TOTAL"))
    # RU_TR
    for s in sectors_meta.get("RU_TR", []):
        series_list.append(("RU_TR", str(s["hs2"]).zfill(2)))
    series_list.append(("RU_TR", "TOTAL"))

    grid = generate_candidate_grid()
    all_candidate_rows: list[dict[str, Any]] = []
    selected_series_orders: list[dict[str, Any]] = []

    # Strict pre-validation training window
    train_window_start = "2012-01"
    train_window_end = "2016-12"

    for direction, hs2 in series_list:
        sub = feature_df[
            (feature_df["direction"] == direction)
            & (feature_df["hs2"] == hs2)
            & (feature_df["period"] >= train_window_start)
            & (feature_df["period"] <= train_window_end)
        ].sort_values("period")

        if len(sub) != 60:
            raise ValueError(
                f"Expected exactly 60 initial training observations for ({direction}, {hs2}), "
                f"found {len(sub)}"
            )

        y = sub["target"].values.astype(float)
        sel_rec, cand_recs = select_order_for_series(y, direction, hs2, grid)
        selected_series_orders.append(sel_rec)
        all_candidate_rows.extend(cand_recs)

    # Format search audit dataframe
    audit_cols = [
        "direction",
        "hs2",
        "order",
        "seasonal_order",
        "trend",
        "optimizer",
        "converged",
        "aic",
        "bic",
        "nobs",
        "warning_count",
        "failure_type",
        "failure_message",
    ]
    audit_df = pd.DataFrame(all_candidate_rows)[audit_cols]

    # Save audit parquet
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(output_parquet_path, index=False)

    # Format frozen order manifest
    now_iso = datetime.now(UTC).isoformat()
    manifest_data: dict[str, Any] = {
        "spec_version": "1.1",
        "sarima_version": "1.1.0",
        "created_at": now_iso,
        "order_selection_window": {
            "start": train_window_start,
            "end": train_window_end,
            "observations_per_series": 60,
        },
        "seasonal_period": 12,
        "search_grid": {
            "p": [0, 1],
            "d": [0, 1],
            "q": [0, 1],
            "P": [0, 1],
            "D": [0, 1],
            "Q": [0, 1],
            "s": 12,
            "max_candidates_per_series": len(grid),
            "total_candidate_attempts": len(series_list) * len(grid),
        },
        "selection_criterion": {
            "primary": "lowest finite aic",
            "tie_break_1": "lowest finite bic",
            "tie_break_2": "lowest structural complexity (p+d+q+P+D+Q)",
            "tie_break_3": "lexicographic tuple (p,d,q,P,D,Q)",
            "float_tolerance": 1e-8,
        },
        "trend_policy": {
            "rule": "trend='c' if d==0 and D==0 else 'n'",
        },
        "optimizer_policy": {
            "primary_optimizer": "lbfgs",
            "primary_maxiter": 200,
            "retry_optimizer": "powell",
            "retry_maxiter": 500,
        },
        "series_orders": selected_series_orders,
    }

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    return manifest_data, audit_df
