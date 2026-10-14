"""Pre-validation time-aware hyperparameter tuner for M9 XGBoost.

Implements Section 10, 11, 12, 13:
- Pre-official-validation tuning block strictly in calendar year 2016.
- 16-candidate controlled Cartesian grid.
- Label embargo applied strictly inside every tuning origin fit.
- No early stopping.
- Selection by arithmetic mean of 14 series sMAPEs, then median sMAPE, then candidate_id.
- One selected parameter set per horizon, shared across all 14 series.
- Full persistence of all 48 candidate results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.metrics import calculate_smape
from tr_ru_forecast.models.ml.constants import (
    FORECAST_HORIZONS,
    TUNING_ORIGINS,
    generate_candidate_grid,
)
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
)

logger = get_logger(__name__)


def evaluate_tuning_candidate(
    dev_features_df: pd.DataFrame,
    series_list: list[dict[str, Any]],
    horizon: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a single candidate parameter set across all 14 series on the 12 pre-validation tuning origins.

    Returns dict with candidate metrics, fit counts, and series-level sMAPEs.
    """
    candidate_id = candidate["candidate_id"]
    origins = TUNING_ORIGINS[horizon]

    series_smapes: list[float] = []
    total_fits = 0
    failed_fits = 0
    failure_messages: list[str] = []

    for series_info in series_list:
        dir_name = series_info["direction"]
        hs2_code = series_info["hs2"]

        series_df = dev_features_df[
            (dev_features_df["direction"] == dir_name)
            & (dev_features_df["hs2"] == hs2_code)
        ].copy()

        series_actuals: list[float] = []
        series_preds: list[float] = []

        for orig in origins:
            total_fits += 1
            try:
                # 1. Enforce strict label embargo
                train_sub = get_training_subset(series_df, orig, horizon)
                if train_sub.empty:
                    raise ValueError(f"No eligible training rows for series ({dir_name}, {hs2_code}) at origin {orig}")

                X_train, y_train = prepare_training_xy(train_sub, horizon)
                X_origin = extract_origin_features(series_df, orig)

                # 2. Build and fit model
                reg = build_xgb_regressor(candidate)
                reg.fit(X_train, y_train)

                # 3. Predict origin row
                y_pred = float(reg.predict(X_origin)[0])

                # 4. Extract actual target label
                label_col = f"target_h{horizon}"
                actual_val = float(series_df.loc[series_df["period"] == orig, label_col].iloc[0])

                series_actuals.append(actual_val)
                series_preds.append(y_pred)
            except Exception as e:
                failed_fits += 1
                failure_messages.append(f"{dir_name}-{hs2_code}-{orig}: {e}")
                logger.warning(
                    "Tuning fit failed for candidate %s, series (%s, %s), origin %s, h=%d: %s",
                    candidate_id, dir_name, hs2_code, orig, horizon, e,
                )

        if len(series_actuals) == len(origins) and failed_fits == 0:
            s_smape = calculate_smape(np.array(series_actuals), np.array(series_preds))
            series_smapes.append(s_smape)
        else:
            series_smapes.append(np.nan)

    if failed_fits > 0 or len(series_smapes) != len(series_list) or any(np.isnan(s) for s in series_smapes):
        status = "INVALID"
        mean_smape = np.nan
        median_smape = np.nan
    else:
        status = "VALID"
        mean_smape = float(np.mean(series_smapes))
        median_smape = float(np.median(series_smapes))

    result = {
        "horizon": int(horizon),
        "candidate_id": str(candidate_id),
        "max_depth": int(candidate["max_depth"]),
        "min_child_weight": int(candidate["min_child_weight"]),
        "learning_rate": float(candidate["learning_rate"]),
        "n_estimators": int(candidate["n_estimators"]),
        "status": status,
        "mean_series_smape": mean_smape,
        "median_series_smape": median_smape,
        "fit_count": total_fits - failed_fits,
        "failure_count": failed_fits,
        "failure_details": "; ".join(failure_messages[:5]) if failure_messages else "",
    }
    return result


def run_pre_validation_tuning(
    dev_features_df: pd.DataFrame,
    series_list: list[dict[str, Any]],
    horizons: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Execute pre-validation time-aware tuning grid search across all horizons.

    Returns:
        tuning_results_df: DataFrame of 48 candidate evaluation results (3 horizons x 16 candidates).
        selected_params: Dict mapping horizon -> winning candidate dict with candidate_id and params.
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    candidates = generate_candidate_grid()
    records: list[dict[str, Any]] = []
    selected_by_horizon: dict[int, dict[str, Any]] = {}

    for h in horizons:
        logger.info("Starting M9 pre-validation tuning for horizon h=%d (16 candidates)", h)
        h_records: list[dict[str, Any]] = []

        for cand in candidates:
            rec = evaluate_tuning_candidate(dev_features_df, series_list, h, cand)
            h_records.append(rec)

        # Convert to DataFrame for deterministic ranking
        h_df = pd.DataFrame(h_records)

        # Filter valid candidates
        valid_df = h_df[h_df["status"] == "VALID"].copy()
        if valid_df.empty:
            raise RuntimeError(f"CRITICAL: All 16 tuning candidates failed for horizon h={h}!")

        # Sort according to Section 12:
        # 1. lowest mean_series_smape
        # 2. lowest median_series_smape
        # 3. deterministic candidate_id ascending
        sorted_valid = valid_df.sort_values(
            by=["mean_series_smape", "median_series_smape", "candidate_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        winning_id = sorted_valid.iloc[0]["candidate_id"]
        logger.info(
            "Selected winning candidate for h=%d: %s (mean sMAPE: %.3f%%, median sMAPE: %.3f%%)",
            h, winning_id, sorted_valid.iloc[0]["mean_series_smape"], sorted_valid.iloc[0]["median_series_smape"],
        )

        # Mark selected column
        for rec in h_records:
            rec["selected"] = bool(rec["candidate_id"] == winning_id)
            records.append(rec)

        winning_candidate_dict = next(c for c in candidates if c["candidate_id"] == winning_id)
        selected_by_horizon[h] = winning_candidate_dict

    tuning_df = pd.DataFrame(records)
    # Sort final table systematically
    tuning_df = tuning_df.sort_values(by=["horizon", "candidate_id"]).reset_index(drop=True)

    return tuning_df, selected_by_horizon
