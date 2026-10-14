"""Constants, predictor whitelist, and hyperparameter specifications for M9 XGBoost.

Strictly defines:
- The exact 21 predictor whitelist.
- The 16-candidate hyperparameter search grid.
- Fixed hyperparameters for deterministic, reproducible training.
- Pre-validation tuning block origins (calendar year 2016 target period).
- Prequential conformal calibration parameters.
"""

from __future__ import annotations

from typing import Any, Final

M9_RUN_ID: Final[str] = "run_m9_xgboost_v1_1"
SPEC_VERSION: Final[str] = "1.1"
MODEL_VERSION: Final[str] = "1.1.0"
MODEL_TYPE: Final[str] = "xgboost_direct"

# Section 5: Exact 21 Predictor Whitelist
M9_PREDICTORS: Final[list[str]] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "mom_growth",
    "yoy_growth",
    "month",
    "quarter",
    "year",
    "usd_try_lagged",
    "rub_usd_lagged",
    "brent_lagged",
    "tr_ip_lagged",
    "ru_ip_lagged",
]

# Columns explicitly forbidden from entering models as predictors
FORBIDDEN_COLUMNS: Final[list[str]] = [
    "target",
    "target_h1",
    "target_h3",
    "target_h6",
    "period",
    "direction",
    "hs2",
    "hs_description",
    "is_total",
    "feature_snapshot_date",
]

FORECAST_HORIZONS: Final[list[int]] = [1, 3, 6]

# Section 11: Fixed Hyperparameters
FIXED_XGB_PARAMS: Final[dict[str, Any]] = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "tree_method": "hist",
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "random_state": 42,
    "n_jobs": 1,
}

# Section 11: Exact Controlled Search Space (16 Candidates)
GRID_MAX_DEPTH: Final[list[int]] = [2, 3]
GRID_MIN_CHILD_WEIGHT: Final[list[int]] = [1, 5]
GRID_LEARNING_RATE: Final[list[float]] = [0.03, 0.10]
GRID_N_ESTIMATORS: Final[list[int]] = [150, 300]

# Section 10: Frozen Pre-Validation Tuning Block Origins
# Targets in calendar year 2016 (2016-01 to 2016-12)
TUNING_ORIGINS: Final[dict[int, list[str]]] = {
    1: [
        "2015-12", "2016-01", "2016-02", "2016-03",
        "2016-04", "2016-05", "2016-06", "2016-07",
        "2016-08", "2016-09", "2016-10", "2016-11",
    ],
    3: [
        "2015-10", "2015-11", "2015-12", "2016-01",
        "2016-02", "2016-03", "2016-04", "2016-05",
        "2016-06", "2016-07", "2016-08", "2016-09",
    ],
    6: [
        "2015-07", "2015-08", "2015-09", "2015-10",
        "2015-11", "2015-12", "2016-01", "2016-02",
        "2016-03", "2016-04", "2016-05", "2016-06",
    ],
}

# Section 17 & 18: Conformal Uncertainty Parameters
MIN_CALIBRATION_SIZE: Final[int] = 19
CONFORMAL_LEVELS: Final[list[float]] = [0.80, 0.95]


def generate_candidate_grid() -> list[dict[str, Any]]:
    """Generate the deterministic 16-candidate hyperparameter grid.

    Ordered lexicographically by (max_depth, min_child_weight, learning_rate, n_estimators).
    Candidate IDs are fixed from cand_00 to cand_15 prior to evaluation.
    """
    candidates = []
    idx = 0
    for depth in sorted(GRID_MAX_DEPTH):
        for mcw in sorted(GRID_MIN_CHILD_WEIGHT):
            for lr in sorted(GRID_LEARNING_RATE):
                for n_est in sorted(GRID_N_ESTIMATORS):
                    candidate_id = f"cand_{idx:02d}"
                    candidates.append({
                        "candidate_id": candidate_id,
                        "max_depth": depth,
                        "min_child_weight": mcw,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                    })
                    idx += 1
    return candidates
