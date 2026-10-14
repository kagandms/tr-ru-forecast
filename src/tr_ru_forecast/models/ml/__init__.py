"""Machine learning direct forecasting models module (XGBoost).

Exports:
- M9_PREDICTORS
- run_m9_xgboost_pipeline
- run_pre_validation_tuning
- PrequentialResidualCalibrator
- explain_fold_prediction
- compute_global_shap_importance
"""

from __future__ import annotations

from tr_ru_forecast.models.ml.conformal import PrequentialResidualCalibrator
from tr_ru_forecast.models.ml.constants import (
    FIXED_XGB_PARAMS,
    FORECAST_HORIZONS,
    M9_PREDICTORS,
    M9_RUN_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    SPEC_VERSION,
    generate_candidate_grid,
)
from tr_ru_forecast.models.ml.pipeline import run_m9_xgboost_pipeline
from tr_ru_forecast.models.ml.shap_explainer import (
    compute_global_shap_importance,
    explain_fold_prediction,
)
from tr_ru_forecast.models.ml.tuner import run_pre_validation_tuning
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
    validate_predictor_features,
)

__all__ = [
    "FIXED_XGB_PARAMS",
    "FORECAST_HORIZONS",
    "M9_PREDICTORS",
    "M9_RUN_ID",
    "MODEL_TYPE",
    "MODEL_VERSION",
    "PrequentialResidualCalibrator",
    "SPEC_VERSION",
    "build_xgb_regressor",
    "compute_global_shap_importance",
    "explain_fold_prediction",
    "extract_origin_features",
    "generate_candidate_grid",
    "get_training_subset",
    "prepare_training_xy",
    "run_m9_xgboost_pipeline",
    "run_pre_validation_tuning",
    "validate_predictor_features",
]
