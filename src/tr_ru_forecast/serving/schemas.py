"""Immutable shared API contracts with JSON-safe numeric diagnostics."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

Direction = Literal["TR_RU", "RU_TR"]
ModelFamily = Literal["sarima", "sarimax", "xgboost", "seasonal_naive"]
ChampionFamily = Literal["sarima", "xgboost"]
Month = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
HoldoutMonth = Annotated[str, Field(pattern=r"^2025-(0[1-9]|1[0-2])$")]
EVALUATION_SCOPE = "retrospective_latest_vintage_deterministic_lag"


def normalize_number(value: object) -> float | None:
    """Parse a diagnostic scalar. Args: value. Returns: finite float/null. Raises: ValueError."""
    if value is None:
        return None
    number = float(str(value))
    return number if math.isfinite(number) else None


def parse_horizon(value: object) -> int:
    """Normalize URL horizons. Args: value. Returns: 1/3/6. Raises: ValueError."""
    if type(value) is int and value in (1, 3, 6):
        return value
    if isinstance(value, str) and value in {"1", "3", "6"}:
        return int(value)
    raise ValueError("horizon must be 1, 3 or 6")


Horizon = Annotated[Literal[1, 3, 6], BeforeValidator(parse_horizon)]


Number = Annotated[float | None, BeforeValidator(normalize_number)]


class FrozenModel(BaseModel):
    """Immutable wire model; unexpected fields are contract violations."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorDetail(FrozenModel):
    """Immutable error detail contract."""

    code: str
    message: str


class ErrorResponse(FrozenModel):
    """Immutable error response contract."""

    error: ErrorDetail


class MetricSummary(FrozenModel):
    """Immutable metric summary contract."""

    mean_smape: Number
    mean_mae: Number
    mean_rmse: Number
    valid_origins: int
    expected_origins: int
    skill_vs_naive_pct: Number = None


class IntervalDiagnostics(FrozenModel):
    """Immutable interval diagnostics contract."""

    empirical_coverage_80: Number
    empirical_coverage_95: Number
    interval_origins: int
    mean_width_80: Number
    mean_width_95: Number
    calibration_pool_size: int | None = None
    coverage_unit: Literal["fraction"] = "fraction"
