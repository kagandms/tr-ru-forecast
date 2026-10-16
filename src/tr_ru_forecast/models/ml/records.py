"""Typed serialization values and immutable forecast identities for ML workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

Scalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = Scalar | list["JsonValue"] | tuple["JsonValue", ...] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class ResidualRecord(TypedDict):
    """Chronological out-of-sample residual evidence."""

    forecast_origin: str
    forecast_period: str
    actual: float
    prediction: float
    absolute_residual: float


@dataclass(frozen=True)
class ForecastIdentity:
    """Model-independent forecast calendar and series identity."""

    direction: str
    hs2: str
    horizon: int
    forecast_origin: str
    forecast_period: str
