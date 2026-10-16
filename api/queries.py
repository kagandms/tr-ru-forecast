"""Validated HTTP query contracts, separate from domain filters."""

from __future__ import annotations

from pydantic import Field, model_validator

from tr_ru_forecast.serving.schemas import (
    ChampionFamily,
    Direction,
    FrozenModel,
    HoldoutMonth,
    Horizon,
    Month,
)


class TaskQuery(FrozenModel):
    """Immutable task query contract."""

    direction: Direction | None = None
    hs2: str | None = None
    horizon: Horizon | None = None
    champion_model: ChampionFamily | None = None


class ForecastQuery(FrozenModel):
    """Immutable forecast query contract."""

    horizon: Horizon | None = None


class ForecastSectorQuery(ForecastQuery):
    """Immutable forecast sector query contract."""

    sector_id: str


class ExplainQuery(FrozenModel):
    """Immutable explain query contract."""

    horizon: Horizon
    target_month: HoldoutMonth | None = None


class PageQuery(FrozenModel):
    """Immutable page query contract."""

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=1000, ge=1, le=10000)
    start_month: Month | None = None
    end_month: Month | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> PageQuery:
        """Reject reversed bounds. Args: self. Returns: query. Raises: ValueError."""
        if self.start_month and self.end_month and self.start_month > self.end_month:
            raise ValueError("start_month must not exceed end_month")
        return self


class TradeQuery(PageQuery):
    """Immutable trade query contract."""

    sector_id: str


class IndicatorQuery(PageQuery):
    """Immutable indicator query contract."""

    indicator_id: str | None = None


class MirrorQuery(PageQuery):
    """Immutable mirror query contract."""

    sector_id: str | None = None
    hs2: str | None = None
    direction: Direction | None = None
