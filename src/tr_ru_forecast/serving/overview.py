"""TOTAL-only aggregates avoid counting sector components twice."""

from __future__ import annotations

from decimal import Decimal

from tr_ru_forecast.serving.indicator_schemas import OverviewResponse
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE
from tr_ru_forecast.serving.snapshot import FrozenSnapshot


def build_overview(snapshot: FrozenSnapshot) -> OverviewResponse:
    """Aggregate two aligned TOTAL observations. Args: snapshot. Returns: overview. Raises: ValueError."""
    exports = next(item for item in snapshot.forecasts if item.sector_id == "TR_RU:TOTAL")
    imports = next(item for item in snapshot.forecasts if item.sector_id == "RU_TR:TOTAL")
    common = min(exports.observation_cutoff, imports.observation_cutoff)
    export_usd = next(point.trade_value_usd for point in exports.history if point.period == common)
    import_usd = next(point.trade_value_usd for point in imports.history if point.period == common)
    total = balance = None
    if export_usd is not None and import_usd is not None:
        total = str(Decimal(export_usd) + Decimal(import_usd))
        balance = str(Decimal(export_usd) - Decimal(import_usd))
    return OverviewResponse(
        evaluation_scope=EVALUATION_SCOPE, observation_cutoff=common, summary=snapshot.kpis,
        tr_ru_total_usd=export_usd, ru_tr_total_usd=import_usd,
        bilateral_total_usd=total, turkey_trade_balance_usd=balance,
    )
