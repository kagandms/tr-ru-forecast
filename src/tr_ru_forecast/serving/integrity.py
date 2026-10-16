"""Snapshot-level serving contracts independent of model execution."""

from __future__ import annotations

import pandas as pd

from tr_ru_forecast.serving.artifacts import ArtifactError, ArtifactTables


def validate_table_contracts(tables: ArtifactTables) -> None:
    """Reject incomplete/contradictory projections. Args: tables. Returns: none. Raises: ArtifactError."""
    keys = ["sector_id", "horizon"]
    matrix = tables.matrix.sort_values(keys).reset_index(drop=True)
    holdout = tables.holdout.sort_values(keys).reset_index(drop=True)
    if matrix.duplicated(keys).any() or holdout.duplicated(keys).any():
        raise ArtifactError("Duplicate task keys")
    for column in [*keys, "champion_model", "fallback_model"]:
        if not matrix[column].equals(holdout[column]):
            raise ArtifactError("Frozen registry matrices disagree")
    predictions = tables.predictions
    if predictions.duplicated([*keys, "forecast_period", "role"]).any():
        raise ArtifactError("Duplicate prediction role")
    counts = predictions.groupby([*keys, "role"]).forecast_period.nunique()
    if len(counts) != 126 or not counts.eq(12).all():
        raise ArtifactError("Expected 12 targets per task and role")
    if not set(predictions.role) == {"champion", "fallback", "baseline"}:
        raise ArtifactError("Unexpected prediction roles")
    if set(predictions.forecast_period) != set(pd.period_range("2025-01", "2025-12", freq="M").astype(str)):
        raise ArtifactError("Unexpected holdout targets")
