"""Query contracts accept valid URL strings without weakening horizon validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.queries import ForecastQuery


@pytest.mark.parametrize("horizon", ["1", "3", "6"])
def test_accepts_supported_horizon_query(horizon: str) -> None:
    """Valid URL strings are normalized to the corresponding integer literal."""
    query = ForecastQuery.model_validate({"horizon": horizon})

    assert query.horizon == int(horizon)


@pytest.mark.parametrize("horizon", ["0", "2", "12", "1.0", "bad", True])
def test_rejects_unsupported_horizon(horizon: object) -> None:
    """Unsupported inputs cannot bypass the literal domain."""
    with pytest.raises(ValidationError):
        ForecastQuery.model_validate({"horizon": horizon})
