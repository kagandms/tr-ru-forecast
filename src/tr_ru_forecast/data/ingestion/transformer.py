"""Data transformation and normalization for UN Comtrade records.

Converts raw Comtrade API records into the standardized trade_monthly schema
in strict compliance with DATA_SPEC.md v1.0.
"""

import re
from datetime import UTC, datetime
from typing import Any, NamedTuple

from tr_ru_forecast.core.constants import (
    CANONICAL_FLOWS,
    FLOW_EXPORT_NAME,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.ingestion.transformer")


class ConflictingDuplicateError(Exception):
    """Raised when conflicting duplicate records are found and fail-closed is enabled."""


class DeduplicationResult(NamedTuple):
    """Container for deduplication outcomes."""

    clean_records: list[dict[str, Any]]
    exact_duplicate_count: int
    conflict_count: int
    conflict_keys: list[str]
    quarantined_records: list[dict[str, Any]]


STANDARDIZED_COLUMNS: list[str] = [
    "period",
    "direction",
    "reporter_code",
    "reporter_name",
    "partner_code",
    "partner_name",
    "flow",
    "hs2",
    "hs_description",
    "trade_value_usd",
    "net_weight_kg",
    "quantity",
    "classification",
    "classification_code",
    "is_original_classification",
    "is_total",
    "source",
    "retrieved_at",
    "data_status",
]


def normalize_period(period_raw: Any) -> str | None:
    """Normalize raw period into standard YYYY-MM format.

    Handles:
        '202301' -> '2023-01'
        '2023-01' -> '2023-01'
        202301 -> '2023-01'

    Returns:
        Formatted period string 'YYYY-MM' or None if invalid.
    """
    if period_raw is None:
        return None

    period_str = str(period_raw).strip()
    if re.match(r"^\d{4}-\d{2}$", period_str):
        return period_str

    if re.match(r"^\d{6}$", period_str):
        return f"{period_str[:4]}-{period_str[4:6]}"

    logger.warning("Unrecognized period format: %r", period_raw)
    return None


def normalize_hs2(cmd_code_raw: Any) -> tuple[str | None, bool]:
    """Normalize commodity code to HS2 or TOTAL.

    Args:
        cmd_code_raw: Raw commodity code string or int.

    Returns:
        tuple (hs2_code, is_total):
            - ('TOTAL', True) for total trade aggregates
            - ('01'..'99', False) for standard HS2 chapters
            - (None, False) if code is neither TOTAL nor valid HS2
    """
    if cmd_code_raw is None:
        return None, False

    code_str = str(cmd_code_raw).strip().upper()
    if code_str in ("TOTAL", "TOTALS", "999999", "00"):
        return "TOTAL", True

    if code_str.isdigit():
        code_int = int(code_str)
        if 1 <= code_int <= 99 and code_int != 77:
            return f"{code_int:02d}", False

    return None, False


def transform_raw_record(
    raw_record: dict[str, Any],
    expected_direction: str | None = None,
    default_retrieved_at: str | None = None,
) -> dict[str, Any] | None:
    """Transform a single UN Comtrade raw record into the standardized trade_monthly schema.

    Args:
        raw_record: Raw dictionary from Comtrade 'data' array.
        expected_direction: Expected direction ('TR_RU' or 'RU_TR') for cross-check.
        default_retrieved_at: Fallback ISO timestamp.

    Returns:
        Standardized record dictionary or None if row is invalid/out-of-scope.
    """
    if not isinstance(raw_record, dict):
        logger.warning("Skipping non-dict raw record: %r", raw_record)
        return None

    # 1. Period
    period = normalize_period(raw_record.get("period"))
    if not period:
        ref_year = raw_record.get("refYear")
        ref_month = raw_record.get("refMonth")
        if ref_year and ref_month:
            period = f"{int(ref_year):04d}-{int(ref_month):02d}"
    if not period:
        logger.warning("Skipping record without valid period: %r", raw_record)
        return None

    # 2. HS2 & TOTAL identification
    cmd_code = raw_record.get("cmdCode")
    hs2, is_total = normalize_hs2(cmd_code)
    if hs2 is None:
        # Ignore HS4, HS6, or unparseable codes
        return None

    # 3. Country identification and direction determination
    reporter_code = raw_record.get("reporterCode")
    partner_code = raw_record.get("partnerCode")
    if reporter_code is None or partner_code is None:
        logger.warning("Skipping record missing reporter or partner code: %r", raw_record)
        return None

    try:
        reporter_code_int = int(reporter_code)
        partner_code_int = int(partner_code)
    except (ValueError, TypeError):
        logger.warning("Non-integer country code in record: %r", raw_record)
        return None

    # Direction resolution
    if reporter_code_int == 792 and partner_code_int == 643:
        direction = TradeDirection.TR_RU.value
    elif reporter_code_int == 643 and partner_code_int == 792:
        direction = TradeDirection.RU_TR.value
    else:
        logger.debug(
            "Record has non-target reporter/partner pair (%s, %s); skipping",
            reporter_code_int,
            partner_code_int,
        )
        return None

    if expected_direction and direction != expected_direction:
        logger.warning(
            "Direction mismatch: record evaluated to %s, expected %s",
            direction,
            expected_direction,
        )
        return None

    canonical_info = CANONICAL_FLOWS[TradeDirection(direction)]
    reporter_name = raw_record.get("reporterDesc") or canonical_info["reporter_name"]
    partner_name = raw_record.get("partnerDesc") or canonical_info["partner_name"]

    # 4. Values (primaryValue or fobvalue for exports)
    primary_val = raw_record.get("primaryValue")
    if primary_val is None:
        primary_val = raw_record.get("fobvalue")

    # Strict Missing != Zero policy
    if primary_val is None:
        trade_value_usd: float | None = None
        data_status = "missing"
    else:
        try:
            val_float = float(primary_val)
            if val_float == 0.0:
                trade_value_usd = 0.0
                data_status = "reported_zero"
            else:
                trade_value_usd = val_float
                data_status = "observed"
        except (ValueError, TypeError):
            logger.warning("Unparseable trade value: %r", primary_val)
            trade_value_usd = None
            data_status = "missing"

    # Weight and Quantity (nullable, never fabricated)
    net_weight_raw = raw_record.get("netWgt")
    try:
        net_weight_kg = float(net_weight_raw) if net_weight_raw is not None else None
    except (ValueError, TypeError):
        net_weight_kg = None

    qty_raw = raw_record.get("qty")
    try:
        quantity = float(qty_raw) if qty_raw is not None else None
    except (ValueError, TypeError):
        quantity = None

    # Classification / Revision metadata
    classification = str(raw_record.get("classificationSearchCode") or "HS")
    classification_code_raw = raw_record.get("classificationCode")
    classification_code = (
        str(classification_code_raw) if classification_code_raw is not None else None
    )
    is_orig_raw = raw_record.get("isOriginalClassification")
    is_original_classification = bool(is_orig_raw) if is_orig_raw is not None else None

    retrieved_at = default_retrieved_at or datetime.now(UTC).isoformat()
    hs_desc = raw_record.get("cmdDesc")

    return {
        "period": period,
        "direction": direction,
        "reporter_code": reporter_code_int,
        "reporter_name": str(reporter_name),
        "partner_code": partner_code_int,
        "partner_name": str(partner_name),
        "flow": FLOW_EXPORT_NAME,
        "hs2": hs2,
        "hs_description": str(hs_desc) if hs_desc is not None else None,
        "trade_value_usd": trade_value_usd,
        "net_weight_kg": net_weight_kg,
        "quantity": quantity,
        "classification": classification,
        "classification_code": classification_code,
        "is_original_classification": is_original_classification,
        "is_total": is_total,
        "source": "UN Comtrade",
        "retrieved_at": retrieved_at,
        "data_status": data_status,
    }


def deduplicate_records(
    records: list[dict[str, Any]],
    fail_closed: bool = False,
) -> DeduplicationResult:
    """Deduplicate records based on canonical uniqueness key.

    Uniqueness Key:
        (period, reporter_code, partner_code, flow, hs2, classification_code or classification)

    Policy (M1 Revision Gate):
        - Same canonical key + same payload/value: exact duplicate, collapsed deterministically to one record.
        - Same canonical key + conflicting economic values: DO NOT pick max(trade_value_usd) or any value automatically.
          Quarantine conflicting records so they are NOT included in the clean standardized canonical output.
        - Conflicting duplicate count and conflict keys are explicitly reported for QA tracking.
        - If fail_closed is True, raise ConflictingDuplicateError immediately upon conflict.

    Returns:
        DeduplicationResult containing clean records, exact duplicate count, conflict count,
        conflict keys, and quarantined records.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for rec in records:
        class_key = rec.get("classification_code") or rec.get("classification")
        key = (
            rec["period"],
            rec["reporter_code"],
            rec["partner_code"],
            rec["flow"],
            rec["hs2"],
            class_key,
        )
        groups.setdefault(key, []).append(rec)

    clean_records: list[dict[str, Any]] = []
    quarantined_records: list[dict[str, Any]] = []
    conflict_keys: list[str] = []
    exact_duplicate_count = 0
    conflict_count = 0

    for key, group in groups.items():
        key_str = ":".join(str(k) for k in key)
        if len(group) == 1:
            clean_records.append(group[0])
            continue

        # Check for economic value conflicts (trade_value_usd)
        distinct_values = {r.get("trade_value_usd") for r in group}
        if len(distinct_values) == 1:
            # Exact duplicate: collapse deterministically
            exact_duplicate_count += len(group) - 1
            clean_records.append(group[0])
            logger.debug(
                "Exact duplicate collapsed for key %s (occurrences=%d, value=%s)",
                key_str,
                len(group),
                group[0].get("trade_value_usd"),
            )
        else:
            # Conflicting duplicate: quarantine
            conflict_count += 1
            conflict_keys.append(key_str)
            quarantined_records.extend(group)
            logger.warning(
                "Conflicting duplicate detected for canonical key %s with distinct values %s. "
                "Quarantining %d records from standardized output.",
                key_str,
                distinct_values,
                len(group),
            )

    if fail_closed and conflict_count > 0:
        raise ConflictingDuplicateError(
            f"Ingestion fail-closed: {conflict_count} conflicting duplicate key(s) detected: {conflict_keys}"
        )

    return DeduplicationResult(
        clean_records=clean_records,
        exact_duplicate_count=exact_duplicate_count,
        conflict_count=conflict_count,
        conflict_keys=conflict_keys,
        quarantined_records=quarantined_records,
    )
