"""Transformation and deduplication for mirror trade data.

Processes UN Comtrade import flows reported by counterparty partners,
mapping them strictly to the corresponding physical trade flow direction
in accordance with DATA_SPEC.md and PROJECT_SPEC.md.
"""

from datetime import UTC, datetime
from typing import Any

from tr_ru_forecast.core.constants import (
    FLOW_IMPORT_NAME,
    MIRROR_FLOWS,
    RUSSIA_M49_CODE,
    TURKEY_M49_CODE,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.ingestion.transformer import (
    ConflictingDuplicateError,
    DeduplicationResult,
    normalize_hs2,
    normalize_period,
)

logger = get_logger("tr_ru_forecast.data.quality.mirror_transformer")


def transform_mirror_raw_record(
    raw_record: dict[str, Any],
    expected_direction: str | None = None,
    default_retrieved_at: str | None = None,
) -> dict[str, Any] | None:
    """Transform a single UN Comtrade raw import record into standardized mirror schema.

    Direction Mapping (Physical Trade Flow):
        - Reporter=643 (RU), Partner=792 (TR), Flow='M' -> Direction 'TR_RU' (Russia importing from Turkey)
        - Reporter=792 (TR), Partner=643 (RU), Flow='M' -> Direction 'RU_TR' (Turkey importing from Russia)

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

    # 1. Period normalization
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
        return None

    # 3. Country identification and physical direction determination
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

    # Direction resolution: Mirror data preserves the physical trade flow direction
    if reporter_code_int == RUSSIA_M49_CODE and partner_code_int == TURKEY_M49_CODE:
        direction = TradeDirection.TR_RU.value
    elif reporter_code_int == TURKEY_M49_CODE and partner_code_int == RUSSIA_M49_CODE:
        direction = TradeDirection.RU_TR.value
    else:
        logger.debug(
            "Mirror record has non-target reporter/partner pair (%s, %s); skipping",
            reporter_code_int,
            partner_code_int,
        )
        return None

    if expected_direction and direction != expected_direction:
        logger.warning(
            "Mirror direction mismatch: record evaluated to %s, expected %s",
            direction,
            expected_direction,
        )
        return None

    mirror_info = MIRROR_FLOWS[TradeDirection(direction)]
    reporter_name = raw_record.get("reporterDesc") or mirror_info["reporter_name"]
    partner_name = raw_record.get("partnerDesc") or mirror_info["partner_name"]

    # 4. Values (primaryValue, cifvalue, or fobvalue for imports)
    primary_val = raw_record.get("primaryValue")
    if primary_val is None:
        primary_val = raw_record.get("cifvalue")
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
            logger.warning("Unparseable mirror trade value: %r", primary_val)
            trade_value_usd = None
            data_status = "missing"

    # Weight and Quantity
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
        "flow": FLOW_IMPORT_NAME,
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


def deduplicate_mirror_records(
    records: list[dict[str, Any]],
    fail_closed: bool = False,
) -> DeduplicationResult:
    """Deduplicate mirror records using canonical uniqueness key and conflict quarantine.

    Uniqueness Key:
        (period, reporter_code, partner_code, flow, hs2, classification_code or classification)

    Policy:
        - Same canonical key + same payload/value: exact duplicate, collapsed deterministically.
        - Same canonical key + conflicting economic values: quarantined from standardized output.
        - If fail_closed is True, raise ConflictingDuplicateError immediately upon conflict.
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

        distinct_values = {r.get("trade_value_usd") for r in group}
        if len(distinct_values) == 1:
            exact_duplicate_count += len(group) - 1
            clean_records.append(group[0])
            logger.debug(
                "Mirror exact duplicate collapsed for key %s (occurrences=%d, value=%s)",
                key_str,
                len(group),
                group[0].get("trade_value_usd"),
            )
        else:
            conflict_count += 1
            conflict_keys.append(key_str)
            quarantined_records.extend(group)
            logger.warning(
                "Mirror conflicting duplicate detected for key %s with distinct values %s. "
                "Quarantining %d records from standardized output.",
                key_str,
                distinct_values,
                len(group),
            )

    if fail_closed and conflict_count > 0:
        raise ConflictingDuplicateError(
            f"Mirror ingestion fail-closed: {conflict_count} conflicting duplicate key(s) detected: {conflict_keys}"
        )

    return DeduplicationResult(
        clean_records=clean_records,
        exact_duplicate_count=exact_duplicate_count,
        conflict_count=conflict_count,
        conflict_keys=conflict_keys,
        quarantined_records=quarantined_records,
    )
