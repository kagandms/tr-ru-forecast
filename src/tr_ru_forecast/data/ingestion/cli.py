"""Command-line interface for UN Comtrade trade data ingestion."""

import argparse
import json
import sys

from tr_ru_forecast.config import get_settings
from tr_ru_forecast.core.constants import TradeDirection
from tr_ru_forecast.core.logging import get_logger, setup_logging
from tr_ru_forecast.data.ingestion.pipeline import (
    TradeIngestionPipeline,
    resolve_latest_complete_month,
)

logger = get_logger("tr_ru_forecast.data.ingestion.cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the trade ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest canonical monthly bilateral trade data from UN Comtrade (M1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["TR_RU", "RU_TR", "both"],
        default="both",
        help="Bilateral trade direction to ingest.",
    )
    parser.add_argument(
        "--start-period",
        type=str,
        default=None,
        help="Starting month in YYYY-MM or YYYYMM format (default from config: 2012-01).",
    )
    parser.add_argument(
        "--end-period",
        type=str,
        default=None,
        help="Ending month in YYYY-MM or YYYYMM format (default: latest complete month).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use existing local raw snapshots when available before querying API.",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        default=False,
        help="Immediately abort execution if conflicting duplicate records are found.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def main(args_list: list[str] | None = None) -> int:
    """CLI entrypoint function."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(level=args.log_level)
    settings = get_settings()

    start_period = args.start_period or settings.start_period
    end_period = args.end_period or resolve_latest_complete_month()

    logger.info(
        "Initiating trade ingestion: direction=%s, range=%s to %s",
        args.direction,
        start_period,
        end_period,
    )

    pipeline = TradeIngestionPipeline(settings=settings)

    try:
        if args.direction == "both":
            summaries = pipeline.run_all(
                start_period=start_period,
                end_period=end_period,
                use_cache=args.use_cache,
                fail_closed=args.fail_closed,
            )
            print("\n=== Ingestion QA Summaries ===")
            print(json.dumps(summaries, indent=2))
        else:
            direction = TradeDirection(args.direction)
            _, summary = pipeline.run_ingestion_for_direction(
                direction=direction,
                start_period=start_period,
                end_period=end_period,
                use_cache=args.use_cache,
                fail_closed=args.fail_closed,
            )
            print(f"\n=== Ingestion QA Summary ({direction.value}) ===")
            print(json.dumps(summary, indent=2))

        logger.info("Trade ingestion process completed successfully.")
        return 0

    except Exception as err:
        logger.error("Trade ingestion failed: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
