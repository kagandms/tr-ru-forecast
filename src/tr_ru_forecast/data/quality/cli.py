"""Command-line interface for bilateral mirror statistics and data quality layer."""

import argparse
import json
import sys

from tr_ru_forecast.config import get_settings
from tr_ru_forecast.core.constants import TradeDirection
from tr_ru_forecast.core.logging import get_logger, setup_logging
from tr_ru_forecast.data.ingestion.pipeline import resolve_latest_complete_month
from tr_ru_forecast.data.quality.coverage_analyzer import BilateralCoverageAnalyzer
from tr_ru_forecast.data.quality.mirror_pipeline import MirrorQualityPipeline

logger = get_logger("tr_ru_forecast.data.quality.cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the mirror data quality CLI."""
    parser = argparse.ArgumentParser(
        description="Run bilateral mirror statistics and data quality layer (M2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["TR_RU", "RU_TR", "both"],
        default="both",
        help="Bilateral trade direction to analyze.",
    )
    parser.add_argument(
        "--start-period",
        type=str,
        default=None,
        help="Starting month in YYYY-MM or YYYYMM format (default: config start_period).",
    )
    parser.add_argument(
        "--end-period",
        type=str,
        default=None,
        help="Ending month in YYYY-MM or YYYYMM format (default: latest complete month).",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default=None,
        help="Explicit comma-separated periods to analyze (e.g. 2021-01,2023-01).",
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
    """CLI entrypoint function for mirror statistics."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(level=args.log_level)
    settings = get_settings()

    periods_list = (
        [p.strip() for p in args.periods.split(",") if p.strip()] if args.periods else None
    )
    start_period = args.start_period or settings.start_period
    end_period = args.end_period or resolve_latest_complete_month()

    logger.info(
        "Initiating mirror quality analysis: direction=%s, range=%s to %s, periods=%s",
        args.direction,
        start_period,
        end_period,
        periods_list,
    )

    pipeline = MirrorQualityPipeline(settings=settings, fail_closed=args.fail_closed)

    try:
        if args.direction == "both":
            _, summary = pipeline.run_all(
                start_period=start_period,
                end_period=end_period,
                use_cache=args.use_cache,
                fail_closed=args.fail_closed,
                periods=periods_list,
            )
            print("\n========================================================")
            print("         M2 — BILATERAL MIRROR QUALITY AUDIT SUMMARY      ")
            print("========================================================")
            print(json.dumps(summary, indent=2))
        else:
            direction = TradeDirection(args.direction)
            paired_df, qa_info = pipeline.pair_and_evaluate_direction(
                direction=direction,
                start_period=start_period,
                end_period=end_period,
                use_cache=args.use_cache,
                fail_closed=args.fail_closed,
                periods=periods_list,
            )
            analyzer = BilateralCoverageAnalyzer(
                paired_df=paired_df,
                full_history_verified=bool(pipeline.client.is_authenticated),
                expected_period_count=len(periods_list) if periods_list else None,
            )
            dir_report = analyzer.generate_full_report()
            output = {
                "direction": direction.value,
                "verification_scope": "FULL" if pipeline.client.is_authenticated else "PARTIAL",
                "full_history_verified": bool(pipeline.client.is_authenticated),
                "mirror_ingestion_qa": qa_info,
                "quality_audit": dir_report,
            }
            print(f"\n=== M2 Quality Summary: {direction.value} ===")
            print(json.dumps(output, indent=2))

        return 0
    except Exception as err:
        logger.exception("Mirror pipeline failed: %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
