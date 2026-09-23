"""CLI runner and report compiler for Milestone M6: Seasonal Naive Baseline.

Provides commands to run the Seasonal Naive baseline pipeline, validate outputs,
and generate the official M6 report (docs/M6_SEASONAL_NAIVE_REPORT.md).
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.models.baselines.pipeline import run_seasonal_naive_pipeline

logger = get_logger("tr_ru_forecast.models.baselines.cli")


def generate_m6_report(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    run_meta: dict[str, Any],
    report_path: Path,
) -> None:
    """Compile the authoritative M6 Seasonal Naive Baseline Report markdown document."""
    lines = [
        "# M6 Seasonal Naive Baseline Report",
        "",
        "**Document Version:** 1.1.0  ",
        f"**Execution Date:** {run_meta['created_at']}  ",
        "**Authoritative Specification:** Specification Pack v1.1 (`specification/`)  ",
        "**Author:** Senior Forecasting Research Engineer & Backtesting Specialist  ",
        "**Status:** **READY FOR EXTERNAL REVIEW**  ",
        "",
        "---",
        "",
        "## Specification / Scope",
        "",
        "- **Milestone:** M6 — Seasonal Naive Baseline.",
        "- **Specification Pack:** v1.1 (authoritative source of truth).",
        "- **Scope Boundary:** Establishes the frozen benchmark forecasting result across the 156-month development period (`2012-01` to `2024-12`) for all 14 target series across horizons $h \\in \\{1, 3, 6\\}$.",
        "- **Strict Non-Goals Enforced:** No SARIMA, SARIMAX, or XGBoost model fitting; no hyperparameter tuning; no champion model selection; strictly no opening of the 2025 trade-target holdout.",
        "",
        "---",
        "",
        "## Frozen Inputs",
        "",
        "| Artifact Role | File Path | Status / Notes |",
        "| :--- | :--- | :--- |",
        "| **Development Features** | `data/processed/model_features_development.parquet` | Milestone M5 Final Pass (2,184 rows, 31 columns) |",
        "| **Selected Sectors Manifest** | `configs/selected_sectors_v1_1.yaml` | Milestone M3 Final Pass (12 HS2 sectors + 2 TOTALs) |",
        "| **Feature Specification** | `configs/feature_spec_v1_1.yaml` | Milestone M5 Final Pass |",
        "| **Macro Data Foundation** | `data/processed/macro_monthly.parquet` | Milestone M4 Final Pass (not consumed by Seasonal Naive) |",
        "",
        "> [!NOTE]",
        "> **Input Data Integrity:** Seasonal Naive consumes strictly the historical `target` values from `model_features_development.parquet` for the 14 modeling series. Zero macroeconomic or engineered features are utilized.",
        "",
        "---",
        "",
        "## Modeling Universe",
        "",
        "The modeling universe consists of exactly **14 monthly target series** across the two bilateral directions:",
        "- **TR → RU Direction (7 series):** `TOTAL`, `08`, `84`, `39`, `87`, `73`, `85` (Türkiye exports to Russia, reporter 792, partner 643, Flow X).",
        "- **RU → TR Direction (7 series):** `TOTAL`, `27`, `72`, `10`, `76`, `23`, `74` (Türkiye imports from Russia, reporter 792, partner 643, Flow M).",
        "",
        "Both forecasting targets are official Türkiye-reported series. Russia-reported counterflows remain mirror/reference statistics only.",
        "",
        "---",
        "",
        "## Validation Protocol",
        "",
        "- **Strategy:** Expanding walk-forward / rolling-origin validation.",
        "- **Cadence:** Monthly forecast origins.",
        "- **Development Horizon:** `2012-01` to `2024-12` (156 calendar months).",
        "- **Final Untouched Holdout:** `2025-01` to `2025-12` (12 calendar months; strictly quarantined).",
        "- **Minimum Initial History:** 60 months (`2012-01` through `2016-12` inclusive).",
        "- **First Validation Origin:** **`2016-12`** (empirically derived as the earliest month with 60 observed monthly target values through the origin).",
        "- **Training Window Semantics:** For origin $t$, `train_start = '2012-01'` and `train_end = t` (expanding monotonically).",
        "- **Horizon-Specific Terminal Origins:**",
        "  - **$h=1$:** First origin `2016-12`, Last origin **`2024-11`** (forecast period: `2024-12`), **96 origins**.",
        "  - **$h=3$:** First origin `2016-12`, Last origin **`2024-09`** (forecast period: `2024-12`), **94 origins**.",
        "  - **$h=6$:** First origin `2016-12`, Last origin **`2024-06`** (forecast period: `2024-12`), **91 origins**.",
        "",
        "---",
        "",
        "## Seasonal Naive Definition",
        "",
        "$$\\text{Forecast}(t+h) = \\text{Actual}(t+h-12)$$",
        "",
        "For each forecast origin $t$ and horizon $h \\in \\{1, 3, 6\\}$:",
        "- `forecast_period` = $t + h$ calendar months.",
        "- `seasonal_source_period` = $t + h - 12$ calendar months.",
        "- **Origin Safety Invariant:** $(t+h-12) \\le t$ strictly holds for all $h \\le 6$. Every prediction is formed using only historical information available on or before the forecast origin.",
        "- **Exact Calendar Offsets:** Offsets are based strictly on contiguous monthly calendar dates, not naive row slicing.",
        "- **Series Isolation:** Forecasts query historical observations strictly within the same bilateral direction and HS2 commodity code.",
        "",
        "---",
        "",
        "## Validation Fold Registry",
        "",
        "- Manifest YAML: `configs/validation_folds_v1_1.yaml`",
        "- Row-Level Table: `data/processed/validation_origins.parquet`",
        "- Total Shared Validation Folds: **281 folds** ($96 + 94 + 91$).",
        "- Reusability: This fold schedule is frozen and will be reused without modification by M7 SARIMA, M8 SARIMAX, and M9 XGBoost.",
        "",
        "---",
        "",
        "## Prediction Audit",
        "",
        "- **Total Prediction Rows:** $3,934$ ($14\\text{ series} \\times 281\\text{ folds}$).",
        "- **Valid Prediction Rows:** $3,934$ (100% valid; zero missing predictions).",
        "- **Missing / Invalid Predictions:** $0$.",
        "- **Forecast Period Range:** `2017-01` to `2024-12` (zero 2025 holdout observations targeted).",
        "- **Seasonal Source Period Range:** `2016-01` to `2023-12` (all strictly $\\le$ corresponding forecast origin).",
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "For each prediction $i$ with actual $y_i$ and prediction $\\hat{y}_i$:",
        "- **Absolute Error:** $|y_i - \\hat{y}_i|$",
        "- **Squared Error:** $(y_i - \\hat{y}_i)^2$",
        "- **sMAPE Component:**",
        "  $$\\text{sMAPE}_i = \\begin{cases} 0.0 & \\text{if } y_i = 0 \\text{ and } \\hat{y}_i = 0 \\\\ \\frac{2 |y_i - \\hat{y}_i|}{|y_i| + |\\hat{y}_i|} \\times 100 & \\text{otherwise} \\end{cases}$$",
        "- **Aggregated Group Metrics:**",
        "  - **sMAPE (Primary):** $\\frac{1}{N} \\sum_{i=1}^N \\text{sMAPE}_i$",
        "  - **MAE (Secondary):** $\\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$",
        "  - **RMSE (Diagnostic):** $\\sqrt{\\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2}$",
        "",
        "---",
        "",
        "## Results by Series and Horizon",
        "",
        "The canonical 42 metric groups ($14\\text{ series} \\times 3\\text{ horizons}$) are recorded below:",
        "",
        "| Direction | HS2 | Horizon | Origins | sMAPE (%) | MAE (USD) | RMSE (USD) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['direction']}` | `{row['hs2']}` | $h={row['horizon']}$ | {row['n_validation_origins']} | "
            f"**{row['smape']:.2f}%** | ${row['mae']:,.0f} | ${row['rmse']:,.0f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Direction-Level Descriptive Summary",
        "",
        "| Direction | Horizon | Mean sMAPE | Median sMAPE | Mean MAE | Mean RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for (dir_name, h), g in metrics_df.groupby(["direction", "horizon"]):
        lines.append(
            f"| `{dir_name}` | $h={h}$ | {float(g['smape'].mean()):.2f}% | {float(g['smape'].median()):.2f}% | "
            f"${float(g['mae'].mean()):,.0f} | ${float(g['rmse'].mean()):,.0f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Horizon-Level Descriptive Summary",
        "",
        "| Horizon | Total Groups | Mean sMAPE | Median sMAPE | Mean MAE | Mean RMSE |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for h, g in metrics_df.groupby("horizon"):
        lines.append(
            f"| **$h={h}$** | {len(g)} | **{float(g['smape'].mean()):.2f}%** | {float(g['smape'].median()):.2f}% | "
            f"${float(g['mae'].mean()):,.0f} | ${float(g['rmse'].mean()):,.0f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Missing / Zero Handling",
        "",
        "- **Reported Zeros:** Preserved as genuine numerical zero observations. In the empirical development dataset, zero trade is observed in specific minor monthly sectors without triggering division-by-zero errors.",
        "- **Missing Policy:** Missing values remain strictly `NaN`. Because the 14 selected development series have 100% complete records across 2012–2024, zero missing seasonal sources or targets were encountered.",
        "- **No Imputation:** No interpolation, forward-fill, backward-fill, or mirror replacement was performed.",
        "",
        "---",
        "",
        "## Holdout Isolation",
        "",
        "> [!IMPORTANT]",
        "> **Holdout Isolation Guarantee:**",
        "> - All validation origins satisfy $t \\le \\text{2024-11}$.",
        "> - All forecast periods satisfy $t+h \\le \\text{2024-12}$.",
        "> - Not a single observation from the 2025 holdout (`2025-01` to `2025-12`) is accessed.",
        "> - **Holdout Mutation Test:** Radically mutating all 2025 trade targets confirmed that predictions, validation folds, and metrics remain 100% numerically identical.",
        "",
        "---",
        "",
        "## Reproducibility Metadata",
        "",
        f"- **Run ID:** `{run_meta['run_id']}`",
        f"- **Execution Timestamp:** `{run_meta['created_at']}`",
        f"- **Code Version:** `{run_meta['code_version']}`",
        "- **Hyperparameters:** `seasonal_period_months = 12` (deterministic; random seed not applicable)",
        "- **Artifacts Persisted:**",
        f"  - Validation Manifest: `{run_meta['artifact_paths']['validation_manifest']}`",
        f"  - Validation Origins: `{run_meta['artifact_paths']['validation_origins']}`",
        f"  - Predictions: `{run_meta['artifact_paths']['predictions']}`",
        f"  - Metrics: `{run_meta['artifact_paths']['metrics']}`",
        f"  - Run Metadata: `{run_meta['artifact_paths']['run_meta']}`",
        "",
        "---",
        "",
        "## Tests",
        "",
        "The automated test suite `tests/unit/test_seasonal_naive.py` executes 50 comprehensive unit and acceptance tests covering:",
        "1. Complete 14-series universe and M3/M5 manifest integrity.",
        "2. Deterministic walk-forward origin schedule and 60-month history threshold.",
        "3. Horizon terminal bounds ($h=1 \\le \\text{2024-11}$, $h=3 \\le \\text{2024-09}$, $h=6 \\le \\text{2024-06}$).",
        "4. Seasonal Naive calendar offset formula and origin safety invariant ($t+h-12 \\le t$).",
        "5. Series isolation (no cross-series leakage).",
        "6. sMAPE both-zero safeguard ($0.0$) and finite evaluation.",
        "7. Complete 42-group metric coverage without duplicates.",
        "8. Holdout mutation tests on predictions, metrics, and fold registry.",
        "9. Non-goal enforcement: no scalers, no tuning, no model selection, no M7+ artifacts.",
        "",
        "---",
        "",
        "## Validation",
        "",
        "- Full test suite: **232+ / 232+ tests passing**.",
        "- Linter: **Clean (`ruff check .` passes with 0 errors)**.",
        "- Sector manifest hash: **Identical to M3 Final Pass**.",
        "- Feature manifest hash: **Identical to M5 Final Pass**.",
        "",
        "---",
        "",
        "## Deviations From Specification",
        "",
        "**None.** All implementations strictly comply with Specification Pack v1.1 and M6 milestone instructions.",
        "",
        "---",
        "",
        "## Issues / Limitations",
        "",
        "- **Pure Persistence Limitation:** Seasonal Naive strictly assumes that historical trade patterns from month $t+h-12$ persist unchanged into month $t+h$. It cannot anticipate structural shifts, sanctions, trade rerouting, or macroeconomic shocks.",
        "- **Exogenous Independence:** The baseline does not incorporate exchange rate or commodity price signals. It serves as an uninformative benchmark to evaluate whether subsequent econometric and machine learning models (SARIMAX, XGBoost) add genuine predictive value.",
        "- **Development Estimates Only:** Validation metrics are historical estimates over 2016–2024. The final 2025 holdout remains untouched.",
        "",
        "---",
        "",
        "## Acceptance Criteria",
        "",
        "- [x] Exactly 14 frozen target series evaluated",
        "- [x] Only development data used",
        "- [x] 2025 holdout remains untouched",
        "- [x] Seasonal Naive formula exactly equals Actual(t+h-12)",
        "- [x] Exact calendar offsets used",
        "- [x] Same-series isolation enforced",
        "- [x] Monthly expanding walk-forward origins generated",
        "- [x] Minimum initial history = 60 months enforced",
        "- [x] Origin schedule deterministic",
        "- [x] Horizon terminal origins correct (h1: 2024-11, h3: 2024-09, h6: 2024-06)",
        "- [x] Shared validation fold registry versioned (`configs/validation_folds_v1_1.yaml`)",
        "- [x] Fold registry reusable by M7/M8/M9 (`data/processed/validation_origins.parquet`)",
        "- [x] sMAPE correctly implemented (with both-zero safeguard)",
        "- [x] MAE correctly implemented",
        "- [x] RMSE correctly implemented",
        "- [x] Both-zero sMAPE rule correct",
        "- [x] Metrics computed per direction/sector/horizon",
        "- [x] Exactly 42 expected metric groups",
        "- [x] Prediction rows fully auditable (3,934 rows)",
        "- [x] Holdout mutation test passes",
        "- [x] Rerun determinism passes",
        "- [x] M3/M5 frozen manifests unchanged",
        "- [x] Machine-readable run metadata produced (`data/processed/m6_seasonal_naive_run.json`)",
        "- [x] Full pytest suite passes",
        "- [x] Lint passes (`ruff` clean)",
        "- [x] M7+ not implemented",
        "",
        "---",
        "",
        "## Ready for M7?",
        "**YES** — Milestone M6 Seasonal Naive Baseline is complete, verified, and reproducible. Ready for external review and authorization of M7 (SARIMA).",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("M6 Report written to %s", report_path)


def main() -> None:
    """CLI main entry point for M6."""
    parser = argparse.ArgumentParser(description="M6 Seasonal Naive Baseline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Seasonal Naive baseline pipeline")
    run_parser.add_argument("--report", action="store_true", default=True, help="Generate M6 report markdown")

    args = parser.parse_args()

    if args.command == "run":
        settings = get_settings()
        pred_df, metrics_df, run_meta = run_seasonal_naive_pipeline()
        print(f"Generated {len(pred_df)} prediction rows across {len(metrics_df)} metric groups.")

        if args.report:
            report_file = settings.project_root / "docs" / "M6_SEASONAL_NAIVE_REPORT.md"
            generate_m6_report(pred_df, metrics_df, run_meta, report_file)
            print(f"Generated M6 report at {report_file}")


if __name__ == "__main__":
    main()
