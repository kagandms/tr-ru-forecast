"""Reproducible report tables from frozen CV selections, without model refitting."""

from __future__ import annotations

import pandas as pd

from tr_ru_forecast.evaluation.champion_io import ChampionRegistry
from tr_ru_forecast.evaluation.champion_selector import MODEL_FAMILIES, Candidate


def build_win_share_table(registry: ChampionRegistry) -> str:
    """Render family wins. Args: registry. Returns: Markdown. Raises: none."""
    lines = [
        "| Family | h1 | h3 | h6 | Overall | Share | Mean skill % | Median skill % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in MODEL_FAMILIES:
        winners = [row for row in registry.champions if row.champion_model == family]
        counts = [sum(row.horizon == horizon for row in winners) for horizon in (1, 3, 6)]
        skills = pd.Series([row.skill_vs_naive_pct for row in winners], dtype=float)
        summary = f"{skills.mean():.2f} | {skills.median():.2f}" if len(skills) else "— | —"
        lines.append(
            f"| {family} | {counts[0]} | {counts[1]} | {counts[2]} | {len(winners)} | {len(winners) / 42:.1%} | {summary} |"
        )
    return "\n".join(lines)


def build_matrix_table(registry: ChampionRegistry) -> str:
    """Render all 42 decisions. Args: registry. Returns: Markdown. Raises: none."""
    lines = [
        "| Sector | h | Champion | Fallback | CV sMAPE % | Std | Worst | Naive % | Skill % | Reason |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in registry.champions:
        lines.append(
            f"| {row.sector_id} | {row.horizon} | {row.champion_model} | {row.fallback_model} | "
            f"{row.mean_smape:.2f} | {row.std_smape:.2f} | {row.max_fold_smape:.2f} | "
            f"{row.baseline_reference.mean_smape:.2f} | {row.skill_vs_naive_pct:.2f} | {row.selection_reason} |"
        )
    return "\n".join(lines)


def build_m10_report(registry: ChampionRegistry, candidates: tuple[Candidate, ...]) -> str:
    """Generate evidence report. Args: registry, leaderboard. Returns: Markdown. Raises: none."""
    lines = _build_summary_lines(registry) + [
        "## 2. Selection Protocol & Governance Rules",
        "",
        _build_protocol_text(),
        "",
        "## 3. Global Model Family Win Share",
        "",
        build_win_share_table(registry),
        "",
        "## 4. Complete 42-Task Champion Matrix",
        "",
        build_matrix_table(registry),
        "",
        "## 5. Eligibility Audit",
        "",
        _build_eligibility_text(candidates),
        "",
        "## 6. Economic & Domain Interpretation",
        "",
        _build_interpretation(registry),
        "",
    ]
    return "\n".join(lines + _build_provenance_lines(registry))


def _build_protocol_text() -> str:
    return (
        "The frozen calendar contains 96/94/91 monthly rolling origins for h1/h3/h6 (281 shared folds), "
        "not five folds. The pasted five-fold/RMSE-first proposal is superseded by the explicitly authoritative protocol. "
        "Each task requires complete finite, nonnegative predictions and successful converged fits. Failed candidates remain in the leaderboard. "
        "Seasonal Naive must be eligible, otherwise selection fails closed. No silent fallback or clipping repairs CV evidence.\n\n"
        "Rank by mean sMAPE, MAE, diagnostic RMSE, population standard deviation, then complexity tier. "
        "Starting from the raw winner, select the simplest eligible model whose relative sMAPE loss against that winner is strictly below 2%, "
        "using the simpler model as the denominator (protocol §8). Exactly 2% retains the complex winner. "
        "This deterministic tolerance is not a test of statistical indistinguishability.\n\n"
        "A worst-origin ratio above 1.25 against an eligible candidate in the same parsimony window flags instability; "
        "it does not override the frozen selection rule. The cap is a configurable descriptive threshold, not a fitted parameter. "
        "Fallback is the best ranked lower-complexity candidate for SARIMAX/XGBoost champions; otherwise the best other eligible candidate. "
        "If only the baseline is eligible, fallback explicitly repeats the baseline.\n\n"
        "sMAPE/MAE are means over origins; std uses ddof=0. `mean_rmse` preserves the M6–M9 pooled RMSE "
        "(square root of mean squared error), since the mean of single-observation RMSEs would equal MAE. "
        "Skill is (naive−candidate)/naive, with percentage multiplied by 100. A zero baseline uses skill 0 for an exact tie "
        "and −1 for a positive-error candidate, an explicit finite sentinel. Missing-candidate metrics remain null. Finite negative forecasts remain in descriptive scores while invalidating their candidate.\n\n"
        "Model references identify frozen refit specifications, not serialized serving models. Seasonal Naive intervals are "
        "explicitly unavailable until a separate approved method is supplied; native classical and M9 prequential interval policies are retained."
    )


def _build_eligibility_text(candidates: tuple[Candidate, ...]) -> str:
    lines = [
        "| Family | Eligible / 42 | Negative forecasts | Ineligible tasks |",
        "| --- | ---: | ---: | --- |",
    ]
    for family in MODEL_FAMILIES:
        rows = [row for row in candidates if row.model_family == family]
        excluded = (
            ", ".join(
                f"{row.direction}:{row.hs2}/h{row.horizon}"
                for row in rows
                if row.eligibility_status != "ELIGIBLE"
            )
            or "none"
        )
        lines.append(
            f"| {family} | {sum(row.eligibility_status == 'ELIGIBLE' for row in rows)} | {sum(row.n_negative_predictions for row in rows)} | {excluded} |"
        )
    return "\n".join(lines)


def _build_interpretation(registry: ChampionRegistry) -> str:
    macro = [row for row in registry.champions if row.champion_model in ("sarimax", "xgboost")]
    names = ", ".join(f"{row.sector_id}/h{row.horizon} ({row.champion_model})" for row in macro)
    return (
        f"Exogenous-capable families win {len(macro)} tasks: {names or 'none'}. "
        "These are observed predictive comparisons; XGBoost also uses trade lags/calendar features, so its wins do not isolate macro contributions. "
        "SARIMAX uses unchanged univariate orders with a constant origin-known macro path; longer horizons cannot benefit from future macro realizations. "
        "SARIMA wins are consistent with useful within-series dynamics and seasonality, but do not establish a causal economic mechanism.\n\n"
        "TOTAL aggregates and HS2 components have different volatility and seasonal structure. Fruit exports (08), energy imports (27), "
        "and machinery/metal flows are economically different series; the table supports sector/horizon heterogeneity without attributing shocks to specific causes. "
        "Worst-origin and dispersion diagnostics retain large development errors, including origins around 2022. "
        "No break-specific effect estimate or statistical significance claim is made.\n\n"
        "The comparison uses retrospective latest-vintage development artifacts. Approximate publication lags do not establish historical "
        "macro vintage fidelity or actual trade decision-time availability. Winner shares and CV skill are selection-period results; "
        "independent 2025 performance remains unknown. Negative SARIMAX/XGBoost point forecasts are retained as eligibility evidence."
    )


def _build_provenance_lines(registry: ChampionRegistry) -> list[str]:
    provenance = registry.provenance
    return [
        "## 7. Reproducibility & Provenance Evidence",
        "",
        f"Git commit: `{provenance.git_commit}`; dirty working tree: `{provenance.is_working_tree_dirty}`. "
        f"Source fingerprint: `{provenance.source_fingerprint}`.",
        "",
        "Commands: `make m10-select`; reproduce independently with `make m10-select M10_OUTPUT_DIR=/tmp/m10-reproduction`. "
        "Existing output files are never overwritten. Compare champion decisions separately from run timestamps and Git dirty-state metadata.",
        "",
        "Artifacts: `configs/champion_registry_v1_1.yaml`, `data/processed/m10_cv_leaderboard.parquet` (168 rows), "
        "`m10_cv_fold_details.parquet` (15,736 observed CV rows), `m10_champion_matrix.parquet` (42 rows), "
        "`m10_cv_leaderboard.csv`, and `m10_selection_run.json`. Registry holds input/output checksums; run JSON also hashes the registry and report.",
        "",
        "The M9 accepted rerun config, prediction and fit-log hashes are verified against its run metadata. "
        "Every candidate is checked against the frozen calendar and shared baseline actuals. M10 reads only allowlisted CV artifacts, "
        "configuration, run metadata and source files; raw trade/target/macro/feature datasets are not opened. "
        "This boundary does not establish that earlier milestones never read 2025 data.",
        "",
        "Legacy discrepancies: "
        + ("; ".join(provenance.legacy_metadata_discrepancies) or "none")
        + ".",
        "",
        "Executed check results and test counts are recorded in `docs/M10_ACCEPTANCE_REPORT.md`; the generator does not assert unexecuted checks.",
        "",
    ]


def _build_summary_lines(registry: ChampionRegistry) -> list[str]:
    return [
        "# M10 Champion Selection Report",
        "",
        f"**Generated:** {registry.provenance.created_at}",
        "**Specification:** v1.1",
        "**Status:** CV selection frozen; M11 acceptance gate remains closed.",
        "",
        "## 1. Executive Summary & Acceptance Verdict",
        "",
        "Computed 168 candidate entries and 42 deterministic champion/fallback decisions from the existing development CV artifacts. "
        "Selection implementation is complete; full experiment release is conditional on the open M9 A2 review and the legacy M8 provenance discrepancy. "
        "No holdout evaluation, model refit, API implementation or causal interpretation is included.",
        "",
    ]
