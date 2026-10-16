"""Project frozen CV and holdout metrics into one explicit task contract."""

from __future__ import annotations

import pandas as pd

from tr_ru_forecast.serving.artifacts import ArtifactError, ArtifactTables
from tr_ru_forecast.serving.catalog_schemas import CvCandidate, GlobalKpis, TaskComparison
from tr_ru_forecast.serving.schemas import IntervalDiagnostics, MetricSummary


def build_comparisons(tables: ArtifactTables) -> tuple[TaskComparison, ...]:
    """Join frozen task matrices. Args: tables. Returns: immutable comparisons. Raises: ArtifactError."""
    joined = tables.holdout.merge(
        tables.matrix[["sector_id", "horizon", "skill_vs_naive_pct", "n_valid_folds", "mean_mae", "mean_rmse", "selection_reason", "has_stability_flag"]],
        on=["sector_id", "horizon"], suffixes=("", "_cv"), validate="one_to_one",
    ).sort_values(["sector_id", "horizon"])
    if len(joined) != 42:
        raise ArtifactError("Expected 42 frozen task comparisons")
    return tuple(_build_comparison(record, tables.candidates) for _, record in joined.iterrows())


def _build_comparison(record: pd.Series, candidates: pd.DataFrame) -> TaskComparison:
    return TaskComparison(
        sector_id=record.sector_id, direction=record.direction, hs2=record.hs2,
        horizon=record.horizon, champion_model=record.champion_model,
        fallback_model=record.fallback_model, selection_reason=record.selection_reason,
        has_stability_flag=bool(record.has_stability_flag), cv=_build_cv(record),
        champion=build_metrics(record, ""), operational=build_metrics(record, "operational_"),
        fallback=build_metrics(record, "fallback_"), baseline=build_metrics(record, "baseline_"),
        cv_candidates=_build_candidates(record, candidates),
        raw_champion_smape=record.raw_mean_smape,
        cv_vs_holdout_drift=record.cv_vs_holdout_smape_drift,
        fallback_replacements=record.n_operational_fallbacks,
        intervals=build_interval_diagnostics(record),
    )


def build_metrics(record: pd.Series, prefix: str) -> MetricSummary:
    """Project metric view. Args: record, prefix. Returns: metrics. Raises: KeyError."""
    valid_key = "n_valid_operational_origins" if prefix == "operational_" else prefix + "n_valid_origins"
    return MetricSummary(
        mean_smape=record[prefix + "mean_smape"], mean_mae=record[prefix + "mean_mae"],
        mean_rmse=record[prefix + "mean_rmse"], valid_origins=int(record[valid_key]),
        expected_origins=int(record.n_expected_origins),
        skill_vs_naive_pct=record.skill_vs_naive_pct if not prefix else None,
    )


def _build_cv(record: pd.Series) -> MetricSummary:
    return MetricSummary(
        mean_smape=record.cv_mean_smape, mean_mae=record.mean_mae_cv,
        mean_rmse=record.mean_rmse_cv, valid_origins=record.n_valid_folds,
        expected_origins=record.n_valid_folds, skill_vs_naive_pct=record.skill_vs_naive_pct_cv,
    )


def build_interval_diagnostics(record: pd.Series) -> IntervalDiagnostics:
    """Project primary coverage. Args: record. Returns: diagnostics. Raises: KeyError."""
    return IntervalDiagnostics(
        empirical_coverage_80=record.empirical_coverage_80,
        empirical_coverage_95=record.empirical_coverage_95,
        interval_origins=record.n_interval_origins,
        mean_width_80=record.mean_interval_width_80, mean_width_95=record.mean_interval_width_95,
    )


def build_global_kpis(tasks: tuple[TaskComparison, ...]) -> GlobalKpis:
    """Average task-level means, not sector trade volumes. Args: tasks. Returns: KPIs. Raises: ValueError."""
    return GlobalKpis(
        operational_mean_smape=_mean(tuple(task.operational.mean_smape for task in tasks)),
        baseline_mean_smape=_mean(tuple(task.baseline.mean_smape for task in tasks)),
        cv_mean_skill_pct=_mean(tuple(task.cv.skill_vs_naive_pct for task in tasks)),
        sarima_champions=sum(task.champion_model == "sarima" for task in tasks),
        xgboost_champions=sum(task.champion_model == "xgboost" for task in tasks),
        fallback_replacements=sum(task.fallback_replacements for task in tasks), total_tasks=len(tasks),
    )


def _mean(values: tuple[float | None, ...]) -> float | None:
    observed = tuple(value for value in values if value is not None)
    return sum(observed) / len(observed) if observed else None


def _build_candidates(record: pd.Series, candidates: pd.DataFrame) -> tuple[CvCandidate, ...]:
    selected = candidates.loc[candidates.direction.eq(record.direction) & candidates.hs2.eq(record.hs2)
                              & candidates.horizon.eq(record.horizon)].sort_values("model_family")
    return tuple(CvCandidate.model_validate(candidate) for candidate in
                 selected[list(CvCandidate.model_fields)].to_dict(orient="records"))
