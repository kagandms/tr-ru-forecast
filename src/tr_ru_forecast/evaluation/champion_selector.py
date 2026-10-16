"""Pure CV aggregation and protocol-v1.1 champion selection; no file access."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ModelFamily = Literal["seasonal_naive", "sarima", "sarimax", "xgboost"]
SelectionReason = Literal[
    "CLEAR_SMAPE_WINNER",
    "PARSIMONY_PREFERENCE",
    "TIE_BREAK_MAE",
    "TIE_BREAK_RMSE",
    "TIE_BREAK_STABILITY",
    "TIE_BREAK_COMPLEXITY",
]
MODEL_FAMILIES: tuple[ModelFamily, ...] = ("seasonal_naive", "sarima", "sarimax", "xgboost")
TASK_COLUMNS = ["direction", "hs2", "is_total", "horizon"]
FOLD_COLUMNS = TASK_COLUMNS + ["forecast_origin", "forecast_period"]


class SelectionSpec(BaseModel):
    """Frozen selection configuration; unsupported protocol changes fail validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    spec_version: Literal["1.1"]
    selection_version: Literal["1.1.0"]
    parsimony_relative_threshold: Literal[0.02]
    complexity_tiers: dict[ModelFamily, int]
    tie_breaks: tuple[str, ...]
    stability_worst_fold_ratio: FiniteFloat = Field(gt=1)
    stability_policy: Literal["diagnostic_only"]
    negative_prediction_policy: Literal["ineligible_no_clipping"]
    fold_policy: Literal["frozen_monthly_origins"]
    inputs: dict[ModelFamily, str]
    xgboost_fit_log: str
    xgboost_run: str
    xgboost_config: str
    output_directory: str
    registry_path: str
    report_path: str


class Candidate(BaseModel):
    """Immutable per-task CV evidence, retaining invalid candidates for audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    direction: Literal["TR_RU", "RU_TR"]
    hs2: str
    is_total: bool
    horizon: Literal[1, 3, 6]
    model_family: ModelFamily
    complexity_tier: int
    mean_smape: FiniteFloat | None
    std_smape: FiniteFloat | None
    median_smape: FiniteFloat | None
    max_fold_smape: FiniteFloat | None
    mean_mae: FiniteFloat | None
    mean_rmse: FiniteFloat | None
    n_expected_folds: int
    n_observed_folds: int
    n_valid_folds: int
    n_negative_predictions: int
    convergence_rate: FiniteFloat
    eligibility_status: Literal["ELIGIBLE", "INCOMPLETE", "INVALID_PREDICTION", "FIT_FAILURE"]
    delta_smape_vs_naive_pct: FiniteFloat = 0
    skill_score_smape: FiniteFloat = 0


class Selection(BaseModel):
    """Champion, runner-up and diagnostic evidence for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    champion: Candidate
    fallback: Candidate
    baseline_reference: Candidate
    raw_best_model: ModelFamily
    selection_reason: SelectionReason
    has_stability_flag: bool


def validate_selection_spec(spec: SelectionSpec) -> None:
    """Validate spec invariants. Args: spec. Returns: None. Raises: ValueError."""
    expected_tiers = dict(zip(MODEL_FAMILIES, range(4), strict=True))
    if spec.complexity_tiers != expected_tiers or set(spec.inputs) != set(MODEL_FAMILIES):
        raise ValueError("All four frozen model families and complexity tiers are required")
    if spec.tie_breaks != ("mean_smape", "mean_mae", "mean_rmse", "std_smape", "complexity_tier"):
        raise ValueError("Protocol v1.1 requires MAE before diagnostic RMSE")


def compute_skill(baseline_smape: float, candidate_smape: float) -> tuple[float, float]:
    """Return relative reduction and percent; (baseline-candidate)/baseline, protocol §8.

    Args: nonnegative finite sMAPEs. Returns: (skill, percent). Raises: ValueError.
    Zero baseline gives zero for a zero candidate and -1 for a worse candidate.
    """
    if (
        not np.isfinite([baseline_smape, candidate_smape]).all()
        or min(baseline_smape, candidate_smape) < 0
    ):
        raise ValueError("sMAPE must be finite and nonnegative")
    if baseline_smape == 0:
        return (0.0, 0.0) if candidate_smape == 0 else (-1.0, -100.0)
    skill = (baseline_smape - candidate_smape) / baseline_smape
    return skill, skill * 100


def _compute_group_summary(group: pd.DataFrame) -> dict[str, float | int | None]:
    valid = group.loc[np.isfinite(group[["actual_value", "prediction"]]).all(axis=1)]
    absolute = (valid["actual_value"] - valid["prediction"]).abs()
    denominator = valid["actual_value"].abs() + valid["prediction"].abs()
    smape = absolute.div(denominator.where(denominator != 0, 1)).mul(200)
    return {
        "mean_smape": float(smape.mean()) if len(valid) else None,
        "std_smape": float(smape.std(ddof=0)) if len(valid) else None,
        "median_smape": float(smape.median()) if len(valid) else None,
        "max_fold_smape": float(smape.max()) if len(valid) else None,
        "mean_mae": float(absolute.mean()) if len(valid) else None,
        "mean_rmse": float(np.sqrt(absolute.pow(2).mean())) if len(valid) else None,
        "n_observed_folds": len(group),
        "n_valid_folds": int(group["is_valid_prediction"].sum()),
        "n_negative_predictions": int(group["prediction"].lt(0).sum()),
        "convergence_rate": float(group["has_successful_fit"].sum() / len(group))
        if len(group)
        else 0,
    }


def _get_eligibility(group: pd.DataFrame, expected_count: int) -> str:
    if len(group) != expected_count:
        return "INCOMPLETE"
    if not group["has_successful_fit"].all():
        return "FIT_FAILURE"
    if not group["is_valid_prediction"].all():
        return "INVALID_PREDICTION"
    return "ELIGIBLE"


def aggregate_candidates(
    folds: pd.DataFrame, tasks: pd.DataFrame, spec: SelectionSpec
) -> tuple[Candidate, ...]:
    """Aggregate CV rows. Args: normalized folds, task grid, spec. Returns: candidates.

    Raises: ValueError for an ineligible baseline. RMSE is pooled over origins, as in M6–M9.
    Finite negative predictions remain in descriptive scores but invalidate the candidate.
    """
    candidates: list[Candidate] = []
    for task in tasks.to_dict("records"):
        mask = (
            folds[TASK_COLUMNS].eq(pd.Series({key: task[key] for key in TASK_COLUMNS})).all(axis=1)
        )
        group = folds.loc[mask]
        for family in MODEL_FAMILIES:
            family_group = group.loc[group["model_family"] == family]
            identity = {
                **task,
                "model_family": family,
                "complexity_tier": spec.complexity_tiers[family],
            }
            candidates.append(_build_candidate(family_group, identity))
    return _attach_baseline_skills(tuple(candidates))


def _build_candidate(group: pd.DataFrame, identity: dict[str, object]) -> Candidate:
    expected_count = int(str(identity["n_expected_folds"]))
    summary = _compute_group_summary(group)
    summary["convergence_rate"] = float(group["has_successful_fit"].sum() / expected_count)
    return Candidate.model_validate(
        {**identity, **summary, "eligibility_status": _get_eligibility(group, expected_count)}
    )


def _attach_baseline_skills(candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    baselines = {
        (row.direction, row.hs2, row.horizon): row
        for row in candidates
        if row.model_family == "seasonal_naive"
    }
    updated: list[Candidate] = []
    for candidate in candidates:
        baseline = baselines[candidate.direction, candidate.hs2, candidate.horizon]
        if baseline.eligibility_status != "ELIGIBLE" or baseline.mean_smape is None:
            raise ValueError("Seasonal Naive must be a complete, valid baseline for every task")
        if candidate.mean_smape is None:
            updated.append(candidate)
            continue
        skill, percent = compute_skill(baseline.mean_smape, candidate.mean_smape)
        updated.append(
            candidate.model_copy(
                update={"skill_score_smape": skill, "delta_smape_vs_naive_pct": percent}
            )
        )
    return tuple(updated)


def _rank_candidate(candidate: Candidate) -> tuple[float, ...]:
    metrics = (candidate.mean_smape, candidate.mean_mae, candidate.mean_rmse, candidate.std_smape)
    if candidate.eligibility_status != "ELIGIBLE" or any(value is None for value in metrics):
        raise ValueError("Cannot rank incomplete candidate")
    return tuple(float(value) for value in metrics if value is not None) + (
        float(candidate.complexity_tier),
    )


def _get_tie_reason(ranked: list[Candidate]) -> SelectionReason:
    if len(ranked) == 1 or ranked[0].mean_smape != ranked[1].mean_smape:
        return "CLEAR_SMAPE_WINNER"
    tie_reasons: tuple[tuple[str, SelectionReason], ...] = (
        ("mean_mae", "TIE_BREAK_MAE"),
        ("mean_rmse", "TIE_BREAK_RMSE"),
        ("std_smape", "TIE_BREAK_STABILITY"),
    )
    for attribute, reason in tie_reasons:
        if getattr(ranked[0], attribute) != getattr(ranked[1], attribute):
            return reason
    return "TIE_BREAK_COMPLEXITY"


def select_champion(candidates: tuple[Candidate, ...], spec: SelectionSpec) -> Selection:
    """Select one task. Args: candidates, frozen spec. Returns: audited selection.

    Raises: ValueError for mixed tasks or invalid baseline. Simplest within STRICTLY <2%
    relative loss to raw winner wins; exact 2% preserves the complex winner.
    """
    if len({(row.direction, row.hs2, row.horizon) for row in candidates}) != 1:
        raise ValueError("Selection must contain exactly one task")
    if len({row.model_family for row in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate family")
    ranked = sorted(
        (row for row in candidates if row.eligibility_status == "ELIGIBLE"), key=_rank_candidate
    )
    baseline = next((row for row in ranked if row.model_family == "seasonal_naive"), None)
    if baseline is None:
        raise ValueError("Eligible baseline is required")
    raw_best = ranked[0]
    comparable = _get_comparable_candidates(ranked, spec.parsimony_relative_threshold)
    champion = min(comparable, key=lambda row: (row.complexity_tier, _rank_candidate(row)))
    reason = "PARSIMONY_PREFERENCE" if champion != raw_best else _get_tie_reason(ranked)
    return Selection(
        champion=champion,
        fallback=_get_fallback(champion, ranked),
        baseline_reference=baseline,
        raw_best_model=raw_best.model_family,
        selection_reason=reason,
        has_stability_flag=_has_stability_flag(
            champion, comparable, spec.stability_worst_fold_ratio
        ),
    )


def _get_comparable_candidates(ranked: list[Candidate], threshold: float) -> list[Candidate]:
    best_smape = ranked[0].mean_smape
    if best_smape is None:
        raise ValueError("Eligible winner must have a finite sMAPE")
    return [
        row
        for row in ranked
        if row.mean_smape is not None
        and (row == ranked[0] or compute_skill(row.mean_smape, best_smape)[0] < threshold)
    ]


def _get_fallback(champion: Candidate, ranked: list[Candidate]) -> Candidate:
    remaining = [row for row in ranked if row != champion]
    lower = [row for row in remaining if row.complexity_tier < champion.complexity_tier]
    if champion.complexity_tier >= 2 and lower:
        return lower[0]
    return (remaining or [champion])[0]


def _has_stability_flag(champion: Candidate, comparable: list[Candidate], ratio: float) -> bool:
    worst = champion.max_fold_smape
    return worst is not None and any(
        row != champion and row.max_fold_smape is not None and worst > ratio * row.max_fold_smape
        for row in comparable
    )


def build_champion_matrix(
    candidates: tuple[Candidate, ...], spec: SelectionSpec
) -> tuple[Selection, ...]:
    """Select the task grid. Args: candidates, spec. Returns: sorted selections. Raises: ValueError."""
    keys = sorted({(row.direction, row.hs2, row.horizon) for row in candidates})
    return tuple(
        select_champion(
            tuple(row for row in candidates if (row.direction, row.hs2, row.horizon) == key), spec
        )
        for key in keys
    )
