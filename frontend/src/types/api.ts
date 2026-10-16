export type Direction = 'TR_RU' | 'RU_TR';
export type ModelFamily = 'sarima' | 'sarimax' | 'xgboost' | 'seasonal_naive';
export type Horizon = 1 | 3 | 6;
export type NumberValue = number | null;

export interface MetricSummary {
  readonly mean_smape: NumberValue;
  readonly mean_mae: NumberValue;
  readonly mean_rmse: NumberValue;
  readonly valid_origins: number;
  readonly expected_origins: number;
  readonly skill_vs_naive_pct: NumberValue;
}

export interface IntervalDiagnostics {
  readonly empirical_coverage_80: NumberValue;
  readonly empirical_coverage_95: NumberValue;
  readonly interval_origins: number;
  readonly mean_width_80: NumberValue;
  readonly mean_width_95: NumberValue;
  readonly calibration_pool_size: number | null;
  readonly coverage_unit: 'fraction';
}

export interface CvCandidate {
  readonly model_family: ModelFamily;
  readonly mean_smape: NumberValue;
  readonly mean_mae: NumberValue;
  readonly mean_rmse: NumberValue;
  readonly eligibility_status: string;
  readonly n_expected_folds: number;
  readonly n_valid_folds: number;
  readonly n_negative_predictions: number;
  readonly convergence_rate: NumberValue;
}

export interface TaskComparison {
  readonly sector_id: string;
  readonly direction: Direction;
  readonly hs2: string;
  readonly horizon: Horizon;
  readonly champion_model: ModelFamily;
  readonly fallback_model: ModelFamily;
  readonly selection_reason: string;
  readonly has_stability_flag: boolean;
  readonly cv: MetricSummary;
  readonly champion: MetricSummary;
  readonly operational: MetricSummary;
  readonly fallback: MetricSummary;
  readonly baseline: MetricSummary;
  readonly cv_candidates: readonly CvCandidate[];
  readonly raw_champion_smape: NumberValue;
  readonly cv_vs_holdout_drift: NumberValue;
  readonly fallback_replacements: number;
  readonly intervals: IntervalDiagnostics;
}

export interface GlobalKpis {
  readonly operational_mean_smape: NumberValue;
  readonly baseline_mean_smape: NumberValue;
  readonly cv_mean_skill_pct: NumberValue;
  readonly sarima_champions: number;
  readonly xgboost_champions: number;
  readonly fallback_replacements: number;
  readonly total_tasks: number;
  readonly weighting: 'equal_task';
}

export interface LeaderboardResponse {
  readonly evaluation_scope: string;
  readonly summary: GlobalKpis;
  readonly tasks: readonly TaskComparison[];
}

export interface TaskAssignment {
  readonly horizon: Horizon;
  readonly champion_model: ModelFamily;
  readonly fallback_model: ModelFamily;
}

export interface Sector {
  readonly sector_id: string;
  readonly direction: Direction;
  readonly hs2: string;
  readonly title: string;
  readonly is_total: boolean;
  readonly history_start: string;
  readonly history_end: string;
  readonly latest_trade_value_usd: string | null;
  readonly source: string;
  readonly reporter_code: number;
  readonly flow: string;
  readonly retrieved_at: string;
  readonly tasks: readonly TaskAssignment[];
}

export interface SectorCatalog {
  readonly evaluation_scope: string;
  readonly sectors: readonly Sector[];
}

export interface ForecastPoint {
  readonly origin_date: string;
  readonly target_date: string;
  readonly actual: NumberValue;
  readonly operational_forecast: NumberValue;
  readonly champion_forecast: NumberValue;
  readonly fallback_forecast: NumberValue;
  readonly seasonal_naive_forecast: NumberValue;
  readonly used_fallback: boolean;
  readonly operational_model: ModelFamily | null;
  readonly status: 'available' | 'unavailable';
  readonly champion_is_valid: boolean;
  readonly fallback_is_valid: boolean;
  readonly lower_80: NumberValue;
  readonly upper_80: NumberValue;
  readonly lower_95: NumberValue;
  readonly upper_95: NumberValue;
  readonly interval_policy: 'native' | 'prequential_conformal' | 'unavailable';
  readonly calibration_pool_size: number;
  readonly train_end: string;
  readonly pre_freeze_origin: boolean;
}

export interface ForecastTask {
  readonly horizon: Horizon;
  readonly champion_model: ModelFamily;
  readonly fallback_model: ModelFamily;
  readonly metrics: MetricSummary;
  readonly primary_metrics: MetricSummary;
  readonly intervals: IntervalDiagnostics;
  readonly points: readonly ForecastPoint[];
}

export interface HistoricalPoint {
  readonly period: string;
  readonly trade_value_usd: string | null;
  readonly data_status: string;
  readonly source: string;
  readonly reporter_code: number;
  readonly partner_code: number;
  readonly flow: string;
  readonly retrieved_at: string;
}

export interface ForecastResponse {
  readonly sector_id: string;
  readonly direction: Direction;
  readonly hs2: string;
  readonly evaluation_scope: string;
  readonly forecast_kind: 'historical_holdout';
  readonly unit: 'USD';
  readonly observation_cutoff: string;
  readonly history: readonly HistoricalPoint[];
  readonly tasks: readonly ForecastTask[];
}

export interface FeatureImportance {
  readonly feature: string;
  readonly mean_abs_shap: NumberValue;
  readonly mean_signed_shap: NumberValue;
  readonly category: 'trade_lag' | 'rolling_stat' | 'seasonality' | 'macro_indicator';
  readonly n_origins: number;
}

export interface Attribution {
  readonly feature: string;
  readonly feature_value: NumberValue;
  readonly shap_value: NumberValue;
  readonly category: 'trade_lag' | 'rolling_stat' | 'seasonality' | 'macro_indicator';
}

export interface LocalExplanation {
  readonly origin_month: string;
  readonly target_month: string;
  readonly base_value: NumberValue;
  readonly prediction: NumberValue;
  readonly contributions: readonly Attribution[];
  readonly top_positive: readonly Attribution[];
  readonly top_negative: readonly Attribution[];
}

export type ExplanationResponse =
  | {
      readonly status: 'available';
      readonly sector_id: string;
      readonly horizon: Horizon;
      readonly model_family: 'xgboost';
      readonly role: 'champion' | 'fallback';
      readonly evaluation_scope: string;
      readonly global_importance: readonly FeatureImportance[];
      readonly local: readonly LocalExplanation[];
      readonly explanation_scope: 'predictive_attribution_not_causality';
    }
  | {
      readonly status: 'unavailable';
      readonly sector_id: string;
      readonly horizon: Horizon;
      readonly evaluation_scope: string;
      readonly reason: 'no_xgboost_role';
    };

export interface MirrorPoint {
  readonly sector_id: string;
  readonly direction: Direction;
  readonly hs2: string;
  readonly period: string;
  readonly canonical_value_usd: string | null;
  readonly mirror_value_usd: string | null;
  readonly mirror_gap_usd: string | null;
  readonly mirror_gap_pct: NumberValue;
  readonly abs_mirror_gap_pct: NumberValue;
  readonly pair_status: string;
  readonly classification_mismatch: boolean;
  readonly gap_flag: string;
  readonly canonical_retrieved_at: string | null;
  readonly mirror_retrieved_at: string | null;
  readonly forecast_target_column: 'canonical_value_usd' | 'mirror_value_usd';
}

export interface MirrorSummary {
  readonly period_group: 'pre_2022' | 'post_2022';
  readonly observation_count: number;
  readonly paired_count: number;
  readonly mean_abs_gap_pct: NumberValue;
  readonly median_abs_gap_pct: NumberValue;
}

export interface MirrorResponse {
  readonly source: 'UN Comtrade';
  readonly reporter_sources: readonly string[];
  readonly gap_definition: string;
  readonly valuation_note: string;
  readonly structural_shift_status: 'available' | 'unavailable_no_paired_post_2022';
  readonly structural_shift_pp: NumberValue;
  readonly summaries: readonly MirrorSummary[];
  readonly observations: readonly MirrorPoint[];
  readonly total_observations: number;
  readonly offset: number;
  readonly limit: number;
}

export interface HealthResponse {
  readonly status: 'healthy' | 'degraded';
  readonly timestamp?: string;
  readonly version: string;
  readonly environment: string;
  readonly specification_version: '1.1';
  readonly evaluation_scope: string;
  readonly checksum_status: 'verified' | 'failed';
  readonly verification_scope: string;
  readonly verified_file_count: number;
  readonly serving_manifest_sha256: string | null;
  readonly total_tasks: number;
  readonly total_holdout_predictions: number;
  readonly total_shap_rows: number;
  readonly readiness: { readonly m10: boolean; readonly m11: boolean; readonly history: boolean; readonly mirror: boolean; readonly macro: boolean };
  readonly m10: { readonly git_commit: string; readonly source_fingerprint: string; readonly manifest_sha256: string; readonly created_at: string; readonly run_id: string | null } | null;
  readonly m11: { readonly git_commit: string; readonly source_fingerprint: string; readonly manifest_sha256: string; readonly created_at: string; readonly run_id: string | null } | null;
  readonly error: string | null;
}

export interface OverviewResponse {
  readonly evaluation_scope: string;
  readonly observation_cutoff: string;
  readonly summary: GlobalKpis;
  readonly tr_ru_total_usd: string | null;
  readonly ru_tr_total_usd: string | null;
  readonly bilateral_total_usd: string | null;
  readonly turkey_trade_balance_usd: string | null;
}

export interface MethodologyResponse {
  readonly specification_version: string;
  readonly evaluation_scope: string;
  readonly forecast_kind: string;
  readonly target_policy: readonly string[];
  readonly limitations: readonly string[];
  readonly unit: string;
  readonly monetary_serialization: string;
  readonly interval_coverage_unit: string;
  readonly m10_run_digest: string;
  readonly m11_run_digest: string;
}

export interface SnapshotManifest {
  readonly schema_version: 1;
  readonly snapshot_file: 'snapshot.json';
  readonly snapshot_sha256: string;
  readonly scope: string;
  readonly health_timestamp: string;
}

export interface DashboardData {
  readonly health: HealthResponse;
  readonly sectors: SectorCatalog;
  readonly leaderboard: LeaderboardResponse;
  readonly forecasts: Readonly<Record<string, ForecastResponse>>;
  readonly explanations: Readonly<Record<string, ExplanationResponse>>;
  readonly mirror: MirrorResponse;
  readonly overview: OverviewResponse;
  readonly methodology: MethodologyResponse;
}
