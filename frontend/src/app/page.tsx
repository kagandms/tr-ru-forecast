'use client';

import { useEffect, useMemo, useState } from 'react';
import { loadDashboard, type LoadedDashboard } from '@/lib/api';
import { useLocale, type Locale } from '@/locales';
import type { Direction, ExplanationResponse, ForecastPoint, Horizon, MirrorPoint, ModelFamily, TaskComparison } from '@/types/api';
import styles from './dashboard.module.css';

type Panel = 'executive' | 'explorer' | 'model-lab' | 'explainability' | 'mirror';
type Band = '80' | '95' | 'off';
const HORIZONS: readonly Horizon[] = [1, 3, 6];
const PANELS: readonly { id: Panel; label: string }[] = [
  { id: 'executive', label: 'Executive' }, { id: 'explorer', label: 'Trade explorer' },
  { id: 'model-lab', label: 'Model lab' }, { id: 'explainability', label: 'Explainability' }, { id: 'mirror', label: 'Mirror monitor' },
];
const percent = (value: number | null, locale: 'tr' | 'en' | 'ru', digits = 2): string => value === null ? '—' : `${new Intl.NumberFormat(locale === 'tr' ? 'tr-TR' : locale === 'ru' ? 'ru-RU' : 'en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value)}%`;
const fixed = (value: number | null, formatNumber: (value: number, maximumFractionDigits?: number) => string): string => value === null ? '—' : formatNumber(value);
const operationalSkill = (task: TaskComparison): number | null => {
  const actual = task.operational.mean_smape;
  const baseline = task.baseline.mean_smape;
  if (actual === null || baseline === null || baseline === 0) return null;
  return ((baseline - actual) / baseline) * 100;
};
const relativeReduction = (actual: number | null, baseline: number | null): number | null => {
  if (actual === null || baseline === null || baseline === 0) return null;
  return ((baseline - actual) / baseline) * 100;
};

function sectorTitle(sectorId: string, title: string, locale: Locale, translate: (key: string) => string): string {
  if (sectorId === 'RU_TR:TOTAL') {
    if (locale === 'tr') return 'Toplam İthalat — Rusya → Türkiye';
    if (locale === 'ru') return 'Общий импорт — Россия → Турция';
    return 'Total Imports — Russia → Türkiye';
  }
  if (sectorId === 'TR_RU:TOTAL') {
    if (locale === 'tr') return 'Toplam İhracat — Türkiye → Rusya';
    if (locale === 'ru') return 'Общий экспорт — Турция → Россия';
    return 'Total Exports — Türkiye → Russia';
  }
  return translate(title);
}

function SeriesChart({ history, points, band, title, t, locale }: {
  readonly history: readonly { period: string; value: number }[];
  readonly points: readonly ForecastPoint[];
  readonly band: Band;
  readonly title: string;
  readonly t: (key: string) => string;
  readonly locale: Locale;
}) {
  const width = 920;
  const height = 300;
  const historical = history.slice(-42);
  const values = [...historical.map((item) => item.value), ...points.flatMap((item) => [item.actual ?? 0, item.operational_forecast ?? 0, item.champion_forecast ?? 0, item.seasonal_naive_forecast ?? 0, item.lower_95 ?? 0, item.upper_95 ?? 0])];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const x = (index: number, total: number): number => 38 + (index / Math.max(total - 1, 1)) * (width - 70);
  const y = (value: number): number => height - 28 - ((value - min) / spread) * (height - 54);
  const polyline = (valuesToPlot: readonly number[], start: number, total: number): string => valuesToPlot.map((value, index) => `${x(start + index, total)},${y(value)}`).join(' ');
  const combinedLength = historical.length + points.length;
  const actual = points.map((point) => point.actual ?? 0);
  const operational = points.map((point) => point.operational_forecast ?? 0);
  const champion = points.map((point) => point.champion_forecast ?? 0);
  const naive = points.map((point) => point.seasonal_naive_forecast ?? 0);
  const lowKey = band === '80' ? 'lower_80' : 'lower_95';
  const highKey = band === '80' ? 'upper_80' : 'upper_95';
  const low = points.map((point) => point[lowKey] ?? 0);
  const high = points.map((point) => point[highKey] ?? 0);
  const bandPolygon = `${high.map((value, index) => `${x(historical.length + index, combinedLength)},${y(value)}`).join(' ')} ${low.map((value, index) => `${x(historical.length + low.length - 1 - index, combinedLength)},${y(low[low.length - 1 - index] ?? 0)}`).join(' ')}`;
  return <div className={styles.chartWrap}>
    <div className={styles.chartLegend}><span className={styles.legendActual}>{t('Actual')}</span><span className={styles.legendOperational}>{t('Operational')}</span><span className={styles.legendChampion}>{t('Champion raw')}</span><span className={styles.legendNaive}>{t('Seasonal naive')}</span>{band !== 'off' && <span className={styles.legendBand}>{band}{locale === 'tr' ? '% aralık' : locale === 'ru' ? '% интервал' : '% interval'}</span>}</div>
    <svg className={styles.chart} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
      {[0, 1, 2, 3].map((line) => <line key={line} x1="38" x2={width - 32} y1={48 + line * 65} y2={48 + line * 65} className={styles.gridLine} />)}
      {band !== 'off' && <polygon points={bandPolygon} className={styles.intervalBand} />}
      <line x1={x(historical.length, combinedLength)} x2={x(historical.length, combinedLength)} y1="18" y2={height - 26} className={styles.holdoutLine} />
      <line x1={x(Math.max(0, historical.findIndex((item) => item.period >= '2022-02')), combinedLength)} x2={x(Math.max(0, historical.findIndex((item) => item.period >= '2022-02')), combinedLength)} y1="18" y2={height - 26} className={styles.breakLine} />
      <polyline points={polyline(historical.map((item) => item.value), 0, combinedLength)} className={styles.historicalLine} />
      <polyline points={polyline(actual, historical.length, combinedLength)} className={styles.actualLine} />
      <polyline points={polyline(operational, historical.length, combinedLength)} className={styles.operationalLine} />
      <polyline points={polyline(champion, historical.length, combinedLength)} className={styles.championLine} />
      <polyline points={polyline(naive, historical.length, combinedLength)} className={styles.naiveLine} />
      {points.map((point, index) => {
        const localeTag = locale === 'tr' ? 'tr-TR' : locale === 'ru' ? 'ru-RU' : 'en-US';
        const formatTooltipValue = (value: number | null): string => value === null ? '—' : new Intl.NumberFormat(localeTag, { maximumFractionDigits: 0 }).format(value);
        const formatTooltipDate = new Intl.DateTimeFormat(localeTag, { year: 'numeric', month: 'short', timeZone: 'UTC' }).format(new Date(`${point.target_date.slice(0, 10)}T00:00:00Z`));
        return <circle key={point.target_date} cx={x(historical.length + index, combinedLength)} cy={y(point.operational_forecast ?? 0)} r="4" className={styles.forecastPoint}><title>{`${formatTooltipDate} · ${t('Actual')}: ${formatTooltipValue(point.actual)} · ${t('Operational')}: ${formatTooltipValue(point.operational_forecast)} · ${t('Champion raw')}: ${formatTooltipValue(point.champion_forecast)} · ${t('80% interval')}: ${formatTooltipValue(point.lower_80)}–${formatTooltipValue(point.upper_80)} · ${t('95% interval')}: ${formatTooltipValue(point.lower_95)}–${formatTooltipValue(point.upper_95)}`}</title></circle>;
      })}
      <text x={width - 87} y={height - 7} className={styles.axisLabel}>{t('2025 HOLDOUT')}</text>
      <text x="40" y="16" className={styles.axisLabel}>{t('USD · MONTHLY')}</text>
    </svg>
  </div>;
}

function MetricCard({ label, value, detail }: { readonly label: string; readonly value: string; readonly detail: string }) {
  return <article className={styles.metricCard}><span className={styles.metricLabel}>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function MirrorTrend({ observations, sectorId, t, locale }: { readonly observations: readonly MirrorPoint[]; readonly sectorId: string | null; readonly t: (key: string) => string; readonly locale: Locale }) {
  const monthly = new Map<string, number[]>();
  observations.filter((point) => sectorId === null || point.sector_id === sectorId).forEach((point) => {
    if (point.abs_mirror_gap_pct === null) return;
    const existing = monthly.get(point.period) ?? [];
    existing.push(point.abs_mirror_gap_pct);
    monthly.set(point.period, existing);
  });
  const values = Array.from(monthly, ([month, entries]) => ({ month, value: entries.reduce((sum, entry) => sum + entry, 0) / entries.length }));
  const formatMonth = (month: string | undefined): string => {
    if (!month) return '';
    return new Intl.DateTimeFormat(locale === 'tr' ? 'tr-TR' : locale === 'ru' ? 'ru-RU' : 'en-US', { year: 'numeric', month: 'short', timeZone: 'UTC' }).format(new Date(`${month}-01T00:00:00Z`));
  };
  if (!values.length) return <p className={styles.note}>{t('No paired monthly mirror gaps for this selection.')}</p>;
  const max = Math.max(...values.map((item) => item.value), 1);
  const points = values.map((item, index) => `${24 + (index / Math.max(values.length - 1, 1)) * 850},${155 - (item.value / max) * 125}`).join(' ');
  const breakIndex = values.findIndex((item) => item.month >= '2022-02');
  const breakX = breakIndex < 0 ? null : 24 + (breakIndex / Math.max(values.length - 1, 1)) * 850;
  return <div className={styles.trendWrap}><div className={styles.trendTitle}>{t('Monthly mean absolute reporter gap · %')}</div><svg viewBox="0 0 900 180" role="img" aria-label={t('Monthly mirror discrepancy trend')}><line x1="24" y1="155" x2="875" y2="155" className={styles.gridLine} />{breakX !== null && <line x1={breakX} x2={breakX} y1="16" y2="155" className={styles.breakLine} />}<polyline points={points} className={styles.mirrorLine} /><text x="25" y="174" className={styles.axisLabel}>{formatMonth(values[0]?.month)}</text><text x="820" y="174" className={styles.axisLabel}>{formatMonth(values[values.length - 1]?.month)}</text></svg></div>;
}

export default function Home() {
  const { locale, setLocale, t, formatNumber } = useLocale();
  const [loaded, setLoaded] = useState<LoadedDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [panel, setPanel] = useState<Panel>('executive');
  const [direction, setDirection] = useState<'ALL' | Direction>('ALL');
  const [sectorId, setSectorId] = useState('RU_TR:10');
  const [horizon, setHorizon] = useState<Horizon>(3);
  const [targetMonth, setTargetMonth] = useState('2025-01');
  const [band, setBand] = useState<Band>('80');
  const [sortBy, setSortBy] = useState<'cv' | 'holdout' | 'family'>('holdout');
  const [candidateDirection, setCandidateDirection] = useState<'ALL' | Direction>('ALL');
  const [candidateHorizon, setCandidateHorizon] = useState<'ALL' | Horizon>('ALL');
  const [candidateFamily, setCandidateFamily] = useState<'ALL' | ModelFamily>('ALL');
  const [candidateEligibility, setCandidateEligibility] = useState('ALL');
  const [mirrorSector, setMirrorSector] = useState('ALL');

  useEffect(() => {
    let isActive = true;
    loadDashboard().then((result) => { if (isActive) setLoaded(result); }).catch((cause: unknown) => {
      if (isActive) setError(cause instanceof Error ? cause.message : 'Snapshot load failed');
    });
    return () => { isActive = false; };
  }, []);

  const data = loaded?.data;
  const percentValue = (value: number | null, digits = 2): string => percent(value, locale, digits);
  const fixedValue = (value: number | null): string => fixed(value, formatNumber);
  const formatDate = (value: string): string => {
    const date = new Date(`${value.slice(0, 10)}T00:00:00Z`);
    return new Intl.DateTimeFormat(locale === 'tr' ? 'tr-TR' : locale === 'ru' ? 'ru-RU' : 'en-US', { year: 'numeric', month: 'short', timeZone: 'UTC' }).format(date);
  };
  const selectedSector = data?.sectors.sectors.find((sector) => sector.sector_id === sectorId);
  const task = data?.leaderboard.tasks.find((item) => item.sector_id === sectorId && item.horizon === horizon);
  const xgboostHorizonSixWins = data?.leaderboard.tasks.filter((item) => item.horizon === 6 && item.champion_model === 'xgboost').length ?? 0;
  const forecast = data?.forecasts[sectorId];
  const forecastTask = forecast?.tasks.find((item) => item.horizon === horizon);
  const explanation: ExplanationResponse | undefined = data?.explanations[`${sectorId}|${horizon}`];
  const localExplanation = explanation?.status === 'available' ? explanation.local.find((item) => item.target_month === targetMonth) : undefined;
  const filteredSectors = useMemo(() => data?.sectors.sectors.filter((sector) => direction === 'ALL' || sector.direction === direction) ?? [], [data, direction]);
  const comparisons = useMemo(() => {
    if (!data) return [];
    const rows = data.leaderboard.tasks.flatMap((item) => item.cv_candidates
      .filter((candidate) => candidateDirection === 'ALL' || item.direction === candidateDirection)
      .filter((candidate) => candidateHorizon === 'ALL' || item.horizon === candidateHorizon)
      .filter((candidate) => candidateFamily === 'ALL' || candidate.model_family === candidateFamily)
      .filter((candidate) => candidateEligibility === 'ALL' || candidate.eligibility_status === candidateEligibility)
      .map((candidate) => ({ task: item, candidate })));
    return rows.sort((left, right) => {
      if (sortBy === 'family') return left.candidate.model_family.localeCompare(right.candidate.model_family);
      const leftValue = sortBy === 'cv' ? left.candidate.mean_smape : left.task.operational.mean_smape;
      const rightValue = sortBy === 'cv' ? right.candidate.mean_smape : right.task.operational.mean_smape;
      return (leftValue ?? Number.POSITIVE_INFINITY) - (rightValue ?? Number.POSITIVE_INFINITY);
    });
  }, [candidateDirection, candidateEligibility, candidateFamily, candidateHorizon, data, sortBy]);
  const historical = forecast?.history.flatMap((point) => point.period >= '2025-01' || point.trade_value_usd === null
    ? []
    : [{ period: point.period, value: Number(point.trade_value_usd) }]).slice(-42) ?? [];

  function chooseTask(nextSector: string, nextHorizon: Horizon): void {
    setSectorId(nextSector);
    setHorizon(nextHorizon);
    setPanel('explorer');
  }

  if (error) return <main className={styles.loading}><b>{t('Snapshot unavailable')}</b><p>{error}</p></main>;
  if (!data || !loaded) return <main className={styles.loading}><div className={styles.brandMark}>TR/RU</div><p>{t('Loading verified evaluation artifacts…')}</p></main>;

  return <main className={styles.shell}>
    <header className={styles.topbar}>
      <div className={styles.brand}><div className={styles.brandMark}>TR/RU</div><div><p className={styles.eyebrow}>{t('BILATERAL TRADE INTELLIGENCE')}</p><h1>{t('Decision support terminal')}</h1></div></div>
      <div className={styles.headerActions}><div className={styles.provenance}><span className={loaded.source === 'LIVE_API' ? styles.liveDot : styles.snapshotDot} />{loaded.source === 'LIVE_API' ? (locale === 'tr' ? 'CANLI API' : locale === 'ru' ? 'АКТИВНЫЙ API' : 'LIVE API') : (locale === 'tr' ? 'ANLIK GÖRÜNTÜ' : locale === 'ru' ? 'СНИМОК' : 'SNAPSHOT')}<span className={styles.hash}>SHA-256 {loaded.sha256.slice(0, 12)}…</span></div><div className={styles.localeSwitch} role="group" aria-label={t('Language selector')}><button type="button" aria-pressed={locale === 'tr'} onClick={() => setLocale('tr')}>TR</button><button type="button" aria-pressed={locale === 'en'} onClick={() => setLocale('en')}>EN</button><button type="button" aria-pressed={locale === 'ru'} onClick={() => setLocale('ru')}>RU</button></div></div>
    </header>
    <div className={styles.scopeLine}><span>{t('RETROSPECTIVE LATEST-VINTAGE HOLDOUT · DETERMINISTIC LAG RULES (v1.1)')}</span><span>{t('M12 BASELINE · 511 TESTS PASSED · 102 ARTIFACTS SHA-256 VERIFIED')}</span></div>
    <nav className={styles.tabs} aria-label={t('Analytics modules')} role="tablist">{PANELS.map((item) => <button key={item.id} role="tab" aria-selected={panel === item.id} className={panel === item.id ? styles.activeTab : ''} onClick={() => setPanel(item.id)}>{t(item.label)}</button>)}</nav>

    <section className={styles.kpiGrid} aria-label={t('Holdout key performance indicators')}>
      <MetricCard label={t('HOLDOUT OPERATIONAL sMAPE')} value={percentValue(data.leaderboard.summary.operational_mean_smape)} detail={`${t('Seasonal naive')} ${percentValue(data.leaderboard.summary.baseline_mean_smape)} · ${percentValue(relativeReduction(data.leaderboard.summary.operational_mean_smape, data.leaderboard.summary.baseline_mean_smape))} ${t('relative reduction')}`} />
      <MetricCard label={t('CV MEAN SKILL VS NAIVE')} value={percentValue(data.leaderboard.summary.cv_mean_skill_pct)} detail={t('281 rolling origins · 15,736 candidate rows')} />
      <MetricCard label={t('CHAMPION DISTRIBUTION')} value={`${data.leaderboard.summary.sarima_champions} / ${data.leaderboard.summary.xgboost_champions}`} detail={`${t('SARIMA / XGBoost')} · ${xgboostHorizonSixWins} ${t('of')} ${data.leaderboard.summary.xgboost_champions} ${t('XGBoost wins at h=6')}`} />
      <MetricCard label={t('HOLDOUT EVIDENCE')} value={formatNumber(1512)} detail={`${t('Role forecasts')} · ${formatNumber(9072)} ${t('local SHAP rows')}`} />
    </section>

    {panel === 'executive' && <section className={styles.twoColumn}>
      <article className={styles.panel}><div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('FROZEN EVALUATION')}</p><h2>{t('Model portfolio at a glance')}</h2></div><span className={styles.tag}>{t('42 TASKS')}</span></div>
        <div className={styles.portfolio}><div><b>{data.leaderboard.summary.sarima_champions}</b><span>{t('SARIMA champions')}</span></div><div><b>{data.leaderboard.summary.xgboost_champions}</b><span>{t('XGBoost champions')}</span></div><div><b>{data.leaderboard.summary.fallback_replacements}</b><span>{t('Governed replacement')}</span></div></div>
        <p className={styles.note}>{t('Forecasts cover Jan–Dec 2025. Values are retrospective latest-vintage holdout estimates under deterministic trade and macro lag rules.')}</p>
      </article>
      <article className={styles.panel}><div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('DATA PROVENANCE')}</p><h2>{t('Verified serving scope')}</h2></div><span className={styles.verified}>{t('● VERIFIED')}</span></div>
        <dl className={styles.provenanceList}><div><dt>{t('Evaluation cutoff')}</dt><dd>{formatDate(`${data.overview.observation_cutoff}-01`)}</dd></div><div><dt>{t('M10 selection digest')}</dt><dd>{data.methodology.m10_run_digest.slice(0, 16)}…</dd></div><div><dt>{t('M11 holdout digest')}</dt><dd>{data.methodology.m11_run_digest.slice(0, 16)}…</dd></div><div><dt>{t('Snapshot SHA-256')}</dt><dd>{loaded.sha256}</dd></div></dl>
      </article>
    </section>}

    {panel === 'explorer' && <section className={styles.panel}>
      <div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('MODULE B · TRADE & FORECAST EXPLORER')}</p><h2>{t('Historical context and governed holdout')}</h2></div><span className={styles.tag}>{t('USD · MONTHLY')}</span></div>
      <div className={styles.guideStrip}><b>{t('Forecast read guide')}</b><span>{t('1 · Dark line: observed value')}</span><span>{t('2 · Teal: operational forecast; fallback use is flagged in the table')}</span><span>{t('3 · Shading: selected conformal prediction interval')}</span></div>
      <div className={styles.controls}>
        <label>{t('Direction')}<select value={direction} onChange={(event) => { const next = event.target.value as 'ALL' | Direction; setDirection(next); const first = data.sectors.sectors.find((sector) => next === 'ALL' || sector.direction === next); if (first) setSectorId(first.sector_id); }}><option value="ALL">{t('All directions')}</option><option value="RU_TR">{t('RU_TR · Imports from Russia')}</option><option value="TR_RU">{t('TR_RU · Exports to Russia')}</option></select></label>
        <label>{t('Series')}<select value={sectorId} onChange={(event) => setSectorId(event.target.value)}>{filteredSectors.map((sector) => <option key={sector.sector_id} value={sector.sector_id}>{sector.sector_id} · {sectorTitle(sector.sector_id, sector.title, locale, t)}</option>)}</select></label>
        <label>{t('Horizon')}<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value) as Horizon)}>{HORIZONS.map((value) => <option key={value} value={value}>{locale === 'tr' ? `h = ${value} ay` : locale === 'ru' ? `h = ${value} мес.` : `h = ${value} month${value > 1 ? 's' : ''}`}</option>)}</select></label>
        <label>{t('Interval')}<select value={band} onChange={(event) => setBand(event.target.value as Band)}><option value="80">{t('80% band')}</option><option value="95">{t('95% band')}</option><option value="off">{t('Off')}</option></select></label>
      </div>
      <div className={styles.chartPanel}><SeriesChart history={historical} points={forecastTask?.points ?? []} band={band} title={`${sectorId} ${horizon} ${t('Horizon forecast')}`} t={t} locale={locale} /></div>
      <div className={styles.chartAnnotations}><span>│ {t('2022-02 structural break')}</span><span>│ {t('2025-01 → 2025-12 holdout')}</span></div>
      {task && <div className={styles.scorecard}><div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('TASK SCORECARD')} · {selectedSector ? (sectorTitle(selectedSector.sector_id, selectedSector.title, locale, t)) : ''}</p><h3>{task.champion_model.toUpperCase()} {t('champion')} <span className={styles.muted}>· {t('fallback')} {task.fallback_model.toUpperCase()}</span></h3></div><span className={styles.tag}>{t(task.selection_reason)}</span></div>
          <div className={styles.scoreGrid}><span>{t('CV sMAPE')} <b>{percentValue(task.cv.mean_smape)}</b></span><span>{t('Holdout sMAPE')} <b>{percentValue(task.operational.mean_smape)}</b></span><span>{t('MAE')} <b>{fixedValue(task.operational.mean_mae)}</b></span><span>{t('RMSE')} <b>{fixedValue(task.operational.mean_rmse)}</b></span><span>{t('Skill vs naive')} <b>{percentValue(operationalSkill(task))}</b></span><span>{t('80% coverage')} <b>{percentValue(task.intervals.empirical_coverage_80 === null ? null : task.intervals.empirical_coverage_80 * 100)}</b></span><span>{t('95% coverage')} <b>{percentValue(task.intervals.empirical_coverage_95 === null ? null : task.intervals.empirical_coverage_95 * 100)}</b></span><span>{t('Interval policy')} <b>{forecastTask?.points.some((point) => point.interval_policy === 'prequential_conformal') ? 'prequential_conformal_80_95' : 'native_80_95'}</b></span><span>{t('Calibration pool')} <b>{forecastTask?.intervals.calibration_pool_size ?? 0}</b></span></div>
      </div>}
      <div className={styles.tableWrap}><table><caption>{t('2025 diagnostic points · raw champion retained alongside operational forecast')}</caption><thead><tr><th>{t('Month')}</th><th>{t('Origin')}</th><th>{t('Actual')}</th><th>{t('Operational')}</th><th>{t('Champion raw')}</th><th>{t('Fallback')}</th><th>{t('Naive')}</th><th>{t('80% interval')}</th><th>{t('95% interval')}</th><th>{t('Status')}</th></tr></thead><tbody>{forecastTask?.points.map((point) => <tr key={point.target_date} className={!point.champion_is_valid ? styles.invalidRow : ''}><td>{formatDate(point.target_date)}</td><td>{formatDate(point.origin_date)}</td><td>{fixedValue(point.actual)}</td><td>{fixedValue(point.operational_forecast)}{point.used_fallback && <b className={styles.alert}> {t('FALLBACK USED')}</b>}</td><td>{fixedValue(point.champion_forecast)}{!point.champion_is_valid && <b className={styles.alert}> {t('INVALID')}</b>}</td><td>{fixedValue(point.fallback_forecast)}</td><td>{fixedValue(point.seasonal_naive_forecast)}</td><td>{fixedValue(point.lower_80)} – {fixedValue(point.upper_80)}</td><td>{fixedValue(point.lower_95)} – {fixedValue(point.upper_95)}</td><td>{t(point.status === 'available' ? 'Available' : 'Unavailable')}</td></tr>)}</tbody></table></div>
    </section>}

    {panel === 'model-lab' && <section className={styles.panel}>
      <div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('MODULE C · MODEL LAB')}</p><h2>{t('Champion matrix & candidate audit')}</h2></div><span className={styles.tag}>{t('168 CV CANDIDATES · 42 TASKS')}</span></div>
      <div className={styles.matrixWrap}><table className={styles.matrix}><thead><tr><th>{t('Trade series')}</th>{HORIZONS.map((item) => <th key={item}>h={item}</th>)}</tr></thead><tbody>{data.sectors.sectors.map((sector) => <tr key={sector.sector_id}><th>{sector.sector_id}<small>{sectorTitle(sector.sector_id, sector.title, locale, t)}</small></th>{HORIZONS.map((item) => { const row = data.leaderboard.tasks.find((candidate) => candidate.sector_id === sector.sector_id && candidate.horizon === item); return <td key={item}>{row && <button className={styles.matrixCell} onClick={() => chooseTask(sector.sector_id, item)}><b className={row.champion_model === 'xgboost' ? styles.xgbBadge : styles.sarimaBadge}>{row.champion_model.toUpperCase()}</b><span>{percentValue(row.operational.mean_smape)}</span><small>{t('skill')} {percentValue(operationalSkill(row))}</small>{row.selection_reason === 'PARSIMONY_PREFERENCE' && <em>{t('PARSIMONY')}</em>}{row.has_stability_flag && <em>{t('STABILITY FLAG')}</em>}{row.fallback_replacements > 0 && <em>{t('FALLBACK USED')}</em>}</button>}</td>; })}</tr>)}</tbody></table></div>
      <div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('CROSS-TASK CANDIDATE LEADERBOARD')}</p><h3>{t('Eligibility and comparative error')}</h3></div><label className={styles.sortControl}>{t('Sort')}<select value={sortBy} onChange={(event) => setSortBy(event.target.value as typeof sortBy)}><option value="holdout">{t('Holdout sMAPE')}</option><option value="cv">{t('CV sMAPE')}</option><option value="family">{t('Model family')}</option></select></label></div>
      <div className={styles.controls}><label>{t('Direction')}<select value={candidateDirection} onChange={(event) => setCandidateDirection(event.target.value as 'ALL' | Direction)}><option value="ALL">{t('All directions')}</option><option value="RU_TR">RU_TR</option><option value="TR_RU">TR_RU</option></select></label><label>{t('Horizon')}<select value={candidateHorizon} onChange={(event) => setCandidateHorizon(event.target.value === 'ALL' ? 'ALL' : Number(event.target.value) as Horizon)}><option value="ALL">{t('All horizons')}</option>{HORIZONS.map((value) => <option key={value} value={value}>h={value}</option>)}</select></label><label>{t('Family')}<select value={candidateFamily} onChange={(event) => setCandidateFamily(event.target.value as typeof candidateFamily)}><option value="ALL">{t('All families')}</option><option value="sarima">SARIMA</option><option value="sarimax">SARIMAX</option><option value="xgboost">XGBoost</option><option value="seasonal_naive">{t('Seasonal naive')}</option></select></label><label>{t('Eligibility')}<select value={candidateEligibility} onChange={(event) => setCandidateEligibility(event.target.value)}><option value="ALL">{t('All statuses')}</option>{Array.from(new Set(data.leaderboard.tasks.flatMap((row) => row.cv_candidates.map((candidate) => candidate.eligibility_status)))).map((status) => <option key={status} value={status}>{t(status)}</option>)}</select></label></div>
      <div className={styles.tableWrap}><table><thead><tr><th>{t('Direction · HS2')}</th><th>h</th><th>{t('Family')}</th><th>{t('CV sMAPE')}</th><th>{t('Holdout task sMAPE')}</th><th>{t('Eligibility')}</th><th>{t('Negative CV')}</th></tr></thead><tbody>{comparisons.map(({ task: row, candidate }, index) => <tr key={`${row.sector_id}-${row.horizon}-${candidate.model_family}-${index}`}><td>{row.direction} · {row.hs2}</td><td>{row.horizon}</td><td>{candidate.model_family === 'seasonal_naive' ? t('Seasonal naive') : candidate.model_family.toUpperCase()}</td><td>{percentValue(candidate.mean_smape)}</td><td>{percentValue(row.operational.mean_smape)}</td><td><span className={candidate.eligibility_status.toUpperCase() === 'ELIGIBLE' ? styles.eligible : styles.ineligible}>{t(candidate.eligibility_status)}</span></td><td>{candidate.n_negative_predictions}</td></tr>)}</tbody></table></div>
      {comparisons.some(({ candidate }) => candidate.n_negative_predictions > 0) && <div className={styles.notice}>{t('Excluded due to negative forecast')}: {formatNumber(comparisons.filter(({ candidate }) => candidate.n_negative_predictions > 0).length)} {t('candidates were ineligible because at least one CV forecast was negative.')}</div>}
      <p className={styles.note}>{t('Eligibility counts are model-level rolling-origin audit values. Candidate holdout sMAPE is the selected task’s operational result, not a separately evaluated holdout score for every challenger.')}</p>
    </section>}

    {panel === 'explainability' && <section className={styles.panel}>
      <div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('MODULE D · EXPLAINABILITY INSPECTOR')}</p><h2>{t('Predictive attribution, not causality')}</h2></div><span className={explanation?.status === 'available' ? styles.verified : styles.tag}>{explanation?.status === 'available' ? `${t(explanation.role === 'champion' ? 'Assigned Champion' : 'Fallback-Challenger')} · XGBOOST` : 'XGBOOST ROLE UNAVAILABLE'}</span></div>
      <div className={styles.controls}><label>{t('Task series')}<select value={sectorId} onChange={(event) => setSectorId(event.target.value)}>{data.sectors.sectors.map((sector) => <option key={sector.sector_id} value={sector.sector_id}>{sector.sector_id} · {sectorTitle(sector.sector_id, sector.title, locale, t)}</option>)}</select></label><label>{t('Horizon')}<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value) as Horizon)}>{HORIZONS.map((value) => <option key={value} value={value}>h = {value}</option>)}</select></label><label>{t('Target month')}<select value={targetMonth} onChange={(event) => setTargetMonth(event.target.value)}>{Array.from({ length: 12 }, (_, index) => `2025-${String(index + 1).padStart(2, '0')}`).map((month) => <option key={month} value={month}>{formatDate(`${month}-01`)}</option>)}</select></label></div>
      {explanation?.status === 'available' ? <div className={styles.twoColumn}><article className={styles.subpanel}><h3>{t('Global mean |SHAP| · 12 holdout origins')}</h3><div className={styles.familyLegend}><span>{t('Trade Lag')}</span><span>{t('Rolling Stat')}</span><span>{t('Seasonality & Calendar')}</span><span>{t('Macro Indicator')}</span></div><div className={styles.bars}>{explanation.global_importance.slice(0, 21).map((feature) => <div className={styles.barRow} key={feature.feature}><span>{t(feature.feature)}</span><div><i style={{ width: `${Math.max(1, ((feature.mean_abs_shap ?? 0) / Math.max(...explanation.global_importance.map((entry) => entry.mean_abs_shap ?? 0))) * 100)}%` }} className={styles[`family_${feature.category}`]} /></div><b>{fixedValue(feature.mean_abs_shap)}</b></div>)}</div></article>
        <article className={styles.subpanel}><h3>{formatDate(`${targetMonth}-01`)} {t('local contributions')}</h3>{localExplanation ? <><p className={styles.note}>{t('Base')} {fixedValue(localExplanation.base_value)} → {t('prediction')} {fixedValue(localExplanation.prediction)}</p><div className={styles.bars}>{[...localExplanation.top_positive, ...localExplanation.top_negative].map((item) => <div className={styles.barRow} key={item.feature}><span>{t(item.feature)}</span><div className={styles.signedBar}><i style={{ width: `${Math.min(100, Math.abs(item.shap_value ?? 0) / Math.max(1, ...localExplanation.contributions.map((entry) => Math.abs(entry.shap_value ?? 0))) * 100)}%`, marginLeft: item.shap_value !== null && item.shap_value < 0 ? 'auto' : 0 }} className={item.shap_value !== null && item.shap_value < 0 ? styles.negativeBar : styles.positiveBar} /></div><b>{fixedValue(item.shap_value)}</b></div>)}</div></> : <p className={styles.note}>{t('No local explanation for this target month.')}</p>}</article></div> : <div className={styles.notice}>{t('This task has no XGBoost champion or fallback role in the frozen M10 assignment. No SHAP values are synthesized.')}</div>}
      <p className={styles.note}>{t('Attributions describe predictive contributions from trade lags, rolling statistics, calendar features and macro indicators under approximate release lags. They do not estimate causal effects.')}</p>
    </section>}

    {panel === 'mirror' && <section className={styles.panel}>
      <div className={styles.panelHeading}><div><p className={styles.eyebrow}>{t('MODULE E · MIRROR STATISTICS')}</p><h2>{t('Reporter asymmetry monitor')}</h2></div><span className={styles.tag}>{data.mirror.source} · {formatNumber(data.mirror.total_observations)} {t('OBSERVATIONS')}</span></div>
      <div className={styles.controls}><label>{t('Sector trend')}<select value={mirrorSector} onChange={(event) => setMirrorSector(event.target.value)}><option value="ALL">{t('All mirror sectors')}</option>{data.sectors.sectors.map((sector) => <option key={sector.sector_id} value={sector.sector_id}>{sector.sector_id} · {sectorTitle(sector.sector_id, sector.title, locale, t)}</option>)}</select></label></div>
      <MirrorTrend observations={data.mirror.observations} sectorId={mirrorSector === 'ALL' ? null : mirrorSector} t={t} locale={locale} />
      <div className={styles.mirrorStats}>{data.mirror.summaries.map((summary) => <article key={summary.period_group}><span>{summary.period_group === 'pre_2022' ? (locale === 'tr' ? '2022 ÖNCESİ' : locale === 'ru' ? 'ДО 2022 ГОДА' : 'PRE-2022') : (locale === 'tr' ? '2022 SONRASI' : locale === 'ru' ? 'ПОСЛЕ 2022 ГОДА' : 'POST-2022')}</span><b>{percentValue(summary.mean_abs_gap_pct)}</b><small>{formatNumber(summary.paired_count)} {t('paired')} / {formatNumber(summary.observation_count)} {t('observations')}</small></article>)}</div>
      <div className={styles.notice}>{data.mirror.structural_shift_status === 'unavailable_no_paired_post_2022' ? t('Post-2022 paired exporter/importer observations are unavailable. Structural shift is not estimated.') : `${t('Structural asymmetry shift')}: ${percentValue(data.mirror.structural_shift_pp)}`}</div>
      <p className={styles.note}>{locale === 'tr' ? 'Fark = ihracatçı bildirimi − ithalatçı bildirimi; yüzde = fark / ihracatçı değeri × 100 (M2); tahmin hedefi eksi referans değildir.' : locale === 'ru' ? 'Расхождение = отчет экспортера − отчет импортера; процент = расхождение / значение экспортера × 100 (M2); цель прогноза не является разностью отчетов.' : data.mirror.gap_definition} {locale === 'tr' ? 'İhracat FOB, ithalat CIF değerindedir; farklar bildirim ve sınıflandırma etkileri içerir. Nedensel navlun düzeltmesi yapılmamıştır. Bağımsız ulusal veri setleri mevcut değildir.' : locale === 'ru' ? 'Экспорт оценивается на условиях FOB, импорт — CIF; расхождения также отражают различия отчетности и классификации. Причинная поправка на стоимость перевозки не выполнялась. Независимые национальные наборы данных отсутствуют.' : data.mirror.valuation_note}</p>
      <div className={styles.mirrorPlot}>{data.mirror.summaries.map((summary) => <div key={summary.period_group}><span>{summary.period_group === 'pre_2022' ? (locale === 'tr' ? '2022 öncesi' : locale === 'ru' ? 'до 2022 года' : 'pre-2022') : (locale === 'tr' ? '2022 sonrası' : locale === 'ru' ? 'после 2022 года' : 'post-2022')}</span><div><i style={{ width: `${Math.min(100, summary.mean_abs_gap_pct ?? 0)}%` }} /></div><b>{percentValue(summary.mean_abs_gap_pct)}</b></div>)}</div>
      <p className={styles.note}>{t('Source coverage is the paired Türkiye and Russia reporter series distributed by UN Comtrade; independently acquired TurkStat and Rosstat feeds are not present. Mirror discrepancies may include CIF/FOB valuation differences and are not a freight-only estimate.')}</p>
    </section>}

    <footer className={styles.footer}><span>{t('Evaluation scope')}: {t(data.health.evaluation_scope)}</span><span>{t('Trade history and macro inputs follow deterministic lag/origin cutoffs · frozen retrospective artifact serving')}</span></footer>
  </main>;
}
