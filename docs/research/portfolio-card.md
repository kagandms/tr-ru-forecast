# Portfolio Card

## TR–RU Trade Forecasting and Mirror Statistics

**A three-language decision-support dashboard for monthly Türkiye–Russia trade forecasting.**

This research platform compares statistical and machine-learning forecasts for 14 monthly trade series across 1-, 3- and 6-month horizons. It preserves reporter-level mirror discrepancies, selects models using development-only rolling-origin evaluation, and evaluates the frozen assignments on a separate 2025 holdout.

**Verified result:** governed operational forecasts recorded 26.17% mean sMAPE over 504 task-month cells, compared with 37.50% for Seasonal Naive. The 2025 evaluation is retrospective latest-vintage under deterministic lag rules; it is not proof of historical publication-time availability.

**What it demonstrates:** time-series validation and leakage controls; artifact provenance with SHA-256 integrity; typed read-only FastAPI contracts; Next.js analytics and Russian/Turkish/English localization; local SHAP prediction explanations; honest display of fallback use, uncertainty and source gaps.

**Technology:** Python, pandas, statsmodels, XGBoost, SHAP, FastAPI, Pydantic, TypeScript and Next.js.

- **Live dashboard:** [frontend-ten-psi-12.vercel.app](https://frontend-ten-psi-12.vercel.app)
- **Source repository:** [github.com/kagandms/tr-ru-forecast](https://github.com/kagandms/tr-ru-forecast)
- **Research package:** [English working paper](working-paper-en.md), [Turkish summary](project-summary-tr.md), [Russian abstract](abstract-ru.md)
- **Evaluation details:** [M11 holdout acceptance](../M11_ACCEPTANCE_REPORT.md), [task-level result table](results/holdout-task-results.csv)

**Limits:** only the selected HS2 and total series are covered; the holdout is one year; SHAP is not causal evidence; XGBoost nominal 80% interval coverage was 67.59%; historical vintages were not archived. Macro-source rights review remains open as documented in the [source register](sources-and-licensing.md).
