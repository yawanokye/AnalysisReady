# StatReady AI, Phase 2

StatReady AI is a transparent statistical analysis and reporting application for CSV and Excel data. It connects research objectives, hypotheses, conceptual-framework roles and dataset structure to statistical methods, diagnostics, defensible responses and reproducible exports.

## Phase 1 foundation

- CSV, XLSX and XLS upload with Excel sheet selection
- Preserved original dataset and separate analysis copy
- Dataset profile, missingness, duplicates and outlier screening
- Structured conceptual-framework and variable-role editor
- Rule-based method recommendation from objectives and hypotheses
- Automatic descriptive statistics for every inferential analysis
- Cronbach's alpha, correlations, t-tests, ANOVA, chi-square and Fisher's exact test
- OLS with HC3 inference and automatic ridge sensitivity for material VIF
- Binary logistic regression
- Basic moderation and bootstrap mediation
- Complete treatment and analysis audit trail
- DOCX, Excel, CSV, code and reproducibility ZIP exports

## Phase 2 methods

### Measurement and latent-variable models

- Exploratory factor analysis
  - KMO overall and item-level diagnostics
  - Bartlett test of sphericity
  - parallel analysis for factor retention
  - maximum-likelihood factor extraction
  - optional varimax rotation
  - communalities, uniqueness, variance explained and factor scores
- Confirmatory factor analysis
  - explicit construct-to-item specification
  - maximum-likelihood covariance fitting
  - CFI, TLI, RMSEA, SRMR and model chi-square
  - standardised loadings and item R-squared
  - composite reliability and average variance extracted
  - latent-factor correlations and residual-correlation review
- Covariance-based structural equation modelling
  - explicit construct measurement specification
  - directed acyclic structural paths
  - simultaneous measurement and structural covariance fitting
  - structural path estimates, fit indices and latent covariance matrix
  - numerical standard-error sensitivity output

### Longitudinal and clustered analysis

- Repeated-measures ANOVA
  - within-subject descriptives
  - approximate Mauchly sphericity assessment
  - Greenhouse-Geisser correction when required
  - Holm-adjusted paired comparisons and Cohen's dz
- Linear mixed-effects models
  - random intercept
  - optional random slope
  - fixed effects, variance components and ICC
  - cluster-size and convergence diagnostics
- Panel-data models
  - pooled OLS with entity-clustered standard errors
  - entity fixed effects
  - random effects
  - entity-effects F test
  - Hausman specification comparison
  - automatic or user-selected model
  - optional time fixed effects

### Advanced conditional-process analysis

- Advanced moderation
  - HC3 robust interaction model
  - simple slopes at the moderator mean and plus or minus one standard deviation
  - Johnson-Neyman boundaries
- Parallel multiple mediation
  - mediator-specific indirect effects
  - total indirect effect
  - bootstrap confidence intervals
- First-stage moderated mediation
  - conditional indirect effects at three moderator levels
  - index of moderated mediation
  - bootstrap confidence intervals

## Integrity safeguards

The app does not change data to produce statistical significance. It:

1. Preserves the uploaded dataset as the original copy.
2. Applies treatments only to a separate analysis copy.
3. Records every treatment, reason, affected variable and sample-size effect.
4. Uses robust, corrected or alternative inference when diagnostics require it.
5. Keeps primary and sensitivity results visible.
6. Does not automatically delete indicators, add SEM paths, correlate errors or remove observations to improve fit.
7. Warns when advanced estimates require confirmation in a specialist package.

## Construct specification syntax

For CFA and SEM, enter one construct per line:

```text
DigitalCompetence: dc1, dc2, dc3, dc4
TeachingEffectiveness: te1, te2, te3, te4
InstitutionalSupport: is1, is2, is3
```

For SEM, enter one directed path per line:

```text
DigitalCompetence -> TeachingEffectiveness
InstitutionalSupport -> TeachingEffectiveness
```

Phase 2 supports acyclic structural models. Every observed item can load on only one construct in the current implementation.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pip install pytest
PYTHONPATH=. pytest -q
```

The included suite tests Phase 1 methods, automatic descriptives, exports, EFA, CFA, SEM, repeated measures, mixed effects, panel selection, advanced moderation, parallel mediation and moderated mediation.

## Deploy to Render

1. Push the extracted project folder to a private GitHub repository.
2. Keep `Dockerfile`, `render.yaml`, `requirements.txt` and `app.py` at the repository root.
3. In Render, select **New → Blueprint**.
4. Connect the repository.
5. Render will create the Docker web service from `render.yaml`.

Current Blueprint defaults:

- service name: `statready-ai`
- region: Frankfurt
- plan: Starter
- health check: `/_stcore/health`
- automatic deployment after each committed update

Use Standard or higher for larger files, CFA/SEM, mixed models or intensive bootstrap analyses.

## Suggested demonstrations

### Phase 1 sample

Use `sample_data/sample_research_data.csv`.

- OLS: `performance` on `training`, `motivation`, `support`
- ANOVA: `performance` by `group`
- logistic regression: `passed` on `training`, `motivation`, `support`

### Phase 2 factor sample

Use `sample_data/phase2_factor_sample.csv`.

EFA items:

```text
a1, a2, a3, a4, b1, b2, b3, b4
```

CFA constructs:

```text
ConstructA: a1, a2, a3, a4
ConstructB: b1, b2, b3, b4
```

SEM path:

```text
ConstructA -> ConstructB
```

### Phase 2 longitudinal sample

Use `sample_data/phase2_longitudinal_sample.csv`.

- mixed-effects outcome: `y`
- fixed predictors: `x`, `time`
- cluster: `entity`
- optional random slope: `time`
- panel entity: `entity`
- panel time: `time`

## SEM and CFA diagrams

CFA and SEM analyses automatically generate a fitted diagram. The app displays the diagram above the inferential tables and provides a separate PNG download. The diagram is also embedded in the DOCX report and Excel workbook, and included as a PNG in the reproducibility package.

The current SEM module is a latent-variable structural model. A separate observed-variable path-analysis estimator is not yet exposed as its own menu item.

## Project structure

```text
app.py                         Streamlit user interface
statready/dispatch.py          Analysis routing and automatic descriptives
statready/methods.py           Phase 1 statistical methods
statready/phase2.py            Factor, SEM, longitudinal, panel and conditional-process methods
statready/figures.py           CFA measurement and SEM path-diagram generation
statready/diagnostics.py       Assumption and diagnostic tests
statready/treatments.py        Logged data treatments
statready/profiling.py         Dataset screening
statready/recommender.py       Objective-to-method rules
statready/reports.py           DOCX, Excel and ZIP exports
statready/literature.py        Curated methodological sources
sample_data/                   Phase 1 and Phase 2 demonstration datasets
tests/                         Engine and export tests
```

## Important limitations

- CFA and SEM are implemented with an internal maximum-likelihood covariance engine. Complex models, correlated residuals, categorical indicators, multigroup invariance and publication-critical estimates should be confirmed in specialist SEM software.
- SEM numerical standard errors are approximate and are clearly labelled as such.
- The mixed-effects implementation does not yet provide small-cluster degrees-of-freedom corrections.
- The panel module does not yet include dynamic panel GMM, cointegration or cross-sectional-dependence estimators.
- Multiple imputation, survey weights, ordinal CFA, PLS-SEM and time-series methods remain later development items.
- Uploaded projects remain session-based until authentication, database and object storage are added.
