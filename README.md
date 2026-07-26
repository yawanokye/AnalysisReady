# StatReady AI, Phase 1

StatReady AI is a transparent statistical analysis and reporting application for CSV and Excel data. It links research objectives, hypotheses and conceptual-framework roles to statistical methods, diagnostics, defensible responses and reproducible exports.

## What Phase 1 includes

- CSV, XLSX and XLS upload
- Excel sheet selection
- Preserved original dataset and separate analysis copy
- Dataset profile, missingness, duplicate and outlier screening
- Structured conceptual-framework and variable-role editor
- Rule-based method recommendation from objectives and hypotheses
- Automatic descriptive statistics for every analysis, including sample overview, missingness, numeric summaries, categorical frequencies and grouped summaries where relevant
- Cronbach's alpha and item diagnostics
- Pearson and Spearman correlations
- Independent and paired t-tests
- One-way and Welch ANOVA
- Chi-square and Fisher's exact tests
- OLS regression with HC3 robust inference
- Binary logistic regression
- Moderation analysis
- Bootstrap mediation analysis
- Assumption and diagnostic summaries
- Complete audit trail for data treatments and analysis adjustments
- DOCX, Excel, cleaned CSV, code and full reproducibility ZIP exports
- Curated supporting methodological literature

## Integrity safeguards

The app does not change data to produce statistical significance. It:

1. Preserves the uploaded dataset as the original copy.
2. Applies treatments only to a separate analysis copy.
3. Records every treatment, reason, affected variable and sample-size effect.
4. Uses robust or alternative inference when assumptions fail.
5. Keeps the untreated and sensitivity results available for comparison.
6. Warns against automatic outlier deletion, unjustified category combination and significance-driven transformations.

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

Open the local address displayed by Streamlit.

## Run the tests

```bash
pip install pytest
pytest -q
```

## Deploy to Render

### Docker deployment

1. Push this folder to a GitHub repository.
2. In Render, create a new Blueprint and select the repository.
3. Render will use `render.yaml` and `Dockerfile`.
4. Deploy the service.

The Docker command uses Render's `PORT` variable automatically.

### Manual web service

- Runtime: Docker
- Health check: `/_stcore/health`
- Instance: Starter or higher for larger Excel files and bootstrap analyses

## Suggested first demonstration

Upload `sample_data/sample_research_data.csv` and try:

- Reliability: `item1`, `item2`, `item3`, `item4`
- ANOVA: outcome `performance`, group `group`
- OLS: outcome `performance`, predictors `training`, `motivation`, `support`
- Logistic regression: outcome `passed`, predictors `training`, `motivation`, `support`
- Moderation: outcome `performance`, predictor `training`, moderator `support`, control `motivation`
- Mediation: outcome `performance`, predictor `training`, mediator `motivation`, control `support`

## Production hardening before public release

- Add authentication, project ownership and role-based access.
- Store files in encrypted object storage rather than browser session memory.
- Add a background worker for large bootstraps and exports.
- Add row and file-size limits by subscription plan.
- Add automated deletion schedules and institutional data-processing terms.
- Add survey weights, clustered sampling and multilevel designs in later phases.
- Connect literature retrieval to Crossref/OpenAlex and verify it through CiteIntegrity.
- Add an optional LLM interpretation service, while retaining the deterministic statistical engine as the sole source of numerical results.

## Main project structure

```text
app.py                         Streamlit user interface
statready/dispatch.py          Analysis routing
statready/methods.py           Statistical methods
statready/diagnostics.py       Assumption and diagnostic tests
statready/treatments.py        Logged data treatments
statready/profiling.py         Dataset screening
statready/recommender.py       Objective-to-method rules
statready/reports.py           DOCX, Excel and ZIP exports
statready/literature.py        Curated methodological sources
sample_data/                   Demonstration dataset
tests/                         Core-engine and export tests
```

## Current limitation

Phase 1 is an MVP. It does not yet cover SEM, CFA, EFA, panel data, time series, mixed models, survey-weighted analysis, multiple imputation or advanced causal inference. It also does not infer a conceptual framework reliably from an uploaded image. The structured framework editor is used instead.


## Multicollinearity response

OLS models now calculate VIF with an intercept. When any VIF is 10 or higher, the app automatically runs a standardised cross-validated ridge regression as a documented sensitivity model, compares coefficient directions, records the action in the audit trail and warns against isolated interpretation of unstable OLS coefficients. No observations are altered or removed.


## Automatic descriptive statistics

Every inferential analysis now begins with descriptive statistics for the exact variables used. The output includes a sample overview, valid and missing counts, mean, standard deviation, median, quartiles, minimum, maximum, skewness and kurtosis for numeric variables. Categorical and low-cardinality variables receive frequency and percentage tables. Group comparison methods also receive descriptive statistics by group. Inferential analyses use the complete-case analytical sample so the descriptive and model sample sizes remain aligned. Users can also select additional demographic or profile variables, which are reported separately using available observations and do not enter or reduce the inferential model sample.
