# StatReady AI, Phase 2.4

StatReady AI is a transparent statistical analysis and reporting application for CSV and Excel data. It links research objectives, hypotheses, conceptual-framework roles and dataset structure to statistical methods, diagnostics, defensible responses and reproducible exports.

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

## Phase 2.2 foundation

### Structured construct measurement builder

CFA, CB-SEM and PLS-SEM no longer require free-text construct syntax.

For each construct, the user can:

1. Enter the construct name.
2. Select indicators from dropdown menus populated from numeric dataset columns.
3. Specify reflective or formative measurement for PLS-SEM.
4. Review duplicate-item and incomplete-construct validation before estimation.

CFA and covariance-based SEM currently estimate reflective common-factor constructs. PLS-SEM supports reflective Mode A and formative Mode B blocks.

### Structured relationship builder

Structural relationships are selected from entered construct names.

- **Direct:** predictor construct to outcome construct
- **Mediator:** predictor, mediator and outcome, with an optional direct path
- **Moderator:** predictor, moderator and outcome

Mediation specifications generate direct, indirect and total-effect tables. PLS-SEM estimates moderation through a two-stage latent-score interaction. The internal covariance-based SEM engine records latent moderation requests but directs them to PLS-SEM or the dedicated moderation module rather than silently omitting the interaction.

## Phase 2.3 additions

### AI Guided Mode

The new AI research agent helps novice and experienced users move from a study idea to a defensible analysis specification. The agent is transparent and confirmation-based. It does not alter data, delete observations, change estimators or run a final model without user approval.

The guided workflow provides:

- readiness checks for the objective, hypothesis, dataset, outcome type, conceptual framework and variable roles
- conservative variable-role suggestions with confidence levels and explanations
- method recommendations from study wording and design information
- dataset findings for rows, variables, missing cells and duplicate records
- construct-block suggestions from repeated item names such as `trust1`, `trust2` or `trust_1`, `trust_2`
- one-click loading of suggested construct blocks into CFA, CB-SEM or PLS-SEM for confirmation
- user-selectable Novice, Assisted and Expert co-pilot guidance levels
- explicit confirmation before suggested roles or construct specifications are applied

The current agent combines deterministic research-method rules and dataset diagnostics. This makes the guidance reproducible and keeps the app functional without an external language-model API. A provider-backed generative explanation layer can be added later without changing the statistical engine.

### Flexible path and measurement diagrams

CFA, CB-SEM and PLS-SEM diagrams now support:

- left-to-right layout
- top-to-bottom layout
- bottom-to-top layout
- radial layout
- hierarchical layout
- measurement-first layout
- structural-first layout
- compact publication layout
- straight or curved structural arrows
- optional custom construct ordering
- show or hide observed indicators, indicator names, factor loadings, path coefficients, p-values and model-fit indices
- optional significance colouring
- journal monochrome mode
- transparent-background PNG export
- standard or high-resolution rendering
- dashed moderation arrows directed to the focal structural path

The selected diagram settings are stored in the analysis plan, applied to the in-app figure and retained in DOCX, Excel and reproducibility-package exports.


## Phase 2.4 additions

### Persistent drag-and-click path editor

CFA, covariance-based SEM and PLS-SEM results now include an interactive path editor. Users can drag constructs directly, click a construct and nudge it with arrow controls, switch among automatic layouts, reset the model or save the arrangement. Saved node coordinates are applied to the publication-quality PNG and all newly generated DOCX, Excel and reproducibility-package exports. The arrangement can also be downloaded as JSON.

### Autonomous AI-completed analysis specification

The AI research agent now reads the objectives, hypotheses, framework narrative, confirmed variable roles and dataset structure to complete the method-specific inputs. It can identify outcomes, predictors, mediators, moderators, controls, groups, clusters, entity and time identifiers, repeated measures, construct blocks, structural paths and network construction rules.

The agent provides a complete specification table and a one-click guided analysis. It pauses only when a critical human decision cannot be inferred safely, such as an ambiguous outcome, unresolved construct direction, missing cluster identifier or unclear network node/edge meaning. Applying the specification also updates provisional variable roles and the framework narrative. Data deletion, transformations and final causal direction remain human-controlled.

### Comprehensive network analysis

Network analysis supports:

- edge-list data with source, target and optional weight columns
- square adjacency matrices
- Pearson and Spearman correlation networks
- regularised partial-correlation networks using Graphical Lasso CV
- directed, undirected, weighted, unweighted and signed networks
- optional self-loops and duplicate-edge aggregation
- two-group network comparison with permutation tests

Node-level measures include degree, absolute and signed strength, in/out degree and strength, betweenness, closeness, harmonic, eigenvector and PageRank centrality, hub and authority scores, average-neighbour degree, effective size, constraint, local clustering, k-core and community membership.

Graph-level measures include density, connected components, isolates, clustering, modularity, reciprocity, assortativity, shortest path length, diameter, global and local efficiency, clique size, degree and betweenness centralisation, and small-world sensitivity against random graphs. Edge outputs include edge betweenness, bridge status, articulation nodes and directed triad census where applicable.

Correlation networks can include bootstrap edge-inclusion probabilities and centrality-rank stability. The output provides an interactive draggable network, network structure, community, centrality, degree-distribution and adjacency-heatmap figures.

The DOCX and Excel outputs include paper-ready abstract, methods, results, diagnostics, robustness, discussion, limitations, figure captions and a network-reporting checklist.

## Statistical methods

### Exploratory factor analysis

- KMO overall and item-level diagnostics
- Bartlett test of sphericity
- parallel analysis for factor retention
- maximum-likelihood factor extraction
- optional varimax rotation
- communalities, uniqueness, variance explained and factor scores

### Confirmatory factor analysis

- dropdown-based construct-to-item specification
- ML, GLS, ULS and DWLS covariance objectives
- CFI, TLI, RMSEA, SRMR and model chi-square
- standardised loadings and item R-squared
- composite reliability and average variance extracted
- latent-factor correlations
- HTMT and Fornell-Larcker discriminant-validity assessment
- Mardia multivariate normality screening
- covariance eigenvalue and condition screening
- observations-per-parameter information screen
- residual-correlation diagnostics

### Covariance-based structural equation modelling

- dropdown-based measurement and structural specification
- direct and mediation relationships
- ML, GLS, ULS and DWLS covariance objectives
- simultaneous measurement and structural covariance fitting
- structural path estimates and specified mediation effects
- reliability, AVE, HTMT and Fornell-Larcker diagnostics
- Mardia normality, covariance, identification and residual diagnostics
- CFI, TLI, RMSEA, SRMR and model chi-square
- fitted measurement and path diagram

### Partial least squares structural equation modelling

- reflective Mode A and formative Mode B measurement blocks
- direct, mediation and two-stage latent-score moderation relationships
- selectable path, centroid and factorial inner-weighting schemes
- iterative PLS outer-weight estimation with convergence reporting
- outer loadings and outer weights
- bootstrap standard errors, p-values and percentile confidence intervals for paths, reflective loadings and formative weights
- joint-bootstrap indirect and total effects for specified mediation relationships
- Cronbach's alpha, approximate rho_A, composite reliability and AVE
- cross-loadings, Fornell-Larcker and HTMT
- outer and inner VIF
- path coefficients and specified indirect, direct and total effects
- endogenous R-squared and adjusted R-squared
- f-squared effect sizes
- cross-validated Q-squared predictive relevance
- approximate SRMR and d_ULS residual fit
- latent construct scores and fitted path diagram

The PLS-SEM engine does not automatically delete indicators or paths to improve thresholds. Weak results are documented for theory-led review and sensitivity analysis.

### Repeated-measures ANOVA

- within-subject descriptives
- approximate Mauchly sphericity assessment
- Greenhouse-Geisser correction when required
- Holm-adjusted paired comparisons and Cohen's dz

### Multilevel and clustered-outcome analysis

- ML and REML random-effects estimation for continuous outcomes
- robust GEE population-average estimation for continuous, binary and count outcomes
- exchangeable, independent and AR(1) GEE working correlations with independence-correlation sensitivity results
- odds-ratio reporting for binary outcomes and incidence-rate-ratio reporting for count outcomes
- random intercept and optional random slope
- separate level-1 and level-2 predictor selection
- group-mean centring with contextual effects
- grand-mean centring or no centring
- verification that level-2 variables are constant within clusters
- ICC(1), ICC(2) and design effect
- marginal and conditional R-squared
- cluster-size support diagnostics
- fixed-effect VIF
- residual normality and heteroskedasticity screening for linear mixed models
- outcome-support, Pearson-dispersion and link-family diagnostics for binary and count GEE
- random-effect normality and covariance singularity screening
- influential-cluster screening
- optimiser and convergence diagnostics
- automatic GEE or REML sensitivity coefficients

### Panel-data models

- pooled OLS with entity-clustered standard errors
- entity fixed effects
- random effects
- entity-effects F test
- Hausman specification comparison
- automatic or user-selected model
- optional time fixed effects

### Advanced conditional-process analysis

- advanced moderation with HC3 inference, simple slopes and Johnson-Neyman boundaries
- parallel multiple mediation with bootstrap confidence intervals
- first-stage moderated mediation with conditional indirect effects

## Estimation guidance

- **ML:** continuous indicators with approximately normal multivariate distributions
- **GLS:** covariance-residual fitting scaled by the observed covariance structure
- **ULS:** unweighted covariance-residual fitting
- **DWLS:** diagonally weighted fitting, useful as an internal sensitivity objective for ordinal or non-normal indicators
- **PLS-SEM:** composite-based, prediction-oriented latent-variable modelling with path, centroid or factorial inner weighting
- **REML:** variance-component estimation for multilevel models
- **ML:** multilevel model comparison involving fixed effects
- **GEE robust:** population-average clustered analysis for Gaussian, binomial or Poisson mean functions with robust covariance

The internal non-ML covariance objectives and SEM standard errors are labelled as approximations. Publication-critical models should be confirmed in specialist software.

## Integrity safeguards

The app does not change data to produce statistical significance. It:

1. Preserves the uploaded dataset as the original copy.
2. Applies treatments only to a separate analysis copy.
3. Records every treatment, reason, affected variable and sample-size effect.
4. Uses robust, corrected or alternative inference when diagnostics require it.
5. Keeps primary and sensitivity results visible.
6. Does not automatically delete indicators, add SEM paths, correlate errors or remove observations to improve fit.
7. Warns when advanced estimates require confirmation in a specialist package.

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
python -m pytest -q
```

The included suite contains 25 tests covering Phase 1 methods, automatic descriptives, exports, EFA, CFA, multiple covariance estimators, CB-SEM, PLS-SEM, repeated measures, mixed effects, multilevel diagnostics, panel selection, conditional-process models, AI-completed specifications, interactive path layouts, comprehensive network analysis, network stability and reproducibility assets.

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

Use Standard or higher for larger files, intensive PLS bootstrapping, CB-SEM or random-slope multilevel models.

## Demonstration datasets

### Phase 1

Use `sample_data/sample_research_data.csv`.

### Factor, CFA, CB-SEM and PLS-SEM

Use `sample_data/phase2_factor_sample.csv`.

Suggested constructs:

- ConstructA: `a1`, `a2`, `a3`, `a4`
- ConstructB: `b1`, `b2`, `b3`, `b4`
- Mediator: `m1`, `m2`

Suggested relationships:

- ConstructA directly predicts ConstructB
- ConstructA predicts Mediator, which predicts ConstructB
- Mediator may also be selected as a moderator for a demonstration of the latent-score interaction workflow

### Multilevel and panel data

Use `sample_data/phase2_longitudinal_sample.csv`.

- outcome: `y`
- level-1 predictor: `x`
- cluster: `entity`
- panel time: `time`

A separate `phase2_multilevel_sample.csv` is included with a level-2 predictor for centring and contextual-effect demonstrations. It also includes `completed` for binomial GEE and `support_contacts` for Poisson GEE demonstrations.

## Project structure

```text
app.py                         Streamlit user interface and structured builders
statready/dispatch.py          Analysis routing, relation effects and automatic descriptives
statready/methods.py           Phase 1 statistical methods
statready/phase2.py            EFA, CFA, CB-SEM, repeated measures, panel and conditional-process methods
statready/pls_sem.py           PLS-SEM estimation, bootstrap and diagnostics
statready/multilevel.py        ML, REML and GEE multilevel analysis
statready/figures.py           CFA, CB-SEM and PLS-SEM diagrams
statready/diagnostics.py       Assumption and diagnostic tests
statready/treatments.py        Logged data treatments
statready/profiling.py         Dataset screening
statready/recommender.py       Objective-to-method rules
statready/reports.py           DOCX, Excel and ZIP exports
statready/literature.py        Curated methodological sources
sample_data/                   Demonstration datasets
tests/                         Engine and export tests
```

## Important limitations

- The covariance-based SEM engine supports acyclic direct and mediation paths. Latent interactions are estimated through PLS-SEM or the dedicated moderation module.
- CFA and CB-SEM non-ML objectives and numerical standard errors are internal approximations. Confirm complex, ordinal, multigroup, invariance or publication-critical models in specialist software.
- PLS-SEM rho_A and residual-fit statistics are approximate. Confirm publication-critical PLS estimates independently.
- Continuous random-effects models are available through ML and REML. Binary and count outcomes currently use population-average GEE rather than subject-specific GLMM estimation.
- Small-cluster degrees-of-freedom corrections, negative-binomial clustered models and cluster bootstrap are not yet implemented.
- The panel module does not yet include dynamic panel GMM, cointegration or cross-sectional-dependence estimators.
- Uploaded projects remain session-based until authentication, database and object storage are added.

### Render path-editor deployment safeguard

Version 2.4.1 no longer depends on a committed directory named `build`. The path editor is shipped in `statready/path_editor_assets/index.html` and is also embedded in `statready/path_editor_component.py`. If the packaged HTML is absent, the app recreates it in the container temporary directory before Streamlit starts. This prevents the `No such component directory` startup failure.

Before pushing to GitHub, you can verify the deployment asset with:

```bash
python -c "from statready.path_editor_component import component_asset_status; print(component_asset_status())"
```

### Render dependency verification

Version 2.4.2 declares Matplotlib explicitly because the network-analysis module uses it to generate publication figures. The Docker image now runs `python -m statready.dependency_check` during the build. If any required package is missing, the build stops before Render starts the web service. After replacing an older release, use **Manual Deploy → Clear build cache & deploy** so Render installs the revised requirements.
