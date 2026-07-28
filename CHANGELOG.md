# Changelog

## 2.6.0

- Added cost-aware model routing for conceptual-framework vision and analysis planning.
- Set GPT-5.4 nano as the primary framework reader and GPT-5.6 Luna as the controlled fallback.
- Added image resizing and configurable low-detail primary versus high-detail fallback extraction.
- Added graph validation for duplicate constructs, unknown path endpoints, self-paths and incomplete mediation or moderation definitions.
- Added DeepSeek V4 Flash as the primary objective-specific planning model and V4 Pro as the controlled fallback.
- Added Pydantic validation for DeepSeek JSON responses and deterministic validation of every proposed method configuration.
- Added safe overlay of provider suggestions onto verified dataset columns and StatReady method fields.
- Added provider, model-attempt and fallback status to the Streamlit interface.
- Added a deterministic local fallback when API keys, providers or schema-valid responses are unavailable.
- Prevented raw dataset rows from being sent to either planning provider. Only aggregate schema metadata is shared with the reasoning provider.
- Added separate Render secrets and model environment variables for OpenAI and DeepSeek.
- Expanded the automated test suite from 33 to 37 passing tests.

## 2.5.0

- Reorganised the Streamlit interface into six focused sidebar workspaces with professional branding, progress indicators, status badges, cards and simplified result navigation.
- Added multiple objectives and hypotheses with objective-by-objective analysis mapping.
- Added a complete AI analysis programme that fills method-specific fields from objectives, hypotheses, framework evidence, variable roles and dataset structure.
- Added critical-decision gates so ambiguous causal direction, measurement mode, identifier structure and unmatched variables require human confirmation.
- Added conceptual-framework image upload and structured vision extraction using the OpenAI Responses API when `OPENAI_API_KEY` is configured.
- Added diagram-to-dataset variable reconciliation with editable indicator matches.
- Added proposed, non-estimated path diagrams before analysis and final estimated diagrams after analysis.
- Added objective and hypothesis mapping to every result, DOCX report, Excel workbook and reproducibility package.
- Added a combined objective-specific analysis ZIP export.
- Added Render environment-variable declarations for `OPENAI_API_KEY` and `OPENAI_VISION_MODEL`.
- Expanded the automated test suite from 29 to 33 passing tests.

## 2.4.2

- Fixed Render startup failure caused by the undeclared `matplotlib` runtime dependency used by network diagrams.
- Added `matplotlib>=3.9,<4` to `requirements.txt`.
- Added a Docker build-time dependency smoke test covering every third-party runtime package.
- Changed network-analysis loading to be lazy, so a plotting dependency problem cannot prevent the rest of the application from starting.
- Added regression tests that compare required runtime packages with the dependency manifest.

## 2.4.0

- Added a persistent drag-and-click path-diagram editor for CFA, CB-SEM and PLS-SEM results.
- Added click-to-nudge controls, automatic layouts, saved node positions and arrangement JSON export.
- Applied saved path positions to in-app PNGs, DOCX, Excel and reproducibility-package exports.
- Expanded the AI research agent to use objectives, hypotheses, framework wording and dataset structure to complete the full analysis specification.
- Added automatic outcome, predictor, mediator, moderator, control, group, cluster, entity, time and construct assignment where evidence is sufficient.
- Added a one-click guided run that pauses only for critical unresolved design or variable decisions.
- Added comprehensive network analysis for edge lists, adjacency matrices, correlation networks and regularised partial-correlation networks.
- Added directed, undirected, weighted, unweighted and signed-network options.
- Added node degree, strength, in/out degree, betweenness, closeness, harmonic, eigenvector, PageRank, hub, authority, structural-hole, clustering, k-core and community measures.
- Added graph density, components, isolates, clustering, modularity, reciprocity, assortativity, path length, diameter, efficiency, clique size, centralisation and small-world assessment.
- Added edge betweenness, bridge edges, articulation nodes, directed triad census, bootstrap edge stability and centrality stability.
- Added two-group network comparison with permutation tests.
- Added interactive draggable network HTML and five publication-ready network diagrams.
- Added paper-ready abstract, methods, results, diagnostics, robustness, discussion, limitations, figure captions and reporting checklist.
- Expanded the automated test suite from 21 to 25 passing tests.

## 2.3.0

- Added AI Guided Mode with Novice, Assisted and Expert co-pilot levels.
- Added readiness checks for study objectives, hypotheses, datasets, outcome types, conceptual frameworks and variable roles.
- Added conservative variable-role suggestions with confidence ratings and reviewable explanations.
- Added dataset-driven construct suggestions based on repeated item-name patterns.
- Added confirmation-controlled loading of role and construct suggestions.
- Added left-to-right, top-to-bottom, bottom-to-top, radial, hierarchical, measurement-first, structural-first and compact publication path-diagram layouts.
- Added straight and curved arrows, construct ordering, significance highlighting, monochrome rendering and transparent-background exports.
- Added diagram controls for indicators, item names, loadings, coefficients, p-values and fit indices.
- Added visible moderation arrows to the focal structural path.
- Added diagram settings to the analysis plan and retained them in all figure exports.
- Expanded the automated test suite from 18 to 21 passing tests.

## 2.2.0

- Replaced free-text construct syntax with named construct builders and indicator dropdowns.
- Added direct, mediation and moderation relationship builders using entered construct names.
- Added PLS-SEM with reflective Mode A and formative Mode B blocks.
- Added path, centroid and factorial PLS inner-weighting schemes.
- Added bootstrap inference for structural paths, reflective loadings, formative weights and joint indirect effects.
- Added PLS diagnostics for convergence, reliability, AVE, HTMT, Fornell-Larcker, cross-loadings, outer and inner VIF, R-squared, f-squared, Q-squared, SRMR, d_ULS and bootstrap stability.
- Expanded CFA and CB-SEM estimators to ML, GLS, ULS and DWLS with distribution, identification, convergence, admissibility, validity and residual diagnostics.
- Added endogenous latent R-squared and latent structural VIF to CB-SEM.
- Added multilevel ML, REML and robust GEE workflows with level-1 and level-2 predictors, centring and random slopes for continuous outcomes.
- Added robust GEE for binary and count clustered outcomes with working-correlation sensitivity, effect ratios, outcome-support and dispersion diagnostics.
- Added updated DOCX, Excel and reproducibility exports and demonstration datasets.

## 2.4.1

- Fixed Render startup failure when Git or deployment tooling omitted the nested `frontend/build` directory.
- Moved the packaged path-editor asset to `statready/path_editor_assets`, which is less likely to be ignored by Git tooling.
- Embedded the complete component HTML in the Python module as a runtime fallback.
- Added automatic recreation of the component in a writable temporary directory before Streamlit registers it.
- Added a Docker build-time component health check and automated regression tests for missing packaged assets.
