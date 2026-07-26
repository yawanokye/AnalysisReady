from __future__ import annotations

import pandas as pd


CURATED_REFERENCES = [
    {
        "topic": "reliability",
        "citation": "Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16, 297–334.",
        "doi": "10.1007/BF02310555",
        "supports": "Internal-consistency reliability and coefficient alpha.",
    },
    {
        "topic": "heteroskedasticity",
        "citation": "Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometrica, 47(5), 1287–1294.",
        "doi": "10.2307/1911963",
        "supports": "Breusch–Pagan testing for non-constant error variance.",
    },
    {
        "topic": "robust_standard_errors",
        "citation": "White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48(4), 817–838.",
        "doi": "10.2307/1912934",
        "supports": "Heteroskedasticity-consistent covariance estimation.",
    },
    {
        "topic": "mediation",
        "citation": "Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. Behavior Research Methods, 40, 879–891.",
        "doi": "10.3758/BRM.40.3.879",
        "supports": "Bootstrap confidence intervals for indirect effects.",
    },
    {
        "topic": "bootstrap",
        "citation": "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.",
        "doi": "",
        "supports": "General bootstrap estimation and uncertainty assessment.",
    },
    {
        "topic": "missing_data",
        "citation": "Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.). Wiley.",
        "doi": "10.1002/9781119482260",
        "supports": "Principles for analysing and documenting missing data.",
    },
    {
        "topic": "multiple_imputation",
        "citation": "van Buuren, S. (2018). Flexible Imputation of Missing Data (2nd ed.). CRC Press.",
        "doi": "10.1201/9780429492259",
        "supports": "Practical multiple-imputation procedures and diagnostics.",
    },
    {
        "topic": "regression_diagnostics",
        "citation": "Cook, R. D. (1977). Detection of influential observation in linear regression. Technometrics, 19(1), 15–18.",
        "doi": "10.2307/1268249",
        "supports": "Influence assessment using Cook’s distance.",
    },
    {
        "topic": "multicollinearity",
        "citation": "Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression Diagnostics: Identifying Influential Data and Sources of Collinearity. Wiley.",
        "doi": "",
        "supports": "Diagnosis and interpretation of collinearity in regression models.",
    },
    {
        "topic": "ridge_regression",
        "citation": "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67.",
        "doi": "10.1080/00401706.1970.10488634",
        "supports": "Ridge regression as a stabilising alternative when predictors are strongly collinear.",
    },
    {
        "topic": "model_specification",
        "citation": "Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. Journal of the Royal Statistical Society: Series B, 31(2), 350–371.",
        "doi": "10.1111/j.2517-6161.1969.tb00796.x",
        "supports": "RESET testing for functional-form and specification concerns.",
    },
    {
        "topic": "exploratory_factor_analysis",
        "citation": "Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. Psychological Methods, 4(3), 272–299.",
        "doi": "10.1037/1082-989X.4.3.272",
        "supports": "Factor extraction, retention, rotation and interpretation in exploratory factor analysis.",
    },
    {
        "topic": "parallel_analysis",
        "citation": "Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika, 30, 179–185.",
        "doi": "10.1007/BF02289447",
        "supports": "Parallel analysis for empirical factor retention.",
    },
    {
        "topic": "cfa_sem",
        "citation": "Brown, T. A. (2015). Confirmatory Factor Analysis for Applied Research (2nd ed.). Guilford Press.",
        "doi": "",
        "supports": "Specification, estimation and evaluation of confirmatory factor models.",
    },
    {
        "topic": "sem_fit",
        "citation": "Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis: Conventional criteria versus new alternatives. Structural Equation Modeling, 6(1), 1–55.",
        "doi": "10.1080/10705519909540118",
        "supports": "Joint interpretation of CFI, TLI, RMSEA and SRMR in covariance-structure models.",
    },
    {
        "topic": "repeated_measures",
        "citation": "Greenhouse, S. W., & Geisser, S. (1959). On methods in the analysis of profile data. Psychometrika, 24, 95–112.",
        "doi": "10.1007/BF02289823",
        "supports": "Greenhouse–Geisser correction when repeated-measures sphericity is not supported.",
    },
    {
        "topic": "mixed_effects",
        "citation": "Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. Biometrics, 38(4), 963–974.",
        "doi": "10.2307/2529876",
        "supports": "Random-effects modelling for clustered and longitudinal observations.",
    },
    {
        "topic": "panel_data",
        "citation": "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data (2nd ed.). MIT Press.",
        "doi": "",
        "supports": "Pooled, fixed-effects and random-effects panel-data modelling and specification decisions.",
    },
    {
        "topic": "conditional_process",
        "citation": "Hayes, A. F. (2022). Introduction to Mediation, Moderation, and Conditional Process Analysis (3rd ed.). Guilford Press.",
        "doi": "",
        "supports": "Simple slopes, Johnson–Neyman analysis, multiple mediation and moderated mediation.",
    },
]


def reference_table(topics: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(CURATED_REFERENCES)
    if topics:
        frame = frame[frame["topic"].isin(topics)]
    return frame.reset_index(drop=True)


def references_for_method(method: str, diagnostics: pd.DataFrame | None = None) -> pd.DataFrame:
    method_lower = method.lower()
    topics: set[str] = set()
    if "reliability" in method_lower:
        topics.add("reliability")
    if "mediation" in method_lower:
        topics.update({"mediation", "bootstrap"})
    if "parallel multiple mediation" in method_lower or "moderated mediation" in method_lower or "advanced moderated" in method_lower:
        topics.update({"conditional_process", "bootstrap"})
    if "exploratory factor" in method_lower:
        topics.update({"exploratory_factor_analysis", "parallel_analysis"})
    if "confirmatory factor" in method_lower or "structural equation" in method_lower:
        topics.update({"cfa_sem", "sem_fit"})
    if "repeated-measures" in method_lower:
        topics.add("repeated_measures")
    if "mixed-effects" in method_lower:
        topics.add("mixed_effects")
    if "fixed-effects" in method_lower or "random-effects" in method_lower or "panel" in method_lower:
        topics.add("panel_data")
    if "regression" in method_lower or "ols" in method_lower:
        topics.update({"regression_diagnostics", "model_specification"})
    if diagnostics is not None and not diagnostics.empty:
        tests = " ".join(diagnostics.get("test", pd.Series(dtype=str)).astype(str)).lower()
        if "breusch" in tests:
            topics.update({"heteroskedasticity", "robust_standard_errors"})
        if "cook" in tests:
            topics.add("regression_diagnostics")
        if "reset" in tests:
            topics.add("model_specification")
        if "variance inflation factor" in tests or "ridge" in tests:
            topics.update({"multicollinearity", "ridge_regression"})
    if not topics:
        topics.add("bootstrap")
    return reference_table(sorted(topics))
