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
