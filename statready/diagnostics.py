from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def _status(p_value: float | None, alpha: float, null_desired: bool = True) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "Cannot determine"
    passed = p_value >= alpha if null_desired else p_value < alpha
    return "Satisfied" if passed else "Material concern"


def normality_diagnostic(values: pd.Series, alpha: float = 0.05, label: str = "Variable") -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return {
            "diagnostic": f"Normality of {label}",
            "test": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Cannot determine",
            "interpretation": "At least three complete observations are required.",
            "recommended_response": "Collect or retain more valid observations.",
        }
    tested = clean.sample(min(len(clean), 5000), random_state=42) if len(clean) > 5000 else clean
    stat, p = stats.shapiro(tested)
    return {
        "diagnostic": f"Normality of {label}",
        "test": "Shapiro-Wilk",
        "statistic": float(stat),
        "p_value": float(p),
        "status": _status(float(p), alpha, null_desired=True),
        "interpretation": "No strong evidence against normality." if p >= alpha else "The distribution departs from normality.",
        "recommended_response": "Use robust or non-parametric inference when the departure is material, and inspect plots and sample size." if p < alpha else "No remedial action is required solely from this test.",
    }


def variance_equality_diagnostic(groups: list[pd.Series], alpha: float = 0.05) -> dict[str, object]:
    clean_groups = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    clean_groups = [g for g in clean_groups if len(g) >= 2]
    if len(clean_groups) < 2:
        return {
            "diagnostic": "Equality of group variances",
            "test": "Levene",
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Cannot determine",
            "interpretation": "At least two groups with two observations each are required.",
            "recommended_response": "Check group coding and sample size.",
        }
    stat, p = stats.levene(*clean_groups, center="median")
    return {
        "diagnostic": "Equality of group variances",
        "test": "Levene",
        "statistic": float(stat),
        "p_value": float(p),
        "status": _status(float(p), alpha, null_desired=True),
        "interpretation": "Group variances are reasonably similar." if p >= alpha else "Group variances differ materially.",
        "recommended_response": "Use Welch's test or heteroskedasticity-robust inference." if p < alpha else "The equal-variance form is acceptable, subject to design considerations.",
    }


def vif_table(exog: pd.DataFrame) -> pd.DataFrame:
    """Calculate predictor VIFs using auxiliary regressions with an intercept.

    Omitting the intercept can seriously inflate VIFs when predictors have
    non-zero means. Constant and near-constant predictors are reported as
    non-estimable rather than silently distorting the diagnostic.
    """
    numeric = exog.select_dtypes(include=np.number).dropna().copy()
    if numeric.empty:
        return pd.DataFrame(columns=["variable", "vif", "tolerance", "status"])

    rows: list[dict[str, object]] = []
    variable_columns: list[str] = []
    for column in numeric.columns:
        variance = float(numeric[column].var(ddof=0))
        if not np.isfinite(variance) or variance <= 1e-12:
            rows.append({
                "variable": column,
                "vif": np.inf,
                "tolerance": 0.0,
                "status": "Material concern",
            })
        else:
            variable_columns.append(column)

    if len(variable_columns) == 1:
        rows.append({
            "variable": variable_columns[0],
            "vif": 1.0,
            "tolerance": 1.0,
            "status": "Satisfied",
        })
        return pd.DataFrame(rows)
    if not variable_columns:
        return pd.DataFrame(rows)

    design = sm.add_constant(numeric[variable_columns].astype(float), has_constant="add")
    values = design.to_numpy(dtype=float)
    for idx, column in enumerate(variable_columns, start=1):
        try:
            vif = float(variance_inflation_factor(values, idx))
        except Exception:
            vif = np.inf
        tolerance = 0.0 if not np.isfinite(vif) or vif <= 0 else 1.0 / vif
        status = "Satisfied" if vif < 5 else "Minor concern" if vif < 10 else "Material concern"
        rows.append({"variable": column, "vif": vif, "tolerance": tolerance, "status": status})
    return pd.DataFrame(rows)


def ols_diagnostics(model, alpha: float = 0.05) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    residuals = pd.Series(model.resid)

    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    rows.append({
        "diagnostic": "Residual distribution",
        "test": "Jarque-Bera",
        "statistic": float(jb_stat),
        "p_value": float(jb_p),
        "status": _status(float(jb_p), alpha, null_desired=True),
        "interpretation": f"Residual skewness={skew:.3f}, kurtosis={kurtosis:.3f}.",
        "recommended_response": "Use bootstrap or robust inference and inspect influential observations when departure is material." if jb_p < alpha else "No remedial action is required solely from this test.",
    })

    try:
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)
        rows.append({
            "diagnostic": "Constant error variance",
            "test": "Breusch-Pagan",
            "statistic": float(lm_stat),
            "p_value": float(lm_p),
            "status": _status(float(lm_p), alpha, null_desired=True),
            "interpretation": "No strong evidence of heteroskedasticity." if lm_p >= alpha else "The residual variance changes across fitted values or predictors.",
            "recommended_response": "Use HC3 robust standard errors and report both conventional and robust inference." if lm_p < alpha else "Conventional standard errors are acceptable, subject to other diagnostics.",
        })
    except Exception as exc:
        rows.append({
            "diagnostic": "Constant error variance",
            "test": "Breusch-Pagan",
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Cannot determine",
            "interpretation": str(exc),
            "recommended_response": "Inspect residual plots and verify model specification.",
        })

    dw = float(durbin_watson(model.resid))
    dw_status = "Satisfied" if 1.5 <= dw <= 2.5 else "Minor concern"
    rows.append({
        "diagnostic": "Residual independence",
        "test": "Durbin-Watson",
        "statistic": dw,
        "p_value": np.nan,
        "status": dw_status,
        "interpretation": "The statistic is near 2." if dw_status == "Satisfied" else "The statistic suggests possible serial dependence.",
        "recommended_response": "Use design-appropriate clustered or HAC standard errors when observations are ordered or grouped." if dw_status != "Satisfied" else "No remedial action is indicated by this statistic.",
    })

    try:
        reset = linear_reset(model, power=2, use_f=True)
        p = float(np.asarray(reset.pvalue).squeeze())
        stat = float(np.asarray(reset.fvalue).squeeze())
        rows.append({
            "diagnostic": "Functional form",
            "test": "Ramsey RESET",
            "statistic": stat,
            "p_value": p,
            "status": _status(p, alpha, null_desired=True),
            "interpretation": "No strong evidence of omitted nonlinear structure." if p >= alpha else "The fitted functional form may be incomplete.",
            "recommended_response": "Revisit theory, transformations, interactions, and nonlinear terms. Do not search specifications solely for significance." if p < alpha else "Retain the specified form unless theory suggests otherwise.",
        })
    except Exception as exc:
        rows.append({
            "diagnostic": "Functional form",
            "test": "Ramsey RESET",
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Cannot determine",
            "interpretation": str(exc),
            "recommended_response": "Inspect residual and component-plus-residual plots.",
        })

    try:
        influence = OLSInfluence(model)
        cooks = influence.cooks_distance[0]
        threshold = 4 / max(int(model.nobs), 1)
        influential_count = int(np.sum(cooks > threshold))
        share = influential_count / max(int(model.nobs), 1)
        status = "Satisfied" if influential_count == 0 else "Minor concern" if share < 0.05 else "Material concern"
        rows.append({
            "diagnostic": "Influential observations",
            "test": "Cook's distance",
            "statistic": float(np.nanmax(cooks)) if len(cooks) else np.nan,
            "p_value": np.nan,
            "status": status,
            "interpretation": f"{influential_count} observation(s) exceed 4/n={threshold:.4f}.",
            "recommended_response": "Verify values and compare full-sample and sensitivity models. Do not delete observations automatically." if influential_count else "No observation exceeds the screening threshold.",
        })
    except Exception as exc:
        rows.append({
            "diagnostic": "Influential observations",
            "test": "Cook's distance",
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Cannot determine",
            "interpretation": str(exc),
            "recommended_response": "Inspect leverage and residual plots manually.",
        })

    return pd.DataFrame(rows)
