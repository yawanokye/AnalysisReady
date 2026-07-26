from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable, Independence, Autoregressive
from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.diagnostic import het_breuschpagan

from .diagnostics import normality_diagnostic, vif_table
from .models import AnalysisResult, AuditEntry


def _fixed_table(params: pd.Series, bse: pd.Series, alpha: float, exponentiate: bool = False) -> pd.DataFrame:
    params = pd.Series(params)
    bse = pd.Series(bse, index=params.index)
    statistic = params / bse.replace(0, np.nan)
    pvalues = 2 * stats.norm.sf(np.abs(statistic))
    critical = stats.norm.ppf(1 - alpha / 2)
    table = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": bse.values,
        "statistic": statistic.values,
        "p_value": pvalues,
        "ci_lower": params.values - critical * bse.values,
        "ci_upper": params.values + critical * bse.values,
    })
    if exponentiate:
        table["effect_ratio"] = np.exp(table["estimate"])
        table["effect_ratio_ci_lower"] = np.exp(table["ci_lower"])
        table["effect_ratio_ci_upper"] = np.exp(table["ci_upper"])
    return table


def _centre_predictors(
    data: pd.DataFrame,
    cluster: str,
    level1: list[str],
    level2: list[str],
    centering: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    work = data.copy()
    model_predictors: list[str] = []
    rows: list[dict] = []
    for variable in level1:
        if centering == "Group-mean with contextual effect":
            group_mean = work.groupby(cluster)[variable].transform("mean")
            within_name = f"{variable}__within"
            between_name = f"{variable}__between"
            work[within_name] = work[variable] - group_mean
            work[between_name] = group_mean - float(group_mean.mean())
            model_predictors.extend([within_name, between_name])
            rows.extend([
                {"original_variable": variable, "model_term": within_name, "level": "Level 1 within-cluster", "centering": "Group-mean centred"},
                {"original_variable": variable, "model_term": between_name, "level": "Cluster contextual effect", "centering": "Cluster mean, grand-mean centred"},
            ])
        elif centering == "Grand-mean":
            name = f"{variable}__grand_centered"
            work[name] = work[variable] - float(work[variable].mean())
            model_predictors.append(name)
            rows.append({"original_variable": variable, "model_term": name, "level": "Level 1", "centering": "Grand-mean centred"})
        else:
            model_predictors.append(variable)
            rows.append({"original_variable": variable, "model_term": variable, "level": "Level 1", "centering": "None"})
    for variable in level2:
        if centering in {"Grand-mean", "Group-mean with contextual effect"}:
            name = f"{variable}__grand_centered"
            work[name] = work[variable] - float(work[variable].mean())
            model_predictors.append(name)
            rows.append({"original_variable": variable, "model_term": name, "level": "Level 2", "centering": "Grand-mean centred"})
        else:
            model_predictors.append(variable)
            rows.append({"original_variable": variable, "model_term": variable, "level": "Level 2", "centering": "None"})
    return work, model_predictors, pd.DataFrame(rows)


def _fit_mixed(
    y: pd.Series,
    x: pd.DataFrame,
    groups: pd.Series,
    random_slope_term: str | None,
    reml: bool,
    optimizer: str,
):
    exog = sm.add_constant(x, has_constant="add")
    if random_slope_term and random_slope_term in x.columns:
        exog_re = sm.add_constant(x[[random_slope_term]], has_constant="add")
    else:
        exog_re = np.ones((len(x), 1))
    model = sm.MixedLM(y, exog, groups=groups, exog_re=exog_re)
    optimizers = list(dict.fromkeys([optimizer, "lbfgs", "powell", "cg"]))
    last_error = None
    for method in optimizers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(reml=reml, method=method, maxiter=2000, disp=False)
            if np.all(np.isfinite(fitted.params)):
                fitted._statready_optimizer = method
                return fitted
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"The multilevel model could not be estimated: {last_error}")


def _fit_null(y: pd.Series, groups: pd.Series, reml: bool = True):
    exog = np.ones((len(y), 1))
    model = sm.MixedLM(y, exog, groups=groups, exog_re=exog)
    last_error = None
    for method in ["lbfgs", "powell", "cg"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(reml=reml, method=method, maxiter=1500, disp=False)
            if np.all(np.isfinite(fitted.params)):
                return fitted
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"The multilevel null model could not be estimated: {last_error}")


def _gee_cov_structure(name: str):
    if name == "Independence":
        return Independence()
    if name == "AR(1)":
        return Autoregressive()
    return Exchangeable()


def _gee_family(name: str):
    key = str(name or "Gaussian").strip().lower()
    if key == "binary":
        return Binomial(), "Binary"
    if key == "count":
        return Poisson(), "Count"
    return Gaussian(), "Continuous"


def _prepare_outcome(data: pd.DataFrame, outcome: str, family_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    work = data.copy()
    mapping: dict[str, object] = {}
    if family_name == "Binary":
        nonmissing = work[outcome].dropna()
        unique = list(pd.unique(nonmissing))
        if len(unique) != 2:
            raise ValueError("A binary multilevel outcome must contain exactly two observed categories.")
        ordered = sorted(unique, key=lambda value: str(value))
        mapping = {"reference_category": ordered[0], "event_category": ordered[1]}
        work[outcome] = work[outcome].map({ordered[0]: 0.0, ordered[1]: 1.0})
    else:
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        if family_name == "Count":
            observed = work[outcome].dropna()
            if (observed < 0).any() or not np.allclose(observed, np.round(observed), atol=1e-8):
                raise ValueError("A count multilevel outcome must contain non-negative integer values.")
    return work, mapping


def _oneway_icc_screen(y: pd.Series, groups: pd.Series) -> tuple[float, float, float, float, float]:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "group": np.asarray(groups)})
    summaries = frame.groupby("group")["y"].agg(["count", "mean", "var"])
    k = len(summaries)
    n = int(summaries["count"].sum())
    grand = float(frame["y"].mean())
    ss_between = float(np.sum(summaries["count"] * (summaries["mean"] - grand) ** 2))
    ss_within = float(np.nansum((summaries["count"] - 1) * summaries["var"].fillna(0)))
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(n - k, 1)
    n0 = (n - float(np.sum(summaries["count"] ** 2)) / max(n, 1)) / max(k - 1, 1)
    icc1 = (ms_between - ms_within) / max(ms_between + (n0 - 1) * ms_within, 1e-12)
    icc1 = float(np.clip(icc1, 0.0, 1.0))
    mean_cluster = float(summaries["count"].mean())
    icc2 = icc1 / max(icc1 + (1 - icc1) / max(mean_cluster, 1), 1e-12)
    design_effect = 1 + (mean_cluster - 1) * icc1
    return icc1, float(icc2), float(design_effect), ms_between, ms_within


def _pseudo_r_squared(y: np.ndarray, fitted: np.ndarray, family_name: str) -> tuple[str, float]:
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    if family_name == "Binary":
        if np.any(y == 1) and np.any(y == 0):
            return "Tjur discrimination R squared", float(np.mean(fitted[y == 1]) - np.mean(fitted[y == 0]))
        return "Tjur discrimination R squared", np.nan
    if family_name == "Count":
        mean_y = max(float(np.mean(y)), 1e-12)
        def poisson_deviance(obs, mu):
            mu = np.clip(mu, 1e-12, None)
            term = np.zeros_like(obs, dtype=float)
            positive = obs > 0
            term[positive] = obs[positive] * np.log(obs[positive] / mu[positive])
            return float(2 * np.sum(term - (obs - mu)))
        null_dev = poisson_deviance(y, np.full_like(y, mean_y))
        model_dev = poisson_deviance(y, fitted)
        return "Poisson deviance explained", float(1 - model_dev / null_dev) if null_dev > 1e-12 else np.nan
    correlation = np.corrcoef(y, fitted)[0, 1] if np.std(y, ddof=1) > 1e-12 and np.std(fitted, ddof=1) > 1e-12 else np.nan
    return "Squared observed-fitted correlation", float(correlation ** 2) if np.isfinite(correlation) else np.nan


def multilevel_linear_model(
    df: pd.DataFrame,
    outcome: str,
    level1_predictors: list[str],
    level2_predictors: list[str],
    cluster: str,
    random_slope: str | None = None,
    estimator: str = "REML",
    centering: str = "Group-mean with contextual effect",
    optimizer: str = "lbfgs",
    gee_correlation: str = "Exchangeable",
    outcome_family: str = "Continuous",
    alpha: float = 0.05,
) -> AnalysisResult:
    level1_predictors = list(dict.fromkeys(level1_predictors or []))
    level2_predictors = list(dict.fromkeys(level2_predictors or []))
    all_predictors = list(dict.fromkeys(level1_predictors + level2_predictors))
    if not all_predictors:
        raise ValueError("Select at least one level-1 or level-2 predictor.")
    if random_slope and random_slope not in level1_predictors:
        raise ValueError("A random-slope variable must also be selected as a level-1 predictor.")

    family, family_name = _gee_family(outcome_family)
    if family_name != "Continuous" and estimator != "GEE robust":
        raise ValueError("Binary and count multilevel outcomes currently require GEE robust estimation in this build.")
    if estimator == "GEE robust" and random_slope:
        raise ValueError("GEE estimates population-average clustered effects and does not estimate random slopes. Clear the random-slope selection or choose ML/REML for a continuous outcome.")

    columns = list(dict.fromkeys([outcome, cluster] + all_predictors))
    data = df.loc[:, columns].copy()
    data, outcome_mapping = _prepare_outcome(data, outcome, family_name)
    for variable in all_predictors:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data.dropna().copy()
    n_clusters = int(data[cluster].nunique())
    if n_clusters < 3:
        raise ValueError("Multilevel analysis requires at least three clusters. Ten or more are strongly preferred for stable clustered inference.")
    if len(data) <= len(all_predictors) + n_clusters:
        raise ValueError("The sample is too small for the requested multilevel specification.")

    level2_check_rows = []
    for variable in level2_predictors:
        maximum_unique = int(data.groupby(cluster)[variable].nunique(dropna=True).max())
        level2_check_rows.append({
            "variable": variable,
            "maximum_values_within_cluster": maximum_unique,
            "status": "Satisfied" if maximum_unique <= 1 else "Material concern",
        })
        if maximum_unique > 1:
            raise ValueError(f"Level-2 predictor '{variable}' varies within clusters. Move it to level 1 or correct the data structure.")

    work, model_predictors, centering_table = _centre_predictors(data, cluster, level1_predictors, level2_predictors, centering)
    random_slope_term = None
    if random_slope:
        matching = centering_table[centering_table["original_variable"] == random_slope]
        if centering == "Group-mean with contextual effect":
            matching = matching[matching["level"] == "Level 1 within-cluster"]
        random_slope_term = str(matching.iloc[0]["model_term"]) if not matching.empty else random_slope

    x = work[model_predictors].astype(float)
    y = work[outcome].astype(float)
    groups = work[cluster]
    cluster_sizes = work.groupby(cluster).size()
    mean_cluster_size = float(cluster_sizes.mean())

    if family_name == "Continuous":
        null_model = _fit_null(y, groups, reml=True)
        null_cov = np.asarray(null_model.cov_re)
        null_between = float(null_cov[0, 0]) if null_cov.size else 0.0
        null_within = float(null_model.scale)
        icc1 = null_between / max(null_between + null_within, 1e-12)
        icc2 = icc1 / max(icc1 + (1 - icc1) / max(mean_cluster_size, 1), 1e-12)
        design_effect = 1 + (mean_cluster_size - 1) * icc1
    else:
        icc1, icc2, design_effect, ms_between, ms_within = _oneway_icc_screen(y, groups)
        null_between = max(ms_between - ms_within, 0.0) / max(mean_cluster_size, 1)
        null_within = max(ms_within, 0.0)

    sensitivity_table = pd.DataFrame()
    random_cov = pd.DataFrame()
    random_effect_values = np.asarray([])
    dispersion = np.nan

    if estimator in {"REML", "ML"}:
        fitted = _fit_mixed(y, x, groups, random_slope_term, reml=estimator == "REML", optimizer=optimizer)
        fixed = _fixed_table(fitted.fe_params, fitted.bse_fe, alpha)
        covariance = np.asarray(fitted.cov_re)
        random_intercept_variance = float(covariance[0, 0]) if covariance.size else np.nan
        residual_variance = float(fitted.scale)
        fitted_values = np.asarray(fitted.fittedvalues)
        residuals = np.asarray(fitted.resid)
        convergence = bool(fitted.converged)
        log_likelihood = float(fitted.llf)
        aic = float(fitted.aic) if np.isfinite(fitted.aic) else np.nan
        bic = float(fitted.bic) if np.isfinite(fitted.bic) else np.nan
        random_cov = pd.DataFrame(covariance)
        random_labels = ["Intercept"] + ([random_slope] if random_slope and covariance.shape[0] > 1 else [])
        random_cov.insert(0, "random_effect", random_labels[: len(random_cov)])
        try:
            random_effect_values = np.asarray([np.asarray(value).reshape(-1)[0] for value in fitted.random_effects.values()])
        except Exception:
            random_effect_values = np.asarray([])
        try:
            gee = GEE(y, sm.add_constant(x, has_constant="add"), groups=groups, family=Gaussian(), cov_struct=_gee_cov_structure(gee_correlation)).fit()
            sensitivity_table = _fixed_table(gee.params, gee.bse, alpha)
            sensitivity_table.insert(0, "model", f"GEE robust sensitivity ({gee_correlation})")
        except Exception:
            sensitivity_table = pd.DataFrame()
    else:
        gee = GEE(
            y,
            sm.add_constant(x, has_constant="add"),
            groups=groups,
            family=family,
            cov_struct=_gee_cov_structure(gee_correlation),
        ).fit()
        fixed = _fixed_table(gee.params, gee.bse, alpha, exponentiate=family_name in {"Binary", "Count"})
        fitted_values = np.asarray(gee.fittedvalues)
        residuals = np.asarray(getattr(gee, "resid_pearson", gee.resid_response))
        convergence = bool(getattr(gee, "converged", True))
        log_likelihood = float(getattr(gee, "llf", np.nan))
        aic_value = getattr(gee, "aic", np.nan)
        aic = float(aic_value) if np.isfinite(aic_value) else np.nan
        bic = np.nan
        random_intercept_variance = null_between
        residual_variance = null_within
        random_cov = pd.DataFrame([{"random_effect": "Not estimated by GEE", "variance": np.nan}])
        df_resid = max(int(getattr(gee, "df_resid", len(y) - len(x.columns) - 1)), 1)
        dispersion = float(np.sum(residuals ** 2) / df_resid)
        try:
            independent = GEE(
                y,
                sm.add_constant(x, has_constant="add"),
                groups=groups,
                family=family,
                cov_struct=Independence(),
            ).fit()
            sensitivity_table = _fixed_table(independent.params, independent.bse, alpha, exponentiate=family_name in {"Binary", "Count"})
            sensitivity_table.insert(0, "model", "GEE independence-correlation sensitivity")
        except Exception:
            sensitivity_table = pd.DataFrame()

    fixed_variance = float(np.var(fitted_values, ddof=1)) if len(fitted_values) > 1 else 0.0
    total_variance = fixed_variance + max(random_intercept_variance, 0) + max(residual_variance, 0)
    marginal_r2 = fixed_variance / max(total_variance, 1e-12)
    conditional_r2 = (fixed_variance + max(random_intercept_variance, 0)) / max(total_variance, 1e-12)
    pseudo_r2_name, pseudo_r2 = _pseudo_r_squared(y.to_numpy(), fitted_values, family_name)

    predictor_vif = vif_table(x)
    max_vif = float(predictor_vif["vif"].replace([np.inf, -np.inf], np.nan).max()) if not predictor_vif.empty else np.nan

    if family_name == "Continuous":
        residual_normal = normality_diagnostic(pd.Series(residuals), alpha, "multilevel residuals")
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm.add_constant(x, has_constant="add"))
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        if len(random_effect_values) >= 3:
            random_normal = normality_diagnostic(pd.Series(random_effect_values), alpha, "cluster random intercepts")
        else:
            random_normal = {
                "diagnostic": "Normality of cluster random intercepts", "test": "Shapiro-Wilk", "statistic": np.nan,
                "p_value": np.nan, "status": "Cannot determine", "interpretation": "Random effects were unavailable or too few.",
                "recommended_response": "Use graphical checks and cautious inference, particularly with few clusters.",
            }
        if estimator in {"REML", "ML"} and not random_cov.empty:
            numeric_cov = random_cov.drop(columns=["random_effect"], errors="ignore").to_numpy(dtype=float)
            eigenvalues = np.linalg.eigvalsh(numeric_cov)
        else:
            eigenvalues = np.asarray([random_intercept_variance])
        min_random_eigen = float(np.nanmin(eigenvalues)) if eigenvalues.size else np.nan
    else:
        residual_normal = {
            "diagnostic": "Residual distribution", "test": "Pearson residual normality", "statistic": np.nan,
            "p_value": np.nan, "status": "Not applicable", "interpretation": f"Normal residuals are not assumed for a {family_name.lower()} GEE outcome.",
            "recommended_response": "Assess outcome support, link function, dispersion, influential clusters and working-correlation sensitivity instead.",
        }
        bp_stat = bp_p = np.nan
        random_normal = {
            "diagnostic": "Random-effect distribution", "test": "GEE population-average model", "statistic": np.nan,
            "p_value": np.nan, "status": "Not applicable", "interpretation": "GEE does not estimate a random-effects distribution.",
            "recommended_response": "Use a specialist GLMM when subject-specific random effects are required.",
        }
        min_random_eigen = np.nan

    cluster_residual = pd.DataFrame({cluster: groups.to_numpy(), "residual": residuals}).groupby(cluster).agg(
        observations=("residual", "size"),
        mean_residual=("residual", "mean"),
        rms_residual=("residual", lambda value: float(np.sqrt(np.mean(np.asarray(value) ** 2)))),
    ).reset_index()
    residual_sd = float(cluster_residual["mean_residual"].std(ddof=1))
    cluster_residual["standardised_mean_residual"] = cluster_residual["mean_residual"] / residual_sd if residual_sd > 1e-12 else 0.0
    cluster_residual["influence_flag"] = cluster_residual["standardised_mean_residual"].abs() >= 3

    diagnostics_rows: list[dict] = [
        {
            "diagnostic": "Need for multilevel modelling", "test": "ICC(1) clustering screen", "statistic": icc1, "p_value": np.nan,
            "status": "Satisfied" if icc1 >= 0.05 else "Minor concern",
            "interpretation": f"ICC(1)={icc1:.3f}; design effect={design_effect:.3f}. For non-Gaussian outcomes this is an observed-scale screening estimate.",
            "recommended_response": "Retain clustered analysis when the design is nested even if the empirical ICC is modest. Compare working-correlation or marginal specifications.",
        },
        {
            "diagnostic": "Reliability of cluster means", "test": "ICC(2) screening", "statistic": icc2, "p_value": np.nan,
            "status": "Satisfied" if icc2 >= 0.70 else "Minor concern" if icc2 >= 0.50 else "Material concern",
            "interpretation": "ICC(2) screens whether cluster aggregates are measured reliably.",
            "recommended_response": "Use caution with cluster-level conclusions when ICC(2) is low, and consider more observations per cluster.",
        },
        {
            "diagnostic": "Cluster support", "test": "Number and size of clusters", "statistic": float(n_clusters), "p_value": np.nan,
            "status": "Satisfied" if n_clusters >= 30 else "Minor concern" if n_clusters >= 10 else "Material concern",
            "interpretation": f"Clusters={n_clusters}; minimum={int(cluster_sizes.min())}; median={cluster_sizes.median():.1f}; maximum={int(cluster_sizes.max())}.",
            "recommended_response": "Use cautious inference with few clusters. Confirm publication-critical results using small-sample corrections or cluster bootstrap.",
        },
        {
            "diagnostic": "Fixed-effect collinearity", "test": "Maximum VIF", "statistic": max_vif, "p_value": np.nan,
            "status": "Cannot determine" if not np.isfinite(max_vif) else "Satisfied" if max_vif < 5 else "Minor concern" if max_vif < 10 else "Material concern",
            "interpretation": "High VIF makes fixed or population-average effects unstable, including contextual-effect terms.",
            "recommended_response": "Review variable overlap, centering and contextual decomposition. Report sensitivity specifications rather than deleting variables to obtain significance.",
        },
        residual_normal,
        random_normal,
        {
            "diagnostic": "Optimisation convergence", "test": "Estimator convergence flag", "statistic": float(convergence), "p_value": np.nan,
            "status": "Satisfied" if convergence else "Material concern",
            "interpretation": "The estimator converged." if convergence else "The estimator did not confirm convergence.",
            "recommended_response": "Rescale variables, change the optimiser or working correlation, simplify unsupported structures, or collect more cluster information.",
        },
        {
            "diagnostic": "Influential clusters", "test": "Absolute standardised cluster mean residual >= 3", "statistic": float(cluster_residual["influence_flag"].sum()), "p_value": np.nan,
            "status": "Satisfied" if not cluster_residual["influence_flag"].any() else "Minor concern",
            "interpretation": f"{int(cluster_residual['influence_flag'].sum())} cluster(s) were flagged for review.",
            "recommended_response": "Verify data and run leave-one-cluster-out sensitivity checks. Do not delete a cluster solely because it changes significance.",
        },
    ]

    if family_name == "Continuous":
        diagnostics_rows.extend([
            {
                "diagnostic": "Conditional residual variance", "test": "Breusch-Pagan screening", "statistic": bp_stat, "p_value": bp_p,
                "status": "Cannot determine" if not np.isfinite(bp_p) else "Satisfied" if bp_p >= alpha else "Material concern",
                "interpretation": "A small p-value suggests residual variance changes with the fixed predictors.",
                "recommended_response": "Inspect level-specific residual plots and consider variance functions or robust/marginal sensitivity analysis.",
            },
            {
                "diagnostic": "Random-effects singularity", "test": "Minimum random-effects covariance eigenvalue", "statistic": min_random_eigen, "p_value": np.nan,
                "status": "Not applicable" if estimator == "GEE robust" else "Cannot determine" if not np.isfinite(min_random_eigen) else "Material concern" if min_random_eigen < 1e-6 else "Satisfied",
                "interpretation": "A near-zero eigenvalue suggests an over-complex or unsupported random-effects structure.",
                "recommended_response": "Simplify the random-effects structure only with theoretical justification and compare the simplified model transparently.",
            },
        ])
    else:
        outcome_mean = float(y.mean())
        events = int(y.sum()) if family_name == "Binary" else int((y > 0).sum())
        zeros = int((y == 0).sum())
        support_ratio = min(events, len(y) - events) / max(len(model_predictors), 1) if family_name == "Binary" else len(y) / max(len(model_predictors), 1)
        diagnostics_rows.extend([
            {
                "diagnostic": "Outcome support", "test": "Events or observations per model term", "statistic": support_ratio, "p_value": np.nan,
                "status": "Satisfied" if support_ratio >= 20 else "Minor concern" if support_ratio >= 10 else "Material concern",
                "interpretation": f"Outcome mean={outcome_mean:.3f}; zero observations={zeros}; support ratio={support_ratio:.2f}.",
                "recommended_response": "Reduce unsupported complexity, collect more outcome events, or use penalised/specialist methods when the support ratio is low.",
            },
            {
                "diagnostic": "Conditional dispersion", "test": "Pearson chi-square divided by residual degrees of freedom", "statistic": dispersion, "p_value": np.nan,
                "status": "Cannot determine" if not np.isfinite(dispersion) else "Satisfied" if 0.75 <= dispersion <= 1.50 else "Minor concern" if 0.50 <= dispersion <= 2.00 else "Material concern",
                "interpretation": "Values far above one indicate overdispersion; values far below one indicate underdispersion relative to the selected mean-variance function.",
                "recommended_response": "For material count overdispersion, consider negative-binomial or other specialist clustered models. For binary outcomes, inspect omitted heterogeneity and model specification.",
            },
            {
                "diagnostic": "Working-correlation sensitivity", "test": f"Selected {gee_correlation} versus independence", "statistic": np.nan, "p_value": np.nan,
                "status": "Satisfied" if not sensitivity_table.empty else "Cannot determine",
                "interpretation": "A second GEE specification is retained so coefficient direction and uncertainty can be compared across working correlations.",
                "recommended_response": "Treat materially different conclusions as sensitivity to the assumed within-cluster correlation structure.",
            },
        ])

    fit_table = pd.DataFrame([{
        "outcome_family": family_name,
        "estimator": estimator,
        "optimizer": getattr(locals().get("fitted", None), "_statready_optimizer", optimizer) if estimator in {"REML", "ML"} else "iterative GEE",
        "gee_working_correlation": gee_correlation if estimator == "GEE robust" else "Sensitivity model only",
        "n": len(work),
        "clusters": n_clusters,
        "minimum_cluster_size": int(cluster_sizes.min()),
        "median_cluster_size": float(cluster_sizes.median()),
        "maximum_cluster_size": int(cluster_sizes.max()),
        "icc_1": icc1,
        "icc_2": icc2,
        "icc_1_screen": icc1,
        "icc_2_screen": icc2,
        "design_effect": design_effect,
        "marginal_r_squared_variance_screen": marginal_r2,
        "conditional_r_squared_variance_screen": conditional_r2,
        "pseudo_r_squared_name": pseudo_r2_name,
        "pseudo_r_squared": pseudo_r2,
        "pearson_dispersion": dispersion,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "random_intercept_variance": random_intercept_variance if estimator in {"REML", "ML"} else np.nan,
        "residual_variance": residual_variance if estimator in {"REML", "ML"} else np.nan,
        "converged": convergence,
    }])

    warnings_list: list[str] = []
    if n_clusters < 10:
        warnings_list.append("Fewer than ten clusters provide weak support for clustered inference. Treat p-values and variance estimates as provisional.")
    if not convergence:
        warnings_list.append("The selected multilevel estimator did not converge and should not be used for substantive inference.")
    if family_name != "Continuous":
        warnings_list.append("Binary and count outcomes are estimated with population-average GEE in this build. Use specialist GLMM software when subject-specific random effects are required.")

    tables = {
        "Multilevel fixed effects": fixed,
        "Multilevel model fit and variance partition": fit_table,
        "Random-effects covariance": random_cov,
        "Predictor centering and level decomposition": centering_table,
        "Fixed-effect VIF": predictor_vif,
        "Cluster sizes": cluster_sizes.reset_index(name="observations"),
        "Cluster residual influence screening": cluster_residual,
        "Level-2 variable check": pd.DataFrame(level2_check_rows),
    }
    if outcome_mapping:
        tables["Binary outcome coding"] = pd.DataFrame([outcome_mapping])
    if not sensitivity_table.empty:
        tables["Alternative estimator sensitivity coefficients"] = sensitivity_table

    method_name = "Multilevel linear mixed model" if estimator in {"REML", "ML"} else f"Population-average multilevel GEE ({family_name.lower()} outcome)"
    return AnalysisResult(
        method=method_name,
        summary=(
            f"The {estimator} clustered analysis used {len(work)} observations in {n_clusters} clusters with a {family_name.lower()} outcome. "
            f"ICC screening={icc1:.3f}, design effect={design_effect:.3f}, and {pseudo_r2_name}={pseudo_r2:.3f}. "
            f"The selected estimator converged={convergence}."
        ),
        tables=tables,
        diagnostics=pd.DataFrame(diagnostics_rows),
        metadata={
            "outcome": outcome,
            "outcome_family": family_name,
            "outcome_mapping": outcome_mapping,
            "level1_predictors": level1_predictors,
            "level2_predictors": level2_predictors,
            "cluster": cluster,
            "random_slope": random_slope,
            "estimator": estimator,
            "centering": centering,
            "gee_correlation": gee_correlation,
        },
        warnings=warnings_list,
        treatment_log=[AuditEntry(
            action="Estimated multilevel dependence structure",
            variable=cluster,
            details=f"Used {estimator} for a {family_name.lower()} outcome with {n_clusters} clusters, {len(level1_predictors)} level-1 predictor(s), {len(level2_predictors)} level-2 predictor(s), and {'a random slope' if random_slope else 'no random slope'}.",
            justification="The model separates within-cluster and between-cluster information where requested. A working-correlation or estimator sensitivity specification is retained when available.",
            before_n=len(work), after_n=len(work),
        )],
        reproducible_code=(
            "# Verify level-2 variables, centre level-1 predictors, fit ML/REML MixedLM for continuous outcomes or robust GEE for continuous, binary or count outcomes; "
            "assess ICC, design effect, collinearity, residual or dispersion diagnostics, convergence, working-correlation sensitivity and cluster influence.\n"
        ),
    )
