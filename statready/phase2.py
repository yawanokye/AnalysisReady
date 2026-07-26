from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from .diagnostics import normality_diagnostic, ols_diagnostics, vif_table
from .models import AnalysisResult, AuditEntry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _complete_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns))
    data = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if data.empty:
        raise ValueError("No complete numeric observations remain for the selected variables.")
    return data


def _tidy_model(model, exponentiate: bool = False) -> pd.DataFrame:
    params = pd.Series(model.params)
    bse = pd.Series(model.bse, index=params.index)
    stat_values = pd.Series(model.tvalues, index=params.index)
    pvalues = pd.Series(model.pvalues, index=params.index)
    conf = pd.DataFrame(model.conf_int(), index=params.index)
    table = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": bse.values,
        "statistic": stat_values.values,
        "p_value": pvalues.values,
        "ci_lower": conf.iloc[:, 0].values,
        "ci_upper": conf.iloc[:, 1].values,
    })
    if exponentiate:
        table["odds_ratio"] = np.exp(table["estimate"])
        table["or_ci_lower"] = np.exp(table["ci_lower"])
        table["or_ci_upper"] = np.exp(table["ci_upper"])
    return table


def _varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 500, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal varimax rotation with the rotation matrix returned."""
    p, k = loadings.shape
    rotation = np.eye(k)
    objective = 0.0
    for _ in range(max_iter):
        previous = objective
        transformed = loadings @ rotation
        u, singular, vh = np.linalg.svd(
            loadings.T @ (transformed ** 3 - (gamma / p) * transformed @ np.diag(np.diag(transformed.T @ transformed)))
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if previous and objective - previous < tol:
            break
    return loadings @ rotation, rotation


def _bartlett_sphericity(correlation: np.ndarray, n: int) -> tuple[float, int, float]:
    p = correlation.shape[0]
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        return np.inf, p * (p - 1) // 2, 0.0
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) // 2
    return float(chi2), int(df), float(stats.chi2.sf(chi2, df))


def _kmo(correlation: np.ndarray) -> tuple[float, np.ndarray]:
    inverse = np.linalg.pinv(correlation)
    scale = np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    partial = -inverse / np.where(scale == 0, np.nan, scale)
    np.fill_diagonal(partial, 0.0)
    corr_sq = correlation ** 2
    partial_sq = np.nan_to_num(partial ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_sq, 0.0)
    numerator = corr_sq.sum()
    denominator = numerator + partial_sq.sum()
    overall = numerator / denominator if denominator > 0 else np.nan
    item_num = corr_sq.sum(axis=0)
    item_den = item_num + partial_sq.sum(axis=0)
    item = np.divide(item_num, item_den, out=np.full_like(item_num, np.nan), where=item_den > 0)
    return float(overall), item


def _parallel_analysis(z: np.ndarray, iterations: int = 100, percentile: float = 95.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(random_state)
    observed = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))[::-1]
    random_eigen = []
    for _ in range(iterations):
        random_data = rng.normal(size=z.shape)
        random_eigen.append(np.linalg.eigvalsh(np.corrcoef(random_data, rowvar=False))[::-1])
    threshold = np.percentile(np.asarray(random_eigen), percentile, axis=0)
    retained = int(max(1, np.sum(observed > threshold)))
    return observed, threshold, retained


def _fit_index_diagnostics(fit: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    rows.append({
        "diagnostic": "Overall model fit",
        "test": "Model chi-square",
        "statistic": fit.get("chi_square", np.nan),
        "p_value": fit.get("chi_square_p", np.nan),
        "status": "Minor concern" if np.isfinite(fit.get("chi_square_p", np.nan)) and fit["chi_square_p"] < alpha else "Satisfied",
        "interpretation": "A significant chi-square can reflect exact-fit departure and is sensitive to sample size.",
        "recommended_response": "Interpret chi-square with CFI, TLI, RMSEA, SRMR, theory and residuals. Do not modify a model solely to improve fit.",
    })
    for name, value, good, adequate, direction in [
        ("CFI", fit.get("cfi", np.nan), 0.95, 0.90, "high"),
        ("TLI", fit.get("tli", np.nan), 0.95, 0.90, "high"),
        ("RMSEA", fit.get("rmsea", np.nan), 0.05, 0.08, "low"),
        ("SRMR", fit.get("srmr", np.nan), 0.05, 0.08, "low"),
    ]:
        if not np.isfinite(value):
            status = "Cannot determine"
        elif direction == "high":
            status = "Satisfied" if value >= good else "Minor concern" if value >= adequate else "Material concern"
        else:
            status = "Satisfied" if value <= good else "Minor concern" if value <= adequate else "Material concern"
        rows.append({
            "diagnostic": f"Approximate fit: {name}",
            "test": name,
            "statistic": value,
            "p_value": np.nan,
            "status": status,
            "interpretation": f"{name}={value:.3f}" if np.isfinite(value) else f"{name} could not be calculated.",
            "recommended_response": "Review theory, residual correlations and measurement quality when fit is weak. Any re-specification must be declared and cross-validated.",
        })
    return pd.DataFrame(rows)


def _ml_discrepancy(sample_cov: np.ndarray, model_cov: np.ndarray) -> float:
    p = sample_cov.shape[0]
    sign_s, logdet_s = np.linalg.slogdet(sample_cov)
    sign_m, logdet_m = np.linalg.slogdet(model_cov)
    if sign_s <= 0 or sign_m <= 0:
        return 1e12
    try:
        trace_term = float(np.trace(np.linalg.solve(model_cov, sample_cov)))
    except np.linalg.LinAlgError:
        return 1e12
    value = logdet_m + trace_term - logdet_s - p
    return float(value) if np.isfinite(value) else 1e12


def _fit_indices(sample_cov: np.ndarray, model_cov: np.ndarray, n: int, n_params: int) -> dict[str, float]:
    p = sample_cov.shape[0]
    f_model = max(_ml_discrepancy(sample_cov, model_cov), 0.0)
    chi_square = max((n - 1) * f_model, 0.0)
    df = p * (p + 1) / 2 - n_params
    chi_p = float(stats.chi2.sf(chi_square, df)) if df > 0 else np.nan

    independence = np.diag(np.diag(sample_cov))
    f_base = max(_ml_discrepancy(sample_cov, independence), 0.0)
    chi_base = max((n - 1) * f_base, 0.0)
    df_base = p * (p - 1) / 2
    denominator = max(chi_base - df_base, 1e-12)
    cfi = float(np.clip(1.0 - max(chi_square - max(df, 0), 0.0) / denominator, 0.0, 1.0))
    tli = ((chi_base / df_base) - (chi_square / df)) / ((chi_base / df_base) - 1) if df > 0 and df_base > 0 and chi_base > df_base else np.nan
    rmsea = math.sqrt(max((chi_square - df) / (df * max(n - 1, 1)), 0.0)) if df > 0 else np.nan

    sd = np.sqrt(np.diag(sample_cov))
    sample_corr = sample_cov / np.outer(sd, sd)
    model_sd = np.sqrt(np.diag(model_cov))
    model_corr = model_cov / np.outer(model_sd, model_sd)
    residual = sample_corr - model_corr
    off_diag = residual[np.triu_indices(p, k=1)]
    srmr = float(np.sqrt(np.mean(off_diag ** 2))) if off_diag.size else 0.0
    return {
        "chi_square": float(chi_square),
        "degrees_of_freedom": float(df),
        "chi_square_p": chi_p,
        "cfi": float(cfi),
        "tli": float(tli) if np.isfinite(tli) else np.nan,
        "rmsea": float(rmsea) if np.isfinite(rmsea) else np.nan,
        "srmr": srmr,
        "ml_discrepancy": f_model,
    }


# ---------------------------------------------------------------------------
# Exploratory factor analysis
# ---------------------------------------------------------------------------


def exploratory_factor_analysis(
    df: pd.DataFrame,
    items: list[str],
    n_factors: int | None = None,
    rotation: str = "varimax",
    parallel_iterations: int = 100,
    random_state: int = 42,
    alpha: float = 0.05,
) -> AnalysisResult:
    if len(items) < 3:
        raise ValueError("Exploratory factor analysis requires at least three items.")
    data = _complete_numeric(df, items)
    if len(data) < max(20, len(items) + 5):
        raise ValueError("The complete sample is too small for the selected number of items.")
    zero_variance = [column for column in items if float(data[column].var(ddof=0)) <= 1e-12]
    if zero_variance:
        raise ValueError(f"Remove constant items before factor analysis: {', '.join(zero_variance)}")

    z = StandardScaler().fit_transform(data)
    correlation = np.corrcoef(z, rowvar=False)
    bartlett_chi, bartlett_df, bartlett_p = _bartlett_sphericity(correlation, len(data))
    kmo_overall, kmo_item = _kmo(correlation)
    observed_eigen, random_eigen, parallel_count = _parallel_analysis(
        z, iterations=int(parallel_iterations), random_state=random_state
    )
    factor_count = int(n_factors or parallel_count)
    factor_count = max(1, min(factor_count, len(items) - 1))

    estimator = FactorAnalysis(n_components=factor_count, random_state=random_state, rotation=None)
    scores = estimator.fit_transform(z)
    loadings = estimator.components_.T
    rotation_matrix = np.eye(factor_count)
    if rotation == "varimax" and factor_count > 1:
        loadings, rotation_matrix = _varimax(loadings)
        scores = scores @ rotation_matrix

    communalities = np.sum(loadings ** 2, axis=1)
    uniqueness = np.clip(1.0 - communalities, 0.0, None)
    loading_table = pd.DataFrame(loadings, columns=[f"Factor_{i+1}" for i in range(factor_count)])
    loading_table.insert(0, "item", items)
    loading_table["communality"] = communalities
    loading_table["uniqueness"] = uniqueness
    loading_table["primary_factor"] = [f"Factor_{int(np.argmax(np.abs(row)))+1}" for row in loadings]
    loading_table["primary_loading"] = np.max(np.abs(loadings), axis=1)
    if factor_count > 1:
        sorted_abs = np.sort(np.abs(loadings), axis=1)
        loading_table["cross_loading_gap"] = sorted_abs[:, -1] - sorted_abs[:, -2]
    else:
        loading_table["cross_loading_gap"] = np.nan

    variance = np.sum(loadings ** 2, axis=0)
    variance_table = pd.DataFrame({
        "factor": [f"Factor_{i+1}" for i in range(factor_count)],
        "sum_squared_loadings": variance,
        "percent_variance": variance / len(items) * 100,
        "cumulative_percent": np.cumsum(variance / len(items) * 100),
    })
    retention_table = pd.DataFrame({
        "component": np.arange(1, len(items) + 1),
        "observed_eigenvalue": observed_eigen,
        "parallel_95th_percentile": random_eigen,
        "retain_by_parallel_analysis": observed_eigen > random_eigen,
    })
    kmo_table = pd.DataFrame({"item": items, "kmo": kmo_item})
    score_table = pd.DataFrame(scores, columns=[f"Factor_{i+1}_score" for i in range(factor_count)])
    score_table.insert(0, "source_row", data.index)

    diagnostics = pd.DataFrame([
        {
            "diagnostic": "Overall sampling adequacy",
            "test": "Kaiser-Meyer-Olkin",
            "statistic": kmo_overall,
            "p_value": np.nan,
            "status": "Satisfied" if kmo_overall >= 0.70 else "Minor concern" if kmo_overall >= 0.60 else "Material concern",
            "interpretation": f"Overall KMO={kmo_overall:.3f}.",
            "recommended_response": "Review weak items and the theoretical domain when KMO is below 0.60. Do not remove items solely to improve fit.",
        },
        {
            "diagnostic": "Factorability of the correlation matrix",
            "test": "Bartlett test of sphericity",
            "statistic": bartlett_chi,
            "p_value": bartlett_p,
            "status": "Satisfied" if bartlett_p < alpha else "Material concern",
            "interpretation": f"Chi-square({bartlett_df})={bartlett_chi:.3f}, p={bartlett_p:.4g}.",
            "recommended_response": "Factor analysis requires meaningful item correlations. Reconsider the item set if Bartlett's test is not significant.",
        },
        {
            "diagnostic": "Sample size relative to items",
            "test": "Cases-to-item ratio",
            "statistic": len(data) / len(items),
            "p_value": np.nan,
            "status": "Satisfied" if len(data) / len(items) >= 10 else "Minor concern" if len(data) / len(items) >= 5 else "Material concern",
            "interpretation": f"{len(data)} complete cases for {len(items)} items.",
            "recommended_response": "Interpret factor recovery cautiously when the ratio is low, especially with weak communalities or cross-loadings.",
        },
    ])
    if n_factors is not None and factor_count != parallel_count:
        diagnostics = pd.concat([diagnostics, pd.DataFrame([{
            "diagnostic": "Factor-retention decision",
            "test": "Parallel analysis comparison",
            "statistic": float(parallel_count),
            "p_value": np.nan,
            "status": "Minor concern",
            "interpretation": f"The user requested {factor_count} factor(s), while parallel analysis suggested {parallel_count}.",
            "recommended_response": "Report both the empirical retention evidence and the theoretical reason for the requested factor count.",
        }])], ignore_index=True)

    weak_items = loading_table.loc[(loading_table["primary_loading"] < 0.40) | (loading_table["communality"] < 0.20), "item"].tolist()
    warnings = []
    if weak_items:
        warnings.append("Weak item evidence was detected for: " + ", ".join(weak_items) + ". Review content validity before any deletion.")

    return AnalysisResult(
        method="Exploratory factor analysis",
        summary=(
            f"EFA used {len(data)} complete observations and retained {factor_count} factor(s). "
            f"Parallel analysis suggested {parallel_count}. Overall KMO was {kmo_overall:.3f}, and Bartlett's test p={bartlett_p:.4g}."
        ),
        tables={
            "Factor loadings": loading_table,
            "Factor variance explained": variance_table,
            "Parallel analysis": retention_table,
            "KMO by item": kmo_table,
            "Factor scores": score_table,
            "Item correlation matrix": pd.DataFrame(correlation, index=items, columns=items).reset_index(names="item"),
        },
        diagnostics=diagnostics,
        metadata={
            "n": len(data), "items": items, "n_factors": factor_count,
            "parallel_recommended_factors": parallel_count, "rotation": rotation,
        },
        warnings=warnings,
        treatment_log=[AuditEntry(
            action="Applied empirical factor-retention check",
            variable=", ".join(items),
            details=f"Parallel analysis with {parallel_iterations} simulated datasets suggested {parallel_count} factor(s).",
            justification="Parallel analysis provides a documented retention benchmark. The final factor count must also reflect theory and item content.",
            before_n=len(data), after_n=len(data),
        )],
        reproducible_code=(
            "# Standardise selected items, evaluate KMO and Bartlett tests, run parallel analysis,\n"
            "# estimate maximum-likelihood factors and apply varimax rotation when requested.\n"
        ),
    )


# ---------------------------------------------------------------------------
# Confirmatory factor analysis
# ---------------------------------------------------------------------------


@dataclass
class _MeasurementSpec:
    constructs: list[str]
    items: list[str]
    item_factor: np.ndarray
    free_loading_positions: list[tuple[int, int]]


def _measurement_spec(construct_map: dict[str, list[str]]) -> _MeasurementSpec:
    clean = {str(k).strip(): list(dict.fromkeys(v)) for k, v in construct_map.items() if str(k).strip() and v}
    if not clean:
        raise ValueError("Define at least one construct with observed items.")
    seen: set[str] = set()
    for construct, items in clean.items():
        if len(items) < 2:
            raise ValueError(f"Construct '{construct}' requires at least two observed items.")
        overlap = seen.intersection(items)
        if overlap:
            raise ValueError(f"Each item can load on only one construct in Phase 2. Repeated item(s): {', '.join(sorted(overlap))}")
        seen.update(items)
    constructs = list(clean)
    items = [item for construct in constructs for item in clean[construct]]
    item_factor = []
    free = []
    row = 0
    for factor_index, construct in enumerate(constructs):
        for item_index, _ in enumerate(clean[construct]):
            item_factor.append(factor_index)
            if item_index > 0:
                free.append((row, factor_index))
            row += 1
    return _MeasurementSpec(constructs, items, np.asarray(item_factor, dtype=int), free)


def _covariance_from_cholesky(values: np.ndarray, q: int) -> np.ndarray:
    lower = np.zeros((q, q), dtype=float)
    index = 0
    for i in range(q):
        for j in range(i + 1):
            lower[i, j] = math.exp(values[index]) if i == j else values[index]
            index += 1
    return lower @ lower.T


def _cfa_unpack(params: np.ndarray, spec: _MeasurementSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = len(spec.items)
    q = len(spec.constructs)
    loadings = np.zeros((p, q), dtype=float)
    for row, factor in enumerate(spec.item_factor):
        loadings[row, factor] = 1.0
    idx = 0
    for row, factor in spec.free_loading_positions:
        loadings[row, factor] = params[idx]
        idx += 1
    corr_count = q * (q + 1) // 2
    phi = _covariance_from_cholesky(params[idx: idx + corr_count], q)
    idx += corr_count
    theta = np.exp(params[idx: idx + p])
    return loadings, phi, theta


def _cfa_initial(spec: _MeasurementSpec, sample_corr: np.ndarray) -> np.ndarray:
    q = len(spec.constructs)
    free_loadings = np.full(len(spec.free_loading_positions), 0.7)
    chol = []
    for i in range(q):
        for j in range(i + 1):
            chol.append(0.0 if i == j else 0.05)
    theta = np.log(np.full(len(spec.items), 0.50))
    return np.concatenate([free_loadings, np.asarray(chol), theta])


def _fit_cfa_covariance(data: pd.DataFrame, construct_map: dict[str, list[str]], random_state: int = 42) -> dict[str, object]:
    spec = _measurement_spec(construct_map)
    numeric = _complete_numeric(data, spec.items)
    z = pd.DataFrame(StandardScaler().fit_transform(numeric), columns=spec.items, index=numeric.index)
    sample_cov = np.cov(z, rowvar=False, ddof=1)
    initial = _cfa_initial(spec, sample_cov)

    def objective(params: np.ndarray) -> float:
        loadings, phi, theta = _cfa_unpack(params, spec)
        sigma = loadings @ phi @ loadings.T + np.diag(theta)
        return _ml_discrepancy(sample_cov, sigma)

    rng = np.random.default_rng(random_state)
    best = None
    starts = [initial] + [initial + rng.normal(0, 0.05, size=initial.size) for _ in range(3)]
    for start in starts:
        candidate = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 2500, "ftol": 1e-10})
        if best is None or candidate.fun < best.fun:
            best = candidate
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("The CFA optimisation failed. Check item variation, sample size and model identification.")

    loadings, phi, theta = _cfa_unpack(best.x, spec)
    sigma = loadings @ phi @ loadings.T + np.diag(theta)
    fit = _fit_indices(sample_cov, sigma, len(z), len(best.x))

    latent_var = np.diag(phi)
    standardized = loadings * np.sqrt(latent_var)[None, :] / np.sqrt(np.diag(sigma))[:, None]
    loading_rows = []
    for row, item in enumerate(spec.items):
        factor = spec.item_factor[row]
        loading_rows.append({
            "construct": spec.constructs[factor],
            "item": item,
            "loading": loadings[row, factor],
            "standardized_loading": standardized[row, factor],
            "residual_variance": theta[row],
            "r_squared": standardized[row, factor] ** 2,
        })
    loading_table = pd.DataFrame(loading_rows)

    phi_sd = np.sqrt(np.diag(phi))
    phi_correlation = phi / np.outer(phi_sd, phi_sd)
    correlation_table = pd.DataFrame(phi_correlation, index=spec.constructs, columns=spec.constructs).reset_index(names="construct")
    quality_rows = []
    for factor_index, construct in enumerate(spec.constructs):
        subset = loading_table[loading_table["construct"] == construct]
        lambdas = subset["standardized_loading"].to_numpy(dtype=float)
        residuals = np.maximum(1.0 - lambdas ** 2, 0.0)
        cr = (lambdas.sum() ** 2) / ((lambdas.sum() ** 2) + residuals.sum()) if len(lambdas) else np.nan
        ave = float(np.mean(lambdas ** 2)) if len(lambdas) else np.nan
        quality_rows.append({"construct": construct, "composite_reliability": cr, "average_variance_extracted": ave, "sqrt_ave": math.sqrt(max(ave, 0))})
    quality = pd.DataFrame(quality_rows)

    sample_sd = np.sqrt(np.diag(sample_cov))
    model_sd = np.sqrt(np.diag(sigma))
    residual_corr = sample_cov / np.outer(sample_sd, sample_sd) - sigma / np.outer(model_sd, model_sd)
    residual_rows = []
    for i in range(len(spec.items)):
        for j in range(i + 1, len(spec.items)):
            residual_rows.append({"item_1": spec.items[i], "item_2": spec.items[j], "residual_correlation": residual_corr[i, j], "absolute_residual": abs(residual_corr[i, j])})
    residual_table = pd.DataFrame(residual_rows).sort_values("absolute_residual", ascending=False).reset_index(drop=True)

    return {
        "spec": spec, "data": z, "sample_cov": sample_cov, "model_cov": sigma,
        "loadings": loadings, "phi": phi, "theta": theta, "fit": fit,
        "loading_table": loading_table, "correlation_table": correlation_table,
        "quality_table": quality, "residual_table": residual_table,
        "optimizer": best,
    }


def confirmatory_factor_analysis(
    df: pd.DataFrame,
    construct_map: dict[str, list[str]],
    alpha: float = 0.05,
    random_state: int = 42,
) -> AnalysisResult:
    fitted = _fit_cfa_covariance(df, construct_map, random_state=random_state)
    fit = fitted["fit"]
    diagnostics = _fit_index_diagnostics(fit, alpha)
    weak = fitted["loading_table"][fitted["loading_table"]["standardized_loading"].abs() < 0.50]
    if not weak.empty:
        diagnostics = pd.concat([diagnostics, pd.DataFrame([{
            "diagnostic": "Indicator strength",
            "test": "Standardised loading review",
            "statistic": float(weak["standardized_loading"].abs().min()),
            "p_value": np.nan,
            "status": "Minor concern" if weak["standardized_loading"].abs().min() >= 0.40 else "Material concern",
            "interpretation": f"{len(weak)} indicator(s) had an absolute standardised loading below 0.50.",
            "recommended_response": "Review item wording and construct coverage. Remove an indicator only with theoretical and measurement justification, then disclose the re-specification.",
        }])], ignore_index=True)

    fit_table = pd.DataFrame([{"n": len(fitted["data"]), "constructs": len(fitted["spec"].constructs), "items": len(fitted["spec"].items), **fit, "optimizer_success": bool(fitted["optimizer"].success)}])
    warnings = []
    if not fitted["optimizer"].success:
        warnings.append("The optimiser returned a caution message. Treat the estimates as provisional and inspect convergence and residuals.")
    return AnalysisResult(
        method="Confirmatory factor analysis",
        summary=(
            f"CFA estimated {len(fitted['spec'].constructs)} construct(s) from {len(fitted['spec'].items)} items using {len(fitted['data'])} complete observations. "
            f"CFI={fit['cfi']:.3f}, TLI={fit['tli']:.3f}, RMSEA={fit['rmsea']:.3f}, and SRMR={fit['srmr']:.3f}."
        ),
        tables={
            "CFA fit indices": fit_table,
            "CFA standardised loadings": fitted["loading_table"],
            "Latent factor correlations": fitted["correlation_table"],
            "Construct reliability and validity": fitted["quality_table"],
            "Largest residual correlations": fitted["residual_table"].head(20),
            "Observed covariance matrix": pd.DataFrame(fitted["sample_cov"], index=fitted["spec"].items, columns=fitted["spec"].items).reset_index(names="item"),
            "Model-implied covariance matrix": pd.DataFrame(fitted["model_cov"], index=fitted["spec"].items, columns=fitted["spec"].items).reset_index(names="item"),
        },
        diagnostics=diagnostics,
        metadata={"construct_map": construct_map, "n": len(fitted["data"]), "estimator": "Maximum-likelihood covariance fitting"},
        warnings=warnings,
        treatment_log=[AuditEntry(
            action="Estimated prespecified measurement model",
            variable=", ".join(fitted["spec"].items),
            details="Each item was assigned to one construct, with factor scaling imposed for model identification.",
            justification="The CFA model was fitted as specified. Residual correlations are diagnostic evidence, not automatic permission to correlate errors or delete indicators.",
            before_n=len(fitted["data"]), after_n=len(fitted["data"]),
        )],
        reproducible_code="# Standardise items and fit the prespecified CFA covariance model by maximum likelihood; report fit indices, loadings, reliability, AVE and residual correlations.\n",
    )


# ---------------------------------------------------------------------------
# Covariance-based structural equation modelling
# ---------------------------------------------------------------------------


def _validate_paths(constructs: list[str], paths: list[tuple[str, str]]) -> None:
    seen = set()
    for predictor, outcome in paths:
        if predictor not in constructs or outcome not in constructs:
            raise ValueError(f"Structural path '{predictor} -> {outcome}' uses an undefined construct.")
        if predictor == outcome:
            raise ValueError("A construct cannot predict itself.")
        if (predictor, outcome) in seen:
            raise ValueError(f"Duplicate path: {predictor} -> {outcome}")
        seen.add((predictor, outcome))
    graph = {name: [] for name in constructs}
    for predictor, outcome in paths:
        graph[predictor].append(outcome)
    visiting, visited = set(), set()
    def visit(node: str):
        if node in visiting:
            raise ValueError("The structural paths contain a directed cycle. Phase 2 requires an acyclic model.")
        if node in visited:
            return
        visiting.add(node)
        for nxt in graph[node]:
            visit(nxt)
        visiting.remove(node)
        visited.add(node)
    for node in constructs:
        visit(node)


def structural_equation_model(
    df: pd.DataFrame,
    construct_map: dict[str, list[str]],
    paths: list[tuple[str, str]],
    alpha: float = 0.05,
    random_state: int = 42,
) -> AnalysisResult:
    spec = _measurement_spec(construct_map)
    _validate_paths(spec.constructs, paths)
    data = _complete_numeric(df, spec.items)
    z = pd.DataFrame(StandardScaler().fit_transform(data), columns=spec.items, index=data.index)
    sample_cov = np.cov(z, rowvar=False, ddof=1)
    p = len(spec.items)
    q = len(spec.constructs)
    index_map = {name: i for i, name in enumerate(spec.constructs)}
    endogenous = sorted({outcome for _, outcome in paths}, key=spec.constructs.index)
    exogenous = [name for name in spec.constructs if name not in endogenous]
    if not exogenous:
        raise ValueError("The structural model needs at least one exogenous construct.")
    path_positions = [(index_map[outcome], index_map[predictor], predictor, outcome) for predictor, outcome in paths]

    # Parameters: non-marker loadings, path coefficients, exogenous covariance Cholesky,
    # endogenous disturbance log-variances, indicator residual log-variances.
    n_free_loadings = len(spec.free_loading_positions)
    n_paths = len(path_positions)
    n_exog_cov = len(exogenous) * (len(exogenous) + 1) // 2
    n_disturbance = len(endogenous)
    initial = np.concatenate([
        np.full(n_free_loadings, 0.7),
        np.full(n_paths, 0.20),
        np.asarray([0.0 if i == j else 0.05 for i in range(len(exogenous)) for j in range(i + 1)]),
        np.log(np.full(n_disturbance, 0.50)),
        np.log(np.full(p, 0.50)),
    ])

    exog_indices = [index_map[name] for name in exogenous]
    endo_indices = [index_map[name] for name in endogenous]

    def unpack(params: np.ndarray):
        loadings = np.zeros((p, q))
        for row, factor in enumerate(spec.item_factor):
            loadings[row, factor] = 1.0
        idx = 0
        for row, factor in spec.free_loading_positions:
            loadings[row, factor] = params[idx]
            idx += 1
        B = np.zeros((q, q))
        for outcome_i, predictor_i, _, _ in path_positions:
            B[outcome_i, predictor_i] = params[idx]
            idx += 1
        exog_cov = _covariance_from_cholesky(params[idx: idx + n_exog_cov], len(exogenous))
        idx += n_exog_cov
        disturbances = np.exp(params[idx: idx + n_disturbance])
        idx += n_disturbance
        theta = np.exp(params[idx: idx + p])
        psi = np.zeros((q, q))
        for local_i, global_i in enumerate(exog_indices):
            for local_j, global_j in enumerate(exog_indices):
                psi[global_i, global_j] = exog_cov[local_i, local_j]
        for local_i, global_i in enumerate(endo_indices):
            psi[global_i, global_i] = disturbances[local_i]
        try:
            inv = np.linalg.inv(np.eye(q) - B)
        except np.linalg.LinAlgError:
            return loadings, B, psi, theta, None, None
        latent_cov = inv @ psi @ inv.T
        sigma = loadings @ latent_cov @ loadings.T + np.diag(theta)
        return loadings, B, psi, theta, latent_cov, sigma

    def objective(params: np.ndarray) -> float:
        *_, sigma = unpack(params)
        if sigma is None:
            return 1e12
        return _ml_discrepancy(sample_cov, sigma)

    rng = np.random.default_rng(random_state)
    best = None
    for start in [initial] + [initial + rng.normal(0, 0.04, size=initial.size) for _ in range(3)]:
        candidate = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 3500, "ftol": 1e-10})
        if best is None or candidate.fun < best.fun:
            best = candidate
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("The SEM optimisation failed. Check model identification, sample size and construct specification.")
    loadings, B, psi, theta, latent_cov, sigma = unpack(best.x)
    fit = _fit_indices(sample_cov, sigma, len(z), len(best.x))

    # Approximate standard errors from the inverse Hessian returned by L-BFGS-B.
    try:
        inverse_hessian = np.asarray(best.hess_inv.todense(), dtype=float) / max(len(z) - 1, 1)
        se_all = np.sqrt(np.maximum(np.diag(inverse_hessian), 0.0))
    except Exception:
        se_all = np.full(len(best.x), np.nan)
    path_start = n_free_loadings
    latent_sd = np.sqrt(np.diag(latent_cov))
    path_rows = []
    for position, (outcome_i, predictor_i, predictor, outcome) in enumerate(path_positions):
        estimate = B[outcome_i, predictor_i]
        se = se_all[path_start + position] if path_start + position < len(se_all) else np.nan
        z_stat = estimate / se if np.isfinite(se) and se > 0 else np.nan
        p_value = float(2 * stats.norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
        standardized = estimate * latent_sd[predictor_i] / latent_sd[outcome_i] if latent_sd[outcome_i] > 0 else np.nan
        path_rows.append({
            "predictor": predictor, "outcome": outcome, "estimate": estimate,
            "std_error_approx": se, "z_statistic_approx": z_stat,
            "p_value_approx": p_value, "standardized_estimate": standardized,
        })
    path_table = pd.DataFrame(path_rows)

    loading_rows = []
    model_diag = np.diag(sigma)
    for row, item in enumerate(spec.items):
        factor_i = spec.item_factor[row]
        standardized = loadings[row, factor_i] * latent_sd[factor_i] / math.sqrt(max(model_diag[row], 1e-12))
        loading_rows.append({"construct": spec.constructs[factor_i], "item": item, "loading": loadings[row, factor_i], "standardized_loading": standardized, "residual_variance": theta[row]})
    loading_table = pd.DataFrame(loading_rows)
    latent_cov_table = pd.DataFrame(latent_cov, index=spec.constructs, columns=spec.constructs).reset_index(names="construct")
    fit_table = pd.DataFrame([{"n": len(z), "constructs": q, "items": p, "paths": len(paths), **fit, "optimizer_success": bool(best.success)}])
    diagnostics = _fit_index_diagnostics(fit, alpha)

    warnings = [
        "SEM standard errors and p-values are numerical approximations from the optimisation Hessian. Confirm important results in a specialist SEM package before publication."
    ]
    if not best.success:
        warnings.append("The optimiser returned a caution message. Treat the model as provisional.")

    return AnalysisResult(
        method="Covariance-based structural equation model",
        summary=(
            f"SEM estimated {len(paths)} structural path(s) among {q} latent constructs using {len(z)} complete observations. "
            f"CFI={fit['cfi']:.3f}, TLI={fit['tli']:.3f}, RMSEA={fit['rmsea']:.3f}, and SRMR={fit['srmr']:.3f}."
        ),
        tables={
            "SEM fit indices": fit_table,
            "Structural path estimates": path_table,
            "SEM standardised loadings": loading_table,
            "Latent covariance matrix": latent_cov_table,
            "Observed covariance matrix": pd.DataFrame(sample_cov, index=spec.items, columns=spec.items).reset_index(names="item"),
            "Model-implied covariance matrix": pd.DataFrame(sigma, index=spec.items, columns=spec.items).reset_index(names="item"),
        },
        diagnostics=diagnostics,
        metadata={"construct_map": construct_map, "paths": paths, "n": len(z), "estimator": "Maximum-likelihood covariance fitting"},
        warnings=warnings,
        treatment_log=[AuditEntry(
            action="Estimated prespecified latent-variable structural model",
            variable=", ".join(spec.constructs),
            details=f"Fitted {len(paths)} directed structural path(s) with an acyclic latent model.",
            justification="The model was fitted as specified. Weak fit should trigger theory-led review and independent validation, not automated path addition or deletion.",
            before_n=len(z), after_n=len(z),
        )],
        reproducible_code="# Fit the measurement and acyclic structural covariance model simultaneously by maximum likelihood; report fit, loadings and structural paths.\n",
    )


# ---------------------------------------------------------------------------
# Repeated measures and mixed effects
# ---------------------------------------------------------------------------


def _mauchly_sphericity(matrix: np.ndarray) -> tuple[float, float, int, float, float]:
    n, k = matrix.shape
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    centering = np.eye(k) - np.ones((k, k)) / k
    transformed = centering @ covariance @ centering
    eigen = np.linalg.eigvalsh(transformed)
    eigen = eigen[eigen > 1e-10]
    d = k - 1
    if len(eigen) < d or np.any(eigen <= 0):
        return np.nan, np.nan, int(k * (k - 1) / 2 - 1), np.nan, np.nan
    w = float(np.prod(eigen) / (np.mean(eigen) ** d))
    correction = n - 1 - (2 * d + 1) / 6
    chi2 = -correction * math.log(max(w, 1e-12))
    df = int(k * (k - 1) / 2 - 1)
    p = float(stats.chi2.sf(chi2, df)) if df > 0 else np.nan
    gg = float((eigen.sum() ** 2) / (d * np.sum(eigen ** 2)))
    return w, chi2, df, p, gg


def repeated_measures_anova(
    df: pd.DataFrame,
    measurements: list[str],
    subject_id: str | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    if len(measurements) < 2:
        raise ValueError("Select at least two repeated measurements.")
    columns = measurements + ([subject_id] if subject_id else [])
    data = df.loc[:, columns].copy()
    for column in measurements:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=measurements)
    if len(data) < 5:
        raise ValueError("At least five complete subjects are required.")
    data["__subject__"] = data[subject_id].astype(str) if subject_id else data.index.astype(str)
    long = data.melt(id_vars="__subject__", value_vars=measurements, var_name="condition", value_name="outcome")
    model = AnovaRM(long, depvar="outcome", subject="__subject__", within=["condition"]).fit()
    anova = model.anova_table.reset_index(names="effect")
    anova = anova.rename(columns={"F Value": "f_statistic", "Num DF": "df_numerator", "Den DF": "df_denominator", "Pr > F": "p_value"})

    values = data[measurements].to_numpy(dtype=float)
    w, mauchly_chi, mauchly_df, mauchly_p, gg = _mauchly_sphericity(values) if len(measurements) > 2 else (1.0, 0.0, 0, 1.0, 1.0)
    f_value = float(anova.iloc[0]["f_statistic"])
    df1 = float(anova.iloc[0]["df_numerator"])
    df2 = float(anova.iloc[0]["df_denominator"])
    gg_p = float(stats.f.sf(f_value, df1 * gg, df2 * gg)) if np.isfinite(gg) else np.nan
    use_gg = len(measurements) > 2 and np.isfinite(mauchly_p) and mauchly_p < alpha
    selected_p = gg_p if use_gg else float(anova.iloc[0]["p_value"])
    anova["greenhouse_geisser_epsilon"] = gg
    anova["greenhouse_geisser_corrected_p"] = gg_p
    anova["selected_p_value"] = selected_p

    desc = pd.DataFrame([{
        "condition": column, "n": int(data[column].notna().sum()), "mean": float(data[column].mean()),
        "std_dev": float(data[column].std(ddof=1)), "median": float(data[column].median()),
    } for column in measurements])
    pairwise_rows = []
    raw_p = []
    pairs = []
    for i in range(len(measurements)):
        for j in range(i + 1, len(measurements)):
            a, b = measurements[i], measurements[j]
            diff = data[b] - data[a]
            t, p = stats.ttest_rel(data[b], data[a])
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan
            pairs.append((a, b, t, p, float(diff.mean()), dz))
            raw_p.append(p)
    adjusted = multipletests(raw_p, alpha=alpha, method="holm")[1] if raw_p else []
    for (a, b, t, p, mean_diff, dz), adj in zip(pairs, adjusted):
        pairwise_rows.append({"condition_1": a, "condition_2": b, "mean_difference_2_minus_1": mean_diff, "t_statistic": t, "raw_p_value": p, "holm_adjusted_p": adj, "cohens_dz": dz})

    diagnostics = pd.DataFrame([{
        "diagnostic": "Sphericity",
        "test": "Mauchly approximate test",
        "statistic": w,
        "p_value": mauchly_p,
        "status": "Satisfied" if len(measurements) <= 2 or mauchly_p >= alpha else "Material concern",
        "interpretation": "Sphericity is automatic with two conditions." if len(measurements) <= 2 else f"W={w:.3f}; chi-square({mauchly_df})={mauchly_chi:.3f}; epsilon={gg:.3f}.",
        "recommended_response": "Use Greenhouse-Geisser corrected degrees of freedom and p-value when sphericity is not supported." if use_gg else "Use the uncorrected repeated-measures result, subject to the design assumptions.",
    }])
    for column in measurements:
        diagnostics = pd.concat([diagnostics, pd.DataFrame([normality_diagnostic(data[column], alpha, column)])], ignore_index=True)

    treatment = []
    if use_gg:
        treatment.append(AuditEntry(
            action="Applied Greenhouse-Geisser correction",
            variable=", ".join(measurements),
            details=f"Adjusted repeated-measures degrees of freedom using epsilon={gg:.3f}.",
            justification="The approximate Mauchly test raised a material sphericity concern. No observations were changed.",
            before_n=len(data), after_n=len(data),
        ))
    return AnalysisResult(
        method="Repeated-measures ANOVA",
        summary=f"Repeated-measures ANOVA analysed {len(data)} complete subjects. F={f_value:.3f}, selected p={selected_p:.4g}. {'Greenhouse-Geisser correction was used.' if use_gg else 'No sphericity correction was required.'}",
        tables={"Repeated-measures descriptives": desc, "Repeated-measures ANOVA": anova, "Holm-adjusted pairwise comparisons": pd.DataFrame(pairwise_rows)},
        diagnostics=diagnostics,
        metadata={"n_subjects": len(data), "measurements": measurements, "greenhouse_geisser_used": use_gg},
        treatment_log=treatment,
        reproducible_code="# Reshape wide repeated measures to long form, fit AnovaRM, evaluate sphericity and apply Greenhouse-Geisser correction when required.\n",
    )


def mixed_effects_model(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    cluster: str,
    random_slope: str | None = None,
    reml: bool = True,
    alpha: float = 0.05,
) -> AnalysisResult:
    if not predictors:
        raise ValueError("Select at least one fixed-effect predictor.")
    columns = [outcome, cluster] + predictors + ([random_slope] if random_slope else [])
    columns = list(dict.fromkeys(columns))
    data = df.loc[:, columns].copy()
    for column in [outcome] + predictors + ([random_slope] if random_slope else []):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data[cluster].nunique() < 3:
        raise ValueError("Mixed-effects modelling requires at least three clusters.")
    if len(data) <= len(predictors) + data[cluster].nunique():
        raise ValueError("The sample is too small for the requested mixed model.")

    safe = data.rename(columns={column: f"v{i}" for i, column in enumerate(columns)})
    reverse = {f"v{i}": column for i, column in enumerate(columns)}
    y_name = next(k for k, v in reverse.items() if v == outcome)
    cluster_name = next(k for k, v in reverse.items() if v == cluster)
    predictor_names = [next(k for k, v in reverse.items() if v == column) for column in predictors]
    formula = y_name + " ~ " + " + ".join(predictor_names)
    re_formula = "1"
    if random_slope:
        slope_name = next(k for k, v in reverse.items() if v == random_slope)
        re_formula = f"1 + {slope_name}"
    model = mixedlm(formula, safe, groups=safe[cluster_name], re_formula=re_formula).fit(reml=reml, method="lbfgs", maxiter=1000, disp=False)

    fixed = pd.DataFrame({
        "term": [reverse.get(term, term) for term in model.fe_params.index],
        "estimate": model.fe_params.values,
        "std_error": model.bse_fe.values,
        "statistic": model.fe_params.values / model.bse_fe.values,
    })
    fixed["p_value"] = 2 * stats.norm.sf(np.abs(fixed["statistic"]))
    fixed["ci_lower"] = fixed["estimate"] - stats.norm.ppf(1 - alpha / 2) * fixed["std_error"]
    fixed["ci_upper"] = fixed["estimate"] + stats.norm.ppf(1 - alpha / 2) * fixed["std_error"]

    covariance = np.asarray(model.cov_re)
    random_intercept_variance = float(covariance[0, 0]) if covariance.size else np.nan
    residual_variance = float(model.scale)
    icc = random_intercept_variance / (random_intercept_variance + residual_variance) if random_intercept_variance + residual_variance > 0 else np.nan
    cluster_sizes = data.groupby(cluster).size()
    fit_table = pd.DataFrame([{
        "n": len(data), "clusters": int(data[cluster].nunique()), "minimum_cluster_size": int(cluster_sizes.min()),
        "median_cluster_size": float(cluster_sizes.median()), "maximum_cluster_size": int(cluster_sizes.max()),
        "log_likelihood": model.llf, "aic": model.aic, "bic": model.bic,
        "random_intercept_variance": random_intercept_variance, "residual_variance": residual_variance,
        "intraclass_correlation": icc, "converged": bool(model.converged), "reml": reml,
    }])
    random_cov = pd.DataFrame(covariance)
    random_cov.insert(0, "random_effect", ["Intercept"] + ([random_slope] if random_slope and covariance.shape[0] > 1 else []))

    residual_diag = normality_diagnostic(pd.Series(model.resid), alpha, "mixed-model residuals")
    diagnostics = pd.DataFrame([
        residual_diag,
        {
            "diagnostic": "Cluster support",
            "test": "Number and size of clusters",
            "statistic": float(data[cluster].nunique()),
            "p_value": np.nan,
            "status": "Satisfied" if data[cluster].nunique() >= 30 else "Minor concern" if data[cluster].nunique() >= 10 else "Material concern",
            "interpretation": f"The model contains {data[cluster].nunique()} clusters; minimum cluster size={cluster_sizes.min()}.",
            "recommended_response": "Use cautious inference with few clusters and consider small-sample corrections or bootstrap validation in a specialist package.",
        },
        {
            "diagnostic": "Optimisation convergence",
            "test": "MixedLM convergence flag",
            "statistic": float(bool(model.converged)),
            "p_value": np.nan,
            "status": "Satisfied" if model.converged else "Material concern",
            "interpretation": "The optimiser converged." if model.converged else "The optimiser did not confirm convergence.",
            "recommended_response": "Simplify the random-effects structure, rescale predictors, or collect more cluster information when convergence fails.",
        },
    ])
    warnings = [] if model.converged else ["The mixed-effects model did not converge and should not be used for substantive inference."]
    return AnalysisResult(
        method="Linear mixed-effects model",
        summary=f"The mixed model analysed {len(data)} observations nested in {data[cluster].nunique()} clusters. The estimated ICC was {icc:.3f}. Convergence status: {model.converged}.",
        tables={"Fixed effects": fixed, "Mixed-model fit": fit_table, "Random-effects covariance": random_cov, "Cluster sizes": cluster_sizes.reset_index(name="observations")},
        diagnostics=diagnostics,
        metadata={"outcome": outcome, "predictors": predictors, "cluster": cluster, "random_slope": random_slope, "reml": reml},
        warnings=warnings,
        treatment_log=[AuditEntry(
            action="Modelled clustered dependence",
            variable=cluster,
            details=f"Estimated a random-intercept{' and random-slope' if random_slope else ''} model.",
            justification="The mixed model represents non-independence within clusters rather than treating clustered observations as independent.",
            before_n=len(data), after_n=len(data),
        )],
        reproducible_code="# Fit statsmodels MixedLM with fixed effects, cluster-level random intercept and optional random slope; report ICC and convergence.\n",
    )


# ---------------------------------------------------------------------------
# Panel-data analysis
# ---------------------------------------------------------------------------


def _cluster_robust_ols(y: pd.Series, x: pd.DataFrame, groups: pd.Series, add_constant: bool = True):
    design = sm.add_constant(x, has_constant="add") if add_constant else x
    return sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups})


def panel_data_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    entity: str,
    time: str,
    model_choice: str = "automatic",
    include_time_effects: bool = False,
    alpha: float = 0.05,
) -> AnalysisResult:
    if not predictors:
        raise ValueError("Select at least one panel predictor.")
    columns = [outcome] + predictors + [entity, time]
    data = df.loc[:, columns].copy()
    for column in [outcome] + predictors:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values([entity, time]).copy()
    if data[entity].nunique() < 3 or data[time].nunique() < 2:
        raise ValueError("Panel analysis requires at least three entities and two time periods.")
    if data.duplicated([entity, time]).any():
        raise ValueError("Entity-time combinations must be unique.")

    x_base = data[predictors].astype(float).copy()
    if include_time_effects:
        time_dummies = pd.get_dummies(data[time].astype(str), prefix="time", drop_first=True, dtype=float)
        x_base = pd.concat([x_base.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
        data = data.reset_index(drop=True)
    y = data[outcome].astype(float)
    groups = data[entity]

    pooled = _cluster_robust_ols(y, x_base, groups)

    group_means_y = data.groupby(entity)[outcome].transform("mean")
    group_means_x = x_base.groupby(groups.values).transform("mean")
    y_within = y - group_means_y
    x_within = x_base - group_means_x
    varying = [column for column in x_within.columns if float(x_within[column].var(ddof=0)) > 1e-12]
    if not varying:
        raise ValueError("No predictor varies within entities, so fixed effects cannot be estimated.")
    fixed = _cluster_robust_ols(y_within, x_within[varying], groups, add_constant=False)

    # Entity-effects F test using conventional sums of squares.
    pooled_plain = sm.OLS(y, sm.add_constant(x_base, has_constant="add")).fit()
    fixed_plain = sm.OLS(y_within, x_within[varying]).fit()
    n, n_entities, k = len(data), data[entity].nunique(), len(varying)
    numerator_df = n_entities - 1
    denominator_df = max(n - n_entities - k, 1)
    f_entity = ((pooled_plain.ssr - fixed_plain.ssr) / max(numerator_df, 1)) / (fixed_plain.ssr / denominator_df)
    f_entity = max(float(f_entity), 0.0)
    p_entity = float(stats.f.sf(f_entity, numerator_df, denominator_df))

    # Swamy-Arora style random-effects transformation.
    sigma_e2 = float(fixed_plain.ssr / denominator_df)
    entity_means = data.groupby(entity)[[outcome] + predictors].mean()
    between_y = entity_means[outcome]
    between_x = sm.add_constant(entity_means[predictors], has_constant="add")
    between = sm.OLS(between_y, between_x).fit()
    t_bar = float(data.groupby(entity).size().mean())
    sigma_u2_raw = float(between.resid.var(ddof=max(len(predictors) + 1, 1)) - sigma_e2 / max(t_bar, 1))
    sigma_u2 = max(sigma_u2_raw, 0.0) if np.isfinite(sigma_u2_raw) else 0.0
    counts = data.groupby(entity)[entity].transform("size").astype(float)
    theta = 1.0 - np.sqrt(sigma_e2 / np.maximum(sigma_e2 + counts * sigma_u2, 1e-12))
    re_y = y - theta * group_means_y
    re_x = x_base.copy()
    for column in re_x.columns:
        re_x[column] = x_base[column] - theta * group_means_x[column]
    random = _cluster_robust_ols(re_y, re_x, groups, add_constant=True)

    common = [column for column in varying if column in random.params.index]
    hausman_stat, hausman_p, hausman_df = np.nan, np.nan, len(common)
    if common:
        b_fe = fixed.params[common].to_numpy()
        b_re = random.params[common].to_numpy()
        cov_diff = fixed.cov_params().loc[common, common].to_numpy() - random.cov_params().loc[common, common].to_numpy()
        try:
            hausman_stat = float((b_fe - b_re).T @ np.linalg.pinv(cov_diff) @ (b_fe - b_re))
            hausman_stat = max(hausman_stat, 0.0)
            hausman_p = float(stats.chi2.sf(hausman_stat, len(common)))
        except Exception:
            pass

    if model_choice == "automatic":
        if p_entity >= alpha:
            selected_name, selected = "Pooled OLS with entity-clustered standard errors", pooled
            decision = "The entity-effects test did not indicate material unobserved entity heterogeneity."
        elif np.isfinite(hausman_p) and hausman_p < alpha:
            selected_name, selected = "Entity fixed-effects model", fixed
            decision = "Entity effects were material and the Hausman comparison favoured fixed effects."
        else:
            selected_name, selected = "Random-effects model", random
            decision = "Entity effects were material, while the Hausman comparison did not reject the random-effects specification."
    elif model_choice == "fixed":
        selected_name, selected, decision = "Entity fixed-effects model", fixed, "The user selected fixed effects."
    elif model_choice == "random":
        selected_name, selected, decision = "Random-effects model", random, "The user selected random effects."
    else:
        selected_name, selected, decision = "Pooled OLS with entity-clustered standard errors", pooled, "The user selected pooled OLS."

    def tidy_panel(model, label: str) -> pd.DataFrame:
        table = _tidy_model(model)
        table.insert(0, "model", label)
        return table

    model_comparison = pd.DataFrame([
        {"model": "Pooled OLS", "n": n, "r_squared": pooled.rsquared, "adjusted_r_squared": pooled.rsquared_adj},
        {"model": "Entity fixed effects", "n": n, "r_squared_within": fixed.rsquared, "adjusted_r_squared_within": fixed.rsquared_adj},
        {"model": "Random effects", "n": n, "r_squared_transformed": random.rsquared, "adjusted_r_squared_transformed": random.rsquared_adj},
    ])
    decision_table = pd.DataFrame([{
        "selected_model": selected_name, "entity_effects_f": f_entity, "entity_effects_p": p_entity,
        "hausman_chi_square": hausman_stat, "hausman_df": hausman_df, "hausman_p": hausman_p,
        "sigma_entity": sigma_u2, "sigma_idiosyncratic": sigma_e2, "decision": decision,
    }])

    # Approximate within-entity residual serial correlation diagnostic.
    selected_resid = pd.Series(np.asarray(selected.resid), index=data.index)
    lagged = selected_resid.groupby(data[entity]).shift(1)
    valid = pd.concat([selected_resid.rename("resid"), lagged.rename("lag")], axis=1).dropna()
    rho = float(valid.corr().iloc[0, 1]) if len(valid) > 2 else np.nan
    diagnostics = pd.DataFrame([
        {
            "diagnostic": "Unobserved entity effects",
            "test": "Pooled versus entity fixed-effects F test",
            "statistic": f_entity,
            "p_value": p_entity,
            "status": "Satisfied" if p_entity >= alpha else "Material concern",
            "interpretation": "No material entity effects detected." if p_entity >= alpha else "Entity-specific heterogeneity is material.",
            "recommended_response": "Use fixed or random effects rather than pooled OLS when entity effects are material.",
        },
        {
            "diagnostic": "Fixed versus random effects",
            "test": "Hausman comparison",
            "statistic": hausman_stat,
            "p_value": hausman_p,
            "status": "Cannot determine" if not np.isfinite(hausman_p) else "Material concern" if hausman_p < alpha else "Satisfied",
            "interpretation": "A small p-value favours fixed effects because random-effects orthogonality is doubtful.",
            "recommended_response": "Use theory and the Hausman result jointly. Report both specifications when the decision is sensitive.",
        },
        {
            "diagnostic": "Within-entity residual dependence",
            "test": "Lag-one residual correlation screening",
            "statistic": rho,
            "p_value": np.nan,
            "status": "Cannot determine" if not np.isfinite(rho) else "Satisfied" if abs(rho) < 0.30 else "Minor concern" if abs(rho) < 0.50 else "Material concern",
            "interpretation": f"Approximate lag-one residual correlation={rho:.3f}." if np.isfinite(rho) else "Insufficient repeated residual pairs.",
            "recommended_response": "Retain entity-clustered inference and consider dynamic or serial-correlation models when dependence is material.",
        },
    ])
    return AnalysisResult(
        method=selected_name,
        summary=f"Panel analysis used {n} observations from {n_entities} entities across {data[time].nunique()} time periods. {decision} Selected model: {selected_name}.",
        tables={
            "Panel model decision": decision_table,
            "Selected panel coefficients": tidy_panel(selected, selected_name),
            "Pooled OLS coefficients": tidy_panel(pooled, "Pooled OLS"),
            "Fixed-effects coefficients": tidy_panel(fixed, "Entity fixed effects"),
            "Random-effects coefficients": tidy_panel(random, "Random effects"),
            "Panel model comparison": model_comparison,
            "Panel structure": data.groupby(entity).size().describe().rename_axis("measure").reset_index(name="value"),
        },
        diagnostics=diagnostics,
        metadata={"outcome": outcome, "predictors": predictors, "entity": entity, "time": time, "selected_model": selected_name, "include_time_effects": include_time_effects},
        treatment_log=[AuditEntry(
            action="Compared pooled, fixed-effects and random-effects panel specifications",
            variable=entity,
            details=f"Selected {selected_name} using the requested rule and reported entity-clustered standard errors.",
            justification="Panel dependence and time-invariant entity heterogeneity require explicit model comparison rather than ordinary cross-sectional OLS.",
            before_n=n, after_n=n,
        )],
        reproducible_code="# Sort entity-time observations; estimate pooled OLS, within-entity fixed effects and quasi-demeaned random effects; compare entity-effects and Hausman diagnostics.\n",
    )


# ---------------------------------------------------------------------------
# Advanced moderation and mediation
# ---------------------------------------------------------------------------


def advanced_moderation_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    moderator: str,
    controls: list[str] | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    controls = controls or []
    data = _complete_numeric(df, [outcome, predictor, moderator] + controls)
    x_centered = data[predictor] - data[predictor].mean()
    w_centered = data[moderator] - data[moderator].mean()
    interaction_name = f"{predictor}_x_{moderator}"
    design = pd.DataFrame({f"c_{predictor}": x_centered, f"c_{moderator}": w_centered, interaction_name: x_centered * w_centered}, index=data.index)
    for control in controls:
        design[control] = data[control]
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(data[outcome], design).fit(cov_type="HC3")
    table = _tidy_model(model)
    b1 = float(model.params[f"c_{predictor}"])
    b3 = float(model.params[interaction_name])
    cov = model.cov_params()
    w_sd = float(data[moderator].std(ddof=1))
    levels = [(-w_sd, "Minus 1 SD"), (0.0, "Mean"), (w_sd, "Plus 1 SD")]
    slope_rows = []
    critical = stats.norm.ppf(1 - alpha / 2)
    for value, label in levels:
        slope = b1 + b3 * value
        variance = cov.loc[f"c_{predictor}", f"c_{predictor}"] + value ** 2 * cov.loc[interaction_name, interaction_name] + 2 * value * cov.loc[f"c_{predictor}", interaction_name]
        se = math.sqrt(max(float(variance), 0.0))
        z = slope / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        slope_rows.append({"moderator_level": label, "centered_moderator_value": value, "simple_slope": slope, "std_error": se, "z_statistic": z, "p_value": p, "ci_lower": slope - critical * se, "ci_upper": slope + critical * se})

    # Johnson-Neyman roots from (b1+b3*w)^2 = zcrit^2 Var(b1+b3*w).
    v11 = float(cov.loc[f"c_{predictor}", f"c_{predictor}"])
    v33 = float(cov.loc[interaction_name, interaction_name])
    v13 = float(cov.loc[f"c_{predictor}", interaction_name])
    a = b3 ** 2 - critical ** 2 * v33
    b = 2 * b1 * b3 - 2 * critical ** 2 * v13
    c = b1 ** 2 - critical ** 2 * v11
    roots = np.roots([a, b, c]) if abs(a) > 1e-12 else np.roots([b, c]) if abs(b) > 1e-12 else np.asarray([])
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < 1e-7)
    jn_table = pd.DataFrame([{"root_number": i + 1, "centered_moderator_value": root, "original_moderator_value": root + data[moderator].mean()} for i, root in enumerate(real_roots)])
    diagnostics = ols_diagnostics(model, alpha)
    vif = vif_table(design.drop(columns="const", errors="ignore"))
    if not vif.empty:
        tables_vif = vif
    else:
        tables_vif = pd.DataFrame()
    interaction = table[table["term"] == interaction_name].iloc[0]
    return AnalysisResult(
        method="Advanced moderated regression",
        summary=f"The interaction estimate was {interaction['estimate']:.4g}, p={interaction['p_value']:.4g}. Conditional slopes were estimated at the moderator mean and plus or minus one standard deviation, with Johnson-Neyman boundaries where estimable.",
        tables={"Moderation coefficients": table, "Conditional simple slopes": pd.DataFrame(slope_rows), "Johnson-Neyman boundaries": jn_table, "Moderation VIF": tables_vif, "Model fit": pd.DataFrame([{"n": len(data), "r_squared": model.rsquared, "adjusted_r_squared": model.rsquared_adj, "f_statistic": model.fvalue, "f_p_value": model.f_pvalue}])},
        diagnostics=diagnostics,
        metadata={"interaction": interaction_name, "moderator_mean": float(data[moderator].mean()), "moderator_sd": w_sd},
        treatment_log=[AuditEntry(
            action="Mean-centred focal variables and estimated conditional effects",
            variable=f"{predictor}, {moderator}",
            details="Created a product interaction and estimated simple slopes plus Johnson-Neyman boundaries.",
            justification="Centring supports interpretable conditional effects but does not remove underlying correlation between distinct main-effect predictors.",
            before_n=len(data), after_n=len(data),
        )],
        reproducible_code="# Mean-centre predictor and moderator; fit HC3 OLS with interaction; calculate simple slopes and Johnson-Neyman boundaries from the robust covariance matrix.\n",
    )


def parallel_mediation_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    mediators: list[str],
    controls: list[str] | None = None,
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> AnalysisResult:
    controls = controls or []
    mediators = list(dict.fromkeys(mediators))
    if len(mediators) < 2:
        raise ValueError("Parallel mediation requires at least two mediators.")
    data = _complete_numeric(df, [outcome, predictor] + mediators + controls)
    if len(data) < max(30, len(mediators) + len(controls) + 10):
        raise ValueError("The complete sample is too small for parallel mediation.")

    mediator_models = {}
    a_paths = {}
    for mediator in mediators:
        model = sm.OLS(data[mediator], sm.add_constant(data[[predictor] + controls], has_constant="add")).fit(cov_type="HC3")
        mediator_models[mediator] = model
        a_paths[mediator] = float(model.params[predictor])
    outcome_model = sm.OLS(data[outcome], sm.add_constant(data[[predictor] + mediators + controls], has_constant="add")).fit(cov_type="HC3")
    total_model = sm.OLS(data[outcome], sm.add_constant(data[[predictor] + controls], has_constant="add")).fit(cov_type="HC3")
    indirect = {mediator: a_paths[mediator] * float(outcome_model.params[mediator]) for mediator in mediators}

    rng = np.random.default_rng(random_state)
    samples = {mediator: [] for mediator in mediators}
    samples["total_indirect"] = []
    n = len(data)
    for _ in range(int(bootstrap_samples)):
        sample = data.iloc[rng.integers(0, n, n)]
        try:
            a_boot = {}
            for mediator in mediators:
                a_boot[mediator] = float(sm.OLS(sample[mediator], sm.add_constant(sample[[predictor] + controls], has_constant="add")).fit().params[predictor])
            b_boot = sm.OLS(sample[outcome], sm.add_constant(sample[[predictor] + mediators + controls], has_constant="add")).fit()
            total_boot = 0.0
            for mediator in mediators:
                effect = a_boot[mediator] * float(b_boot.params[mediator])
                samples[mediator].append(effect)
                total_boot += effect
            samples["total_indirect"].append(total_boot)
        except Exception:
            continue
    successful = len(samples["total_indirect"])
    if successful < max(200, bootstrap_samples // 2):
        raise RuntimeError("Too many parallel-mediation bootstrap samples failed.")
    rows = []
    for mediator in mediators:
        low, high = np.quantile(samples[mediator], [alpha / 2, 1 - alpha / 2])
        rows.append({"mediator": mediator, "indirect_effect": indirect[mediator], "bootstrap_ci_lower": low, "bootstrap_ci_upper": high, "interval_excludes_zero": not (low <= 0 <= high)})
    total_indirect = sum(indirect.values())
    low, high = np.quantile(samples["total_indirect"], [alpha / 2, 1 - alpha / 2])
    rows.append({"mediator": "Total indirect effect", "indirect_effect": total_indirect, "bootstrap_ci_lower": low, "bootstrap_ci_upper": high, "interval_excludes_zero": not (low <= 0 <= high)})
    path_rows = []
    for mediator in mediators:
        path_rows.append({"path": f"{predictor} -> {mediator}", "estimate": a_paths[mediator], "p_value": mediator_models[mediator].pvalues[predictor]})
        path_rows.append({"path": f"{mediator} -> {outcome} controlling other mediators", "estimate": outcome_model.params[mediator], "p_value": outcome_model.pvalues[mediator]})
    path_rows.extend([
        {"path": f"Direct {predictor} -> {outcome}", "estimate": outcome_model.params[predictor], "p_value": outcome_model.pvalues[predictor]},
        {"path": f"Total {predictor} -> {outcome}", "estimate": total_model.params[predictor], "p_value": total_model.pvalues[predictor]},
    ])
    diagnostics = []
    for name, model in [(f"Mediator model: {m}", mediator_models[m]) for m in mediators] + [("Outcome model", outcome_model)]:
        diag = ols_diagnostics(model, alpha)
        diag["diagnostic"] = name + ": " + diag["diagnostic"].astype(str)
        diagnostics.append(diag)
    return AnalysisResult(
        method="Parallel multiple mediation analysis",
        summary=f"Parallel mediation analysed {len(data)} complete observations with {successful} successful bootstrap samples. The total indirect effect was {total_indirect:.4g}, with a {(1-alpha):.0%} bootstrap interval from {low:.4g} to {high:.4g}.",
        tables={"Parallel indirect effects": pd.DataFrame(rows), "Mediation path estimates": pd.DataFrame(path_rows), "Outcome model coefficients": _tidy_model(outcome_model)},
        diagnostics=pd.concat(diagnostics, ignore_index=True),
        metadata={"mediators": mediators, "bootstrap_samples_successful": successful, "random_state": random_state},
        treatment_log=[AuditEntry(
            action="Estimated parallel mediator-specific indirect effects",
            variable=", ".join(mediators),
            details=f"Generated {successful} successful bootstrap samples for individual and total indirect effects.",
            justification="Parallel mediation estimates each indirect pathway while controlling for the other mediators. Bootstrap intervals avoid normal-theory assumptions for products of coefficients.",
            before_n=n, after_n=n,
        )],
        reproducible_code="# Fit one mediator model per mediator, a joint outcome model, and bootstrap mediator-specific plus total indirect effects.\n",
    )


def moderated_mediation_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    mediator: str,
    moderator: str,
    controls: list[str] | None = None,
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> AnalysisResult:
    controls = controls or []
    data = _complete_numeric(df, [outcome, predictor, mediator, moderator] + controls)
    if len(data) < max(35, len(controls) + 10):
        raise ValueError("The complete sample is too small for moderated mediation.")
    data = data.copy()
    data["c_x"] = data[predictor] - data[predictor].mean()
    data["c_w"] = data[moderator] - data[moderator].mean()
    data["xw"] = data["c_x"] * data["c_w"]
    mediator_predictors = ["c_x", "c_w", "xw"] + controls
    mediator_model = sm.OLS(data[mediator], sm.add_constant(data[mediator_predictors], has_constant="add")).fit(cov_type="HC3")
    outcome_predictors = ["c_x", mediator, "c_w"] + controls
    outcome_model = sm.OLS(data[outcome], sm.add_constant(data[outcome_predictors], has_constant="add")).fit(cov_type="HC3")
    a1 = float(mediator_model.params["c_x"])
    a3 = float(mediator_model.params["xw"])
    b_path = float(outcome_model.params[mediator])
    w_sd = float(data[moderator].std(ddof=1))
    level_values = [(-w_sd, "Minus 1 SD"), (0.0, "Mean"), (w_sd, "Plus 1 SD")]

    rng = np.random.default_rng(random_state)
    boot = {label: [] for _, label in level_values}
    n = len(data)
    for _ in range(int(bootstrap_samples)):
        sample = data.iloc[rng.integers(0, n, n)]
        try:
            m_fit = sm.OLS(sample[mediator], sm.add_constant(sample[mediator_predictors], has_constant="add")).fit()
            y_fit = sm.OLS(sample[outcome], sm.add_constant(sample[outcome_predictors], has_constant="add")).fit()
            for value, label in level_values:
                boot[label].append((float(m_fit.params["c_x"]) + float(m_fit.params["xw"]) * value) * float(y_fit.params[mediator]))
        except Exception:
            continue
    successful = min(len(values) for values in boot.values())
    if successful < max(200, bootstrap_samples // 2):
        raise RuntimeError("Too many moderated-mediation bootstrap samples failed.")
    rows = []
    for value, label in level_values:
        effect = (a1 + a3 * value) * b_path
        low, high = np.quantile(boot[label], [alpha / 2, 1 - alpha / 2])
        rows.append({"moderator_level": label, "centered_moderator_value": value, "conditional_indirect_effect": effect, "bootstrap_ci_lower": low, "bootstrap_ci_upper": high, "interval_excludes_zero": not (low <= 0 <= high)})
    index_of_moderated_mediation = a3 * b_path
    return AnalysisResult(
        method="First-stage moderated mediation analysis",
        summary=f"The index of moderated mediation was {index_of_moderated_mediation:.4g}. Conditional indirect effects were estimated at the moderator mean and plus or minus one standard deviation using {successful} successful bootstrap samples.",
        tables={"Conditional indirect effects": pd.DataFrame(rows), "Mediator model coefficients": _tidy_model(mediator_model), "Outcome model coefficients": _tidy_model(outcome_model), "Moderated mediation index": pd.DataFrame([{"index": index_of_moderated_mediation, "a3_interaction_path": a3, "b_mediator_path": b_path}])},
        diagnostics=pd.concat([
            ols_diagnostics(mediator_model, alpha).assign(diagnostic=lambda x: "Mediator model: " + x["diagnostic"].astype(str)),
            ols_diagnostics(outcome_model, alpha).assign(diagnostic=lambda x: "Outcome model: " + x["diagnostic"].astype(str)),
        ], ignore_index=True),
        metadata={"bootstrap_samples_successful": successful, "moderation_stage": "first-stage", "random_state": random_state},
        treatment_log=[AuditEntry(
            action="Estimated first-stage moderated indirect effects",
            variable=f"{predictor}, {mediator}, {moderator}",
            details="Modelled the predictor-by-moderator interaction in the mediator equation and bootstrapped conditional indirect effects.",
            justification="The conditional indirect effect changes with the moderator through the first-stage path. Bootstrap intervals provide the primary uncertainty assessment.",
            before_n=n, after_n=n,
        )],
        reproducible_code="# Mean-centre predictor and moderator, estimate first-stage interaction in the mediator model, then bootstrap conditional indirect effects at moderator mean and plus/minus one SD.\n",
    )
