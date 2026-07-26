from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .diagnostics import (
    normality_diagnostic,
    ols_diagnostics,
    variance_equality_diagnostic,
    vif_table,
)
from .models import AnalysisResult, AuditEntry


def _tidy_params(model, exponentiate: bool = False) -> pd.DataFrame:
    conf = np.asarray(model.conf_int())
    index = list(model.params.index) if hasattr(model.params, "index") else [f"term_{i}" for i in range(len(model.params))]
    params = np.asarray(model.params)
    bse = np.asarray(model.bse)
    tvalues = np.asarray(model.tvalues)
    pvalues = np.asarray(model.pvalues)
    table = pd.DataFrame({
        "term": index,
        "estimate": params,
        "std_error": bse,
        "statistic": tvalues,
        "p_value": pvalues,
        "ci_lower": conf[:, 0],
        "ci_upper": conf[:, 1],
    })
    if exponentiate:
        table["odds_ratio"] = np.exp(table["estimate"])
        table["or_ci_lower"] = np.exp(table["ci_lower"])
        table["or_ci_upper"] = np.exp(table["ci_upper"])
    return table


def _complete_case(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns))
    return df.loc[:, columns].dropna().copy()


def descriptive_statistics(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    group_by: str | None = None,
    sample_basis: str = "available-case",
) -> AnalysisResult:
    selected = list(dict.fromkeys(variables or list(df.columns)))
    selected = [column for column in selected if column in df.columns]
    if not selected:
        raise ValueError("Select at least one variable for descriptive statistics.")

    data = df.loc[:, selected].copy()
    numeric_columns = list(data.select_dtypes(include=np.number).columns)
    categorical_columns = list(data.select_dtypes(exclude=np.number).columns)
    discrete_numeric = [
        column for column in numeric_columns
        if data[column].nunique(dropna=True) <= 10
    ]
    frequency_columns = list(dict.fromkeys(categorical_columns + discrete_numeric))

    tables: dict[str, pd.DataFrame] = {}
    sample_overview = pd.DataFrame([{
        "sample_basis": sample_basis,
        "dataset_rows": int(len(data)),
        "variables_described": int(len(selected)),
        "complete_rows_across_selected_variables": int(data.dropna().shape[0]),
        "rows_with_any_missing_selected_value": int(data.isna().any(axis=1).sum()),
    }])
    tables["Descriptive sample overview"] = sample_overview

    if numeric_columns:
        rows: list[dict[str, object]] = []
        for column in numeric_columns:
            series = pd.to_numeric(data[column], errors="coerce")
            valid = series.dropna()
            rows.append({
                "variable": column,
                "valid_n": int(valid.shape[0]),
                "missing_n": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std_dev": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "minimum": float(valid.min()) if not valid.empty else np.nan,
                "q1": float(valid.quantile(0.25)) if not valid.empty else np.nan,
                "q3": float(valid.quantile(0.75)) if not valid.empty else np.nan,
                "maximum": float(valid.max()) if not valid.empty else np.nan,
                "skewness": float(valid.skew()) if len(valid) > 2 else np.nan,
                "kurtosis": float(valid.kurtosis()) if len(valid) > 3 else np.nan,
            })
        tables["Descriptive statistics - Numeric variables"] = pd.DataFrame(rows)

    categorical_summary_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    for column in frequency_columns:
        series = data[column]
        valid = series.dropna()
        counts = series.value_counts(dropna=False)
        valid_counts = valid.value_counts(dropna=False)
        mode_value = valid_counts.index[0] if not valid_counts.empty else np.nan
        mode_count = int(valid_counts.iloc[0]) if not valid_counts.empty else 0
        categorical_summary_rows.append({
            "variable": column,
            "valid_n": int(valid.shape[0]),
            "missing_n": int(series.isna().sum()),
            "missing_percent": float(series.isna().mean() * 100),
            "number_of_categories": int(valid.nunique(dropna=True)),
            "mode": "" if pd.isna(mode_value) else mode_value,
            "mode_count": mode_count,
        })
        for value, count in counts.items():
            is_missing = pd.isna(value)
            denominator = max(len(valid), 1)
            frequency_rows.append({
                "variable": column,
                "category": "<missing>" if is_missing else value,
                "count": int(count),
                "percent_of_valid": np.nan if is_missing else float(count / denominator * 100),
                "percent_of_total": float(count / max(len(series), 1) * 100),
            })
    if categorical_summary_rows:
        tables["Descriptive statistics - Categorical summary"] = pd.DataFrame(categorical_summary_rows)
    if frequency_rows:
        tables["Descriptive statistics - Frequencies"] = pd.DataFrame(frequency_rows)

    if group_by and group_by in df.columns:
        group_numeric = [column for column in numeric_columns if column != group_by]
        grouped_rows: list[dict[str, object]] = []
        if group_numeric:
            grouped_source = df[[group_by] + group_numeric].copy()
            for group_value, group_frame in grouped_source.groupby(group_by, dropna=False, observed=True):
                group_label = "<missing>" if pd.isna(group_value) else group_value
                for column in group_numeric:
                    values = pd.to_numeric(group_frame[column], errors="coerce").dropna()
                    grouped_rows.append({
                        "group_variable": group_by,
                        "group": group_label,
                        "variable": column,
                        "n": int(values.shape[0]),
                        "mean": float(values.mean()) if not values.empty else np.nan,
                        "std_dev": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        "median": float(values.median()) if not values.empty else np.nan,
                        "minimum": float(values.min()) if not values.empty else np.nan,
                        "maximum": float(values.max()) if not values.empty else np.nan,
                    })
        if grouped_rows:
            tables["Descriptive statistics - By group"] = pd.DataFrame(grouped_rows)

    summary = (
        f"Descriptive statistics were produced for {len(selected)} analysis variable(s). "
        f"Numeric variables include valid and missing counts, mean, standard deviation, median, quartiles, range, skewness and kurtosis. "
        f"Categorical and low-cardinality variables include frequencies and percentages."
    )
    return AnalysisResult(
        method="Descriptive statistics",
        summary=summary,
        tables=tables,
        metadata={
            "n_rows": len(data),
            "n_variables": len(selected),
            "variables": selected,
            "group_by": group_by,
            "sample_basis": sample_basis,
        },
        reproducible_code=(
            f"# Descriptive statistics for the analysis variables\n"
            f"selected = df[{selected!r}]\n"
            "numeric_summary = selected.describe().T\n"
            "frequency_tables = {c: selected[c].value_counts(dropna=False) for c in selected.columns}\n"
        ),
    )


def cronbach_alpha(df: pd.DataFrame, items: list[str]) -> AnalysisResult:
    data = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if len(items) < 2:
        raise ValueError("Select at least two scale items.")
    if len(data) < 2:
        raise ValueError("At least two complete records are required.")

    item_variances = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    k = len(items)
    alpha = np.nan if total_variance == 0 else k / (k - 1) * (1 - item_variances.sum() / total_variance)

    rows = []
    for item in items:
        remaining = [x for x in items if x != item]
        reduced = data[remaining]
        kr = len(remaining)
        reduced_total_var = reduced.sum(axis=1).var(ddof=1)
        alpha_deleted = np.nan
        if kr > 1 and reduced_total_var != 0:
            alpha_deleted = kr / (kr - 1) * (1 - reduced.var(axis=0, ddof=1).sum() / reduced_total_var)
        corrected_item_total = data[item].corr(data[remaining].sum(axis=1))
        rows.append({
            "item": item,
            "mean": float(data[item].mean()),
            "std_dev": float(data[item].std(ddof=1)),
            "corrected_item_total_correlation": float(corrected_item_total),
            "alpha_if_deleted": float(alpha_deleted) if np.isfinite(alpha_deleted) else np.nan,
        })

    status = "acceptable" if alpha >= 0.70 else "below the common 0.70 guideline"
    return AnalysisResult(
        method="Reliability analysis",
        summary=f"Cronbach's alpha was {alpha:.3f}, which is {status}. Interpret the value alongside construct breadth and item content.",
        tables={
            "Reliability summary": pd.DataFrame([{"items": k, "complete_cases": len(data), "cronbach_alpha": alpha}]),
            "Item statistics": pd.DataFrame(rows),
        },
        metadata={"cronbach_alpha": float(alpha), "n": len(data), "k": k},
        reproducible_code="# Cronbach alpha calculated from item and total-score variances.",
    )


def correlation_analysis(df: pd.DataFrame, variables: list[str], method: str = "pearson", alpha: float = 0.05) -> AnalysisResult:
    if len(variables) < 2:
        raise ValueError("Select at least two variables.")
    data = df[variables].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(method=method)
    pairs: list[dict[str, object]] = []
    for i, left in enumerate(variables):
        for right in variables[i + 1:]:
            pair = data[[left, right]].dropna()
            if len(pair) < 3:
                r, p = np.nan, np.nan
            elif method == "spearman":
                r, p = stats.spearmanr(pair[left], pair[right])
            else:
                r, p = stats.pearsonr(pair[left], pair[right])
            pairs.append({
                "variable_1": left,
                "variable_2": right,
                "n": len(pair),
                "coefficient": float(r) if np.isfinite(r) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "significant_at_alpha": bool(p < alpha) if np.isfinite(p) else False,
            })

    return AnalysisResult(
        method=f"{method.title()} correlation",
        summary=f"A {method} correlation matrix was estimated for {len(variables)} variables. Correlation describes association and does not establish causality.",
        tables={"Correlation matrix": corr.reset_index(names="variable"), "Pairwise tests": pd.DataFrame(pairs)},
        metadata={"alpha": alpha, "method": method},
        reproducible_code=f"df[{variables!r}].corr(method={method!r})",
    )


def independent_t_test(df: pd.DataFrame, outcome: str, group: str, alpha: float = 0.05) -> AnalysisResult:
    data = _complete_case(df, [outcome, group])
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    levels = list(pd.unique(data[group]))
    if len(levels) != 2:
        raise ValueError("The grouping variable must have exactly two observed categories.")
    a = data.loc[data[group] == levels[0], outcome]
    b = data.loc[data[group] == levels[1], outcome]

    variance_diag = variance_equality_diagnostic([a, b], alpha)
    use_equal_var = variance_diag["status"] == "Satisfied"
    t_stat, p = stats.ttest_ind(a, b, equal_var=use_equal_var, nan_policy="omit")
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    group_stats = pd.DataFrame([
        {"group": levels[0], "n": len(a), "mean": a.mean(), "std_dev": a.std(ddof=1)},
        {"group": levels[1], "n": len(b), "mean": b.mean(), "std_dev": b.std(ddof=1)},
    ])
    diagnostics = pd.DataFrame([
        normality_diagnostic(a, alpha, str(levels[0])),
        normality_diagnostic(b, alpha, str(levels[1])),
        variance_diag,
    ])
    test_name = "Student independent-samples t-test" if use_equal_var else "Welch independent-samples t-test"
    decision = "statistically significant" if p < alpha else "not statistically significant"

    treatment_log: list[AuditEntry] = []
    if not use_equal_var:
        treatment_log.append(AuditEntry(
            action="Used Welch correction",
            variable=outcome,
            details="The unequal-variance form of the independent-samples t-test was used.",
            justification="Levene's test indicated materially unequal group variances. No data values were changed.",
            before_n=len(data),
            after_n=len(data),
        ))

    return AnalysisResult(
        method=test_name,
        summary=f"The difference between the two group means was {decision}, t={t_stat:.3f}, p={p:.4g}, Cohen's d={d:.3f}.",
        tables={
            "Group statistics": group_stats,
            "Test result": pd.DataFrame([{
                "test": test_name, "t_statistic": t_stat, "p_value": p,
                "mean_difference": a.mean() - b.mean(), "cohens_d": d, "alpha": alpha,
            }]),
        },
        diagnostics=diagnostics,
        metadata={"levels": [str(x) for x in levels], "equal_var": use_equal_var},
        treatment_log=treatment_log,
        reproducible_code=f"scipy.stats.ttest_ind(group1, group2, equal_var={use_equal_var})",
    )


def paired_t_test(df: pd.DataFrame, before: str, after: str, alpha: float = 0.05) -> AnalysisResult:
    data = _complete_case(df, [before, after]).apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2:
        raise ValueError("At least two complete pairs are required.")
    difference = data[after] - data[before]
    t_stat, p = stats.ttest_rel(data[after], data[before], nan_policy="omit")
    dz = difference.mean() / difference.std(ddof=1) if difference.std(ddof=1) > 0 else np.nan
    diagnostics = pd.DataFrame([normality_diagnostic(difference, alpha, "paired differences")])
    return AnalysisResult(
        method="Paired-samples t-test",
        summary=f"The mean paired change was {difference.mean():.3f}, t={t_stat:.3f}, p={p:.4g}, Cohen's dz={dz:.3f}.",
        tables={
            "Paired descriptives": pd.DataFrame([
                {"variable": before, "n": len(data), "mean": data[before].mean(), "std_dev": data[before].std(ddof=1)},
                {"variable": after, "n": len(data), "mean": data[after].mean(), "std_dev": data[after].std(ddof=1)},
            ]),
            "Test result": pd.DataFrame([{"t_statistic": t_stat, "p_value": p, "mean_change": difference.mean(), "cohens_dz": dz, "alpha": alpha}]),
        },
        diagnostics=diagnostics,
        reproducible_code=f"scipy.stats.ttest_rel(df[{after!r}], df[{before!r}])",
    )


def one_way_anova(df: pd.DataFrame, outcome: str, group: str, alpha: float = 0.05) -> AnalysisResult:
    data = _complete_case(df, [outcome, group])
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    levels = list(pd.unique(data[group]))
    if len(levels) < 3:
        raise ValueError("ANOVA requires at least three observed groups.")
    groups = [data.loc[data[group] == level, outcome] for level in levels]
    variance_diag = variance_equality_diagnostic(groups, alpha)
    use_welch = variance_diag["status"] != "Satisfied"

    if use_welch:
        test = anova_oneway(groups, use_var="unequal", welch_correction=True)
        statistic, p_value = float(test.statistic), float(test.pvalue)
        df_num, df_denom = float(test.df_num), float(test.df_denom)
        test_name = "Welch one-way ANOVA"
    else:
        statistic, p_value = stats.f_oneway(*groups)
        df_num, df_denom = len(groups) - 1, len(data) - len(groups)
        test_name = "One-way ANOVA"

    grand_mean = data[outcome].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((data[outcome] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    descriptives = pd.DataFrame([
        {"group": level, "n": len(values), "mean": values.mean(), "std_dev": values.std(ddof=1)}
        for level, values in zip(levels, groups)
    ])
    diagnostics = pd.DataFrame([variance_diag] + [normality_diagnostic(g, alpha, str(level)) for level, g in zip(levels, groups)])
    treatment_log: list[AuditEntry] = []
    if use_welch:
        treatment_log.append(AuditEntry(
            action="Used Welch ANOVA",
            variable=outcome,
            details="The unequal-variance ANOVA was selected automatically.",
            justification="Levene's test indicated unequal group variances. No observations were changed or removed.",
            before_n=len(data), after_n=len(data),
        ))

    return AnalysisResult(
        method=test_name,
        summary=f"{test_name} returned F={statistic:.3f}, p={p_value:.4g}, with eta-squared={eta_sq:.3f}.",
        tables={
            "Group descriptives": descriptives,
            "ANOVA result": pd.DataFrame([{
                "test": test_name, "f_statistic": statistic, "df_numerator": df_num,
                "df_denominator": df_denom, "p_value": p_value, "eta_squared": eta_sq,
            }]),
        },
        diagnostics=diagnostics,
        treatment_log=treatment_log,
        reproducible_code="# One-way ANOVA; Welch correction is applied when Levene's test indicates unequal variances.",
    )


def chi_square_test(df: pd.DataFrame, row_variable: str, column_variable: str, alpha: float = 0.05) -> AnalysisResult:
    data = _complete_case(df, [row_variable, column_variable])
    table = pd.crosstab(data[row_variable], data[column_variable])
    if table.empty:
        raise ValueError("The contingency table is empty.")

    chi2, p, dof, expected = stats.chi2_contingency(table)
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    low_expected_percent = float((expected < 5).mean() * 100)
    fisher_used = table.shape == (2, 2) and bool((expected < 5).any())
    fisher_p = np.nan
    if fisher_used:
        _, fisher_p = stats.fisher_exact(table.to_numpy())

    n = table.to_numpy().sum()
    denom = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * denom)) if n > 0 and denom > 0 else np.nan
    diagnostic = pd.DataFrame([{
        "diagnostic": "Expected cell frequencies",
        "test": "Chi-square adequacy check",
        "statistic": low_expected_percent,
        "p_value": np.nan,
        "status": "Satisfied" if low_expected_percent <= 20 and expected.min() >= 1 else "Material concern",
        "interpretation": f"{low_expected_percent:.1f}% of expected counts are below 5; minimum expected count={expected.min():.3f}.",
        "recommended_response": "Use Fisher's exact test for a 2x2 table, or combine categories only when theoretically justified." if low_expected_percent > 20 or expected.min() < 1 else "The chi-square approximation is acceptable.",
    }])

    summary_p = fisher_p if fisher_used else p
    summary_test = "Fisher's exact test" if fisher_used else "Chi-square test"
    treatment_log: list[AuditEntry] = []
    if fisher_used:
        treatment_log.append(AuditEntry(
            action="Used Fisher's exact test",
            details="Fisher's exact p-value was used for inference in the 2x2 table.",
            justification="One or more expected cell frequencies were small. No categories were altered.",
            before_n=len(data), after_n=len(data),
        ))

    return AnalysisResult(
        method=summary_test,
        summary=f"{summary_test} produced p={summary_p:.4g}. Cramer's V={cramers_v:.3f} describes association strength.",
        tables={
            "Observed counts": table.reset_index(),
            "Expected counts": expected_df.reset_index(),
            "Test result": pd.DataFrame([{
                "chi_square": chi2, "degrees_of_freedom": dof, "chi_square_p": p,
                "fisher_exact_p": fisher_p, "cramers_v": cramers_v,
            }]),
        },
        diagnostics=diagnostic,
        treatment_log=treatment_log,
        reproducible_code=f"scipy.stats.chi2_contingency(pd.crosstab(df[{row_variable!r}], df[{column_variable!r}]))",
    )


def _prepare_design(df: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    data = _complete_case(df, [outcome] + predictors)
    y = pd.to_numeric(data[outcome], errors="coerce")
    x_raw = data[predictors].copy()
    x = pd.get_dummies(x_raw, drop_first=True, dtype=float)
    x = x.apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([y.rename(outcome), x], axis=1).dropna()
    y = combined[outcome].astype(float)
    x = combined.drop(columns=[outcome]).astype(float)
    if x.empty:
        raise ValueError("No usable predictor columns remain after preparation.")
    x_const = sm.add_constant(x, has_constant="add")
    return y, x_const, combined


def _ridge_multicollinearity_sensitivity(
    y: pd.Series,
    predictors: pd.DataFrame,
    conventional_table: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit a cross-validated ridge model without altering observed values.

    Ridge is used as a stability/sensitivity model when OLS coefficient
    decomposition is unreliable because predictors overlap strongly. It is not
    used to manufacture statistical significance and it does not replace
    theory-guided construct decisions.
    """
    x_numeric = predictors.select_dtypes(include=np.number).astype(float)
    if x_numeric.shape[1] < 2 or len(y) < 6:
        raise ValueError("Ridge sensitivity requires at least two predictors and six observations.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_numeric)
    alphas = np.logspace(-4, 4, 81)
    n_splits = min(5, max(2, len(y) // 4), len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring="neg_root_mean_squared_error")
    ridge_cv.fit(x_scaled, y)
    best_alpha = float(ridge_cv.alpha_)

    cv_predictions = cross_val_predict(Ridge(alpha=best_alpha), x_scaled, y, cv=cv)
    cv_rmse = float(np.sqrt(mean_squared_error(y, cv_predictions)))
    cv_r2 = float(r2_score(y, cv_predictions))
    training_r2 = float(ridge_cv.score(x_scaled, y))
    y_sd = float(pd.Series(y).std(ddof=1))

    ridge_table = pd.DataFrame({
        "term": x_numeric.columns,
        "ridge_coefficient_per_1_sd_predictor": np.asarray(ridge_cv.coef_, dtype=float),
    })
    ridge_table["standardized_coefficient"] = (
        ridge_table["ridge_coefficient_per_1_sd_predictor"] / y_sd if y_sd > 0 else np.nan
    )
    ridge_table["absolute_standardized_coefficient"] = ridge_table["standardized_coefficient"].abs()
    ridge_table["importance_rank"] = ridge_table["absolute_standardized_coefficient"].rank(
        method="dense", ascending=False
    ).astype(int)
    ridge_table = ridge_table.sort_values("importance_rank").reset_index(drop=True)

    x_sd = x_numeric.std(ddof=1)
    ols = conventional_table.loc[conventional_table["term"] != "const", ["term", "estimate"]].copy()
    ols["ols_standardized_coefficient"] = ols.apply(
        lambda row: row["estimate"] * x_sd.get(row["term"], np.nan) / y_sd if y_sd > 0 else np.nan,
        axis=1,
    )
    comparison = ols.merge(
        ridge_table[["term", "standardized_coefficient"]], on="term", how="outer"
    ).rename(columns={"standardized_coefficient": "ridge_standardized_coefficient"})
    comparison["direction_agrees"] = np.sign(comparison["ols_standardized_coefficient"]) == np.sign(
        comparison["ridge_standardized_coefficient"]
    )
    comparable = comparison.dropna(subset=["ols_standardized_coefficient", "ridge_standardized_coefficient"])
    direction_agreement = float(comparable["direction_agrees"].mean()) if not comparable.empty else np.nan

    fit_table = pd.DataFrame([{
        "selected_penalty_alpha": best_alpha,
        "cross_validation_folds": n_splits,
        "cross_validated_rmse": cv_rmse,
        "cross_validated_r_squared": cv_r2,
        "training_r_squared": training_r2,
        "n": len(y),
        "predictor_columns": x_numeric.shape[1],
    }])
    return {
        "coefficient_table": ridge_table,
        "fit_table": fit_table,
        "comparison_table": comparison,
        "alpha": best_alpha,
        "cv_folds": n_splits,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "direction_agreement": direction_agreement,
    }


def ols_regression(df: pd.DataFrame, outcome: str, predictors: list[str], alpha: float = 0.05) -> AnalysisResult:
    y, x, combined = _prepare_design(df, outcome, predictors)
    x_predictors = x.drop(columns=["const"], errors="ignore")
    model = sm.OLS(y, x).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    conventional_table = _tidy_params(model)
    robust_conf = np.asarray(robust.conf_int())
    robust_table = pd.DataFrame({
        "term": model.params.index,
        "estimate": np.asarray(robust.params),
        "std_error": np.asarray(robust.bse),
        "statistic": np.asarray(robust.tvalues),
        "p_value": np.asarray(robust.pvalues),
        "ci_lower": robust_conf[:, 0],
        "ci_upper": robust_conf[:, 1],
    })

    diagnostics = ols_diagnostics(model, alpha)
    bp_row = diagnostics.loc[diagnostics["test"] == "Breusch-Pagan"]
    use_robust = not bp_row.empty and bp_row.iloc[0]["status"] != "Satisfied"

    vif = vif_table(x_predictors)
    if not vif.empty:
        vif_diag = vif.rename(columns={"variable": "diagnostic", "vif": "statistic"}).copy()
        vif_diag["diagnostic"] = "Multicollinearity: " + vif_diag["diagnostic"].astype(str)
        vif_diag["test"] = "Variance inflation factor"
        vif_diag["p_value"] = np.nan
        vif_diag["interpretation"] = vif.apply(
            lambda row: f"VIF={row['vif']:.3f}; tolerance={row['tolerance']:.4f}.", axis=1
        )
        vif_diag["recommended_response"] = vif["status"].map({
            "Satisfied": "No remedial action indicated.",
            "Minor concern": "Review predictor overlap and compare coefficient stability in a sensitivity model.",
            "Material concern": "Run a theory-preserving alternative such as ridge or principal-component sensitivity analysis. Do not drop predictors solely to gain significance.",
        })
        diagnostics = pd.concat([diagnostics, vif_diag[diagnostics.columns]], ignore_index=True)

    influence = OLSInfluence(model)
    cooks = np.asarray(influence.cooks_distance[0])
    threshold = 4 / max(len(y), 1)
    keep = cooks <= threshold
    influence_sensitivity_table = pd.DataFrame()
    if keep.sum() >= max(x.shape[1] + 2, 5) and keep.sum() < len(keep):
        sensitivity = sm.OLS(y.iloc[keep], x.iloc[keep]).fit(cov_type="HC3")
        influence_sensitivity_table = _tidy_params(sensitivity)

    selected_table = robust_table if use_robust else conventional_table
    treatment_log: list[AuditEntry] = []
    warnings: list[str] = []
    if use_robust:
        treatment_log.append(AuditEntry(
            action="Applied HC3 robust standard errors",
            variable=outcome,
            details="The coefficient estimates were retained while inference used heteroskedasticity-consistent standard errors.",
            justification="The Breusch-Pagan diagnostic raised a material concern about non-constant residual variance. No data values were altered.",
            before_n=len(y), after_n=len(y),
        ))
    if not influence_sensitivity_table.empty:
        treatment_log.append(AuditEntry(
            action="Ran influential-observation sensitivity model",
            variable=outcome,
            details=f"Compared the full model with a model excluding {int((~keep).sum())} observation(s) exceeding Cook's distance 4/n.",
            justification="This is a documented sensitivity analysis, not an automatic deletion rule. The full-sample model remains primary unless theory and data verification support another choice.",
            before_n=len(y), after_n=int(keep.sum()),
        ))

    tables: dict[str, pd.DataFrame] = {
        "Selected coefficient table": selected_table,
        "Conventional coefficients": conventional_table,
        "HC3 robust coefficients": robust_table,
        "VIF": vif,
    }
    if not influence_sensitivity_table.empty:
        tables["Influential-observation sensitivity coefficients"] = influence_sensitivity_table

    max_vif = float(vif["vif"].replace([np.inf, -np.inf], np.nan).max()) if not vif.empty else np.nan
    material_multicollinearity = bool((vif["status"] == "Material concern").any()) if not vif.empty else False
    diagnostic_response = ""
    ridge_info: dict[str, object] | None = None
    if material_multicollinearity:
        ridge_info = _ridge_multicollinearity_sensitivity(y, x_predictors, conventional_table)
        agreement = ridge_info["direction_agreement"]
        agreement_text = (
            f"{agreement:.0%} of comparable coefficient directions agreed"
            if np.isfinite(agreement) else "coefficient-direction agreement could not be calculated"
        )
        diagnostic_response = (
            f"Material multicollinearity was detected, with a maximum correctly specified VIF of {max_vif:.2f}. "
            f"A standardized ridge regression was therefore estimated as a sensitivity model. "
            f"The penalty alpha={ridge_info['alpha']:.4g} was selected using {ridge_info['cv_folds']}-fold cross-validation; "
            f"cross-validated RMSE={ridge_info['cv_rmse']:.3f} and R-squared={ridge_info['cv_r2']:.3f}. "
            f"{agreement_text}. Ridge stabilises overlapping coefficients but does not provide conventional OLS p-values. "
            "The observed data were not transformed or deleted, and the alternative model was not selected on the basis of significance."
        )
        action_summary = pd.DataFrame([{
            "issue": "Material multicollinearity",
            "initial_evidence": f"Maximum VIF={max_vif:.3f}",
            "action_taken": "Standardised cross-validated ridge regression sensitivity model",
            "post_action_evidence": f"alpha={ridge_info['alpha']:.4g}; CV RMSE={ridge_info['cv_rmse']:.3f}; CV R-squared={ridge_info['cv_r2']:.3f}",
            "interpretive_decision": "Do not interpret individual full-model OLS coefficients as isolated effects without theory. Use the ridge model to assess coefficient direction and relative stability.",
            "data_values_changed": "No",
        }])
        tables["Multicollinearity action summary"] = action_summary
        tables["Predictor correlation matrix"] = x_predictors.corr().reset_index(names="variable")
        tables["Ridge sensitivity coefficients"] = ridge_info["coefficient_table"]
        tables["Ridge sensitivity model fit"] = ridge_info["fit_table"]
        tables["OLS-Ridge coefficient comparison"] = ridge_info["comparison_table"]
        diagnostics = pd.concat([diagnostics, pd.DataFrame([{
            "diagnostic": "Multicollinearity response",
            "test": "Cross-validated ridge sensitivity",
            "statistic": float(ridge_info["alpha"]),
            "p_value": np.nan,
            "status": "Addressed by sensitivity analysis",
            "interpretation": f"A ridge alternative was estimated; {agreement_text}.",
            "recommended_response": "Report the full OLS diagnostic and ridge sensitivity together. Re-specify or combine constructs only when theory and measurement evidence justify it.",
        }])], ignore_index=True)
        treatment_log.append(AuditEntry(
            action="Ran cross-validated ridge sensitivity model",
            variable=", ".join(map(str, x_predictors.columns)),
            details=f"Standardised all encoded predictors and selected ridge alpha={ridge_info['alpha']:.4g} using {ridge_info['cv_folds']}-fold cross-validation.",
            justification="Material VIF indicates unstable separation of overlapping OLS predictor effects. Ridge provides a defensible stability analysis without changing observed values or choosing a model to improve p-values.",
            before_n=len(y), after_n=len(y),
        ))
        warnings.append(
            "Material multicollinearity remains in the full OLS specification. Individual OLS coefficients and p-values may be unstable. Interpret them only with the ridge sensitivity results and the conceptual framework."
        )

    model_fit = pd.DataFrame([{
        "n": int(model.nobs), "r_squared": model.rsquared, "adjusted_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue, "f_p_value": model.f_pvalue, "aic": model.aic, "bic": model.bic,
        "selected_inference": "HC3 robust" if use_robust else "Conventional",
        "maximum_vif": max_vif,
        "multicollinearity_response": "Cross-validated ridge sensitivity" if material_multicollinearity else "Not required",
    }])
    tables["Model fit"] = model_fit

    summary = (
        f"The OLS model used {int(model.nobs)} complete observations and explained {model.rsquared_adj:.1%} of adjusted outcome variance. "
        f"{'HC3 robust inference was selected because heteroskedasticity was detected.' if use_robust else 'Conventional inference was retained because the heteroskedasticity diagnostic did not fail.'}"
    )
    if diagnostic_response:
        summary += " " + diagnostic_response

    ridge_code = ""
    if ridge_info is not None:
        ridge_code = f"""
# Multicollinearity sensitivity model
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
X_scaled = StandardScaler().fit_transform(X.drop(columns=['const'], errors='ignore'))
cv = KFold(n_splits={ridge_info['cv_folds']}, shuffle=True, random_state=42)
ridge = RidgeCV(
    alphas=np.logspace(-4, 4, 81),
    cv=cv,
    scoring='neg_root_mean_squared_error',
).fit(X_scaled, y)
print(ridge.alpha_, ridge.coef_)
"""

    base_code = f"""import pandas as pd
import statsmodels.api as sm
X = pd.get_dummies(df[{predictors!r}], drop_first=True, dtype=float)
X = sm.add_constant(X, has_constant='add')
y = df[{outcome!r}]
model = sm.OLS(y, X, missing='drop').fit()
robust = model.get_robustcov_results(cov_type='HC3')
"""

    return AnalysisResult(
        method="Ordinary least squares regression",
        summary=summary,
        tables=tables,
        diagnostics=diagnostics,
        metadata={
            "outcome": outcome, "predictors": predictors, "alpha": alpha, "robust_selected": use_robust,
            "maximum_vif": max_vif, "material_multicollinearity": material_multicollinearity,
            "multicollinearity_addressed": material_multicollinearity,
            "diagnostic_response": diagnostic_response,
            "ridge_alpha": ridge_info["alpha"] if ridge_info else None,
        },
        warnings=warnings,
        treatment_log=treatment_log,
        reproducible_code=base_code + ridge_code,
    )


def _encode_binary(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    clean = series.dropna()
    levels = list(pd.unique(clean))
    if len(levels) != 2:
        raise ValueError("The dependent variable must have exactly two observed categories.")
    if set(levels) == {0, 1}:
        return pd.to_numeric(series, errors="coerce"), {"0": 0, "1": 1}
    mapping = {str(levels[0]): 0, str(levels[1]): 1}
    encoded = series.map({levels[0]: 0, levels[1]: 1})
    return encoded, mapping


def _hosmer_lemeshow(y: pd.Series, predicted: pd.Series, groups: int = 10) -> tuple[float, float, int]:
    frame = pd.DataFrame({"y": y, "p": predicted}).dropna()
    unique = frame["p"].nunique()
    g = min(groups, unique, max(len(frame) // 5, 2))
    if g < 2:
        return np.nan, np.nan, 0
    frame["bin"] = pd.qcut(frame["p"], q=g, duplicates="drop")
    agg = frame.groupby("bin", observed=True).agg(observed=("y", "sum"), expected=("p", "sum"), n=("y", "size"))
    denom1 = agg["expected"].clip(lower=1e-9)
    denom0 = (agg["n"] - agg["expected"]).clip(lower=1e-9)
    statistic = (((agg["observed"] - agg["expected"]) ** 2 / denom1) + (((agg["n"] - agg["observed"]) - (agg["n"] - agg["expected"])) ** 2 / denom0)).sum()
    dof = max(len(agg) - 2, 1)
    p = stats.chi2.sf(statistic, dof)
    return float(statistic), float(p), dof


def logistic_regression(df: pd.DataFrame, outcome: str, predictors: list[str], alpha: float = 0.05) -> AnalysisResult:
    required = [outcome] + predictors
    data = _complete_case(df, required)
    y, mapping = _encode_binary(data[outcome])
    x = pd.get_dummies(data[predictors], drop_first=True, dtype=float).apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([y.rename(outcome), x], axis=1).dropna()
    y = combined[outcome].astype(float)
    x = sm.add_constant(combined.drop(columns=[outcome]).astype(float), has_constant="add")
    model = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    robust = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="HC3")

    predictions = model.predict(x)
    predicted_class = (predictions >= 0.5).astype(int)
    accuracy = float((predicted_class == y).mean())
    sensitivity = float(((predicted_class == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    specificity = float(((predicted_class == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    hl_stat, hl_p, hl_df = _hosmer_lemeshow(y, predictions)

    diagnostics = pd.DataFrame([{
        "diagnostic": "Model calibration",
        "test": "Hosmer-Lemeshow grouped check",
        "statistic": hl_stat,
        "p_value": hl_p,
        "status": "Cannot determine" if not np.isfinite(hl_p) else ("Satisfied" if hl_p >= alpha else "Material concern"),
        "interpretation": "No strong evidence of calibration failure." if np.isfinite(hl_p) and hl_p >= alpha else "Predicted and observed event rates may differ across risk groups.",
        "recommended_response": "Inspect calibration plots, nonlinear terms, interactions, and validation performance." if np.isfinite(hl_p) and hl_p < alpha else "Retain the model subject to discrimination and design checks.",
    }])
    vif = vif_table(x.drop(columns=["const"], errors="ignore"))

    return AnalysisResult(
        method="Binary logistic regression",
        summary=f"The model analysed {len(y)} complete observations. At a 0.50 threshold, accuracy={accuracy:.1%}, sensitivity={sensitivity:.1%}, and specificity={specificity:.1%}. The event category coded 1 was {next(k for k, v in mapping.items() if v == 1)}.",
        tables={
            "Coefficients and odds ratios": _tidy_params(model, exponentiate=True),
            "HC3 robust coefficients": _tidy_params(robust, exponentiate=True),
            "Model performance": pd.DataFrame([{
                "n": len(y), "aic": model.aic, "deviance": model.deviance,
                "null_deviance": model.null_deviance, "accuracy": accuracy,
                "sensitivity": sensitivity, "specificity": specificity,
                "hosmer_lemeshow": hl_stat, "hl_df": hl_df, "hl_p_value": hl_p,
            }]),
            "VIF": vif,
        },
        diagnostics=diagnostics,
        metadata={"binary_mapping": mapping, "threshold": 0.5},
        treatment_log=[AuditEntry(
            action="Reported HC3 robust sensitivity estimates",
            variable=outcome,
            details="Both model-based and HC3 robust standard errors are supplied.",
            justification="Robust estimates provide a transparent inference sensitivity check without modifying observed data.",
            before_n=len(y), after_n=len(y),
        )],
        reproducible_code=(
            "X = pd.get_dummies(df[predictors], drop_first=True, dtype=float)\n"
            "X = sm.add_constant(X, has_constant='add')\n"
            "model = sm.GLM(y, X, family=sm.families.Binomial()).fit()\n"
        ),
    )


def moderation_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    moderator: str,
    controls: list[str] | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    controls = controls or []
    data = _complete_case(df, [outcome, predictor, moderator] + controls)
    for column in [outcome, predictor, moderator] + controls:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data[f"c_{predictor}"] = data[predictor] - data[predictor].mean()
    data[f"c_{moderator}"] = data[moderator] - data[moderator].mean()
    interaction = f"{predictor}_x_{moderator}"
    data[interaction] = data[f"c_{predictor}"] * data[f"c_{moderator}"]
    predictors = [f"c_{predictor}", f"c_{moderator}", interaction] + controls
    result = ols_regression(data, outcome, predictors, alpha)
    result.method = "Moderated multiple regression"
    interaction_row = result.tables["Selected coefficient table"].loc[result.tables["Selected coefficient table"]["term"] == interaction]
    if not interaction_row.empty:
        estimate = interaction_row.iloc[0]["estimate"]
        p = interaction_row.iloc[0]["p_value"]
        result.summary = f"The interaction coefficient was {estimate:.4g} with p={p:.4g}. A significant interaction indicates that the predictor-outcome relationship changes across moderator levels."
    result.metadata.update({"predictor": predictor, "moderator": moderator, "interaction": interaction})
    result.reproducible_code += "# Mean-centre predictor and moderator, create their product term, then estimate OLS.\n"
    return result


def mediation_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    mediator: str,
    controls: list[str] | None = None,
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> AnalysisResult:
    controls = controls or []
    columns = [outcome, predictor, mediator] + controls
    data = _complete_case(df, columns)
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < max(20, len(controls) + 5):
        raise ValueError("The mediation model has too few complete observations for a stable bootstrap analysis.")

    x_a = sm.add_constant(data[[predictor] + controls], has_constant="add")
    model_a = sm.OLS(data[mediator], x_a).fit(cov_type="HC3")
    x_b = sm.add_constant(data[[predictor, mediator] + controls], has_constant="add")
    model_b = sm.OLS(data[outcome], x_b).fit(cov_type="HC3")
    x_total = sm.add_constant(data[[predictor] + controls], has_constant="add")
    model_total = sm.OLS(data[outcome], x_total).fit(cov_type="HC3")

    a = float(model_a.params[predictor])
    b = float(model_b.params[mediator])
    direct = float(model_b.params[predictor])
    total = float(model_total.params[predictor])
    indirect = a * b

    rng = np.random.default_rng(random_state)
    indirect_samples: list[float] = []
    n = len(data)
    for _ in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, n)
        sample = data.iloc[idx]
        try:
            a_fit = sm.OLS(sample[mediator], sm.add_constant(sample[[predictor] + controls], has_constant="add")).fit()
            b_fit = sm.OLS(sample[outcome], sm.add_constant(sample[[predictor, mediator] + controls], has_constant="add")).fit()
            indirect_samples.append(float(a_fit.params[predictor] * b_fit.params[mediator]))
        except Exception:
            continue
    if len(indirect_samples) < max(100, bootstrap_samples // 2):
        raise RuntimeError("Too many bootstrap resamples failed. Check collinearity and variable variation.")
    lower, upper = np.quantile(indirect_samples, [alpha / 2, 1 - alpha / 2])
    mediated = not (lower <= 0 <= upper)

    path_table = pd.DataFrame([
        {"effect": "a: predictor to mediator", "estimate": a, "p_value": model_a.pvalues[predictor]},
        {"effect": "b: mediator to outcome controlling predictor", "estimate": b, "p_value": model_b.pvalues[mediator]},
        {"effect": "direct effect c-prime", "estimate": direct, "p_value": model_b.pvalues[predictor]},
        {"effect": "total effect c", "estimate": total, "p_value": model_total.pvalues[predictor]},
        {"effect": "indirect effect a*b", "estimate": indirect, "p_value": np.nan},
    ])
    indirect_table = pd.DataFrame([{
        "indirect_effect": indirect, "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_successful": len(indirect_samples), "ci_lower": lower,
        "ci_upper": upper, "confidence_level": 1 - alpha,
        "interval_excludes_zero": mediated,
    }])

    diagnostic_rows = []
    for label, model in [("Mediator model", model_a), ("Outcome model", model_b)]:
        diag = ols_diagnostics(model, alpha)
        diag["diagnostic"] = label + ": " + diag["diagnostic"].astype(str)
        diagnostic_rows.append(diag)

    return AnalysisResult(
        method="Bootstrap mediation analysis",
        summary=f"The estimated indirect effect was {indirect:.4g}, with a {(1-alpha):.0%} bootstrap confidence interval from {lower:.4g} to {upper:.4g}. The interval {'excluded' if mediated else 'included'} zero.",
        tables={
            "Path estimates": path_table,
            "Indirect effect": indirect_table,
            "Mediator model coefficients": _tidy_params(model_a),
            "Outcome model coefficients": _tidy_params(model_b),
        },
        diagnostics=pd.concat(diagnostic_rows, ignore_index=True),
        metadata={"bootstrap_samples": bootstrap_samples, "random_state": random_state},
        treatment_log=[AuditEntry(
            action="Used non-parametric bootstrap confidence interval",
            variable=mediator,
            details=f"Generated {len(indirect_samples)} successful resamples for the indirect effect.",
            justification="The product of mediation paths commonly has a non-normal sampling distribution. Bootstrapping avoids relying on a normal-theory indirect-effect test.",
            before_n=n, after_n=n,
        )],
        reproducible_code="# Fit paths a, b, total and direct effects with OLS, then bootstrap the product a*b using a fixed random seed.",
    )
