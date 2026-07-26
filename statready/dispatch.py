from __future__ import annotations

from typing import Any

import pandas as pd

from .methods import (
    chi_square_test,
    correlation_analysis,
    cronbach_alpha,
    descriptive_statistics,
    independent_t_test,
    logistic_regression,
    mediation_analysis,
    moderation_analysis,
    ols_regression,
    one_way_anova,
    paired_t_test,
)


def _analysis_variables(method_key: str, config: dict[str, Any]) -> list[str]:
    mappings = {
        "descriptive": config.get("variables") or [],
        "reliability": config.get("items") or [],
        "correlation": config.get("variables") or [],
        "independent_t": [config.get("outcome"), config.get("group")],
        "paired_t": [config.get("before"), config.get("after")],
        "anova": [config.get("outcome"), config.get("group")],
        "chi_square": [config.get("row_variable"), config.get("column_variable")],
        "ols": [config.get("outcome"), *(config.get("predictors") or [])],
        "logistic": [config.get("outcome"), *(config.get("predictors") or [])],
        "moderation": [
            config.get("outcome"), config.get("predictor"), config.get("moderator"),
            *(config.get("controls") or []),
        ],
        "mediation": [
            config.get("outcome"), config.get("predictor"), config.get("mediator"),
            *(config.get("controls") or []),
        ],
    }
    return list(dict.fromkeys(value for value in mappings.get(method_key, []) if value))


def _group_variable(method_key: str, config: dict[str, Any]) -> str | None:
    if method_key in {"independent_t", "anova"}:
        return config.get("group")
    return None


def _attach_automatic_descriptives(
    df: pd.DataFrame,
    method_key: str,
    config: dict[str, Any],
    result,
):
    variables = _analysis_variables(method_key, config)
    if not variables:
        return result

    # Inferential descriptives should match the complete-case analytical sample.
    # The descriptive-only method retains available-case summaries.
    analysis_sample = df.loc[:, [column for column in variables if column in df.columns]].copy()
    if method_key != "descriptive":
        analysis_sample = analysis_sample.dropna()
        sample_basis = "complete-case analytical sample"
    else:
        sample_basis = "available-case selected data"

    descriptive = descriptive_statistics(
        analysis_sample,
        variables=[column for column in variables if column in analysis_sample.columns],
        group_by=_group_variable(method_key, config),
        sample_basis=sample_basis,
    )

    if method_key == "descriptive":
        return descriptive

    profile_variables = [
        column for column in (config.get("profile_variables") or [])
        if column in df.columns and column not in variables
    ]
    profile_tables: dict[str, pd.DataFrame] = {}
    profile_code = ""
    if profile_variables:
        profile = descriptive_statistics(
            df,
            variables=profile_variables,
            sample_basis="available-case profile data",
        )
        for name, table in profile.tables.items():
            renamed = name.replace("Descriptive sample overview", "Descriptive profile overview")
            renamed = renamed.replace("Descriptive statistics - ", "Descriptive profile - ")
            profile_tables[renamed] = table
        profile_code = "\n# Additional demographic/profile descriptives\n" + profile.reproducible_code

    result.tables = {**descriptive.tables, **profile_tables, **result.tables}
    result.metadata["descriptive_statistics_included"] = True
    result.metadata["descriptive_variables"] = variables
    result.metadata["profile_descriptive_variables"] = profile_variables
    result.metadata["descriptive_sample_basis"] = sample_basis
    result.metadata["descriptive_summary"] = descriptive.summary
    result.reproducible_code = descriptive.reproducible_code + profile_code + "\n" + result.reproducible_code
    return result


def run_analysis(df: pd.DataFrame, method_key: str, config: dict[str, Any]):
    alpha = float(config.get("alpha", 0.05))
    if method_key == "descriptive":
        result = descriptive_statistics(df, config.get("variables") or None)
    elif method_key == "reliability":
        result = cronbach_alpha(df, config["items"])
    elif method_key == "correlation":
        result = correlation_analysis(df, config["variables"], config.get("correlation_method", "pearson"), alpha)
    elif method_key == "independent_t":
        result = independent_t_test(df, config["outcome"], config["group"], alpha)
    elif method_key == "paired_t":
        result = paired_t_test(df, config["before"], config["after"], alpha)
    elif method_key == "anova":
        result = one_way_anova(df, config["outcome"], config["group"], alpha)
    elif method_key == "chi_square":
        result = chi_square_test(df, config["row_variable"], config["column_variable"], alpha)
    elif method_key == "ols":
        result = ols_regression(df, config["outcome"], config["predictors"], alpha)
    elif method_key == "logistic":
        result = logistic_regression(df, config["outcome"], config["predictors"], alpha)
    elif method_key == "moderation":
        result = moderation_analysis(
            df, config["outcome"], config["predictor"], config["moderator"],
            config.get("controls") or [], alpha,
        )
    elif method_key == "mediation":
        result = mediation_analysis(
            df, config["outcome"], config["predictor"], config["mediator"],
            config.get("controls") or [], alpha,
            int(config.get("bootstrap_samples", 1000)),
            int(config.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown method: {method_key}")

    return _attach_automatic_descriptives(df, method_key, config, result)
