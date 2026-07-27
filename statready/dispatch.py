from __future__ import annotations

from typing import Any
import math

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
from .figures import attach_latent_variable_figures
from .pls_sem import partial_least_squares_sem
from .network_analysis import network_analysis
from .multilevel import multilevel_linear_model
from .phase2 import (
    exploratory_factor_analysis,
    confirmatory_factor_analysis,
    structural_equation_model,
    repeated_measures_anova,
    mixed_effects_model,
    panel_data_analysis,
    advanced_moderation_analysis,
    parallel_mediation_analysis,
    moderated_mediation_analysis,
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
        "efa": config.get("items") or [],
        "cfa": [item for items in (config.get("construct_map") or {}).values() for item in items],
        "sem": [item for items in (config.get("construct_map") or {}).values() for item in items],
        "pls_sem": [item for items in (config.get("construct_map") or {}).values() for item in items],
        "repeated_measures": config.get("measurements") or [],
        "mixed_effects": [config.get("outcome"), config.get("cluster"), *(config.get("predictors") or []), config.get("random_slope")],
        "multilevel": [config.get("outcome"), config.get("cluster"), *(config.get("level1_predictors") or []), *(config.get("level2_predictors") or []), config.get("random_slope")],
        "panel": [config.get("outcome"), config.get("entity"), config.get("time"), *(config.get("predictors") or [])],
        "advanced_moderation": [config.get("outcome"), config.get("predictor"), config.get("moderator"), *(config.get("controls") or [])],
        "parallel_mediation": [config.get("outcome"), config.get("predictor"), *(config.get("mediators") or []), *(config.get("controls") or [])],
        "moderated_mediation": [config.get("outcome"), config.get("predictor"), config.get("mediator"), config.get("moderator"), *(config.get("controls") or [])],
        "network": (
            [config.get("source"), config.get("target"), config.get("weight")]
            if config.get("network_input") == "Edge list" else
            ([config.get("node_label"), *(config.get("adjacency_columns") or [])]
             if config.get("network_input") == "Adjacency matrix" else
             [*(config.get("variables") or []), config.get("group_variable")])
        ),
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



def _find_structural_table(result) -> pd.DataFrame:
    for name in ["PLS structural path estimates", "Structural path estimates"]:
        table = result.tables.get(name)
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    return pd.DataFrame()


def _path_values(table: pd.DataFrame, predictor: str, outcome: str) -> tuple[float, float, float]:
    if table.empty:
        return float("nan"), float("nan"), float("nan")
    row = table[(table["predictor"].astype(str) == str(predictor)) & (table["outcome"].astype(str) == str(outcome))]
    if row.empty:
        return float("nan"), float("nan"), float("nan")
    row = row.iloc[0]
    estimate = row.get("standardized_estimate", row.get("estimate", float("nan")))
    se = row.get("bootstrap_std_error", row.get("std_error_approx", row.get("std_error", float("nan"))))
    p = row.get("bootstrap_p", row.get("p_value_approx", row.get("p_value", float("nan"))))
    return float(estimate), float(se) if pd.notna(se) else float("nan"), float(p) if pd.notna(p) else float("nan")


def _attach_specified_relation_effects(result, relations: list[dict] | None, alpha: float):
    relations = relations or []
    table = _find_structural_table(result)
    if table.empty or not relations:
        return result
    import numpy as np
    from scipy import stats
    mediation_rows = []
    moderation_rows = []
    critical = stats.norm.ppf(1 - alpha / 2)
    for relation in relations:
        relation_type = relation.get("type")
        predictor = relation.get("predictor")
        outcome = relation.get("outcome")
        if relation_type == "Mediator":
            mediator = relation.get("mediator")
            a, se_a, p_a = _path_values(table, predictor, mediator)
            b, se_b, p_b = _path_values(table, mediator, outcome)
            direct, se_direct, p_direct = _path_values(table, predictor, outcome)
            indirect = a * b if np.isfinite(a) and np.isfinite(b) else np.nan
            indirect_se = math.sqrt((b ** 2) * (se_a ** 2) + (a ** 2) * (se_b ** 2)) if all(np.isfinite(v) for v in [a, b, se_a, se_b]) else np.nan
            z_value = indirect / indirect_se if np.isfinite(indirect_se) and indirect_se > 0 else np.nan
            indirect_p = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            mediation_rows.append({
                "predictor": predictor, "mediator": mediator, "outcome": outcome,
                "a_path": a, "a_path_p": p_a, "b_path": b, "b_path_p": p_b,
                "indirect_effect": indirect, "indirect_std_error_delta": indirect_se,
                "indirect_z_delta": z_value, "indirect_p_delta": indirect_p,
                "indirect_ci_lower_delta": indirect - critical * indirect_se if np.isfinite(indirect_se) else np.nan,
                "indirect_ci_upper_delta": indirect + critical * indirect_se if np.isfinite(indirect_se) else np.nan,
                "direct_effect": direct, "direct_effect_p": p_direct,
                "total_effect": direct + indirect if np.isfinite(direct) and np.isfinite(indirect) else np.nan,
                "direct_path_requested": bool(relation.get("include_direct", True)),
            })
        elif relation_type == "Moderator":
            moderator = relation.get("moderator")
            interaction = f"{predictor} × {moderator}"
            estimate, se, p_value = _path_values(table, interaction, outcome)
            moderation_rows.append({
                "predictor": predictor, "moderator": moderator, "outcome": outcome,
                "interaction_term": interaction, "interaction_effect": estimate,
                "std_error": se, "p_value": p_value,
                "status": "Not estimated in CB-SEM" if not np.isfinite(estimate) else "Estimated",
            })
    if mediation_rows and "Specified mediation effects" not in result.tables:
        result.tables["Specified mediation effects"] = pd.DataFrame(mediation_rows)
        result.warnings.append("Specified mediation indirect-effect inference uses a delta-method approximation in this build. Confirm important indirect effects with joint bootstrap confidence intervals in specialist software.")
    if moderation_rows:
        result.tables["Specified moderation effects"] = pd.DataFrame(moderation_rows)
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
    elif method_key == "efa":
        result = exploratory_factor_analysis(
            df, config["items"], config.get("n_factors"), config.get("rotation", "varimax"),
            int(config.get("parallel_iterations", 100)), int(config.get("random_state", 42)), alpha,
        )
    elif method_key == "cfa":
        result = confirmatory_factor_analysis(
            df, config["construct_map"], alpha, int(config.get("random_state", 42)), config.get("estimator", "ML")
        )
    elif method_key == "sem":
        result = structural_equation_model(
            df, config["construct_map"], config["paths"], alpha, int(config.get("random_state", 42)), config.get("estimator", "ML")
        )
        if config.get("unsupported_moderations"):
            result.warnings.append("Latent moderation relations were recorded but not estimated in the internal covariance-based SEM engine. Use PLS-SEM or the advanced moderation module for those interactions.")
    elif method_key == "pls_sem":
        result = partial_least_squares_sem(
            df=df, construct_map=config["construct_map"], paths=config["paths"],
            measurement_modes=config.get("measurement_modes") or {},
            moderations=config.get("moderations") or [],
            structural_relations=config.get("structural_relations") or [],
            bootstrap_samples=int(config.get("bootstrap_samples", 500)),
            max_iter=int(config.get("max_iter", 300)),
            tolerance=float(config.get("tolerance", 1e-7)),
            weighting_scheme=config.get("weighting_scheme", "Path"),
            random_state=int(config.get("random_state", 42)), alpha=alpha,
        )
    elif method_key == "repeated_measures":
        result = repeated_measures_anova(df, config["measurements"], config.get("subject_id"), alpha)
    elif method_key == "mixed_effects":
        result = mixed_effects_model(
            df, config["outcome"], config["predictors"], config["cluster"],
            config.get("random_slope"), bool(config.get("reml", True)), alpha,
        )
    elif method_key == "multilevel":
        result = multilevel_linear_model(
            df=df, outcome=config["outcome"],
            level1_predictors=config.get("level1_predictors") or [],
            level2_predictors=config.get("level2_predictors") or [],
            cluster=config["cluster"], random_slope=config.get("random_slope"),
            estimator=config.get("estimator", "REML"),
            centering=config.get("centering", "Group-mean with contextual effect"),
            optimizer=config.get("optimizer", "lbfgs"),
            gee_correlation=config.get("gee_correlation", "Exchangeable"),
            outcome_family=config.get("outcome_family", "Continuous"), alpha=alpha,
        )
    elif method_key == "panel":
        result = panel_data_analysis(
            df, config["outcome"], config["predictors"], config["entity"], config["time"],
            config.get("model_choice", "automatic"), bool(config.get("include_time_effects", False)), alpha,
        )
    elif method_key == "advanced_moderation":
        result = advanced_moderation_analysis(
            df, config["outcome"], config["predictor"], config["moderator"], config.get("controls") or [], alpha,
        )
    elif method_key == "parallel_mediation":
        result = parallel_mediation_analysis(
            df, config["outcome"], config["predictor"], config["mediators"], config.get("controls") or [], alpha,
            int(config.get("bootstrap_samples", 1000)), int(config.get("random_state", 42)),
        )
    elif method_key == "network":
        result = network_analysis(df, config)
    elif method_key == "moderated_mediation":
        result = moderated_mediation_analysis(
            df, config["outcome"], config["predictor"], config["mediator"], config["moderator"],
            config.get("controls") or [], alpha, int(config.get("bootstrap_samples", 1000)), int(config.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown method: {method_key}")

    result = _attach_specified_relation_effects(result, config.get("structural_relations"), alpha)
    result.metadata["diagram_settings"] = config.get("diagram_settings") or {}
    result.metadata["structural_relations"] = config.get("structural_relations") or []
    result = attach_latent_variable_figures(result)
    return _attach_automatic_descriptives(df, method_key, config, result)
