from __future__ import annotations

from typing import Any

import pandas as pd

from .figures import render_latent_path_diagram


def specification_graph(method_key: str, config: dict[str, Any]) -> tuple[dict[str, list[str]], list[tuple[str, str]], list[dict]]:
    relations = config.get("structural_relations") or []
    if method_key in {"cfa", "sem", "pls_sem"}:
        return config.get("construct_map") or {}, [tuple(path) for path in config.get("paths") or []], relations

    paths: list[tuple[str, str]] = []
    nodes: list[str] = []
    outcome = config.get("outcome")
    if method_key in {"ols", "logistic", "panel", "mixed_effects"}:
        predictors = config.get("predictors") or []
        paths = [(str(p), str(outcome)) for p in predictors if p and outcome]
        nodes = [*predictors, outcome]
    elif method_key == "multilevel":
        predictors = [*(config.get("level1_predictors") or []), *(config.get("level2_predictors") or [])]
        paths = [(str(p), str(outcome)) for p in predictors if p and outcome]
        nodes = [*predictors, outcome]
    elif method_key in {"moderation", "advanced_moderation"}:
        predictor, moderator = config.get("predictor"), config.get("moderator")
        nodes = [predictor, moderator, outcome]
        paths = [(str(predictor), str(outcome)), (str(moderator), str(outcome))]
        relations = [{"type": "Moderator", "predictor": predictor, "moderator": moderator, "outcome": outcome}]
    elif method_key == "mediation":
        predictor, mediator = config.get("predictor"), config.get("mediator")
        nodes = [predictor, mediator, outcome]
        paths = [(str(predictor), str(mediator)), (str(mediator), str(outcome)), (str(predictor), str(outcome))]
        relations = [{"type": "Mediator", "predictor": predictor, "mediator": mediator, "outcome": outcome, "include_direct": True}]
    elif method_key == "parallel_mediation":
        predictor, mediators = config.get("predictor"), config.get("mediators") or []
        nodes = [predictor, *mediators, outcome]
        for mediator in mediators:
            paths.extend([(str(predictor), str(mediator)), (str(mediator), str(outcome))])
        paths.append((str(predictor), str(outcome)))
        relations = [{"type": "Mediator", "predictor": predictor, "mediator": m, "outcome": outcome, "include_direct": True} for m in mediators]
    elif method_key == "moderated_mediation":
        predictor, mediator, moderator = config.get("predictor"), config.get("mediator"), config.get("moderator")
        nodes = [predictor, mediator, moderator, outcome]
        paths = [(str(predictor), str(mediator)), (str(mediator), str(outcome)), (str(predictor), str(outcome)), (str(moderator), str(mediator))]
        relations = [
            {"type": "Mediator", "predictor": predictor, "mediator": mediator, "outcome": outcome, "include_direct": True},
            {"type": "Moderator", "predictor": predictor, "moderator": moderator, "outcome": mediator},
        ]
    else:
        return {}, [], []
    clean_nodes = [str(node) for node in nodes if node]
    return {node: [] for node in dict.fromkeys(clean_nodes)}, list(dict.fromkeys(paths)), relations


def proposed_diagram(method_key: str, config: dict[str, Any], title: str = "Proposed analysis model") -> bytes | None:
    construct_map, paths, relations = specification_graph(method_key, config)
    if not construct_map or not paths:
        return None
    settings = dict(config.get("diagram_settings") or {})
    settings.update({
        "show_indicators": bool(any(construct_map.values())),
        "show_loadings": False,
        "show_coefficients": False,
        "show_p_values": False,
        "show_fit": False,
    })
    settings.setdefault("layout", "Left to right")
    settings.setdefault("arrow_style", "Curved")
    return render_latent_path_diagram(
        construct_map=construct_map,
        loading_table=pd.DataFrame(),
        paths=paths,
        path_table=pd.DataFrame(),
        fit_table=pd.DataFrame(),
        title=title,
        settings=settings,
        structural_relations=relations,
    )


def _coefficient_table(result) -> pd.DataFrame:
    candidates = [
        "Selected coefficient table", "Coefficients and odds ratios", "Regression coefficients",
        "Moderation coefficients", "Fixed effects", "Multilevel fixed effects", "Panel coefficients",
    ]
    for name in candidates:
        table = result.tables.get(name)
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    return pd.DataFrame()


def _term_columns(table: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    term = next((c for c in ["term", "variable", "predictor"] if c in table.columns), None)
    estimate = next((c for c in ["standardized_estimate", "estimate", "coefficient", "log_odds"] if c in table.columns), None)
    pvalue = next((c for c in ["p_value", "p_value_approx", "bootstrap_p"] if c in table.columns), None)
    return term, estimate, pvalue


def _normalised_term(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _estimate_from_table(table: pd.DataFrame, variable: str) -> tuple[float | None, float | None]:
    if table is None or table.empty:
        return None, None
    term_col, estimate_col, p_col = _term_columns(table)
    if not term_col or not estimate_col:
        return None, None
    target = _normalised_term(variable)
    terms = table[term_col].astype(str)
    exact = table[terms.map(_normalised_term) == target]
    if exact.empty:
        # Moderation procedures often prefix centred variables with c_.
        exact = table[terms.map(_normalised_term).map(lambda value: value.endswith(target) or value.startswith(target))]
    if exact.empty:
        return None, None
    row = exact.iloc[0]
    estimate = row.get(estimate_col)
    pvalue = row.get(p_col) if p_col else None
    return (float(estimate) if pd.notna(estimate) else None, float(pvalue) if pvalue is not None and pd.notna(pvalue) else None)


def _observed_path_table(result, method_key: str, config: dict[str, Any], paths: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if method_key == "mediation":
        predictor, mediator, outcome = config.get("predictor"), config.get("mediator"), config.get("outcome")
        a_est, a_p = _estimate_from_table(result.tables.get("Mediator model coefficients", pd.DataFrame()), predictor)
        b_est, b_p = _estimate_from_table(result.tables.get("Outcome model coefficients", pd.DataFrame()), mediator)
        direct, direct_p = _estimate_from_table(result.tables.get("Outcome model coefficients", pd.DataFrame()), predictor)
        for source, target, estimate, pvalue in [
            (predictor, mediator, a_est, a_p), (mediator, outcome, b_est, b_p), (predictor, outcome, direct, direct_p),
        ]:
            if source and target:
                rows.append({"predictor": source, "outcome": target, "estimate": estimate, "p_value": pvalue})
        return pd.DataFrame(rows)
    coef = _coefficient_table(result)
    for source, target in paths:
        estimate, pvalue = _estimate_from_table(coef, source)
        rows.append({"predictor": source, "outcome": target, "estimate": estimate, "p_value": pvalue})
    return pd.DataFrame(rows)


def attach_observed_path_figures(result, method_key: str, config: dict[str, Any]):
    construct_map, paths, relations = specification_graph(method_key, config)
    if not construct_map or not paths or method_key in {"cfa", "sem", "pls_sem"}:
        return result
    proposed = proposed_diagram(method_key, config, "Proposed model before estimation")
    if proposed:
        result.figures = {"Proposed model before estimation": proposed, **result.figures}

    path_table = _observed_path_table(result, method_key, config, paths)
    settings = dict(config.get("diagram_settings") or {})
    settings.update({"show_indicators": False, "show_loadings": False, "show_fit": False})
    settings.setdefault("layout", "Left to right")
    settings.setdefault("arrow_style", "Curved")
    estimated = render_latent_path_diagram(
        construct_map=construct_map,
        loading_table=pd.DataFrame(),
        paths=paths,
        path_table=path_table,
        fit_table=pd.DataFrame(),
        title="Estimated analysis model",
        settings=settings,
        structural_relations=relations,
    )
    result.figures["Estimated analysis model"] = estimated
    return result
