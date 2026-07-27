from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .profiling import dataset_profile
from .recommender import METHOD_LABELS, recommend_method


@dataclass
class AutoSpecification:
    method_key: str
    method_label: str
    config: dict[str, Any]
    confidence: str
    completed_fields: pd.DataFrame
    critical_blockers: list[str]
    assumptions_for_confirmation: list[str]
    rationale: str
    role_assignments: dict[str, str]
    framework_narrative: str


@dataclass
class GuidedReview:
    method_key: str
    method_label: str
    reason: str
    readiness: pd.DataFrame
    role_suggestions: pd.DataFrame
    construct_suggestions: list[dict[str, Any]]
    data_findings: pd.DataFrame
    next_actions: list[str]
    auto_specification: AutoSpecification | None = None


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _plain_name(value: str) -> str:
    return _normalise_name(value).replace("_", " ")


def _measurement(series: pd.Series) -> str:
    unique = int(series.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/time"
    if pd.api.types.is_numeric_dtype(series):
        if unique == 2:
            return "Binary"
        if pd.api.types.is_integer_dtype(series) and unique <= max(20, int(len(series) * 0.10)):
            return "Ordinal" if unique <= 7 else "Count"
        return "Continuous"
    return "Binary" if unique == 2 else "Nominal"


def suggest_variable_roles(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcome_claimed = False
    for column in df.columns:
        series = df[column]
        name = _normalise_name(column)
        unique = int(series.nunique(dropna=True))
        ratio = unique / max(int(series.notna().sum()), 1)
        role = "Unassigned"
        confidence = "Low"
        rationale = "No strong naming or data-pattern signal was detected."

        if re.search(r"(^|_)(id|uuid|identifier|student_id|respondent_id|case_id)($|_)", name) or (ratio > 0.98 and not pd.api.types.is_float_dtype(series)):
            role, confidence = "Identifier", "High"
            rationale = "The name or near-unique values indicate an identifier."
        elif re.search(r"(^|_)(year|time|wave|period|date|month|quarter|semester)($|_)", name):
            role, confidence = "Time", "High"
            rationale = "The variable name indicates a time or measurement occasion."
        elif re.search(r"(^|_)(cluster|school|class|centre|center|site|facility|team|department|district)($|_)", name):
            role, confidence = "Cluster", "Medium"
            rationale = "The name suggests a higher-level grouping or cluster variable."
        elif re.search(r"(^|_)(group|treatment|arm|cohort|gender|sex|region|category)($|_)", name) or (not pd.api.types.is_numeric_dtype(series) and unique <= 12):
            role, confidence = "Group", "Medium"
            rationale = "The variable has a small set of categories suitable for grouping or profiling."
        elif re.search(r"(^|_)(outcome|performance|result|score|achievement|effectiveness|success|passed|dependent|dv|y)($|_)", name) and not outcome_claimed:
            role, confidence = "Outcome", "Medium"
            rationale = "The name commonly denotes a study outcome. Confirm this against the objective."
            outcome_claimed = True
        elif re.search(r"(^|_)(mediator|mediation|mediating|indirect|motivation)($|_)", name):
            role, confidence = "Mediator", "Low"
            rationale = "The name may denote a mediator, but the conceptual framework should confirm it."
        elif re.search(r"(^|_)(moderator|moderating|interaction)($|_)", name):
            role, confidence = "Moderator", "Medium"
            rationale = "The name explicitly indicates a moderator or interaction variable."
        elif re.search(r"(?:^|_)[a-z][a-z_]*\d+$", name) or re.search(r"(^|_)item_?\d+$", name):
            role, confidence = "Scale item", "Medium"
            rationale = "The name follows a common numbered-item pattern."
        elif pd.api.types.is_numeric_dtype(series):
            role, confidence = "Predictor", "Low"
            rationale = "The numeric variable may be a predictor, control or scale item. Confirm its role."

        rows.append({
            "variable": column,
            "suggested_role": role,
            "measurement": _measurement(series),
            "confidence": confidence,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)


def suggest_constructs(df: pd.DataFrame, minimum_items: int = 2) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for column in df.select_dtypes(include="number").columns:
        raw = str(column)
        match = re.match(r"^(.*?)[_\- ]?(\d+)$", raw)
        if not match:
            continue
        prefix = match.group(1).strip(" _-")
        if not prefix:
            continue
        key = _normalise_name(prefix)
        groups.setdefault(key, []).append(raw)

    suggestions: list[dict[str, Any]] = []
    for prefix, items in groups.items():
        if len(items) < minimum_items:
            continue
        items = sorted(items, key=lambda value: int(re.search(r"(\d+)$", value).group(1)))
        display = prefix.replace("_", " ").title()
        suggestions.append({
            "name": display,
            "mode": "Reflective (Mode A)",
            "items": items,
            "confidence": "Medium",
            "rationale": "The numeric columns share a repeated item prefix. Confirm the construct name and measurement logic.",
        })
    return suggestions


def _data_findings(df: pd.DataFrame) -> pd.DataFrame:
    profile = dataset_profile(df)
    overview = profile["overview"].iloc[0]
    numeric = len(df.select_dtypes(include="number").columns)
    categorical = df.shape[1] - numeric
    findings = [
        {"check": "Rows", "finding": int(overview["rows"]), "interpretation": "Available records before method-specific complete-case selection."},
        {"check": "Variables", "finding": int(df.shape[1]), "interpretation": f"{numeric} numeric and {categorical} non-numeric variables."},
        {"check": "Missing cells", "finding": int(overview["total_missing_cells"]), "interpretation": "Review the missingness table before choosing a treatment."},
        {"check": "Duplicate rows", "finding": int(overview["duplicate_rows"]), "interpretation": "Verify whether duplicates are errors or legitimate repeated observations."},
    ]
    return pd.DataFrame(findings)


def _readiness(study: dict[str, Any], df: pd.DataFrame | None, framework: pd.DataFrame | None) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(component: str, complete: bool, guidance: str):
        checks.append({"component": component, "status": "Ready" if complete else "Needs attention", "guidance": guidance})

    add("Study objective", bool(str(study.get("objective", "")).strip()), "State what relationship, difference, prediction or latent structure will be examined.")
    add("Hypothesis", bool(str(study.get("hypothesis", "")).strip()), "A hypothesis is optional for exploratory or descriptive work but should be stated for confirmatory analysis.")
    add("Dataset", df is not None and not df.empty, "Upload CSV or Excel data and confirm the correct sheet.")
    add("Outcome type", bool(study.get("outcome_type")), "Confirm whether the outcome is continuous, binary, categorical, ordinal or count.")
    assigned = False
    if isinstance(framework, pd.DataFrame) and not framework.empty and "role" in framework:
        assigned = bool((framework["role"].astype(str) != "Unassigned").any())
    add("Variable roles", assigned, "The agent can complete roles from the objective and hypothesis, but ambiguous roles need confirmation.")
    add("Conceptual framework", bool(str(study.get("framework_notes", "")).strip()), "The agent can infer a provisional framework from the objective and hypotheses when wording is explicit.")
    return pd.DataFrame(checks)


def _role_map(framework: pd.DataFrame | None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(framework, pd.DataFrame) and not framework.empty and {"variable", "role"}.issubset(framework.columns):
        for _, row in framework.iterrows():
            role = str(row.get("role", "Unassigned"))
            variable = str(row.get("variable", ""))
            if variable and role != "Unassigned":
                result.setdefault(role, []).append(variable)
    return result


def _mentioned_columns(text: str, columns: list[str]) -> list[str]:
    lower = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    matches: list[tuple[int, int, str]] = []
    for column in columns:
        phrase = _plain_name(column)
        if not phrase:
            continue
        match = re.search(rf"\b{re.escape(phrase)}\b", lower)
        if match:
            matches.append((match.start(), -len(phrase), column))
            continue
        tokens = [token for token in phrase.split() if len(token) > 2]
        score = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", lower))
        if tokens and score == len(tokens):
            matches.append((lower.find(tokens[0]), -len(phrase), column))
    return [item[2] for item in sorted(matches)]


def _pick_first(candidates: list[str], allowed: list[str], exclude: set[str] | None = None) -> str | None:
    excluded = exclude or set()
    return next((value for value in candidates if value in allowed and value not in excluded), None)


def _field_rows(config: dict[str, Any], source: str = "AI-completed") -> pd.DataFrame:
    rows = []
    for key, value in config.items():
        if key in {"diagram_settings"}:
            display = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, dict):
            display = "; ".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in value.items())
        elif isinstance(value, list):
            display = "; ".join(" -> ".join(v) if isinstance(v, tuple) else str(v) for v in value)
        else:
            display = value
        rows.append({"field": key.replace("_", " ").title(), "value": display, "source": source})
    return pd.DataFrame(rows)


def _roles_from_config(method_key: str, config: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for field, role in [
        ("outcome", "Outcome"), ("predictor", "Predictor"), ("mediator", "Mediator"),
        ("moderator", "Moderator"), ("group", "Group"), ("cluster", "Cluster"),
        ("entity", "Entity"), ("time", "Time"), ("subject_id", "Identifier"),
    ]:
        value = config.get(field)
        if value:
            roles[str(value)] = role
    for value in config.get("predictors") or []:
        roles[str(value)] = "Predictor"
    for value in config.get("controls") or []:
        roles[str(value)] = "Control"
    for value in config.get("mediators") or []:
        roles[str(value)] = "Mediator"
    for value in config.get("level1_predictors") or []:
        roles[str(value)] = "Predictor"
    for value in config.get("level2_predictors") or []:
        roles[str(value)] = "Predictor"
    for value in config.get("measurements") or []:
        roles[str(value)] = "Scale item"
    for value in config.get("items") or []:
        roles[str(value)] = "Scale item"
    for items in (config.get("construct_map") or {}).values():
        for value in items:
            roles[str(value)] = "Scale item"
    if method_key == "network" and config.get("network_input") == "Correlation or partial-correlation network":
        for value in config.get("variables") or []:
            roles.setdefault(str(value), "Scale item")
    return roles


def _framework_narrative(method_key: str, config: dict[str, Any]) -> str:
    if method_key in {"sem", "pls_sem"}:
        relations = []
        for relation in config.get("structural_relations") or []:
            if relation.get("type") == "Mediator":
                relations.append(f"{relation.get('predictor')} affects {relation.get('outcome')} through {relation.get('mediator')}")
            elif relation.get("type") == "Moderator":
                relations.append(f"{relation.get('moderator')} moderates the relationship between {relation.get('predictor')} and {relation.get('outcome')}")
            else:
                relations.append(f"{relation.get('predictor')} predicts {relation.get('outcome')}")
        return ". ".join(relations) + ("." if relations else "")
    if method_key in {"mediation", "moderated_mediation"}:
        text = f"{config.get('predictor')} predicts {config.get('outcome')} through {config.get('mediator')}"
        if config.get("moderator"):
            text += f", with the first-stage effect conditional on {config.get('moderator')}"
        return text + "."
    if method_key in {"moderation", "advanced_moderation"}:
        return f"{config.get('moderator')} changes the relationship between {config.get('predictor')} and {config.get('outcome')}."
    if method_key in {"ols", "logistic", "panel", "multilevel"}:
        predictors = config.get("predictors") or config.get("level1_predictors") or []
        return f"{', '.join(map(str, predictors))} predict {config.get('outcome')}."
    if method_key == "network":
        return "The selected nodes are analysed as an interconnected system. Edges represent the confirmed tie, correlation or partial-correlation construction rule."
    return ""


def build_auto_specification(
    study: dict[str, Any],
    df: pd.DataFrame | None,
    framework: pd.DataFrame | None,
    method_key: str | None = None,
) -> AutoSpecification | None:
    if df is None or df.empty:
        return None
    objective = str(study.get("objective", ""))
    hypothesis = str(study.get("hypothesis", ""))
    framework_notes = str(study.get("framework_notes", ""))
    text = " ".join([objective, hypothesis, framework_notes])
    recommendation = recommend_method(objective, hypothesis, str(study.get("outcome_type", "continuous")), int(study.get("group_count", 0)) or None, bool(study.get("paired", False)))
    key = method_key or recommendation["method_key"]
    columns = list(df.columns)
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = [column for column in columns if column not in numeric]
    roles = _role_map(framework)
    suggestions = suggest_variable_roles(df)
    inferred_roles = {role: suggestions.loc[suggestions["suggested_role"] == role, "variable"].tolist() for role in suggestions["suggested_role"].unique()}
    mentioned = _mentioned_columns(text, columns)
    blockers: list[str] = []
    assumptions: list[str] = []
    alpha = float(study.get("alpha", 0.05))
    config: dict[str, Any] = {"alpha": alpha}

    # Explicit framework roles take precedence. Name-based outcome signals come next.
    # In effect/prediction wording, the final mentioned variable is usually the dependent variable.
    outcome_candidates = roles.get("Outcome", []) + inferred_roles.get("Outcome", []) + list(reversed([v for v in mentioned if v in numeric]))
    predictor_candidates = roles.get("Predictor", []) + [v for v in mentioned if v in numeric] + inferred_roles.get("Predictor", [])
    group_candidates = roles.get("Group", []) + [v for v in mentioned if v in columns] + inferred_roles.get("Group", [])
    mediator_candidates = roles.get("Mediator", []) + [v for v in mentioned if v in numeric] + inferred_roles.get("Mediator", [])
    moderator_candidates = roles.get("Moderator", []) + [v for v in mentioned if v in numeric] + inferred_roles.get("Moderator", [])
    control_candidates = roles.get("Control", [])
    outcome = _pick_first(outcome_candidates, columns)

    if key == "descriptive":
        config["variables"] = mentioned or columns[: min(12, len(columns))]
    elif key == "reliability":
        items = roles.get("Scale item", []) or [item for group in suggest_constructs(df) for item in group["items"]]
        config["items"] = list(dict.fromkeys(item for item in items if item in numeric))
        if len(config["items"]) < 2:
            blockers.append("The agent could not identify at least two scale items. Select the items that form the scale.")
    elif key in {"correlation", "efa"}:
        variables = [v for v in mentioned if v in numeric] or roles.get("Scale item", []) or numeric[: min(10, len(numeric))]
        if key == "correlation":
            config.update({"variables": variables, "correlation_method": "pearson"})
            if len(variables) < 2: blockers.append("At least two numeric variables are required.")
        else:
            config.update({"items": variables, "n_factors": None, "rotation": "varimax", "parallel_iterations": 100, "random_state": 42})
            if len(variables) < 3: blockers.append("At least three observed items are required for EFA.")
    elif key in {"independent_t", "anova"}:
        config["outcome"] = outcome
        config["group"] = _pick_first(group_candidates, columns, {outcome} if outcome else set())
        if not config["outcome"]: blockers.append("The continuous outcome could not be identified from the objective, hypothesis or variable roles.")
        if not config["group"]: blockers.append("The grouping variable could not be identified.")
    elif key == "paired_t":
        measures = [v for v in mentioned if v in numeric]
        config["before"] = measures[0] if measures else None
        config["after"] = measures[1] if len(measures) > 1 else None
        if not config["before"] or not config["after"]: blockers.append("Identify the two repeated measurement columns.")
    elif key == "chi_square":
        choices = [v for v in mentioned if v in columns] or group_candidates + categorical
        config["row_variable"] = choices[0] if choices else None
        config["column_variable"] = next((v for v in choices[1:] if v != config["row_variable"]), None)
        if not config["row_variable"] or not config["column_variable"]: blockers.append("Two categorical variables are required.")
    elif key in {"ols", "logistic"}:
        config["outcome"] = outcome
        config["predictors"] = list(dict.fromkeys(v for v in predictor_candidates + control_candidates if v in columns and v != outcome))
        if not config["outcome"]: blockers.append("The dependent variable could not be identified.")
        if not config["predictors"]: blockers.append("No predictors could be identified from the objective, hypothesis or framework.")
    elif key in {"mediation", "moderation", "advanced_moderation", "moderated_mediation"}:
        predictor = _pick_first(predictor_candidates, numeric, {outcome} if outcome else set())
        mediator = _pick_first(mediator_candidates, numeric, {outcome, predictor})
        moderator = _pick_first(moderator_candidates, numeric, {outcome, predictor, mediator})
        config.update({"outcome": outcome, "predictor": predictor, "controls": [v for v in control_candidates if v in numeric and v not in {outcome, predictor, mediator, moderator}]})
        if key in {"mediation", "moderated_mediation"}: config["mediator"] = mediator
        if key in {"moderation", "advanced_moderation", "moderated_mediation"}: config["moderator"] = moderator
        if key in {"mediation", "moderated_mediation"}: config.update({"bootstrap_samples": 1000, "random_state": 42})
        for field in ["outcome", "predictor"] + (["mediator"] if key in {"mediation", "moderated_mediation"} else []) + (["moderator"] if key in {"moderation", "advanced_moderation", "moderated_mediation"} else []):
            if not config.get(field): blockers.append(f"The {field} could not be identified unambiguously.")
    elif key == "parallel_mediation":
        predictor = _pick_first(predictor_candidates, numeric, {outcome} if outcome else set())
        mediators = list(dict.fromkeys(v for v in mediator_candidates if v in numeric and v not in {outcome, predictor}))
        config.update({"outcome": outcome, "predictor": predictor, "mediators": mediators, "controls": [v for v in control_candidates if v in numeric and v not in {outcome, predictor, *mediators}], "bootstrap_samples": 1000, "random_state": 42})
        if not outcome or not predictor or len(mediators) < 2: blockers.append("Parallel mediation requires an outcome, predictor and at least two mediators.")
    elif key in {"cfa", "sem", "pls_sem"}:
        constructs = suggest_constructs(df)
        construct_map = {item["name"]: item["items"] for item in constructs}
        config["construct_map"] = construct_map
        config["measurement_modes"] = {item["name"]: "reflective" for item in constructs}
        if not construct_map or len(construct_map) < (2 if key != "cfa" else 1):
            blockers.append("The agent could not identify enough item blocks for the latent-variable model. Confirm construct names and indicators.")
        if key in {"sem", "pls_sem"}:
            construct_mentions = _mentioned_columns(text, list(construct_map))
            ordered = construct_mentions or list(construct_map)
            paths = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)] if len(ordered) >= 2 else []
            config["paths"] = paths
            config["structural_relations"] = [{"type": "Direct", "predictor": a, "outcome": b} for a, b in paths]
            config["moderations"] = []
            assumptions.append("Structural paths were provisionally ordered from construct mentions or construct order. Confirm causal direction before analysis.")
            if not paths: blockers.append("No structural path could be inferred among the constructs.")
        config["estimator"] = "ML" if key in {"cfa", "sem"} else "Path"
        if key == "pls_sem":
            config.update({"weighting_scheme": "Path", "bootstrap_samples": 500, "max_iter": 300, "tolerance": 1e-7})
        config["random_state"] = 42
        config["diagram_settings"] = {"layout": "Left to right", "arrow_style": "Curved", "show_indicators": True, "show_loadings": True, "show_indicator_names": True, "show_coefficients": True, "show_p_values": True, "show_fit": True, "significance_colours": True, "monochrome": False, "transparent": False, "resolution": "High resolution", "construct_order": list(construct_map)}
    elif key == "repeated_measures":
        measures = [v for v in mentioned if v in numeric]
        if len(measures) < 2:
            measures = [v for v in numeric if re.search(r"(?:time|wave|t|score)[_ -]?\d+$", _normalise_name(v))]
        config["measurements"] = measures
        config["subject_id"] = _pick_first(roles.get("Identifier", []) + inferred_roles.get("Identifier", []), columns)
        if len(measures) < 2: blockers.append("At least two repeated measurement columns must be identified.")
    elif key in {"multilevel", "panel"}:
        if key == "multilevel":
            cluster = _pick_first(roles.get("Cluster", []) + inferred_roles.get("Cluster", []) + roles.get("Group", []), columns)
            predictors = [v for v in predictor_candidates if v in numeric and v != outcome]
            config.update({"outcome_family": str(study.get("outcome_type", "continuous")).title(), "outcome": outcome, "cluster": cluster, "level1_predictors": predictors, "level2_predictors": [], "centering": "Group-mean with contextual effect", "estimator": "REML" if str(study.get("outcome_type", "continuous")) == "continuous" else "GEE robust", "random_slope": None, "optimizer": "lbfgs", "gee_correlation": "Exchangeable"})
            if not outcome or not cluster or not predictors: blockers.append("Multilevel analysis requires a clear outcome, cluster identifier and at least one predictor.")
        else:
            entity = _pick_first(roles.get("Entity", []) + roles.get("Identifier", []) + inferred_roles.get("Identifier", []), columns)
            time = _pick_first(roles.get("Time", []) + inferred_roles.get("Time", []), columns)
            predictors = [v for v in predictor_candidates if v in numeric and v != outcome]
            config.update({"outcome": outcome, "entity": entity, "time": time, "predictors": predictors, "model_choice": "automatic", "include_time_effects": False})
            if not outcome or not entity or not time or not predictors: blockers.append("Panel analysis requires outcome, entity, time and predictor variables.")
    elif key == "network":
        normalized = {_normalise_name(column): column for column in columns}
        source = next((normalized[name] for name in normalized if re.search(r"(^|_)(source|from|sender|ego)($|_)", name)), None)
        target = next((normalized[name] for name in normalized if re.search(r"(^|_)(target|to|receiver|alter)($|_)", name)), None)
        weight = next((normalized[name] for name in normalized if re.search(r"(^|_)(weight|strength|frequency|tie)($|_)", name)), None)
        if source and target:
            config.update({"network_input": "Edge list", "source": source, "target": target, "weight": weight, "directed": bool(re.search(r"directed|sender|receiver|from|to", text.lower())), "allow_self_loops": False, "layout": "Spring", "random_graph_iterations": 50, "random_state": 42})
        else:
            variables = [v for v in mentioned if v in numeric] or numeric[: min(20, len(numeric))]
            config.update({"network_input": "Correlation or partial-correlation network", "variables": variables, "network_estimator": "Partial correlation (Graphical Lasso)" if len(df) >= max(50, len(variables) * 5) else "Pearson correlation", "edge_threshold": 0.20, "retain_negative": True, "bootstrap_samples": 200, "layout": "Spring", "random_graph_iterations": 50, "random_state": 42, "group_variable": None, "group_values": [], "permutation_samples": 0})
            if len(variables) < 3: blockers.append("Network analysis requires source and target columns or at least three numeric variables.")
        assumptions.append("The agent selected a network construction rule from the dataset structure. Confirm node, edge, weight and threshold meanings before interpreting the network.")
    else:
        blockers.append("The selected method is not yet supported by the auto-completion agent.")

    if key not in {"descriptive", "network"}:
        config.setdefault("profile_variables", [v for v in group_candidates[:3] if v in columns])
    confidence = "High" if not blockers and not assumptions else "Medium" if not blockers else "Low"
    return AutoSpecification(
        method_key=key,
        method_label=METHOD_LABELS.get(key, key),
        config=config,
        confidence=confidence,
        completed_fields=_field_rows(config),
        critical_blockers=blockers,
        assumptions_for_confirmation=assumptions,
        rationale=recommendation["reason"],
        role_assignments=_roles_from_config(key, config),
        framework_narrative=_framework_narrative(key, config),
    )


def build_guided_review(
    study: dict[str, Any],
    df: pd.DataFrame | None = None,
    framework: pd.DataFrame | None = None,
) -> GuidedReview:
    recommendation = recommend_method(
        str(study.get("objective", "")),
        str(study.get("hypothesis", "")),
        str(study.get("outcome_type", "continuous")),
        int(study.get("group_count", 0)) or None,
        bool(study.get("paired", False)),
    )
    role_suggestions = suggest_variable_roles(df) if df is not None and not df.empty else pd.DataFrame(
        columns=["variable", "suggested_role", "measurement", "confidence", "rationale"]
    )
    constructs = suggest_constructs(df) if df is not None and not df.empty else []
    findings = _data_findings(df) if df is not None and not df.empty else pd.DataFrame(
        [{"check": "Dataset", "finding": "Not loaded", "interpretation": "Upload a dataset to receive data-specific guidance."}]
    )
    readiness = _readiness(study, df, framework)
    auto_spec = build_auto_specification(study, df, framework, recommendation["method_key"])
    next_actions = []
    if auto_spec and auto_spec.critical_blockers:
        next_actions.extend(auto_spec.critical_blockers)
    for _, row in readiness[readiness["status"] != "Ready"].iterrows():
        if row["component"] not in {"Variable roles", "Conceptual framework"}:
            next_actions.append(f"Complete {str(row['component']).lower()}: {row['guidance']}")
    if not next_actions:
        next_actions.append("Review the AI-completed specification, confirm provisional assumptions, then run the guided analysis.")
    return GuidedReview(
        method_key=recommendation["method_key"],
        method_label=METHOD_LABELS[recommendation["method_key"]],
        reason=recommendation["reason"],
        readiness=readiness,
        role_suggestions=role_suggestions,
        construct_suggestions=constructs,
        data_findings=findings,
        next_actions=list(dict.fromkeys(next_actions)),
        auto_specification=auto_spec,
    )
