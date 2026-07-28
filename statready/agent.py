from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .profiling import dataset_profile
from .recommender import METHOD_LABELS, recommend_method
from .ai_providers import AgentAssignmentDraft, AgentProviderResult, generate_analysis_programme_with_provider


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


@dataclass
class AnalysisAssignment:
    objective_id: str
    objective: str
    hypothesis_id: str
    hypothesis: str
    method_key: str
    method_label: str
    rationale: str
    specification: AutoSpecification | None
    status: str


@dataclass
class AnalysisProgramme:
    assignments: list[AnalysisAssignment]
    mapping_table: pd.DataFrame
    critical_questions: list[str]
    readiness_summary: str
    provider: str = "Deterministic local agent"
    models_attempted: list[str] = field(default_factory=list)
    fallback_used: bool = False
    provider_warnings: list[str] = field(default_factory=list)


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


def _structured_framework(study: dict[str, Any]) -> dict[str, Any]:
    value = study.get("framework_structured") or {}
    return value if isinstance(value, dict) else {}


def _framework_method_override(
    objective: str, hypothesis: str, default_key: str, structured: dict[str, Any]
) -> str:
    if not structured:
        return default_key
    text = f"{objective} {hypothesis}".lower()
    construct_map = structured.get("construct_map") or {}
    relations = structured.get("structural_relations") or []
    latent = sum(1 for items in construct_map.values() if len(items) >= 2) >= 2
    formative = any(str(mode).lower() == "formative" for mode in (structured.get("measurement_modes") or {}).values())
    has_mediation = any(str(r.get("type", "")).lower() == "mediator" for r in relations) or "mediat" in text
    has_moderation = any(str(r.get("type", "")).lower() == "moderator" for r in relations) or "moderat" in text
    if latent and (has_mediation or has_moderation or default_key in {"ols", "mediation", "moderation", "sem", "pls_sem"}):
        return "pls_sem" if formative or has_moderation else "sem"
    return default_key


def parse_research_items(text: str, prefix: str) -> list[tuple[str, str]]:
    raw_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not raw_lines and str(text).strip():
        raw_lines = [str(text).strip()]
    items: list[tuple[str, str]] = []
    for index, line in enumerate(raw_lines, start=1):
        cleaned = re.sub(r"^\s*(?:[A-Za-z]?\d+|[ivxlcdm]+)[\.\):\-]\s*", "", line, flags=re.I).strip()
        items.append((f"{prefix}{index}", cleaned or line))
    return items


def _valid_column(value: Any, columns: list[str]) -> str | None:
    return str(value) if value is not None and str(value) in columns else None


def _valid_column_list(value: Any, columns: list[str]) -> list[str]:
    values = value if isinstance(value, list) else [value] if value else []
    return list(dict.fromkeys(str(item) for item in values if str(item) in columns))


def _normalise_relation_type(value: str) -> str:
    mapping = {"mediation": "Mediator", "mediator": "Mediator", "moderation": "Moderator", "moderator": "Moderator", "covariance": "Covariance"}
    return mapping.get(str(value).lower(), "Direct")


def _overlay_provider_draft(
    spec: AutoSpecification,
    draft: AgentAssignmentDraft,
    df: pd.DataFrame,
) -> AutoSpecification:
    columns = list(df.columns)
    config = dict(spec.config)
    roles = {str(key).lower(): value for key, value in (draft.roles or {}).items()}
    hints = dict(draft.config_hints or {})

    single_fields = [
        "outcome", "predictor", "mediator", "moderator", "group", "cluster", "entity", "time",
        "subject_id", "before", "after", "row_variable", "column_variable", "source", "target", "weight",
        "random_slope", "group_variable",
    ]
    list_fields = [
        "predictors", "controls", "mediators", "variables", "items", "measurements", "level1_predictors",
        "level2_predictors", "profile_variables", "group_values",
    ]
    role_aliases = {
        "outcome": "outcome", "dependent": "outcome", "dependent_variable": "outcome",
        "predictor": "predictor", "independent": "predictor", "independent_variable": "predictor",
        "predictors": "predictors", "independent_variables": "predictors",
        "mediator": "mediator", "mediators": "mediators", "moderator": "moderator",
        "controls": "controls", "control": "controls", "group": "group", "cluster": "cluster",
        "entity": "entity", "time": "time", "identifier": "subject_id", "subject_id": "subject_id",
    }
    for role_name, field_name in role_aliases.items():
        if role_name in roles and field_name not in hints:
            hints[field_name] = roles[role_name]

    for field_name in single_fields:
        if field_name in hints:
            value = _valid_column(hints[field_name], columns)
            if value:
                config[field_name] = value
    for field_name in list_fields:
        if field_name in hints:
            values = _valid_column_list(hints[field_name], columns)
            if values:
                config[field_name] = values

    safe_scalar_fields = {
        "alpha", "bootstrap_samples", "random_state", "estimator", "rotation", "n_factors",
        "parallel_iterations", "correlation_method", "outcome_family", "centering", "optimizer",
        "gee_correlation", "model_choice", "include_time_effects", "network_input", "network_estimator",
        "edge_threshold", "retain_negative", "directed", "allow_self_loops", "layout",
        "random_graph_iterations", "permutation_samples", "weighting_scheme", "max_iter", "tolerance",
        "covariance_structure",
    }
    for field_name in safe_scalar_fields:
        if field_name in hints and hints[field_name] is not None:
            config[field_name] = hints[field_name]

    if draft.constructs:
        construct_map: dict[str, list[str]] = {}
        measurement_modes: dict[str, str] = {}
        for construct in draft.constructs:
            items = _valid_column_list(construct.indicators, columns)
            if items:
                construct_map[construct.name] = items
                measurement_modes[construct.name] = construct.measurement_mode
        if construct_map:
            config["construct_map"] = construct_map
            config["measurement_modes"] = measurement_modes

    construct_names = set((config.get("construct_map") or {}).keys())
    if draft.structural_relations and construct_names:
        relations: list[dict[str, Any]] = []
        paths: list[tuple[str, str]] = []
        moderations: list[dict[str, str]] = []
        for item in draft.structural_relations:
            if item.predictor not in construct_names or item.outcome not in construct_names:
                continue
            relation_type = _normalise_relation_type(item.relationship_type)
            relation: dict[str, Any] = {
                "type": relation_type,
                "predictor": item.predictor,
                "outcome": item.outcome,
                "expected_sign": item.expected_sign,
            }
            if relation_type == "Mediator" and item.mediator in construct_names:
                relation.update({"mediator": item.mediator, "include_direct": item.include_direct})
                paths.extend([(item.predictor, item.mediator), (item.mediator, item.outcome)])
                if item.include_direct:
                    paths.append((item.predictor, item.outcome))
            elif relation_type == "Moderator" and item.moderator in construct_names:
                relation["moderator"] = item.moderator
                paths.append((item.predictor, item.outcome))
                moderations.append({"predictor": item.predictor, "moderator": item.moderator, "outcome": item.outcome})
            else:
                paths.append((item.predictor, item.outcome))
            relations.append(relation)
        if relations:
            config["structural_relations"] = relations
            config["paths"] = list(dict.fromkeys(paths))
            config["moderations"] = moderations

    blockers = _critical_config_issues(draft.method_key, config)
    blockers.extend(question for question in draft.critical_questions if question)
    assumptions = list(dict.fromkeys(spec.assumptions_for_confirmation + draft.assumptions_for_confirmation))
    confidence = draft.confidence.title()
    return AutoSpecification(
        method_key=draft.method_key,
        method_label=METHOD_LABELS.get(draft.method_key, draft.method_key),
        config=config,
        confidence=confidence,
        completed_fields=_field_rows(config, source="AI provider + deterministic validation"),
        critical_blockers=list(dict.fromkeys(blockers)),
        assumptions_for_confirmation=assumptions,
        rationale=draft.rationale or spec.rationale,
        role_assignments=_roles_from_config(draft.method_key, config),
        framework_narrative=_framework_narrative(draft.method_key, config),
    )


def _critical_config_issues(method_key: str, config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    def require(field_name: str, message: str) -> None:
        value = config.get(field_name)
        if value is None or value == "" or value == [] or value == {}:
            issues.append(message)

    if method_key == "reliability":
        if len(config.get("items") or []) < 2: issues.append("Select at least two scale items.")
    elif method_key == "correlation":
        if len(config.get("variables") or []) < 2: issues.append("Select at least two variables.")
    elif method_key == "efa":
        if len(config.get("items") or []) < 3: issues.append("Select at least three observed items.")
    elif method_key in {"independent_t", "anova"}:
        require("outcome", "Confirm the outcome variable."); require("group", "Confirm the grouping variable.")
    elif method_key == "paired_t":
        require("before", "Confirm the first repeated measure."); require("after", "Confirm the second repeated measure.")
    elif method_key == "chi_square":
        require("row_variable", "Confirm the first categorical variable."); require("column_variable", "Confirm the second categorical variable.")
    elif method_key in {"ols", "logistic"}:
        require("outcome", "Confirm the dependent variable.")
        if not config.get("predictors"): issues.append("Confirm at least one predictor.")
    elif method_key in {"mediation", "moderated_mediation"}:
        require("outcome", "Confirm the outcome."); require("predictor", "Confirm the predictor."); require("mediator", "Confirm the mediator.")
        if method_key == "moderated_mediation": require("moderator", "Confirm the moderator.")
    elif method_key in {"moderation", "advanced_moderation"}:
        require("outcome", "Confirm the outcome."); require("predictor", "Confirm the predictor."); require("moderator", "Confirm the moderator.")
    elif method_key == "parallel_mediation":
        require("outcome", "Confirm the outcome."); require("predictor", "Confirm the predictor.")
        if len(config.get("mediators") or []) < 2: issues.append("Confirm at least two mediators.")
    elif method_key in {"cfa", "sem", "pls_sem"}:
        construct_map = config.get("construct_map") or {}
        minimum = 1 if method_key == "cfa" else 2
        if len(construct_map) < minimum: issues.append(f"Confirm at least {minimum} construct measurement block(s).")
        for name, items in construct_map.items():
            if len(items) < 2: issues.append(f"{name} needs at least two matched indicators.")
        if method_key in {"sem", "pls_sem"} and not config.get("paths"): issues.append("Confirm at least one structural path.")
    elif method_key == "repeated_measures":
        if len(config.get("measurements") or []) < 2: issues.append("Confirm at least two repeated measurements.")
    elif method_key == "multilevel":
        require("outcome", "Confirm the multilevel outcome."); require("cluster", "Confirm the cluster identifier.")
        if not config.get("level1_predictors") and not config.get("level2_predictors"): issues.append("Confirm at least one predictor.")
    elif method_key == "panel":
        require("outcome", "Confirm the panel outcome."); require("entity", "Confirm the entity identifier."); require("time", "Confirm the time identifier.")
        if not config.get("predictors"): issues.append("Confirm at least one predictor.")
    elif method_key == "network":
        if config.get("network_input") == "Edge list":
            require("source", "Confirm the source-node column."); require("target", "Confirm the target-node column.")
        elif len(config.get("variables") or []) < 3:
            issues.append("Confirm at least three variables for a correlation network.")
    return list(dict.fromkeys(issues))


def _deterministic_analysis_program(
    study: dict[str, Any],
    df: pd.DataFrame | None,
    framework: pd.DataFrame | None,
    provider_result: AgentProviderResult | None = None,
) -> AnalysisProgramme:
    objectives = parse_research_items(study.get("objectives") or study.get("objective", ""), "O")
    hypotheses = parse_research_items(study.get("hypotheses") or study.get("hypothesis", ""), "H")
    if not objectives:
        objectives = [("O1", "Describe the study variables and analytical sample.")]
    structured = _structured_framework(study)
    assignments: list[AnalysisAssignment] = []
    critical: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, (objective_id, objective) in enumerate(objectives):
        hypothesis_id, hypothesis = hypotheses[index] if index < len(hypotheses) else ("", "")
        recommendation = recommend_method(
            objective, hypothesis, str(study.get("outcome_type", "continuous")),
            int(study.get("group_count", 0)) or None, bool(study.get("paired", False)),
        )
        method_key = _framework_method_override(objective, hypothesis, recommendation["method_key"], structured)
        local_study = dict(study)
        local_study.update({"objective": objective, "hypothesis": hypothesis})
        spec = build_auto_specification(local_study, df, framework, method_key) if df is not None else None
        blockers = spec.critical_blockers if spec else ["Upload a dataset before the analysis can be completed."]
        status = "Ready for confirmation" if not blockers else "Human input required"
        for blocker in blockers:
            critical.append(f"{objective_id}: {blocker}")
        assignments.append(AnalysisAssignment(
            objective_id=objective_id, objective=objective, hypothesis_id=hypothesis_id, hypothesis=hypothesis,
            method_key=method_key, method_label=METHOD_LABELS.get(method_key, method_key),
            rationale=recommendation["reason"], specification=spec, status=status,
        ))
        rows.append({
            "objective_id": objective_id, "objective": objective,
            "hypothesis_id": hypothesis_id or "Not stated", "hypothesis": hypothesis or "Exploratory or descriptive objective",
            "analysis": METHOD_LABELS.get(method_key, method_key),
            "status": status,
            "critical_human_input": "; ".join(blockers),
        })
    provider_result = provider_result or AgentProviderResult(draft=None, provider="Deterministic local agent")
    return AnalysisProgramme(
        assignments=assignments, mapping_table=pd.DataFrame(rows),
        critical_questions=list(dict.fromkeys(critical)),
        readiness_summary=f"{sum(a.status == 'Ready for confirmation' for a in assignments)} of {len(assignments)} objective-specific analyses are ready for confirmation.",
        provider=provider_result.provider,
        models_attempted=provider_result.models_attempted,
        fallback_used=provider_result.fallback_used,
        provider_warnings=list(dict.fromkeys(provider_result.warnings + ([provider_result.error] if provider_result.error else []))),
    )


def build_analysis_program(
    study: dict[str, Any],
    df: pd.DataFrame | None,
    framework: pd.DataFrame | None = None,
    api_key: str | None = None,
    use_provider: bool = True,
) -> AnalysisProgramme:
    if df is None or df.empty or not use_provider:
        return _deterministic_analysis_program(study, df, framework)

    provider_result = generate_analysis_programme_with_provider(study, df, framework, api_key=api_key)
    if provider_result.draft is None:
        return _deterministic_analysis_program(study, df, framework, provider_result)

    objectives = parse_research_items(study.get("objectives") or study.get("objective", ""), "O")
    hypotheses = parse_research_items(study.get("hypotheses") or study.get("hypothesis", ""), "H")
    objective_lookup = {identifier: value for identifier, value in objectives}
    hypothesis_lookup = {identifier: value for identifier, value in hypotheses}
    assignments: list[AnalysisAssignment] = []
    rows: list[dict[str, Any]] = []
    critical: list[str] = list(provider_result.draft.global_critical_questions)

    seen_objectives: set[str] = set()
    for index, draft in enumerate(provider_result.draft.assignments):
        proposed_id = draft.objective_id
        if proposed_id not in objective_lookup or proposed_id in seen_objectives:
            proposed_id = next((identifier for identifier, _ in objectives if identifier not in seen_objectives), proposed_id or f"O{index + 1}")
        objective_id = proposed_id
        seen_objectives.add(objective_id)
        objective = objective_lookup.get(objective_id, draft.objective)
        objective_index = next((i for i, item in enumerate(objectives) if item[0] == objective_id), index)
        hypothesis_id = draft.hypothesis_id or (hypotheses[objective_index][0] if objective_index < len(hypotheses) else "")
        hypothesis = hypothesis_lookup.get(hypothesis_id, draft.hypothesis or (hypotheses[objective_index][1] if objective_index < len(hypotheses) else ""))
        method_key = draft.method_key if draft.method_key in METHOD_LABELS else recommend_method(objective, hypothesis)["method_key"]
        local_study = dict(study)
        local_study.update({"objective": objective, "hypothesis": hypothesis})
        base_spec = build_auto_specification(local_study, df, framework, method_key)
        if base_spec is None:
            spec = None
            blockers = ["Upload a valid dataset before the analysis can be completed."]
        else:
            draft.method_key = method_key
            spec = _overlay_provider_draft(base_spec, draft, df)
            blockers = spec.critical_blockers
        status = "Ready for confirmation" if not blockers else "Human input required"
        critical.extend(f"{objective_id}: {item}" for item in blockers)
        assignments.append(AnalysisAssignment(
            objective_id=objective_id,
            objective=objective,
            hypothesis_id=hypothesis_id,
            hypothesis=hypothesis,
            method_key=method_key,
            method_label=METHOD_LABELS.get(method_key, method_key),
            rationale=draft.rationale,
            specification=spec,
            status=status,
        ))
        rows.append({
            "objective_id": objective_id,
            "objective": objective,
            "hypothesis_id": hypothesis_id or "Not stated",
            "hypothesis": hypothesis or "Exploratory or descriptive objective",
            "analysis": METHOD_LABELS.get(method_key, method_key),
            "status": status,
            "critical_human_input": "; ".join(blockers),
        })

    missing_objectives = [identifier for identifier, _ in objectives if identifier not in seen_objectives]
    if missing_objectives:
        local_programme = _deterministic_analysis_program(study, df, framework)
        local_lookup = {item.objective_id: item for item in local_programme.assignments}
        for objective_id in missing_objectives:
            item = local_lookup[objective_id]
            assignments.append(item)
            blockers = item.specification.critical_blockers if item.specification else ["A specification could not be generated."]
            rows.append({
                "objective_id": item.objective_id,
                "objective": item.objective,
                "hypothesis_id": item.hypothesis_id or "Not stated",
                "hypothesis": item.hypothesis or "Exploratory or descriptive objective",
                "analysis": item.method_label,
                "status": item.status,
                "critical_human_input": "; ".join(blockers),
            })
            critical.extend(f"{item.objective_id}: {blocker}" for blocker in blockers)
        provider_result.warnings.append("The provider omitted one or more objectives. The deterministic local agent completed the missing assignments.")

    if not assignments:
        provider_result.warnings.append("The reasoning provider returned no assignments, so the local agent was used.")
        return _deterministic_analysis_program(study, df, framework, provider_result)

    order = {identifier: index for index, (identifier, _) in enumerate(objectives)}
    assignments.sort(key=lambda item: order.get(item.objective_id, len(order)))
    rows.sort(key=lambda item: order.get(str(item.get("objective_id")), len(order)))

    return AnalysisProgramme(
        assignments=assignments,
        mapping_table=pd.DataFrame(rows),
        critical_questions=list(dict.fromkeys(critical)),
        readiness_summary=f"{sum(a.status == 'Ready for confirmation' for a in assignments)} of {len(assignments)} objective-specific analyses are ready for confirmation.",
        provider=provider_result.provider,
        models_attempted=provider_result.models_attempted,
        fallback_used=provider_result.fallback_used,
        provider_warnings=provider_result.warnings,
    )


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
    structured_framework = _structured_framework(study)
    key = method_key or _framework_method_override(objective, hypothesis, recommendation["method_key"], structured_framework)
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
        if structured_framework.get("construct_map"):
            construct_map = {
                str(name): [item for item in items if item in columns]
                for name, items in (structured_framework.get("construct_map") or {}).items()
            }
            construct_map = {name: items for name, items in construct_map.items() if items}
            config["construct_map"] = construct_map
            config["measurement_modes"] = {
                name: str((structured_framework.get("measurement_modes") or {}).get(name, "reflective")).lower()
                for name in construct_map
            }
        else:
            constructs = suggest_constructs(df)
            construct_map = {item["name"]: item["items"] for item in constructs}
            config["construct_map"] = construct_map
            config["measurement_modes"] = {item["name"]: "reflective" for item in constructs}
        if not construct_map or len(construct_map) < (2 if key != "cfa" else 1):
            blockers.append("The agent could not identify enough item blocks for the latent-variable model. Confirm construct names and indicators.")
        for construct, items in construct_map.items():
            if len(items) < 2:
                blockers.append(f"{construct} has fewer than two matched indicators and needs human confirmation.")
        if key in {"sem", "pls_sem"}:
            if structured_framework.get("paths"):
                paths = [tuple(path) for path in structured_framework.get("paths") if len(path) == 2 and path[0] in construct_map and path[1] in construct_map]
                config["structural_relations"] = [
                    relation for relation in (structured_framework.get("structural_relations") or [])
                    if relation.get("predictor") in construct_map and relation.get("outcome") in construct_map
                ]
                config["moderations"] = [
                    relation for relation in (structured_framework.get("moderations") or [])
                    if relation.get("predictor") in construct_map and relation.get("moderator") in construct_map and relation.get("outcome") in construct_map
                ]
            else:
                construct_mentions = _mentioned_columns(text, list(construct_map))
                ordered = construct_mentions or list(construct_map)
                paths = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)] if len(ordered) >= 2 else []
                config["structural_relations"] = [{"type": "Direct", "predictor": a, "outcome": b} for a, b in paths]
                config["moderations"] = []
                assumptions.append("Structural paths were provisionally ordered from construct mentions or construct order. Confirm causal direction before analysis.")
            config["paths"] = paths
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
