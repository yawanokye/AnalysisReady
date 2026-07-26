from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .profiling import dataset_profile
from .recommender import METHOD_LABELS, recommend_method


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


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


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
    """Conservative, reviewable role suggestions based on names, types and uniqueness."""
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
            rationale = "The name may denote a mediator, but the conceptual framework must confirm it."
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
    """Suggest item blocks from repeated prefixes such as trust1/trust2 or trust_1/trust_2."""
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
    add("Hypothesis", bool(str(study.get("hypothesis", "")).strip()), "A hypothesis is optional for descriptive work but should be stated for confirmatory analysis.")
    add("Dataset", df is not None and not df.empty, "Upload CSV or Excel data and confirm the correct sheet.")
    add("Outcome type", bool(study.get("outcome_type")), "Confirm whether the outcome is continuous, binary, categorical, ordinal or count.")
    assigned = False
    if isinstance(framework, pd.DataFrame) and not framework.empty and "role" in framework:
        assigned = bool((framework["role"].astype(str) != "Unassigned").any())
    add("Variable roles", assigned, "Confirm outcome, predictors, mediators, moderators, controls, groups, clusters and time identifiers.")
    add("Conceptual framework", bool(str(study.get("framework_notes", "")).strip()), "Describe the expected directions and any mediation or moderation structure.")
    return pd.DataFrame(checks)


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
    next_actions = []
    for _, row in readiness[readiness["status"] != "Ready"].iterrows():
        next_actions.append(f"Complete {str(row['component']).lower()}: {row['guidance']}")
    if not next_actions:
        next_actions.append("Review the recommended method and confirm the analysis configuration before running it.")
    return GuidedReview(
        method_key=recommendation["method_key"],
        method_label=METHOD_LABELS[recommendation["method_key"]],
        reason=recommendation["reason"],
        readiness=readiness,
        role_suggestions=role_suggestions,
        construct_suggestions=constructs,
        data_findings=findings,
        next_actions=next_actions,
    )
