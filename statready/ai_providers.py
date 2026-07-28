from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from .recommender import METHOD_LABELS


class AgentConstructDraft(BaseModel):
    name: str
    indicators: list[str] = Field(default_factory=list)
    measurement_mode: Literal["reflective", "formative", "observed", "unclear"] = "unclear"
    role: Literal["predictor", "outcome", "mediator", "moderator", "control", "unspecified"] = "unspecified"


class AgentRelationshipDraft(BaseModel):
    relationship_type: Literal["direct", "mediation", "moderation", "covariance"] = "direct"
    predictor: str
    outcome: str
    mediator: str | None = None
    moderator: str | None = None
    include_direct: bool = True
    expected_sign: Literal["positive", "negative", "unspecified"] = "unspecified"


class AgentAssignmentDraft(BaseModel):
    objective_id: str
    objective: str
    hypothesis_id: str = ""
    hypothesis: str = ""
    method_key: str
    rationale: str
    confidence: Literal["high", "medium", "low"] = "medium"
    roles: dict[str, Any] = Field(default_factory=dict)
    config_hints: dict[str, Any] = Field(default_factory=dict)
    constructs: list[AgentConstructDraft] = Field(default_factory=list)
    structural_relations: list[AgentRelationshipDraft] = Field(default_factory=list)
    assumptions_for_confirmation: list[str] = Field(default_factory=list)
    critical_questions: list[str] = Field(default_factory=list)
    escalation_required: bool = False


class AgentProgrammeDraft(BaseModel):
    assignments: list[AgentAssignmentDraft]
    programme_summary: str
    global_critical_questions: list[str] = Field(default_factory=list)
    escalation_recommended: bool = False


@dataclass
class AgentProviderResult:
    draft: AgentProgrammeDraft | None
    provider: str
    models_attempted: list[str] = field(default_factory=list)
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def _measurement_type(series: pd.Series) -> str:
    unique = int(series.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date_time"
    if pd.api.types.is_numeric_dtype(series):
        if unique == 2:
            return "binary"
        if pd.api.types.is_integer_dtype(series) and unique <= max(20, int(len(series) * 0.10)):
            return "ordinal" if unique <= 7 else "count"
        return "continuous"
    return "binary" if unique == 2 else "nominal"


def dataset_schema_for_agent(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return aggregate schema metadata only, never raw rows."""
    rows: list[dict[str, Any]] = []
    n = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        rows.append({
            "name": str(column),
            "storage_type": str(series.dtype),
            "measurement_type": _measurement_type(series),
            "unique_count": int(series.nunique(dropna=True)),
            "missing_percent": round(float(series.isna().sum() / n * 100), 2),
        })
    return rows


def _programme_prompt(
    study: dict[str, Any],
    schema: list[dict[str, Any]],
    framework: pd.DataFrame | None,
) -> str:
    confirmed_roles: list[dict[str, str]] = []
    if isinstance(framework, pd.DataFrame) and not framework.empty and {"variable", "role"}.issubset(framework.columns):
        for _, row in framework.iterrows():
            role = str(row.get("role", "Unassigned"))
            if role != "Unassigned":
                confirmed_roles.append({"variable": str(row.get("variable", "")), "role": role})

    structured = study.get("framework_structured") or {}
    allowed_methods = sorted(METHOD_LABELS)
    output_schema = AgentProgrammeDraft.model_json_schema()
    return f"""
You are the planning layer for StatReady AI. Build one defensible statistical analysis assignment for every stated objective and map the matching hypothesis. Return JSON only and conform to the supplied JSON schema.

Important controls:
- Use only dataset columns listed in DATASET_SCHEMA. Never invent a variable.
- Do not request or infer raw observations. DATASET_SCHEMA contains aggregate metadata only.
- Respect confirmed variable roles and the reviewed conceptual framework.
- Use one of ALLOWED_METHOD_KEYS exactly.
- Complete every non-critical field that can be inferred from the objective, hypothesis, framework and schema.
- Put only genuinely critical unresolved design decisions in critical_questions.
- Never delete observations, transform variables, change causal direction, or select a model because it may produce significance.
- Mark escalation_required true when the objective contains complex latent interactions, mixed measurement modes, multilevel longitudinal structure, competing plausible outcomes, or contradictory framework evidence.
- Keep rationale concise and decision-focused. Do not reveal chain-of-thought.
- For config_hints, use StatReady field names such as outcome, predictors, predictor, mediator, mediators, moderator, controls, group, cluster, entity, time, subject_id, measurements, items, variables, construct_map, measurement_modes, structural_relations, estimator, outcome_family, covariance_structure, alpha, bootstrap_samples and profile_variables.

STUDY:
{json.dumps({
    "title": study.get("title", ""),
    "objectives": study.get("objectives") or study.get("objective", ""),
    "hypotheses": study.get("hypotheses") or study.get("hypothesis", ""),
    "research_design": study.get("research_design", ""),
    "outcome_type": study.get("outcome_type", ""),
    "paired": study.get("paired", False),
    "group_count": study.get("group_count", 0),
    "alpha": study.get("alpha", 0.05),
    "framework_notes": study.get("framework_notes", ""),
}, ensure_ascii=False)}

REVIEWED_FRAMEWORK:
{json.dumps(structured, ensure_ascii=False, default=str)}

CONFIRMED_VARIABLE_ROLES:
{json.dumps(confirmed_roles, ensure_ascii=False)}

DATASET_SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

ALLOWED_METHOD_KEYS:
{json.dumps(allowed_methods)}

OUTPUT_JSON_SCHEMA:
{json.dumps(output_schema, ensure_ascii=False)}
""".strip()


def _parse_json_content(content: str | None) -> AgentProgrammeDraft:
    if not content or not content.strip():
        raise ValueError("The reasoning provider returned an empty response.")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    payload = json.loads(text)
    return AgentProgrammeDraft.model_validate(payload)


def _call_deepseek(
    api_key: str,
    model: str,
    prompt: str,
    base_url: str,
) -> AgentProgrammeDraft:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a statistical-analysis planning service. Return a single valid JSON object only. Do not include markdown or hidden reasoning.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "12000")),
    )
    message = response.choices[0].message if response.choices else None
    return _parse_json_content(getattr(message, "content", None))


def _draft_needs_escalation(draft: AgentProgrammeDraft) -> bool:
    if draft.escalation_recommended:
        return True
    if any(item.escalation_required or item.confidence == "low" for item in draft.assignments):
        return True
    unresolved = len(draft.global_critical_questions) + sum(len(item.critical_questions) for item in draft.assignments)
    return unresolved > max(2, len(draft.assignments))


def generate_analysis_programme_with_provider(
    study: dict[str, Any],
    df: pd.DataFrame,
    framework: pd.DataFrame | None = None,
    api_key: str | None = None,
    primary_model: str | None = None,
    fallback_model: str | None = None,
) -> AgentProviderResult:
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    primary = primary_model or os.getenv("AGENT_REASONING_MODEL", "deepseek-v4-flash")
    fallback = fallback_model or os.getenv("AGENT_REASONING_FALLBACK_MODEL", "deepseek-v4-pro")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not key:
        return AgentProviderResult(
            draft=None,
            provider="Deterministic local agent",
            warnings=["DEEPSEEK_API_KEY is not configured. The reproducible local planning engine was used."],
        )

    schema = dataset_schema_for_agent(df)
    prompt = _programme_prompt(study, schema, framework)
    attempted: list[str] = []
    warnings: list[str] = []
    primary_error: Exception | None = None

    try:
        attempted.append(primary)
        draft = _call_deepseek(key, primary, prompt, base_url)
        if not _draft_needs_escalation(draft) or primary == fallback:
            return AgentProviderResult(
                draft=draft,
                provider=f"DeepSeek {primary}",
                models_attempted=attempted,
                fallback_used=False,
                warnings=warnings,
            )
        warnings.append("The primary planning model identified complexity or unresolved ambiguity, so the request was escalated.")
    except Exception as exc:
        primary_error = exc
        warnings.append(f"Primary planning model failed validation: {exc}")

    try:
        attempted.append(fallback)
        retry_prompt = prompt + "\n\nThe previous attempt was incomplete or failed schema validation. Re-check every objective and return one complete JSON object matching the schema."
        draft = _call_deepseek(key, fallback, retry_prompt, base_url)
        return AgentProviderResult(
            draft=draft,
            provider=f"DeepSeek {fallback}",
            models_attempted=attempted,
            fallback_used=True,
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"Fallback planning model failed validation: {exc}")
        detail = f"Primary: {primary_error}; fallback: {exc}" if primary_error else str(exc)
        return AgentProviderResult(
            draft=None,
            provider="Deterministic local agent",
            models_attempted=attempted,
            fallback_used=True,
            warnings=warnings,
            error=f"Provider-backed planning was unavailable. {detail}",
        )
