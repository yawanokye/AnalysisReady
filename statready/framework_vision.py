from __future__ import annotations

import base64
import difflib
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field


class VisionConstruct(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    role: Literal["predictor", "outcome", "mediator", "moderator", "control", "unspecified"] = "unspecified"
    measurement_mode: Literal["reflective", "formative", "observed", "unclear"] = "unclear"


class VisionRelationship(BaseModel):
    predictor: str
    outcome: str
    relationship_type: Literal["direct", "mediation", "moderation", "covariance"] = "direct"
    mediator: str | None = None
    moderator: str | None = None
    expected_sign: Literal["positive", "negative", "unspecified"] = "unspecified"
    label: str | None = None


class FrameworkVisionExtraction(BaseModel):
    diagram_summary: str
    constructs: list[VisionConstruct]
    relationships: list[VisionRelationship]
    confidence: Literal["high", "medium", "low"]
    ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class FrameworkVisionResult:
    extraction: FrameworkVisionExtraction | None
    mapped_framework: dict[str, Any]
    mapping_table: list[dict[str, Any]]
    provider: str
    error: str | None = None
    models_attempted: list[str] = field(default_factory=list)
    fallback_used: bool = False
    validation_issues: list[str] = field(default_factory=list)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _best_column(label: str, columns: list[str]) -> tuple[str | None, float]:
    target = _normalise(label)
    if not target:
        return None, 0.0
    exact = next((column for column in columns if _normalise(column) == target), None)
    if exact:
        return exact, 1.0
    scored = []
    for column in columns:
        candidate = _normalise(column)
        ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
        if target in candidate or candidate in target:
            ratio = max(ratio, min(len(target), len(candidate)) / max(len(target), len(candidate)))
        scored.append((ratio, column))
    score, column = max(scored, default=(0.0, None))
    return (column if score >= 0.58 else None), float(score)


def validate_framework_graph(extraction: FrameworkVisionExtraction) -> list[str]:
    issues: list[str] = []
    names = [construct.name.strip() for construct in extraction.constructs if construct.name.strip()]
    normalised = [_normalise(name) for name in names]
    if not names:
        issues.append("No constructs were extracted.")
        return issues
    if len(normalised) != len(set(normalised)):
        issues.append("Duplicate construct names were extracted.")
    valid_names = set(names)
    if len(names) > 1 and not extraction.relationships:
        issues.append("Multiple constructs were extracted but no relationships were identified.")
    for relation in extraction.relationships:
        if relation.predictor not in valid_names:
            issues.append(f"Unknown predictor construct: {relation.predictor}.")
        if relation.outcome not in valid_names:
            issues.append(f"Unknown outcome construct: {relation.outcome}.")
        if relation.predictor == relation.outcome:
            issues.append(f"Self-directed relationship found for {relation.predictor}.")
        if relation.relationship_type == "mediation":
            if not relation.mediator:
                issues.append(f"Mediation path {relation.predictor} to {relation.outcome} has no mediator.")
            elif relation.mediator not in valid_names:
                issues.append(f"Unknown mediator construct: {relation.mediator}.")
        if relation.relationship_type == "moderation":
            if not relation.moderator:
                issues.append(f"Moderation path {relation.predictor} to {relation.outcome} has no moderator.")
            elif relation.moderator not in valid_names:
                issues.append(f"Unknown moderator construct: {relation.moderator}.")
    return list(dict.fromkeys(issues))


def reconcile_framework(extraction: FrameworkVisionExtraction, columns: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    construct_map: dict[str, list[str]] = {}
    modes: dict[str, str] = {}
    roles: dict[str, str] = {}
    mapping_rows: list[dict[str, Any]] = []
    names = {construct.name for construct in extraction.constructs}

    for construct in extraction.constructs:
        mapped_items: list[str] = []
        candidates = construct.indicators or construct.aliases or [construct.name]
        for item in candidates:
            mapped, score = _best_column(item, columns)
            mapping_rows.append({
                "construct": construct.name,
                "diagram_label": item,
                "dataset_variable": mapped or "",
                "match_score": round(score, 3),
                "status": "Matched" if mapped else "Human confirmation required",
            })
            if mapped and mapped not in mapped_items:
                mapped_items.append(mapped)
        construct_map[construct.name] = mapped_items
        modes[construct.name] = construct.measurement_mode
        roles[construct.name] = construct.role

    relations: list[dict[str, Any]] = []
    paths: list[tuple[str, str]] = []
    moderations: list[dict[str, str]] = []
    for relationship in extraction.relationships:
        if relationship.predictor not in names or relationship.outcome not in names:
            continue
        relation_type = relationship.relationship_type.title()
        if relation_type == "Mediation":
            relation_type = "Mediator"
        elif relation_type == "Moderation":
            relation_type = "Moderator"
        elif relation_type == "Covariance":
            relation_type = "Covariance"
        else:
            relation_type = "Direct"
        relation = {
            "type": relation_type,
            "predictor": relationship.predictor,
            "outcome": relationship.outcome,
            "expected_sign": relationship.expected_sign,
        }
        if relation_type == "Mediator" and relationship.mediator:
            relation["mediator"] = relationship.mediator
            relation["include_direct"] = True
            paths.extend([
                (relationship.predictor, relationship.mediator),
                (relationship.mediator, relationship.outcome),
                (relationship.predictor, relationship.outcome),
            ])
        elif relation_type == "Moderator" and relationship.moderator:
            relation["moderator"] = relationship.moderator
            paths.extend([(relationship.predictor, relationship.outcome), (relationship.moderator, relationship.outcome)])
            moderations.append({
                "predictor": relationship.predictor,
                "moderator": relationship.moderator,
                "outcome": relationship.outcome,
            })
        else:
            paths.append((relationship.predictor, relationship.outcome))
        relations.append(relation)

    mapped = {
        "summary": extraction.diagram_summary,
        "confidence": extraction.confidence,
        "construct_map": construct_map,
        "measurement_modes": modes,
        "construct_roles": roles,
        "structural_relations": relations,
        "paths": list(dict.fromkeys(paths)),
        "moderations": moderations,
        "ambiguities": extraction.ambiguities,
        "warnings": extraction.warnings,
    }
    return mapped, mapping_rows


def _prepare_image(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    max_dimension = int(os.getenv("OPENAI_VISION_MAX_DIMENSION", "1800"))
    quality = int(os.getenv("OPENAI_VISION_JPEG_QUALITY", "88"))
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
        buffer = io.BytesIO()
        if has_alpha or mime_type.lower() == "image/png":
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if has_alpha else "RGB")
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue(), "image/png"
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def _call_openai_vision(
    key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    detail: str,
) -> FrameworkVisionExtraction:
    from openai import OpenAI

    client = OpenAI(api_key=key)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    response = client.responses.parse(
        model=model_name,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded}",
                    "detail": detail,
                },
            ],
        }],
        text_format=FrameworkVisionExtraction,
    )
    extraction = response.output_parsed
    if extraction is None:
        raise RuntimeError("The vision model returned no structured framework extraction.")
    return extraction


def analyse_framework_image(
    image_bytes: bytes,
    mime_type: str,
    objective_text: str,
    hypothesis_text: str,
    columns: list[str],
    api_key: str | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
) -> FrameworkVisionResult:
    key = api_key or os.getenv("OPENAI_API_KEY")
    primary_model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-nano")
    secondary_model = fallback_model or os.getenv("OPENAI_VISION_FALLBACK_MODEL", "gpt-5.6-luna")
    primary_detail = os.getenv("OPENAI_VISION_DETAIL", "low")
    fallback_detail = os.getenv("OPENAI_VISION_FALLBACK_DETAIL", "high")
    if not key:
        return FrameworkVisionResult(
            extraction=None,
            mapped_framework={},
            mapping_table=[],
            provider="Manual framework editor",
            error="OPENAI_API_KEY is not configured. Add it as a secret environment variable on Render to enable conceptual-framework image interpretation.",
        )

    try:
        prepared_bytes, prepared_mime = _prepare_image(image_bytes, mime_type)
    except Exception as exc:
        return FrameworkVisionResult(
            extraction=None,
            mapped_framework={},
            mapping_table=[],
            provider="Manual framework editor",
            error=f"The uploaded conceptual-framework image could not be prepared: {exc}",
        )
    prompt = f"""
Interpret the uploaded conceptual framework as a research-methods expert.
Return only the structured schema. Read construct names, observed indicators, arrow directions,
mediation, moderation, covariance links and expected signs. Do not invent relationships that are
not visible or directly supported by the study wording. Treat ambiguous arrow direction or unclear
measurement mode as an ambiguity requiring human confirmation.

Objectives:
{objective_text}

Hypotheses:
{hypothesis_text}

Available dataset columns:
{', '.join(columns)}

Use dataset column names only as matching evidence. Keep the diagram's original construct names.
""".strip()

    attempted: list[str] = []
    primary_error: Exception | None = None
    primary_issues: list[str] = []
    try:
        attempted.append(primary_model)
        extraction = _call_openai_vision(key, primary_model, prepared_bytes, prepared_mime, prompt, primary_detail)
        primary_issues = validate_framework_graph(extraction)
        requires_fallback = bool(primary_issues) or extraction.confidence == "low"
        if not requires_fallback or primary_model == secondary_model:
            mapped, rows = reconcile_framework(extraction, columns)
            return FrameworkVisionResult(
                extraction=extraction,
                mapped_framework=mapped,
                mapping_table=rows,
                provider=f"OpenAI {primary_model}",
                models_attempted=attempted,
                validation_issues=primary_issues,
            )
    except Exception as exc:
        primary_error = exc

    try:
        attempted.append(secondary_model)
        retry_prompt = prompt
        if primary_issues:
            retry_prompt += "\n\nThe first extraction had these graph-validation issues. Resolve them only when supported by the visible diagram: " + "; ".join(primary_issues)
        extraction = _call_openai_vision(key, secondary_model, prepared_bytes, prepared_mime, retry_prompt, fallback_detail)
        issues = validate_framework_graph(extraction)
        mapped, rows = reconcile_framework(extraction, columns)
        mapped["warnings"] = list(dict.fromkeys((mapped.get("warnings") or []) + issues))
        return FrameworkVisionResult(
            extraction=extraction,
            mapped_framework=mapped,
            mapping_table=rows,
            provider=f"OpenAI {secondary_model}",
            models_attempted=attempted,
            fallback_used=True,
            validation_issues=issues,
            error=None if not issues else "The fallback extraction completed but some relationships still require human confirmation.",
        )
    except Exception as exc:
        detail = f"Primary error: {primary_error}; fallback error: {exc}" if primary_error else str(exc)
        return FrameworkVisionResult(
            extraction=None,
            mapped_framework={},
            mapping_table=[],
            provider="Manual framework editor",
            models_attempted=attempted,
            fallback_used=True,
            validation_issues=primary_issues,
            error=f"Conceptual-framework interpretation failed after the controlled fallback. {detail}",
        )
