from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import os
import zipfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from statready.agent import build_guided_review, build_analysis_program, parse_research_items
from statready.figures import render_latent_path_diagram
from statready.analysis_diagrams import proposed_diagram
from statready.framework_vision import analyse_framework_image
from statready.ui_theme import apply_professional_theme, hero, card, status_badge
from statready.path_editor_component import path_editor
from statready.dispatch import run_analysis
from statready.io import list_excel_sheets, load_tabular_file
from statready.literature import references_for_method
from statready.models import AuditEntry
from statready.profiling import dataset_profile, normalise_missing_codes, outlier_summary
from statready.recommender import METHOD_LABELS, recommend_method
from statready.reports import (
    audit_frame,
    build_docx_report,
    build_excel_report,
    build_reproducibility_package,
)
from statready.treatments import drop_duplicate_rows, impute_missing, log1p_transform, winsorise


st.set_page_config(page_title="StatReady AI", page_icon="📊", layout="wide")


METHOD_OPTIONS = {label: key for key, label in METHOD_LABELS.items()}
ROLE_OPTIONS = ["Unassigned", "Outcome", "Predictor", "Mediator", "Moderator", "Control", "Group", "Scale item", "Cluster", "Entity", "Identifier", "Time"]
MEASUREMENT_OPTIONS = ["Unknown", "Binary", "Nominal", "Ordinal", "Continuous", "Count", "Date/time", "Identifier"]


def init_state() -> None:
    defaults = {
        "original_df": None,
        "analysis_df": None,
        "audit_entries": [],
        "analysis_result": None,
        "analysis_plan": pd.DataFrame(),
        "framework": pd.DataFrame(),
        "recommended_method_key": "descriptive",
        "study": {},
        "source_name": "",
        "agent_review": None,
        "experience_mode": "AI Guided",
        "auto_specification": None,
        "diagram_positions": {},
        "analysis_results": {},
        "analysis_program": None,
        "active_result_key": None,
        "framework_image_bytes": None,
        "framework_image_name": "",
        "framework_image_mime": "",
        "framework_vision": None,
        "navigation": "Study design",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_result() -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_plan = pd.DataFrame()


def add_audit(entry: AuditEntry) -> None:
    st.session_state.audit_entries.append(entry)
    reset_analysis_result()


def create_framework(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        dtype = df[column].dtype
        unique = df[column].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(dtype):
            measurement = "Binary" if unique == 2 else "Count" if pd.api.types.is_integer_dtype(dtype) else "Continuous"
        else:
            measurement = "Binary" if unique == 2 else "Nominal"
        rows.append({
            "variable": column,
            "construct_label": column.replace("_", " ").title(),
            "role": "Unassigned",
            "measurement": measurement,
            "expected_relationship": "",
            "coding_notes": "",
        })
    return pd.DataFrame(rows)


def load_uploaded_data(uploaded_file, selected_sheet=None) -> None:
    raw_bytes = uploaded_file.getvalue()
    dataframe = load_tabular_file(raw_bytes, uploaded_file.name, sheet_name=selected_sheet or 0)
    st.session_state.original_df = dataframe.copy()
    st.session_state.analysis_df = dataframe.copy()
    st.session_state.source_name = uploaded_file.name
    st.session_state.audit_entries = []
    st.session_state.framework = create_framework(dataframe)
    st.session_state.agent_review = None
    reset_analysis_result()
    add_audit(AuditEntry(
        action="Imported dataset",
        details=f"Loaded {uploaded_file.name} with {len(dataframe)} rows and {dataframe.shape[1]} columns.",
        justification="The uploaded file is preserved as the original dataset. Later treatments apply only to the analysis copy.",
        before_n=len(dataframe),
        after_n=len(dataframe),
    ))


def framework_defaults() -> dict[str, list[str]]:
    defaults = {role: [] for role in ROLE_OPTIONS}
    frame = st.session_state.framework
    if isinstance(frame, pd.DataFrame) and not frame.empty and {"variable", "role"}.issubset(frame.columns):
        for _, row in frame.iterrows():
            role = str(row.get("role", "Unassigned"))
            variable = str(row.get("variable", ""))
            if role in defaults and variable:
                defaults[role].append(variable)
    return defaults


def analysis_plan_frame(method_label: str, config: dict) -> pd.DataFrame:
    study = st.session_state.study
    role_map = framework_defaults()
    role_text = "; ".join(f"{role}: {', '.join(values)}" for role, values in role_map.items() if values and role != "Unassigned")
    fields = {
        "Objective": study.get("objective", ""),
        "Hypothesis": study.get("hypothesis", ""),
        "Guidance level": study.get("guidance_level", ""),
        "Conceptual framework": study.get("framework_notes", ""),
        "Framework role mapping": role_text,
        "Method": method_label,
        "Significance level": config.get("alpha", 0.05),
        "Outcome": config.get("outcome", config.get("after", "")),
        "Predictors": ", ".join(config.get("predictors", [])),
        "Group variable": config.get("group", ""),
        "Mediator": config.get("mediator", ""),
        "Moderator": config.get("moderator", ""),
        "Controls": ", ".join(config.get("controls", [])),
        "Variables": ", ".join(config.get("variables", config.get("items", []))),
        "Additional descriptive profile variables": ", ".join(config.get("profile_variables", [])),
        "Factor items": ", ".join(config.get("items", [])),
        "Construct measurement model": "; ".join(f"{construct} [{(config.get('measurement_modes') or {}).get(construct, 'reflective')}]: {', '.join(items)}" for construct, items in (config.get("construct_map") or {}).items()),
        "Structural relationships": "; ".join(
            f"{relation.get('type')}: " + " -> ".join(str(relation.get(key)) for key in ['predictor', 'mediator', 'moderator', 'outcome'] if relation.get(key))
            for relation in (config.get("structural_relations") or [])
        ),
        "Structural paths estimated": "; ".join(f"{predictor} -> {outcome}" for predictor, outcome in (config.get("paths") or [])),
        "Path diagram layout": (config.get("diagram_settings") or {}).get("layout", ""),
        "Path diagram display": ", ".join(key.replace("show_", "").replace("_", " ") for key, value in (config.get("diagram_settings") or {}).items() if key.startswith("show_") and value),
        "Estimator": config.get("estimator", ""),
        "Bootstrap resamples": config.get("bootstrap_samples", ""),
        "PLS weighting scheme": config.get("weighting_scheme", ""),
        "Repeated measurements": ", ".join(config.get("measurements", [])),
        "Subject identifier": config.get("subject_id", ""),
        "Cluster variable": config.get("cluster", ""),
        "Multilevel outcome family": config.get("outcome_family", ""),
        "Level-1 predictors": ", ".join(config.get("level1_predictors", [])),
        "Level-2 predictors": ", ".join(config.get("level2_predictors", [])),
        "Multilevel centring": config.get("centering", ""),
        "Random slope": config.get("random_slope", ""),
        "GEE working correlation": config.get("gee_correlation", ""),
        "Entity identifier": config.get("entity", ""),
        "Time identifier": config.get("time", ""),
        "Panel model choice": config.get("model_choice", ""),
        "Analysis dataset rows": len(st.session_state.analysis_df) if st.session_state.analysis_df is not None else 0,
    }
    return pd.DataFrame([{"component": key, "specification": value} for key, value in fields.items() if value not in ("", [], None)])


def variable_selector(label: str, columns: list[str], key: str, allow_none: bool = False, default: str | None = None):
    options = ["<none>"] + columns if allow_none else columns
    if not options:
        st.warning(f"No variables are available for {label.lower()}.")
        return None
    default_value = default if default in options else options[0]
    selected = st.selectbox(label, options, index=options.index(default_value), key=key)
    return None if selected == "<none>" else selected


def parse_construct_map(text: str, columns: list[str]) -> tuple[dict[str, list[str]], str | None]:
    mapping: dict[str, list[str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            return {}, f"Construct specification line {line_number} must use 'Construct: item1, item2'."
        construct, raw_items = line.split(":", 1)
        construct = construct.strip()
        items = [item.strip() for item in raw_items.split(",") if item.strip()]
        if not construct or len(items) < 2:
            return {}, f"Construct specification line {line_number} requires a name and at least two items."
        unknown = [item for item in items if item not in columns]
        if unknown:
            return {}, f"Unknown item(s) on line {line_number}: {', '.join(unknown)}."
        mapping[construct] = items
    return mapping, None


def parse_structural_paths(text: str) -> tuple[list[tuple[str, str]], str | None]:
    paths: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "->" not in line:
            return [], f"Structural path line {line_number} must use 'Predictor -> Outcome'."
        predictor, outcome = [part.strip() for part in line.split("->", 1)]
        if not predictor or not outcome:
            return [], f"Structural path line {line_number} is incomplete."
        paths.append((predictor, outcome))
    return paths, None


def render_construct_builder(method_key: str, numeric_columns: list[str]) -> tuple[dict[str, list[str]], dict[str, str], str | None]:
    """Structured construct builder with typed names and indicator dropdowns."""
    saved = st.session_state.study.get(f"{method_key}_construct_definitions", [])
    default_count = max(1, len(saved) or (3 if method_key in {"sem", "pls_sem"} else 2))
    count = int(st.number_input(
        "Number of constructs", min_value=1, max_value=20, value=default_count, step=1,
        key=f"{method_key}_construct_count",
    ))
    definitions = []
    mapping: dict[str, list[str]] = {}
    modes: dict[str, str] = {}
    errors: list[str] = []
    used_items: dict[str, str] = {}
    st.caption("Enter each construct name, then select its indicators from the uploaded dataset.")
    for index in range(count):
        prior = saved[index] if index < len(saved) else {}
        with st.expander(f"Construct {index + 1}", expanded=True):
            name = st.text_input(
                "Construct name", value=str(prior.get("name", "")),
                placeholder="Example: Digital competence", key=f"{method_key}_construct_name_{index}",
            ).strip()
            mode_options = ["Reflective (Mode A)", "Formative (Mode B)"] if method_key == "pls_sem" else ["Reflective common-factor"]
            prior_mode = str(prior.get("mode", mode_options[0]))
            mode = st.selectbox(
                "Measurement specification", mode_options,
                index=mode_options.index(prior_mode) if prior_mode in mode_options else 0,
                key=f"{method_key}_construct_mode_{index}",
                help="CB-SEM and CFA currently estimate reflective common-factor constructs. PLS-SEM supports reflective Mode A and formative Mode B blocks.",
            )
            items = st.multiselect(
                "Select indicators/items", numeric_columns,
                default=[item for item in prior.get("items", []) if item in numeric_columns],
                key=f"{method_key}_construct_items_{index}",
                help="Items are selected directly from numeric columns in the uploaded dataset.",
            )
        definitions.append({"name": name, "mode": mode, "items": items})
        if not name:
            errors.append(f"Construct {index + 1} requires a name.")
            continue
        if name in mapping:
            errors.append(f"Construct name '{name}' is repeated.")
            continue
        if len(items) < 2:
            errors.append(f"Construct '{name}' requires at least two indicators.")
        for item in items:
            if item in used_items:
                errors.append(f"Indicator '{item}' is assigned to both '{used_items[item]}' and '{name}'.")
            used_items[item] = name
        mapping[name] = items
        modes[name] = "formative" if mode.startswith("Formative") else "reflective"
    st.session_state.study[f"{method_key}_construct_definitions"] = definitions
    return mapping, modes, errors[0] if errors else None


def _construct_select(label: str, constructs: list[str], key: str, default: str | None = None) -> str | None:
    options = ["<select>"] + constructs
    chosen = default if default in constructs else options[0]
    selected = st.selectbox(label, options, index=options.index(chosen), key=key)
    return None if selected == "<select>" else selected


def render_structural_relation_builder(
    method_key: str, constructs: list[str]
) -> tuple[list[dict], list[tuple[str, str]], list[dict], str | None]:
    """Build direct, mediation and moderation relations from entered constructs."""
    if len(constructs) < 2:
        st.info("Enter at least two valid constructs before defining structural relationships.")
        return [], [], [], "At least two constructs are required for structural relationships."
    saved = st.session_state.study.get(f"{method_key}_structural_relations", [])
    default_count = max(1, len(saved) or 1)
    count = int(st.number_input(
        "Number of structural relationships", min_value=1, max_value=30, value=default_count, step=1,
        key=f"{method_key}_relation_count",
    ))
    relations: list[dict] = []
    paths: list[tuple[str, str]] = []
    moderations: list[dict] = []
    errors: list[str] = []
    for index in range(count):
        prior = saved[index] if index < len(saved) else {}
        relation_type = st.selectbox(
            f"Relationship {index + 1} type", ["Direct", "Mediator", "Moderator"],
            index=["Direct", "Mediator", "Moderator"].index(prior.get("type", "Direct")) if prior.get("type", "Direct") in ["Direct", "Mediator", "Moderator"] else 0,
            key=f"{method_key}_relation_type_{index}",
        )
        with st.expander(f"Relationship {index + 1}: {relation_type}", expanded=True):
            predictor = _construct_select("Predictor construct", constructs, f"{method_key}_relation_predictor_{index}", prior.get("predictor"))
            if relation_type == "Direct":
                outcome = _construct_select("Outcome construct", constructs, f"{method_key}_relation_outcome_{index}", prior.get("outcome"))
                relation = {"type": relation_type, "predictor": predictor, "outcome": outcome}
                if predictor and outcome:
                    paths.append((predictor, outcome))
            elif relation_type == "Mediator":
                mediator = _construct_select("Mediator construct", constructs, f"{method_key}_relation_mediator_{index}", prior.get("mediator"))
                outcome = _construct_select("Outcome construct", constructs, f"{method_key}_relation_outcome_{index}", prior.get("outcome"))
                include_direct = st.checkbox(
                    "Also estimate the direct predictor-to-outcome path", value=bool(prior.get("include_direct", True)),
                    key=f"{method_key}_relation_direct_{index}",
                )
                relation = {"type": relation_type, "predictor": predictor, "mediator": mediator, "outcome": outcome, "include_direct": include_direct}
                if predictor and mediator and outcome:
                    paths.extend([(predictor, mediator), (mediator, outcome)])
                    if include_direct:
                        paths.append((predictor, outcome))
            else:
                moderator = _construct_select("Moderator construct", constructs, f"{method_key}_relation_moderator_{index}", prior.get("moderator"))
                outcome = _construct_select("Outcome construct", constructs, f"{method_key}_relation_outcome_{index}", prior.get("outcome"))
                relation = {"type": relation_type, "predictor": predictor, "moderator": moderator, "outcome": outcome}
                if predictor and moderator and outcome:
                    paths.extend([(predictor, outcome), (moderator, outcome)])
                    moderations.append({"predictor": predictor, "moderator": moderator, "outcome": outcome})
            relations.append(relation)
            selected = [value for key, value in relation.items() if key in {"predictor", "mediator", "moderator", "outcome"} and value]
            if len(selected) != len(set(selected)):
                errors.append(f"Relationship {index + 1} must use different constructs for its roles.")
            if not predictor or not relation.get("outcome"):
                errors.append(f"Relationship {index + 1} is incomplete.")
            if relation_type == "Mediator" and not relation.get("mediator"):
                errors.append(f"Relationship {index + 1} requires a mediator construct.")
            if relation_type == "Moderator" and not relation.get("moderator"):
                errors.append(f"Relationship {index + 1} requires a moderator construct.")
    st.session_state.study[f"{method_key}_structural_relations"] = relations
    paths = list(dict.fromkeys(paths))
    moderations = [dict(item) for item in {tuple(sorted(item.items())) for item in moderations}]
    return relations, paths, moderations, errors[0] if errors else None


def render_diagram_settings(method_key: str, constructs: list[str]) -> dict:
    """User-controlled path and measurement diagram settings."""
    saved = st.session_state.study.get(f"{method_key}_diagram_settings", {})
    layout_options = [
        "Left to right",
        "Top to bottom",
        "Bottom to top",
        "Radial",
        "Hierarchical",
        "Measurement first",
        "Structural first",
        "Compact publication",
    ]
    with st.expander("Path diagram settings", expanded=False):
        layout = st.selectbox(
            "Diagram orientation and layout",
            layout_options,
            index=layout_options.index(saved.get("layout", "Left to right")) if saved.get("layout", "Left to right") in layout_options else 0,
            key=f"{method_key}_diagram_layout",
        )
        arrow_style = st.selectbox(
            "Structural arrow style", ["Straight", "Curved"],
            index=1 if saved.get("arrow_style") == "Curved" else 0,
            key=f"{method_key}_diagram_arrows",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            show_indicators = st.checkbox("Show observed indicators", value=bool(saved.get("show_indicators", True)), key=f"{method_key}_show_indicators")
            show_loadings = st.checkbox("Show factor loadings", value=bool(saved.get("show_loadings", True)), key=f"{method_key}_show_loadings")
            show_indicator_names = st.checkbox("Show indicator names", value=bool(saved.get("show_indicator_names", True)), key=f"{method_key}_show_indicator_names")
        with c2:
            show_coefficients = st.checkbox("Show path coefficients", value=bool(saved.get("show_coefficients", True)), key=f"{method_key}_show_coefficients")
            show_p_values = st.checkbox("Show path p-values", value=bool(saved.get("show_p_values", True)), key=f"{method_key}_show_p_values")
            show_fit = st.checkbox("Show model-fit indices", value=bool(saved.get("show_fit", True)), key=f"{method_key}_show_fit")
        with c3:
            significance_colours = st.checkbox("Highlight path significance", value=bool(saved.get("significance_colours", True)), key=f"{method_key}_significance_colours")
            monochrome = st.checkbox("Journal monochrome", value=bool(saved.get("monochrome", False)), key=f"{method_key}_monochrome")
            transparent = st.checkbox("Transparent background", value=bool(saved.get("transparent", False)), key=f"{method_key}_transparent")
        resolution = st.selectbox(
            "Export resolution", ["Standard", "High resolution"],
            index=1 if saved.get("resolution") == "High resolution" else 0,
            key=f"{method_key}_diagram_resolution",
        )
        custom_order_text = st.text_input(
            "Optional construct order",
            value=", ".join(saved.get("construct_order", [])),
            placeholder=", ".join(constructs),
            help="Enter construct names separated by commas. Unlisted constructs are appended automatically.",
            key=f"{method_key}_diagram_order",
        )
    requested_order = [value.strip() for value in custom_order_text.split(",") if value.strip()]
    construct_order = [value for value in requested_order if value in constructs]
    construct_order += [value for value in constructs if value not in construct_order]
    settings = {
        "layout": layout,
        "arrow_style": arrow_style,
        "show_indicators": show_indicators,
        "show_loadings": show_loadings,
        "show_indicator_names": show_indicator_names,
        "show_coefficients": show_coefficients,
        "show_p_values": show_p_values,
        "show_fit": show_fit,
        "significance_colours": significance_colours,
        "monochrome": monochrome,
        "transparent": transparent,
        "resolution": resolution,
        "construct_order": construct_order,
    }
    st.session_state.study[f"{method_key}_diagram_settings"] = settings
    return settings


def apply_auto_spec_to_state(auto_spec) -> None:
    st.session_state.auto_specification = auto_spec
    st.session_state.recommended_method_key = auto_spec.method_key
    st.session_state["analysis_method_label"] = auto_spec.method_label
    frame = st.session_state.framework
    if isinstance(frame, pd.DataFrame) and not frame.empty and auto_spec.role_assignments:
        updated = frame.copy()
        updated["role"] = [auto_spec.role_assignments.get(str(variable), role) for variable, role in zip(updated["variable"], updated["role"])]
        st.session_state.framework = updated
    if auto_spec.framework_narrative and not str(st.session_state.study.get("framework_notes", "")).strip():
        st.session_state.study["framework_notes"] = auto_spec.framework_narrative
    config = auto_spec.config
    if auto_spec.method_key in {"cfa", "sem", "pls_sem"}:
        definitions = []
        for name, items in (config.get("construct_map") or {}).items():
            mode = "Formative (Mode B)" if (config.get("measurement_modes") or {}).get(name) == "formative" else ("Reflective (Mode A)" if auto_spec.method_key == "pls_sem" else "Reflective common-factor")
            definitions.append({"name": name, "mode": mode, "items": items})
        st.session_state.study[f"{auto_spec.method_key}_construct_definitions"] = definitions
        if config.get("structural_relations") is not None:
            st.session_state.study[f"{auto_spec.method_key}_structural_relations"] = config.get("structural_relations") or []


def latent_diagram_payload(result):
    method = getattr(result, "method", "")
    construct_map = result.metadata.get("construct_map") or {}
    paths = [tuple(path) for path in (result.metadata.get("paths") or [])]
    if method == "Covariance-based structural equation model":
        return construct_map, paths, result.tables.get("SEM standardised loadings", pd.DataFrame()), result.tables.get("Structural path estimates", pd.DataFrame()), result.tables.get("SEM fit indices", pd.DataFrame()), "SEM path diagram", "Structural equation model path diagram"
    if method == "Confirmatory factor analysis":
        return construct_map, [], result.tables.get("CFA standardised loadings", pd.DataFrame()), pd.DataFrame(), result.tables.get("CFA fit indices", pd.DataFrame()), "CFA measurement diagram", "Confirmatory factor analysis measurement diagram"
    if method == "Partial least squares structural equation model":
        return construct_map, paths, result.tables.get("PLS outer loadings", pd.DataFrame()), result.tables.get("PLS structural path estimates", pd.DataFrame()), result.tables.get("PLS-SEM model summary", pd.DataFrame()), "PLS-SEM path diagram", "Partial least squares SEM path diagram"
    return None


def update_result_diagram(result, custom_positions: dict[str, dict[str, float]]) -> None:
    payload = latent_diagram_payload(result)
    if not payload:
        return
    construct_map, paths, loading_table, path_table, fit_table, figure_name, title = payload
    settings = dict(result.metadata.get("diagram_settings") or {})
    settings["custom_positions"] = custom_positions
    result.metadata["diagram_settings"] = settings
    result.figures[figure_name] = render_latent_path_diagram(
        construct_map=construct_map, loading_table=loading_table, paths=paths,
        path_table=path_table, fit_table=fit_table, title=title, settings=settings,
        structural_relations=result.metadata.get("structural_relations") or [],
    )


def render_method_configuration(method_key: str, columns: list[str], numeric_columns: list[str]) -> dict:
    config: dict = {"alpha": st.number_input("Significance level", min_value=0.001, max_value=0.20, value=float(st.session_state.study.get("alpha", 0.05)), step=0.01, format="%.3f")}
    roles = framework_defaults()
    default_outcome = next((v for v in roles["Outcome"] if v in columns), None)
    default_predictors = [v for v in roles["Predictor"] + roles["Control"] if v in columns]
    default_group = next((v for v in roles["Group"] if v in columns), None)
    default_mediator = next((v for v in roles["Mediator"] if v in columns), None)
    default_moderator = next((v for v in roles["Moderator"] if v in columns), None)

    if method_key == "descriptive":
        config["variables"] = st.multiselect("Variables", columns, default=columns[: min(8, len(columns))])
    elif method_key == "reliability":
        config["items"] = st.multiselect("Scale items", numeric_columns, default=[v for v in roles["Scale item"] if v in numeric_columns])
    elif method_key == "correlation":
        correlation_defaults = [v for v in roles["Outcome"] + roles["Predictor"] if v in numeric_columns]
        config["variables"] = st.multiselect("Variables", numeric_columns, default=correlation_defaults)
        config["correlation_method"] = st.radio("Correlation type", ["pearson", "spearman"], horizontal=True)
    elif method_key == "independent_t":
        config["outcome"] = variable_selector("Continuous outcome", numeric_columns, "tt_outcome", default=default_outcome)
        config["group"] = variable_selector("Two-category group", columns, "tt_group", default=default_group)
    elif method_key == "paired_t":
        config["before"] = variable_selector("First measurement", numeric_columns, "paired_before")
        config["after"] = variable_selector("Second measurement", numeric_columns, "paired_after")
    elif method_key == "anova":
        config["outcome"] = variable_selector("Continuous outcome", numeric_columns, "anova_outcome", default=default_outcome)
        config["group"] = variable_selector("Group variable", columns, "anova_group", default=default_group)
    elif method_key == "chi_square":
        config["row_variable"] = variable_selector("First categorical variable", columns, "chi_row")
        config["column_variable"] = variable_selector("Second categorical variable", columns, "chi_col")
    elif method_key in {"ols", "logistic"}:
        config["outcome"] = variable_selector("Dependent variable", columns if method_key == "logistic" else numeric_columns, f"{method_key}_outcome", default=default_outcome)
        candidates = [c for c in columns if c != config["outcome"]]
        config["predictors"] = st.multiselect("Predictors and controls", candidates, default=[v for v in default_predictors if v in candidates])
    elif method_key == "moderation":
        config["outcome"] = variable_selector("Outcome", numeric_columns, "mod_outcome", default=default_outcome)
        config["predictor"] = variable_selector("Focal predictor", numeric_columns, "mod_predictor", default=next((v for v in roles["Predictor"] if v in numeric_columns), None))
        config["moderator"] = variable_selector("Moderator", numeric_columns, "mod_moderator", default=default_moderator)
        excluded = {config["outcome"], config["predictor"], config["moderator"]}
        config["controls"] = st.multiselect("Optional controls", [c for c in numeric_columns if c not in excluded])
    elif method_key == "mediation":
        config["outcome"] = variable_selector("Outcome", numeric_columns, "med_outcome", default=default_outcome)
        config["predictor"] = variable_selector("Predictor", numeric_columns, "med_predictor", default=next((v for v in roles["Predictor"] if v in numeric_columns), None))
        config["mediator"] = variable_selector("Mediator", numeric_columns, "med_mediator", default=default_mediator)
        excluded = {config["outcome"], config["predictor"], config["mediator"]}
        config["controls"] = st.multiselect("Optional controls", [c for c in numeric_columns if c not in excluded])
        config["bootstrap_samples"] = st.number_input("Bootstrap resamples", min_value=500, max_value=5000, value=1000, step=500)
        config["random_state"] = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    elif method_key == "efa":
        config["items"] = st.multiselect("Observed items", numeric_columns, default=[v for v in roles["Scale item"] if v in numeric_columns])
        automatic = st.checkbox("Choose factor count using parallel analysis", value=True)
        config["n_factors"] = None if automatic else int(st.number_input("Number of factors", min_value=1, max_value=max(1, len(config["items"]) - 1), value=1, step=1))
        config["rotation"] = st.selectbox("Rotation", ["varimax", "none"])
        config["parallel_iterations"] = int(st.number_input("Parallel-analysis simulations", min_value=50, max_value=500, value=100, step=50))
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="efa_seed"))
    elif method_key in {"cfa", "sem", "pls_sem"}:
        st.markdown("#### Construct measurement specification")
        config["construct_map"], config["measurement_modes"], config["construct_parse_error"] = render_construct_builder(method_key, numeric_columns)
        if method_key in {"sem", "pls_sem"}:
            st.markdown("#### Structural paths")
            config["structural_relations"], config["paths"], config["moderations"], config["path_parse_error"] = render_structural_relation_builder(method_key, list(config["construct_map"]))
            config["unsupported_moderations"] = config["moderations"] if method_key == "sem" else []
        if method_key in {"cfa", "sem"}:
            config["estimator"] = st.selectbox(
                "Covariance estimation method", ["ML", "GLS", "ULS", "DWLS"],
                help="ML is appropriate for approximately continuous, normally distributed indicators. DWLS is often preferred for ordinal or non-normal indicators. GLS and ULS provide alternative covariance fitting objectives.",
                key=f"{method_key}_estimator",
            )
        else:
            config["weighting_scheme"] = st.selectbox(
                "PLS inner weighting scheme", ["Path", "Centroid", "Factorial"],
                help="Path weighting is the default for recursive structural models. Centroid uses correlation signs, while factorial uses the correlations themselves.",
                key="pls_weighting_scheme",
            )
            config["bootstrap_samples"] = int(st.number_input("Bootstrap resamples", min_value=100, max_value=5000, value=500, step=100, key="pls_bootstrap"))
            config["max_iter"] = int(st.number_input("Maximum PLS iterations", min_value=50, max_value=2000, value=300, step=50, key="pls_iterations"))
            config["tolerance"] = float(st.selectbox("Convergence tolerance", [1e-5, 1e-6, 1e-7, 1e-8], index=2, key="pls_tolerance"))
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key=f"{method_key}_seed"))
        config["diagram_settings"] = render_diagram_settings(method_key, list(config.get("construct_map", {})))
    elif method_key == "repeated_measures":
        config["measurements"] = st.multiselect("Repeated measurement columns", numeric_columns)
        config["subject_id"] = variable_selector("Subject identifier (optional)", columns, "rm_subject", allow_none=True, default=next((v for v in roles["Identifier"] if v in columns), None))
    elif method_key == "multilevel":
        config["outcome_family"] = st.selectbox(
            "Outcome family", ["Continuous", "Binary", "Count"],
            help="Continuous outcomes may use ML, REML or robust GEE. Binary and count outcomes use population-average robust GEE in this build.",
            key="ml_outcome_family",
        )
        outcome_options = columns if config["outcome_family"] == "Binary" else numeric_columns
        config["outcome"] = variable_selector(
            f"{config['outcome_family']} outcome", outcome_options, "ml_outcome", default=default_outcome,
        )
        config["cluster"] = variable_selector("Level-2 cluster identifier", columns, "ml_cluster", default=next((v for v in roles["Cluster"] + roles["Group"] + roles["Identifier"] if v in columns), None))
        candidates = [c for c in numeric_columns if c != config["outcome"]]
        config["level1_predictors"] = st.multiselect(
            "Level-1 predictors that vary within clusters", candidates,
            default=[v for v in default_predictors if v in candidates], key="ml_level1",
        )
        remaining = [c for c in candidates if c not in config["level1_predictors"]]
        config["level2_predictors"] = st.multiselect(
            "Level-2 predictors that are constant within clusters", remaining, key="ml_level2",
        )
        config["centering"] = st.selectbox(
            "Predictor centring", ["Group-mean with contextual effect", "Grand-mean", "None"],
            help="Group-mean centring separates within-cluster and between-cluster effects for level-1 predictors.",
        )
        estimator_options = ["REML", "ML", "GEE robust"] if config["outcome_family"] == "Continuous" else ["GEE robust"]
        config["estimator"] = st.selectbox("Estimation method", estimator_options, key="ml_estimator")
        if config["estimator"] in {"REML", "ML"}:
            config["random_slope"] = variable_selector(
                "Optional random-slope level-1 predictor", config["level1_predictors"], "ml_random_slope", allow_none=True,
            )
            config["optimizer"] = st.selectbox("Mixed-model optimiser", ["lbfgs", "powell", "cg"], key="ml_optimizer")
        else:
            config["random_slope"] = None
            config["optimizer"] = "lbfgs"
        config["gee_correlation"] = st.selectbox("GEE working correlation", ["Exchangeable", "Independence", "AR(1)"], key="ml_gee_corr")
    elif method_key == "panel":
        config["outcome"] = variable_selector("Continuous outcome", numeric_columns, "panel_outcome", default=default_outcome)
        config["entity"] = variable_selector("Entity identifier", columns, "panel_entity", default=next((v for v in roles["Entity"] + roles["Identifier"] if v in columns), None))
        config["time"] = variable_selector("Time variable", columns, "panel_time", default=next((v for v in roles["Time"] if v in columns), None))
        excluded = {config["outcome"], config["entity"], config["time"]}
        config["predictors"] = st.multiselect("Time-varying predictors", [c for c in numeric_columns if c not in excluded], default=[v for v in default_predictors if v not in excluded])
        config["model_choice"] = st.selectbox("Panel specification", ["automatic", "pooled", "fixed", "random"])
        config["include_time_effects"] = st.checkbox("Include time fixed effects", value=False)
    elif method_key == "advanced_moderation":
        config["outcome"] = variable_selector("Outcome", numeric_columns, "advmod_outcome", default=default_outcome)
        config["predictor"] = variable_selector("Focal predictor", numeric_columns, "advmod_predictor", default=next((v for v in roles["Predictor"] if v in numeric_columns), None))
        config["moderator"] = variable_selector("Continuous moderator", numeric_columns, "advmod_moderator", default=default_moderator)
        excluded = {config["outcome"], config["predictor"], config["moderator"]}
        config["controls"] = st.multiselect("Optional controls", [c for c in numeric_columns if c not in excluded], key="advmod_controls")
    elif method_key == "parallel_mediation":
        config["outcome"] = variable_selector("Outcome", numeric_columns, "pmed_outcome", default=default_outcome)
        config["predictor"] = variable_selector("Predictor", numeric_columns, "pmed_predictor", default=next((v for v in roles["Predictor"] if v in numeric_columns), None))
        excluded = {config["outcome"], config["predictor"]}
        config["mediators"] = st.multiselect("Parallel mediators", [c for c in numeric_columns if c not in excluded], default=[v for v in roles["Mediator"] if v in numeric_columns])
        excluded.update(config["mediators"])
        config["controls"] = st.multiselect("Optional controls", [c for c in numeric_columns if c not in excluded], key="pmed_controls")
        config["bootstrap_samples"] = int(st.number_input("Bootstrap resamples", min_value=500, max_value=5000, value=1000, step=500, key="pmed_boot"))
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="pmed_seed"))
    elif method_key == "moderated_mediation":
        config["outcome"] = variable_selector("Outcome", numeric_columns, "mm_outcome", default=default_outcome)
        config["predictor"] = variable_selector("Predictor", numeric_columns, "mm_predictor", default=next((v for v in roles["Predictor"] if v in numeric_columns), None))
        config["mediator"] = variable_selector("Mediator", numeric_columns, "mm_mediator", default=default_mediator)
        config["moderator"] = variable_selector("First-stage moderator", numeric_columns, "mm_moderator", default=default_moderator)
        excluded = {config["outcome"], config["predictor"], config["mediator"], config["moderator"]}
        config["controls"] = st.multiselect("Optional controls", [c for c in numeric_columns if c not in excluded], key="mm_controls")
        config["bootstrap_samples"] = int(st.number_input("Bootstrap resamples", min_value=500, max_value=5000, value=1000, step=500, key="mm_boot"))
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="mm_seed"))
    elif method_key == "network":
        st.markdown("#### Network construction")
        config["network_input"] = st.selectbox(
            "Network input structure",
            ["Edge list", "Correlation or partial-correlation network", "Adjacency matrix"],
            help="Use an edge list for relational data, a correlation network for variables measured on cases, or a square adjacency matrix.",
            key="network_input_mode",
        )
        if config["network_input"] == "Edge list":
            config["source"] = variable_selector("Source or origin node", columns, "network_source")
            config["target"] = variable_selector("Target or destination node", columns, "network_target")
            config["weight"] = variable_selector("Optional edge weight", numeric_columns, "network_weight", allow_none=True)
            config["directed"] = st.checkbox("Directed network", value=False, key="network_directed")
            config["allow_self_loops"] = st.checkbox("Retain substantively meaningful self-loops", value=False, key="network_self_loops")
        elif config["network_input"] == "Correlation or partial-correlation network":
            config["variables"] = st.multiselect(
                "Network variables or nodes", numeric_columns,
                default=[v for v in roles["Scale item"] + roles["Outcome"] + roles["Predictor"] if v in numeric_columns][:20],
                key="network_variables",
            )
            config["network_estimator"] = st.selectbox(
                "Association estimator",
                ["Pearson correlation", "Spearman correlation", "Partial correlation (Graphical Lasso)"],
                help="Graphical Lasso estimates a regularised partial-correlation network. Pearson and Spearman networks retain marginal associations.",
                key="network_estimator",
            )
            config["edge_threshold"] = float(st.slider("Minimum absolute edge value", 0.00, 0.80, 0.20, 0.01, key="network_threshold"))
            config["retain_negative"] = st.checkbox("Retain negative edges", value=True, key="network_negative")
            config["bootstrap_samples"] = int(st.number_input("Bootstrap stability resamples", min_value=0, max_value=2000, value=200, step=100, key="network_bootstrap"))
            group_options = ["<none>"] + columns
            selected_group = st.selectbox("Optional two-group network comparison", group_options, key="network_group")
            config["group_variable"] = None if selected_group == "<none>" else selected_group
            config["group_values"] = []
            config["permutation_samples"] = 0
            if config["group_variable"]:
                values = list(st.session_state.analysis_df[config["group_variable"]].dropna().unique()) if st.session_state.analysis_df is not None else []
                config["group_values"] = st.multiselect("Select exactly two groups", values, max_selections=2, key="network_group_values")
                config["permutation_samples"] = int(st.number_input("Network-comparison permutations", min_value=100, max_value=5000, value=500, step=100, key="network_permutations"))
        else:
            config["node_label"] = variable_selector("Optional row/node label column", columns, "network_node_label", allow_none=True)
            config["adjacency_columns"] = st.multiselect("Square adjacency-matrix columns", numeric_columns, key="network_adjacency_columns")
            config["directed"] = st.checkbox("Directed adjacency matrix", value=False, key="network_adjacency_directed")
            config["allow_self_loops"] = st.checkbox("Retain self-loops", value=False, key="network_adjacency_loops")
        st.markdown("#### Network diagnostics and diagrams")
        config["layout"] = st.selectbox("Primary network layout", ["Spring", "Kamada-Kawai", "Circular", "Shell", "Spectral"], key="network_layout")
        config["random_graph_iterations"] = int(st.number_input("Random graphs for small-world sensitivity", min_value=10, max_value=500, value=50, step=10, key="network_random_graphs"))
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="network_seed"))

    if method_key != "descriptive":
        config["profile_variables"] = st.multiselect(
            "Additional demographic or profile variables for descriptive reporting (optional)",
            columns,
            default=[],
            help="These variables are summarised separately using available observations and do not enter the inferential model.",
        )
    return config


def validate_config(method_key: str, config: dict) -> str | None:
    required = {
        "reliability": ["items"],
        "correlation": ["variables"],
        "independent_t": ["outcome", "group"],
        "paired_t": ["before", "after"],
        "anova": ["outcome", "group"],
        "chi_square": ["row_variable", "column_variable"],
        "ols": ["outcome", "predictors"],
        "logistic": ["outcome", "predictors"],
        "moderation": ["outcome", "predictor", "moderator"],
        "mediation": ["outcome", "predictor", "mediator"],
        "efa": ["items"],
        "cfa": ["construct_map"],
        "sem": ["construct_map", "paths"],
        "pls_sem": ["construct_map", "paths"],
        "repeated_measures": ["measurements"],
        "mixed_effects": ["outcome", "predictors", "cluster"],
        "multilevel": ["outcome", "cluster"],
        "panel": ["outcome", "predictors", "entity", "time"],
        "advanced_moderation": ["outcome", "predictor", "moderator"],
        "parallel_mediation": ["outcome", "predictor", "mediators"],
        "moderated_mediation": ["outcome", "predictor", "mediator", "moderator"],
        "network": ["network_input"],
    }.get(method_key, [])
    for field in required:
        value = config.get(field)
        if value is None or value == "" or value == []:
            return f"Complete the required field: {field.replace('_', ' ')}."
    if method_key in {"reliability", "correlation"} and len(config.get(required[0], [])) < 2:
        return "Select at least two variables."
    if method_key == "paired_t" and config.get("before") == config.get("after"):
        return "Select two different measurements."
    if method_key == "chi_square" and config.get("row_variable") == config.get("column_variable"):
        return "Select two different categorical variables."
    if method_key == "efa" and len(config.get("items", [])) < 3:
        return "Select at least three observed items for EFA."
    if method_key in {"cfa", "sem", "pls_sem"} and config.get("construct_parse_error"):
        return config["construct_parse_error"]
    if method_key in {"sem", "pls_sem"} and config.get("path_parse_error"):
        return config["path_parse_error"]
    if method_key == "repeated_measures" and len(config.get("measurements", [])) < 2:
        return "Select at least two repeated measurements."
    if method_key == "parallel_mediation" and len(config.get("mediators", [])) < 2:
        return "Select at least two parallel mediators."
    if method_key == "multilevel" and not (config.get("level1_predictors") or config.get("level2_predictors")):
        return "Select at least one level-1 or level-2 predictor."
    if method_key == "multilevel" and config.get("random_slope") and config.get("random_slope") not in (config.get("level1_predictors") or []):
        return "The random-slope variable must be selected as a level-1 predictor."
    if method_key == "panel" and config.get("entity") == config.get("time"):
        return "Entity and time identifiers must be different variables."
    if method_key == "network":
        mode = config.get("network_input")
        if mode == "Edge list":
            if not config.get("source") or not config.get("target"):
                return "Select source and target node columns."
            if config.get("source") == config.get("target"):
                return "Source and target must use different columns."
        elif mode == "Correlation or partial-correlation network":
            if len(config.get("variables") or []) < 3:
                return "Select at least three variables for a correlation network."
            if config.get("group_variable") and len(config.get("group_values") or []) != 2:
                return "Select exactly two groups for network comparison or remove the group variable."
        elif mode == "Adjacency matrix":
            if len(config.get("adjacency_columns") or []) < 2:
                return "Select at least two adjacency-matrix columns."
            if st.session_state.analysis_df is not None and len(st.session_state.analysis_df) != len(config.get("adjacency_columns") or []):
                return "The adjacency matrix must be square: selected columns must equal the number of rows."
    return None


init_state()
apply_professional_theme()


def _research_items(text: str, prefix: str) -> list[tuple[str, str]]:
    return parse_research_items(text, prefix)


def _step_statuses() -> dict[str, bool]:
    study_ready = bool(str(st.session_state.study.get("objectives") or st.session_state.study.get("objective", "")).strip())
    data_ready = st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty
    framework_ready = bool(st.session_state.study.get("framework_structured") or str(st.session_state.study.get("framework_notes", "")).strip())
    programme_ready = st.session_state.analysis_program is not None
    results_ready = bool(st.session_state.analysis_results or st.session_state.analysis_result is not None)
    return {
        "Study design": study_ready,
        "Data and preparation": data_ready,
        "Conceptual framework": framework_ready,
        "AI analysis plan": programme_ready,
        "Manual analysis": data_ready,
        "Results and exports": results_ready,
    }


def _sidebar() -> str:
    steps = list(_step_statuses())
    statuses = _step_statuses()
    st.sidebar.markdown("## StatReady AI")
    st.sidebar.caption("Defensible analysis, from research question to reproducible report")
    completed = sum(statuses.values())
    st.sidebar.progress(completed / len(statuses), text=f"Workflow readiness: {completed}/{len(statuses)}")
    st.sidebar.markdown("---")
    selected = st.sidebar.radio(
        "Workspace",
        steps,
        index=steps.index(st.session_state.navigation) if st.session_state.navigation in steps else 0,
        label_visibility="collapsed",
    )
    st.session_state.navigation = selected
    st.sidebar.markdown("---")
    st.sidebar.selectbox(
        "Guidance level",
        ["Novice guided", "Assisted", "Expert"],
        index=["Novice guided", "Assisted", "Expert"].index(st.session_state.study.get("guidance_level", "Novice guided"))
        if st.session_state.study.get("guidance_level", "Novice guided") in ["Novice guided", "Assisted", "Expert"] else 0,
        key="guidance_level_control",
        help="Novice guided mode completes defensible fields and explains each decision. Expert mode keeps the review gates but reduces guidance text.",
    )
    st.session_state.study["guidance_level"] = st.session_state.guidance_level_control
    st.sidebar.markdown(
        status_badge("Original data preserved", "green")
        + status_badge("Audit trail active", "blue"),
        unsafe_allow_html=True,
    )
    st.sidebar.caption("StatReady never changes data merely to obtain significance.")
    return selected


def _go(page: str) -> None:
    st.session_state.navigation = page
    st.rerun()


def _assignment_mapping_table(objective_id: str, objective: str, hypothesis_id: str, hypothesis: str, method: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "objective_id": objective_id,
        "objective_addressed": objective,
        "hypothesis_id": hypothesis_id or "Not stated",
        "hypothesis_addressed": hypothesis or "Exploratory or descriptive objective",
        "statistical_analysis": method,
    }])


def _store_result(key: str, result, config: dict, method_key: str, method_label: str, objective_id: str, objective: str, hypothesis_id: str, hypothesis: str) -> None:
    mapping = _assignment_mapping_table(objective_id, objective, hypothesis_id, hypothesis, method_label)
    result.tables = {"Objective and hypothesis addressed": mapping, **result.tables}
    result.metadata.update({
        "objective_id": objective_id,
        "objective": objective,
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "method_key": method_key,
    })
    local_study = {
        **st.session_state.study,
        "objective": objective,
        "hypothesis": hypothesis,
        "method": method_label,
    }
    original_objective = st.session_state.study.get("objective", "")
    original_hypothesis = st.session_state.study.get("hypothesis", "")
    st.session_state.study["objective"] = objective
    st.session_state.study["hypothesis"] = hypothesis
    plan = analysis_plan_frame(method_label, config)
    st.session_state.study["objective"] = original_objective
    st.session_state.study["hypothesis"] = original_hypothesis
    st.session_state.analysis_results[key] = {
        "result": result,
        "config": config,
        "method_key": method_key,
        "method_label": method_label,
        "study": local_study,
        "plan": plan,
        "objective_id": objective_id,
        "objective": objective,
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
    }
    st.session_state.active_result_key = key
    st.session_state.analysis_result = result
    st.session_state.analysis_plan = plan


def _run_one_spec(key: str, objective_id: str, objective: str, hypothesis_id: str, hypothesis: str, method_key: str, method_label: str, config: dict):
    error = validate_config(method_key, config)
    if error:
        raise ValueError(error)
    result = run_analysis(st.session_state.analysis_df, method_key, config)
    _store_result(key, result, config, method_key, method_label, objective_id, objective, hypothesis_id, hypothesis)
    return result


def _batch_export() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, record in st.session_state.analysis_results.items():
            safe = "".join(ch if ch.isalnum() else "_" for ch in key).strip("_") or "analysis"
            result = record["result"]
            docx_bytes = build_docx_report(result, record["study"], record["plan"], st.session_state.audit_entries)
            xlsx_bytes = build_excel_report(st.session_state.original_df, st.session_state.analysis_df, result, record["plan"], st.session_state.audit_entries)
            package = build_reproducibility_package(st.session_state.original_df, st.session_state.analysis_df, result, record["study"], record["plan"], st.session_state.audit_entries)
            archive.writestr(f"{safe}/{safe}_Report.docx", docx_bytes)
            archive.writestr(f"{safe}/{safe}_Results.xlsx", xlsx_bytes)
            archive.writestr(f"{safe}/{safe}_Reproducibility.zip", package)
        if st.session_state.analysis_program is not None:
            archive.writestr("Analysis_Objective_Mapping.csv", st.session_state.analysis_program.mapping_table.to_csv(index=False))
    return output.getvalue()


def page_study() -> None:
    hero("Design the analysis around the study", "Enter each objective and hypothesis on a separate line. The agent will build an objective-specific analysis programme rather than forcing the whole study into one test.")
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        st.markdown("### Research purpose")
        title = st.text_input("Study title", value=st.session_state.study.get("title", ""), placeholder="Enter a clear working title")
        objectives = st.text_area(
            "Study objectives, one per line",
            value=st.session_state.study.get("objectives", st.session_state.study.get("objective", "")),
            height=190,
            placeholder="1. Examine the effect of ...\n2. Test the mediating role of ...",
        )
        hypotheses = st.text_area(
            "Hypotheses, one per line and in matching order",
            value=st.session_state.study.get("hypotheses", st.session_state.study.get("hypothesis", "")),
            height=170,
            placeholder="H1. X has a positive effect on Y.\nH2. M mediates the relationship between X and Y.",
        )
        framework_notes = st.text_area(
            "Conceptual framework description, optional when a diagram will be uploaded",
            value=st.session_state.study.get("framework_notes", ""),
            height=110,
            placeholder="Describe the expected direction, mediator, moderator and controls.",
        )
    with c2:
        st.markdown("### Design details")
        outcome_options = ["continuous", "binary", "categorical", "ordinal", "count"]
        current_outcome = st.session_state.study.get("outcome_type", "continuous")
        outcome_type = st.selectbox("Expected primary outcome type", outcome_options, index=outcome_options.index(current_outcome) if current_outcome in outcome_options else 0)
        study_design = st.selectbox(
            "Study design",
            ["Cross-sectional", "Experimental or quasi-experimental", "Longitudinal or repeated measures", "Panel data", "Multilevel or clustered", "Exploratory"],
            index=0,
        )
        group_count = st.number_input("Number of comparison groups, where relevant", min_value=0, max_value=50, value=int(st.session_state.study.get("group_count", 0)))
        paired = st.checkbox("The same participants or units are measured repeatedly", value=bool(st.session_state.study.get("paired", False)))
        alpha = st.number_input("Significance level", min_value=0.001, max_value=0.20, value=float(st.session_state.study.get("alpha", 0.05)), step=0.01, format="%.3f")
        objective_items = _research_items(objectives, "O")
        hypothesis_items = _research_items(hypotheses, "H")
        st.markdown("### Entry check")
        m1, m2 = st.columns(2)
        m1.metric("Objectives", len(objective_items))
        m2.metric("Hypotheses", len(hypothesis_items))
        if hypothesis_items and len(hypothesis_items) != len(objective_items):
            st.warning("The agent will map hypotheses by order, but unmatched items will be marked for review.")
        else:
            st.success("The objectives and hypotheses are ready for objective-specific mapping.")

    first_objective = objective_items[0][1] if objective_items else ""
    first_hypothesis = hypothesis_items[0][1] if hypothesis_items else ""
    st.session_state.study.update({
        "title": title,
        "objectives": objectives,
        "hypotheses": hypotheses,
        "objective": first_objective,
        "hypothesis": first_hypothesis,
        "framework_notes": framework_notes,
        "outcome_type": outcome_type,
        "study_design": study_design,
        "group_count": int(group_count),
        "paired": paired,
        "alpha": float(alpha),
    })
    st.markdown("---")
    cnext1, cnext2 = st.columns([1, 3])
    if cnext1.button("Continue to data", type="primary", use_container_width=True):
        _go("Data and preparation")
    cnext2.caption("The agent can refine the outcome type and design from the dataset and conceptual framework. Critical causal or measurement decisions still require confirmation.")


def page_data() -> None:
    hero("Prepare and understand the data", "Upload CSV or Excel data. The original file is preserved, while every treatment is documented on a separate analysis copy.")
    with st.container(border=True):
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], help="Maximum recommended upload size on the Render Starter plan is 100 MB.")
        selected_sheet = None
        if uploaded is not None and Path(uploaded.name).suffix.lower() in {".xlsx", ".xls"}:
            try:
                selected_sheet = st.selectbox("Excel sheet", list_excel_sheets(uploaded.getvalue()))
            except Exception as exc:
                st.error(f"Could not read workbook sheets: {exc}")
        if uploaded is not None:
            if getattr(uploaded, "size", 0) > 100 * 1024 * 1024:
                st.error("The file exceeds 100 MB. Use a higher-capacity private deployment or reduce the file.")
            elif st.button("Load and profile dataset", type="primary"):
                try:
                    load_uploaded_data(uploaded, selected_sheet)
                    st.success("Dataset loaded and profiled. The original copy is preserved.")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    if st.session_state.analysis_df is None:
        card("No dataset loaded", "Upload a CSV or Excel file to activate screening, conceptual-framework matching and analysis.", "Required")
        return

    original_df = st.session_state.original_df
    analysis_df = st.session_state.analysis_df
    profile = dataset_profile(analysis_df)
    overview = profile["overview"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{int(overview['rows']):,}")
    m2.metric("Variables", f"{analysis_df.shape[1]:,}")
    m3.metric("Missing cells", f"{int(overview['total_missing_cells']):,}")
    m4.metric("Duplicate rows", f"{int(overview['duplicate_rows']):,}")

    overview_tab, quality_tab, treatment_tab = st.tabs(["Data preview", "Quality review", "Documented preparation"])
    with overview_tab:
        st.caption(f"Source: {st.session_state.source_name} | Original rows: {len(original_df):,} | Current analysis rows: {len(analysis_df):,}")
        st.dataframe(analysis_df.head(100), use_container_width=True)
    with quality_tab:
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("#### Variable profile")
            st.dataframe(profile["variables"], use_container_width=True, hide_index=True)
        with q2:
            st.markdown("#### Missingness")
            st.dataframe(profile["missingness"], use_container_width=True, hide_index=True)
        st.markdown("#### Outlier screening")
        st.dataframe(outlier_summary(analysis_df), use_container_width=True, hide_index=True)
    with treatment_tab:
        st.warning("Treatments are never selected to improve significance. The untreated model and original data remain available.")
        a, b, c = st.columns(3)
        with a:
            st.markdown("#### Basic cleaning")
            if st.button("Normalise missing-value codes", use_container_width=True):
                cleaned, changes = normalise_missing_codes(st.session_state.analysis_df)
                st.session_state.analysis_df = cleaned
                for _, row in changes.iterrows():
                    add_audit(AuditEntry(action=str(row["action"]), variable=str(row["variable"]), details=f"Converted {int(row['values_changed'])} coded value(s) to missing.", justification="The values matched common missing-data codes. The original dataset remains unchanged.", before_n=len(cleaned), after_n=len(cleaned)))
                st.success(f"Recorded {len(changes)} variable-level change(s).")
            if st.button("Drop exact duplicate rows", use_container_width=True):
                cleaned, entry = drop_duplicate_rows(st.session_state.analysis_df)
                st.session_state.analysis_df = cleaned
                add_audit(entry)
                st.success(entry.details)
        with b:
            st.markdown("#### Missing data")
            selected = st.multiselect("Variables to impute", list(analysis_df.columns), key="pro_impute_columns")
            strategy = st.selectbox("Simple strategy", ["median", "mean", "mode"], key="pro_impute_strategy")
            if st.button("Apply documented imputation", use_container_width=True):
                cleaned, entries = impute_missing(st.session_state.analysis_df, selected, strategy)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Applied {len(entries)} documented treatment(s).")
        with c:
            st.markdown("#### Sensitivity copy")
            numeric = list(analysis_df.select_dtypes(include="number").columns)
            selected_w = st.multiselect("Winsorise variables", numeric, key="pro_winsor_columns")
            if st.button("Winsorise at 1% and 99%", use_container_width=True):
                cleaned, entries = winsorise(st.session_state.analysis_df, selected_w, 0.01, 0.99)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Applied {len(entries)} documented sensitivity treatment(s).")
            selected_l = st.multiselect("Create log1p variables", numeric, key="pro_log_columns")
            if st.button("Create transformed variables", use_container_width=True):
                cleaned, entries = log1p_transform(st.session_state.analysis_df, selected_l)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Created {len(entries)} transformed variable(s). Original variables were retained.")
        st.markdown("---")
        if st.button("Reset analysis data to original", type="secondary"):
            st.session_state.analysis_df = st.session_state.original_df.copy()
            st.session_state.audit_entries = [AuditEntry(action="Reset analysis data", details="Restored the analysis copy from the preserved original dataset.", justification="User-requested reset.", before_n=len(analysis_df), after_n=len(original_df))]
            reset_analysis_result()
            st.success("Analysis data restored from the original copy.")

    if st.button("Continue to conceptual framework", type="primary"):
        _go("Conceptual framework")


def _apply_structured_framework_to_variable_roles(mapped: dict) -> None:
    frame = st.session_state.framework.copy()
    if frame.empty:
        return
    item_set = {item for items in (mapped.get("construct_map") or {}).values() for item in items}
    frame.loc[frame["variable"].isin(item_set), "role"] = "Scale item"
    frame.loc[frame["variable"].isin(item_set), "coding_notes"] = "Matched to uploaded conceptual framework diagram"
    st.session_state.framework = frame


def page_framework() -> None:
    hero("Confirm the conceptual framework", "Upload the framework diagram, let the vision agent extract constructs and paths, then confirm the variable matches before any model is estimated.")
    if st.session_state.analysis_df is None:
        card("Dataset required", "Load the dataset first so diagram labels can be matched to actual columns.", "Required")
        return
    columns = list(st.session_state.analysis_df.columns)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("### Diagram upload")
        framework_file = st.file_uploader("Conceptual framework diagram", type=["png", "jpg", "jpeg", "webp"], key="framework_diagram_upload")
        if framework_file is not None:
            st.session_state.framework_image_bytes = framework_file.getvalue()
            st.session_state.framework_image_name = framework_file.name
            st.session_state.framework_image_mime = framework_file.type or "image/png"
        if st.session_state.framework_image_bytes:
            st.image(st.session_state.framework_image_bytes, caption=st.session_state.framework_image_name, use_container_width=True)
            api_configured = bool(os.getenv("OPENAI_API_KEY") or st.session_state.get("temporary_openai_key"))
            st.markdown(status_badge("Vision agent ready" if api_configured else "Vision API key required", "green" if api_configured else "gold"), unsafe_allow_html=True)
            if not os.getenv("OPENAI_API_KEY"):
                st.text_input("Temporary OpenAI API key for this session", type="password", key="temporary_openai_key", help="For production, store OPENAI_API_KEY as a secret environment variable on Render. Do not hard-code it in the repository.")
            st.caption("Only the conceptual-framework image, objectives, hypotheses and dataset column names are sent for interpretation. The dataset values are not sent.")
            if st.button("Interpret diagram with AI", type="primary", use_container_width=True):
                with st.spinner("Reading constructs, indicators and arrow directions..."):
                    vision = analyse_framework_image(
                        st.session_state.framework_image_bytes,
                        st.session_state.framework_image_mime,
                        st.session_state.study.get("objectives", ""),
                        st.session_state.study.get("hypotheses", ""),
                        columns,
                        api_key=st.session_state.get("temporary_openai_key") or None,
                    )
                st.session_state.framework_vision = vision
                if vision.extraction is None:
                    st.error(vision.error or "The framework could not be interpreted.")
                else:
                    st.session_state.study["framework_structured"] = vision.mapped_framework
                    st.session_state.study["framework_notes"] = vision.mapped_framework.get("summary", "")
                    _apply_structured_framework_to_variable_roles(vision.mapped_framework)
                    st.success(f"The diagram was interpreted with {vision.provider}. Review every construct, item match and relationship before confirmation.")
                    if vision.fallback_used:
                        st.info("The low-cost vision model required escalation to the fallback model after confidence or graph validation checks.")
                    for issue in vision.validation_issues:
                        st.warning(issue)
    with right:
        st.markdown("### Extracted framework")
        vision = st.session_state.framework_vision
        mapped = st.session_state.study.get("framework_structured") or {}
        if vision is None and not mapped:
            card("No framework extraction yet", "Upload a clear diagram or use the structured variable editor below.", "Awaiting input")
        else:
            if vision is not None and getattr(vision, "extraction", None) is not None:
                st.caption(f"Provider: {vision.provider} | Models attempted: {', '.join(vision.models_attempted) or 'Not recorded'}")
                st.write(vision.extraction.diagram_summary)
                construct_rows = [c.model_dump() for c in vision.extraction.constructs]
                relationship_rows = [r.model_dump() for r in vision.extraction.relationships]
                st.markdown("#### Constructs")
                st.dataframe(pd.DataFrame(construct_rows), use_container_width=True, hide_index=True)
                st.markdown("#### Relationships")
                st.dataframe(pd.DataFrame(relationship_rows), use_container_width=True, hide_index=True)
                if vision.mapping_table:
                    st.markdown("#### Diagram-to-data matches")
                    mapping_df = pd.DataFrame(vision.mapping_table)
                    edited_mapping = st.data_editor(
                        mapping_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "dataset_variable": st.column_config.SelectboxColumn("Dataset variable", options=[""] + columns),
                            "status": st.column_config.TextColumn("Status", disabled=True),
                            "match_score": st.column_config.NumberColumn("Match score", format="%.3f", disabled=True),
                        },
                        key="framework_mapping_editor",
                    )
                    if st.button("Apply reviewed item matches"):
                        new_map: dict[str, list[str]] = {}
                        for _, row in edited_mapping.iterrows():
                            if row.get("dataset_variable"):
                                new_map.setdefault(str(row["construct"]), []).append(str(row["dataset_variable"]))
                        mapped["construct_map"] = {name: list(dict.fromkeys(items)) for name, items in new_map.items()}
                        st.session_state.study["framework_structured"] = mapped
                        _apply_structured_framework_to_variable_roles(mapped)
                        st.success("Reviewed diagram-to-data matches applied.")
            if mapped.get("construct_map") and mapped.get("paths"):
                preview_config = {
                    "construct_map": mapped.get("construct_map"),
                    "paths": mapped.get("paths"),
                    "structural_relations": mapped.get("structural_relations") or [],
                    "diagram_settings": {"layout": "Left to right", "arrow_style": "Curved", "show_indicators": True, "show_indicator_names": True},
                }
                preview = proposed_diagram("pls_sem" if any(str(v).lower() == "formative" for v in (mapped.get("measurement_modes") or {}).values()) else "sem", preview_config, "Conceptual model before estimation")
                if preview:
                    st.markdown("#### Proposed model, not estimated")
                    st.image(preview, use_container_width=True)

    st.markdown("---")
    st.markdown("### Variable dictionary and role confirmation")
    edited = st.data_editor(
        st.session_state.framework,
        use_container_width=True,
        hide_index=True,
        column_config={
            "variable": st.column_config.TextColumn("Dataset variable", disabled=True),
            "construct_label": st.column_config.TextColumn("Readable label"),
            "role": st.column_config.SelectboxColumn("Analytical role", options=ROLE_OPTIONS),
            "measurement": st.column_config.SelectboxColumn("Measurement level", options=MEASUREMENT_OPTIONS),
            "expected_relationship": st.column_config.TextColumn("Expected relationship"),
            "coding_notes": st.column_config.TextColumn("Coding or scale notes"),
        },
        key="professional_framework_editor",
    )
    if st.button("Save framework and continue", type="primary"):
        st.session_state.framework = edited
        if not st.session_state.study.get("framework_notes"):
            roles = edited[edited["role"] != "Unassigned"]
            st.session_state.study["framework_notes"] = "; ".join(f"{row.variable}: {row.role}" for row in roles.itertuples())
        st.session_state.analysis_program = None
        _go("AI analysis plan")


def page_agent() -> None:
    hero("AI-guided analysis programme", "The agent maps every objective and hypothesis to a separate analysis, completes defensible fields from the study, diagram and data, and flags only decisions that require human judgement.")
    if st.session_state.analysis_df is None:
        card("Dataset required", "Load the dataset before the agent can complete variable and estimator settings.", "Required")
        return

    deepseek_configured = bool(os.getenv("DEEPSEEK_API_KEY") or st.session_state.get("temporary_deepseek_key"))
    left_status, right_status = st.columns([1, 1.4])
    with left_status:
        st.markdown(status_badge("DeepSeek planning agent ready" if deepseek_configured else "Local planning fallback", "green" if deepseek_configured else "gold"), unsafe_allow_html=True)
    with right_status:
        st.caption("Only objectives, hypotheses, reviewed framework, variable names, data types, unique counts and missingness percentages are sent. Raw dataset rows are never sent to the reasoning provider.")
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.text_input("Temporary DeepSeek API key for this session", type="password", key="temporary_deepseek_key", help="For production, store DEEPSEEK_API_KEY as a secret environment variable on Render. Do not commit it to GitHub.")

    if st.button("Build or refresh the complete analysis programme", type="primary"):
        with st.spinner("Mapping objectives, hypotheses and validated framework evidence..."):
            st.session_state.analysis_program = build_analysis_program(
                st.session_state.study,
                st.session_state.analysis_df,
                st.session_state.framework,
                api_key=st.session_state.get("temporary_deepseek_key") or None,
            )
        st.success("Objective-specific analysis programme generated.")
    programme = st.session_state.analysis_program
    if programme is None:
        st.info("Select the button above to generate the full programme.")
        return

    provider_note = f"Planning provider: {programme.provider}"
    if programme.models_attempted:
        provider_note += f" | Models attempted: {', '.join(programme.models_attempted)}"
    if programme.fallback_used:
        provider_note += " | Controlled fallback used"
    st.caption(provider_note)
    for warning in programme.provider_warnings:
        st.info(warning)

    ready = sum(a.status == "Ready for confirmation" for a in programme.assignments)
    m1, m2, m3 = st.columns(3)
    m1.metric("Objective analyses", len(programme.assignments))
    m2.metric("Ready for confirmation", ready)
    m3.metric("Human intervention required", len(programme.assignments) - ready)
    st.markdown("### Objective and hypothesis coverage")
    st.dataframe(programme.mapping_table, use_container_width=True, hide_index=True)

    for assignment in programme.assignments:
        status_style = "green" if assignment.status == "Ready for confirmation" else "gold"
        with st.expander(f"{assignment.objective_id}: {assignment.method_label}", expanded=assignment.status != "Ready for confirmation"):
            st.markdown(status_badge(assignment.status, status_style), unsafe_allow_html=True)
            st.write(f"**Objective:** {assignment.objective}")
            st.write(f"**Hypothesis:** {assignment.hypothesis or 'No confirmatory hypothesis stated'}")
            st.caption(assignment.rationale)
            spec = assignment.specification
            if spec is None:
                st.error("A specification could not be generated.")
                continue
            st.dataframe(spec.completed_fields, use_container_width=True, hide_index=True)
            preview = proposed_diagram(assignment.method_key, spec.config, f"{assignment.objective_id} proposed model before estimation")
            if preview:
                st.markdown("#### Model before estimation")
                st.image(preview, use_container_width=True)
            if spec.assumptions_for_confirmation:
                st.markdown("#### Provisional assumptions")
                for item in spec.assumptions_for_confirmation:
                    st.info(item)
            if spec.critical_blockers:
                st.markdown("#### Critical human decisions")
                for blocker in spec.critical_blockers:
                    st.warning(blocker)
            else:
                if st.button(f"Run {assignment.objective_id} analysis", key=f"run_assignment_{assignment.objective_id}"):
                    try:
                        with st.spinner(f"Running {assignment.objective_id}..."):
                            _run_one_spec(assignment.objective_id, assignment.objective_id, assignment.objective, assignment.hypothesis_id, assignment.hypothesis, assignment.method_key, assignment.method_label, spec.config)
                        st.success(f"{assignment.objective_id} completed and linked to its objective and hypothesis.")
                    except Exception as exc:
                        st.error(f"{assignment.objective_id} could not be completed: {exc}")

    st.markdown("---")
    confirm_all = st.checkbox("I confirm the agent's variable roles, causal directions, measurement blocks and estimator choices for all analyses marked ready.")
    if st.button("Run all ready objective-specific analyses", type="primary", disabled=not confirm_all):
        progress = st.progress(0, text="Starting analysis programme...")
        completed = 0
        failures = []
        runnable = [a for a in programme.assignments if a.specification and not a.specification.critical_blockers]
        for index, assignment in enumerate(runnable, start=1):
            try:
                progress.progress((index - 1) / max(len(runnable), 1), text=f"Running {assignment.objective_id}: {assignment.method_label}")
                _run_one_spec(assignment.objective_id, assignment.objective_id, assignment.objective, assignment.hypothesis_id, assignment.hypothesis, assignment.method_key, assignment.method_label, assignment.specification.config)
                completed += 1
            except Exception as exc:
                failures.append(f"{assignment.objective_id}: {exc}")
        progress.progress(1.0, text="Analysis programme finished")
        if completed:
            st.success(f"Completed {completed} objective-specific analysis or analyses.")
        for failure in failures:
            st.error(failure)
        if completed:
            if st.button("Open results", key="open_batch_results"):
                _go("Results and exports")


def page_manual() -> None:
    hero("Manual analysis workspace", "Advanced users can override the guided plan. A non-estimated diagram and complete specification are shown before the model runs.")
    if st.session_state.analysis_df is None:
        card("Dataset required", "Load the dataset before configuring an analysis.", "Required")
        return
    df = st.session_state.analysis_df
    columns = list(df.columns)
    numeric_columns = list(df.select_dtypes(include="number").columns)
    labels = list(METHOD_OPTIONS.keys())
    default_label = METHOD_LABELS.get(st.session_state.recommended_method_key, METHOD_LABELS["descriptive"])
    selected_label = st.selectbox("Statistical method", labels, index=labels.index(default_label) if default_label in labels else 0, key="professional_analysis_method")
    selected_key = METHOD_OPTIONS[selected_label]
    with st.container(border=True):
        config = render_method_configuration(selected_key, columns, numeric_columns)
    config["alpha"] = float(config.get("alpha", st.session_state.study.get("alpha", 0.05)))
    st.markdown("### Pre-estimation review")
    preview_plan = analysis_plan_frame(selected_label, config)
    st.dataframe(preview_plan, use_container_width=True, hide_index=True)
    preview = proposed_diagram(selected_key, config, "Selected model before estimation")
    if preview:
        st.image(preview, caption="Proposed relationships and selected variables, not yet estimated", use_container_width=True)
    st.warning("Confirm causal direction, variable measurement, unit of analysis and sampling design. The app will not invent design information absent from the study or data.")
    if st.button("Run confirmed analysis", type="primary"):
        objectives = _research_items(st.session_state.study.get("objectives", ""), "O")
        hypotheses = _research_items(st.session_state.study.get("hypotheses", ""), "H")
        objective_id, objective = objectives[0] if objectives else ("Manual", st.session_state.study.get("objective", "Manual analysis"))
        hypothesis_id, hypothesis = hypotheses[0] if hypotheses else ("", st.session_state.study.get("hypothesis", ""))
        try:
            with st.spinner("Running analysis, diagnostics and sensitivity procedures..."):
                _run_one_spec("Manual", objective_id, objective, hypothesis_id, hypothesis, selected_key, selected_label, config)
            st.success("Analysis completed. The proposed and estimated diagrams are available in Results and exports.")
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")


def _render_result_figures(record: dict) -> None:
    result = record["result"]
    latent_payload = latent_diagram_payload(result)
    if latent_payload:
        construct_map, paths, _, _, _, editor_figure_name, _ = latent_payload
        st.markdown("#### Refine the estimated path diagram")
        st.caption("Drag constructs or click a construct and use the movement controls. Saved positions update the PNG and all new exports.")
        saved_positions = (result.metadata.get("diagram_settings") or {}).get("custom_positions") or st.session_state.diagram_positions.get(editor_figure_name, {})
        edited_positions = path_editor(nodes=list(construct_map), edges=paths, positions=saved_positions, height=680, key=f"path_editor_{record['objective_id']}_{editor_figure_name}")
        if edited_positions and edited_positions != saved_positions:
            st.session_state.diagram_positions[editor_figure_name] = edited_positions
            update_result_diagram(result, edited_positions)
            st.success("The estimated diagram and future exports now use the saved arrangement.")
    if result.metadata.get("interactive_network_html"):
        st.markdown("#### Interactive network")
        components.html(result.metadata["interactive_network_html"], height=720, scrolling=False)
    for index, (name, figure) in enumerate(result.figures.items(), start=1):
        st.markdown(f"#### {name}")
        st.image(figure, use_container_width=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        st.download_button(f"Download {name}", figure, f"{safe}.png", "image/png", key=f"professional_fig_{record['objective_id']}_{index}")


def page_results() -> None:
    hero("Results linked to every objective", "Review the proposed model, final estimates, assumptions, sensitivity analyses and reproducibility files for each objective and hypothesis.")
    if not st.session_state.analysis_results:
        if st.session_state.analysis_result is None:
            card("No results yet", "Run the guided programme or a manual analysis.", "Awaiting analysis")
            return
        _store_result("Latest", st.session_state.analysis_result, {}, "", st.session_state.analysis_result.method, "O1", st.session_state.study.get("objective", ""), "H1", st.session_state.study.get("hypothesis", ""))
    keys = list(st.session_state.analysis_results)
    active = st.session_state.active_result_key if st.session_state.active_result_key in keys else keys[0]
    selected = st.selectbox("Select objective-specific result", keys, index=keys.index(active))
    st.session_state.active_result_key = selected
    record = st.session_state.analysis_results[selected]
    result = record["result"]

    st.markdown(
        status_badge(record["objective_id"], "teal")
        + status_badge(record["hypothesis_id"] or "No hypothesis", "blue")
        + status_badge(record["method_label"], "green"),
        unsafe_allow_html=True,
    )
    st.markdown(f"### {record['objective']}")
    if record["hypothesis"]:
        st.write(f"**Hypothesis addressed:** {record['hypothesis']}")
    st.write(result.summary)
    for warning in result.warnings:
        st.warning(warning)

    overview_tab, figure_tab, table_tab, diagnostic_tab, export_tab = st.tabs(["Overview", "Diagrams", "Results tables", "Diagnostics", "Exports"])
    with overview_tab:
        st.markdown("#### Analysis specification")
        st.dataframe(record["plan"], use_container_width=True, hide_index=True)
        if result.metadata.get("diagnostic_response"):
            st.info(result.metadata["diagnostic_response"])
        if result.metadata.get("descriptive_summary"):
            st.markdown("#### Descriptive summary")
            st.write(result.metadata["descriptive_summary"])
    with figure_tab:
        if result.figures or result.metadata.get("interactive_network_html"):
            _render_result_figures(record)
        else:
            st.info("This analysis does not require a path or network diagram.")
    with table_tab:
        for name, table in result.tables.items():
            with st.expander(name, expanded=name in {"Objective and hypothesis addressed", "Selected coefficient table", "Model fit", "Structural path estimates", "PLS structural path estimates", "Test result"}):
                st.dataframe(table, use_container_width=True, hide_index=True)
    with diagnostic_tab:
        st.markdown("#### Assumptions and diagnostics")
        if result.diagnostics.empty:
            st.info("No method-specific diagnostic table was generated.")
        else:
            st.dataframe(result.diagnostics, use_container_width=True, hide_index=True)
            concerns = result.diagnostics[result.diagnostics["status"].isin(["Minor concern", "Material concern"])] if "status" in result.diagnostics else pd.DataFrame()
            if concerns.empty:
                st.success("No diagnostic was classified as a minor or material concern.")
            else:
                st.warning(f"{len(concerns)} diagnostic item(s) require interpretation or a documented sensitivity response.")
        st.markdown("#### Audit trail")
        st.dataframe(audit_frame(st.session_state.audit_entries + result.treatment_log), use_container_width=True, hide_index=True)
        st.markdown("#### Methodological literature")
        st.dataframe(references_for_method(result.method, result.diagnostics), use_container_width=True, hide_index=True)
    with export_tab:
        try:
            docx_bytes = build_docx_report(result, record["study"], record["plan"], st.session_state.audit_entries)
            xlsx_bytes = build_excel_report(st.session_state.original_df, st.session_state.analysis_df, result, record["plan"], st.session_state.audit_entries)
            zip_bytes = build_reproducibility_package(st.session_state.original_df, st.session_state.analysis_df, result, record["study"], record["plan"], st.session_state.audit_entries)
            c1, c2, c3 = st.columns(3)
            c1.download_button("DOCX report", docx_bytes, f"StatReady_{selected}_Report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            c2.download_button("Excel results", xlsx_bytes, f"StatReady_{selected}_Results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            c3.download_button("Reproducibility ZIP", zip_bytes, f"StatReady_{selected}_Reproducibility.zip", "application/zip", use_container_width=True)
            if len(st.session_state.analysis_results) > 1:
                st.download_button("Download complete objective-by-objective analysis package", _batch_export(), "StatReady_Complete_Analysis_Programme.zip", "application/zip", type="primary")
        except Exception as exc:
            st.error(f"Export generation failed: {exc}")
        with st.expander("Generated reproducibility code"):
            st.code(result.reproducible_code, language="python")


page = _sidebar()
if page == "Study design":
    page_study()
elif page == "Data and preparation":
    page_data()
elif page == "Conceptual framework":
    page_framework()
elif page == "AI analysis plan":
    page_agent()
elif page == "Manual analysis":
    page_manual()
else:
    page_results()

st.markdown("---")
st.caption("StatReady AI Phase 2.6 | Cost-aware hybrid AI | Validated framework vision | Objective-specific planning | Deterministic statistical computation | Reproducible reporting")
