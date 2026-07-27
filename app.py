from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from statready.agent import build_guided_review
from statready.figures import render_latent_path_diagram
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

st.title("StatReady AI")
st.caption("Phase 2.4: interactive path editing, autonomous AI specification, comprehensive network analysis, advanced models and reproducible reporting")
st.info("The app never changes data merely to obtain significance. It preserves the original dataset, records every treatment, uses robust or alternative methods where justified, and keeps sensitivity results visible.")
st.session_state.experience_mode = st.segmented_control(
    "Working mode", ["AI Guided", "Standard"],
    default=st.session_state.get("experience_mode", "AI Guided"),
    help="AI Guided Mode provides reviewable recommendations and readiness checks. It does not make irreversible decisions or alter data without confirmation.",
) or "AI Guided"

study_tab, data_tab, framework_tab, agent_tab, analysis_tab, results_tab = st.tabs([
    "1. Study design", "2. Data preparation", "3. Conceptual framework", "4. AI research agent", "5. Run analysis", "6. Results and exports"
])

with study_tab:
    st.subheader("Research specification")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Study title", value=st.session_state.study.get("title", ""))
        objective = st.text_area("Objective", value=st.session_state.study.get("objective", ""), height=110)
        hypothesis = st.text_area("Hypothesis", value=st.session_state.study.get("hypothesis", ""), height=90)
    with col2:
        outcome_type = st.selectbox("Expected outcome type", ["continuous", "binary", "categorical", "ordinal", "count"])
        group_count = st.number_input("Number of comparison groups, where relevant", min_value=0, max_value=50, value=0)
        paired = st.checkbox("The same participants or units are measured twice")
        alpha = st.number_input("Default significance level", min_value=0.001, max_value=0.20, value=float(st.session_state.study.get("alpha", 0.05)), step=0.01, format="%.3f")

    st.session_state.study = {
        **st.session_state.study,
        "title": title,
        "objective": objective,
        "hypothesis": hypothesis,
        "outcome_type": outcome_type,
        "group_count": int(group_count),
        "paired": paired,
        "alpha": alpha,
    }

    if st.button("Recommend statistical method", type="primary"):
        recommendation = recommend_method(objective, hypothesis, outcome_type, int(group_count) or None, paired)
        st.session_state.recommended_method_key = recommendation["method_key"]
        st.session_state.study["recommendation_reason"] = recommendation["reason"]
        st.session_state["analysis_method_label"] = METHOD_LABELS[recommendation["method_key"]]

    recommended_key = st.session_state.recommended_method_key
    st.success(f"Recommended starting method: **{METHOD_LABELS[recommended_key]}**")
    if st.session_state.study.get("recommendation_reason"):
        st.write(st.session_state.study["recommendation_reason"])
    st.caption("The recommendation is rule-based and must be confirmed against variable measurement, sampling design and the conceptual framework.")

with data_tab:
    st.subheader("Upload and screen the dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    upload_allowed = True
    if uploaded is not None and getattr(uploaded, "size", 0) > 100 * 1024 * 1024:
        st.error("The file exceeds the 100 MB deployment limit. Reduce the file or use a higher-capacity private deployment.")
        upload_allowed = False
    selected_sheet = None
    if uploaded is not None and upload_allowed and Path(uploaded.name).suffix.lower() in {".xlsx", ".xls"}:
        try:
            sheets = list_excel_sheets(uploaded.getvalue())
            selected_sheet = st.selectbox("Excel sheet", sheets)
        except Exception as exc:
            st.error(f"Could not read workbook sheets: {exc}")

    if uploaded is not None and upload_allowed and st.button("Load dataset", type="primary"):
        try:
            load_uploaded_data(uploaded, selected_sheet)
            st.success("Dataset loaded. The original copy is preserved.")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.session_state.analysis_df is not None:
        original_df = st.session_state.original_df
        analysis_df = st.session_state.analysis_df
        st.write(f"**Source:** {st.session_state.source_name} | **Original rows:** {len(original_df):,} | **Analysis rows:** {len(analysis_df):,}")
        st.dataframe(analysis_df.head(100), use_container_width=True)

        profile = dataset_profile(analysis_df)
        c1, c2, c3 = st.columns(3)
        overview = profile["overview"].iloc[0]
        c1.metric("Rows", f"{int(overview['rows']):,}")
        c2.metric("Missing cells", f"{int(overview['total_missing_cells']):,}")
        c3.metric("Duplicate rows", f"{int(overview['duplicate_rows']):,}")
        with st.expander("Variable profile", expanded=False):
            st.dataframe(profile["variables"], use_container_width=True)
        with st.expander("Missingness", expanded=False):
            st.dataframe(profile["missingness"], use_container_width=True)
        with st.expander("Outlier screening", expanded=False):
            st.dataframe(outlier_summary(analysis_df), use_container_width=True)

        st.markdown("#### Documented data preparation")
        t1, t2, t3, t4, t5 = st.tabs(["Missing codes", "Duplicates", "Imputation", "Winsorisation", "Transformation"])
        with t1:
            st.write("Convert common text codes such as NA, missing and null to true missing values in the analysis copy.")
            if st.button("Normalise missing-value codes"):
                cleaned, changes = normalise_missing_codes(st.session_state.analysis_df)
                st.session_state.analysis_df = cleaned
                for _, row in changes.iterrows():
                    add_audit(AuditEntry(
                        action=str(row["action"]), variable=str(row["variable"]),
                        details=f"Converted {int(row['values_changed'])} coded value(s) to missing.",
                        justification="The values matched common missing-data codes. The original dataset remains unchanged.",
                        before_n=len(cleaned), after_n=len(cleaned),
                    ))
                st.success(f"Recorded {len(changes)} variable-level change(s).")
        with t2:
            if st.button("Drop exact duplicate rows"):
                cleaned, entry = drop_duplicate_rows(st.session_state.analysis_df)
                st.session_state.analysis_df = cleaned
                add_audit(entry)
                st.success(entry.details)
        with t3:
            columns = list(st.session_state.analysis_df.columns)
            selected = st.multiselect("Variables to impute", columns, key="impute_columns")
            strategy = st.selectbox("Strategy", ["median", "mean", "mode"])
            st.warning("Simple imputation can understate uncertainty. Compare results with complete-case analysis and use multiple imputation in later phases where appropriate.")
            if st.button("Apply documented imputation"):
                cleaned, entries = impute_missing(st.session_state.analysis_df, selected, strategy)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Applied {len(entries)} documented variable treatment(s).")
        with t4:
            numeric = list(st.session_state.analysis_df.select_dtypes(include="number").columns)
            selected = st.multiselect("Numeric variables", numeric, key="winsor_columns")
            limits = st.selectbox("Limits", ["1% and 99%", "5% and 95%"])
            st.warning("Winsorisation is offered only as a defensible sensitivity treatment. The untreated model must remain available.")
            if st.button("Create winsorised analysis copy"):
                lower, upper = (0.01, 0.99) if limits.startswith("1") else (0.05, 0.95)
                cleaned, entries = winsorise(st.session_state.analysis_df, selected, lower, upper)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Applied {len(entries)} documented treatment(s).")
        with t5:
            numeric = list(st.session_state.analysis_df.select_dtypes(include="number").columns)
            selected = st.multiselect("Non-negative variables", numeric, key="log_columns")
            if st.button("Create log1p variables"):
                cleaned, entries = log1p_transform(st.session_state.analysis_df, selected)
                st.session_state.analysis_df = cleaned
                for entry in entries:
                    add_audit(entry)
                st.success(f"Created {len(entries)} transformed variable(s). Original variables were retained.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset analysis data to original"):
                st.session_state.analysis_df = st.session_state.original_df.copy()
                st.session_state.audit_entries = [AuditEntry(
                    action="Reset analysis data",
                    details="Restored the analysis copy from the preserved original dataset.",
                    justification="User-requested reset. All previous result objects were cleared.",
                    before_n=len(original_df), after_n=len(original_df),
                )]
                reset_analysis_result()
                st.success("Analysis copy reset.")
        with c2:
            st.download_button("Download current analysis data", data=analysis_df.to_csv(index=False), file_name="StatReady_Analysis_Data.csv", mime="text/csv")

with framework_tab:
    st.subheader("Conceptual framework and variable roles")
    if st.session_state.analysis_df is None:
        st.info("Load a dataset first.")
    else:
        st.write("Confirm how each dataset variable functions in the study. These roles guide method selection and reporting.")
        if st.session_state.framework.empty or set(st.session_state.framework["variable"]) != set(st.session_state.analysis_df.columns):
            st.session_state.framework = create_framework(st.session_state.analysis_df)
        edited = st.data_editor(
            st.session_state.framework,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "role": st.column_config.SelectboxColumn("Role", options=ROLE_OPTIONS, required=True),
                "measurement": st.column_config.SelectboxColumn("Measurement", options=MEASUREMENT_OPTIONS, required=True),
            },
            disabled=["variable"],
            key="framework_editor",
        )
        st.session_state.framework = edited
        framework_notes = st.text_area(
            "Framework narrative",
            value=st.session_state.study.get("framework_notes", ""),
            placeholder="Example: Digital competence predicts online teaching effectiveness. Motivation mediates the relationship, while institutional support moderates it.",
        )
        st.session_state.study["framework_notes"] = framework_notes
        st.caption("Image extraction can be added later. Phase 2 uses confirmed structured roles and explicit construct specifications to avoid incorrect automatic interpretation.")

with agent_tab:
    st.subheader("AI Guided Mode for research and statistical decisions")
    st.write("The guided agent uses the objectives, hypotheses, framework wording and dataset structure to complete the required analysis specification. It pauses only when a critical variable or design decision cannot be inferred safely. Data deletion and transformations always require human approval.")
    guidance_level = st.selectbox(
        "Guidance level", ["Novice", "Assisted", "Expert co-pilot"],
        index=["Novice", "Assisted", "Expert co-pilot"].index(st.session_state.study.get("guidance_level", "Novice")),
    )
    st.session_state.study["guidance_level"] = guidance_level

    if st.button("Review my study and dataset", type="primary", key="agent_review_button"):
        st.session_state.agent_review = build_guided_review(
            st.session_state.study, st.session_state.analysis_df, st.session_state.framework
        )

    review = st.session_state.get("agent_review")
    if review is None:
        st.info("Complete as much of the study design as possible, upload the dataset, then run the guided review.")
    else:
        st.markdown("### Recommended starting method")
        st.success(f"**{review.method_label}**")
        st.write(review.reason)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use this method in analysis setup", key="agent_use_method"):
                st.session_state.recommended_method_key = review.method_key
                st.session_state.study["recommendation_reason"] = review.reason
                st.session_state["analysis_method_label"] = review.method_label
                st.success("The method has been selected as the starting method. Review its configuration before running it.")
        with c2:
            if st.button("Refresh review", key="agent_refresh"):
                st.session_state.agent_review = build_guided_review(
                    st.session_state.study, st.session_state.analysis_df, st.session_state.framework
                )
                st.rerun()

        st.markdown("### Analysis readiness")
        st.dataframe(review.readiness, use_container_width=True, hide_index=True)
        outstanding = review.readiness[review.readiness["status"] != "Ready"]
        if outstanding.empty:
            st.success("The core study and data inputs are ready for method-specific configuration.")
        else:
            st.warning(f"{len(outstanding)} readiness item(s) still need attention.")

        st.markdown("### Dataset findings")
        st.dataframe(review.data_findings, use_container_width=True, hide_index=True)

        if not review.role_suggestions.empty:
            st.markdown("### Suggested variable roles")
            st.caption("These are conservative suggestions, not final classifications. Review them against the objective and conceptual framework.")
            st.dataframe(review.role_suggestions, use_container_width=True, hide_index=True)
            apply_confidence = st.multiselect(
                "Apply suggestions with confidence", ["High", "Medium", "Low"], default=["High", "Medium"],
                key="agent_role_confidence",
            )
            if st.button("Apply selected role suggestions", key="agent_apply_roles"):
                frame = st.session_state.framework.copy()
                suggestions = review.role_suggestions[
                    review.role_suggestions["confidence"].isin(apply_confidence)
                    & (review.role_suggestions["suggested_role"] != "Unassigned")
                ]
                role_map = dict(zip(suggestions["variable"], suggestions["suggested_role"]))
                measurement_map = dict(zip(suggestions["variable"], suggestions["measurement"]))
                frame["role"] = [role_map.get(variable, role) for variable, role in zip(frame["variable"], frame["role"])]
                frame["measurement"] = [measurement_map.get(variable, measurement) for variable, measurement in zip(frame["variable"], frame["measurement"])]
                st.session_state.framework = frame
                st.session_state.agent_review = build_guided_review(
                    st.session_state.study, st.session_state.analysis_df, st.session_state.framework
                )
                st.success(f"Applied {len(role_map)} reviewable suggestion(s). Open the conceptual-framework tab to confirm or edit them.")

        if review.construct_suggestions:
            st.markdown("### Suggested construct blocks")
            construct_frame = pd.DataFrame([{
                "construct": item["name"],
                "suggested_items": ", ".join(item["items"]),
                "measurement": item["mode"],
                "confidence": item["confidence"],
                "rationale": item["rationale"],
            } for item in review.construct_suggestions])
            st.dataframe(construct_frame, use_container_width=True, hide_index=True)
            target_method = st.selectbox("Load suggestions into", ["PLS-SEM", "Covariance-based SEM", "CFA"], key="agent_construct_target")
            if st.button("Load construct suggestions for confirmation", key="agent_apply_constructs"):
                target_key = {"PLS-SEM": "pls_sem", "Covariance-based SEM": "sem", "CFA": "cfa"}[target_method]
                definitions = []
                for suggestion in review.construct_suggestions:
                    mode = suggestion["mode"] if target_key == "pls_sem" else "Reflective common-factor"
                    definitions.append({"name": suggestion["name"], "mode": mode, "items": suggestion["items"]})
                st.session_state.study[f"{target_key}_construct_definitions"] = definitions
                st.session_state.recommended_method_key = target_key
                st.session_state["analysis_method_label"] = METHOD_LABELS[target_key]
                st.success("Suggestions were loaded into the construct builder. Confirm names, indicators, measurement modes and structural relationships before analysis.")

        st.markdown("### AI-completed analysis specification")
        auto_spec = review.auto_specification
        if auto_spec is None:
            st.info("Upload a dataset to generate a complete analysis specification.")
        else:
            st.write(f"**Completion confidence: {auto_spec.confidence}**")
            st.dataframe(auto_spec.completed_fields, use_container_width=True, hide_index=True)
            if auto_spec.assumptions_for_confirmation:
                st.markdown("**Provisional assumptions to confirm**")
                for assumption in auto_spec.assumptions_for_confirmation:
                    st.info(assumption)
            if auto_spec.critical_blockers:
                st.error("Critical human input is required before the model can run.")
                for blocker in auto_spec.critical_blockers:
                    st.write(f"• {blocker}")
            else:
                confirm_spec = st.checkbox(
                    "I confirm that the inferred variables, directions, measurement blocks and construction rules match the study.",
                    key="agent_confirm_auto_spec",
                )
                cauto1, cauto2 = st.columns(2)
                with cauto1:
                    if st.button("Use the complete AI specification", key="agent_apply_full_spec"):
                        apply_auto_spec_to_state(auto_spec)
                        st.success("The complete specification, variable roles and provisional framework have been applied. It can be run here or reviewed in the analysis workspace.")
                with cauto2:
                    if st.button("Run AI-completed analysis", type="primary", disabled=not confirm_spec, key="agent_run_full_spec"):
                        try:
                            validation_error = validate_config(auto_spec.method_key, auto_spec.config)
                            if validation_error:
                                st.error(validation_error)
                            else:
                                with st.spinner("Running the AI-completed analysis and diagnostics..."):
                                    result = run_analysis(st.session_state.analysis_df, auto_spec.method_key, auto_spec.config)
                                st.session_state.analysis_result = result
                                st.session_state.analysis_plan = analysis_plan_frame(auto_spec.method_label, auto_spec.config)
                                st.session_state.study["method"] = auto_spec.method_label
                                st.session_state.study["alpha"] = auto_spec.config.get("alpha", st.session_state.study.get("alpha", 0.05))
                                apply_auto_spec_to_state(auto_spec)
                                st.success("Analysis completed. Review the diagnostics, diagrams and paper-ready reporting outputs.")
                        except Exception as exc:
                            st.error(f"The guided analysis could not be completed: {exc}")

        st.markdown("### Guided next actions")
        for number, action in enumerate(review.next_actions, start=1):
            st.write(f"{number}. {action}")

with analysis_tab:
    st.subheader("Configure and run analysis")
    if st.session_state.analysis_df is None:
        st.info("Load a dataset before configuring an analysis.")
    else:
        df = st.session_state.analysis_df
        columns = list(df.columns)
        numeric_columns = list(df.select_dtypes(include="number").columns)
        default_label = METHOD_LABELS.get(st.session_state.recommended_method_key, METHOD_LABELS["descriptive"])
        labels = list(METHOD_OPTIONS.keys())
        default_index = labels.index(default_label) if default_label in labels else 0
        if st.session_state.get("analysis_method_label") not in labels:
            st.session_state["analysis_method_label"] = default_label
        selected_label = st.selectbox("Statistical method", labels, index=default_index, key="analysis_method_label")
        selected_key = METHOD_OPTIONS[selected_label]
        config = render_method_configuration(selected_key, columns, numeric_columns)
        saved_auto_spec = st.session_state.get("auto_specification")
        if saved_auto_spec is not None and getattr(saved_auto_spec, "method_key", None) == selected_key:
            with st.expander("AI-completed configuration available", expanded=True):
                st.dataframe(saved_auto_spec.completed_fields, use_container_width=True, hide_index=True)
                use_auto = st.checkbox("Use these completed values for this run", value=True, key="analysis_use_auto_spec")
                if use_auto:
                    config = dict(saved_auto_spec.config)
                    st.caption("The AI-completed values replace the manual controls above for this run. Clear the checkbox to use the manual selections.")
        config["alpha"] = float(config.get("alpha", st.session_state.study.get("alpha", 0.05)))
        if selected_key in {"cfa", "sem"}:
            st.warning("CFA and covariance-based SEM use an internal covariance-fitting engine with ML, GLS, ULS and DWLS objectives. Confirm publication-critical models in specialist SEM software, especially for ordinal indicators, complex models or small samples.")
        if selected_key == "pls_sem":
            st.warning("The internal PLS-SEM engine supports reflective and formative blocks, mediation and two-stage latent-score moderation. Confirm publication-critical estimates in specialist PLS-SEM software.")

        st.markdown("#### Pre-analysis confirmation")
        preview_plan = analysis_plan_frame(selected_label, config)
        st.dataframe(preview_plan, use_container_width=True, hide_index=True)
        st.warning("Confirm that the variable roles, measurement levels, unit of analysis and sampling design support this method. The app cannot infer design features that are absent from the data and study description.")

        if st.button("Run statistical analysis", type="primary"):
            validation_error = validate_config(selected_key, config)
            if validation_error:
                st.error(validation_error)
            else:
                try:
                    with st.spinner("Running the analysis and diagnostics..."):
                        result = run_analysis(df, selected_key, config)
                    st.session_state.analysis_result = result
                    st.session_state.analysis_plan = preview_plan
                    st.session_state.study["method"] = selected_label
                    st.session_state.study["alpha"] = config["alpha"]
                    st.success("Analysis completed. Review the diagnostics before using the results.")
                except Exception as exc:
                    st.error(f"Analysis could not be completed: {exc}")

with results_tab:
    st.subheader("Results, diagnostics and reproducibility package")
    result = st.session_state.analysis_result
    if result is None:
        st.info("Run an analysis to generate results and exports.")
    else:
        st.markdown(f"### {result.method}")
        st.write(result.summary)
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        latent_payload = latent_diagram_payload(result)
        if latent_payload:
            construct_map, paths, _, _, _, editor_figure_name, _ = latent_payload
            st.markdown("### Interactive path-diagram editor")
            st.caption("Drag constructs directly, or click a construct and use the arrow controls. Select Save arrangement to update the publication-quality PNG, DOCX, Excel and reproducibility exports.")
            saved_positions = (result.metadata.get("diagram_settings") or {}).get("custom_positions") or st.session_state.diagram_positions.get(editor_figure_name, {})
            edited_positions = path_editor(
                nodes=list(construct_map), edges=paths, positions=saved_positions,
                height=680, key=f"path_editor_{editor_figure_name}",
            )
            if edited_positions and edited_positions != saved_positions:
                st.session_state.diagram_positions[editor_figure_name] = edited_positions
                update_result_diagram(result, edited_positions)
                st.success("The saved arrangement has been applied to the result diagram and all new exports.")
            arrangement = (result.metadata.get("diagram_settings") or {}).get("custom_positions") or edited_positions or saved_positions
            if arrangement:
                import json as _json
                st.download_button(
                    "Download diagram arrangement", _json.dumps(arrangement, indent=2),
                    file_name="StatReady_path_diagram_arrangement.json", mime="application/json",
                    key="download_diagram_arrangement",
                )

        interactive_network = result.metadata.get("interactive_network_html")
        if interactive_network:
            st.markdown("### Interactive network diagram")
            st.caption("Drag nodes and click a node to inspect its centrality and community measures. The static publication diagrams remain available below.")
            components.html(interactive_network, height=720, scrolling=False)
            st.download_button(
                "Download interactive network HTML", interactive_network,
                file_name="StatReady_Interactive_Network.html", mime="text/html",
                key="download_interactive_network",
            )

        if result.figures:
            st.markdown("### Diagrams and figures")
            st.caption("Interpret figures together with coefficient, diagnostic, stability and model-fit tables.")
            for figure_index, (figure_name, figure_bytes) in enumerate(result.figures.items(), start=1):
                st.image(figure_bytes, caption=figure_name, use_container_width=True)
                safe_name = "".join(ch if ch.isalnum() else "_" for ch in figure_name).strip("_")
                st.download_button(
                    f"Download {figure_name} (PNG)",
                    figure_bytes,
                    file_name=f"{safe_name}.png",
                    mime="image/png",
                    key=f"figure_download_{figure_index}_{safe_name}",
                )

        descriptive_tables = {name: table for name, table in result.tables.items() if name.startswith("Descriptive ")}
        inferential_tables = {name: table for name, table in result.tables.items() if not name.startswith("Descriptive ")}

        if descriptive_tables:
            st.markdown("### Descriptive statistics")
            st.caption(result.metadata.get("descriptive_summary", "Automatically generated for the variables used in this analysis."))
            descriptive_display_names = {
                "Descriptive sample overview": "Analysis sample overview",
                "Descriptive statistics - Numeric variables": "Numeric variables",
                "Descriptive statistics - Categorical summary": "Categorical summary",
                "Descriptive statistics - Frequencies": "Frequencies",
                "Descriptive statistics - By group": "Descriptive statistics by group",
                "Descriptive profile overview": "Profile sample overview",
                "Descriptive profile - Numeric variables": "Profile numeric variables",
                "Descriptive profile - Categorical summary": "Profile categorical summary",
                "Descriptive profile - Frequencies": "Profile frequencies",
            }
            for name, table in descriptive_tables.items():
                display_name = descriptive_display_names.get(
                    name, name.replace("Descriptive statistics - ", "").replace("Descriptive ", "")
                )
                with st.expander(display_name, expanded=display_name in {"Analysis sample overview", "Numeric variables"}):
                    st.dataframe(table, use_container_width=True, hide_index=True)

        if inferential_tables:
            st.markdown("### Inferential results")
            for name, table in inferential_tables.items():
                with st.expander(name, expanded=name in {"Selected coefficient table", "Test result", "Model fit", "Indirect effect", "Multicollinearity action summary", "Ridge sensitivity model fit", "CFA fit indices", "SEM fit indices", "Structural path estimates", "PLS-SEM model summary", "PLS structural path estimates", "Construct reliability and convergent validity", "HTMT discriminant validity", "Multilevel model fit and variance partition", "Multilevel fixed effects", "Panel model decision", "Repeated-measures ANOVA", "Fixed effects", "Parallel indirect effects", "Conditional indirect effects"}):
                    st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("### Diagnostics and assumptions")
        if result.diagnostics.empty:
            st.write("No method-specific diagnostic table was generated.")
        else:
            st.dataframe(result.diagnostics, use_container_width=True, hide_index=True)
            concerns = result.diagnostics[result.diagnostics["status"].isin(["Minor concern", "Material concern"])]
            if concerns.empty:
                st.success("No diagnostic was classified as a material or minor concern.")
            else:
                st.warning(f"{len(concerns)} diagnostic item(s) require interpretation or sensitivity analysis. Review the recommended responses rather than altering data to seek significance.")

        if result.metadata.get("diagnostic_response"):
            st.markdown("### Diagnostic response and alternative model")
            st.info(result.metadata["diagnostic_response"])

        st.markdown("### Treatment and analysis audit trail")
        combined_audit = audit_frame(st.session_state.audit_entries + result.treatment_log)
        st.dataframe(combined_audit, use_container_width=True, hide_index=True)

        st.markdown("### Supporting methodological literature")
        refs = references_for_method(result.method, result.diagnostics)
        st.dataframe(refs, use_container_width=True, hide_index=True)
        st.caption("Phase 2 uses a curated methodological library. Live DOI and metadata verification can be connected to CiteIntegrity in the next integration step.")

        st.markdown("### Export")
        study = st.session_state.study
        plan = st.session_state.analysis_plan
        original_df = st.session_state.original_df
        analysis_df = st.session_state.analysis_df
        try:
            docx_bytes = build_docx_report(result, study, plan, st.session_state.audit_entries)
            xlsx_bytes = build_excel_report(original_df, analysis_df, result, plan, st.session_state.audit_entries)
            zip_bytes = build_reproducibility_package(original_df, analysis_df, result, study, plan, st.session_state.audit_entries)
            c1, c2, c3 = st.columns(3)
            c1.download_button("Download DOCX report", docx_bytes, "StatReady_Report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            c2.download_button("Download Excel results", xlsx_bytes, "StatReady_Results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            c3.download_button("Download full reproducibility ZIP", zip_bytes, "StatReady_Reproducibility_Package.zip", "application/zip")
        except Exception as exc:
            st.error(f"Export generation failed: {exc}")

        with st.expander("Generated reproducibility code"):
            st.code(result.reproducible_code, language="python")

st.divider()
st.caption("StatReady AI Phase 2.4 | Drag-and-click path editing | Autonomous guided analysis | Comprehensive network analysis | Original data preserved")
