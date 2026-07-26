from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

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
        "Construct measurement model": "; ".join(f"{construct}: {', '.join(items)}" for construct, items in (config.get("construct_map") or {}).items()),
        "Structural paths": "; ".join(f"{predictor} -> {outcome}" for predictor, outcome in (config.get("paths") or [])),
        "Repeated measurements": ", ".join(config.get("measurements", [])),
        "Subject identifier": config.get("subject_id", ""),
        "Cluster variable": config.get("cluster", ""),
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
    elif method_key in {"cfa", "sem"}:
        default_spec = st.session_state.study.get(f"{method_key}_construct_spec", "")
        construct_text = st.text_area(
            "Construct measurement specification", value=default_spec, height=150,
            placeholder="DigitalCompetence: dc1, dc2, dc3\nTeachingEffectiveness: te1, te2, te3",
            help="Enter one construct per line. Every item must match a numeric dataset column and can appear only once.",
            key=f"{method_key}_construct_text",
        )
        st.session_state.study[f"{method_key}_construct_spec"] = construct_text
        config["construct_map"], config["construct_parse_error"] = parse_construct_map(construct_text, numeric_columns)
        if method_key == "sem":
            path_text = st.text_area(
                "Structural paths", value=st.session_state.study.get("sem_paths", ""), height=120,
                placeholder="DigitalCompetence -> TeachingEffectiveness",
                help="Enter one directed path per line. Phase 2 supports acyclic structural models.",
                key="sem_path_text",
            )
            st.session_state.study["sem_paths"] = path_text
            config["paths"], config["path_parse_error"] = parse_structural_paths(path_text)
        config["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key=f"{method_key}_seed"))
    elif method_key == "repeated_measures":
        config["measurements"] = st.multiselect("Repeated measurement columns", numeric_columns)
        config["subject_id"] = variable_selector("Subject identifier (optional)", columns, "rm_subject", allow_none=True, default=next((v for v in roles["Identifier"] if v in columns), None))
    elif method_key == "mixed_effects":
        config["outcome"] = variable_selector("Continuous outcome", numeric_columns, "mixed_outcome", default=default_outcome)
        config["cluster"] = variable_selector("Cluster or subject identifier", columns, "mixed_cluster", default=next((v for v in roles["Cluster"] + roles["Group"] + roles["Identifier"] if v in columns), None))
        candidates = [c for c in numeric_columns if c != config["outcome"]]
        config["predictors"] = st.multiselect("Fixed-effect predictors", candidates, default=[v for v in default_predictors if v in candidates])
        config["random_slope"] = variable_selector("Optional random-slope variable", candidates, "mixed_slope", allow_none=True)
        config["reml"] = st.checkbox("Use restricted maximum likelihood", value=True)
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
        "repeated_measures": ["measurements"],
        "mixed_effects": ["outcome", "predictors", "cluster"],
        "panel": ["outcome", "predictors", "entity", "time"],
        "advanced_moderation": ["outcome", "predictor", "moderator"],
        "parallel_mediation": ["outcome", "predictor", "mediators"],
        "moderated_mediation": ["outcome", "predictor", "mediator", "moderator"],
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
    if method_key in {"cfa", "sem"} and config.get("construct_parse_error"):
        return config["construct_parse_error"]
    if method_key == "sem" and config.get("path_parse_error"):
        return config["path_parse_error"]
    if method_key == "repeated_measures" and len(config.get("measurements", [])) < 2:
        return "Select at least two repeated measurements."
    if method_key == "parallel_mediation" and len(config.get("mediators", [])) < 2:
        return "Select at least two parallel mediators."
    if method_key == "panel" and config.get("entity") == config.get("time"):
        return "Entity and time identifiers must be different variables."
    return None


init_state()

st.title("StatReady AI")
st.caption("Phase 2: objective-aligned statistics, latent-variable analysis, longitudinal models, diagnostics and reproducible reporting")
st.info("The app never changes data merely to obtain significance. It preserves the original dataset, records every treatment, uses robust or alternative methods where justified, and keeps sensitivity results visible.")

study_tab, data_tab, framework_tab, analysis_tab, results_tab = st.tabs([
    "1. Study design", "2. Data preparation", "3. Conceptual framework", "4. Run analysis", "5. Results and exports"
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
        selected_label = st.selectbox("Statistical method", labels, index=default_index)
        selected_key = METHOD_OPTIONS[selected_label]
        config = render_method_configuration(selected_key, columns, numeric_columns)
        config["alpha"] = float(config.get("alpha", st.session_state.study.get("alpha", 0.05)))
        if selected_key in {"cfa", "sem"}:
            st.warning("Phase 2 CFA and SEM use the internal maximum-likelihood covariance engine. Confirm publication-critical models in specialist SEM software, especially when using complex specifications or smaller samples.")

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

        if result.figures:
            st.markdown("### Path and measurement diagram")
            st.caption("Standardised estimates are displayed. Interpret the figure together with the coefficient tables, diagnostics and model-fit indices.")
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
                with st.expander(name, expanded=name in {"Selected coefficient table", "Test result", "Model fit", "Indirect effect", "Multicollinearity action summary", "Ridge sensitivity model fit", "CFA fit indices", "SEM fit indices", "Structural path estimates", "Panel model decision", "Repeated-measures ANOVA", "Fixed effects", "Parallel indirect effects", "Conditional indirect effects"}):
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
st.caption("StatReady AI Phase 2 | Original data preserved | Advanced models audited | Alternatives documented | No significance-seeking data manipulation")
