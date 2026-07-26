from pathlib import Path
import pandas as pd

from statready.dispatch import run_analysis
from statready.reports import build_docx_report, build_excel_report, build_reproducibility_package

base = Path(__file__).resolve().parent
out = base / "demo_output"
out.mkdir(exist_ok=True)


def export_demo(prefix, data, result, study, plan):
    (out / f"{prefix}_Report.docx").write_bytes(build_docx_report(result, study, plan, []))
    (out / f"{prefix}_Results.xlsx").write_bytes(build_excel_report(data, data.copy(), result, plan, []))
    (out / f"{prefix}_Reproducibility.zip").write_bytes(build_reproducibility_package(data, data.copy(), result, study, plan, []))
    for name, content in result.figures.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        (out / f"{prefix}_{safe}.png").write_bytes(content)


factor = pd.read_csv(base / "sample_data" / "phase2_factor_sample.csv")
construct_map = {
    "ConstructA": ["a1", "a2", "a3", "a4"],
    "ConstructB": ["b1", "b2", "b3", "b4"],
    "Mediator": ["m1", "m2"],
}
relations = [
    {"type": "Mediator", "predictor": "ConstructA", "mediator": "Mediator", "outcome": "ConstructB", "include_direct": True},
    {"type": "Moderator", "predictor": "ConstructA", "moderator": "Mediator", "outcome": "ConstructB"},
]
pls_config = {
    "construct_map": construct_map,
    "measurement_modes": {"ConstructA": "reflective", "ConstructB": "reflective", "Mediator": "reflective"},
    "structural_relations": relations,
    "paths": [("ConstructA", "Mediator"), ("Mediator", "ConstructB"), ("ConstructA", "ConstructB")],
    "moderations": [{"predictor": "ConstructA", "moderator": "Mediator", "outcome": "ConstructB"}],
    "bootstrap_samples": 300,
    "max_iter": 300,
    "tolerance": 1e-7,
    "weighting_scheme": "Path",
    "random_state": 42,
    "alpha": 0.05,
}
pls_result = run_analysis(factor, "pls_sem", pls_config)
pls_study = {
    "title": "StatReady Phase 2.2 Demonstration: PLS-SEM Mediation and Moderation",
    "objective": "Estimate the direct, mediated and moderated relationships among three latent constructs.",
    "hypothesis": "Construct A influences Construct B directly and through the Mediator, and the relationship varies with the Mediator score.",
    "alpha": 0.05,
    "method": pls_result.method,
    "framework_notes": "Construct A predicts the Mediator and Construct B. The Mediator predicts Construct B and is also used in a two-stage interaction demonstration.",
}
pls_plan = pd.DataFrame([
    {"component": "Method", "specification": pls_result.method},
    {"component": "Measurement model", "specification": "ConstructA and ConstructB use four reflective items; Mediator uses two reflective items."},
    {"component": "Structural relationships", "specification": "Direct, mediation and moderation"},
    {"component": "Bootstrap resamples", "specification": 300},
    {"component": "Analysis dataset rows", "specification": len(factor)},
])
export_demo("StatReady_Phase2_2_PLS_SEM", factor, pls_result, pls_study, pls_plan)

multilevel = pd.read_csv(base / "sample_data" / "phase2_multilevel_sample.csv")
ml_config = {
    "outcome": "performance",
    "cluster": "school_id",
    "level1_predictors": ["engagement", "prior_achievement"],
    "level2_predictors": ["school_support"],
    "random_slope": "engagement",
    "estimator": "REML",
    "centering": "Group-mean with contextual effect",
    "optimizer": "lbfgs",
    "gee_correlation": "Exchangeable",
    "outcome_family": "Continuous",
    "alpha": 0.05,
}
ml_result = run_analysis(multilevel, "multilevel", ml_config)
ml_study = {
    "title": "StatReady Phase 2.2 Demonstration: Students Nested in Schools",
    "objective": "Estimate within-school and between-school effects on student performance.",
    "hypothesis": "Student engagement and prior achievement predict performance, while school support contributes at level 2.",
    "alpha": 0.05,
    "method": ml_result.method,
    "framework_notes": "Students are nested within schools. Engagement and prior achievement vary at level 1, while school support is a level-2 predictor.",
}
ml_plan = pd.DataFrame([
    {"component": "Method", "specification": "REML multilevel linear model with GEE sensitivity"},
    {"component": "Outcome", "specification": "performance"},
    {"component": "Level-1 predictors", "specification": "engagement; prior_achievement"},
    {"component": "Level-2 predictor", "specification": "school_support"},
    {"component": "Cluster", "specification": "school_id"},
    {"component": "Random slope", "specification": "engagement"},
    {"component": "Centring", "specification": "Group-mean with contextual effect"},
])
export_demo("StatReady_Phase2_2_Multilevel", multilevel, ml_result, ml_study, ml_plan)

binary_config = {
    "outcome": "completed",
    "cluster": "school_id",
    "level1_predictors": ["engagement", "prior_achievement"],
    "level2_predictors": ["school_support"],
    "random_slope": None,
    "estimator": "GEE robust",
    "centering": "Group-mean with contextual effect",
    "optimizer": "lbfgs",
    "gee_correlation": "Exchangeable",
    "outcome_family": "Binary",
    "alpha": 0.05,
}
binary_result = run_analysis(multilevel, "multilevel", binary_config)
binary_study = {
    "title": "StatReady Phase 2.2 Demonstration: Binary Clustered Outcome",
    "objective": "Estimate population-average predictors of completion for students nested within schools.",
    "hypothesis": "Engagement, prior achievement and school support predict programme completion.",
    "alpha": 0.05,
    "method": binary_result.method,
    "framework_notes": "Students are nested within schools and completion is binary.",
}
binary_plan = pd.DataFrame([
    {"component": "Method", "specification": "Robust binomial GEE"},
    {"component": "Outcome", "specification": "completed"},
    {"component": "Level-1 predictors", "specification": "engagement; prior_achievement"},
    {"component": "Level-2 predictor", "specification": "school_support"},
    {"component": "Cluster", "specification": "school_id"},
    {"component": "Working correlation", "specification": "Exchangeable"},
])
export_demo("StatReady_Phase2_2_Binary_GEE", multilevel, binary_result, binary_study, binary_plan)

print(pls_result.summary)
print(ml_result.summary)
print(binary_result.summary)
