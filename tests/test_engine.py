from pathlib import Path

import pandas as pd

from statready.dispatch import run_analysis
from statready.methods import (
    chi_square_test,
    correlation_analysis,
    cronbach_alpha,
    descriptive_statistics,
    independent_t_test,
    logistic_regression,
    mediation_analysis,
    moderation_analysis,
    ols_regression,
    one_way_anova,
    paired_t_test,
)
from statready.profiling import dataset_profile
from statready.recommender import recommend_method
from statready.reports import build_docx_report, build_excel_report, build_reproducibility_package


DATA = Path(__file__).parents[1] / "sample_data" / "sample_research_data.csv"


def load_data():
    return pd.read_csv(DATA)


def test_profile_and_recommendation():
    df = load_data()
    profile = dataset_profile(df)
    assert int(profile["overview"].iloc[0]["rows"]) == 20
    recommendation = recommend_method("Assess the mediating role of motivation")
    assert recommendation["method_key"] == "mediation"


def test_descriptive_reliability_and_correlation():
    df = load_data()
    descriptive = descriptive_statistics(df, ["training", "performance", "gender"])
    assert "Descriptive statistics - Numeric variables" in descriptive.tables
    reliability = cronbach_alpha(df, ["item1", "item2", "item3", "item4"])
    assert reliability.metadata["cronbach_alpha"] > 0.5
    correlation = correlation_analysis(df, ["training", "motivation", "performance"])
    assert len(correlation.tables["Pairwise tests"]) == 3


def test_group_tests():
    df = load_data()
    ttest = independent_t_test(df[df["group"].isin(["A", "B"])], "performance", "group")
    assert "Test result" in ttest.tables
    paired = paired_t_test(df, "before_score", "after_score")
    assert paired.tables["Test result"].iloc[0]["mean_change"] > 0
    anova = one_way_anova(df, "performance", "group")
    assert "ANOVA result" in anova.tables
    chi = chi_square_test(df, "group", "passed")
    assert "Observed counts" in chi.tables


def test_regressions_and_mediation():
    df = load_data()
    ols = ols_regression(df, "performance", ["training", "motivation", "support"])
    assert "Selected coefficient table" in ols.tables
    assert ols.metadata["material_multicollinearity"] is True
    assert "Ridge sensitivity coefficients" in ols.tables
    assert "Multicollinearity action summary" in ols.tables
    assert any(entry.action == "Ran cross-validated ridge sensitivity model" for entry in ols.treatment_log)
    assert ols.tables["VIF"]["vif"].max() < 100
    logistic = logistic_regression(df, "passed", ["training", "motivation", "support"])
    assert "Coefficients and odds ratios" in logistic.tables
    moderation = moderation_analysis(df, "performance", "training", "support", ["motivation"])
    assert "interaction" in moderation.metadata
    mediation = mediation_analysis(
        df, "performance", "training", "motivation", ["support"], bootstrap_samples=500
    )
    assert "Indirect effect" in mediation.tables



def test_automatic_descriptives_for_inferential_analysis():
    df = load_data()
    result = run_analysis(df, "ols", {
        "outcome": "performance",
        "predictors": ["training", "motivation", "support"],
        "alpha": 0.05,
    })
    assert result.metadata["descriptive_statistics_included"] is True
    assert "Descriptive sample overview" in result.tables
    assert "Descriptive statistics - Numeric variables" in result.tables
    numeric = result.tables["Descriptive statistics - Numeric variables"]
    assert set(["performance", "training", "motivation", "support"]).issubset(set(numeric["variable"]))
    assert {"mean", "std_dev", "median", "minimum", "maximum", "skewness", "kurtosis"}.issubset(numeric.columns)


def test_grouped_descriptives_for_group_comparison():
    df = load_data()
    result = run_analysis(df, "anova", {
        "outcome": "performance",
        "group": "group",
        "alpha": 0.05,
    })
    assert "Descriptive statistics - By group" in result.tables
    grouped = result.tables["Descriptive statistics - By group"]
    assert grouped["group"].nunique() == df["group"].nunique()


def test_optional_profile_descriptives_do_not_enter_model():
    df = load_data()
    result = run_analysis(df, "ols", {
        "outcome": "performance",
        "predictors": ["training", "motivation", "support"],
        "profile_variables": ["gender", "group"],
        "alpha": 0.05,
    })
    assert result.metadata["profile_descriptive_variables"] == ["gender", "group"]
    assert "Descriptive profile - Frequencies" in result.tables
    assert result.tables["Model fit"].iloc[0]["n"] == len(df)

def test_exports():
    df = load_data()
    result = ols_regression(df, "performance", ["training", "motivation", "support"])
    study = {"title": "Sample", "objective": "Estimate effects", "hypothesis": "H1", "alpha": 0.05, "method": result.method}
    plan = pd.DataFrame([{"component": "Method", "specification": result.method}])
    docx = build_docx_report(result, study, plan, [])
    xlsx = build_excel_report(df, df, result, plan, [])
    package = build_reproducibility_package(df, df, result, study, plan, [])
    assert docx[:2] == b"PK"
    assert xlsx[:2] == b"PK"
    assert package[:2] == b"PK"


def make_phase2_data(seed: int = 7):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = 140
    factor_a = rng.normal(size=n)
    factor_b = 0.40 * factor_a + rng.normal(scale=0.92, size=n)
    frame = {}
    for index, loading in enumerate([0.82, 0.76, 0.71, 0.66], start=1):
        frame[f"a{index}"] = loading * factor_a + rng.normal(scale=np.sqrt(1 - loading ** 2), size=n)
    for index, loading in enumerate([0.84, 0.78, 0.72, 0.68], start=1):
        frame[f"b{index}"] = loading * factor_b + rng.normal(scale=np.sqrt(1 - loading ** 2), size=n)
    frame["x"] = factor_a
    frame["w"] = factor_b
    frame["m1"] = 0.65 * factor_a + rng.normal(size=n)
    frame["m2"] = 0.35 * factor_a + rng.normal(size=n)
    frame["y"] = 0.35 * factor_a + 0.55 * frame["m1"] + 0.20 * factor_b + rng.normal(size=n)
    return pd.DataFrame(frame)


def make_longitudinal_data(seed: int = 11):
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(24):
        entity_effect = rng.normal(0, 1.8)
        for time in range(5):
            x = rng.normal()
            y = 3.0 + 1.1 * x + 0.25 * time + entity_effect + rng.normal()
            rows.append((entity, time, x, y))
    return pd.DataFrame(rows, columns=["entity", "time", "x", "y"])


def test_phase2_factor_models():
    from statready.phase2 import exploratory_factor_analysis, confirmatory_factor_analysis, structural_equation_model
    df = make_phase2_data()
    items = [f"a{i}" for i in range(1, 5)] + [f"b{i}" for i in range(1, 5)]
    construct_map = {"ConstructA": [f"a{i}" for i in range(1, 5)], "ConstructB": [f"b{i}" for i in range(1, 5)]}
    efa = exploratory_factor_analysis(df, items, n_factors=2, parallel_iterations=50)
    assert efa.metadata["n_factors"] == 2
    assert "Factor loadings" in efa.tables
    cfa = confirmatory_factor_analysis(df, construct_map)
    assert cfa.tables["CFA fit indices"].iloc[0]["cfi"] > 0.85
    sem = structural_equation_model(df, construct_map, [("ConstructA", "ConstructB")])
    assert "Structural path estimates" in sem.tables
    assert len(sem.tables["Structural path estimates"]) == 1


def test_phase2_longitudinal_models():
    from statready.phase2 import repeated_measures_anova, mixed_effects_model, panel_data_analysis
    import numpy as np
    rng = np.random.default_rng(20)
    base = rng.normal(50, 5, 36)
    wide = pd.DataFrame({
        "id": range(36),
        "t1": base,
        "t2": base + 2 + rng.normal(0, 2, 36),
        "t3": base + 4 + rng.normal(0, 2, 36),
    })
    repeated = repeated_measures_anova(wide, ["t1", "t2", "t3"], "id")
    assert "Repeated-measures ANOVA" in repeated.tables
    long = make_longitudinal_data()
    mixed = mixed_effects_model(long, "y", ["x", "time"], "entity", "time")
    assert mixed.tables["Mixed-model fit"].iloc[0]["converged"]
    panel = panel_data_analysis(long, "y", ["x"], "entity", "time")
    assert "Panel model decision" in panel.tables
    assert panel.metadata["selected_model"]


def test_phase2_conditional_process_models():
    from statready.phase2 import advanced_moderation_analysis, parallel_mediation_analysis, moderated_mediation_analysis
    df = make_phase2_data()
    moderation = advanced_moderation_analysis(df, "y", "x", "w")
    assert len(moderation.tables["Conditional simple slopes"]) == 3
    parallel = parallel_mediation_analysis(df, "y", "x", ["m1", "m2"], bootstrap_samples=500)
    assert len(parallel.tables["Parallel indirect effects"]) == 3
    moderated = moderated_mediation_analysis(df, "y", "x", "m1", "w", bootstrap_samples=500)
    assert len(moderated.tables["Conditional indirect effects"]) == 3


def test_phase2_dispatch_and_exports():
    df = make_phase2_data()
    construct_map = {"ConstructA": [f"a{i}" for i in range(1, 5)], "ConstructB": [f"b{i}" for i in range(1, 5)]}
    result = run_analysis(df, "cfa", {"construct_map": construct_map, "alpha": 0.05, "random_state": 42})
    assert result.metadata["descriptive_statistics_included"] is True
    assert "CFA fit indices" in result.tables
    assert "CFA measurement diagram" in result.figures
    assert result.figures["CFA measurement diagram"][:8] == b"\x89PNG\r\n\x1a\n"
    study = {"title": "Phase 2 sample", "objective": "Confirm the measurement model", "hypothesis": "", "alpha": 0.05, "method": result.method}
    plan = pd.DataFrame([{"component": "Method", "specification": result.method}])
    assert build_docx_report(result, study, plan, [])[:2] == b"PK"
    assert build_excel_report(df, df, result, plan, [])[:2] == b"PK"


def test_phase22_pls_sem_measurement_structural_and_diagnostics():
    from statready.pls_sem import partial_least_squares_sem
    df = make_phase2_data()
    construct_map = {
        "ConstructA": [f"a{i}" for i in range(1, 5)],
        "ConstructB": [f"b{i}" for i in range(1, 5)],
        "Mediator": ["m1", "m2"],
    }
    result = partial_least_squares_sem(
        df,
        construct_map=construct_map,
        paths=[("ConstructA", "Mediator"), ("Mediator", "ConstructB"), ("ConstructA", "ConstructB")],
        measurement_modes={"ConstructA": "reflective", "ConstructB": "reflective", "Mediator": "reflective"},
        moderations=[{"predictor": "ConstructA", "moderator": "Mediator", "outcome": "ConstructB"}],
        bootstrap_samples=100,
        random_state=9,
    )
    assert "PLS outer loadings" in result.tables
    assert "HTMT discriminant validity" in result.tables
    assert "Inner VIF" in result.tables
    assert "Endogenous construct R squared" in result.tables
    assert "Structural effect sizes f squared" in result.tables
    assert "Predictive relevance Q squared" in result.tables
    assert len(result.tables["PLS structural path estimates"]) >= 4
    assert set(["Reflective indicator reliability", "Discriminant validity", "Structural-model collinearity"]).issubset(set(result.diagnostics["diagnostic"]))


def test_phase22_multilevel_estimators_and_diagnostics():
    from statready.multilevel import multilevel_linear_model
    import numpy as np
    rng = np.random.default_rng(31)
    clusters = np.repeat(np.arange(18), 10)
    x = rng.normal(size=len(clusters))
    z_cluster = rng.normal(size=18)
    z = np.repeat(z_cluster, 10)
    y = 1.0 + 0.75 * x + 0.45 * z + np.repeat(rng.normal(scale=0.7, size=18), 10) + rng.normal(scale=0.8, size=len(clusters))
    frame = pd.DataFrame({"cluster": clusters, "x": x, "z": z, "y": y})
    result = multilevel_linear_model(
        frame, "y", ["x"], ["z"], "cluster", random_slope="x",
        estimator="REML", centering="Group-mean with contextual effect",
    )
    fit = result.tables["Multilevel model fit and variance partition"].iloc[0]
    assert fit["converged"]
    assert 0 <= fit["icc_1"] <= 1
    assert "Alternative estimator sensitivity coefficients" in result.tables
    expected = {"Need for multilevel modelling", "Reliability of cluster means", "Random-effects singularity", "Influential clusters"}
    assert expected.issubset(set(result.diagnostics["diagnostic"]))


def test_phase22_additional_covariance_estimators_and_dispatch():
    df = make_phase2_data()
    construct_map = {"ConstructA": [f"a{i}" for i in range(1, 5)], "ConstructB": [f"b{i}" for i in range(1, 5)]}
    for estimator in ["ML", "GLS", "ULS", "DWLS"]:
        result = run_analysis(df, "cfa", {
            "construct_map": construct_map, "alpha": 0.05, "random_state": 42, "estimator": estimator,
        })
        assert result.tables["CFA fit indices"].iloc[0]["estimator"] == estimator
    pls = run_analysis(df, "pls_sem", {
        "construct_map": construct_map,
        "measurement_modes": {"ConstructA": "reflective", "ConstructB": "reflective"},
        "paths": [("ConstructA", "ConstructB")],
        "moderations": [],
        "bootstrap_samples": 100,
        "random_state": 42,
        "alpha": 0.05,
    })
    assert "PLS-SEM path diagram" in pls.figures
    assert pls.figures["PLS-SEM path diagram"][:8] == b"\x89PNG\r\n\x1a\n"


def test_phase22_pls_weighting_formative_bootstrap_and_joint_mediation():
    from statready.pls_sem import partial_least_squares_sem
    df = make_phase2_data()
    construct_map = {
        "ConstructA": [f"a{i}" for i in range(1, 5)],
        "ConstructB": [f"b{i}" for i in range(1, 5)],
        "Mediator": ["m1", "m2"],
    }
    relations = [{
        "type": "Mediator", "predictor": "ConstructA", "mediator": "Mediator",
        "outcome": "ConstructB", "include_direct": True,
    }]
    result = partial_least_squares_sem(
        df,
        construct_map=construct_map,
        paths=[("ConstructA", "Mediator"), ("Mediator", "ConstructB"), ("ConstructA", "ConstructB")],
        measurement_modes={"ConstructA": "reflective", "ConstructB": "reflective", "Mediator": "formative"},
        structural_relations=relations,
        bootstrap_samples=60,
        weighting_scheme="Factorial",
        random_state=13,
    )
    assert result.metadata["weighting_scheme"] == "Factorial"
    assert "bootstrap_p" in result.tables["PLS outer weights"].columns
    assert "Specified mediation effects" in result.tables
    assert result.tables["Specified mediation effects"].iloc[0]["bootstrap_draws"] >= 20
    assert "Formative indicator contribution" in set(result.diagnostics["diagnostic"])


def test_phase22_sem_diagnostics_r_squared_and_latent_vif():
    from statready.phase2 import structural_equation_model
    df = make_phase2_data()
    construct_map = {
        "ConstructA": [f"a{i}" for i in range(1, 5)],
        "ConstructB": [f"b{i}" for i in range(1, 5)],
    }
    result = structural_equation_model(df, construct_map, [("ConstructA", "ConstructB")], estimator="GLS")
    assert "Endogenous construct R squared" in result.tables
    assert "Latent structural VIF" in result.tables
    expected = {"Model identification", "Numerical convergence", "Estimator suitability", "Admissible measurement solution"}
    assert expected.issubset(set(result.diagnostics["diagnostic"]))


def test_phase22_multilevel_binary_and_count_gee():
    from statready.multilevel import multilevel_linear_model
    import numpy as np
    rng = np.random.default_rng(41)
    clusters = np.repeat(np.arange(24), 16)
    x = rng.normal(size=len(clusters))
    z_cluster = rng.normal(size=24)
    z = np.repeat(z_cluster, 16)
    u = np.repeat(rng.normal(scale=0.55, size=24), 16)
    probability = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.35 * z + u)))
    binary = rng.binomial(1, probability)
    count = rng.poisson(np.exp(0.2 + 0.25 * x + 0.2 * z + u))
    frame = pd.DataFrame({"cluster": clusters, "x": x, "z": z, "binary": binary, "count": count})

    binary_result = multilevel_linear_model(
        frame, "binary", ["x"], ["z"], "cluster",
        estimator="GEE robust", outcome_family="Binary", gee_correlation="Exchangeable",
    )
    assert binary_result.metadata["outcome_family"] == "Binary"
    assert "effect_ratio" in binary_result.tables["Multilevel fixed effects"].columns
    assert "Conditional dispersion" in set(binary_result.diagnostics["diagnostic"])

    count_result = multilevel_linear_model(
        frame, "count", ["x"], ["z"], "cluster",
        estimator="GEE robust", outcome_family="Count", gee_correlation="Independence",
    )
    assert count_result.metadata["outcome_family"] == "Count"
    assert count_result.tables["Multilevel model fit and variance partition"].iloc[0]["pseudo_r_squared_name"] == "Poisson deviance explained"


def test_ai_guided_review_and_construct_suggestions():
    from statready.agent import build_guided_review, suggest_constructs
    df = make_phase2_data()
    study = {
        "objective": "Examine the structural effect of ConstructA on ConstructB using SEM",
        "hypothesis": "ConstructA positively predicts ConstructB",
        "outcome_type": "continuous",
        "group_count": 0,
        "paired": False,
        "framework_notes": "ConstructA predicts ConstructB.",
    }
    framework = pd.DataFrame({
        "variable": list(df.columns),
        "role": ["Scale item"] * len(df.columns),
    })
    review = build_guided_review(study, df, framework)
    assert review.method_key == "sem"
    assert not review.role_suggestions.empty
    suggestions = suggest_constructs(df)
    names = {item["name"] for item in suggestions}
    assert {"A", "B"}.issubset(names)


def test_path_diagram_layout_options():
    from PIL import Image
    from io import BytesIO
    from statready.figures import render_latent_path_diagram

    construct_map = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}
    loading_table = pd.DataFrame([
        {"construct": "A", "item": "a1", "standardized_loading": 0.8},
        {"construct": "A", "item": "a2", "standardized_loading": 0.75},
        {"construct": "A", "item": "a3", "standardized_loading": 0.7},
        {"construct": "B", "item": "b1", "standardized_loading": 0.82},
        {"construct": "B", "item": "b2", "standardized_loading": 0.77},
        {"construct": "B", "item": "b3", "standardized_loading": 0.72},
    ])
    path_table = pd.DataFrame([{"predictor": "A", "outcome": "B", "standardized_estimate": 0.45, "p_value": 0.01}])
    for layout in ["Left to right", "Top to bottom", "Bottom to top", "Radial", "Hierarchical", "Compact publication"]:
        output = render_latent_path_diagram(
            construct_map, loading_table, [("A", "B")], path_table,
            settings={"layout": layout, "arrow_style": "Curved", "show_indicators": True},
        )
        assert output[:8] == b"\x89PNG\r\n\x1a\n"
        image = Image.open(BytesIO(output))
        assert image.width >= 1200
        assert image.height >= 650


def test_dispatch_respects_transparent_high_resolution_diagram():
    from PIL import Image
    from io import BytesIO
    df = make_phase2_data()
    construct_map = {"ConstructA": [f"a{i}" for i in range(1, 5)], "ConstructB": [f"b{i}" for i in range(1, 5)]}
    result = run_analysis(df, "sem", {
        "construct_map": construct_map,
        "paths": [("ConstructA", "ConstructB")],
        "structural_relations": [{"type": "Direct", "predictor": "ConstructA", "outcome": "ConstructB"}],
        "diagram_settings": {
            "layout": "Top to bottom",
            "transparent": True,
            "resolution": "High resolution",
            "show_indicators": True,
            "show_loadings": True,
            "show_coefficients": True,
            "show_p_values": False,
        },
        "alpha": 0.05,
        "random_state": 42,
        "estimator": "ML",
    })
    image = Image.open(BytesIO(result.figures["SEM path diagram"]))
    assert image.mode == "RGBA"
    assert image.width > 1600
    assert result.metadata["diagram_settings"]["layout"] == "Top to bottom"


def test_phase24_network_edge_list_complete_outputs():
    from statready.network_analysis import network_analysis
    frame = pd.DataFrame({
        "source": ["A", "A", "B", "C", "D", "D", "E"],
        "target": ["B", "C", "C", "D", "A", "C", "A"],
        "weight": [2.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0],
    })
    result = network_analysis(frame, {
        "network_input": "Edge list", "source": "source", "target": "target",
        "weight": "weight", "directed": False, "layout": "Spring",
        "random_graph_iterations": 10, "random_state": 42,
    })
    assert "Network summary measures" in result.tables
    assert "Node centrality and community measures" in result.tables
    assert "Paper-ready methods narrative" in result.tables
    assert "Network reporting checklist" in result.tables
    assert len(result.figures) >= 5
    assert result.metadata["interactive_network_html"].startswith("<!doctype html>")
    assert set(["degree", "strength", "betweenness", "closeness", "pagerank", "community"]).issubset(
        result.tables["Node centrality and community measures"].columns
    )


def test_phase24_correlation_network_stability_and_dispatch():
    import numpy as np
    rng = np.random.default_rng(44)
    latent = rng.normal(size=120)
    frame = pd.DataFrame({
        "stress": latent + rng.normal(scale=.6, size=120),
        "anxiety": latent + rng.normal(scale=.6, size=120),
        "sleep": -0.5 * latent + rng.normal(size=120),
        "support": rng.normal(size=120),
        "performance": -0.4 * latent + .3 * rng.normal(size=120),
    })
    result = run_analysis(frame, "network", {
        "network_input": "Correlation or partial-correlation network",
        "variables": list(frame.columns), "network_estimator": "Pearson correlation",
        "edge_threshold": 0.10, "retain_negative": True, "bootstrap_samples": 20,
        "layout": "Circular", "random_graph_iterations": 10, "random_state": 42,
        "group_variable": None, "group_values": [], "permutation_samples": 0,
        "alpha": 0.05,
    })
    assert "Bootstrap edge stability" in result.tables
    assert result.metadata["descriptive_statistics_included"] is True
    assert "Adjacency heatmap" in result.figures


def test_phase24_agent_completes_network_and_regression_specs():
    import numpy as np
    from statready.agent import build_auto_specification
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "training": rng.normal(size=80), "motivation": rng.normal(size=80),
        "performance": rng.normal(size=80), "support": rng.normal(size=80),
    })
    regression = build_auto_specification({
        "objective": "Examine the effect of training and motivation on performance",
        "hypothesis": "Training and motivation positively predict performance",
        "outcome_type": "continuous", "alpha": 0.05,
    }, frame, None)
    assert regression.method_key == "ols"
    assert regression.config["outcome"] == "performance"
    assert set(["training", "motivation"]).issubset(regression.config["predictors"])
    assert regression.critical_blockers == []

    network = build_auto_specification({
        "objective": "Analyse the network structure, centrality and communities among training, motivation, performance and support",
        "hypothesis": "Motivation will be a central node",
        "outcome_type": "continuous", "alpha": 0.05,
    }, frame, None)
    assert network.method_key == "network"
    assert network.config["network_input"] == "Correlation or partial-correlation network"
    assert len(network.config["variables"]) == 4
    assert network.critical_blockers == []


def test_phase24_custom_path_positions_and_reproducibility_assets():
    import io
    import zipfile
    from statready.figures import render_latent_path_diagram
    from statready.models import AnalysisResult
    construct_map = {"Training": ["t1", "t2"], "Performance": ["p1", "p2"]}
    settings = {
        "layout": "Left to right", "show_indicators": True, "show_loadings": True,
        "show_indicator_names": True, "show_coefficients": True, "show_p_values": True,
        "show_fit": True, "significance_colours": True, "custom_positions": {
            "Training": {"x": 220, "y": 250}, "Performance": {"x": 850, "y": 360},
        },
    }
    loading = pd.DataFrame([
        {"construct": "Training", "indicator": "t1", "standardized_loading": .8},
        {"construct": "Training", "indicator": "t2", "standardized_loading": .7},
        {"construct": "Performance", "indicator": "p1", "standardized_loading": .85},
        {"construct": "Performance", "indicator": "p2", "standardized_loading": .75},
    ])
    paths = pd.DataFrame([{"predictor": "Training", "outcome": "Performance", "standardized_estimate": .45, "p_value_approx": .01}])
    figure = render_latent_path_diagram(construct_map, loading, [("Training", "Performance")], paths, pd.DataFrame(), settings=settings)
    assert figure[:8] == b"\x89PNG\r\n\x1a\n"

    result = AnalysisResult(
        method="Comprehensive network analysis", summary="Network result",
        figures={"Network structure": figure},
        metadata={"interactive_network_html": "<!doctype html><html></html>", "diagram_settings": settings},
    )
    frame = pd.DataFrame({"x": [1, 2]})
    package = build_reproducibility_package(frame, frame, result, {"title": "Test"}, pd.DataFrame(), [])
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
        assert "Figures/Interactive_Network.html" in names
        assert "Figures/Path_Diagram_Arrangement.json" in names



def test_path_editor_component_runtime_fallback(tmp_path):
    from statready.path_editor_assets import resolve_component_path

    missing_asset = tmp_path / "missing_component_assets"
    fallback_root = tmp_path / "runtime_cache"
    resolved = resolve_component_path(packaged_path=missing_asset, cache_root=fallback_root)
    assert resolved.is_dir()
    assert (resolved / "index.html").is_file()
    assert "Interactive path editor" in (resolved / "index.html").read_text(encoding="utf-8")


def test_path_editor_packaged_asset_exists():
    from statready.path_editor_assets import resolve_component_path

    resolved = resolve_component_path()
    assert resolved.name == "path_editor_assets"
    assert (resolved / "index.html").is_file()


def test_all_runtime_dependencies_are_declared():
    import re
    from pathlib import Path

    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        "streamlit", "pandas", "numpy", "scipy", "statsmodels",
        "scikit-learn", "openpyxl", "xlrd", "python-docx", "plotly",
        "matplotlib", "pillow", "networkx", "openai", "pydantic",
    }
    assert required.issubset(declared), sorted(required - declared)


def test_matplotlib_headless_import():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.close(figure)


def test_phase25_framework_vision_reconciliation_without_api():
    from statready.framework_vision import (
        FrameworkVisionExtraction, VisionConstruct, VisionRelationship, reconcile_framework,
    )
    extraction = FrameworkVisionExtraction(
        diagram_summary="Digital competence predicts performance through motivation.",
        constructs=[
            VisionConstruct(name="Digital competence", indicators=["dc1", "dc2", "dc3"], role="predictor", measurement_mode="reflective"),
            VisionConstruct(name="Motivation", indicators=["mot1", "mot2"], role="mediator", measurement_mode="reflective"),
            VisionConstruct(name="Performance", indicators=["perf1", "perf2"], role="outcome", measurement_mode="reflective"),
        ],
        relationships=[
            VisionRelationship(predictor="Digital competence", outcome="Performance", relationship_type="mediation", mediator="Motivation", expected_sign="positive")
        ],
        confidence="high",
    )
    mapped, rows = reconcile_framework(extraction, ["dc1", "dc2", "dc3", "mot1", "mot2", "perf1", "perf2"])
    assert mapped["construct_map"]["Digital competence"] == ["dc1", "dc2", "dc3"]
    assert ("Digital competence", "Motivation") in mapped["paths"]
    assert any(row["status"] == "Matched" for row in rows)


def test_phase25_objective_specific_analysis_program_uses_framework():
    from statready.agent import build_analysis_program
    df = make_phase2_data()
    structured = {
        "construct_map": {
            "ConstructA": ["a1", "a2", "a3", "a4"],
            "ConstructB": ["b1", "b2", "b3", "b4"],
        },
        "measurement_modes": {"ConstructA": "reflective", "ConstructB": "reflective"},
        "paths": [("ConstructA", "ConstructB")],
        "structural_relations": [{"type": "Direct", "predictor": "ConstructA", "outcome": "ConstructB"}],
        "moderations": [],
    }
    study = {
        "objectives": "1. Examine the effect of ConstructA on ConstructB.\n2. Describe the study variables.",
        "hypotheses": "H1. ConstructA positively predicts ConstructB.",
        "outcome_type": "continuous",
        "group_count": 0,
        "paired": False,
        "alpha": 0.05,
        "framework_structured": structured,
    }
    programme = build_analysis_program(study, df, pd.DataFrame())
    assert len(programme.assignments) == 2
    assert programme.assignments[0].method_key == "sem"
    assert programme.assignments[0].specification.config["paths"] == [("ConstructA", "ConstructB")]
    assert set(programme.mapping_table["objective_id"]) == {"O1", "O2"}


def test_phase25_proposed_and_estimated_diagrams_are_attached():
    from statready.analysis_diagrams import proposed_diagram
    df = load_data()
    config = {
        "outcome": "performance",
        "predictors": ["training", "motivation", "support"],
        "alpha": 0.05,
        "diagram_settings": {"layout": "Left to right", "arrow_style": "Curved"},
    }
    preview = proposed_diagram("ols", config)
    assert preview[:8] == b"\x89PNG\r\n\x1a\n"
    result = run_analysis(df, "ols", config)
    assert "Proposed model before estimation" in result.figures
    assert "Estimated analysis model" in result.figures
    assert result.figures["Estimated analysis model"][:8] == b"\x89PNG\r\n\x1a\n"


def test_phase25_latent_analysis_exports_proposed_and_final_diagrams():
    df = make_phase2_data()
    config = {
        "construct_map": {"ConstructA": ["a1", "a2", "a3", "a4"], "ConstructB": ["b1", "b2", "b3", "b4"]},
        "paths": [("ConstructA", "ConstructB")],
        "structural_relations": [{"type": "Direct", "predictor": "ConstructA", "outcome": "ConstructB"}],
        "diagram_settings": {"layout": "Top to bottom", "arrow_style": "Curved", "show_indicators": True},
        "alpha": 0.05,
        "random_state": 42,
        "estimator": "ML",
    }
    result = run_analysis(df, "sem", config)
    assert "Proposed model before estimation" in result.figures
    assert "SEM path diagram" in result.figures


def test_phase26_render_model_routing():
    text = (Path(__file__).parents[1] / "render.yaml").read_text()
    assert "OPENAI_VISION_MODEL" in text and "gpt-5.4-nano" in text
    assert "OPENAI_VISION_FALLBACK_MODEL" in text and "gpt-5.6-luna" in text
    assert "AGENT_REASONING_MODEL" in text and "deepseek-v4-flash" in text
    assert "AGENT_REASONING_FALLBACK_MODEL" in text and "deepseek-v4-pro" in text
    assert "DEEPSEEK_API_KEY" in text


def test_phase26_framework_graph_validation():
    from statready.framework_vision import (
        FrameworkVisionExtraction,
        VisionConstruct,
        VisionRelationship,
        validate_framework_graph,
    )
    valid = FrameworkVisionExtraction(
        diagram_summary="Training predicts performance through motivation.",
        constructs=[
            VisionConstruct(name="Training"),
            VisionConstruct(name="Motivation", role="mediator"),
            VisionConstruct(name="Performance", role="outcome"),
        ],
        relationships=[
            VisionRelationship(
                predictor="Training",
                outcome="Performance",
                relationship_type="mediation",
                mediator="Motivation",
            )
        ],
        confidence="high",
    )
    assert validate_framework_graph(valid) == []
    invalid = FrameworkVisionExtraction(
        diagram_summary="Invalid graph.",
        constructs=[VisionConstruct(name="Training"), VisionConstruct(name="Training")],
        relationships=[VisionRelationship(predictor="Training", outcome="Unknown")],
        confidence="low",
    )
    issues = validate_framework_graph(invalid)
    assert any("Duplicate" in item for item in issues)
    assert any("Unknown outcome" in item for item in issues)


def test_phase26_provider_absence_uses_local_agent(monkeypatch):
    from statready.agent import build_analysis_program
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    df = load_data()
    study = {
        "objectives": "Examine the effect of training on performance.",
        "hypotheses": "Training significantly predicts performance.",
        "outcome_type": "continuous",
        "alpha": 0.05,
    }
    programme = build_analysis_program(study, df)
    assert programme.provider == "Deterministic local agent"
    assert programme.assignments
    assert any("DEEPSEEK_API_KEY" in warning for warning in programme.provider_warnings)


def test_phase26_validated_provider_plan_overlays_dataset_columns(monkeypatch):
    from statready import agent as agent_module
    from statready.ai_providers import (
        AgentAssignmentDraft,
        AgentProgrammeDraft,
        AgentProviderResult,
    )
    draft = AgentProgrammeDraft(
        programme_summary="One objective-specific regression.",
        assignments=[AgentAssignmentDraft(
            objective_id="O1",
            objective="Examine the effect of training and motivation on performance.",
            hypothesis_id="H1",
            hypothesis="Training and motivation predict performance.",
            method_key="ols",
            rationale="The outcome and predictors are continuous observed variables.",
            confidence="high",
            roles={"outcome": "performance", "predictors": ["training", "motivation"]},
            config_hints={"outcome": "performance", "predictors": ["training", "motivation", "invented_variable"]},
        )],
    )
    monkeypatch.setattr(
        agent_module,
        "generate_analysis_programme_with_provider",
        lambda *args, **kwargs: AgentProviderResult(
            draft=draft,
            provider="DeepSeek deepseek-v4-flash",
            models_attempted=["deepseek-v4-flash"],
        ),
    )
    df = load_data()
    study = {
        "objectives": "Examine the effect of training and motivation on performance.",
        "hypotheses": "Training and motivation predict performance.",
        "outcome_type": "continuous",
        "alpha": 0.05,
    }
    programme = agent_module.build_analysis_program(study, df, api_key="test-key")
    spec = programme.assignments[0].specification
    assert programme.provider == "DeepSeek deepseek-v4-flash"
    assert spec.config["outcome"] == "performance"
    assert spec.config["predictors"] == ["training", "motivation"]
    assert "invented_variable" not in spec.config["predictors"]
