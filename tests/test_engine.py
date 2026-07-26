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
