from pathlib import Path

import pandas as pd

from statready.agent import build_guided_review
from statready.figures import render_latent_path_diagram
from statready.phase2 import structural_equation_model


BASE = Path(__file__).resolve().parent
OUT = BASE / "demo_output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    data = pd.read_csv(BASE / "sample_data" / "phase2_factor_sample.csv")
    construct_map = {
        "Digital Competence": ["a1", "a2", "a3", "a4"],
        "Teaching Effectiveness": ["b1", "b2", "b3", "b4"],
        "Motivation": ["m1", "m2"],
    }
    paths = [
        ("Digital Competence", "Motivation"),
        ("Motivation", "Teaching Effectiveness"),
        ("Digital Competence", "Teaching Effectiveness"),
    ]
    relations = [{
        "type": "Mediator",
        "predictor": "Digital Competence",
        "mediator": "Motivation",
        "outcome": "Teaching Effectiveness",
        "include_direct": True,
    }]
    result = structural_equation_model(data, construct_map, paths, estimator="ML")

    layouts = {
        "Left_to_Right": {
            "layout": "Left to right", "arrow_style": "Curved",
            "show_indicators": True, "show_loadings": True,
            "show_coefficients": True, "show_p_values": True, "show_fit": True,
        },
        "Top_to_Bottom": {
            "layout": "Top to bottom", "arrow_style": "Straight",
            "show_indicators": False, "show_loadings": False,
            "show_coefficients": True, "show_p_values": False, "show_fit": True,
        },
        "Radial": {
            "layout": "Radial", "arrow_style": "Curved",
            "show_indicators": False, "show_loadings": False,
            "show_coefficients": True, "show_p_values": True, "show_fit": True,
        },
        "Compact_Monochrome": {
            "layout": "Compact publication", "arrow_style": "Straight",
            "show_indicators": False, "show_loadings": False,
            "show_coefficients": True, "show_p_values": True, "show_fit": True,
            "monochrome": True,
        },
    }
    for suffix, settings in layouts.items():
        image = render_latent_path_diagram(
            construct_map=construct_map,
            loading_table=result.tables["SEM standardised loadings"],
            paths=paths,
            path_table=result.tables["Structural path estimates"],
            fit_table=result.tables["SEM fit indices"],
            title=f"StatReady AI Phase 2.3, {settings['layout']}",
            settings=settings,
            structural_relations=relations,
        )
        (OUT / f"StatReady_Phase2_3_{suffix}.png").write_bytes(image)

    study = {
        "objective": "Examine the structural effect of digital competence on teaching effectiveness through motivation.",
        "hypothesis": "Digital competence positively predicts teaching effectiveness directly and indirectly through motivation.",
        "outcome_type": "continuous",
        "group_count": 0,
        "paired": False,
        "framework_notes": "Digital competence predicts motivation and teaching effectiveness. Motivation mediates the relationship.",
    }
    framework = pd.DataFrame({"variable": data.columns, "role": ["Scale item"] * len(data.columns)})
    review = build_guided_review(study, data, framework)
    lines = [
        "# StatReady AI Guided Review Demonstration", "",
        f"**Recommended method:** {review.method_label}", "", review.reason, "",
        "## Readiness", "",
    ]
    for _, row in review.readiness.iterrows():
        lines.append(f"- **{row['component']}:** {row['status']}. {row['guidance']}")
    lines += ["", "## Suggested construct blocks", ""]
    for item in review.construct_suggestions:
        lines.append(f"- **{item['name']}:** {', '.join(item['items'])}. {item['rationale']}")
    (OUT / "StatReady_Phase2_3_AI_Guided_Review.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
