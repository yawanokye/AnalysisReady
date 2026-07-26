from __future__ import annotations

import re


METHOD_LABELS = {
    "descriptive": "Descriptive statistics",
    "reliability": "Reliability analysis (Cronbach's alpha)",
    "correlation": "Pearson or Spearman correlation",
    "independent_t": "Independent-samples t-test",
    "paired_t": "Paired-samples t-test",
    "anova": "One-way ANOVA or Welch ANOVA",
    "chi_square": "Chi-square test of independence",
    "ols": "Ordinary least squares regression",
    "logistic": "Binary logistic regression",
    "moderation": "Moderated multiple regression",
    "mediation": "Bootstrap mediation analysis",
}


def recommend_method(
    objective: str,
    hypothesis: str = "",
    outcome_type: str = "continuous",
    group_count: int | None = None,
    paired: bool = False,
) -> dict[str, str]:
    text = f"{objective} {hypothesis}".lower()

    if re.search(r"mediat|indirect effect", text):
        key = "mediation"
        reason = "The wording specifies an indirect or mediating relationship."
    elif re.search(r"moderat|interaction effect", text):
        key = "moderation"
        reason = "The wording specifies that a relationship changes across levels of another variable."
    elif paired or re.search(r"before and after|pre.?test|post.?test|paired|same participants", text):
        key = "paired_t"
        reason = "The same units appear to be measured twice or under two conditions."
    elif re.search(r"difference|compare|variation among|group differences", text):
        if group_count and group_count > 2:
            key = "anova"
            reason = "The objective compares a continuous outcome across more than two groups."
        else:
            key = "independent_t"
            reason = "The objective compares a continuous outcome across two independent groups."
    elif re.search(r"association|relationship|correlat", text) and outcome_type == "categorical":
        key = "chi_square"
        reason = "The objective examines association between categorical variables."
    elif re.search(r"association|relationship|correlat", text):
        key = "correlation"
        reason = "The objective examines the strength and direction of association."
    elif re.search(r"effect|influence|predict|determinant|impact", text):
        if outcome_type == "binary":
            key = "logistic"
            reason = "The objective estimates effects on a binary outcome."
        else:
            key = "ols"
            reason = "The objective estimates the effect of predictors on a continuous outcome."
    elif re.search(r"reliab|internal consistency|scale", text):
        key = "reliability"
        reason = "The objective concerns internal consistency of a multi-item scale."
    else:
        key = "descriptive"
        reason = "The wording does not clearly imply an inferential model, so descriptive analysis is the safest starting point."

    return {"method_key": key, "method": METHOD_LABELS[key], "reason": reason}
