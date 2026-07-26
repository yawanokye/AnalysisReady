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
    "efa": "Exploratory factor analysis",
    "cfa": "Confirmatory factor analysis",
    "sem": "Covariance-based structural equation model",
    "pls_sem": "Partial least squares structural equation model (PLS-SEM)",
    "repeated_measures": "Repeated-measures ANOVA",
    "multilevel": "Multilevel mixed model or robust GEE",
    "panel": "Panel-data model selection",
    "advanced_moderation": "Advanced moderation with simple slopes",
    "parallel_mediation": "Parallel multiple mediation",
    "moderated_mediation": "First-stage moderated mediation",
}


def recommend_method(
    objective: str,
    hypothesis: str = "",
    outcome_type: str = "continuous",
    group_count: int | None = None,
    paired: bool = False,
) -> dict[str, str]:
    text = f"{objective} {hypothesis}".lower()

    if re.search(r"pls.?sem|partial least squares|composite structural", text):
        key = "pls_sem"
        reason = "The wording specifies a composite-based PLS structural equation model with predictive and latent-variable assessment."
    elif re.search(r"structural equation|\bsem\b|latent path|latent variable model", text):
        key = "sem"
        reason = "The wording specifies a latent-variable structural model with linked measurement and structural relationships."
    elif re.search(r"confirmatory factor|\bcfa\b|measurement model|construct validity", text):
        key = "cfa"
        reason = "The objective concerns confirmation of a prespecified latent measurement structure."
    elif re.search(r"exploratory factor|\befa\b|factor structure|dimension reduction|underlying dimensions", text):
        key = "efa"
        reason = "The objective seeks to discover the underlying dimensional structure of a multi-item set."
    elif re.search(r"panel data|fixed effects|random effects|longitudinal firms|entity and time", text):
        key = "panel"
        reason = "The data appear to contain repeated entity observations across time."
    elif re.search(r"mixed effects|multilevel|hierarchical|nested|clustered observations|random intercept|random slope", text):
        key = "multilevel"
        reason = "The objective involves observations nested within clusters or repeated within units."
    elif re.search(r"repeated measures|within-subject|multiple time points|three time points", text):
        key = "repeated_measures"
        reason = "The same units are measured across more than two conditions or occasions."
    elif re.search(r"moderated mediation|conditional indirect|index of moderated mediation", text):
        key = "moderated_mediation"
        reason = "The objective specifies that an indirect effect changes across levels of a moderator."
    elif re.search(r"parallel mediation|multiple mediators|mediators", text) and re.search(r"mediat|indirect effect", text):
        key = "parallel_mediation"
        reason = "The objective specifies more than one mediator operating in parallel."
    elif re.search(r"johnson.?neyman|simple slopes|conditional effect", text):
        key = "advanced_moderation"
        reason = "The objective requires conditional effects and probing of an interaction."
    elif re.search(r"mediat|indirect effect", text):
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
