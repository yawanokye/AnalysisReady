from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import KFold

from .diagnostics import vif_table
from .models import AnalysisResult, AuditEntry


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (values - mean) / sd


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 3 or np.std(a, ddof=1) <= 1e-12 or np.std(b, ddof=1) <= 1e-12:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _cronbach_alpha(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return np.nan
    item_var = np.var(x, axis=0, ddof=1)
    total_var = np.var(np.sum(x, axis=1), ddof=1)
    if total_var <= 1e-12:
        return np.nan
    k = x.shape[1]
    return float(k / (k - 1) * (1.0 - item_var.sum() / total_var))


def _normalise_weight(weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=float)
    norm = float(np.linalg.norm(weight))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.ones_like(weight) / math.sqrt(max(len(weight), 1))
    return weight / norm


def _validate_model(
    construct_map: dict[str, list[str]],
    paths: list[tuple[str, str]],
    moderations: list[dict] | None = None,
) -> None:
    constructs = set(construct_map)
    if len(constructs) < 2:
        raise ValueError("PLS-SEM requires at least two constructs.")
    seen: set[str] = set()
    for construct, items in construct_map.items():
        if not str(construct).strip():
            raise ValueError("Every construct requires a name.")
        if len(items) < 2:
            raise ValueError(f"Construct '{construct}' requires at least two indicators.")
        overlap = seen.intersection(items)
        if overlap:
            raise ValueError(f"Indicators may be assigned to only one construct: {', '.join(sorted(overlap))}.")
        seen.update(items)
    for source, target in paths:
        if source not in constructs or target not in constructs:
            raise ValueError(f"Unknown structural path: {source} -> {target}.")
        if source == target:
            raise ValueError("A construct cannot predict itself.")
    for relation in moderations or []:
        required = [relation.get("predictor"), relation.get("moderator"), relation.get("outcome")]
        if any(value not in constructs for value in required):
            raise ValueError("Every moderation relationship must use entered constructs.")
        if len(set(required)) < 3:
            raise ValueError("Predictor, moderator and outcome must be different constructs.")


@dataclass
class _PLSEstimate:
    scores: pd.DataFrame
    outer_loadings: pd.DataFrame
    outer_weights: pd.DataFrame
    cross_loadings: pd.DataFrame
    paths: pd.DataFrame
    r_squared: pd.DataFrame
    effect_sizes: pd.DataFrame
    q_squared: pd.DataFrame
    latent_correlations: pd.DataFrame
    srmr: float
    d_uls: float
    iterations: int
    converged: bool


def _estimate_pls(
    data: pd.DataFrame,
    construct_map: dict[str, list[str]],
    paths: list[tuple[str, str]],
    measurement_modes: dict[str, str],
    moderations: list[dict],
    max_iter: int,
    tolerance: float,
    random_state: int,
    weighting_scheme: str = "Path",
) -> _PLSEstimate:
    constructs = list(construct_map)
    blocks = {name: _zscore(data[items].to_numpy(dtype=float)) for name, items in construct_map.items()}
    weights = {name: _normalise_weight(np.ones(len(items))) for name, items in construct_map.items()}
    scores = {name: _zscore(blocks[name] @ weights[name]).reshape(-1) for name in constructs}

    neighbours = {name: set() for name in constructs}
    predecessors = {name: [] for name in constructs}
    successors = {name: [] for name in constructs}
    for source, target in paths:
        neighbours[source].add(target)
        neighbours[target].add(source)
        if source not in predecessors[target]:
            predecessors[target].append(source)
        if target not in successors[source]:
            successors[source].append(target)
    weighting_scheme = str(weighting_scheme or "Path").strip().title()
    if weighting_scheme not in {"Path", "Centroid", "Factorial"}:
        raise ValueError("PLS weighting scheme must be Path, Centroid or Factorial.")

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        previous = {name: weights[name].copy() for name in constructs}
        inner: dict[str, np.ndarray] = {}
        for construct in constructs:
            linked = sorted(neighbours[construct])
            if not linked:
                inner[construct] = scores[construct]
                continue
            estimate = np.zeros(len(data), dtype=float)
            if weighting_scheme == "Centroid":
                for other in linked:
                    sign = 1.0 if _corr(scores[construct], scores[other]) >= 0 else -1.0
                    estimate += sign * scores[other]
            elif weighting_scheme == "Factorial":
                for other in linked:
                    estimate += _corr(scores[construct], scores[other]) * scores[other]
            else:
                # Path weighting uses regression coefficients for predecessor constructs
                # and correlations for constructs downstream from the focal construct.
                preds = predecessors[construct]
                if preds:
                    pred_matrix = np.column_stack([scores[name] for name in preds])
                    beta = np.linalg.pinv(pred_matrix) @ scores[construct]
                    for coefficient, other in zip(beta, preds):
                        estimate += float(coefficient) * scores[other]
                for other in successors[construct]:
                    estimate += _corr(scores[construct], scores[other]) * scores[other]
            if np.std(estimate, ddof=1) <= 1e-12:
                estimate = scores[construct]
            inner[construct] = _zscore(estimate).reshape(-1)

        for construct in constructs:
            x = blocks[construct]
            mode = str(measurement_modes.get(construct, "reflective")).lower()
            if mode.startswith("form"):
                candidate = np.linalg.pinv(x) @ inner[construct]
            else:
                candidate = np.asarray([_corr(x[:, idx], inner[construct]) for idx in range(x.shape[1])])
            weights[construct] = _normalise_weight(candidate)
            scores[construct] = _zscore(x @ weights[construct]).reshape(-1)

        maximum_change = max(float(np.max(np.abs(weights[name] - previous[name]))) for name in constructs)
        if maximum_change < tolerance:
            converged = True
            break

    score_frame = pd.DataFrame(scores, index=data.index)
    outer_loading_rows = []
    outer_weight_rows = []
    cross_rows = []
    for construct, items in construct_map.items():
        block = blocks[construct]
        for idx, item in enumerate(items):
            loading = _corr(block[:, idx], scores[construct])
            outer_loading_rows.append({
                "construct": construct,
                "item": item,
                "measurement_mode": measurement_modes.get(construct, "reflective"),
                "loading": loading,
                "standardized_loading": loading,
                "indicator_reliability": loading ** 2,
            })
            outer_weight_rows.append({
                "construct": construct,
                "item": item,
                "measurement_mode": measurement_modes.get(construct, "reflective"),
                "outer_weight": weights[construct][idx],
            })
            row = {"construct": construct, "item": item}
            for latent in constructs:
                row[latent] = _corr(block[:, idx], scores[latent])
            cross_rows.append(row)

    # Latent interaction terms use a two-stage product of standardised construct scores.
    interaction_names: list[str] = []
    moderation_lookup: dict[str, dict] = {}
    for relation in moderations:
        predictor = relation["predictor"]
        moderator = relation["moderator"]
        outcome = relation["outcome"]
        name = f"{predictor} × {moderator}"
        score_frame[name] = _zscore(score_frame[predictor].to_numpy() * score_frame[moderator].to_numpy()).reshape(-1)
        interaction_names.append(name)
        moderation_lookup[name] = {**relation, "outcome": outcome}

    path_rows: list[dict] = []
    r2_rows: list[dict] = []
    f2_rows: list[dict] = []
    q2_rows: list[dict] = []
    all_outcomes = sorted(set(target for _, target in paths) | {item["outcome"] for item in moderations}, key=constructs.index)
    for outcome in all_outcomes:
        predictor_names = list(dict.fromkeys([source for source, target in paths if target == outcome]))
        for interaction in interaction_names:
            relation = moderation_lookup[interaction]
            if relation["outcome"] == outcome:
                predictor_names.extend([relation["predictor"], relation["moderator"], interaction])
        predictor_names = list(dict.fromkeys(predictor_names))
        if not predictor_names:
            continue
        x = score_frame[predictor_names].astype(float)
        y = score_frame[outcome].astype(float)
        model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
        for predictor in predictor_names:
            path_rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "estimate": float(model.params[predictor]),
                "standardized_estimate": float(model.params[predictor]),
                "std_error": float(model.bse[predictor]),
                "t_statistic": float(model.tvalues[predictor]),
                "p_value": float(model.pvalues[predictor]),
                "ci_lower": float(model.conf_int().loc[predictor, 0]),
                "ci_upper": float(model.conf_int().loc[predictor, 1]),
                "relationship_type": "moderation interaction" if predictor in interaction_names else "direct/mediated path",
            })
        r2 = float(model.rsquared)
        r2_rows.append({
            "endogenous_construct": outcome,
            "r_squared": r2,
            "adjusted_r_squared": float(model.rsquared_adj),
            "predictors": ", ".join(predictor_names),
        })
        for predictor in predictor_names:
            reduced_names = [name for name in predictor_names if name != predictor]
            if reduced_names:
                reduced = sm.OLS(y, sm.add_constant(score_frame[reduced_names], has_constant="add")).fit()
                reduced_r2 = float(reduced.rsquared)
            else:
                reduced_r2 = 0.0
            f2 = (r2 - reduced_r2) / max(1.0 - r2, 1e-12)
            f2_rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "r_squared_included": r2,
                "r_squared_excluded": reduced_r2,
                "f_squared": float(max(f2, 0.0)),
            })

        # Five-fold predictive relevance screening on latent scores.
        splits = min(5, max(2, len(data) // 10))
        kfold = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        predictions = np.full(len(data), np.nan)
        x_values = score_frame[predictor_names].to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)
        for train, test in kfold.split(x_values):
            fitted = sm.OLS(y_values[train], sm.add_constant(x_values[train], has_constant="add")).fit()
            predictions[test] = fitted.predict(sm.add_constant(x_values[test], has_constant="add"))
        press = float(np.sum((y_values - predictions) ** 2))
        sso = float(np.sum((y_values - y_values.mean()) ** 2))
        q2 = 1.0 - press / max(sso, 1e-12)
        q2_rows.append({"endogenous_construct": outcome, "q_squared_predict": q2, "folds": splits})

    latent_corr = score_frame[constructs].corr()
    loading_matrix = np.zeros((sum(len(v) for v in construct_map.values()), len(constructs)))
    item_order: list[str] = []
    item_row = 0
    loading_lookup = {(row["construct"], row["item"]): row["loading"] for row in outer_loading_rows}
    for c_idx, construct in enumerate(constructs):
        for item in construct_map[construct]:
            item_order.append(item)
            loading_matrix[item_row, c_idx] = loading_lookup[(construct, item)]
            item_row += 1
    observed_corr = data[item_order].corr().to_numpy(dtype=float)
    implied = loading_matrix @ latent_corr.to_numpy(dtype=float) @ loading_matrix.T
    np.fill_diagonal(implied, 1.0)
    residual = observed_corr - implied
    upper = residual[np.triu_indices_from(residual, k=1)]
    srmr = float(np.sqrt(np.mean(upper ** 2))) if upper.size else 0.0
    d_uls = float(np.sum(upper ** 2))

    return _PLSEstimate(
        scores=score_frame,
        outer_loadings=pd.DataFrame(outer_loading_rows),
        outer_weights=pd.DataFrame(outer_weight_rows),
        cross_loadings=pd.DataFrame(cross_rows),
        paths=pd.DataFrame(path_rows),
        r_squared=pd.DataFrame(r2_rows),
        effect_sizes=pd.DataFrame(f2_rows),
        q_squared=pd.DataFrame(q2_rows),
        latent_correlations=latent_corr.reset_index(names="construct"),
        srmr=srmr,
        d_uls=d_uls,
        iterations=iterations,
        converged=converged,
    )


def _htmt(data: pd.DataFrame, construct_map: dict[str, list[str]]) -> pd.DataFrame:
    corr = data[[item for items in construct_map.values() for item in items]].corr().abs()
    rows = []
    constructs = list(construct_map)
    for i, first in enumerate(constructs):
        first_items = construct_map[first]
        mono_first = [corr.loc[a, b] for idx, a in enumerate(first_items) for b in first_items[idx + 1:]]
        for second in constructs[i + 1:]:
            second_items = construct_map[second]
            mono_second = [corr.loc[a, b] for idx, a in enumerate(second_items) for b in second_items[idx + 1:]]
            hetero = [corr.loc[a, b] for a in first_items for b in second_items]
            denominator = math.sqrt(max(np.mean(mono_first) if mono_first else np.nan, 0) * max(np.mean(mono_second) if mono_second else np.nan, 0))
            value = float(np.mean(hetero) / denominator) if denominator and np.isfinite(denominator) else np.nan
            rows.append({"construct_1": first, "construct_2": second, "htmt": value})
    return pd.DataFrame(rows)


def _measurement_quality(
    data: pd.DataFrame,
    construct_map: dict[str, list[str]],
    modes: dict[str, str],
    loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    vif_rows = []
    for construct, items in construct_map.items():
        mode = str(modes.get(construct, "reflective"))
        subset = loadings[loadings["construct"] == construct]
        lambdas = subset["loading"].to_numpy(dtype=float)
        alpha = _cronbach_alpha(_zscore(data[items].to_numpy(dtype=float)))
        residual = np.maximum(1.0 - lambdas ** 2, 0.0)
        cr = (np.sum(lambdas) ** 2) / max(np.sum(lambdas) ** 2 + residual.sum(), 1e-12)
        ave = float(np.mean(lambdas ** 2))
        # rho_A is reported as an approximation because a full consistent-PLS correction is outside this engine.
        rho_a = float(np.clip((np.sum(np.abs(lambdas)) ** 2) / max(np.sum(np.abs(lambdas)) ** 2 + residual.sum(), 1e-12), 0.0, 1.0))
        rows.append({
            "construct": construct,
            "measurement_mode": mode,
            "cronbach_alpha": alpha if mode.startswith("reflect") else np.nan,
            "rho_A_approx": rho_a if mode.startswith("reflect") else np.nan,
            "composite_reliability_rho_c": cr if mode.startswith("reflect") else np.nan,
            "average_variance_extracted": ave if mode.startswith("reflect") else np.nan,
            "sqrt_ave": math.sqrt(max(ave, 0)) if mode.startswith("reflect") else np.nan,
            "minimum_loading": float(np.min(np.abs(lambdas))),
            "maximum_loading": float(np.max(np.abs(lambdas))),
        })
        block_vif = vif_table(data[items]).copy()
        if not block_vif.empty:
            block_vif.insert(0, "construct", construct)
            vif_rows.extend(block_vif.to_dict("records"))
    quality = pd.DataFrame(rows)
    outer_vif = pd.DataFrame(vif_rows)

    corr = pd.DataFrame(index=list(construct_map), columns=list(construct_map), dtype=float)
    latent = pd.DataFrame({construct: _zscore(data[items].mean(axis=1).to_numpy()).reshape(-1) for construct, items in construct_map.items()})
    corr.loc[:, :] = latent.corr().to_numpy()
    for _, row in quality.iterrows():
        if pd.notna(row["sqrt_ave"]):
            corr.loc[row["construct"], row["construct"]] = row["sqrt_ave"]
    fornell = corr.reset_index(names="construct")
    return quality, outer_vif, fornell


def _inner_vif(scores: pd.DataFrame, paths: list[tuple[str, str]], moderations: list[dict]) -> pd.DataFrame:
    rows = []
    outcomes = sorted(set(target for _, target in paths) | {r["outcome"] for r in moderations})
    work = scores.copy()
    for relation in moderations:
        name = f"{relation['predictor']} × {relation['moderator']}"
        if name not in work:
            work[name] = _zscore(work[relation["predictor"]].to_numpy() * work[relation["moderator"]].to_numpy()).reshape(-1)
    for outcome in outcomes:
        predictors = [source for source, target in paths if target == outcome]
        for relation in moderations:
            if relation["outcome"] == outcome:
                predictors += [relation["predictor"], relation["moderator"], f"{relation['predictor']} × {relation['moderator']}"]
        predictors = list(dict.fromkeys(predictors))
        if not predictors:
            continue
        table = vif_table(work[predictors])
        if not table.empty:
            table.insert(0, "outcome", outcome)
            rows.extend(table.to_dict("records"))
    return pd.DataFrame(rows)


def partial_least_squares_sem(
    df: pd.DataFrame,
    construct_map: dict[str, list[str]],
    paths: list[tuple[str, str]],
    measurement_modes: dict[str, str] | None = None,
    moderations: list[dict] | None = None,
    structural_relations: list[dict] | None = None,
    bootstrap_samples: int = 500,
    max_iter: int = 300,
    tolerance: float = 1e-7,
    weighting_scheme: str = "Path",
    random_state: int = 42,
    alpha: float = 0.05,
) -> AnalysisResult:
    measurement_modes = measurement_modes or {name: "reflective" for name in construct_map}
    moderations = moderations or []
    structural_relations = structural_relations or []
    paths = list(dict.fromkeys(tuple(path) for path in paths))
    _validate_model(construct_map, paths, moderations)
    items = [item for block in construct_map.values() for item in block]
    data = df.loc[:, items].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(data) < max(30, len(items) * 2):
        raise ValueError("The complete sample is too small for a stable PLS-SEM screening model.")
    if any(float(data[item].var(ddof=0)) <= 1e-12 for item in items):
        raise ValueError("Constant indicators must be removed before PLS-SEM.")

    estimate = _estimate_pls(
        data, construct_map, paths, measurement_modes, moderations,
        max_iter, tolerance, random_state, weighting_scheme,
    )
    quality, outer_vif, fornell = _measurement_quality(data, construct_map, measurement_modes, estimate.outer_loadings)
    htmt = _htmt(data, construct_map)
    inner_vif = _inner_vif(estimate.scores, paths, moderations)

    rng = np.random.default_rng(random_state)
    boot_paths: dict[tuple[str, str], list[float]] = {}
    boot_loadings: dict[tuple[str, str], list[float]] = {}
    boot_weights: dict[tuple[str, str], list[float]] = {}
    mediation_relations = [relation for relation in structural_relations if relation.get("type") == "Mediator"]
    boot_indirect: dict[tuple[str, str, str], list[float]] = {}
    boot_total: dict[tuple[str, str, str], list[float]] = {}
    successes = 0
    for _ in range(int(bootstrap_samples)):
        indices = rng.integers(0, len(data), len(data))
        sample = data.iloc[indices].reset_index(drop=True)
        try:
            fitted = _estimate_pls(
                sample, construct_map, paths, measurement_modes, moderations,
                max_iter, tolerance, random_state, weighting_scheme,
            )
            successes += 1
            for _, row in fitted.paths.iterrows():
                boot_paths.setdefault((str(row["predictor"]), str(row["outcome"])), []).append(float(row["standardized_estimate"]))
            for _, row in fitted.outer_loadings.iterrows():
                boot_loadings.setdefault((str(row["construct"]), str(row["item"])), []).append(float(row["loading"]))
            for _, row in fitted.outer_weights.iterrows():
                boot_weights.setdefault((str(row["construct"]), str(row["item"])), []).append(float(row["outer_weight"]))
            fitted_path_map = {
                (str(row["predictor"]), str(row["outcome"])): float(row["standardized_estimate"])
                for _, row in fitted.paths.iterrows()
            }
            for relation in mediation_relations:
                predictor = str(relation.get("predictor"))
                mediator = str(relation.get("mediator"))
                outcome = str(relation.get("outcome"))
                a = fitted_path_map.get((predictor, mediator), np.nan)
                b = fitted_path_map.get((mediator, outcome), np.nan)
                direct = fitted_path_map.get((predictor, outcome), 0.0) if relation.get("include_direct", True) else 0.0
                if np.isfinite(a) and np.isfinite(b):
                    key = (predictor, mediator, outcome)
                    indirect = float(a * b)
                    boot_indirect.setdefault(key, []).append(indirect)
                    boot_total.setdefault(key, []).append(float(direct + indirect) if np.isfinite(direct) else np.nan)
        except Exception:
            continue

    path_table = estimate.paths.copy()
    if not path_table.empty:
        for idx, row in path_table.iterrows():
            values = np.asarray(boot_paths.get((str(row["predictor"]), str(row["outcome"])), []), dtype=float)
            if len(values) >= 20:
                se = float(np.std(values, ddof=1))
                t_value = float(row["standardized_estimate"] / se) if se > 0 else np.nan
                p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
                lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            else:
                se = t_value = p_value = lower = upper = np.nan
            path_table.loc[idx, ["bootstrap_std_error", "bootstrap_t", "bootstrap_p", "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_draws"]] = [se, t_value, p_value, lower, upper, len(values)]

    loading_table = estimate.outer_loadings.copy()
    if not loading_table.empty:
        for idx, row in loading_table.iterrows():
            values = np.asarray(boot_loadings.get((str(row["construct"]), str(row["item"])), []), dtype=float)
            if len(values) >= 20:
                se = float(np.std(values, ddof=1))
                t_value = float(row["loading"] / se) if se > 0 else np.nan
                p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
                lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            else:
                se = t_value = p_value = lower = upper = np.nan
            loading_table.loc[idx, ["bootstrap_std_error", "bootstrap_t", "bootstrap_p", "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_draws"]] = [se, t_value, p_value, lower, upper, len(values)]

    weight_table = estimate.outer_weights.copy()
    if not weight_table.empty:
        for idx, row in weight_table.iterrows():
            values = np.asarray(boot_weights.get((str(row["construct"]), str(row["item"])), []), dtype=float)
            if len(values) >= 20:
                se = float(np.std(values, ddof=1))
                t_value = float(row["outer_weight"] / se) if se > 0 else np.nan
                p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
                lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            else:
                se = t_value = p_value = lower = upper = np.nan
            weight_table.loc[idx, ["bootstrap_std_error", "bootstrap_t", "bootstrap_p", "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_draws"]] = [se, t_value, p_value, lower, upper, len(values)]

    base_path_map = {
        (str(row["predictor"]), str(row["outcome"])): float(row["standardized_estimate"])
        for _, row in estimate.paths.iterrows()
    }
    mediation_rows: list[dict] = []
    for relation in mediation_relations:
        predictor = str(relation.get("predictor"))
        mediator = str(relation.get("mediator"))
        outcome = str(relation.get("outcome"))
        key = (predictor, mediator, outcome)
        a = base_path_map.get((predictor, mediator), np.nan)
        b = base_path_map.get((mediator, outcome), np.nan)
        direct = base_path_map.get((predictor, outcome), 0.0) if relation.get("include_direct", True) else 0.0
        indirect = float(a * b) if np.isfinite(a) and np.isfinite(b) else np.nan
        indirect_values = np.asarray(boot_indirect.get(key, []), dtype=float)
        total_values = np.asarray(boot_total.get(key, []), dtype=float)
        if len(indirect_values) >= 20:
            indirect_se = float(np.std(indirect_values, ddof=1))
            indirect_t = indirect / indirect_se if indirect_se > 0 and np.isfinite(indirect) else np.nan
            indirect_p = float(2 * stats.norm.sf(abs(indirect_t))) if np.isfinite(indirect_t) else np.nan
            indirect_lower, indirect_upper = np.percentile(indirect_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            indirect_se = indirect_t = indirect_p = indirect_lower = indirect_upper = np.nan
        total = direct + indirect if np.isfinite(direct) and np.isfinite(indirect) else np.nan
        if len(total_values) >= 20:
            total_lower, total_upper = np.percentile(total_values[np.isfinite(total_values)], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            total_lower = total_upper = np.nan
        mediation_rows.append({
            "predictor": predictor, "mediator": mediator, "outcome": outcome,
            "a_path": a, "b_path": b, "direct_effect": direct,
            "indirect_effect": indirect, "indirect_bootstrap_std_error": indirect_se,
            "indirect_bootstrap_t": indirect_t, "indirect_bootstrap_p": indirect_p,
            "indirect_ci_lower": indirect_lower, "indirect_ci_upper": indirect_upper,
            "total_effect": total, "total_ci_lower": total_lower, "total_ci_upper": total_upper,
            "bootstrap_draws": len(indirect_values),
        })
    mediation_table = pd.DataFrame(mediation_rows)

    # Diagnostics are deliberately explicit and avoid automatic item deletion.
    diagnostics: list[dict] = [{
        "diagnostic": "PLS algorithm convergence",
        "test": f"{weighting_scheme} inner weighting, maximum outer-weight change",
        "statistic": float(estimate.iterations),
        "p_value": np.nan,
        "status": "Satisfied" if estimate.converged else "Material concern",
        "interpretation": f"The algorithm used {estimate.iterations} iteration(s) with tolerance {tolerance:g}.",
        "recommended_response": "Increase the iteration limit, inspect indicator collinearity and rescale problematic blocks when the algorithm does not converge.",
    }]
    reflective = quality[quality["measurement_mode"].astype(str).str.startswith("reflect")]
    minimum_loading = float(reflective["minimum_loading"].min()) if not reflective.empty else np.nan
    diagnostics.append({
        "diagnostic": "Reflective indicator reliability",
        "test": "Minimum absolute outer loading",
        "statistic": minimum_loading,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(minimum_loading) else "Satisfied" if minimum_loading >= 0.708 else "Minor concern" if minimum_loading >= 0.40 else "Material concern",
        "interpretation": "Reflective indicators are assessed by outer loadings and indicator reliability.",
        "recommended_response": "Review weak indicators against content validity. Do not delete an item automatically or solely to improve statistics.",
    })
    minimum_cr = float(reflective["composite_reliability_rho_c"].min()) if not reflective.empty else np.nan
    diagnostics.append({
        "diagnostic": "Internal consistency reliability",
        "test": "Minimum composite reliability (rho_c)",
        "statistic": minimum_cr,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(minimum_cr) else "Satisfied" if 0.70 <= minimum_cr <= 0.95 else "Minor concern" if 0.60 <= minimum_cr < 0.70 else "Material concern",
        "interpretation": "Very low reliability weakens measurement, while values above 0.95 can indicate redundant indicators.",
        "recommended_response": "Review construct breadth, item wording and redundancy. Preserve theoretically necessary content.",
    })
    minimum_ave = float(reflective["average_variance_extracted"].min()) if not reflective.empty else np.nan
    diagnostics.append({
        "diagnostic": "Convergent validity",
        "test": "Minimum AVE",
        "statistic": minimum_ave,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(minimum_ave) else "Satisfied" if minimum_ave >= 0.50 else "Material concern",
        "interpretation": "AVE summarises the variance captured by reflective constructs relative to indicator error.",
        "recommended_response": "Review weak loadings and construct specification. Any item removal requires theoretical justification and a sensitivity comparison.",
    })
    max_htmt = float(htmt["htmt"].max()) if not htmt.empty else np.nan
    diagnostics.append({
        "diagnostic": "Discriminant validity",
        "test": "Maximum HTMT",
        "statistic": max_htmt,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(max_htmt) else "Satisfied" if max_htmt < 0.85 else "Minor concern" if max_htmt < 0.90 else "Material concern",
        "interpretation": "High HTMT suggests that constructs may not be empirically distinct.",
        "recommended_response": "Revisit conceptual distinctiveness, item overlap and cross-loadings. Do not merge constructs only to improve a threshold.",
    })
    max_outer_vif = float(outer_vif["vif"].replace([np.inf, -np.inf], np.nan).max()) if not outer_vif.empty else np.nan
    diagnostics.append({
        "diagnostic": "Measurement-model collinearity",
        "test": "Maximum outer VIF",
        "statistic": max_outer_vif,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(max_outer_vif) else "Satisfied" if max_outer_vif < 3.3 else "Minor concern" if max_outer_vif < 5 else "Material concern",
        "interpretation": "Outer VIF is especially important for formative blocks and can also reveal redundant reflective indicators.",
        "recommended_response": "Review overlapping indicators and content coverage. Consider a defensible composite or item revision rather than significance-driven deletion.",
    })
    max_inner_vif = float(inner_vif["vif"].replace([np.inf, -np.inf], np.nan).max()) if not inner_vif.empty else np.nan
    diagnostics.append({
        "diagnostic": "Structural-model collinearity",
        "test": "Maximum inner VIF",
        "statistic": max_inner_vif,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(max_inner_vif) else "Satisfied" if max_inner_vif < 3.3 else "Minor concern" if max_inner_vif < 5 else "Material concern",
        "interpretation": "High inner VIF makes individual path effects unstable.",
        "recommended_response": "Inspect construct overlap and report sensitivity models. Do not interpret highly collinear paths as isolated effects.",
    })
    formative_names = [name for name, mode in measurement_modes.items() if str(mode).lower().startswith("form")]
    formative_weights = weight_table[weight_table["construct"].isin(formative_names)] if formative_names and not weight_table.empty else pd.DataFrame()
    if not formative_weights.empty and "bootstrap_ci_lower" in formative_weights:
        significant = ((formative_weights["bootstrap_ci_lower"] > 0) | (formative_weights["bootstrap_ci_upper"] < 0)).sum()
        proportion_significant = float(significant / len(formative_weights))
    else:
        proportion_significant = np.nan
    diagnostics.append({
        "diagnostic": "Formative indicator contribution",
        "test": "Proportion of formative outer-weight bootstrap intervals excluding zero",
        "statistic": proportion_significant,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(proportion_significant) else "Satisfied" if proportion_significant >= 0.80 else "Minor concern" if proportion_significant >= 0.50 else "Material concern",
        "interpretation": "A non-significant formative weight does not automatically make an indicator irrelevant; its loading and content contribution must also be considered.",
        "recommended_response": "Retain content-essential indicators where justified, assess outer loadings and collinearity, and report any alternative formative specification transparently.",
    })

    minimum_q2 = float(estimate.q_squared["q_squared_predict"].min()) if not estimate.q_squared.empty else np.nan
    diagnostics.append({
        "diagnostic": "Predictive relevance",
        "test": "Minimum cross-validated Q²_predict",
        "statistic": minimum_q2,
        "p_value": np.nan,
        "status": "Cannot determine" if not np.isfinite(minimum_q2) else "Satisfied" if minimum_q2 > 0 else "Material concern",
        "interpretation": "Values above zero indicate better prediction than the mean benchmark for the latent outcome score.",
        "recommended_response": "When Q² is weak, reassess theoretical predictors and validate prediction on independent data.",
    })
    diagnostics.append({
        "diagnostic": "Approximate composite-model residual fit",
        "test": "SRMR",
        "statistic": estimate.srmr,
        "p_value": np.nan,
        "status": "Satisfied" if estimate.srmr <= 0.08 else "Minor concern" if estimate.srmr <= 0.10 else "Material concern",
        "interpretation": "SRMR is a descriptive residual-fit screen and should not replace measurement and predictive assessment.",
        "recommended_response": "Review residual patterns, construct validity and predictive performance rather than optimising SRMR alone.",
    })
    bootstrap_rate = successes / max(int(bootstrap_samples), 1)
    diagnostics.append({
        "diagnostic": "Bootstrap stability",
        "test": "Successful bootstrap replications",
        "statistic": bootstrap_rate,
        "p_value": np.nan,
        "status": "Satisfied" if bootstrap_rate >= 0.95 else "Minor concern" if bootstrap_rate >= 0.80 else "Material concern",
        "interpretation": f"{successes} of {int(bootstrap_samples)} bootstrap models were successfully estimated.",
        "recommended_response": "Investigate unstable indicators, collinearity, small samples or complex interaction specifications when replication failures are frequent.",
    })

    fit_table = pd.DataFrame([{
        "n": len(data),
        "constructs": len(construct_map),
        "indicators": len(items),
        "structural_paths": len(paths),
        "moderation_effects": len(moderations),
        "weighting_scheme": weighting_scheme,
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "srmr_approx": estimate.srmr,
        "d_uls_approx": estimate.d_uls,
        "bootstrap_requested": int(bootstrap_samples),
        "bootstrap_successful": successes,
    }])

    warnings = [
        "PLS-SEM is a composite-based estimator. Use covariance-based SEM when the primary aim is strict common-factor model confirmation and the assumptions are appropriate.",
        "rho_A and residual-fit statistics are approximate in this internal engine. Confirm publication-critical models in specialist PLS-SEM software.",
    ]
    if not estimate.converged:
        warnings.append("The PLS outer-weight iteration did not meet the requested tolerance. Treat estimates as provisional.")

    return AnalysisResult(
        method="Partial least squares structural equation model",
        summary=(
            f"PLS-SEM estimated {len(paths)} directed path(s) and {len(moderations)} moderation effect(s) among "
            f"{len(construct_map)} constructs using {len(data)} complete observations. The {weighting_scheme.lower()} weighting algorithm converged={estimate.converged}; "
            f"approximate SRMR={estimate.srmr:.3f}. Review loadings or weights, reliability, AVE, HTMT, VIF, R², f², Q² and bootstrap intervals together."
        ),
        tables={
            "PLS-SEM model summary": fit_table,
            "PLS structural path estimates": path_table,
            "PLS outer loadings": loading_table,
            "PLS outer weights": weight_table,
            "Construct reliability and convergent validity": quality,
            "HTMT discriminant validity": htmt,
            "Fornell-Larcker matrix": fornell,
            "Cross-loadings": estimate.cross_loadings,
            "Outer VIF": outer_vif,
            "Inner VIF": inner_vif,
            "Endogenous construct R squared": estimate.r_squared,
            "Structural effect sizes f squared": estimate.effect_sizes,
            "Predictive relevance Q squared": estimate.q_squared,
            "Latent construct correlations": estimate.latent_correlations,
            "Latent variable scores": estimate.scores.reset_index(drop=True),
            **({"Specified mediation effects": mediation_table} if not mediation_table.empty else {}),
        },
        diagnostics=pd.DataFrame(diagnostics),
        metadata={
            "construct_map": construct_map,
            "measurement_modes": measurement_modes,
            "paths": paths,
            "moderations": moderations,
            "n": len(data),
            "estimator": f"Iterative PLS path modelling ({weighting_scheme} inner weighting) with Mode A reflective and Mode B formative blocks",
            "weighting_scheme": weighting_scheme,
            "bootstrap_samples": int(bootstrap_samples),
        },
        warnings=warnings,
        treatment_log=[AuditEntry(
            action="Estimated prespecified PLS-SEM model",
            variable=", ".join(construct_map),
            details=f"Estimated {len(paths)} structural paths, {len(moderations)} latent-score interaction(s), and {len(items)} indicators.",
            justification="The composite model was estimated exactly as specified. No indicator or path was automatically removed to improve significance, reliability or fit.",
            before_n=len(data),
            after_n=len(data),
        )],
        reproducible_code=(
            f"# Standardise indicators; estimate Mode A/Mode B PLS outer weights using {weighting_scheme} inner weighting; estimate structural regressions; "
            "bootstrap loadings and paths; assess reliability, AVE, HTMT, VIF, R2, f2, Q2 and residual fit.\n"
        ),
    )
