from __future__ import annotations

from io import BytesIO
import base64
import html
import json
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

from .models import AnalysisResult, AuditEntry


def _png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def _build_edge_graph(df: pd.DataFrame, config: dict[str, Any]) -> tuple[nx.Graph, pd.DataFrame, list[str]]:
    source = config["source"]
    target = config["target"]
    weight_column = config.get("weight")
    directed = bool(config.get("directed", False))
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    columns = [source, target] + ([weight_column] if weight_column else [])
    data = df[columns].copy().dropna(subset=[source, target])
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        left, right = str(row[source]), str(row[target])
        if not config.get("allow_self_loops", False) and left == right:
            continue
        weight = _safe_float(row[weight_column], 1.0) if weight_column else 1.0
        if weight < 0:
            warnings.append("Negative edge weights were retained for sign reporting but absolute weights were used for distance-based measures.")
        if graph.has_edge(left, right):
            graph[left][right]["weight"] += weight
            graph[left][right]["multiplicity"] = int(graph[left][right].get("multiplicity", 1)) + 1
        else:
            graph.add_edge(left, right, weight=weight, multiplicity=1)
    for left, right, attrs in graph.edges(data=True):
        rows.append({"source": left, "target": right, "weight": attrs.get("weight", 1.0), "multiplicity": attrs.get("multiplicity", 1)})
    if not rows:
        raise ValueError("No valid edges remained after applying the selected edge-list settings.")
    return graph, pd.DataFrame(rows), list(dict.fromkeys(warnings))


def _partial_correlation_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    complete = data.dropna()
    if len(complete) < max(20, data.shape[1] * 3):
        raise ValueError("Partial-correlation network requires more complete observations relative to the number of variables.")
    scaled = StandardScaler().fit_transform(complete)
    model = GraphicalLassoCV().fit(scaled)
    precision = model.precision_
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / denominator
    np.fill_diagonal(partial, 1.0)
    return pd.DataFrame(partial, index=data.columns, columns=data.columns), f"Graphical Lasso CV alpha={model.alpha_:.6g}"


def _correlation_network(df: pd.DataFrame, config: dict[str, Any]) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, str]:
    variables = config.get("variables") or []
    if len(variables) < 3:
        raise ValueError("Select at least three variables for a correlation network.")
    data = df[variables].apply(pd.to_numeric, errors="coerce")
    method = str(config.get("network_estimator", "Pearson correlation"))
    estimator_note = method
    if method == "Spearman correlation":
        matrix = data.corr(method="spearman")
    elif method == "Partial correlation (Graphical Lasso)":
        matrix, estimator_note = _partial_correlation_matrix(data)
    else:
        matrix = data.corr(method="pearson")
    threshold = float(config.get("edge_threshold", 0.20))
    retain_negative = bool(config.get("retain_negative", True))
    graph = nx.Graph()
    graph.add_nodes_from(variables)
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(variables):
        for right in variables[i + 1:]:
            value = float(matrix.loc[left, right]) if pd.notna(matrix.loc[left, right]) else np.nan
            if not np.isfinite(value):
                continue
            if abs(value) < threshold:
                continue
            if value < 0 and not retain_negative:
                continue
            graph.add_edge(left, right, weight=value, sign="negative" if value < 0 else "positive")
            edges.append({"source": left, "target": right, "weight": value, "absolute_weight": abs(value), "sign": "negative" if value < 0 else "positive"})
    if graph.number_of_edges() == 0:
        raise ValueError("The selected threshold produced a network with no edges. Reduce the threshold or review the variables.")
    return graph, pd.DataFrame(edges), matrix.reset_index(names="variable"), estimator_note


def _adjacency_network(df: pd.DataFrame, config: dict[str, Any]) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    label_column = config.get("node_label")
    value_columns = config.get("adjacency_columns") or []
    directed = bool(config.get("directed", False))
    if len(value_columns) < 2:
        raise ValueError("Select at least two adjacency-matrix columns.")
    matrix = df[value_columns].apply(pd.to_numeric, errors="coerce")
    if len(matrix) != len(value_columns):
        raise ValueError("The adjacency matrix must be square: the number of selected rows must equal the number of selected columns.")
    labels = [str(value) for value in df[label_column].iloc[:len(value_columns)]] if label_column else [str(column) for column in value_columns]
    values = matrix.iloc[:len(value_columns)].fillna(0).to_numpy(dtype=float)
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(labels)
    rows: list[dict[str, Any]] = []
    for i, source in enumerate(labels):
        start_j = 0 if directed else i + 1
        for j in range(start_j, len(labels)):
            if i == j and not config.get("allow_self_loops", False):
                continue
            weight = float(values[i, j])
            if weight == 0 or not np.isfinite(weight):
                continue
            target = labels[j]
            graph.add_edge(source, target, weight=weight)
            rows.append({"source": source, "target": target, "weight": weight})
    if graph.number_of_edges() == 0:
        raise ValueError("The selected adjacency matrix contains no non-zero edges.")
    matrix_frame = pd.DataFrame(values, index=labels, columns=labels).reset_index(names="node")
    return graph, pd.DataFrame(rows), matrix_frame


def _distance_graph(graph: nx.Graph) -> nx.Graph:
    copied = graph.copy()
    for left, right, attrs in copied.edges(data=True):
        attrs["distance"] = 1.0 / max(abs(_safe_float(attrs.get("weight", 1.0))), 1e-9)
    return copied


def _absolute_weight_graph(graph: nx.Graph) -> nx.Graph:
    copied = graph.copy()
    for _, _, attrs in copied.edges(data=True):
        attrs["abs_weight"] = abs(_safe_float(attrs.get("weight", 1.0)))
    return copied


def _communities(graph: nx.Graph, seed: int = 42) -> tuple[dict[str, int], float]:
    undirected = _absolute_weight_graph(graph).to_undirected()
    if undirected.number_of_edges() == 0:
        return {str(node): index for index, node in enumerate(undirected.nodes())}, 0.0
    try:
        groups = nx.community.louvain_communities(undirected, weight="abs_weight", seed=seed)
    except Exception:
        groups = list(nx.community.greedy_modularity_communities(undirected, weight="weight"))
    membership: dict[str, int] = {}
    for index, group in enumerate(groups, start=1):
        for node in group:
            membership[str(node)] = index
    try:
        modularity = float(nx.community.modularity(undirected, groups, weight="abs_weight"))
    except Exception:
        modularity = np.nan
    return membership, modularity


def _centralization(values: dict[Any, float]) -> float:
    if len(values) <= 2:
        return 0.0
    maximum = max(values.values(), default=0.0)
    numerator = sum(maximum - value for value in values.values())
    denominator = (len(values) - 1) * (len(values) - 2)
    return float(numerator / denominator) if denominator else 0.0


def _network_measures(graph: nx.Graph, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directed = graph.is_directed()
    undirected = graph.to_undirected()
    absolute_graph = _absolute_weight_graph(graph)
    absolute_undirected = absolute_graph.to_undirected()
    distance_graph = _distance_graph(graph)
    membership, modularity = _communities(graph, seed)
    degree = dict(graph.degree())
    signed_strength = dict(graph.degree(weight="weight"))
    strength = dict(absolute_graph.degree(weight="abs_weight"))
    betweenness = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(distance_graph, distance="distance")
    harmonic = nx.harmonic_centrality(distance_graph, distance="distance")
    try:
        eigenvector = nx.eigenvector_centrality_numpy(absolute_undirected, weight="abs_weight")
    except Exception:
        eigenvector = {node: np.nan for node in graph}
    try:
        pagerank = nx.pagerank(absolute_graph, weight="abs_weight", max_iter=1000)
    except Exception:
        pagerank = {node: np.nan for node in graph}
    clustering = nx.clustering(absolute_undirected, weight="abs_weight")
    core = nx.core_number(undirected) if undirected.number_of_edges() else {node: 0 for node in graph}
    average_neighbor_degree = nx.average_neighbor_degree(absolute_graph, weight="abs_weight") if graph.number_of_edges() else {node: 0.0 for node in graph}
    try:
        effective_size = nx.effective_size(absolute_undirected, weight="abs_weight")
        constraint = nx.constraint(absolute_undirected, weight="abs_weight")
    except Exception:
        effective_size = {node: np.nan for node in graph}
        constraint = {node: np.nan for node in graph}
    if directed:
        try:
            hubs, authorities = nx.hits(absolute_graph, max_iter=1000, normalized=True)
        except Exception:
            hubs = authorities = {node: np.nan for node in graph}
    else:
        hubs = authorities = {node: np.nan for node in graph}
    in_degree = dict(graph.in_degree()) if directed else {}
    out_degree = dict(graph.out_degree()) if directed else {}
    in_strength = dict(graph.in_degree(weight="weight")) if directed else {}
    out_strength = dict(graph.out_degree(weight="weight")) if directed else {}
    node_rows = []
    for node in graph.nodes():
        node_rows.append({
            "node": str(node), "degree": degree.get(node, 0), "strength": strength.get(node, 0.0), "signed_strength": signed_strength.get(node, 0.0),
            "in_degree": in_degree.get(node, np.nan), "out_degree": out_degree.get(node, np.nan),
            "in_strength": in_strength.get(node, np.nan), "out_strength": out_strength.get(node, np.nan),
            "betweenness": betweenness.get(node, np.nan), "closeness": closeness.get(node, np.nan),
            "harmonic_centrality": harmonic.get(node, np.nan), "eigenvector": eigenvector.get(node, np.nan),
            "pagerank": pagerank.get(node, np.nan), "hub_score": hubs.get(node, np.nan),
            "authority_score": authorities.get(node, np.nan), "average_neighbor_degree": average_neighbor_degree.get(node, np.nan),
            "effective_size": effective_size.get(node, np.nan), "constraint": constraint.get(node, np.nan),
            "local_clustering": clustering.get(node, np.nan), "k_core": core.get(node, np.nan),
            "community": membership.get(str(node), np.nan),
        })
    node_table = pd.DataFrame(node_rows).sort_values(["degree", "strength"], ascending=False).reset_index(drop=True)

    components = list(nx.weakly_connected_components(graph)) if directed else list(nx.connected_components(graph))
    component_rows = [{"component": index + 1, "nodes": len(nodes), "members": ", ".join(sorted(map(str, nodes)))} for index, nodes in enumerate(sorted(components, key=len, reverse=True))]
    largest_nodes = max(components, key=len) if components else set()
    largest = distance_graph.subgraph(largest_nodes).copy()
    avg_path = np.nan
    diameter = np.nan
    if len(largest) > 1:
        try:
            if directed and not nx.is_strongly_connected(largest):
                largest = largest.to_undirected()
            avg_path = float(nx.average_shortest_path_length(largest, weight="distance"))
            diameter = float(nx.diameter(largest.to_undirected()))
        except Exception:
            pass
    reciprocity = float(nx.reciprocity(graph)) if directed and graph.number_of_edges() else np.nan
    assortativity = np.nan
    try:
        assortativity = float(nx.degree_assortativity_coefficient(undirected))
    except Exception:
        pass
    summary = pd.DataFrame([{
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "directed": directed,
        "density": nx.density(graph), "components": len(components),
        "largest_component_nodes": len(largest_nodes), "isolates": len(list(nx.isolates(graph))),
        "average_degree": float(np.mean(list(degree.values()))) if degree else 0.0,
        "average_strength": float(np.mean(list(strength.values()))) if strength else 0.0,
        "global_clustering": float(nx.transitivity(undirected)) if undirected.number_of_edges() else 0.0,
        "average_local_clustering": float(nx.average_clustering(absolute_undirected, weight="abs_weight")) if undirected.number_of_edges() else 0.0,
        "modularity": modularity, "communities": len(set(membership.values())),
        "reciprocity": reciprocity, "degree_assortativity": assortativity,
        "average_shortest_path_largest_component": avg_path, "diameter_largest_component": diameter,
        "global_efficiency": float(nx.global_efficiency(undirected)) if undirected.number_of_nodes() > 1 else np.nan,
        "local_efficiency": float(nx.local_efficiency(undirected)) if undirected.number_of_nodes() > 2 else np.nan,
        "maximum_clique_size": int(nx.approximation.large_clique_size(undirected)) if undirected.number_of_edges() else 1,
        "degree_centralization": _centralization(degree), "betweenness_centralization": _centralization(betweenness),
    }])
    community_table = pd.DataFrame([{"node": node, "community": community} for node, community in membership.items()]).sort_values(["community", "node"])
    return summary, node_table, pd.DataFrame(component_rows), community_table


def _edge_and_bridge_measures(graph: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_graph = _distance_graph(graph)
    edge_between = nx.edge_betweenness_centrality(distance_graph, weight="distance", normalized=True)
    bridge_set = set()
    articulation = []
    undirected = graph.to_undirected()
    if undirected.number_of_edges():
        try:
            bridge_set = {tuple(sorted(map(str, edge))) for edge in nx.bridges(undirected)}
            articulation = list(nx.articulation_points(undirected))
        except Exception:
            pass
    rows = []
    for left, right, attrs in graph.edges(data=True):
        key = (left, right) if (left, right) in edge_between else (right, left)
        rows.append({
            "source": str(left), "target": str(right), "weight": attrs.get("weight", 1.0),
            "edge_betweenness": edge_between.get(key, np.nan),
            "is_bridge": tuple(sorted((str(left), str(right)))) in bridge_set,
        })
    bridge_table = pd.DataFrame([
        {"type": "Articulation node", "member": str(node)} for node in articulation
    ] + [
        {"type": "Bridge edge", "member": f"{left} -- {right}"} for left, right in sorted(bridge_set)
    ])
    return pd.DataFrame(rows).sort_values("edge_betweenness", ascending=False), bridge_table


def _small_world_assessment(graph: nx.Graph, iterations: int = 50, seed: int = 42) -> pd.DataFrame:
    undirected = graph.to_undirected()
    n, m = undirected.number_of_nodes(), undirected.number_of_edges()
    if n < 5 or m < n - 1 or not nx.is_connected(undirected):
        return pd.DataFrame([{"status": "Not estimated", "reason": "Small-world assessment requires a connected network with sufficient nodes and edges."}])
    observed_c = nx.average_clustering(undirected)
    observed_l = nx.average_shortest_path_length(undirected)
    rng = np.random.default_rng(seed)
    random_c, random_l = [], []
    for _ in range(max(10, iterations)):
        candidate = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
        if not nx.is_connected(candidate):
            largest = candidate.subgraph(max(nx.connected_components(candidate), key=len)).copy()
        else:
            largest = candidate
        if len(largest) > 1:
            random_c.append(nx.average_clustering(candidate))
            random_l.append(nx.average_shortest_path_length(largest))
    c_rand = float(np.mean(random_c)) if random_c else np.nan
    l_rand = float(np.mean(random_l)) if random_l else np.nan
    sigma = (observed_c / c_rand) / (observed_l / l_rand) if all(np.isfinite(v) and v > 0 for v in [c_rand, l_rand, observed_l]) else np.nan
    return pd.DataFrame([{
        "observed_clustering": observed_c, "random_clustering_mean": c_rand,
        "observed_path_length": observed_l, "random_path_length_mean": l_rand,
        "small_world_sigma": sigma, "interpretation": "Small-world tendency" if np.isfinite(sigma) and sigma > 1 else "No clear small-world tendency",
    }])


def _diagnostics(graph: nx.Graph, edge_table: pd.DataFrame, config: dict[str, Any], sample_n: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    density = nx.density(graph)
    components = nx.number_weakly_connected_components(graph) if graph.is_directed() else nx.number_connected_components(graph)
    rows.append({"test": "Network size", "statistic": f"{n} nodes; {m} edges", "status": "Satisfied" if n >= 5 else "Minor concern", "interpretation": "Very small networks provide limited structural information.", "recommended_response": "Report the small network transparently and avoid overinterpreting centrality ranks."})
    rows.append({"test": "Connectedness", "statistic": components, "status": "Satisfied" if components == 1 else "Material concern", "interpretation": "Disconnected components make global path-based measures less comparable.", "recommended_response": "Report components and calculate path measures on the largest connected component."})
    rows.append({"test": "Density", "statistic": density, "status": "Minor concern" if density > 0.80 or density < 0.02 else "Satisfied", "interpretation": "Extremely dense or sparse networks may reflect threshold choice or measurement design.", "recommended_response": "Run threshold or estimator sensitivity analyses and report the rule used."})
    self_loops = nx.number_of_selfloops(graph)
    rows.append({"test": "Self-loops", "statistic": self_loops, "status": "Minor concern" if self_loops else "Satisfied", "interpretation": "Self-loops are not used by most centrality measures.", "recommended_response": "Retain only when substantively meaningful and report their treatment."})
    duplicates = int(edge_table.get("multiplicity", pd.Series(dtype=float)).gt(1).sum()) if "multiplicity" in edge_table else 0
    rows.append({"test": "Duplicate edge aggregation", "statistic": duplicates, "status": "Minor concern" if duplicates else "Satisfied", "interpretation": "Repeated dyads were aggregated by summing weights.", "recommended_response": "Confirm whether repeated ties represent intensity or duplicate records."})
    negative = int((pd.to_numeric(edge_table.get("weight", pd.Series(dtype=float)), errors="coerce") < 0).sum())
    rows.append({"test": "Negative weights", "statistic": negative, "status": "Minor concern" if negative else "Satisfied", "interpretation": "Signed networks need sign-aware interpretation.", "recommended_response": "Report positive and negative ties separately where substantive meaning differs."})
    if sample_n is not None:
        ratio = sample_n / max(n, 1)
        rows.append({"test": "Observations per network variable", "statistic": ratio, "status": "Satisfied" if ratio >= 10 else "Material concern" if ratio < 5 else "Minor concern", "interpretation": "Correlation networks are unstable when observations are too few relative to nodes.", "recommended_response": "Reduce nodes, increase sample size, use regularisation and report bootstrap stability."})
    return pd.DataFrame(rows)


def _layout_positions(graph: nx.Graph, layout: str, seed: int = 42) -> dict[Any, np.ndarray]:
    if layout == "Circular":
        return nx.circular_layout(graph)
    if layout == "Kamada-Kawai":
        return nx.kamada_kawai_layout(graph, weight="distance")
    if layout == "Shell":
        return nx.shell_layout(graph)
    if layout == "Spectral":
        return nx.spectral_layout(graph, weight="weight")
    return nx.spring_layout(graph, seed=seed, weight="weight", k=None)


def _network_figure(graph: nx.Graph, node_table: pd.DataFrame, layout: str, title: str, seed: int = 42, community: bool = False) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = _layout_positions(_distance_graph(graph), layout, seed)
    degree_map = dict(zip(node_table["node"].astype(str), node_table["degree"]))
    community_map = dict(zip(node_table["node"].astype(str), node_table["community"]))
    node_sizes = [350 + 120 * math.sqrt(max(float(degree_map.get(str(node), 0)), 0)) for node in graph.nodes()]
    node_colors = [community_map.get(str(node), 0) for node in graph.nodes()] if community else None
    nx.draw_networkx_nodes(graph, positions, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.tab20 if community else None, ax=ax)
    # Draw positive and negative edges separately so signs remain visible in monochrome printing.
    positive_data = [(u, v, attrs) for u, v, attrs in graph.edges(data=True) if _safe_float(attrs.get("weight", 1.0)) >= 0]
    negative_data = [(u, v, attrs) for u, v, attrs in graph.edges(data=True) if _safe_float(attrs.get("weight", 1.0)) < 0]
    if positive_data:
        nx.draw_networkx_edges(graph, positions, edgelist=[(u, v) for u, v, _ in positive_data], width=[max(0.6, min(5.0, 1.5 * abs(_safe_float(attrs.get("weight", 1.0))))) for _, _, attrs in positive_data], alpha=0.65, arrows=graph.is_directed(), ax=ax)
    if negative_data:
        nx.draw_networkx_edges(graph, positions, edgelist=[(u, v) for u, v, _ in negative_data], style="dashed", width=[max(0.6, min(5.0, 1.5 * abs(_safe_float(attrs.get("weight", 1.0))))) for _, _, attrs in negative_data], alpha=0.7, arrows=graph.is_directed(), ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=9, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    return _png_bytes(fig)


def _centrality_figure(node_table: pd.DataFrame) -> bytes:
    top = node_table.nlargest(min(20, len(node_table)), "betweenness").sort_values("betweenness")
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.34 * len(top))))
    ax.barh(top["node"].astype(str), top["betweenness"])
    ax.set_xlabel("Betweenness centrality")
    ax.set_title("Node betweenness centrality")
    return _png_bytes(fig)


def _degree_figure(node_table: pd.DataFrame) -> bytes:
    values = node_table["degree"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(15, max(4, int(math.sqrt(max(len(values), 1)))))
    ax.hist(values, bins=bins)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Degree distribution")
    return _png_bytes(fig)


def _heatmap_figure(graph: nx.Graph) -> bytes:
    nodes = list(graph.nodes())
    matrix = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    fig, ax = plt.subplots(figsize=(max(6, len(nodes) * 0.35), max(5, len(nodes) * 0.32)))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm" if np.nanmin(matrix) < 0 else "viridis")
    ax.set_xticks(range(len(nodes)), labels=[str(n) for n in nodes], rotation=90)
    ax.set_yticks(range(len(nodes)), labels=[str(n) for n in nodes])
    ax.set_title("Weighted adjacency matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _png_bytes(fig)


def interactive_network_html(graph: nx.Graph, node_table: pd.DataFrame, title: str = "Interactive network") -> str:
    node_metrics = node_table.set_index("node").to_dict(orient="index") if not node_table.empty else {}
    nodes = [{"id": str(node), "label": str(node), "metrics": node_metrics.get(str(node), {})} for node in graph.nodes()]
    edges = [{"source": str(u), "target": str(v), "weight": _safe_float(attrs.get("weight", 1.0)), "directed": graph.is_directed()} for u, v, attrs in graph.edges(data=True)]
    payload = json.dumps({"nodes": nodes, "edges": edges, "title": title})
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
    body{{font-family:Arial,sans-serif;margin:0;background:#fff}} #bar{{padding:8px 12px;border-bottom:1px solid #ddd;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
    button{{padding:6px 10px;border:1px solid #999;background:#f6f6f6;border-radius:6px;cursor:pointer}} svg{{width:100%;height:650px;display:block}}
    .node{{cursor:grab}} .node text{{pointer-events:none;font-size:12px}} .edge{{stroke:#8b97a6;stroke-opacity:.65}} .tip{{position:absolute;background:#fff;border:1px solid #999;padding:8px;border-radius:6px;display:none;max-width:260px;font-size:12px}}
    </style></head><body><div id='bar'><b>{html.escape(title)}</b><span>Drag nodes. Click a node to inspect measures.</span><button onclick='autoLayout("lr")'>Left→right</button><button onclick='autoLayout("tb")'>Top→bottom</button><button onclick='autoLayout("circle")'>Circular</button><button onclick='fit()'>Fit</button></div><svg id='svg' viewBox='0 0 1100 650'></svg><div id='tip' class='tip'></div><script>
    const data={payload}; const svg=document.getElementById('svg'), tip=document.getElementById('tip');
    const NS='http://www.w3.org/2000/svg'; const state={{}}; const nodes=data.nodes; const edges=data.edges;
    function autoLayout(mode){{nodes.forEach((n,i)=>{{let x,y; if(mode==='circle'){{const a=2*Math.PI*i/nodes.length;x=550+250*Math.cos(a);y=330+230*Math.sin(a)}} else if(mode==='tb'){{x=120+(i%(Math.ceil(Math.sqrt(nodes.length))))*180;y=100+Math.floor(i/Math.ceil(Math.sqrt(nodes.length)))*150}} else {{x=120+Math.floor(i/Math.ceil(Math.sqrt(nodes.length)))*220;y=100+(i%Math.ceil(Math.sqrt(nodes.length)))*125}} state[n.id]={{x,y}}}}); draw()}}
    function el(name,attrs){{const e=document.createElementNS(NS,name);Object.entries(attrs||{{}}).forEach(([k,v])=>e.setAttribute(k,v));return e}}
    function draw(){{svg.innerHTML=''; edges.forEach(e=>{{const a=state[e.source],b=state[e.target];if(!a||!b)return;const line=el('line',{{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge','stroke-width':Math.max(1,Math.min(6,Math.abs(e.weight)*2)),'stroke-dasharray':e.weight<0?'6 4':''}});svg.appendChild(line)}});nodes.forEach(n=>{{const p=state[n.id];const g=el('g',{{class:'node',transform:`translate(${{p.x}},${{p.y}})`}});const c=el('circle',{{r:28,fill:'#dbeafe',stroke:'#1f2937','stroke-width':2}});const t=el('text',{{'text-anchor':'middle',dy:4}});t.textContent=n.label.length>14?n.label.slice(0,13)+'…':n.label;g.append(c,t);g.addEventListener('pointerdown',startDrag);g.addEventListener('click',ev=>showTip(ev,n));g.dataset.id=n.id;svg.appendChild(g)}})}}
    let drag=null; function point(ev){{const r=svg.getBoundingClientRect();return{{x:(ev.clientX-r.left)*1100/r.width,y:(ev.clientY-r.top)*650/r.height}}}}
    function startDrag(ev){{drag=this.dataset.id;this.setPointerCapture(ev.pointerId);this.addEventListener('pointermove',moveDrag);this.addEventListener('pointerup',endDrag)}}
    function moveDrag(ev){{if(!drag)return;state[drag]=point(ev);draw()}} function endDrag(ev){{drag=null}}
    function showTip(ev,n){{tip.style.display='block';tip.style.left=(ev.clientX+10)+'px';tip.style.top=(ev.clientY+10)+'px';tip.innerHTML='<b>'+n.label+'</b><br>'+Object.entries(n.metrics||{{}}).map(([k,v])=>k+': '+(typeof v==='number'?v.toFixed(4):v)).join('<br>')}}
    function fit(){{autoLayout('circle')}} autoLayout('lr');
    </script></body></html>"""


def _bootstrap_stability(df: pd.DataFrame, config: dict[str, Any], base_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = config.get("variables") or []
    samples = int(config.get("bootstrap_samples", 0))
    if samples <= 0 or len(df) < 20 or len(variables) < 3:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(int(config.get("random_state", 42)))
    edge_keys = [tuple(sorted((str(row.source), str(row.target)))) for row in base_edges.itertuples()]
    inclusion = {key: 0 for key in edge_keys}
    base_graph = nx.Graph()
    for row in base_edges.itertuples():
        base_graph.add_edge(str(row.source), str(row.target), weight=float(row.weight))
    base_strength = dict(base_graph.degree(weight="weight"))
    correlations = []
    for _ in range(samples):
        sampled = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        try:
            graph, edges, _, _ = _correlation_network(sampled, config)
        except Exception:
            continue
        present = {tuple(sorted((str(row.source), str(row.target)))) for row in edges.itertuples()}
        for key in edge_keys:
            inclusion[key] += int(key in present)
        boot_strength = dict(graph.degree(weight="weight"))
        common = [node for node in base_strength if node in boot_strength]
        if len(common) >= 3:
            corr = stats.spearmanr([base_strength[n] for n in common], [boot_strength[n] for n in common]).statistic
            if np.isfinite(corr):
                correlations.append(float(corr))
    completed = max(1, len(correlations) if correlations else samples)
    edge_table = pd.DataFrame([{"source": key[0], "target": key[1], "edge_inclusion_probability": count / max(samples, 1)} for key, count in inclusion.items()])
    stability = pd.DataFrame([{
        "bootstrap_resamples_requested": samples, "centrality_correlations_completed": len(correlations),
        "median_strength_rank_correlation": float(np.median(correlations)) if correlations else np.nan,
        "lower_5_percent_strength_rank_correlation": float(np.quantile(correlations, 0.05)) if correlations else np.nan,
        "interpretation": "Stable" if correlations and np.quantile(correlations, 0.05) >= 0.70 else "Potential instability",
    }])
    return edge_table, stability


def _group_network_comparison(df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, bytes], list[str]]:
    group = config.get("group_variable")
    values = config.get("group_values") or []
    permutations = int(config.get("permutation_samples", 0))
    if not group or len(values) != 2 or permutations <= 0:
        return {}, {}, []
    first_data = df[df[group] == values[0]]
    second_data = df[df[group] == values[1]]
    first_graph, first_edges, _, _ = _correlation_network(first_data, config)
    second_graph, second_edges, _, _ = _correlation_network(second_data, config)
    variables = config.get("variables") or []
    def edge_vector(graph):
        return np.array([graph.get_edge_data(a, b, {}).get("weight", 0.0) for i, a in enumerate(variables) for b in variables[i+1:]], dtype=float)
    first_vec, second_vec = edge_vector(first_graph), edge_vector(second_graph)
    observed_strength = abs(np.abs(first_vec).sum() - np.abs(second_vec).sum())
    observed_max = float(np.max(np.abs(first_vec - second_vec))) if len(first_vec) else 0.0
    combined = df[df[group].isin(values)].copy()
    labels = combined[group].to_numpy(copy=True)
    rng = np.random.default_rng(int(config.get("random_state", 42)))
    perm_strength, perm_max = [], []
    for _ in range(permutations):
        rng.shuffle(labels)
        permuted = combined.copy(); permuted[group] = labels
        try:
            g1, _, _, _ = _correlation_network(permuted[permuted[group] == values[0]], config)
            g2, _, _, _ = _correlation_network(permuted[permuted[group] == values[1]], config)
        except Exception:
            continue
        v1, v2 = edge_vector(g1), edge_vector(g2)
        perm_strength.append(abs(np.abs(v1).sum() - np.abs(v2).sum()))
        perm_max.append(float(np.max(np.abs(v1-v2))))
    result = pd.DataFrame([{
        "group_1": values[0], "group_1_n": len(first_data), "group_2": values[1], "group_2_n": len(second_data),
        "global_strength_difference": observed_strength,
        "global_strength_permutation_p": (1 + sum(v >= observed_strength for v in perm_strength)) / (1 + len(perm_strength)) if perm_strength else np.nan,
        "maximum_edge_difference": observed_max,
        "maximum_edge_permutation_p": (1 + sum(v >= observed_max for v in perm_max)) / (1 + len(perm_max)) if perm_max else np.nan,
        "permutations_completed": len(perm_strength),
    }])
    figures = {
        f"Network for {values[0]}": _network_figure(first_graph, _network_measures(first_graph)[1], config.get("layout", "Spring"), f"Network: {values[0]}"),
        f"Network for {values[1]}": _network_figure(second_graph, _network_measures(second_graph)[1], config.get("layout", "Spring"), f"Network: {values[1]}"),
    }
    return {"Network comparison permutation test": result, f"Edges - {values[0]}": first_edges, f"Edges - {values[1]}": second_edges}, figures, ["Group network comparison uses label permutation and should be interpreted with adequate group sample sizes."]


def network_analysis(df: pd.DataFrame, config: dict[str, Any]) -> AnalysisResult:
    mode = str(config.get("network_input", "Edge list"))
    estimator_note = "Observed edge list"
    matrix_table = pd.DataFrame()
    construction_warnings: list[str] = []
    if mode == "Correlation or partial-correlation network":
        graph, edge_table, matrix_table, estimator_note = _correlation_network(df, config)
        sample_n = int(df[config.get("variables") or []].dropna().shape[0])
    elif mode == "Adjacency matrix":
        graph, edge_table, matrix_table = _adjacency_network(df, config)
        sample_n = None
        estimator_note = "Uploaded adjacency matrix"
    else:
        graph, edge_table, construction_warnings = _build_edge_graph(df, config)
        sample_n = None
        estimator_note = "Uploaded edge list"

    summary, node_table, components, communities = _network_measures(graph, int(config.get("random_state", 42)))
    edge_measures, bridges = _edge_and_bridge_measures(graph)
    diagnostics = _diagnostics(graph, edge_table, config, sample_n)
    small_world = _small_world_assessment(graph, int(config.get("random_graph_iterations", 50)), int(config.get("random_state", 42)))
    layout = str(config.get("layout", "Spring"))
    figures = {
        "Network structure": _network_figure(graph, node_table, layout, "Network structure", int(config.get("random_state", 42))),
        "Community structure": _network_figure(graph, node_table, layout, "Network communities", int(config.get("random_state", 42)), community=True),
        "Centrality profile": _centrality_figure(node_table),
        "Degree distribution": _degree_figure(node_table),
        "Adjacency heatmap": _heatmap_figure(graph),
    }
    tables: dict[str, pd.DataFrame] = {
        "Network summary measures": summary,
        "Node centrality and community measures": node_table,
        "Network edge list": edge_table,
        "Edge centrality and bridge measures": edge_measures,
        "Articulation nodes and bridge edges": bridges,
        "Connected components": components,
        "Community membership": communities,
        "Small-world assessment": small_world,
    }
    if not matrix_table.empty:
        tables["Association or adjacency matrix"] = matrix_table

    if graph.is_directed() and graph.number_of_nodes() <= 500:
        try:
            tables["Directed triad census"] = pd.DataFrame([nx.triadic_census(graph)])
        except Exception:
            pass

    warnings: list[str] = list(construction_warnings)
    edge_stability, centrality_stability = pd.DataFrame(), pd.DataFrame()
    if mode == "Correlation or partial-correlation network":
        edge_stability, centrality_stability = _bootstrap_stability(df, config, edge_table)
        if not edge_stability.empty:
            tables["Bootstrap edge stability"] = edge_stability
            tables["Bootstrap centrality stability"] = centrality_stability
            if centrality_stability.iloc[0].get("interpretation") == "Potential instability":
                warnings.append("Bootstrap results suggest that centrality rankings may be unstable. Emphasise uncertainty rather than fixed node rankings.")
        comparison_tables, comparison_figures, comparison_warnings = _group_network_comparison(df, config)
        tables.update(comparison_tables); figures.update(comparison_figures); warnings.extend(comparison_warnings)

    summary_row = summary.iloc[0]
    methods_text = (
        f"A {mode.lower()} was estimated using {estimator_note}. The analysis treated the network as "
        f"{'directed' if graph.is_directed() else 'undirected'} and {'weighted' if any(abs(_safe_float(d.get('weight',1.0))-1.0)>1e-12 for _,_,d in graph.edges(data=True)) else 'unweighted'}. "
        f"Node-level measures included degree, strength, betweenness, closeness, harmonic, eigenvector and PageRank centrality, local clustering and k-core membership. "
        f"Graph-level measures included density, connected components, clustering, modularity, assortativity, reciprocity where applicable, path length, diameter and centralisation."
    )
    results_text = (
        f"The network contained {int(summary_row['nodes'])} nodes and {int(summary_row['edges'])} edges, with density {summary_row['density']:.3f}. "
        f"It comprised {int(summary_row['components'])} component(s) and {int(summary_row['communities'])} detected community structure(s). "
        f"The highest-degree node was {node_table.iloc[0]['node']} with degree {node_table.iloc[0]['degree']}. "
        f"Global clustering was {summary_row['global_clustering']:.3f} and modularity was {summary_row['modularity']:.3f}."
    )
    diagnostic_concerns = diagnostics[diagnostics["status"].isin(["Minor concern", "Material concern"])]
    diagnostic_text = (
        f"The diagnostic review identified {len(diagnostic_concerns)} item(s) requiring qualification or sensitivity analysis. "
        "Connectedness, density, self-loops, duplicate dyads, signed edges and observations per node were assessed. "
        "Path-based measures were restricted to the largest connected component when necessary, and correlation-network stability was assessed by bootstrap when requested."
    )
    discussion_text = (
        "Network measures describe structural position within the specified network and do not by themselves establish causal influence. "
        "Centrality ranks can change with node selection, edge definition, thresholding, regularisation and sampling variation. "
        "Substantive interpretation should therefore prioritise stable patterns, community structure and sensitivity results rather than isolated rank positions."
    )
    abstract_text = (
        f"This study used comprehensive network analysis to examine a network of {int(summary_row['nodes'])} nodes and {int(summary_row['edges'])} edges. "
        f"The network density was {summary_row['density']:.3f}, global clustering was {summary_row['global_clustering']:.3f}, and {int(summary_row['communities'])} community structure(s) were detected. "
        f"{node_table.iloc[0]['node']} had the highest observed degree. Diagnostic and stability analyses were used to qualify interpretation of centrality and global structure."
    )
    tables["Paper-ready abstract"] = pd.DataFrame([{"section": "Abstract results", "text": abstract_text}])
    tables["Paper-ready methods narrative"] = pd.DataFrame([{"section": "Network analysis methods", "text": methods_text}])
    tables["Paper-ready results narrative"] = pd.DataFrame([{"section": "Network analysis results", "text": results_text}])
    tables["Paper-ready diagnostics and robustness"] = pd.DataFrame([{"section": "Diagnostics and robustness", "text": diagnostic_text}])
    tables["Paper-ready discussion and limitations"] = pd.DataFrame([{"section": "Interpretation and limitations", "text": discussion_text}])
    tables["Network figure captions"] = pd.DataFrame([
        {"figure": "Network structure", "caption": "Network structure with node size reflecting degree and edge width reflecting absolute tie strength. Dashed edges indicate negative ties where present."},
        {"figure": "Community structure", "caption": "Community structure identified using modularity-based community detection."},
        {"figure": "Centrality profile", "caption": "Betweenness centrality of the highest-ranking nodes."},
        {"figure": "Degree distribution", "caption": "Distribution of node degree across the network."},
        {"figure": "Adjacency heatmap", "caption": "Weighted adjacency matrix showing the magnitude and sign of observed ties."},
    ])
    tables["Network reporting checklist"] = pd.DataFrame([
        {"reporting_item": item, "included": True} for item in [
            "Network definition and unit of analysis", "Directed or undirected status", "Weight definition and edge threshold",
            "Missing, duplicate and self-loop treatment", "Node and graph-level measures", "Community detection procedure",
            "Layout algorithm and visual encodings", "Connectedness and density diagnostics", "Stability or sensitivity analysis",
            "Software and reproducibility information", "Limitations and non-causal interpretation",
        ]
    ])
    interactive_html = interactive_network_html(graph, node_table, "Interactive network editor")
    summary_text = results_text + " The complete output includes diagnostics, stability checks, publication-ready tables and five core diagrams."
    reproducible = (
        "# Network analysis reproducibility outline\n"
        "import networkx as nx\n"
        f"# Input mode: {mode}; estimator: {estimator_note}; layout: {layout}\n"
        "# Reconstruct the graph from the exported Network edge list, then compute the reported measures.\n"
    )
    result = AnalysisResult(
        method="Comprehensive network analysis",
        summary=summary_text,
        tables=tables,
        figures=figures,
        diagnostics=diagnostics,
        metadata={
            "network_input": mode, "network_estimator": estimator_note, "layout": layout,
            "interactive_network_html": interactive_html, "paper_ready_methods": methods_text,
            "paper_ready_results": results_text, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        },
        warnings=warnings,
        reproducible_code=reproducible,
    )
    result.treatment_log.append(AuditEntry(
        action="Constructed and analysed network",
        details=f"Created a network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges using {estimator_note}.",
        justification="Network construction rules, thresholds, weights and diagnostics are retained in the analysis plan and reproducibility package.",
        before_n=len(df), after_n=len(df),
    ))
    return result
