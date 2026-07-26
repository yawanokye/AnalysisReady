from __future__ import annotations

from io import BytesIO
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .models import AnalysisResult


DEFAULT_PALETTE = {
    "background": "white",
    "ink": "#17365D",
    "muted": "#5B6573",
    "latent_fill": "#D9EAF7",
    "item_fill": "#F7F9FC",
    "accent": "#2F75B5",
    "warning": "#9C5700",
    "border": "#D0D5DD",
}
MONO_PALETTE = {
    "background": "white",
    "ink": "#111111",
    "muted": "#555555",
    "latent_fill": "#E6E6E6",
    "item_fill": "#F8F8F8",
    "accent": "#111111",
    "warning": "#666666",
    "border": "#999999",
}


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _center_text(draw: ImageDraw.ImageDraw, centre: tuple[float, float], text: str, font, fill: str):
    width, height = _text_size(draw, text, font)
    draw.text((centre[0] - width / 2, centre[1] - height / 2), text, font=font, fill=fill)




def _wrap_text(text: str, max_chars: int = 20) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:3]


def _multiline_center(draw: ImageDraw.ImageDraw, centre: tuple[float, float], lines: list[str], font, fill: str, line_gap: float = 4.0):
    sizes = [_text_size(draw, line, font) for line in lines]
    total_height = sum(height for _, height in sizes) + line_gap * max(len(lines) - 1, 0)
    cursor_y = centre[1] - total_height / 2
    for line, (width, height) in zip(lines, sizes):
        draw.text((centre[0] - width / 2, cursor_y), line, font=font, fill=fill)
        cursor_y += height + line_gap


def _node_dimensions(draw: ImageDraw.ImageDraw, text: str, font, scale: float) -> tuple[float, float, list[str]]:
    lines = _wrap_text(text, 21)
    widths = [_text_size(draw, line, font)[0] for line in lines]
    rx = max(105 * scale, min(180 * scale, max(widths, default=0) / 2 + 24 * scale))
    ry = max(42 * scale, 30 * scale + 13 * scale * max(len(lines) - 1, 0))
    return rx, ry, lines


def _path_crosses_other_node(
    source: str, target: str, positions: dict[str, tuple[float, float]], node_sizes: dict[str, tuple[float, float]],
) -> bool:
    if source not in positions or target not in positions:
        return False
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1:
        return False
    for node, (cx, cy) in positions.items():
        if node in {source, target}:
            continue
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / length_sq))
        px, py = x1 + t * dx, y1 + t * dy
        distance = math.hypot(cx - px, cy - py)
        rx, ry = node_sizes.get(node, (100, 45))
        if 0.08 < t < 0.92 and distance < max(rx, ry) * 0.85:
            return True
    return False

def _arrow_head(draw: ImageDraw.ImageDraw, tip: tuple[float, float], direction: tuple[float, float], fill: str, scale: float = 1.0):
    ux, uy = direction
    head_len = 15 * scale
    head_w = 8 * scale
    base_x = tip[0] - ux * head_len
    base_y = tip[1] - uy * head_len
    px, py = -uy, ux
    draw.polygon([
        tip,
        (base_x + px * head_w, base_y + py * head_w),
        (base_x - px * head_w, base_y - py * head_w),
    ], fill=fill)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int = 4,
    dashed: bool = False,
    curved: bool = False,
    scale: float = 1.0,
):
    x1, y1 = start
    x2, y2 = end
    if curved:
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1.0)
        px, py = -dy / length, dx / length
        bend = min(70 * scale, length * 0.20)
        control = ((x1 + x2) / 2 + px * bend, (y1 + y2) / 2 + py * bend)
        points = []
        for value in np.linspace(0, 1, 40):
            inv = 1 - value
            points.append((
                inv * inv * x1 + 2 * inv * value * control[0] + value * value * x2,
                inv * inv * y1 + 2 * inv * value * control[1] + value * value * y2,
            ))
        if dashed:
            for index in range(0, len(points) - 1, 4):
                draw.line(points[index:min(index + 3, len(points))], fill=fill, width=width, joint="curve")
        else:
            draw.line(points, fill=fill, width=width, joint="curve")
        prev = points[-2]
        tip = points[-1]
        vx, vy = tip[0] - prev[0], tip[1] - prev[1]
        vlen = max(math.hypot(vx, vy), 1.0)
        _arrow_head(draw, tip, (vx / vlen, vy / vlen), fill, scale)
        return

    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    if dashed:
        dash, gap = 13 * scale, 8 * scale
        cursor = 0.0
        while cursor < length:
            seg_end = min(cursor + dash, length)
            draw.line((x1 + ux * cursor, y1 + uy * cursor, x1 + ux * seg_end, y1 + uy * seg_end), fill=fill, width=width)
            cursor += dash + gap
    else:
        draw.line((x1, y1, x2, y2), fill=fill, width=width)
    _arrow_head(draw, (x2, y2), (ux, uy), fill, scale)


def _dag_levels(nodes: list[str], paths: list[tuple[str, str]]) -> dict[str, int]:
    parents = {node: [] for node in nodes}
    for source, target in paths:
        if source in parents and target in parents:
            parents[target].append(source)
    levels: dict[str, int] = {}
    remaining = set(nodes)
    for _ in range(len(nodes) + 1):
        progressed = False
        for node in list(remaining):
            if not parents[node]:
                levels[node] = 0
                remaining.remove(node)
                progressed = True
            elif all(parent in levels for parent in parents[node]):
                levels[node] = max(levels[parent] for parent in parents[node]) + 1
                remaining.remove(node)
                progressed = True
        if not remaining or not progressed:
            break
    for node in remaining:
        levels[node] = 0
    return levels


def _ordered_nodes(nodes: list[str], requested: list[str] | None) -> list[str]:
    requested = requested or []
    ordered = [node for node in requested if node in nodes]
    ordered.extend(node for node in nodes if node not in ordered)
    return ordered


def _latent_positions(
    nodes: list[str],
    paths: list[tuple[str, str]],
    width: int,
    structural_bottom: int,
    layout: str,
) -> dict[str, tuple[float, float]]:
    left, right = 180, width - 180
    top, bottom = 155, structural_bottom
    if not nodes:
        return {}
    if layout == "Radial":
        centre_x = width / 2
        centre_y = (top + bottom) / 2 + 10
        radius_x = min((right - left) * 0.42, 520)
        radius_y = min((bottom - top) * 0.42, 175)
        if len(nodes) == 1:
            return {nodes[0]: (centre_x, centre_y)}
        return {
            node: (
                centre_x + radius_x * math.cos(-math.pi / 2 + 2 * math.pi * index / len(nodes)),
                centre_y + radius_y * math.sin(-math.pi / 2 + 2 * math.pi * index / len(nodes)),
            )
            for index, node in enumerate(nodes)
        }

    levels = _dag_levels(nodes, paths)
    max_level = max(levels.values(), default=0)
    groups: dict[int, list[str]] = {}
    for node in nodes:
        groups.setdefault(levels[node], []).append(node)
    for level in groups:
        groups[level] = [node for node in nodes if node in groups[level]]

    if layout in {"Top to bottom", "Bottom to top", "Hierarchical"}:
        positions: dict[str, tuple[float, float]] = {}
        for level, group in sorted(groups.items()):
            y = (top + bottom) / 2 if max_level == 0 else top + (bottom - top) * level / max_level
            if layout == "Bottom to top":
                y = bottom - (y - top)
            xs = [width / 2] if len(group) == 1 else np.linspace(left, right, len(group)).tolist()
            for node, x in zip(group, xs):
                positions[node] = (x, y)
        return positions

    if layout == "Measurement first":
        xs = [width / 2] if len(nodes) == 1 else np.linspace(left, right, len(nodes)).tolist()
        return {node: (x, top + 55) for node, x in zip(nodes, xs)}

    if max_level == 0:
        xs = [width / 2] if len(nodes) == 1 else np.linspace(left, right, len(nodes)).tolist()
        return {node: (x, (top + bottom) / 2) for node, x in zip(nodes, xs)}

    positions = {}
    for level, group in sorted(groups.items()):
        x = left + (right - left) * level / max_level
        ys = [(top + bottom) / 2] if len(group) == 1 else np.linspace(top, bottom, len(group)).tolist()
        for node, y in zip(group, ys):
            positions[node] = (x, y)
    return positions


def _edge_estimate(path_table: pd.DataFrame, source: str, target: str) -> tuple[float | None, float | None]:
    if path_table.empty or not {"predictor", "outcome"}.issubset(path_table.columns):
        return None, None
    rows = path_table[(path_table["predictor"].astype(str) == source) & (path_table["outcome"].astype(str) == target)]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    estimate = row.get("standardized_estimate", row.get("estimate"))
    p_value = row.get("bootstrap_p", row.get("p_value_approx", row.get("p_value")))
    return (float(estimate) if pd.notna(estimate) else None, float(p_value) if pd.notna(p_value) else None)


def _loading_estimates(loading_table: pd.DataFrame, construct: str) -> dict[str, float]:
    if loading_table.empty or "construct" not in loading_table.columns:
        return {}
    rows = loading_table[loading_table["construct"].astype(str) == construct]
    result = {}
    for _, row in rows.iterrows():
        value = row.get("standardized_loading", row.get("loading", row.get("outer_loading")))
        item = row.get("item", row.get("indicator"))
        if item is not None and pd.notna(value):
            result[str(item)] = float(value)
    return result


def _edge_points(
    source: tuple[float, float],
    target: tuple[float, float],
    rx: float,
    ry: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length = max(math.hypot(dx, dy), 1)
    ux, uy = dx / length, dy / length
    return (sx + ux * rx, sy + uy * ry), (tx - ux * rx, ty - uy * ry)


def _edge_points_variable(
    source: tuple[float, float],
    target: tuple[float, float],
    source_size: tuple[float, float],
    target_size: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length

    def ellipse_offset(rx: float, ry: float) -> float:
        denominator = math.sqrt((ux / max(rx, 1.0)) ** 2 + (uy / max(ry, 1.0)) ** 2)
        return 1.0 / max(denominator, 1e-9)

    source_offset = ellipse_offset(*source_size)
    target_offset = ellipse_offset(*target_size)
    return (sx + ux * source_offset, sy + uy * source_offset), (tx - ux * target_offset, ty - uy * target_offset)


def _fit_text(fit_table: pd.DataFrame | None) -> str:
    if fit_table is None or fit_table.empty:
        return ""
    row = fit_table.iloc[0]
    parts = []
    observed = set()
    for key, label in [
        ("cfi", "CFI"), ("tli", "TLI"), ("rmsea", "RMSEA"),
        ("srmr", "SRMR"), ("srmr_approx", "SRMR"), ("r_squared", "R²"),
    ]:
        if label in observed:
            continue
        if key in row and pd.notna(row[key]):
            parts.append(f"{label}={float(row[key]):.3f}")
            observed.add(label)
    return "Model fit: " + " | ".join(parts) if parts else ""


def render_latent_path_diagram(
    construct_map: dict[str, list[str]],
    loading_table: pd.DataFrame,
    paths: list[tuple[str, str]] | None = None,
    path_table: pd.DataFrame | None = None,
    fit_table: pd.DataFrame | None = None,
    title: str = "SEM path diagram",
    settings: dict | None = None,
    structural_relations: list[dict] | None = None,
) -> bytes:
    settings = settings or {}
    layout = str(settings.get("layout", "Left to right"))
    nodes = _ordered_nodes(list(construct_map), settings.get("construct_order"))
    paths = [tuple(path) for path in (paths or [])]
    path_table = path_table if path_table is not None else pd.DataFrame()
    structural_relations = structural_relations or []
    scale = 1.35 if settings.get("resolution") == "High resolution" else 1.0
    compact = layout == "Compact publication"
    show_indicators = bool(settings.get("show_indicators", True))
    show_loadings = bool(settings.get("show_loadings", True))
    show_indicator_names = bool(settings.get("show_indicator_names", True))
    show_coefficients = bool(settings.get("show_coefficients", True))
    show_p_values = bool(settings.get("show_p_values", True))
    show_fit = bool(settings.get("show_fit", True))
    significance_colours = bool(settings.get("significance_colours", True))
    curved = settings.get("arrow_style") == "Curved"
    palette = MONO_PALETTE if settings.get("monochrome") else DEFAULT_PALETTE

    max_items = max((len(construct_map.get(node, [])) for node in nodes), default=1)
    base_width = max(1200 if compact else 1450, (330 if compact else 410) * max(len(nodes), 2))
    item_rows = max(1, math.ceil(max_items / 3))
    base_height = 650 if not show_indicators else max(900, 665 + item_rows * 72)
    width, height = int(base_width * scale), int(base_height * scale)
    structural_bottom = int((430 if show_indicators else 485) * scale)
    background = (255, 255, 255, 0) if settings.get("transparent") else palette["background"]
    mode = "RGBA" if settings.get("transparent") else "RGB"
    image = Image.new(mode, (width, height), background)
    draw = ImageDraw.Draw(image)

    title_font = _font(int((25 if compact else 30) * scale), True)
    node_font = _font(int((17 if compact else 20) * scale), True)
    item_font = _font(int((13 if compact else 16) * scale))
    label_font = _font(int((13 if compact else 16) * scale), True)
    small_font = _font(int((12 if compact else 14) * scale))
    _center_text(draw, (width / 2, 45 * scale), title, title_font, palette["ink"])
    subtitle = f"Layout: {layout} | Ellipses = constructs | Rectangles = indicators | Standardised estimates"
    _center_text(draw, (width / 2, 82 * scale), subtitle, small_font, palette["muted"])

    positions = _latent_positions(nodes, paths, width, structural_bottom, layout)
    node_dimensions = {
        node: _node_dimensions(draw, node, node_font, scale)
        for node in nodes
    }
    node_sizes = {node: (values[0], values[1]) for node, values in node_dimensions.items()}

    # Structural arrows are drawn before nodes. Paths that would pass through another
    # construct are curved automatically, even when straight arrows are selected.
    for source, target in paths:
        if source not in positions or target not in positions:
            continue
        start_point, end_point = _edge_points_variable(
            positions[source], positions[target], node_sizes[source], node_sizes[target]
        )
        estimate, p_value = _edge_estimate(path_table, source, target)
        if significance_colours and p_value is not None and p_value >= 0.05:
            line_colour = palette["warning"]
        else:
            line_colour = palette["accent"]
        edge_curved = curved or _path_crosses_other_node(source, target, positions, node_sizes)
        _arrow(
            draw, start_point, end_point, fill=line_colour,
            width=max(2, int(4 * scale)), curved=edge_curved, scale=scale,
        )
        label_parts = []
        if show_coefficients and estimate is not None:
            label_parts.append(f"β={estimate:.2f}")
        if show_p_values and p_value is not None:
            label_parts.append(f"p={p_value:.3f}" if p_value >= 0.001 else "p<.001")
        if label_parts:
            label = "; ".join(label_parts)
            dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
            length = max(math.hypot(dx, dy), 1.0)
            perpendicular = (-dy / length, dx / length)
            base_mid = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
            if edge_curved:
                bend = min(70 * scale, length * 0.20)
                base_mid = (
                    base_mid[0] + perpendicular[0] * bend * 0.50,
                    base_mid[1] + perpendicular[1] * bend * 0.50,
                )
            label_offset = 18 * scale
            mid = (
                base_mid[0] + perpendicular[0] * label_offset,
                base_mid[1] + perpendicular[1] * label_offset,
            )
            box = draw.textbbox((0, 0), label, font=label_font)
            pad = 6 * scale
            draw.rounded_rectangle(
                (mid[0] - (box[2] - box[0]) / 2 - pad, mid[1] - 12 * scale - pad,
                 mid[0] + (box[2] - box[0]) / 2 + pad, mid[1] + 12 * scale + pad),
                radius=int(7 * scale), fill="white", outline=line_colour, width=max(1, int(2 * scale)),
            )
            _center_text(draw, mid, label, label_font, line_colour)

    # Moderation is shown as a dashed arrow to the focal path midpoint.
    for relation in structural_relations:
        if relation.get("type") != "Moderator":
            continue
        predictor, moderator, outcome = relation.get("predictor"), relation.get("moderator"), relation.get("outcome")
        if not all(value in positions for value in [predictor, moderator, outcome]):
            continue
        focal_start, focal_end = _edge_points_variable(
            positions[predictor], positions[outcome], node_sizes[predictor], node_sizes[outcome]
        )
        midpoint = ((focal_start[0] + focal_end[0]) / 2, (focal_start[1] + focal_end[1]) / 2)
        m_start, _ = _edge_points_variable(
            positions[moderator], midpoint, node_sizes[moderator], (12 * scale, 12 * scale)
        )
        _arrow(
            draw, m_start, midpoint, fill=palette["muted"],
            width=max(2, int(3 * scale)), dashed=True, curved=True, scale=scale,
        )
        interaction_name = f"{predictor} × {moderator}"
        estimate, p_value = _edge_estimate(path_table, interaction_name, outcome)
        label = "moderates"
        if show_coefficients and estimate is not None:
            label += f"; β={estimate:.2f}"
        if show_p_values and p_value is not None:
            label += f"; p={p_value:.3f}" if p_value >= 0.001 else "; p<.001"
        _center_text(
            draw, ((m_start[0] + midpoint[0]) / 2, (m_start[1] + midpoint[1]) / 2 - 14 * scale),
            label, small_font, palette["muted"],
        )

    # Construct nodes. Long names wrap within dynamically sized ellipses.
    for node in nodes:
        cx, cy = positions[node]
        node_rx, node_ry, lines = node_dimensions[node]
        draw.ellipse(
            (cx - node_rx, cy - node_ry, cx + node_rx, cy + node_ry),
            fill=palette["latent_fill"], outline=palette["ink"], width=max(2, int(4 * scale)),
        )
        _multiline_center(draw, (cx, cy), lines, node_font, palette["ink"], 3 * scale)

    # Indicators are grouped in a footer by construct. This keeps every orientation readable.
    if show_indicators:
        footer_top = int(structural_bottom + 80 * scale)
        group_gap = 16 * scale
        available = width - 60 * scale
        group_width = max(220 * scale, (available - group_gap * max(len(nodes) - 1, 0)) / max(len(nodes), 1))
        box_h = 42 * scale
        footer_nodes = sorted(nodes, key=lambda node: positions[node][0])
        for construct_index, construct in enumerate(footer_nodes):
            items = construct_map.get(construct, [])
            loadings = _loading_estimates(loading_table, construct)
            group_left = 30 * scale + construct_index * (group_width + group_gap)
            columns = max(1, min(3, len(items)))
            inner_gap = 8 * scale
            box_w = max(80 * scale, (group_width - inner_gap * (columns - 1)) / columns)
            node_rx, node_ry = node_sizes[construct]
            for item_index, item in enumerate(items):
                row_index, column_index = divmod(item_index, columns)
                x1 = group_left + column_index * (box_w + inner_gap)
                y1 = footer_top + row_index * (box_h + 22 * scale)
                x2, y2 = x1 + box_w, y1 + box_h
                draw.rounded_rectangle(
                    (x1, y1, x2, y2), radius=int(7 * scale),
                    fill=palette["item_fill"], outline=palette["muted"], width=max(1, int(2 * scale)),
                )
                if show_indicator_names:
                    item_lines = _wrap_text(str(item), 16)
                    _multiline_center(draw, ((x1 + x2) / 2, (y1 + y2) / 2), item_lines, item_font, palette["ink"], 2 * scale)
                fan_fraction = 0.0 if len(items) <= 1 else (item_index - (len(items) - 1) / 2) / ((len(items) - 1) / 2)
                start_point = (
                    positions[construct][0] + fan_fraction * node_rx * 0.55,
                    positions[construct][1] + node_ry * 0.90,
                )
                end_point = ((x1 + x2) / 2, y1)
                _arrow(
                    draw, start_point, end_point, fill=palette["muted"],
                    width=max(1, int(2 * scale)), curved=curved, scale=scale,
                )
                loading = loadings.get(str(item))
                if show_loadings and loading is not None:
                    t = 0.70
                    mid_x = start_point[0] * (1 - t) + end_point[0] * t
                    mid_y = start_point[1] * (1 - t) + end_point[1] * t
                    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
                    length = max(math.hypot(dx, dy), 1.0)
                    mid_x += (-dy / length) * 9 * scale
                    mid_y += (dx / length) * 9 * scale
                    label = f"{loading:.2f}"
                    draw.rounded_rectangle(
                        (mid_x - 23 * scale, mid_y - 12 * scale, mid_x + 23 * scale, mid_y + 12 * scale),
                        radius=int(5 * scale), fill="white", outline=palette["border"],
                    )
                    _center_text(draw, (mid_x, mid_y), label, small_font, palette["ink"])

    fit = _fit_text(fit_table) if show_fit else ""
    if fit:
        _center_text(draw, (width / 2, height - 42 * scale), fit, label_font, palette["ink"])

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def attach_latent_variable_figures(result: AnalysisResult) -> AnalysisResult:
    settings = result.metadata.get("diagram_settings") or {}
    relations = result.metadata.get("structural_relations") or []
    if result.method == "Covariance-based structural equation model":
        construct_map = result.metadata.get("construct_map") or {}
        paths = [tuple(path) for path in (result.metadata.get("paths") or [])]
        if construct_map:
            result.figures["SEM path diagram"] = render_latent_path_diagram(
                construct_map=construct_map,
                loading_table=result.tables.get("SEM standardised loadings", pd.DataFrame()),
                paths=paths,
                path_table=result.tables.get("Structural path estimates", pd.DataFrame()),
                fit_table=result.tables.get("SEM fit indices", pd.DataFrame()),
                title="Structural equation model path diagram",
                settings=settings,
                structural_relations=relations,
            )
    elif result.method == "Confirmatory factor analysis":
        construct_map = result.metadata.get("construct_map") or {}
        if construct_map:
            result.figures["CFA measurement diagram"] = render_latent_path_diagram(
                construct_map=construct_map,
                loading_table=result.tables.get("CFA standardised loadings", pd.DataFrame()),
                paths=[],
                path_table=pd.DataFrame(),
                fit_table=result.tables.get("CFA fit indices", pd.DataFrame()),
                title="Confirmatory factor analysis measurement diagram",
                settings=settings,
                structural_relations=[],
            )
    elif result.method == "Partial least squares structural equation model":
        construct_map = result.metadata.get("construct_map") or {}
        paths = [tuple(path) for path in (result.metadata.get("paths") or [])]
        if construct_map:
            result.figures["PLS-SEM path diagram"] = render_latent_path_diagram(
                construct_map=construct_map,
                loading_table=result.tables.get("PLS outer loadings", pd.DataFrame()),
                paths=paths,
                path_table=result.tables.get("PLS structural path estimates", pd.DataFrame()),
                fit_table=result.tables.get("PLS-SEM model summary", pd.DataFrame()),
                title="Partial least squares SEM path diagram",
                settings=settings,
                structural_relations=relations,
            )
    return result
