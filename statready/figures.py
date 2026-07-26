from __future__ import annotations

from io import BytesIO
import math
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .models import AnalysisResult


_BG = "white"
_INK = "#17365D"
_MUTED = "#5B6573"
_LATENT_FILL = "#D9EAF7"
_ITEM_FILL = "#F7F9FC"
_ACCENT = "#2F75B5"
_WARNING = "#9C5700"


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


def _center_text(draw: ImageDraw.ImageDraw, centre: tuple[float, float], text: str, font, fill=_INK):
    width, height = _text_size(draw, text, font)
    draw.text((centre[0] - width / 2, centre[1] - height / 2), text, font=font, fill=fill)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], *, fill=_ACCENT, width=4, double=False, dashed=False):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length

    if dashed:
        dash, gap = 13, 8
        cursor = 0.0
        while cursor < length:
            seg_end = min(cursor + dash, length)
            draw.line((x1 + ux * cursor, y1 + uy * cursor, x1 + ux * seg_end, y1 + uy * seg_end), fill=fill, width=width)
            cursor += dash + gap
    else:
        draw.line((x1, y1, x2, y2), fill=fill, width=width)

    def head(tip_x, tip_y, direction_x, direction_y):
        head_len = 15
        head_w = 8
        base_x = tip_x - direction_x * head_len
        base_y = tip_y - direction_y * head_len
        px, py = -direction_y, direction_x
        draw.polygon([
            (tip_x, tip_y),
            (base_x + px * head_w, base_y + py * head_w),
            (base_x - px * head_w, base_y - py * head_w),
        ], fill=fill)

    head(x2, y2, ux, uy)
    if double:
        head(x1, y1, -ux, -uy)


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


def _latent_positions(nodes: list[str], paths: list[tuple[str, str]], width: int, height: int) -> dict[str, tuple[float, float]]:
    levels = _dag_levels(nodes, paths)
    max_level = max(levels.values(), default=0)
    groups: dict[int, list[str]] = {}
    for node in nodes:
        groups.setdefault(levels[node], []).append(node)
    positions = {}
    left, right = 180, width - 180
    top, bottom = 170, min(height * 0.52, 470)
    if max_level == 0 and len(nodes) > 1:
        xs = np.linspace(left, right, len(nodes)).tolist()
        centre_y = (top + bottom) / 2
        return {node: (x, centre_y) for node, x in zip(nodes, xs)}

    for level, group in sorted(groups.items()):
        x = left if max_level == 0 else left + (right - left) * level / max_level
        if len(group) == 1:
            ys = [(top + bottom) / 2]
        else:
            ys = np.linspace(top, bottom, len(group)).tolist()
        for node, y in zip(group, ys):
            positions[node] = (x, y)
    return positions


def _edge_estimate(path_table: pd.DataFrame, source: str, target: str) -> tuple[float | None, float | None]:
    if path_table.empty:
        return None, None
    rows = path_table[(path_table["predictor"].astype(str) == source) & (path_table["outcome"].astype(str) == target)]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    estimate = row.get("standardized_estimate", row.get("estimate"))
    p_value = row.get("p_value_approx", row.get("p_value"))
    return (float(estimate) if pd.notna(estimate) else None, float(p_value) if pd.notna(p_value) else None)


def _loading_estimates(loading_table: pd.DataFrame, construct: str) -> dict[str, float]:
    if loading_table.empty:
        return {}
    rows = loading_table[loading_table["construct"].astype(str) == construct]
    return {
        str(row["item"]): float(row.get("standardized_loading", row.get("loading")))
        for _, row in rows.iterrows()
        if pd.notna(row.get("standardized_loading", row.get("loading")))
    }


def render_latent_path_diagram(
    construct_map: dict[str, list[str]],
    loading_table: pd.DataFrame,
    paths: list[tuple[str, str]] | None = None,
    path_table: pd.DataFrame | None = None,
    fit_table: pd.DataFrame | None = None,
    title: str = "SEM path diagram",
) -> bytes:
    paths = list(paths or [])
    path_table = path_table if path_table is not None else pd.DataFrame()
    constructs = list(construct_map)
    max_items = max((len(items) for items in construct_map.values()), default=1)
    width = max(1350, 430 * max(len(constructs), 2))
    height = max(820, 640 + max(0, max_items - 4) * 65)
    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(30, True)
    node_font = _font(20, True)
    item_font = _font(16)
    label_font = _font(16, True)
    small_font = _font(14)

    _center_text(draw, (width / 2, 48), title, title_font)
    subtitle = "Ellipses = latent constructs | Rectangles = observed indicators | Values = standardised estimates"
    _center_text(draw, (width / 2, 88), subtitle, small_font, _MUTED)

    positions = _latent_positions(constructs, paths, width, height)
    latent_rx, latent_ry = 112, 45

    # Structural arrows first so nodes sit cleanly above them.
    for source, target in paths:
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        dx, dy = tx - sx, ty - sy
        length = max(math.hypot(dx, dy), 1)
        ux, uy = dx / length, dy / length
        start = (sx + ux * latent_rx, sy + uy * latent_ry)
        end = (tx - ux * latent_rx, ty - uy * latent_ry)
        estimate, p_value = _edge_estimate(path_table, source, target)
        line_colour = _ACCENT if p_value is None or p_value < 0.05 else _WARNING
        _arrow(draw, start, end, fill=line_colour, width=5)
        if estimate is not None:
            label = f"β={estimate:.2f}"
            if p_value is not None:
                label += f"; p={p_value:.3f}" if p_value >= 0.001 else "; p<.001"
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 22)
            box = draw.textbbox((0, 0), label, font=label_font)
            pad = 6
            draw.rounded_rectangle((mid[0] - (box[2]-box[0])/2 - pad, mid[1] - 12 - pad,
                                    mid[0] + (box[2]-box[0])/2 + pad, mid[1] + 12 + pad),
                                   radius=7, fill="white", outline=line_colour, width=2)
            _center_text(draw, mid, label, label_font, line_colour)

    # Latent constructs and observed indicators.
    item_boxes: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for construct in constructs:
        cx, cy = positions[construct]
        draw.ellipse((cx-latent_rx, cy-latent_ry, cx+latent_rx, cy+latent_ry), fill=_LATENT_FILL, outline=_INK, width=4)
        _center_text(draw, (cx, cy), construct, node_font)

        items = construct_map[construct]
        loadings = _loading_estimates(loading_table, construct)
        sorted_constructs = sorted(constructs, key=lambda name: positions[name][0])
        construct_index = sorted_constructs.index(construct)
        previous_x = positions[sorted_constructs[construct_index - 1]][0] if construct_index > 0 else 0
        next_x = positions[sorted_constructs[construct_index + 1]][0] if construct_index < len(sorted_constructs) - 1 else width
        region_left = 25 if construct_index == 0 else (previous_x + cx) / 2 + 10
        region_right = width - 25 if construct_index == len(sorted_constructs) - 1 else (cx + next_x) / 2 - 10
        region_width = max(region_right - region_left, 220)
        gap = 12
        preferred_box_w, box_h = 150, 46
        columns = max(1, min(len(items), int((region_width + gap) // (preferred_box_w + gap))))
        rows = int(math.ceil(len(items) / columns))
        box_w = min(preferred_box_w, (region_width - gap * max(columns - 1, 0)) / columns)
        box_w = max(box_w, 95)
        grid_w = columns * box_w + gap * max(columns - 1, 0)
        start_x = region_left + max((region_width - grid_w) / 2, 0)
        first_y = height - 125 - rows * (box_h + 16)
        for idx, item in enumerate(items):
            row_index, column_index = divmod(idx, columns)
            x1 = start_x + column_index * (box_w + gap)
            y1 = first_y + row_index * (box_h + 16)
            x2, y2 = x1 + box_w, y1 + box_h
            item_boxes[(construct, item)] = (x1, y1, x2, y2)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=7, fill=_ITEM_FILL, outline=_MUTED, width=2)
            _center_text(draw, ((x1+x2)/2, (y1+y2)/2), item, item_font)
            arrow_start = (cx, cy + latent_ry)
            arrow_end = ((x1+x2)/2, y1)
            _arrow(draw, arrow_start, arrow_end, fill=_MUTED, width=2)
            loading = loadings.get(item)
            if loading is not None:
                mid_x = (arrow_start[0] + arrow_end[0]) / 2
                mid_y = (arrow_start[1] + arrow_end[1]) / 2
                label = f"{loading:.2f}"
                draw.rounded_rectangle((mid_x-25, mid_y-13, mid_x+25, mid_y+13), radius=5, fill="white", outline="#D0D5DD")
                _center_text(draw, (mid_x, mid_y), label, small_font, _INK)

    if fit_table is not None and not fit_table.empty:
        row = fit_table.iloc[0]
        fit_parts = []
        for key, label in [("cfi", "CFI"), ("tli", "TLI"), ("rmsea", "RMSEA"), ("srmr", "SRMR"), ("srmr_approx", "SRMR")]:
            if key in row and pd.notna(row[key]):
                fit_parts.append(f"{label}={float(row[key]):.3f}")
        if fit_parts:
            fit_text = "Model fit: " + " | ".join(fit_parts)
            _center_text(draw, (width/2, height-55), fit_text, label_font, _INK)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def attach_latent_variable_figures(result: AnalysisResult) -> AnalysisResult:
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
            )
    return result
