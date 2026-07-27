from __future__ import annotations

from typing import Any

import streamlit.components.v1 as components

from .path_editor_assets import resolve_component_path

_COMPONENT_PATH = resolve_component_path()
_path_editor = components.declare_component("statready_path_editor", path=str(_COMPONENT_PATH))


def component_asset_status() -> dict[str, str | bool]:
    """Return deployment diagnostics for the custom component asset."""
    index_path = _COMPONENT_PATH / "index.html"
    return {
        "component_path": str(_COMPONENT_PATH),
        "index_exists": index_path.is_file(),
        "using_packaged_asset": _COMPONENT_PATH.name == "path_editor_assets",
    }


def path_editor(
    nodes: list[str],
    edges: list[tuple[str, str]],
    positions: dict[str, dict[str, float]] | None = None,
    height: int = 680,
    key: str | None = None,
) -> dict[str, dict[str, float]]:
    default = positions or {}
    value: Any = _path_editor(
        nodes=nodes,
        edges=[{"source": source, "target": target} for source, target in edges],
        positions=default,
        height=height,
        default=default,
        key=key,
    )
    return value if isinstance(value, dict) else default
