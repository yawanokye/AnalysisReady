from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).with_name("path_editor_component") / "frontend" / "build"
_path_editor = components.declare_component("statready_path_editor", path=str(_COMPONENT_PATH))


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
