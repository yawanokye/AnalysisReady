from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class AuditEntry:
    action: str
    variable: str | None = None
    details: str = ""
    justification: str = ""
    before_n: int | None = None
    after_n: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    method: str
    summary: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    treatment_log: list[AuditEntry] = field(default_factory=list)
    reproducible_code: str = ""

    def add_table(self, name: str, table: pd.DataFrame) -> None:
        self.tables[name] = table
