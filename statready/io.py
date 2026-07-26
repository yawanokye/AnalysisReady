from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def list_excel_sheets(file_obj: BinaryIO | bytes) -> list[str]:
    data = file_obj if isinstance(file_obj, bytes) else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    with pd.ExcelFile(BytesIO(data)) as workbook:
        return workbook.sheet_names


def load_tabular_file(
    file_obj: BinaryIO | bytes,
    filename: str,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    data = file_obj if isinstance(file_obj, bytes) else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    stream = BytesIO(data)
    if ext == ".csv":
        try:
            return pd.read_csv(stream)
        except UnicodeDecodeError:
            stream.seek(0)
            return pd.read_csv(stream, encoding="latin-1")
    return pd.read_excel(stream, sheet_name=sheet_name)


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    cleaned = df.copy()
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}

    for original in cleaned.columns:
        base = (
            str(original)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        base = "".join(ch for ch in base if ch.isalnum() or ch == "_") or "variable"
        count = seen.get(base, 0)
        seen[base] = count + 1
        final = base if count == 0 else f"{base}_{count + 1}"
        mapping[str(original)] = final

    cleaned.columns = [mapping[str(c)] for c in df.columns]
    return cleaned, mapping
