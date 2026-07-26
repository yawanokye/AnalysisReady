from __future__ import annotations

import numpy as np
import pandas as pd

from .models import AuditEntry


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, AuditEntry]:
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    return cleaned, AuditEntry(
        action="Dropped exact duplicate rows",
        details=f"Removed {before - len(cleaned)} exact duplicate row(s).",
        justification="Exact duplicate records can over-weight observations. The original dataset remains preserved.",
        before_n=before,
        after_n=len(cleaned),
    )


def impute_missing(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str,
) -> tuple[pd.DataFrame, list[AuditEntry]]:
    cleaned = df.copy()
    entries: list[AuditEntry] = []
    for column in columns:
        if column not in cleaned.columns:
            continue
        missing = int(cleaned[column].isna().sum())
        if missing == 0:
            continue

        if strategy == "median":
            value = pd.to_numeric(cleaned[column], errors="coerce").median()
        elif strategy == "mean":
            value = pd.to_numeric(cleaned[column], errors="coerce").mean()
        elif strategy == "mode":
            mode = cleaned[column].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else np.nan
        else:
            raise ValueError("Strategy must be mean, median, or mode.")

        if pd.isna(value):
            continue
        cleaned[column] = cleaned[column].fillna(value)
        entries.append(AuditEntry(
            action=f"{strategy.title()} imputation",
            variable=column,
            details=f"Replaced {missing} missing value(s) with {value!r}.",
            justification="Applied only after user selection. Sensitivity analysis against complete-case results is recommended.",
            before_n=len(df),
            after_n=len(cleaned),
        ))
    return cleaned, entries


def winsorise(
    df: pd.DataFrame,
    columns: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, list[AuditEntry]]:
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    cleaned = df.copy()
    entries: list[AuditEntry] = []
    for column in columns:
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        if numeric.notna().sum() < 4:
            continue
        lower = float(numeric.quantile(lower_quantile))
        upper = float(numeric.quantile(upper_quantile))
        changed = int(((numeric < lower) | (numeric > upper)).sum())
        cleaned[column] = numeric.clip(lower, upper)
        entries.append(AuditEntry(
            action="Winsorisation",
            variable=column,
            details=f"Capped {changed} value(s) at the {lower_quantile:.0%} and {upper_quantile:.0%} quantiles ({lower:.4g}, {upper:.4g}).",
            justification="A bounded sensitivity treatment for extreme values. The untreated model must remain available for comparison.",
            before_n=len(df),
            after_n=len(cleaned),
        ))
    return cleaned, entries


def log1p_transform(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[AuditEntry]]:
    cleaned = df.copy()
    entries: list[AuditEntry] = []
    for column in columns:
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        if numeric.dropna().empty or numeric.min(skipna=True) < 0:
            continue
        new_name = f"log1p_{column}"
        cleaned[new_name] = np.log1p(numeric)
        entries.append(AuditEntry(
            action="Log1p transformation",
            variable=column,
            details=f"Created {new_name}; original values were retained.",
            justification="The transformation may improve scale or functional form for non-negative variables. It must be theoretically defensible.",
            before_n=len(df),
            after_n=len(cleaned),
        ))
    return cleaned, entries
