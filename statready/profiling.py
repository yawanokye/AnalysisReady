from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


MISSING_CODES = {
    "", " ", "na", "n/a", "nan", "none", "null", "missing", ".", "-", "--",
    "999", "9999", "-999", "-9999",
}


def normalise_missing_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    changes: list[dict[str, object]] = []

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            original = cleaned[column].copy()
            stripped = cleaned[column].astype(str).str.strip()
            mask = stripped.str.lower().isin(MISSING_CODES)
            changed_count = int(mask.sum())
            if changed_count:
                cleaned.loc[mask, column] = np.nan
                changes.append({
                    "variable": column,
                    "action": "Normalised missing-value codes",
                    "values_changed": changed_count,
                })
    return cleaned, pd.DataFrame(changes)


def dataset_profile(df: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)

    overview = pd.DataFrame([
        {
            "rows": len(df),
            "columns": df.shape[1],
            "numeric_variables": numeric.shape[1],
            "non_numeric_variables": categorical.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
        }
    ])

    variables: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        non_missing = series.dropna()
        variables.append({
            "variable": column,
            "dtype": str(series.dtype),
            "non_missing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique": int(non_missing.nunique()),
            "constant": bool(non_missing.nunique() <= 1),
            "sample_values": ", ".join(map(str, non_missing.head(3).tolist())),
        })

    missingness = (
        df.isna().sum()
        .rename("missing")
        .to_frame()
        .assign(missing_percent=lambda x: (x["missing"] / max(len(df), 1) * 100).round(2))
        .reset_index(names="variable")
        .sort_values("missing", ascending=False)
    )

    numeric_summary = numeric.describe(include="all").T.reset_index(names="variable") if not numeric.empty else pd.DataFrame()

    return {
        "overview": overview,
        "variables": pd.DataFrame(variables),
        "missingness": missingness,
        "numeric_summary": numeric_summary,
    }


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.select_dtypes(include=np.number).columns:
        values = df[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        z_count = 0
        if values.std(ddof=0) > 0:
            z_count = int((np.abs(stats.zscore(values, nan_policy="omit")) > 3).sum())
        rows.append({
            "variable": column,
            "iqr_outliers": int(mask.sum()),
            "z_score_outliers": z_count,
            "iqr_lower_bound": float(lower),
            "iqr_upper_bound": float(upper),
        })
    return pd.DataFrame(rows)
