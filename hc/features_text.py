from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hc.constants import CACHE_DIR, TEXT_COMPONENTS


def text_cache_path(gender: str, text_dim: int) -> Path:
    return CACHE_DIR / f"text_embeddings_{gender}_{text_dim}d.parquet"


def load_text_embeddings(gender: str, text_dim: int) -> pd.DataFrame:
    path = text_cache_path(gender, text_dim)
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID"])
    return pd.read_parquet(path)


def _text_component_bases(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {}
    for component in TEXT_COMPONENTS:
        prefix = f"Text{component}_"
        cols = [column for column in frame.columns if column.startswith(prefix)]
        if cols:
            groups[component] = cols
    return groups


def attach_text_matchup_features(df: pd.DataFrame, text_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    if text_df is None or text_df.empty:
        return df, []

    merged = df.copy()
    t1_text = text_df.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2_text = text_df.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    merged = merged.merge(t1_text, on=["Season", "T1"], how="left")
    merged = merged.merge(t2_text, on=["Season", "T2"], how="left")

    feature_cols: list[str] = []
    component_map = _text_component_bases(text_df)
    for component, base_cols in component_map.items():
        for base_col in base_cols:
            suffix = base_col.split("_", 1)[1]
            t1_col = f"T1_{base_col}"
            t2_col = f"T2_{base_col}"
            diff_col = f"D_Text{suffix}"
            abs_col = f"Abs_Text{suffix}"
            mean_col = f"Mean_Text{suffix}"
            merged[diff_col] = merged[t1_col].fillna(0.0) - merged[t2_col].fillna(0.0)
            merged[abs_col] = merged[diff_col].abs()
            merged[mean_col] = (merged[t1_col].fillna(0.0) + merged[t2_col].fillna(0.0)) / 2.0
            feature_cols.extend([diff_col, abs_col, mean_col])

    if {"T1_TextDocCount", "T2_TextDocCount"}.issubset(merged.columns):
        merged["D_TextDocCount"] = merged["T1_TextDocCount"].fillna(0.0) - merged["T2_TextDocCount"].fillna(0.0)
        merged["TextDocCountTotal"] = merged["T1_TextDocCount"].fillna(0.0) + merged["T2_TextDocCount"].fillna(0.0)
        merged["TextDocCountMin"] = merged[["T1_TextDocCount", "T2_TextDocCount"]].fillna(0.0).min(axis=1)
        feature_cols.extend(["D_TextDocCount", "TextDocCountTotal", "TextDocCountMin"])

    if {"T1_TextWindowDocCount", "T2_TextWindowDocCount"}.issubset(merged.columns):
        merged["D_TextWindowDocCount"] = merged["T1_TextWindowDocCount"].fillna(0.0) - merged["T2_TextWindowDocCount"].fillna(0.0)
        feature_cols.append("D_TextWindowDocCount")

    return merged, list(dict.fromkeys(feature_cols))
