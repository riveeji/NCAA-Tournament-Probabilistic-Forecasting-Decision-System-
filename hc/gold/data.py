from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from zizzii_features import build_team_features

from .config import GoldConfig
from .ratings import build_gold_ratings

ROOT = Path(__file__).resolve().parents[2]
NCAA_DATA = ROOT / "ncaa-data"


def load_tournament_results(gender: str) -> pd.DataFrame:
    compact = pd.read_csv(NCAA_DATA / f"{gender}NCAATourneyCompactResults.csv")
    detailed_path = NCAA_DATA / f"{gender}NCAATourneyDetailedResults.csv"
    detailed = (
        pd.read_csv(detailed_path, usecols=["Season", "DayNum", "WTeamID", "LTeamID", "NumOT"])
        if detailed_path.exists()
        else pd.DataFrame()
    )
    compact["Season"] = compact["Season"].astype(int)
    compact["T1"] = compact[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    compact["T2"] = compact[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    compact["Label"] = (compact["WTeamID"] == compact["T1"]).astype(int)
    compact["Margin"] = np.where(
        compact["WTeamID"] == compact["T1"],
        compact["WScore"] - compact["LScore"],
        compact["LScore"] - compact["WScore"],
    ).astype(float)
    base = compact[["Season", "DayNum", "T1", "T2", "Label", "Margin"]].copy()
    if not detailed.empty:
        detailed["T1"] = detailed[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
        detailed["T2"] = detailed[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
        detailed = detailed[["Season", "DayNum", "T1", "T2", "NumOT"]].drop_duplicates(["Season", "DayNum", "T1", "T2"])
        base = base.merge(detailed, on=["Season", "DayNum", "T1", "T2"], how="left")
    else:
        base["NumOT"] = 0
    base["NumOT"] = pd.to_numeric(base["NumOT"], errors="coerce").fillna(0.0)
    return base.sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _cached_base_features(gender: str) -> pd.DataFrame:
    frame = build_team_features(gender, include_external=True).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def load_gold_team_features(config: GoldConfig) -> pd.DataFrame:
    base = _cached_base_features(config.gender)
    ratings = build_gold_ratings(config.gender, config.resolved_rating_source_profile())
    keep = base[["Season", "TeamID"]].copy()
    if "SeedNum" in base.columns:
        keep["SeedNum"] = pd.to_numeric(base["SeedNum"], errors="coerce")
    if "SeedPriorExpectedWins" in base.columns:
        keep["SeedPriorExpectedWins"] = pd.to_numeric(base["SeedPriorExpectedWins"], errors="coerce")
    merged = keep.merge(ratings, on=["Season", "TeamID"], how="left", suffixes=("", "_rating"))
    for column in merged.columns:
        if column not in {"Season", "TeamID"}:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    aggregated = (
        merged.groupby(["Season", "TeamID"], as_index=False)
        .agg({column: "mean" for column in merged.columns if column not in {"Season", "TeamID"}})
    )
    return aggregated


def build_gold_dataset(config: GoldConfig) -> pd.DataFrame:
    results = load_tournament_results(config.gender)
    features = load_gold_team_features(config)
    team_feature_names = list(dict.fromkeys(config.resolved_candidate_features()))

    t1 = features.rename(columns={"TeamID": "T1", **{col: f"T1_{col}" for col in team_feature_names if col in features.columns}})
    t2 = features.rename(columns={"TeamID": "T2", **{col: f"T2_{col}" for col in team_feature_names if col in features.columns}})
    merged = results.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")

    engineered_columns: list[str] = []
    for feature in team_feature_names:
        t1_col = f"T1_{feature}"
        t2_col = f"T2_{feature}"
        if t1_col in merged.columns and t2_col in merged.columns:
            diff_col = f"{feature}_diff"
            merged[diff_col] = merged[t1_col] - merged[t2_col]
            engineered_columns.append(diff_col)

    if "T1_SeedNum" in merged.columns and "T2_SeedNum" in merged.columns:
        merged["SeedAbsGap"] = (merged["T1_SeedNum"] - merged["T2_SeedNum"]).abs()
        merged["SeedPairProduct"] = merged["T1_SeedNum"] * merged["T2_SeedNum"]
        engineered_columns.extend(["SeedAbsGap", "SeedPairProduct"])

    for interaction_name, left_feature, right_feature in config.resolved_interactions():
        t1_left = f"T1_{left_feature}"
        t1_right = f"T1_{right_feature}"
        t2_left = f"T2_{left_feature}"
        t2_right = f"T2_{right_feature}"
        if all(column in merged.columns for column in (t1_left, t1_right, t2_left, t2_right)):
            diff_col = f"{interaction_name}_diff"
            merged[diff_col] = (merged[t1_left] * merged[t1_right]) - (merged[t2_left] * merged[t2_right])
            engineered_columns.append(diff_col)

    merged = merged[["Season", "DayNum", "T1", "T2", "Label", "Margin", "NumOT", *engineered_columns]].copy()
    return merged.sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)
