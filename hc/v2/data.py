from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from zizzii_features import build_team_features

from .config import V2Config

ROOT = Path(__file__).resolve().parents[2]
NCAA_DATA = ROOT / "ncaa-data"
EXTERNAL_DATA = ROOT / "external-data"


def _canonicalize_matchups(df: pd.DataFrame, team1_col: str, team2_col: str, prob_col: str) -> pd.DataFrame:
    base = df.copy()
    t1 = base[team1_col].astype(int)
    t2 = base[team2_col].astype(int)
    swap = t1 > t2
    base["T1"] = np.where(swap, t2, t1)
    base["T2"] = np.where(swap, t1, t2)
    base[prob_col] = np.where(swap, 1.0 - base[prob_col].astype(float), base[prob_col].astype(float))
    keep = ["Season", "T1", "T2", prob_col]
    return base[keep].drop_duplicates(subset=["Season", "T1", "T2"], keep="last")


def load_tournament_results(gender: str) -> pd.DataFrame:
    compact = pd.read_csv(NCAA_DATA / f"{gender}NCAATourneyCompactResults.csv")
    compact["Season"] = compact["Season"].astype(int)
    compact["T1"] = compact[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    compact["T2"] = compact[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    compact["Label"] = (compact["WTeamID"] == compact["T1"]).astype(int)
    compact["Margin"] = np.where(
        compact["WTeamID"] == compact["T1"],
        compact["WScore"] - compact["LScore"],
        compact["LScore"] - compact["WScore"],
    ).astype(float)
    return compact[["Season", "DayNum", "T1", "T2", "Label", "Margin"]].sort_values(["Season", "DayNum", "T1", "T2"])


@lru_cache(maxsize=8)
def _cached_team_features(gender: str, include_external: bool) -> pd.DataFrame:
    return build_team_features(gender, include_external=include_external)


def _strength_pack_frame(features: pd.DataFrame, feature_pack: str) -> pd.DataFrame:
    if feature_pack == "strength_full":
        frame = pd.DataFrame(
            {
                "Season": features["Season"],
                "TeamID": features["TeamID"],
                "SeedNum": pd.to_numeric(features.get("SeedNum"), errors="coerce"),
                "Elo": pd.to_numeric(features.get("Elo"), errors="coerce"),
                "SOS": pd.to_numeric(features.get("SOS"), errors="coerce"),
                "WinRate": pd.to_numeric(features.get("WinRate"), errors="coerce"),
                "AvgMargin": pd.to_numeric(features.get("AvgMargin"), errors="coerce"),
                "Top50WinRate": pd.to_numeric(features.get("Top50WinRate"), errors="coerce"),
                "Last30WinRate": pd.to_numeric(features.get("Last30WinRate"), errors="coerce"),
                "SeedPriorExpectedWins": pd.to_numeric(features.get("SeedPriorExpectedWins"), errors="coerce"),
                "StrengthNet": pd.to_numeric(features.get("AdjNetRtg"), errors="coerce"),
                "StrengthOff": pd.to_numeric(features.get("AdjOffRtg"), errors="coerce"),
                "StrengthDef": -pd.to_numeric(features.get("AdjDefRtg"), errors="coerce"),
                "StrengthTempo": pd.to_numeric(features.get("Tempo"), errors="coerce"),
                "StrengthSOS": pd.to_numeric(features.get("SOS"), errors="coerce"),
                "StrengthTop50": pd.to_numeric(features.get("Top50WinRate"), errors="coerce"),
                "StrengthPath": -pd.to_numeric(features.get("PathDifficultyEarly"), errors="coerce"),
                "StrengthMomentum": pd.to_numeric(features.get("MomentumAdjNetRtg"), errors="coerce"),
                "StrengthOffMomentum": (
                    pd.to_numeric(features.get("RecentEffAdjOffRtg"), errors="coerce")
                    - pd.to_numeric(features.get("AdjOffRtg"), errors="coerce")
                ),
                "StrengthDefMomentum": (
                    pd.to_numeric(features.get("AdjDefRtg"), errors="coerce")
                    - pd.to_numeric(features.get("RecentEffAdjDefRtg"), errors="coerce")
                ),
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "Season": features["Season"],
                "TeamID": features["TeamID"],
                "SeedNum": pd.to_numeric(features.get("SeedNum"), errors="coerce"),
                "Elo": pd.to_numeric(features.get("Elo"), errors="coerce"),
                "SOS": pd.to_numeric(features.get("SOS"), errors="coerce"),
                "WinRate": pd.to_numeric(features.get("WinRate"), errors="coerce"),
                "AvgMargin": pd.to_numeric(features.get("AvgMargin"), errors="coerce"),
                "Top50WinRate": pd.to_numeric(features.get("Top50WinRate"), errors="coerce"),
                "Last30WinRate": pd.to_numeric(features.get("Last30WinRate"), errors="coerce"),
                "SeedPriorExpectedWins": pd.to_numeric(features.get("SeedPriorExpectedWins"), errors="coerce"),
                "StrengthNet": pd.to_numeric(features.get("Recent30EffAdjNetRtg"), errors="coerce"),
                "StrengthOff": pd.to_numeric(features.get("Recent30EffAdjOffRtg"), errors="coerce"),
                "StrengthDef": -pd.to_numeric(features.get("Recent30EffAdjDefRtg"), errors="coerce"),
                "StrengthTempo": pd.to_numeric(features.get("Tempo"), errors="coerce"),
                "StrengthSOS": pd.to_numeric(features.get("Last30SOS"), errors="coerce"),
                "StrengthTop50": pd.to_numeric(features.get("Last30Top50WinRate"), errors="coerce"),
                "StrengthPath": -pd.to_numeric(features.get("PathDifficultyEarly"), errors="coerce"),
                "StrengthMomentum": pd.to_numeric(features.get("Momentum30AdjNetRtg"), errors="coerce"),
                "StrengthOffMomentum": (
                    pd.to_numeric(features.get("Recent30EffAdjOffRtg"), errors="coerce")
                    - pd.to_numeric(features.get("AdjOffRtg"), errors="coerce")
                ),
                "StrengthDefMomentum": (
                    pd.to_numeric(features.get("AdjDefRtg"), errors="coerce")
                    - pd.to_numeric(features.get("Recent30EffAdjDefRtg"), errors="coerce")
                ),
            }
        )
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def _external_base_frame(features: pd.DataFrame, gender: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Season": features["Season"],
            "TeamID": features["TeamID"],
            "SeedNum": pd.to_numeric(features.get("SeedNum"), errors="coerce"),
            "SeedPriorExpectedWins": pd.to_numeric(features.get("SeedPriorExpectedWins"), errors="coerce"),
            "Elo": pd.to_numeric(features.get("Elo"), errors="coerce"),
            "WinRate": pd.to_numeric(features.get("WinRate"), errors="coerce"),
            "AvgMargin": pd.to_numeric(features.get("AvgMargin"), errors="coerce"),
            "Top50WinRate": pd.to_numeric(features.get("Top50WinRate"), errors="coerce"),
            "Last30WinRate": pd.to_numeric(features.get("Last30WinRate"), errors="coerce"),
            "CloseGameWinRate": pd.to_numeric(features.get("CloseGameWinRate"), errors="coerce"),
            "CloseGameMargin": pd.to_numeric(features.get("CloseGameMargin"), errors="coerce"),
            "Last30SOS": pd.to_numeric(features.get("Last30SOS"), errors="coerce"),
            "PageRank": pd.to_numeric(features.get("PageRank"), errors="coerce"),
            "ExternalCompositeStrength": pd.to_numeric(features.get("ExtCompositeStrength"), errors="coerce"),
            "ExternalFallbackElo": pd.to_numeric(features.get("Ext_FallbackElo"), errors="coerce"),
            "ExternalFallbackSOS": pd.to_numeric(features.get("Ext_FallbackSOS"), errors="coerce"),
            "ExternalFallbackMargin": pd.to_numeric(features.get("Ext_FallbackAvgMargin"), errors="coerce"),
        }
    )
    if gender == "M":
        frame = frame.assign(
            ExternalBPIStrength=-pd.to_numeric(features.get("Ext_PublicBPIRank"), errors="coerce"),
            ExternalPOMStrength=-pd.to_numeric(features.get("Ext_PublicPOMRank"), errors="coerce"),
            ExternalNETStrength=-pd.to_numeric(features.get("Ext_PublicNETRank"), errors="coerce"),
            ExternalWABStrength=-pd.to_numeric(features.get("Ext_PublicWABRank"), errors="coerce"),
            ExternalELORankStrength=-pd.to_numeric(features.get("Ext_PublicELORank"), errors="coerce"),
            ExternalSORStrength=-pd.to_numeric(features.get("Ext_PublicSORRank"), errors="coerce"),
            ExternalTRankStrength=-pd.to_numeric(features.get("Ext_PublicTRankRank"), errors="coerce"),
            MasseyPOMStrength=-pd.to_numeric(features.get("Rank_POM"), errors="coerce"),
            MasseyMORStrength=-pd.to_numeric(features.get("Rank_MOR"), errors="coerce"),
            MasseyNETStrength=-pd.to_numeric(features.get("Rank_NET"), errors="coerce"),
        )
    else:
        frame = frame.assign(
            ExternalNETStrength=-pd.to_numeric(features.get("Ext_PublicNETRank"), errors="coerce"),
            ExternalRPIStrength=-pd.to_numeric(features.get("Ext_PublicRPIRank"), errors="coerce"),
            ExternalPredRPIStrength=-pd.to_numeric(features.get("Ext_PublicPredRPIRank"), errors="coerce"),
            ExternalELORankStrength=-pd.to_numeric(features.get("Ext_PublicELORank"), errors="coerce"),
            ExternalAPStrength=-pd.to_numeric(features.get("Ext_PublicAPRank"), errors="coerce"),
            ExternalCoachesStrength=-pd.to_numeric(features.get("Ext_PublicCoachesRank"), errors="coerce"),
        )
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def load_team_features(config: V2Config) -> pd.DataFrame:
    include_external = config.feature_pack not in {"strength_full", "strength_recent"}
    features = _cached_team_features(config.gender, include_external=include_external).copy()
    if config.feature_pack in {"strength_full", "strength_recent"}:
        features = _strength_pack_frame(features, config.feature_pack)
    elif config.feature_pack in {"external_base", "external_base_pruned"}:
        features = _external_base_frame(features, config.gender)
    feature_names = config.resolved_features()
    cols = ["Season", "TeamID", *feature_names]
    existing = [c for c in cols if c in features.columns]
    frame = features[existing].copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    for col in existing[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_v2_dataset(config: V2Config) -> pd.DataFrame:
    results = load_tournament_results(config.gender)
    team_features = load_team_features(config)
    t1 = team_features.rename(columns={"TeamID": "T1", **{col: f"T1_{col}" for col in config.resolved_features() if col in team_features.columns}})
    t2 = team_features.rename(columns={"TeamID": "T2", **{col: f"T2_{col}" for col in config.resolved_features() if col in team_features.columns}})
    merged = results.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")
    diff_columns: list[str] = []
    for feature in config.resolved_features():
        t1_col = f"T1_{feature}"
        t2_col = f"T2_{feature}"
        if t1_col in merged.columns and t2_col in merged.columns:
            diff_col = f"{feature}_diff"
            merged[diff_col] = merged[t1_col] - merged[t2_col]
            diff_columns.append(diff_col)

    for interaction_name, left_feature, right_feature in config.resolved_interactions():
        t1_left = f"T1_{left_feature}"
        t1_right = f"T1_{right_feature}"
        t2_left = f"T2_{left_feature}"
        t2_right = f"T2_{right_feature}"
        if all(column in merged.columns for column in (t1_left, t1_right, t2_left, t2_right)):
            diff_col = f"{interaction_name}_diff"
            merged[diff_col] = (merged[t1_left] * merged[t1_right]) - (merged[t2_left] * merged[t2_right])
            diff_columns.append(diff_col)
    dataset = merged[["Season", "DayNum", "T1", "T2", "Label", "Margin", *diff_columns]].copy()
    keep_diff = [col for col in diff_columns if dataset[col].notna().any()]
    dataset = dataset[["Season", "DayNum", "T1", "T2", "Label", "Margin", *keep_diff]]
    dataset = dataset.sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)
    return dataset


def load_historical_sportsbook_probs(gender: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pattern = f"{gender}MatchupOdds_*.csv"
    for path in sorted(EXTERNAL_DATA.glob(pattern)):
        frame = pd.read_csv(path)
        if frame.empty or "MarketProb" not in frame.columns:
            continue
        season = pd.to_numeric(frame.get("Season"), errors="coerce")
        usable = frame.loc[season.notna()].copy()
        if usable.empty:
            continue
        usable["Season"] = season.loc[usable.index].astype(int)
        rows.append(_canonicalize_matchups(usable, "T1", "T2", "MarketProb").rename(columns={"MarketProb": "sportsbook_prob"}))
    if not rows:
        return pd.DataFrame(columns=["Season", "T1", "T2", "sportsbook_prob"])
    combined = pd.concat(rows, ignore_index=True)
    return combined.drop_duplicates(subset=["Season", "T1", "T2"], keep="last")


def load_prediction_market_probs(gender: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    filenames = [
        EXTERNAL_DATA / f"{gender}KalshiPredictionMarketOdds_2026.csv",
        EXTERNAL_DATA / f"{gender}PolymarketPredictionMarketOdds_2026.csv",
    ]
    for path in filenames:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        season = pd.to_numeric(frame.get("Season"), errors="coerce")
        usable = frame.loc[season.notna() & frame.get("MarketProb").notna()].copy()
        if usable.empty:
            continue
        usable["Season"] = season.loc[usable.index].astype(int)
        rows.append(_canonicalize_matchups(usable, "T1", "T2", "MarketProb").rename(columns={"MarketProb": "prediction_market_prob"}))
    if not rows:
        return pd.DataFrame(columns=["Season", "T1", "T2", "prediction_market_prob"])
    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby(["Season", "T1", "T2"], as_index=False)["prediction_market_prob"].mean()
    return grouped


def attach_market_columns(dataset: pd.DataFrame, gender: str) -> pd.DataFrame:
    sportsbook = load_historical_sportsbook_probs(gender)
    merged = dataset.merge(sportsbook, on=["Season", "T1", "T2"], how="left")
    prediction_market = load_prediction_market_probs(gender)
    merged = merged.merge(prediction_market, on=["Season", "T1", "T2"], how="left")
    return merged
