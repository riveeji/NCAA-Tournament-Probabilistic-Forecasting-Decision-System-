from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from zizzii_features import build_team_features

from .config import JIBaseConfig
from .internal_ratings import load_internal_ratings
from .ratings import build_ji_ratings
from .women_upstream import EXTERNAL_CONSENSUS_SOURCE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
NCAA_DATA = ROOT / "ncaa-data"


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
    return compact[["Season", "DayNum", "T1", "T2", "Label", "Margin"]].sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)


@lru_cache(maxsize=8)
def _cached_base_team_frame(gender: str, include_external: bool) -> pd.DataFrame:
    frame = build_team_features(gender, include_external=False).copy()
    if include_external:
        frame = build_team_features(gender, include_external=True).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def build_team_profile_v2(gender: str, women_ranking_provider: str = "internal_fallback") -> pd.DataFrame:
    include_external = gender == "W" and women_ranking_provider in {"external_consensus_v1", "external_consensus_v2"}
    base = _cached_base_team_frame(gender, include_external)
    profile = base[["Season", "TeamID"]].copy()
    profile["SeedNum"] = pd.to_numeric(base.get("SeedNum"), errors="coerce").fillna(17.0)
    profile["wpct"] = pd.to_numeric(base.get("WinRate"), errors="coerce")
    profile["margin"] = pd.to_numeric(base.get("AvgMargin"), errors="coerce")
    profile["oeff"] = pd.to_numeric(base.get("AdjOffRtg"), errors="coerce").fillna(pd.to_numeric(base.get("OffRtg"), errors="coerce"))
    profile["deff"] = pd.to_numeric(base.get("AdjDefRtg"), errors="coerce").fillna(pd.to_numeric(base.get("DefRtg"), errors="coerce"))
    profile["neff"] = pd.to_numeric(base.get("AdjNetRtg"), errors="coerce").fillna(pd.to_numeric(base.get("NetRtg"), errors="coerce"))
    profile["efg"] = pd.to_numeric(base.get("eFG"), errors="coerce")
    profile["tor"] = pd.to_numeric(base.get("TOVPct"), errors="coerce")
    profile["orpct"] = pd.to_numeric(base.get("ORBPct"), errors="coerce")
    profile["ftr"] = pd.to_numeric(base.get("FTR"), errors="coerce")
    profile["oefg"] = pd.to_numeric(base.get("OppEFG"), errors="coerce")
    profile["otor"] = pd.to_numeric(base.get("OppTOVPct"), errors="coerce")
    profile["oorpct"] = pd.to_numeric(base.get("OppORBPct"), errors="coerce")
    profile["oftr"] = pd.to_numeric(base.get("OppFTR"), errors="coerce")
    profile["pace"] = pd.to_numeric(base.get("Tempo"), errors="coerce")
    profile["stlpg"] = pd.to_numeric(base.get("Stl"), errors="coerce")
    profile["blkpg"] = pd.to_numeric(base.get("Blk"), errors="coerce")
    profile["drbpg"] = (1.0 - pd.to_numeric(base.get("OppORBPct"), errors="coerce")).clip(lower=0.0)
    profile["Elo"] = pd.to_numeric(base.get("Elo"), errors="coerce")
    profile["SOS"] = pd.to_numeric(base.get("SOS"), errors="coerce")
    profile["RPIStyle"] = pd.to_numeric(base.get("RPIStyle"), errors="coerce").fillna(0.0)
    profile["PageRank"] = pd.to_numeric(base.get("PageRank"), errors="coerce").fillna(0.0)
    profile["Games"] = pd.to_numeric(base.get("Games"), errors="coerce")
    profile["Top50WinRate"] = pd.to_numeric(base.get("Top50WinRate"), errors="coerce").fillna(0.0)
    profile["Top100WinRate"] = pd.to_numeric(base.get("Top100WinRate"), errors="coerce").fillna(0.0)
    profile["ConfMeanElo"] = pd.to_numeric(base.get("ConfMeanElo"), errors="coerce").fillna(profile["Elo"])
    profile["AvgBlkDiff"] = pd.to_numeric(base.get("Blk"), errors="coerce") - pd.to_numeric(base.get("OppFTR"), errors="coerce").fillna(0.0)
    for column in EXTERNAL_CONSENSUS_SOURCE_COLUMNS:
        if column in base.columns:
            profile[column] = pd.to_numeric(base.get(column), errors="coerce")
    internal_ratings = load_internal_ratings(gender, profile[["Season", "TeamID"]])
    profile = profile.merge(internal_ratings, on=["Season", "TeamID"], how="left")
    profile["CarryElo"] = pd.to_numeric(profile["CarryElo"], errors="coerce").fillna(1500.0)
    profile["CarryElo80"] = pd.to_numeric(profile["CarryElo80"], errors="coerce").fillna(1500.0)
    profile["CarryElo85"] = pd.to_numeric(profile["CarryElo85"], errors="coerce").fillna(1500.0)
    profile["Colley"] = pd.to_numeric(profile["Colley"], errors="coerce").fillna(0.5)
    profile["ColleyNC"] = pd.to_numeric(profile["ColleyNC"], errors="coerce").fillna(0.5)
    profile["SRS"] = pd.to_numeric(profile["SRS"], errors="coerce").fillna(0.0)
    profile["SRSClip15"] = pd.to_numeric(profile["SRSClip15"], errors="coerce").fillna(0.0)
    profile["SRSClip20"] = pd.to_numeric(profile["SRSClip20"], errors="coerce").fillna(0.0)
    profile["EffSRS"] = pd.to_numeric(profile["EffSRS"], errors="coerce").fillna(0.0)
    profile["Quality"] = np.nan
    return profile


def load_ji_team_features(config: JIBaseConfig) -> pd.DataFrame:
    return _cached_ji_team_features(config.gender, config.women_quality_profile, config.women_ranking_provider)


@lru_cache(maxsize=16)
def _cached_ji_team_features(gender: str, women_quality_profile: str, women_ranking_provider: str) -> pd.DataFrame:
    profile = build_team_profile_v2(gender, women_ranking_provider=women_ranking_provider)
    ratings = build_ji_ratings(
        profile,
        gender=gender,
        women_quality_profile=women_quality_profile,
        women_ranking_provider=women_ranking_provider,
    )
    merged = profile.merge(ratings, on=["Season", "TeamID"], how="left", suffixes=("", "_rating"))
    merged["Quality"] = pd.to_numeric(merged["Quality_rating"], errors="coerce").fillna(pd.to_numeric(merged["Quality"], errors="coerce"))
    if "Quality_rating" in merged.columns:
        merged = merged.drop(columns=["Quality_rating"])
    return merged


def build_submission_feature_frame(ids: pd.DataFrame, team_features: pd.DataFrame, config: JIBaseConfig) -> pd.DataFrame:
    team_feature_names = [
        "SeedNum",
        "Elo",
        "CarryElo",
        "CarryElo80",
        "CarryElo85",
        "Colley",
        "ColleyNC",
        "SRS",
        "SRSClip15",
        "SRSClip20",
        "EffSRS",
        "Quality",
        "WomenCompositeQuality",
        "WomenSeedStrength",
        "WomenQualityWinsStrength",
        "WomenQualityWinsStrengthV2",
        "WomenOpponentTournamentStrength",
        "WomenRimProtectionStrength",
        "WomenCompositeQualityV5",
        "WomenOpponentTournamentStrengthV2",
        "oeff",
        "deff",
        "neff",
        "efg",
        "tor",
        "orpct",
        "ftr",
        "pace",
        "harry_Rating",
        "QualityWins",
        "OpponentQualityTournamentRank",
        "AvgBlkDiff",
    ]
    t1 = team_features.rename(columns={"TeamID": "T1", **{col: f"T1_{col}" for col in team_feature_names if col in team_features.columns}})
    t2 = team_features.rename(columns={"TeamID": "T2", **{col: f"T2_{col}" for col in team_feature_names if col in team_features.columns}})
    merged = ids.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")

    diff_pairs = {
        "Delta_Seed": "SeedNum",
        "Delta_Elo": "Elo",
        "Delta_CarryElo": "CarryElo",
        "Delta_CarryElo80": "CarryElo80",
        "Delta_CarryElo85": "CarryElo85",
        "Delta_Colley": "Colley",
        "Delta_ColleyNC": "ColleyNC",
        "Delta_SRS": "SRS",
        "Delta_SRSClip15": "SRSClip15",
        "Delta_SRSClip20": "SRSClip20",
        "Delta_EffSRS": "EffSRS",
        "Delta_Quality": "Quality",
        "Delta_oeff": "oeff",
        "Delta_deff": "deff",
        "Delta_neff": "neff",
        "Delta_efg": "efg",
        "Delta_tor": "tor",
        "Delta_orpct": "orpct",
        "Delta_ftr": "ftr",
        "Delta_pace": "pace",
        "harry_Rating_diff": "harry_Rating",
        "QualityWins_diff": "QualityWins",
        "OpponentQualityTournamentRank_diff": "OpponentQualityTournamentRank",
        "AvgBlkDiff_diff": "AvgBlkDiff",
        "WomenCompositeQuality_diff": "WomenCompositeQuality",
        "Delta_WomenSeedStrength": "WomenSeedStrength",
        "Delta_WomenQualityWinsStrength": "WomenQualityWinsStrength",
        "Delta_WomenQualityWinsStrengthV2": "WomenQualityWinsStrengthV2",
        "Delta_WomenOpponentTournamentStrength": "WomenOpponentTournamentStrength",
        "Delta_WomenRimProtectionStrength": "WomenRimProtectionStrength",
        "Delta_WomenCompositeQualityV5": "WomenCompositeQualityV5",
        "Delta_WomenOpponentTournamentStrengthV2": "WomenOpponentTournamentStrengthV2",
    }
    for diff_col, base_col in diff_pairs.items():
        t1_col = f"T1_{base_col}"
        t2_col = f"T2_{base_col}"
        if t1_col in merged.columns and t2_col in merged.columns:
            merged[diff_col] = pd.to_numeric(merged[t1_col], errors="coerce").fillna(0.0) - pd.to_numeric(merged[t2_col], errors="coerce").fillna(0.0)

    if config.gender == "W" and config.alpha_profile == "quality_only_women_light":
        merged["QualityWins_diff"] = pd.to_numeric(merged.get("QualityWins_diff"), errors="coerce").fillna(0.0) * 0.80
        merged["OpponentQualityTournamentRank_diff"] = (
            pd.to_numeric(merged.get("OpponentQualityTournamentRank_diff"), errors="coerce").fillna(0.0) * 0.85
        )
    elif config.gender == "W" and config.feature_profile == "women_tossup_quality_conservative":
        tossup_mask = pd.to_numeric(merged.get("Delta_Seed"), errors="coerce").fillna(0.0).abs() <= 1.0
        merged.loc[tossup_mask, "QualityWins_diff"] = (
            pd.to_numeric(merged.loc[tossup_mask, "QualityWins_diff"], errors="coerce").fillna(0.0) * 0.75
        )
        merged.loc[tossup_mask, "OpponentQualityTournamentRank_diff"] = (
            pd.to_numeric(merged.loc[tossup_mask, "OpponentQualityTournamentRank_diff"], errors="coerce").fillna(0.0) * 0.80
        )

    merged["Seed_sum"] = pd.to_numeric(merged.get("T1_SeedNum"), errors="coerce").fillna(17.0) + pd.to_numeric(merged.get("T2_SeedNum"), errors="coerce").fillna(17.0)
    merged["Seed_prod"] = pd.to_numeric(merged.get("T1_SeedNum"), errors="coerce").fillna(17.0) * pd.to_numeric(merged.get("T2_SeedNum"), errors="coerce").fillna(17.0)
    merged["Seed_gap_abs"] = merged["Delta_Seed"].abs()
    merged["EloProb"] = 1.0 / (1.0 + np.power(10.0, -(merged["Delta_Elo"].fillna(0.0) / 400.0)))
    merged["strength_blend"] = 0.45 * merged["Delta_Quality"].fillna(0.0) + 0.35 * merged["Delta_Elo"].fillna(0.0) + 0.20 * merged["Delta_neff"].fillna(0.0)
    quality_wins_diff = (
        pd.to_numeric(merged["QualityWins_diff"], errors="coerce").fillna(0.0)
        if "QualityWins_diff" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    opp_quality_rank_diff = (
        pd.to_numeric(merged["OpponentQualityTournamentRank_diff"], errors="coerce").fillna(0.0)
        if "OpponentQualityTournamentRank_diff" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["strength_blend_alt"] = (
        0.35 * merged["Delta_Quality"].fillna(0.0)
        + 0.25 * merged["Delta_Elo"].fillna(0.0)
        + 0.25 * merged["Delta_neff"].fillna(0.0)
        + 0.15 * quality_wins_diff
    )
    seed_gap_scale = 1.0 + merged["Seed_gap_abs"].fillna(0.0)
    close_game_weight = 1.0 / seed_gap_scale
    upset_alpha = (
        0.45 * merged["Delta_Quality"].fillna(0.0)
        + 0.25 * merged["Delta_Elo"].fillna(0.0)
        + 0.15 * quality_wins_diff
        + 0.15 * opp_quality_rank_diff
    )
    merged["CloseGameStrength"] = (
        0.55 * merged["strength_blend"].fillna(0.0) + 0.45 * upset_alpha
    ) * close_game_weight
    merged["UpsetPressure"] = upset_alpha * close_game_weight
    seed_quality_base = merged["Delta_Quality"].fillna(0.0)
    if config.gender == "W" and config.feature_profile == "seed_quality_interaction_women_conservative":
        seed_quality_base = seed_quality_base.clip(-0.35, 0.35) * 0.70
    merged["Seed_x_Quality"] = merged["Delta_Seed"].fillna(0.0) * seed_quality_base
    colley_diff = (
        pd.to_numeric(merged["Delta_Colley"], errors="coerce").fillna(0.0)
        if "Delta_Colley" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["Seed_x_Colley"] = merged["Delta_Seed"].fillna(0.0) * colley_diff
    colley_nc_diff = (
        pd.to_numeric(merged["Delta_ColleyNC"], errors="coerce").fillna(0.0)
        if "Delta_ColleyNC" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["Seed_x_ColleyNC"] = merged["Delta_Seed"].fillna(0.0) * colley_nc_diff
    women_consensus_diff = (
        pd.to_numeric(merged["WomenCompositeQuality_diff"], errors="coerce").fillna(0.0)
        if "WomenCompositeQuality_diff" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["Seed_x_WomenConsensusQuality"] = merged["Delta_Seed"].fillna(0.0) * women_consensus_diff
    women_opp_tournament_strength_diff = (
        pd.to_numeric(merged["Delta_WomenOpponentTournamentStrength"], errors="coerce").fillna(0.0)
        if "Delta_WomenOpponentTournamentStrength" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["Seed_x_WomenOpponentTournamentStrength"] = (
        pd.to_numeric(merged["Delta_Seed"], errors="coerce").fillna(0.0) * women_opp_tournament_strength_diff
    )
    women_opp_tournament_strength_v2_diff = (
        pd.to_numeric(merged["Delta_WomenOpponentTournamentStrengthV2"], errors="coerce").fillna(0.0)
        if "Delta_WomenOpponentTournamentStrengthV2" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    women_quality_wins_strength_v2_diff = (
        pd.to_numeric(merged["Delta_WomenQualityWinsStrengthV2"], errors="coerce").fillna(0.0)
        if "Delta_WomenQualityWinsStrengthV2" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["Seed_x_WomenOpponentTournamentStrengthV2"] = (
        pd.to_numeric(merged["Delta_Seed"], errors="coerce").fillna(0.0) * women_opp_tournament_strength_v2_diff
    )
    merged["Seed_x_WomenQualityWinsStrengthV2"] = (
        pd.to_numeric(merged["Delta_Seed"], errors="coerce").fillna(0.0) * women_quality_wins_strength_v2_diff
    )
    return merged


def build_ji_dataset(config: JIBaseConfig) -> pd.DataFrame:
    return _cached_ji_dataset(
        config.gender,
        config.alpha_profile,
        config.women_quality_profile,
        config.women_ranking_provider,
        config.feature_profile,
    )


@lru_cache(maxsize=16)
def _cached_ji_dataset(
    gender: str,
    alpha_profile: str,
    women_quality_profile: str,
    women_ranking_provider: str,
    feature_profile: str,
) -> pd.DataFrame:
    config = JIBaseConfig(
        gender=gender,
        alpha_profile=alpha_profile,
        women_quality_profile=women_quality_profile,
        women_ranking_provider=women_ranking_provider,
        feature_profile=feature_profile,  # type: ignore[arg-type]
    )
    results = load_tournament_results(gender)
    features = load_ji_team_features(config)
    merged = build_submission_feature_frame(results[["Season", "T1", "T2"]].copy(), features, config)
    merged["Label"] = results["Label"].to_numpy()
    merged["Margin"] = results["Margin"].to_numpy()
    merged["DayNum"] = results["DayNum"].to_numpy()
    ordered = ["Season", "DayNum", "T1", "T2", "Label", "Margin", *[col for col in merged.columns if col not in {"Season", "DayNum", "T1", "T2", "Label", "Margin"}]]
    return merged[ordered].sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)
