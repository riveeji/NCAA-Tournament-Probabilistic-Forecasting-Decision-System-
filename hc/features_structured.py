from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hc.constants import (
    CACHE_DIR,
    DATA_DIR,
    EXTERNAL_DIR,
    MARKET_POLICY_BY_GENDER,
    MARKET_CONSENSUS_FEATURES,
    PROFILE_AGGRESSIVE,
    MEN_MARKET_FEATURES,
    MEN_PUBLIC_ROUTE_FEATURES,
    MEN_STRUCTURED_FEATURES,
    PUBLIC_ROUTE_VERSION,
    TEXT_ROUTE_STRUCTURED_ANCHORS,
    ENABLE_SILVER_HISTORY,
    WOMEN_PUBLIC_ROUTE_FEATURES,
    WOMEN_STRUCTURED_FEATURES,
)
from hc.data_build import build_all, cache_path, resolve_all_tourney_seasons
from hc.features_text import attach_text_matchup_features, load_text_embeddings
from hc.signals import canonicalize_team_signal_frame, coalesce_team_signal_frames
from zizzii_features import build_team_features, filter_verified_pretourney_snapshot_frame
from zizzii_train import build_matchup_df


def matchup_cache_path(gender: str, market_policy: str, text_dim: int, profile: str) -> Path:
    return CACHE_DIR / f"matchups_{gender}_{market_policy}_{text_dim}d_{profile}_{PUBLIC_ROUTE_VERSION}.parquet"


def available_eval_seasons(matchups: pd.DataFrame) -> list[int]:
    return sorted(pd.to_numeric(matchups["Season"], errors="coerce").dropna().astype(int).unique().tolist())


def _market_cache_path(gender: str, market_policy: str) -> Path:
    return cache_path(f"market_history_{gender}_{market_policy}")


def _team_cache_path(gender: str) -> Path:
    return cache_path(f"team_snapshots_{gender}")


def _dedupe_team_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"Season", "TeamID"}.issubset(frame.columns):
        return frame
    deduped = frame.copy()
    deduped["Season"] = pd.to_numeric(deduped["Season"], errors="coerce")
    deduped["TeamID"] = pd.to_numeric(deduped["TeamID"], errors="coerce")
    deduped = deduped.dropna(subset=["Season", "TeamID"]).copy()
    deduped["Season"] = deduped["Season"].astype(int)
    deduped["TeamID"] = deduped["TeamID"].astype(int)
    if "SnapshotDate" in deduped.columns:
        deduped["SnapshotDate"] = pd.to_datetime(deduped["SnapshotDate"], errors="coerce", utc=True)
        deduped = deduped.sort_values(["Season", "TeamID", "SnapshotDate"])
    else:
        deduped = deduped.sort_values(["Season", "TeamID"])
    deduped = deduped.drop_duplicates(["Season", "TeamID"], keep="last").reset_index(drop=True)
    return deduped


def _canonical_public_rating_map(gender: str) -> dict[str, str]:
    if gender == "M":
        return {
            "PublicNETRank": "HC_PublicNETRank",
            "OfficialNETRank": "HC_PublicNETRank",
            "WN_NET": "HC_PublicNETRank",
            "PublicELORank": "HC_PublicELORank",
            "WN_ELO": "HC_PublicELORank",
            "PublicRPIRank": "HC_PublicRPIRank",
            "WN_RPI": "HC_PublicRPIRank",
            "PublicPredRPIRank": "HC_PublicPredRPIRank",
            "WN_PredRPI": "HC_PublicPredRPIRank",
            "PublicBPIRank": "HC_PublicBPIRank",
            "WN_BPI": "HC_PublicBPIRank",
            "PublicPOMRank": "HC_PublicPOMRank",
            "WN_POM": "HC_PublicPOMRank",
            "PublicKPIRank": "HC_PublicKPIRank",
            "WN_KPI": "HC_PublicKPIRank",
            "PublicSORRank": "HC_PublicSORRank",
            "WN_SOR": "HC_PublicSORRank",
            "PublicAverageRank": "HC_PublicAverageRank",
            "WN_AverageRank": "HC_PublicAverageRank",
            "PublicTRankRank": "HC_PublicTRankRank",
            "WN_TRank": "HC_PublicTRankRank",
            "PublicAvgPredRank": "HC_PublicAvgPredRank",
            "WN_AvgPredRank": "HC_PublicAvgPredRank",
        }
    return {
        "PublicNETRank": "HC_PublicNETRank",
        "OfficialNETRank": "HC_PublicNETRank",
        "WN_NET": "HC_PublicNETRank",
        "PublicELORank": "HC_PublicELORank",
        "WN_ELO": "HC_PublicELORank",
        "PublicRPIRank": "HC_PublicRPIRank",
        "WN_RPI": "HC_PublicRPIRank",
        "PublicPredRPIRank": "HC_PublicPredRPIRank",
        "WN_PredRPI": "HC_PublicPredRPIRank",
    }


def load_aggressive_public_team_ratings(gender: str) -> pd.DataFrame:
    mapping = _canonical_public_rating_map(gender)
    frames: list[pd.DataFrame] = []
    historical_pre = EXTERNAL_DIR / f"{gender}HistoricalTeamRatingsPreTourney.csv"
    current_ratings = EXTERNAL_DIR / f"{gender}TeamRatings.csv"
    source_priority = {
        current_ratings: 1,
        historical_pre: 2,
    }
    for path in [current_ratings, historical_pre]:
        if not path.exists():
            continue
        try:
            current = pd.read_csv(path)
        except Exception:
            continue
        if current.empty or "Season" not in current.columns or "TeamID" not in current.columns:
            continue
        if path == historical_pre:
            if not {"SnapshotDate", "VerifiedPreTourney"}.issubset(current.columns):
                continue
            current = filter_verified_pretourney_snapshot_frame(current, gender, data_dir=DATA_DIR)
            if current.empty:
                continue
        keep = ["Season", "TeamID"]
        rename_map: dict[str, str] = {}
        for source_col, target_col in mapping.items():
            if source_col in current.columns:
                keep.append(source_col)
                rename_map[source_col] = target_col
        if len(keep) == 2:
            continue
        current = current[keep].rename(columns=rename_map).copy()
        value_columns = [column for column in current.columns if column not in {"Season", "TeamID"}]
        if value_columns:
            current = current.T.groupby(level=0).first().T
            for column in value_columns:
                if column in current.columns:
                    current[column] = pd.to_numeric(current[column], errors="coerce")
        frames.append(
            canonicalize_team_signal_frame(
                current,
                source=path.name,
                priority=source_priority.get(path, 0),
            )
        )
    return coalesce_team_signal_frames(frames)


def load_silver_bulletin_team_signals(gender: str, include_history: bool = ENABLE_SILVER_HISTORY) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    patterns = [
        f"{gender}SilverBulletinTeamRatings_*.csv",
        f"{gender}SilverBulletinTournamentProbs_*.csv",
    ]
    for pattern in patterns:
        for path in sorted(EXTERNAL_DIR.glob(pattern)):
            if (not include_history) and "History" in path.name:
                continue
            try:
                current = pd.read_csv(path)
            except Exception:
                continue
            if current.empty or "Season" not in current.columns or "TeamID" not in current.columns:
                continue
            priority = 10
            if {"SnapshotDate", "VerifiedPreTourney"}.issubset(current.columns):
                current = filter_verified_pretourney_snapshot_frame(current, gender, data_dir=DATA_DIR)
                if current.empty:
                    continue
                priority = 30
            elif "TournamentProbs" in path.name:
                priority = 20
            elif "History" in path.name:
                priority = 5
            keep_columns = [column for column in current.columns if column in {"Season", "TeamID", "SnapshotDate"} or column.startswith("SB_")]
            if not keep_columns:
                continue
            current = current[keep_columns].copy()
            value_columns = [column for column in current.columns if column.startswith("SB_") and column != "SB_TeamRegion"]
            for column in value_columns:
                current[column] = pd.to_numeric(current[column], errors="coerce")
            frames.append(canonicalize_team_signal_frame(current, source=path.name, priority=priority))
    aggregated = coalesce_team_signal_frames(frames)
    numeric_cols = [
        column
        for column in aggregated.columns
        if column.startswith("SB_") and column not in {"SB_DisplayConference", "SB_TeamRegion"}
    ]
    for column in numeric_cols:
        aggregated[column] = pd.to_numeric(aggregated[column], errors="coerce")
    return aggregated


def augment_team_snapshots_with_public_ratings(
    team_feats: pd.DataFrame,
    gender: str,
    include_silver_history: bool = ENABLE_SILVER_HISTORY,
) -> pd.DataFrame:
    public_df = load_aggressive_public_team_ratings(gender)
    merged = team_feats
    if not public_df.empty:
        merged = merged.merge(public_df, on=["Season", "TeamID"], how="left")
    silver_df = load_silver_bulletin_team_signals(gender, include_history=include_silver_history)
    if not silver_df.empty:
        merged = merged.merge(silver_df, on=["Season", "TeamID"], how="left")
    return merged


def load_market_history(gender: str, market_policy: Optional[str] = None) -> pd.DataFrame:
    policy = market_policy or MARKET_POLICY_BY_GENDER[gender]
    path = _market_cache_path(gender, policy)
    if not path.exists():
        build_all(genders=(gender,))
    if not path.exists():
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"])
    try:
        return pd.read_parquet(path)
    except Exception:
        build_all(genders=(gender,), force_rebuild=True)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"])


def load_team_snapshots(gender: str, force_rebuild: bool = False) -> pd.DataFrame:
    path = _team_cache_path(gender)
    if force_rebuild or not path.exists():
        frame = _dedupe_team_snapshots(build_team_features(gender=gender, data_dir=DATA_DIR, external_dir=EXTERNAL_DIR))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame
    try:
        return _dedupe_team_snapshots(pd.read_parquet(path))
    except Exception:
        frame = _dedupe_team_snapshots(build_team_features(gender=gender, data_dir=DATA_DIR, external_dir=EXTERNAL_DIR))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame


def merge_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if market_df.empty:
        return df, []
    keep = [
        column
        for column in ["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"] + MARKET_CONSENSUS_FEATURES
        if column in market_df.columns
    ]
    merged = df.merge(market_df[keep], on=["Season", "T1", "T2"], how="left")
    merged["MarketAvailable"] = merged["MarketProb"].notna().astype(int)
    market_cols = [
        column
        for column in ["MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"] + MARKET_CONSENSUS_FEATURES + ["MarketAvailable"]
        if column in merged.columns
    ]
    return merged, market_cols


def build_hc_matchups(
    gender: str,
    market_policy: Optional[str] = None,
    text_dim: int = 32,
    include_text: bool = True,
    profile: str = PROFILE_AGGRESSIVE,
    include_aggressive_public: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    policy = market_policy or MARKET_POLICY_BY_GENDER[gender]
    cache = matchup_cache_path(gender, policy, text_dim, profile)
    if cache.exists() and not force_rebuild:
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass

    team_feats = load_team_snapshots(gender, force_rebuild=force_rebuild)
    if include_aggressive_public:
        team_feats = augment_team_snapshots_with_public_ratings(team_feats, gender, include_silver_history=False)
    tourney = pd.read_csv(DATA_DIR / f"{gender}NCAATourneyCompactResults.csv")
    matchups, _, _, _, _ = build_matchup_df(tourney, team_feats, gender)
    market_df = load_market_history(gender, policy)
    matchups, _ = merge_market_features(matchups, market_df)
    if include_text:
        text_df = load_text_embeddings(gender, text_dim)
        matchups, _ = attach_text_matchup_features(matchups, text_df)
    if "MarketProb" in matchups.columns:
        matchups["MarketProb"] = pd.to_numeric(matchups["MarketProb"], errors="coerce")
    if "LastSpread" in matchups.columns:
        matchups["LastSpread"] = pd.to_numeric(matchups["LastSpread"], errors="coerce")
        matchups["AbsLastSpread"] = matchups["LastSpread"].abs()
    matchups.to_parquet(cache, index=False)
    return matchups


def feature_views(
    matchups: pd.DataFrame,
    gender: str,
    text_enabled: bool,
    tabpfn_enabled: bool,
    include_public_route: bool = True,
) -> dict[str, list[str]]:
    views: dict[str, list[str]] = {}
    if gender == "M":
        views["market_only"] = [column for column in MEN_MARKET_FEATURES if column in matchups.columns]
        if include_public_route:
            views["market_public"] = [column for column in MEN_PUBLIC_ROUTE_FEATURES if column in matchups.columns]
        views["market_plus_structured"] = list(
            dict.fromkeys([column for column in MEN_MARKET_FEATURES + MEN_STRUCTURED_FEATURES if column in matchups.columns])
        )
        views["stats_fallback"] = [column for column in MEN_STRUCTURED_FEATURES if column in matchups.columns]
    else:
        views["women_minimal"] = [column for column in WOMEN_STRUCTURED_FEATURES if column in matchups.columns]
        views["women_market"] = list(
            dict.fromkeys([column for column in ["MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"] + WOMEN_STRUCTURED_FEATURES if column in matchups.columns])
        )
        if include_public_route:
            views["women_public"] = [column for column in WOMEN_PUBLIC_ROUTE_FEATURES if column in matchups.columns]

    text_cols = sorted(
        column for column in matchups.columns
        if column.startswith("D_Text") or column.startswith("Abs_Text") or column.startswith("Mean_Text") or column.startswith("TextDocCount")
    )
    if text_enabled and text_cols:
        structured_anchor = [column for column in TEXT_ROUTE_STRUCTURED_ANCHORS[gender] if column in matchups.columns]
        views["text_fusion"] = list(dict.fromkeys(structured_anchor + text_cols))

    if tabpfn_enabled:
        base = views.get("market_plus_structured" if gender == "M" else "women_minimal", [])
        text_subset = views.get("text_fusion", [])[:16]
        views["tabpfn"] = list(dict.fromkeys(base + text_subset))

    return {key: value for key, value in views.items() if value}


def summarize_matchup_matrix(matchups: pd.DataFrame) -> dict[str, object]:
    seasons = available_eval_seasons(matchups)
    market_cov = {}
    for season in seasons:
        season_df = matchups.loc[matchups["Season"] == season]
        market_cov[int(season)] = float(season_df.get("MarketProb", pd.Series(np.nan, index=season_df.index)).notna().mean())
    return {
        "rows": int(len(matchups)),
        "seasons": seasons,
        "market_coverage_by_season": market_cov,
    }
