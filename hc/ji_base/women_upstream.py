from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "external-data"

EXTERNAL_CONSENSUS_SOURCE_COLUMNS = (
    "Ext_PublicNETRank",
    "Ext_PublicRPIRank",
    "Ext_PublicPredRPIRank",
    "Ext_WN_NET",
    "Ext_WN_ELO",
    "Ext_WN_RPI",
    "Ext_WN_PredRPI",
)
EXTERNAL_CONSENSUS_V2_WEIGHTS = {
    "Ext_PublicNETRank": 1.0,
    "Ext_PublicPredRPIRank": 1.0,
    "Ext_PublicRPIRank": 0.6,
    "Ext_WN_NET": 0.6,
    "Ext_WN_ELO": 0.6,
    "Ext_WN_RPI": 0.6,
    "Ext_WN_PredRPI": 0.6,
}
HISTORICAL_RANK_WEIGHTS = {
    "PublicNETRank": 1.0,
    "PublicELORank": 0.8,
    "PublicPredRPIRank": 0.8,
    "PublicRPIRank": 0.6,
    "PublicAPRank": 0.5,
    "PublicCoachesRank": 0.5,
    "CurrentWN_NET": 1.0,
    "CurrentWN_RPI": 0.6,
    "CurrentWN_PredRPI": 0.8,
    "CurrentOfficialNETRank": 1.0,
}
HISTORICAL_VALUE_WEIGHTS = {
    "FallbackElo": 0.8,
    "FallbackWinRate": 0.6,
    "FallbackAvgMargin": 0.5,
    "SB_BXelo": 0.8,
    "SB_BNetRating": 0.8,
    "CurrentWN_ELO": 1.0,
    "CurrentMarketTitleProb": 0.5,
}


@lru_cache(maxsize=1)
def _load_historical_consensus_snapshots() -> pd.DataFrame:
    path = EXTERNAL_DIR / "WHistoricalConsensusSnapshots.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID"])
    frame = pd.read_csv(path)
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["Season", "TeamID"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def _season_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    mean_ = series.groupby(frame["Season"]).transform("mean")
    std_ = series.groupby(frame["Season"]).transform("std").fillna(0.0).clip(lower=0.1)
    return ((series - mean_) / std_).fillna(0.0)


def _build_internal_fallback_score(profile: pd.DataFrame) -> pd.Series:
    pagerank_z = _season_zscore(
        profile.assign(_pagerank=pd.to_numeric(profile.get("PageRank", 0.0), errors="coerce").fillna(0.0)),
        "_pagerank",
    )
    rpi_z = _season_zscore(
        profile.assign(_rpi=pd.to_numeric(profile.get("RPIStyle", 0.0), errors="coerce").fillna(0.0)),
        "_rpi",
    )
    sos_z = _season_zscore(
        profile.assign(_sos=pd.to_numeric(profile.get("SOS", 0.0), errors="coerce").fillna(0.0)),
        "_sos",
    )
    win_z = _season_zscore(
        profile.assign(_wpct=pd.to_numeric(profile.get("wpct", 0.0), errors="coerce").fillna(0.0)),
        "_wpct",
    )
    return (0.45 * pagerank_z + 0.25 * rpi_z + 0.15 * sos_z + 0.15 * win_z).astype(float)


def _build_historical_snapshot_bundle(profile: pd.DataFrame, internal_score: pd.Series) -> pd.DataFrame:
    snapshots = _load_historical_consensus_snapshots()
    bundle = profile[["Season", "TeamID"]].copy()
    bundle["WomenConsensusRankScore"] = internal_score.astype(float)
    bundle["WomenConsensusRankCoverage"] = 0.0
    bundle["WomenConsensusRankConfidence"] = 0.0
    if snapshots.empty:
        return bundle

    merged = bundle.merge(snapshots, on=["Season", "TeamID"], how="left")

    weighted_available = pd.Series(0.0, index=merged.index, dtype=float)
    weighted_score_sum = pd.Series(0.0, index=merged.index, dtype=float)
    weighted_total = float(sum(HISTORICAL_RANK_WEIGHTS.values()) + sum(HISTORICAL_VALUE_WEIGHTS.values()))

    for column, weight in HISTORICAL_RANK_WEIGHTS.items():
        if column not in merged.columns or weight <= 0.0:
            continue
        raw = pd.to_numeric(merged[column], errors="coerce")
        standardized = -_season_zscore(merged.assign(_rank=raw), "_rank").where(raw.notna(), np.nan)
        available = raw.notna().astype(float)
        weighted_available = weighted_available + available * weight
        weighted_score_sum = weighted_score_sum + standardized.fillna(0.0) * weight

    for column, weight in HISTORICAL_VALUE_WEIGHTS.items():
        if column not in merged.columns or weight <= 0.0:
            continue
        raw = pd.to_numeric(merged[column], errors="coerce")
        standardized = _season_zscore(merged.assign(_value=raw), "_value").where(raw.notna(), np.nan)
        available = raw.notna().astype(float)
        weighted_available = weighted_available + available * weight
        weighted_score_sum = weighted_score_sum + standardized.fillna(0.0) * weight

    if weighted_total <= 0.0:
        return bundle

    external_score = weighted_score_sum / weighted_available.replace(0.0, np.nan)
    coverage = (weighted_available / weighted_total).clip(lower=0.0, upper=1.0)
    verified = pd.to_numeric(merged.get("HasVerifiedPreTourneySnapshot", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    source_count = pd.to_numeric(merged.get("WomenConsensusSourceCount", 0.0), errors="coerce").fillna(0.0)
    source_bonus = (source_count / 6.0).clip(lower=0.0, upper=1.0)
    confidence = np.maximum(((coverage - 0.20) / 0.55).clip(lower=0.0, upper=1.0), 0.15 * source_bonus + 0.20 * verified)
    confidence = confidence.clip(lower=0.0, upper=1.0).where(weighted_available > 0.0, 0.0)
    blended = ((1.0 - confidence) * internal_score + confidence * external_score.fillna(internal_score)).astype(float)

    bundle["WomenConsensusRankScore"] = blended
    bundle["WomenConsensusRankCoverage"] = coverage.astype(float)
    bundle["WomenConsensusRankConfidence"] = confidence.astype(float)
    return bundle


def build_women_ranking_upstream(profile: pd.DataFrame, *, provider: str = "internal_fallback") -> pd.DataFrame:
    bundle = profile[["Season", "TeamID"]].copy()
    internal_score = _build_internal_fallback_score(profile)
    bundle["WomenConsensusRankScore"] = internal_score.astype(float)
    bundle["WomenConsensusRankCoverage"] = 0.0
    bundle["WomenConsensusRankConfidence"] = 0.0

    if provider == "historical_consensus_snapshots_v1":
        return _build_historical_snapshot_bundle(profile, internal_score)

    if provider not in {"external_consensus_v1", "external_consensus_v2"}:
        return bundle

    available_cols = [column for column in EXTERNAL_CONSENSUS_SOURCE_COLUMNS if column in profile.columns]
    if not available_cols:
        return bundle

    if provider == "external_consensus_v2":
        weighted_components: list[pd.Series] = []
        weighted_available = pd.Series(0.0, index=profile.index, dtype=float)
        weighted_total = sum(EXTERNAL_CONSENSUS_V2_WEIGHTS.get(column, 0.0) for column in available_cols)

        for column in available_cols:
            weight = EXTERNAL_CONSENSUS_V2_WEIGHTS.get(column, 0.0)
            if weight <= 0.0:
                continue
            raw = pd.to_numeric(profile[column], errors="coerce")
            availability = raw.notna().astype(float)
            weighted_available = weighted_available + (availability * weight)
            standardized = -_season_zscore(profile.assign(_rank=raw), "_rank").where(raw.notna(), np.nan)
            weighted_components.append(standardized * weight)

        if not weighted_components or weighted_total <= 0.0:
            return bundle

        external_score = pd.concat(weighted_components, axis=1).sum(axis=1, skipna=True) / weighted_available.replace(0.0, np.nan)
        coverage = (weighted_available / float(weighted_total)).clip(lower=0.0, upper=1.0)
        confidence = ((coverage - 0.25) / 0.50).clip(lower=0.0, upper=1.0).where(weighted_available > 0.0, 0.0)
        blended = ((1.0 - confidence) * internal_score + confidence * external_score.fillna(internal_score)).astype(float)

        bundle["WomenConsensusRankScore"] = blended
        bundle["WomenConsensusRankCoverage"] = coverage.astype(float)
        bundle["WomenConsensusRankConfidence"] = confidence.astype(float)
        return bundle

    standardized_components: list[pd.Series] = []
    availability_count = pd.Series(0, index=profile.index, dtype=float)
    for column in available_cols:
        raw = pd.to_numeric(profile[column], errors="coerce")
        availability_count = availability_count + raw.notna().astype(float)
        standardized = -_season_zscore(profile.assign(_rank=raw), "_rank").where(raw.notna(), np.nan)
        standardized_components.append(standardized)

    if not standardized_components:
        return bundle

    external_score = pd.concat(standardized_components, axis=1).mean(axis=1, skipna=True)
    coverage = (availability_count / float(len(available_cols))).clip(lower=0.0, upper=1.0)
    confidence = coverage.where(availability_count > 0, 0.0)
    blended = ((1.0 - confidence) * internal_score + confidence * external_score.fillna(internal_score)).astype(float)

    bundle["WomenConsensusRankScore"] = blended
    bundle["WomenConsensusRankCoverage"] = coverage.astype(float)
    bundle["WomenConsensusRankConfidence"] = confidence.astype(float)
    return bundle
