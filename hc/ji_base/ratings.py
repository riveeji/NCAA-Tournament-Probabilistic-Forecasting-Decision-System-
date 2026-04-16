from __future__ import annotations

import numpy as np
import pandas as pd

from .women_upstream import build_women_ranking_upstream


def _season_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    mean_ = series.groupby(frame["Season"]).transform("mean")
    std_ = series.groupby(frame["Season"]).transform("std").fillna(0.0).clip(lower=0.1)
    return ((series - mean_) / std_).fillna(0.0)


def _quality_wins_component(frame: pd.DataFrame, *, gender: str) -> pd.Series:
    top50 = pd.to_numeric(frame.get("Top50WinRate", 0.0), errors="coerce").fillna(0.0)
    top100 = pd.to_numeric(frame.get("Top100WinRate", 0.0), errors="coerce").fillna(0.0)
    sos_z = _season_zscore(frame.assign(_SOS=pd.to_numeric(frame.get("SOS", 0.0), errors="coerce").fillna(0.0)), "_SOS")
    base = 4.0 * top50 + 2.0 * top100 + 0.15 * sos_z
    if gender == "W":
        base = np.sign(base) * np.sqrt(np.abs(base))
    return pd.Series(base, index=frame.index, dtype=float)


def build_ji_ratings(
    profile: pd.DataFrame,
    *,
    gender: str,
    women_quality_profile: str = "legacy_v1",
    women_ranking_provider: str = "internal_fallback",
) -> pd.DataFrame:
    frame = profile.copy()

    elo_z = _season_zscore(frame.assign(_Elo=frame["Elo"]), "_Elo")
    win_z = _season_zscore(frame.assign(_wpct=frame["wpct"]), "_wpct")
    sos_z = _season_zscore(frame.assign(_sos=pd.to_numeric(frame.get("SOS", 0.0), errors="coerce").fillna(0.0)), "_sos")
    neff = pd.to_numeric(frame["neff"], errors="coerce").fillna(0.0)
    neff_z = _season_zscore(frame.assign(_neff=neff), "_neff")
    margin = pd.to_numeric(frame["margin"], errors="coerce").fillna(0.0)
    margin_term = np.sign(margin) * np.sqrt(np.abs(margin)) if gender == "W" else margin
    margin_z = _season_zscore(frame.assign(_margin=margin_term), "_margin")
    rpi_style = pd.to_numeric(frame.get("RPIStyle", 0.0), errors="coerce").fillna(0.0)
    rpi_z = _season_zscore(frame.assign(_rpi=rpi_style), "_rpi")
    pagerank = pd.to_numeric(frame.get("PageRank", 0.0), errors="coerce").fillna(0.0)
    pagerank_z = _season_zscore(frame.assign(_pagerank=pagerank), "_pagerank")

    quality_wins = _quality_wins_component(frame, gender=gender)
    opp_quality_rank = 0.55 * quality_wins + 0.30 * sos_z + 0.15 * win_z
    women_seed_strength = pd.Series(0.0, index=frame.index, dtype=float)
    women_quality_wins_strength = pd.Series(0.0, index=frame.index, dtype=float)
    women_quality_wins_strength_v2 = pd.Series(0.0, index=frame.index, dtype=float)
    women_opp_tournament_strength = pd.Series(0.0, index=frame.index, dtype=float)
    women_opp_tournament_strength_v2 = pd.Series(0.0, index=frame.index, dtype=float)
    women_rim_protection_strength = pd.Series(0.0, index=frame.index, dtype=float)
    women_composite_v5 = pd.Series(np.nan, index=frame.index, dtype=float)
    women_consensus_rank_score = pd.Series(np.nan, index=frame.index, dtype=float)
    women_consensus_rank_coverage = pd.Series(0.0, index=frame.index, dtype=float)
    women_consensus_rank_confidence = pd.Series(0.0, index=frame.index, dtype=float)

    quality = 0.35 * elo_z + 0.30 * neff_z + 0.20 * win_z + 0.15 * margin_z
    if gender == "W":
        women_upstream = build_women_ranking_upstream(frame, provider=women_ranking_provider)
        frame = frame.merge(women_upstream, on=["Season", "TeamID"], how="left")
        women_consensus_rank_score = pd.to_numeric(frame["WomenConsensusRankScore"], errors="coerce").fillna(0.0)
        women_consensus_rank_coverage = pd.to_numeric(frame["WomenConsensusRankCoverage"], errors="coerce").fillna(0.0)
        women_consensus_rank_confidence = pd.to_numeric(frame["WomenConsensusRankConfidence"], errors="coerce").fillna(0.0)
        legacy_quality = 0.40 * elo_z + 0.25 * neff_z + 0.20 * win_z + 0.15 * margin_z
        if women_quality_profile == "consensus_rebuild_v5":
            seed_raw = pd.to_numeric(frame.get("SeedNum", frame.get("Seed", 17)), errors="coerce").fillna(17.0)
            women_seed_strength = _season_zscore(frame.assign(_seed=-seed_raw), "_seed").clip(lower=-2.0, upper=2.0)
            women_quality_wins_strength = quality_wins.clip(lower=-2.0, upper=2.0)
            women_opp_tournament_strength = (0.55 * women_quality_wins_strength + 0.30 * sos_z + 0.15 * win_z).clip(lower=-2.0, upper=2.0)
            women_rim_protection_strength = _season_zscore(
                frame.assign(_blk=pd.to_numeric(frame.get("AvgBlkDiff", 0.0), errors="coerce").fillna(0.0)),
                "_blk",
            ).clip(lower=-2.0, upper=2.0)
            women_composite_v5 = (
                0.35 * women_seed_strength
                + 0.25 * women_quality_wins_strength
                + 0.25 * women_opp_tournament_strength
                + 0.15 * women_rim_protection_strength
            ).clip(lower=-2.0, upper=2.0)
            quality = 0.75 * legacy_quality + 0.25 * women_composite_v5
            opp_quality_rank = women_opp_tournament_strength
            frame["WomenCompositeQuality"] = women_composite_v5
        elif women_quality_profile == "consensus_rebuild_v4b":
            women_consensus = 0.45 * pagerank_z + 0.25 * rpi_z + 0.15 * sos_z + 0.15 * win_z
            clipped_quality_wins = quality_wins.clip(lower=-2.0, upper=2.0)
            quality = 0.88 * legacy_quality + 0.12 * women_consensus
            opp_quality_rank = 0.55 * clipped_quality_wins + 0.30 * sos_z + 0.15 * women_consensus
            frame["WomenCompositeQuality"] = women_consensus
        elif women_quality_profile == "consensus_rebuild_v4a":
            women_consensus = 0.45 * pagerank_z + 0.25 * rpi_z + 0.15 * sos_z + 0.15 * win_z
            clipped_quality_wins = quality_wins.clip(lower=-2.0, upper=2.0)
            quality = 0.90 * legacy_quality + 0.10 * women_consensus
            opp_quality_rank = 0.58 * clipped_quality_wins + 0.30 * sos_z + 0.12 * women_consensus
            frame["WomenCompositeQuality"] = women_consensus
        elif women_quality_profile == "consensus_rebuild_v6":
            women_consensus = women_consensus_rank_score
            clipped_quality_wins = quality_wins.clip(lower=-2.0, upper=2.0)
            quality = 0.88 * legacy_quality + 0.12 * women_consensus
            opp_quality_rank = 0.55 * clipped_quality_wins + 0.30 * sos_z + 0.15 * women_consensus
            frame["WomenCompositeQuality"] = women_consensus
        elif women_quality_profile == "consensus_rebuild_v4":
            women_consensus = 0.45 * pagerank_z + 0.25 * rpi_z + 0.15 * sos_z + 0.15 * win_z
            clipped_quality_wins = quality_wins.clip(lower=-2.0, upper=2.0)
            quality = 0.88 * legacy_quality + 0.12 * women_consensus
            opp_quality_rank = 0.55 * clipped_quality_wins + 0.30 * sos_z + 0.15 * women_consensus
            frame["WomenCompositeQuality"] = women_consensus
        elif women_quality_profile == "consensus_rebuild_v3":
            women_consensus = 0.45 * pagerank_z + 0.25 * rpi_z + 0.15 * sos_z + 0.15 * win_z
            clipped_quality_wins = quality_wins.clip(lower=-2.0, upper=2.0)
            quality = 0.80 * legacy_quality + 0.20 * women_consensus
            opp_quality_rank = 0.50 * clipped_quality_wins + 0.30 * sos_z + 0.20 * women_consensus
            frame["WomenCompositeQuality"] = women_consensus
        elif women_quality_profile == "consensus_rebuild_v2":
            dominance_margin = np.sign(margin) * np.log1p(np.abs(margin))
            dominance_z = _season_zscore(frame.assign(_dominance=dominance_margin), "_dominance")
            clipped_quality_wins = quality_wins.clip(lower=-2.5, upper=2.5)
            women_consensus = 0.28 * elo_z + 0.22 * neff_z + 0.18 * rpi_z + 0.16 * pagerank_z + 0.10 * win_z + 0.06 * sos_z
            women_quality = 0.60 * women_consensus + 0.25 * clipped_quality_wins + 0.15 * dominance_z
            opp_quality_rank = 0.40 * clipped_quality_wins + 0.20 * sos_z + 0.15 * win_z + 0.15 * rpi_z + 0.10 * pagerank_z
            quality = women_quality
            frame["WomenCompositeQuality"] = women_consensus
        else:
            women_quality = legacy_quality
            quality = women_quality
            frame["WomenCompositeQuality"] = women_quality

        women_quality_wins_strength_v2 = (
            0.70 * quality_wins.clip(lower=-2.0, upper=2.0)
            + 0.20 * win_z
            + 0.10 * pagerank_z
        ).clip(lower=-2.0, upper=2.0)
        women_opp_tournament_strength_v2 = (
            0.50 * quality_wins.clip(lower=-2.0, upper=2.0)
            + 0.30 * sos_z
            + 0.10 * win_z
            + 0.10 * pagerank_z
        ).clip(lower=-2.0, upper=2.0)
    else:
        frame["WomenCompositeQuality"] = np.nan

    conf_mean_elo = pd.to_numeric(frame.get("ConfMeanElo", 0.0), errors="coerce").fillna(0.0)
    conf_factor = 1.0 + 0.08 * _season_zscore(frame.assign(_conf=conf_mean_elo), "_conf")
    opp_factor = 1.0 + 0.10 * np.clip(opp_quality_rank, -2.0, 2.0)
    if gender == "W" and women_quality_profile == "consensus_rebuild_v5":
        women_consensus = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce").fillna(0.0)
        harry = neff * (1.0 + 0.03 * np.clip(women_consensus, -2.0, 2.0))
    elif gender == "W" and women_quality_profile == "consensus_rebuild_v4b":
        women_consensus = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce").fillna(0.0)
        harry = neff * (1.0 + 0.03 * np.clip(women_consensus, -2.0, 2.0))
    elif gender == "W" and women_quality_profile == "consensus_rebuild_v4a":
        women_consensus = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce").fillna(0.0)
        harry = neff * (1.0 + 0.04 * np.clip(women_consensus, -2.0, 2.0))
    elif gender == "W" and women_quality_profile == "consensus_rebuild_v4":
        women_consensus = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce").fillna(0.0)
        harry = neff * (1.0 + 0.04 * np.clip(women_consensus, -2.0, 2.0))
    elif gender == "W" and women_quality_profile == "consensus_rebuild_v3":
        women_consensus = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce").fillna(0.0)
        harry = neff * (1.0 + 0.06 * np.clip(women_consensus, -2.0, 2.0))
    else:
        harry = neff * opp_factor * conf_factor

    output = frame[["Season", "TeamID"]].copy()
    output["QualityWins"] = quality_wins.astype(float)
    output["OpponentQualityTournamentRank"] = opp_quality_rank.astype(float)
    output["Quality"] = quality.astype(float)
    output["harry_Rating"] = harry.astype(float)
    output["WomenCompositeQuality"] = pd.to_numeric(frame["WomenCompositeQuality"], errors="coerce")
    output["WomenSeedStrength"] = women_seed_strength.astype(float)
    output["WomenQualityWinsStrength"] = women_quality_wins_strength.astype(float)
    output["WomenQualityWinsStrengthV2"] = women_quality_wins_strength_v2.astype(float)
    output["WomenOpponentTournamentStrength"] = women_opp_tournament_strength.astype(float)
    output["WomenOpponentTournamentStrengthV2"] = women_opp_tournament_strength_v2.astype(float)
    output["WomenRimProtectionStrength"] = women_rim_protection_strength.astype(float)
    output["WomenCompositeQualityV5"] = women_composite_v5.astype(float)
    output["WomenConsensusRankScore"] = women_consensus_rank_score.astype(float)
    output["WomenConsensusRankCoverage"] = women_consensus_rank_coverage.astype(float)
    output["WomenConsensusRankConfidence"] = women_consensus_rank_confidence.astype(float)
    return output
