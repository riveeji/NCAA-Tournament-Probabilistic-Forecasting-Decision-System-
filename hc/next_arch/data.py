from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from hc.ji_base import JIBaseConfig, build_ji_dataset, load_ji_team_features
from hc.ji_base.data import NCAA_DATA, load_tournament_results

from .config import NextArchConfig
from .season_encoder import build_season_encoder_matchup_features

GRAPH_EMBED_DIM = 8


def _load_regular_season_compact(gender: str) -> pd.DataFrame:
    frame = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonCompactResults.csv")
    frame["Season"] = frame["Season"].astype(int)
    return frame


def _normalize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


@lru_cache(maxsize=4)
def load_static_graph_team_embeddings(gender: str) -> pd.DataFrame:
    regular = _load_regular_season_compact(gender)
    rows: list[dict[str, float | int]] = []
    for season, season_games in regular.groupby("Season", sort=True):
        team_ids = np.array(sorted(set(season_games["WTeamID"]).union(set(season_games["LTeamID"]))), dtype=int)
        index = {team_id: idx for idx, team_id in enumerate(team_ids)}
        n_teams = len(team_ids)
        affinity = np.zeros((n_teams, n_teams), dtype=float)
        signed_flow = np.zeros((n_teams, n_teams), dtype=float)
        games_played = np.zeros(n_teams, dtype=float)

        for game in season_games.itertuples(index=False):
            winner = int(game.WTeamID)
            loser = int(game.LTeamID)
            margin = abs(float(game.WScore) - float(game.LScore))
            weight = 1.0 + min(margin, 15.0) / 15.0
            i = index[winner]
            j = index[loser]
            affinity[i, j] += weight
            affinity[j, i] += weight
            signed_flow[i, j] += weight
            signed_flow[j, i] -= weight
            games_played[i] += 1.0
            games_played[j] += 1.0

        signed_strength = signed_flow.sum(axis=1) / np.clip(games_played, 1.0, None)
        strength = _normalize_series(signed_strength)

        degree = affinity.sum(axis=1)
        if n_teams > 1 and np.any(degree > 0):
            inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
            normalized_adj = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
            eigvals, eigvecs = np.linalg.eigh(normalized_adj)
            order = np.argsort(eigvals)[::-1]
            selected = [idx for idx in order if eigvals[idx] > 1e-9][: GRAPH_EMBED_DIM + 1]
            if selected:
                selected = selected[1 : GRAPH_EMBED_DIM + 1] if len(selected) > 1 else []
            embedding = np.zeros((n_teams, GRAPH_EMBED_DIM), dtype=float)
            if selected:
                kept = eigvecs[:, selected] * np.sqrt(np.abs(eigvals[selected]))[None, :]
                embedding[:, : kept.shape[1]] = kept
        else:
            embedding = np.zeros((n_teams, GRAPH_EMBED_DIM), dtype=float)

        for idx, team_id in enumerate(team_ids):
            row: dict[str, float | int] = {
                "Season": int(season),
                "TeamID": int(team_id),
                "GraphEmbStrength": float(strength[idx]),
            }
            for emb_idx in range(GRAPH_EMBED_DIM):
                row[f"GraphEmb_{emb_idx}"] = float(embedding[idx, emb_idx])
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def _build_graph_matchup_frame(gender: str) -> pd.DataFrame:
    results = load_tournament_results(gender)
    team_embeddings = load_static_graph_team_embeddings(gender)
    embedding_cols = [column for column in team_embeddings.columns if column.startswith("GraphEmb_")]
    t1 = team_embeddings.rename(
        columns={"TeamID": "T1", "GraphEmbStrength": "T1_GraphEmbStrength", **{col: f"T1_{col}" for col in embedding_cols}}
    )
    t2 = team_embeddings.rename(
        columns={"TeamID": "T2", "GraphEmbStrength": "T2_GraphEmbStrength", **{col: f"T2_{col}" for col in embedding_cols}}
    )
    merged = results.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")

    left = merged[[f"T1_{col}" for col in embedding_cols]].fillna(0.0).to_numpy(dtype=float)
    right = merged[[f"T2_{col}" for col in embedding_cols]].fillna(0.0).to_numpy(dtype=float)
    numerator = np.sum(left * right, axis=1)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.where((left_norm > 0) & (right_norm > 0), left_norm * right_norm, 1.0)
    merged["GraphEmbCosSim"] = numerator / denom
    merged["GraphEmbL2"] = np.linalg.norm(left - right, axis=1)
    merged["Delta_GraphEmbStrength"] = (
        pd.to_numeric(merged.get("T1_GraphEmbStrength"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("T2_GraphEmbStrength"), errors="coerce").fillna(0.0)
    )
    ordered = [
        "Season",
        "DayNum",
        "T1",
        "T2",
        "Label",
        "Margin",
        "GraphEmbCosSim",
        "GraphEmbL2",
        "Delta_GraphEmbStrength",
    ]
    return merged[ordered].sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)


def _build_gender_specific_stacker_frame(gender: str) -> pd.DataFrame:
    base_dataset = build_ji_dataset(JIBaseConfig(gender=gender)).copy()
    if gender != "W":
        return base_dataset

    women_config = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        calibration_mode="none",
        alpha_profile="quality_only_men_quality_blocks_women",
        feature_profile="lr_carry_elo_definition_v1",
        women_quality_profile="consensus_rebuild_v6",
        women_ranking_provider="historical_consensus_snapshots_v1",
    )
    results = load_tournament_results("W")[["Season", "DayNum", "T1", "T2"]].copy()
    team_features = load_ji_team_features(women_config)[
        [
            "Season",
            "TeamID",
            "WomenConsensusRankScore",
            "WomenConsensusRankCoverage",
            "WomenConsensusRankConfidence",
        ]
    ].copy()
    t1 = team_features.rename(
        columns={
            "TeamID": "T1",
            "WomenConsensusRankScore": "T1_WomenConsensusRankScore",
            "WomenConsensusRankCoverage": "T1_WomenConsensusRankCoverage",
            "WomenConsensusRankConfidence": "T1_WomenConsensusRankConfidence",
        }
    )
    t2 = team_features.rename(
        columns={
            "TeamID": "T2",
            "WomenConsensusRankScore": "T2_WomenConsensusRankScore",
            "WomenConsensusRankCoverage": "T2_WomenConsensusRankCoverage",
            "WomenConsensusRankConfidence": "T2_WomenConsensusRankConfidence",
        }
    )
    sidecar = results.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")
    sidecar["Delta_WomenConsensusRankScore"] = (
        pd.to_numeric(sidecar["T1_WomenConsensusRankScore"], errors="coerce").fillna(0.0)
        - pd.to_numeric(sidecar["T2_WomenConsensusRankScore"], errors="coerce").fillna(0.0)
    )
    sidecar["WomenConsensusCoverageMean"] = (
        pd.to_numeric(sidecar["T1_WomenConsensusRankCoverage"], errors="coerce").fillna(0.0)
        + pd.to_numeric(sidecar["T2_WomenConsensusRankCoverage"], errors="coerce").fillna(0.0)
    ) / 2.0
    sidecar["WomenConsensusConfidenceMean"] = (
        pd.to_numeric(sidecar["T1_WomenConsensusRankConfidence"], errors="coerce").fillna(0.0)
        + pd.to_numeric(sidecar["T2_WomenConsensusRankConfidence"], errors="coerce").fillna(0.0)
    ) / 2.0
    sidecar = sidecar[
        [
            "Season",
            "DayNum",
            "T1",
            "T2",
            "Delta_WomenConsensusRankScore",
            "WomenConsensusCoverageMean",
            "WomenConsensusConfidenceMean",
        ]
    ]
    merged = base_dataset.merge(sidecar, on=["Season", "DayNum", "T1", "T2"], how="left")
    for column in ("Delta_WomenConsensusRankScore", "WomenConsensusCoverageMean", "WomenConsensusConfidenceMean"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def build_next_arch_dataset(config: NextArchConfig) -> pd.DataFrame:
    if config.experiment_name in {"tabr_v1", "tabr_hybrid_v1", "tabr_feature_fusion_v1", "pairwise_ranking_v1"}:
        return build_ji_dataset(config.base_config()).copy()
    if config.experiment_name == "season_encoder_transformer_v1":
        base_dataset = build_ji_dataset(config.base_config()).copy()
        return build_season_encoder_matchup_features(base_dataset, config.gender)
    if config.experiment_name == "graph_static_embedding_v1":
        return _build_graph_matchup_frame(config.gender)
    if config.experiment_name == "gender_specific_stacker_v1":
        return _build_gender_specific_stacker_frame(config.gender)
    raise KeyError(f"Unknown next-arch experiment: {config.experiment_name}")
