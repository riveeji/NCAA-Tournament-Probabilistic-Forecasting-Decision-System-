from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hc.ji_base.config import build_working_ji_base_config

NEXT_ARCH_EXPERIMENTS = (
    "tabr_v1",
    "tabr_hybrid_v1",
    "tabr_feature_fusion_v1",
    "pairwise_ranking_v1",
    "season_encoder_transformer_v1",
    "graph_static_embedding_v1",
    "gender_specific_stacker_v1",
)


@dataclass(slots=True)
class NextArchConfig:
    gender: Literal["M", "W"]
    experiment_name: Literal[
        "tabr_v1",
        "tabr_hybrid_v1",
        "tabr_feature_fusion_v1",
        "pairwise_ranking_v1",
        "season_encoder_transformer_v1",
        "graph_static_embedding_v1",
        "gender_specific_stacker_v1",
    ]
    recent_window: int = 5

    def base_config(self):
        return build_working_ji_base_config(self.gender)

    def resolved_model_features(self) -> list[str]:
        if self.experiment_name == "graph_static_embedding_v1":
            return ["GraphEmbCosSim", "GraphEmbL2", "Delta_GraphEmbStrength"]
        if self.experiment_name == "season_encoder_transformer_v1":
            return ["SeasonEmbCosSim", "SeasonEmbL2", "Delta_SeasonEmbStrength"]
        if self.experiment_name == "gender_specific_stacker_v1":
            if self.gender == "M":
                return ["BaselineLogit"]
            return [
                "BaselineLogit",
                "Delta_WomenConsensusRankScore",
                "WomenConsensusCoverageMean",
                "WomenConsensusConfidenceMean",
            ]
        features = list(self.base_config().resolved_model_features())
        if self.experiment_name in {"tabr_hybrid_v1", "tabr_feature_fusion_v1"}:
            features.append("BaselineLogit")
        return features

    def resolved_clip_bounds(self) -> tuple[float, float]:
        return self.base_config().resolved_clip_bounds()
