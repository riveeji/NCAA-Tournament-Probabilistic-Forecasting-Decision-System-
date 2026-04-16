from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MODEL_FAMILIES = ("JI_spread_xgb", "JI_lr_control", "JI_lgb_control", "JI_node_control")
CALIBRATION_MODES = ("none", "isotonic_gender")
SIDECAR_PROFILES = ("none", "text_embeddings_v1", "graph_team_embedding_v1")
OVERLAY_SOURCE_PROFILES = ("direct_priority", "direct_only")
FROZEN_OVERLAY_SUBMISSION_PROFILE = "ji_base_overlay_v1_men_best_women_direct_only_weight025"
OVERLAY_SUBMISSION_PROFILES = (
    "ji_base_overlay_v1",
    "ji_base_overlay_v1_conservative_injury",
    "ji_base_overlay_v1_direct_only",
    "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
    "ji_base_overlay_v1_direct_only_injury_confirmed3",
    "ji_base_overlay_v1_direct_only_injury_confirmed4",
    "ji_base_overlay_v1_direct_only_injury_confirmed5",
    "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008",
    "ji_base_overlay_v1_men_best_women_direct_priority",
    "ji_base_overlay_v1_men_best_women_direct_only_weight070",
    "ji_base_overlay_v1_men_best_women_direct_only_weight060",
    "ji_base_overlay_v1_men_best_women_direct_only_weight050",
    "ji_base_overlay_v1_men_best_women_direct_only_weight040",
    "ji_base_overlay_v1_men_best_women_direct_only_weight030",
    "ji_base_overlay_v1_men_best_women_direct_only_weight020",
    "ji_base_overlay_v1_men_best_women_direct_only_weight025",
    "ji_base_overlay_v2_men_player_injury_weight025",
)
ALPHA_PROFILES = (
    "core_alpha_v1",
    "none",
    "harry_only",
    "quality_only",
    "quality_only_women_light",
    "quality_only_men_core_women",
    "quality_only_men_quality_blocks_women",
    "quality_wins_only_men_quality_blocks_women",
    "opp_rank_only_men_quality_blocks_women",
    "quality_only_men_harry_quality_women",
    "quality_only_men_harry_blocks_women",
    "women_blocks_only",
)
FEATURE_PROFILES = (
    "baseline_v1",
    "seed_quality_interaction",
    "seed_quality_interaction_women_conservative",
    "women_tossup_quality_conservative",
    "seed_women_consensus_interaction",
    "seed_quality_plus_women_consensus",
    "strength_blend_alt",
    "tossup_upset_v1",
    "lr_pruned_only_v1",
    "lr_ratings_only_v1",
    "lr_women_fix_only_v1",
    "lr_ratings_core_v2a",
    "lr_ratings_core_v2b",
    "lr_ratings_core_v2c",
    "lr_ratings_definition_v1",
    "lr_carry_elo_definition_v1",
    "lr_carry_elo_definition_confirm80",
    "lr_colley_definition_v1",
    "lr_srs_definition_v1_clip15",
    "lr_srs_definition_confirm20",
    "lr_pruned_core_v1",
    "women_slice_redesign_v1_architecture",
    "women_slice_redesign_v1_no_seed_interaction",
    "women_opp_rank_redesign_v1_architecture",
    "women_opp_rank_redesign_v1_no_seed_interaction",
    "women_qualitywins_redesign_v1_architecture",
    "women_qualitywins_redesign_v1_with_seed_interaction",
)
WOMEN_QUALITY_PROFILES = (
    "legacy_v1",
    "consensus_rebuild_v2",
    "consensus_rebuild_v3",
    "consensus_rebuild_v4",
    "consensus_rebuild_v4a",
    "consensus_rebuild_v4b",
    "consensus_rebuild_v5",
    "consensus_rebuild_v6",
)
WOMEN_RANKING_PROVIDERS = (
    "internal_fallback",
    "external_consensus_v1",
    "external_consensus_v2",
    "historical_consensus_snapshots_v1",
)


@dataclass(slots=True)
class JIBaseConfig:
    gender: Literal["M", "W"]
    model_family: Literal["JI_spread_xgb", "JI_lr_control", "JI_lgb_control", "JI_node_control"] = "JI_spread_xgb"
    calibration_mode: Literal["none", "isotonic_gender"] = "none"
    alpha_profile: Literal[
        "core_alpha_v1",
        "none",
        "harry_only",
        "quality_only",
        "quality_only_women_light",
        "quality_only_men_core_women",
        "quality_only_men_quality_blocks_women",
        "quality_wins_only_men_quality_blocks_women",
        "opp_rank_only_men_quality_blocks_women",
        "quality_only_men_harry_quality_women",
        "quality_only_men_harry_blocks_women",
        "women_blocks_only",
    ] = "core_alpha_v1"
    sidecar_profile: Literal["none", "text_embeddings_v1", "graph_team_embedding_v1"] = "none"
    feature_profile: Literal[
        "baseline_v1",
        "seed_quality_interaction",
        "seed_quality_interaction_women_conservative",
        "women_tossup_quality_conservative",
        "seed_women_consensus_interaction",
        "seed_quality_plus_women_consensus",
        "strength_blend_alt",
        "tossup_upset_v1",
        "lr_pruned_only_v1",
        "lr_ratings_only_v1",
        "lr_women_fix_only_v1",
        "lr_ratings_core_v2a",
        "lr_ratings_core_v2b",
        "lr_ratings_core_v2c",
        "lr_ratings_definition_v1",
        "lr_carry_elo_definition_v1",
        "lr_carry_elo_definition_confirm80",
        "lr_colley_definition_v1",
        "lr_srs_definition_v1_clip15",
        "lr_srs_definition_confirm20",
        "lr_pruned_core_v1",
        "women_slice_redesign_v1_architecture",
        "women_slice_redesign_v1_no_seed_interaction",
        "women_opp_rank_redesign_v1_architecture",
        "women_opp_rank_redesign_v1_no_seed_interaction",
        "women_qualitywins_redesign_v1_architecture",
        "women_qualitywins_redesign_v1_with_seed_interaction",
    ] = "baseline_v1"
    women_quality_profile: Literal[
        "legacy_v1",
        "consensus_rebuild_v2",
        "consensus_rebuild_v3",
        "consensus_rebuild_v4",
        "consensus_rebuild_v4a",
        "consensus_rebuild_v4b",
        "consensus_rebuild_v5",
        "consensus_rebuild_v6",
    ] = "legacy_v1"
    women_ranking_provider: Literal[
        "internal_fallback",
        "external_consensus_v1",
        "external_consensus_v2",
        "historical_consensus_snapshots_v1",
    ] = "internal_fallback"
    isotonic_min_samples: int = 20
    recent_window: int = 5
    lr_c_m: float = 1.0
    lr_c_w: float = 1.0

    def resolved_feature_profile(self) -> str:
        return self.feature_profile

    def resolved_rating_profile(self) -> str:
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v2":
            return "ji_quality_elo_v2_women_consensus"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v3":
            return "ji_quality_elo_v3_women_consensus_legacy_blend"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v4":
            return "ji_quality_elo_v4_women_consensus_shrunk"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v4a":
            return "ji_quality_elo_v4a_women_consensus_more_conservative"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v4b":
            return "ji_quality_elo_v4b_women_harry_more_conservative"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v5":
            return "ji_quality_elo_v5_women_consensus_more_shrunk"
        if self.gender == "W" and self.women_quality_profile == "consensus_rebuild_v6":
            return "ji_quality_elo_v6_women_upstream_consensus"
        return "ji_quality_elo_v1"

    def resolved_selection_objective(self) -> str:
        return "total_cv_brier_calibrated"

    def resolved_lr_c(self) -> float:
        return float(self.lr_c_m if self.gender == "M" else self.lr_c_w)

    def resolved_model_features(self) -> list[str]:
        if self.feature_profile == "lr_carry_elo_definition_confirm80":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo80",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo80",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_carry_elo_definition_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_colley_definition_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_ColleyNC",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_ColleyNC",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_ColleyNC",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_ColleyNC",
            ]

        if self.feature_profile == "lr_srs_definition_v1_clip15":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRSClip15",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRSClip15",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_srs_definition_confirm20":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRSClip20",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRSClip20",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_ratings_definition_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_EffSRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_EffSRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_ratings_core_v2a":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo",
                "Delta_Quality",
                "Delta_Colley",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_ratings_core_v2b":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo",
                "Delta_Quality",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
            ]

        if self.feature_profile == "lr_ratings_core_v2c":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
            ]

        if self.feature_profile == "lr_pruned_only_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_Quality",
                    "Delta_neff",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_Quality",
                "Delta_neff",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Quality",
            ]

        if self.feature_profile == "lr_ratings_only_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_sum",
                    "Seed_prod",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "EloProb",
                    "Delta_Quality",
                    "Delta_oeff",
                    "Delta_deff",
                    "Delta_neff",
                    "Delta_efg",
                    "Delta_tor",
                    "Delta_orpct",
                    "Delta_ftr",
                    "Delta_pace",
                    "strength_blend",
                    "Seed_x_Quality",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Delta_CarryElo",
                    "Delta_Colley",
                    "Delta_SRS",
                ]
            return [
                "Delta_Seed",
                "Seed_sum",
                "Seed_prod",
                "Seed_gap_abs",
                "Delta_Elo",
                "EloProb",
                "Delta_Quality",
                "Delta_oeff",
                "Delta_deff",
                "Delta_neff",
                "Delta_efg",
                "Delta_tor",
                "Delta_orpct",
                "Delta_ftr",
                "Delta_pace",
                "strength_blend",
                "Seed_x_Quality",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Delta_CarryElo",
                "Delta_Colley",
                "Delta_SRS",
            ]

        if self.feature_profile == "lr_women_fix_only_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_sum",
                    "Seed_prod",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "EloProb",
                    "Delta_Quality",
                    "Delta_oeff",
                    "Delta_deff",
                    "Delta_neff",
                    "Delta_efg",
                    "Delta_tor",
                    "Delta_orpct",
                    "Delta_ftr",
                    "Delta_pace",
                    "strength_blend",
                    "Seed_x_Quality",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                ]
            return [
                "Delta_Seed",
                "Seed_sum",
                "Seed_prod",
                "Seed_gap_abs",
                "Delta_Elo",
                "EloProb",
                "Delta_Quality",
                "Delta_oeff",
                "Delta_deff",
                "Delta_neff",
                "Delta_efg",
                "Delta_tor",
                "Delta_orpct",
                "Delta_ftr",
                "Delta_pace",
                "strength_blend",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "lr_pruned_core_v1":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "QualityWins_diff",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "women_slice_redesign_v1_architecture":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_WomenCompositeQualityV5",
                "Delta_WomenQualityWinsStrength",
                "Delta_WomenOpponentTournamentStrength",
                "Delta_WomenRimProtectionStrength",
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Colley",
                "Delta_SRS",
                "Seed_x_WomenOpponentTournamentStrength",
            ]

        if self.feature_profile == "women_slice_redesign_v1_no_seed_interaction":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_WomenCompositeQualityV5",
                "Delta_WomenQualityWinsStrength",
                "Delta_WomenOpponentTournamentStrength",
                "Delta_WomenRimProtectionStrength",
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Colley",
                "Delta_SRS",
            ]

        if self.feature_profile == "women_opp_rank_redesign_v1_architecture":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Colley",
                "Delta_SRS",
                "Delta_Quality",
                "QualityWins_diff",
                "Delta_WomenOpponentTournamentStrengthV2",
                "AvgBlkDiff_diff",
                "Seed_x_WomenOpponentTournamentStrengthV2",
            ]

        if self.feature_profile == "women_opp_rank_redesign_v1_no_seed_interaction":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Colley",
                "Delta_SRS",
                "Delta_Quality",
                "QualityWins_diff",
                "Delta_WomenOpponentTournamentStrengthV2",
                "AvgBlkDiff_diff",
            ]

        if self.feature_profile == "women_qualitywins_redesign_v1_architecture":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "Delta_WomenQualityWinsStrengthV2",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
            ]

        if self.feature_profile == "women_qualitywins_redesign_v1_with_seed_interaction":
            if self.gender == "M":
                return [
                    "Delta_Seed",
                    "Seed_gap_abs",
                    "Delta_Elo",
                    "Delta_CarryElo85",
                    "Delta_Quality",
                    "Delta_neff",
                    "Delta_Colley",
                    "Delta_SRS",
                    "QualityWins_diff",
                    "OpponentQualityTournamentRank_diff",
                    "Seed_x_Quality",
                    "Seed_x_Colley",
                ]
            return [
                "Delta_Seed",
                "Seed_gap_abs",
                "Delta_Elo",
                "Delta_CarryElo85",
                "Delta_Quality",
                "Delta_Colley",
                "Delta_SRS",
                "Delta_WomenQualityWinsStrengthV2",
                "OpponentQualityTournamentRank_diff",
                "AvgBlkDiff_diff",
                "Seed_x_Colley",
                "Seed_x_WomenQualityWinsStrengthV2",
            ]

        base = [
            "Delta_Seed",
            "Seed_sum",
            "Seed_prod",
            "Seed_gap_abs",
            "Delta_Elo",
            "EloProb",
            "Delta_Quality",
            "Delta_oeff",
            "Delta_deff",
            "Delta_neff",
            "Delta_efg",
            "Delta_tor",
            "Delta_orpct",
            "Delta_ftr",
            "Delta_pace",
        ]
        if self.feature_profile == "strength_blend_alt":
            base.append("strength_blend_alt")
        else:
            base.append("strength_blend")
        if self.feature_profile in {
            "seed_quality_interaction",
            "seed_quality_interaction_women_conservative",
            "women_tossup_quality_conservative",
            "seed_quality_plus_women_consensus",
        }:
            base.append("Seed_x_Quality")
        if self.feature_profile in {"seed_women_consensus_interaction", "seed_quality_plus_women_consensus"} and self.gender == "W":
            base.append("Seed_x_WomenConsensusQuality")
        if self.feature_profile == "tossup_upset_v1":
            base.extend(["CloseGameStrength", "UpsetPressure"])

        if self.alpha_profile == "quality_only_men_core_women":
            if self.gender == "M":
                base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
                return base
            base.append("harry_Rating_diff")
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff", "AvgBlkDiff_diff"])
            return base

        if self.alpha_profile == "quality_only_men_quality_blocks_women":
            if self.gender == "M":
                base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
                return base
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff", "AvgBlkDiff_diff"])
            return base

        if self.alpha_profile == "quality_wins_only_men_quality_blocks_women":
            if self.gender == "M":
                base.append("QualityWins_diff")
                return base
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff", "AvgBlkDiff_diff"])
            return base

        if self.alpha_profile == "opp_rank_only_men_quality_blocks_women":
            if self.gender == "M":
                base.append("OpponentQualityTournamentRank_diff")
                return base
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff", "AvgBlkDiff_diff"])
            return base

        if self.alpha_profile == "quality_only_men_harry_quality_women":
            if self.gender == "M":
                base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
                return base
            base.append("harry_Rating_diff")
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
            return base

        if self.alpha_profile == "quality_only_men_harry_blocks_women":
            if self.gender == "M":
                base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
                return base
            base.extend(["harry_Rating_diff", "AvgBlkDiff_diff"])
            return base

        if self.alpha_profile in {"core_alpha_v1", "harry_only"}:
            base.append("harry_Rating_diff")
        if self.alpha_profile in {"core_alpha_v1", "quality_only", "quality_only_women_light"}:
            base.extend(["QualityWins_diff", "OpponentQualityTournamentRank_diff"])
        if self.alpha_profile in {"core_alpha_v1", "women_blocks_only"} and self.gender == "W":
            base.append("AvgBlkDiff_diff")
        return base

    def resolved_clip_bounds(self) -> tuple[float, float]:
        return (0.03, 0.97) if self.gender == "M" else (0.005, 0.995)


def build_working_ji_base_config(gender: Literal["M", "W"]) -> JIBaseConfig:
    return JIBaseConfig(
        gender=gender,
        model_family="JI_lr_control",
        calibration_mode="none",
        alpha_profile="quality_only_men_quality_blocks_women",
        feature_profile="lr_carry_elo_definition_v1",
        women_quality_profile="consensus_rebuild_v4" if gender == "W" else "legacy_v1",
        women_ranking_provider="internal_fallback",
    )


@dataclass(slots=True)
class JIBaseOverlayConfig:
    gender: Literal["M", "W"]
    overlay_source_profile: Literal["direct_priority", "direct_only"] = "direct_priority"
    allow_market: bool = True
    allow_injury: bool = True
    direct_weight: float = 0.85
    max_delta: float = 0.025
    injury_cap: float = 0.02
    injury_min_confirmed_out: int = 1
    injury_min_abs_shift: float = 0.0
    injury_mode: Literal["team_confirmed_gate", "player_level_v2"] = "team_confirmed_gate"

    def resolved_overlay_stack(self) -> str:
        if self.allow_market and self.allow_injury and self.gender == "M":
            return "market_injury"
        if self.allow_market:
            return "market_only"
        return "none"


def build_working_ji_base_overlay_config(gender: Literal["M", "W"]) -> JIBaseOverlayConfig:
    return build_ji_base_overlay_config(gender, FROZEN_OVERLAY_SUBMISSION_PROFILE)


def build_ji_base_overlay_config(
    gender: Literal["M", "W"],
    submission_profile: Literal[
        "ji_base_overlay_v1",
        "ji_base_overlay_v1_conservative_injury",
        "ji_base_overlay_v1_direct_only",
        "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
        "ji_base_overlay_v1_direct_only_injury_confirmed3",
        "ji_base_overlay_v1_direct_only_injury_confirmed4",
        "ji_base_overlay_v1_direct_only_injury_confirmed5",
        "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008",
        "ji_base_overlay_v1_men_best_women_direct_priority",
        "ji_base_overlay_v1_men_best_women_direct_only_weight070",
        "ji_base_overlay_v1_men_best_women_direct_only_weight060",
        "ji_base_overlay_v1_men_best_women_direct_only_weight050",
        "ji_base_overlay_v1_men_best_women_direct_only_weight040",
        "ji_base_overlay_v1_men_best_women_direct_only_weight030",
        "ji_base_overlay_v1_men_best_women_direct_only_weight020",
        "ji_base_overlay_v1_men_best_women_direct_only_weight025",
        "ji_base_overlay_v2_men_player_injury_weight025",
    ] = "ji_base_overlay_v1",
) -> JIBaseOverlayConfig:
    injury_cap = 0.02
    direct_weight = 0.85
    overlay_source_profile: Literal["direct_priority", "direct_only"] = "direct_priority"
    injury_min_confirmed_out = 1
    injury_min_abs_shift = 0.0
    injury_mode: Literal["team_confirmed_gate", "player_level_v2"] = "team_confirmed_gate"
    if submission_profile == "ji_base_overlay_v1_conservative_injury" and gender == "M":
        injury_cap = 0.01
    if submission_profile == "ji_base_overlay_v1_direct_only":
        overlay_source_profile = "direct_only"
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_strict_confirmed":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 2
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed3":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 3
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed4":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed5":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 5
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
            injury_min_abs_shift = 0.08
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_priority":
        if gender == "M":
            overlay_source_profile = "direct_only"
            injury_min_confirmed_out = 4
        else:
            overlay_source_profile = "direct_priority"
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight070":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.70
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight060":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.60
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight050":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.50
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight040":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.40
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight030":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.30
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight020":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.20
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight025":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
        else:
            direct_weight = 0.25
    if submission_profile == "ji_base_overlay_v2_men_player_injury_weight025":
        overlay_source_profile = "direct_only"
        if gender == "M":
            injury_min_confirmed_out = 4
            injury_mode = "player_level_v2"
        else:
            direct_weight = 0.25

    return JIBaseOverlayConfig(
        gender=gender,
        overlay_source_profile=overlay_source_profile,
        allow_market=True,
        allow_injury=(gender == "M"),
        direct_weight=direct_weight,
        max_delta=0.025,
        injury_cap=injury_cap,
        injury_min_confirmed_out=injury_min_confirmed_out,
        injury_min_abs_shift=injury_min_abs_shift,
        injury_mode=injury_mode,
    )
