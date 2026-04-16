from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MODEL_FAMILIES = (
    "gold_linear",
    "gold_harry_lr",
    "gold_harry_xgb_spread",
    "gold_xgb_spread_light",
    "gold_min_lr",
    "gold_min_xgb_spread",
    "gold_tree_control",
    "gold_spread_control",
)
CALIBRATION_MODES = ("none", "isotonic_gender", "monotonic_spline_gender")
FEATURE_PROFILES = (
    "gold_recover_wide",
    "gold_harry_m",
    "gold_harry_w",
    "gold_pruned_m",
    "gold_pruned_w",
    "gold_augmented_m",
    "gold_augmented_w",
    "gold_min_m",
    "gold_min_w",
    "gold_core_m",
    "gold_core_w",
)

COMMON_BASE_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "CarryoverElo",
    "MoveElo",
    "GLMQuality",
    "SRS",
    "Colley",
    "MasseyComposite",
    "CustomNetRating",
    "OpponentQualityScore",
    "AvgMargin",
    "SOS",
    "Last30SOS",
    "CloseGameWinRate",
    "OTNormalizedMargin",
    "QualityWins",
]

MEN_ONLY_FEATURES = ["APStrength"]
WOMEN_ONLY_FEATURES: list[str] = []

PRUNED_MEN_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "MoveElo",
    "GLMQuality",
    "MasseyComposite",
    "CustomNetRating",
    "OpponentQualityScore",
    "QualityWins",
    "AvgMargin",
    "APStrength",
]
PRUNED_WOMEN_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "MoveElo",
    "GLMQuality",
    "MasseyComposite",
    "CustomNetRating",
    "OpponentQualityScore",
    "QualityWins",
    "AvgBlkDiff",
]

AUGMENTED_MEN_FEATURES = list(dict.fromkeys(COMMON_BASE_FEATURES + MEN_ONLY_FEATURES + ["CustomStrengthCore"]))
AUGMENTED_WOMEN_FEATURES = list(dict.fromkeys(COMMON_BASE_FEATURES + ["AvgBlkDiff", "CustomStrengthCore"]))

MIN_MEN_FEATURES = [
    "SeedNum",
    "GoldConsensusStrength",
    "QualityWins",
    "AvgMargin",
    "APStrength",
]
MIN_WOMEN_FEATURES = [
    "SeedNum",
    "GoldConsensusStrength",
    "QualityWins",
    "AvgBlkDiff",
]

HARRY_MEN_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "OpponentQualityTournamentRank",
    "QualityWins",
    "AvgMargin",
    "harry_Rating",
    "InjuryAdjustedStrength",
]
HARRY_WOMEN_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "OpponentQualityTournamentRank",
    "QualityWins",
    "harry_Rating",
    "AvgBlkDiff",
]


@dataclass(slots=True)
class GoldConfig:
    gender: Literal["M", "W"]
    model_family: Literal[
        "gold_linear",
        "gold_harry_lr",
        "gold_harry_xgb_spread",
        "gold_xgb_spread_light",
        "gold_min_lr",
        "gold_min_xgb_spread",
        "gold_tree_control",
        "gold_spread_control",
    ] = "gold_linear"
    calibration_mode: Literal["none", "isotonic_gender", "monotonic_spline_gender"] = "none"
    feature_profile: Literal[
        "gold_recover_wide",
        "gold_harry_m",
        "gold_harry_w",
        "gold_pruned_m",
        "gold_pruned_w",
        "gold_augmented_m",
        "gold_augmented_w",
        "gold_min_m",
        "gold_min_w",
        "gold_core_m",
        "gold_core_w",
    ] | None = None
    recent_window: int = 5
    rating_source_profile: Literal[
        "current_default",
        "a_tier_default",
        "m_ap_removed_only",
        "w_polls_removed_only",
        "b_tier_plus_polls",
        "c_all_external",
    ] = "current_default"
    overlay_source_profile: Literal[
        "current_default",
        "a_tier_default",
        "direct_only",
        "direct_priority",
        "b_tier_with_futures",
        "c_all_sources",
    ] = "current_default"

    def resolved_feature_profile(self) -> str:
        if self.feature_profile is not None:
            return self.feature_profile
        if self.model_family in {"gold_min_lr", "gold_min_xgb_spread"}:
            return "gold_min_m" if self.gender == "M" else "gold_min_w"
        if self.model_family in {"gold_harry_lr", "gold_harry_xgb_spread"}:
            return "gold_harry_m" if self.gender == "M" else "gold_harry_w"
        if self.model_family == "gold_xgb_spread_light":
            return "gold_pruned_m" if self.gender == "M" else "gold_pruned_w"
        return "gold_recover_wide"

    def resolved_candidate_features(self) -> list[str]:
        feature_profile = self.resolved_feature_profile()
        if feature_profile == "gold_recover_wide":
            features = list(COMMON_BASE_FEATURES)
            if self.gender == "M":
                features.extend(MEN_ONLY_FEATURES)
                if self.resolved_rating_source_profile() in {"a_tier_default", "m_ap_removed_only"}:
                    features = [feature for feature in features if feature != "APStrength"]
            return features
        if feature_profile == "gold_pruned_m":
            return list(PRUNED_MEN_FEATURES)
        if feature_profile == "gold_pruned_w":
            return list(PRUNED_WOMEN_FEATURES)
        if feature_profile == "gold_harry_m":
            return list(HARRY_MEN_FEATURES)
        if feature_profile == "gold_harry_w":
            return list(HARRY_WOMEN_FEATURES)
        if feature_profile == "gold_augmented_m":
            return list(AUGMENTED_MEN_FEATURES)
        if feature_profile == "gold_augmented_w":
            return list(AUGMENTED_WOMEN_FEATURES)
        if feature_profile == "gold_min_m":
            return list(MIN_MEN_FEATURES)
        if feature_profile == "gold_min_w":
            return list(MIN_WOMEN_FEATURES)
        features = list(COMMON_BASE_FEATURES)
        if self.gender == "M":
            features.extend(MEN_ONLY_FEATURES)
        else:
            features.extend(WOMEN_ONLY_FEATURES)
        return features

    def resolved_interactions(self) -> list[tuple[str, str, str]]:
        feature_profile = self.resolved_feature_profile()
        if feature_profile == "gold_min_m":
            return [("Seed_x_GoldConsensusStrength", "SeedNum", "GoldConsensusStrength")]
        if feature_profile == "gold_min_w":
            return []
        if feature_profile == "gold_harry_m":
            return [("Seed_x_harry_Rating", "SeedNum", "harry_Rating")]
        if feature_profile == "gold_harry_w":
            return []
        if feature_profile in {"gold_pruned_m", "gold_pruned_w"}:
            return []
        interactions = [
            ("Seed_x_MasseyComposite", "SeedNum", "MasseyComposite"),
            ("Seed_x_CustomNetRating", "SeedNum", "CustomNetRating"),
            ("SOS_x_MasseyComposite", "SOS", "MasseyComposite"),
            ("CloseGameWinRate_x_AvgMargin", "CloseGameWinRate", "AvgMargin"),
        ]
        if self.gender == "M":
            interactions.append(("APStrength_x_MasseyComposite", "APStrength", "MasseyComposite"))
        return interactions

    def resolved_model_features(self) -> list[str]:
        feature_profile = self.resolved_feature_profile()
        if feature_profile == "gold_recover_wide":
            base = [
                "SeedNum_diff",
                "SeedPriorExpectedWins_diff",
                "CarryoverElo_diff",
                "MoveElo_diff",
                "GLMQuality_diff",
                "SRS_diff",
                "Colley_diff",
                "MasseyComposite_diff",
                "CustomNetRating_diff",
                "OpponentQualityScore_diff",
                "AvgMargin_diff",
                "SOS_diff",
                "Last30SOS_diff",
                "CloseGameWinRate_diff",
                "OTNormalizedMargin_diff",
                "QualityWins_diff",
                "SeedAbsGap",
                "SeedPairProduct",
                "Seed_x_MasseyComposite_diff",
                "Seed_x_CustomNetRating_diff",
                "SOS_x_MasseyComposite_diff",
                "CloseGameWinRate_x_AvgMargin_diff",
            ]
            if self.gender == "M":
                base.extend(["APStrength_diff", "APStrength_x_MasseyComposite_diff"])
                if self.resolved_rating_source_profile() in {"a_tier_default", "m_ap_removed_only"}:
                    base = [feature for feature in base if feature not in {"APStrength_diff", "APStrength_x_MasseyComposite_diff"}]
            return base
        if feature_profile == "gold_pruned_m":
            return [
                "SeedNum_diff",
                "SeedPriorExpectedWins_diff",
                "MoveElo_diff",
                "GLMQuality_diff",
                "MasseyComposite_diff",
                "CustomNetRating_diff",
                "OpponentQualityScore_diff",
                "QualityWins_diff",
                "AvgMargin_diff",
                "APStrength_diff",
                "SeedAbsGap",
            ]
        if feature_profile == "gold_pruned_w":
            return [
                "SeedNum_diff",
                "SeedPriorExpectedWins_diff",
                "MoveElo_diff",
                "GLMQuality_diff",
                "MasseyComposite_diff",
                "CustomNetRating_diff",
                "OpponentQualityScore_diff",
                "QualityWins_diff",
                "AvgBlkDiff_diff",
                "SeedAbsGap",
            ]
        if feature_profile == "gold_harry_m":
            return [
                "SeedNum_diff",
                "SeedPriorExpectedWins_diff",
                "SeedAbsGap",
                "OpponentQualityTournamentRank_diff",
                "QualityWins_diff",
                "AvgMargin_diff",
                "harry_Rating_diff",
                "InjuryAdjustedStrength_diff",
                "Seed_x_harry_Rating_diff",
            ]
        if feature_profile == "gold_harry_w":
            return [
                "SeedNum_diff",
                "SeedPriorExpectedWins_diff",
                "SeedAbsGap",
                "OpponentQualityTournamentRank_diff",
                "QualityWins_diff",
                "harry_Rating_diff",
                "AvgBlkDiff_diff",
            ]
        if feature_profile == "gold_augmented_m":
            return self._augmented_base_features(include_ap=True)
        if feature_profile == "gold_augmented_w":
            return self._augmented_base_features(include_ap=False)
        if feature_profile == "gold_min_m":
            return [
                "SeedNum_diff",
                "SeedAbsGap",
                "GoldConsensusStrength_diff",
                "QualityWins_diff",
                "AvgMargin_diff",
                "APStrength_diff",
                "Seed_x_GoldConsensusStrength_diff",
            ]
        if feature_profile == "gold_min_w":
            return [
                "SeedNum_diff",
                "SeedAbsGap",
                "GoldConsensusStrength_diff",
                "QualityWins_diff",
                "AvgBlkDiff_diff",
            ]
        return self._augmented_base_features(include_ap=self.gender == "M")

    def resolved_rating_profile(self) -> str:
        if self.resolved_feature_profile() in {"gold_min_m", "gold_min_w"}:
            return "gold_consensus_minimal"
        if self.resolved_feature_profile() in {"gold_harry_m", "gold_harry_w"}:
            return "harry_rating_core"
        if self.resolved_feature_profile() in {"gold_pruned_m", "gold_pruned_w"}:
            return "gold_pruned_multi_rating"
        if self.resolved_feature_profile() in {"gold_augmented_m", "gold_augmented_w"}:
            return "gold_augmented_custom_strength"
        return "gold_recover_multi_rating"

    def resolved_selection_objective(self) -> str:
        if self.resolved_feature_profile() in {"gold_min_m", "gold_min_w"}:
            return "lb_first_minimal"
        if self.resolved_feature_profile() in {"gold_harry_m", "gold_harry_w"}:
            return "submission_first_harry"
        return "latest_first_lb_proxy"

    def resolved_rating_source_profile(self) -> str:
        return self.rating_source_profile

    def resolved_overlay_source_profile(self) -> str:
        return self.overlay_source_profile

    def resolved_lr_c(self) -> float:
        if self.resolved_feature_profile() in {"gold_min_m", "gold_min_w"}:
            return 6.0 if self.gender == "M" else 1.5
        if self.resolved_feature_profile() in {"gold_harry_m", "gold_harry_w"}:
            return 2.5 if self.gender == "M" else 1.25
        if self.resolved_feature_profile() in {"gold_pruned_m", "gold_pruned_w"}:
            return 25.0 if self.gender == "M" else 0.5
        return 100.0 if self.gender == "M" else 0.15

    def resolved_clip_bounds(self) -> tuple[float, float]:
        if self.gender == "M":
            return 0.03, 0.97
        return 0.005, 0.995

    def _augmented_base_features(self, *, include_ap: bool) -> list[str]:
        base = self.resolved_model_features_for_recover()
        base.append("CustomStrengthCore_diff")
        if not include_ap:
            base.append("AvgBlkDiff_diff")
        return list(dict.fromkeys(base))

    def resolved_model_features_for_recover(self) -> list[str]:
        base = [
            "SeedNum_diff",
            "SeedPriorExpectedWins_diff",
            "CarryoverElo_diff",
            "MoveElo_diff",
            "GLMQuality_diff",
            "SRS_diff",
            "Colley_diff",
            "MasseyComposite_diff",
            "CustomNetRating_diff",
            "OpponentQualityScore_diff",
            "AvgMargin_diff",
            "SOS_diff",
            "Last30SOS_diff",
            "CloseGameWinRate_diff",
            "OTNormalizedMargin_diff",
            "QualityWins_diff",
            "SeedAbsGap",
            "SeedPairProduct",
            "Seed_x_MasseyComposite_diff",
            "Seed_x_CustomNetRating_diff",
            "SOS_x_MasseyComposite_diff",
            "CloseGameWinRate_x_AvgMargin_diff",
        ]
        if include_ap:
            base.extend(["APStrength_diff", "APStrength_x_MasseyComposite_diff"])
        return base
