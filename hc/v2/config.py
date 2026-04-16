from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ROUTES = ("probability", "spread")
MODEL_VARIANTS = ("lr", "tree", "blend")
MARKET_EXPERIMENTS = ("none", "sportsbook", "sportsbook_prediction")
FEATURE_PACKS = ("base", "external_base_pruned", "external_base", "efficiency", "opp_adjusted", "strength_full", "strength_recent")
CALIBRATION_MODES = ("direct", "basecal", "gendercal", "monotoniccal")

BASE_FEATURES = [
    "SeedNum",
    "Elo",
    "SOS",
    "WinRate",
    "AvgMargin",
    "Top50WinRate",
    "Last30WinRate",
    "AdjNetRtg",
    "RecentEffAdjNetRtg",
    "Recent30EffAdjNetRtg",
]

EFFICIENCY_FEATURES = [
    "AdjOffRtg",
    "AdjDefRtg",
    "Tempo",
]

OPPONENT_ADJUSTED_FEATURES = [
    "OppEFG",
    "OppTOVPct",
    "OppFTR",
    "OppORBPct",
]

EXTERNAL_BASE_COMMON_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "Elo",
    "WinRate",
    "AvgMargin",
    "Top50WinRate",
    "Last30WinRate",
    "CloseGameWinRate",
    "CloseGameMargin",
    "Last30SOS",
    "PageRank",
    "ExternalCompositeStrength",
    "ExternalFallbackElo",
    "ExternalFallbackSOS",
    "ExternalFallbackMargin",
]

EXTERNAL_BASE_MEN_FEATURES = [
    "ExternalBPIStrength",
    "ExternalPOMStrength",
    "ExternalNETStrength",
    "ExternalWABStrength",
    "ExternalELORankStrength",
    "ExternalSORStrength",
    "ExternalTRankStrength",
    "MasseyPOMStrength",
    "MasseyMORStrength",
    "MasseyNETStrength",
]

EXTERNAL_BASE_WOMEN_FEATURES = [
    "ExternalNETStrength",
    "ExternalRPIStrength",
    "ExternalPredRPIStrength",
    "ExternalELORankStrength",
    "ExternalAPStrength",
    "ExternalCoachesStrength",
]

EXTERNAL_BASE_PRUNED_COMMON_FEATURES = [
    "SeedNum",
    "SeedPriorExpectedWins",
    "Elo",
    "AvgMargin",
    "CloseGameWinRate",
    "ExternalCompositeStrength",
    "ExternalFallbackElo",
]

EXTERNAL_BASE_PRUNED_MEN_FEATURES = [
    "Last30SOS",
    "ExternalBPIStrength",
    "ExternalPOMStrength",
    "ExternalNETStrength",
    "MasseyPOMStrength",
    "MasseyMORStrength",
]

EXTERNAL_BASE_PRUNED_WOMEN_FEATURES = [
    "ExternalNETStrength",
    "ExternalRPIStrength",
    "ExternalPredRPIStrength",
]

MEN_EXTRA_FEATURES = ["Ext_WN_ELO", "Ext_WN_NET", "ExtCompositeStrength"]
WOMEN_EXTRA_FEATURES = ["Ext_WN_ELO", "Ext_WN_NET"]


@dataclass(slots=True)
class V2Config:
    gender: Literal["M", "W"]
    route: Literal["probability", "spread"] = "probability"
    model_variant: Literal["lr", "tree", "blend"] = "blend"
    market_mode: Literal["none", "sportsbook", "sportsbook_prediction"] = "none"
    feature_pack: Literal["base", "external_base_pruned", "external_base", "efficiency", "opp_adjusted", "strength_full", "strength_recent"] = "opp_adjusted"
    calibration_mode: Literal["direct", "basecal", "gendercal", "monotoniccal"] = "basecal"
    tree_model: Literal["histgb"] = "histgb"
    lr_weight: float = 0.4
    tree_weight: float = 0.6
    clip_low: float = 0.02
    clip_high: float = 0.98
    market_weight: float = 0.2
    bounded_pull_delta: float = 0.075
    recent_window: int = 5
    spread_logit_scale: float = 7.5
    features: list[str] = field(default_factory=list)

    def resolved_features(self) -> list[str]:
        if self.features:
            return list(self.features)
        if self.feature_pack == "base":
            feature_pack = list(BASE_FEATURES)
        elif self.feature_pack == "external_base_pruned":
            feature_pack = list(EXTERNAL_BASE_PRUNED_COMMON_FEATURES)
            if self.gender == "M":
                feature_pack.extend(EXTERNAL_BASE_PRUNED_MEN_FEATURES)
            else:
                feature_pack.extend(EXTERNAL_BASE_PRUNED_WOMEN_FEATURES)
        elif self.feature_pack == "external_base":
            feature_pack = list(EXTERNAL_BASE_COMMON_FEATURES)
            if self.gender == "M":
                feature_pack.extend(EXTERNAL_BASE_MEN_FEATURES)
            else:
                feature_pack.extend(EXTERNAL_BASE_WOMEN_FEATURES)
        elif self.feature_pack == "efficiency":
            feature_pack = [*BASE_FEATURES, *EFFICIENCY_FEATURES]
        elif self.feature_pack in {"strength_full", "strength_recent"}:
            feature_pack = [
                "SeedNum",
                "Elo",
                "SOS",
                "WinRate",
                "AvgMargin",
                "Top50WinRate",
                "Last30WinRate",
                "SeedPriorExpectedWins",
                "StrengthNet",
                "StrengthOff",
                "StrengthDef",
                "StrengthTempo",
                "StrengthSOS",
                "StrengthTop50",
                "StrengthPath",
                "StrengthMomentum",
                "StrengthOffMomentum",
                "StrengthDefMomentum",
            ]
        else:
            feature_pack = [*BASE_FEATURES, *EFFICIENCY_FEATURES, *OPPONENT_ADJUSTED_FEATURES]
        if self.feature_pack in {"strength_full", "strength_recent"}:
            return feature_pack
        if self.feature_pack in {"external_base", "external_base_pruned"}:
            return feature_pack
        if self.gender == "M":
            return feature_pack + MEN_EXTRA_FEATURES
        return feature_pack + WOMEN_EXTRA_FEATURES

    def resolved_calibration_mode(self) -> str:
        if self.route == "probability":
            return "direct"
        return self.calibration_mode

    def resolved_learner_family(self) -> str:
        if self.model_variant == "lr":
            return "linear"
        if self.model_variant == "tree":
            return "tree"
        return "blend"

    def resolved_linear_alpha(self) -> float:
        if self.feature_pack in {"external_base", "external_base_pruned"}:
            return 0.35 if self.gender == "M" else 1.25
        return 1.0

    def resolved_clip_bounds(self) -> tuple[float, float]:
        if self.feature_pack in {"external_base", "external_base_pruned"}:
            if self.gender == "M":
                return 0.03, 0.97
            return 0.01, 0.99
        return self.clip_low, self.clip_high

    def resolved_interactions(self) -> list[tuple[str, str, str]]:
        if self.feature_pack not in {"external_base", "external_base_pruned"}:
            return []

        interactions = [
            ("SeedNum_x_ExternalCompositeStrength", "SeedNum", "ExternalCompositeStrength"),
            ("CloseGameWinRate_x_AvgMargin", "CloseGameWinRate", "AvgMargin"),
        ]
        if self.feature_pack == "external_base":
            interactions.append(("Last30SOS_x_ExternalCompositeStrength", "Last30SOS", "ExternalCompositeStrength"))
        if self.gender == "M":
            interactions.append(("SeedNum_x_ExternalPOMStrength", "SeedNum", "ExternalPOMStrength"))
            if self.feature_pack == "external_base":
                interactions.append(("ExternalSORStrength_x_ExternalCompositeStrength", "ExternalSORStrength", "ExternalCompositeStrength"))
        else:
            interactions.append(("SeedNum_x_ExternalNETStrength", "SeedNum", "ExternalNETStrength"))
            if self.feature_pack == "external_base":
                interactions.append(
                    (
                        "ExternalAPStrength_x_ExternalCompositeStrength",
                        "ExternalAPStrength",
                        "ExternalCompositeStrength",
                    )
                )
        return interactions

    def resolved_gender_profile(self) -> str:
        if self.feature_pack == "external_base_pruned":
            return "men_external_pruned" if self.gender == "M" else "women_external_pruned"
        if self.feature_pack == "external_base":
            return "men_external_full" if self.gender == "M" else "women_external_full"
        return f"{self.gender}_{self.feature_pack}"
