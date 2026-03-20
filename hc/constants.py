from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache" / "hc"
RAW_TEXT_DIR = ROOT / "data" / "raw_text"
TEXT_EMBED_DIR = ROOT / "data" / "text_embeddings"

PRIMARY_BLEND = {"linear": 0.35, "histgb": 0.65}
LEGACY_OVERRIDE_MARGIN = {"M": 0.01, "W": 0.02}
CHALLENGER_TRUST_MARGIN = {"M": 0.015, "W": 0.01}
PRIMARY_TARGET_MAX = 0.1505
HOLDOUT_2Y_ACCEPT_MAX = 0.1230
GUARDRAIL_7Y_ACCEPT_MAX = 0.1550
CACHE_SCHEMA_VERSION = "cachev2"
FUSION_LOGIC_VERSION = "fusionv3"
PUBLIC_ROUTE_VERSION = "publicv10_runtime"
ENABLE_SILVER_HISTORY = False
MARKET_POLICY_VERSION = "marketv1"
HISTORICAL_OVERRIDE_POLICY_VERSION = "ovr_off"
CALIBRATION_POLICY_VERSION = "calibv1_beta"
SELECTOR_POLICY_VERSION = "selectorv1_groupsoft"
LEGACY_TRUST_POLICY_VERSION = "legacytrustv1"
ENABLE_HISTORICAL_SEASON_OVERRIDES = False
MEN_SELECTOR_GATE_START_SEASON = 2021
MEN_SELECTOR_GATE_END_SEASON = 0
MEN_HISTGB_META_GATE_START_SEASON = 2021
MEN_HISTGB_META_GATE_END_SEASON = 0
MEN_LINEAR_META_GATE_START_SEASON = 2022
MEN_LINEAR_META_GATE_END_SEASON = 0
MEN_RECENT_LINEAR_GATE_START_SEASON = 2024
MEN_RECENT_LINEAR_GATE_MARKET_PROB = 0.45
WOMEN_RECENT_SELECTOR_START_SEASON = 2024
WOMEN_HOST_MARKET_MIN_LR_GATE_HOST_MIN = 1.0
WOMEN_HOST_MARKET_MIN_LR_GATE_EARLY_ROUND_MIN = 1.0
MEN_SEASON_ROUTE_OVERRIDES = {}
MEN_SEASON_BLEND_OVERRIDES = {}
WOMEN_SEASON_ROUTE_OVERRIDES = {}
LOSS_SELECTOR_TOP_K = {"M": 3, "W": 3}
LOSS_SELECTOR_META_BLEND = {"M": 0.25, "W": 0.15}
MIN_TRAIN_SEASONS = 8
MIN_META_SEASONS = 5
PRIMARY_YEARS = 5
GUARDRAIL_YEARS = 7
HOLDOUT_YEARS = 2
LB_PROXY_SEASONS = (2022, 2023, 2024, 2025)

MARKET_POLICY_PRE_TIP_ALL = "pre_tip_all_round"
MARKET_POLICY_SELECTION_WEEK = "selection_week_only"
MARKET_POLICY_SELECTION_WEEK_PLUS = "selection_week_plus_pre_tip"
PROFILE_AGGRESSIVE = "aggressive"
PROFILE_CLEAN = "clean"
PROFILE_CHOICES = (PROFILE_AGGRESSIVE, PROFILE_CLEAN)
MARKET_POLICY_BY_GENDER = {
    "M": MARKET_POLICY_PRE_TIP_ALL,
    "W": MARKET_POLICY_SELECTION_WEEK_PLUS,
}
MARKET_POLICY_BY_PROFILE = {
    PROFILE_AGGRESSIVE: {
        "M": MARKET_POLICY_PRE_TIP_ALL,
        "W": MARKET_POLICY_SELECTION_WEEK_PLUS,
    },
    PROFILE_CLEAN: {
        "M": MARKET_POLICY_SELECTION_WEEK,
        "W": MARKET_POLICY_SELECTION_WEEK,
    },
}
MARKET_POLICY_CHOICES = (
    MARKET_POLICY_PRE_TIP_ALL,
    MARKET_POLICY_SELECTION_WEEK,
    MARKET_POLICY_SELECTION_WEEK_PLUS,
)

MARKET_COVERAGE_THRESHOLD = {"M": 0.45, "W": 0.08}
MARKET_ROUTE_MIN_ROWS = {"M": 180, "W": 40}
TEXT_ROUTE_MIN_DOCS = {"M": 12, "W": 12}
TABPFN_MAX_FEATURES = {"M": 32, "W": 24}
TEXT_COMPONENTS = ("Recent3", "Recent5", "Weighted5")
TEXT_DIM_CHOICES = (16, 32, 64)
MARKET_CONSENSUS_FEATURES = [
    "MarketProbMean",
    "MarketProbMedian",
    "MarketProbStd",
    "SpreadMean",
    "SpreadMedian",
    "SpreadStd",
    "BookCountMean",
    "BookCountMax",
    "BookCountTotal",
    "MarketRowCount",
    "MarketSourceCount",
]

RESULT_PREFIX = "hc"
BENCHMARKS_PATH = RESULTS_DIR / "hc_benchmarks.csv"

MEN_STRUCTURED_FEATURES = [
    "D_Elo",
    "D_WinRate",
    "D_AvgMargin",
    "D_SOS",
    "D_NetRtg_z",
    "D_Last30WinRate",
    "D_Recent30EffNetRtg_z",
    "D_DefRtg_z",
    "D_OppEFG",
    "D_OppFTR",
    "D_OppTOVPct",
    "D_OppORBPct",
    "H2HGames",
    "H2HWinPct",
    "H2HMargin",
    "CommonOppCount",
    "CommonOppMarginDiff",
    "CommonOppWinPctDiff",
    "AbsSeedDiff",
    "T1BetterSeed",
    "D_SeedNum",
]

MEN_MARKET_FEATURES = [
    "MarketProb",
    "MarketLogit",
    "MarketConfidence",
    "LastSpread",
    "AbsLastSpread",
    "AbsSeedDiff",
    "T1BetterSeed",
    "D_SeedNum",
]

MEN_PUBLIC_ROUTE_FEATURES = [
    "MarketProb",
    "MarketLogit",
    "MarketConfidence",
    "LastSpread",
    "AbsLastSpread",
    "AbsSeedDiff",
    "T1BetterSeed",
    "D_SeedNum",
    "D_ExtCompositeStrength",
    "D_HC_PublicNETRank",
    "D_HC_PublicELORank",
    "D_HC_PublicRPIRank",
    "D_HC_PublicPredRPIRank",
    "D_HC_PublicBPIRank",
    "D_HC_PublicPOMRank",
    "D_HC_PublicKPIRank",
    "D_HC_PublicSORRank",
    "D_HC_PublicAverageRank",
    "D_HC_PublicTRankRank",
    "D_HC_PublicAvgPredRank",
]

WOMEN_STRUCTURED_FEATURES = [
    "D_SeedNum",
    "AbsSeedDiff",
    "T1BetterSeed",
    "D_Elo",
    "D_WinRate",
    "D_AvgMargin",
    "D_SOS",
    "D_NetRtg_z",
    "D_DefRtg_z",
    "D_DR",
    "D_RecentEffDR",
    "D_Recent30EffDR",
    "D_OppORBPct",
    "D_Recent30EffDefRtg_z",
    "D_Recent30EffNetRtg_z",
    "T1HostLikely",
    "T2HostLikely",
    "D_HostLikely",
    "TourneyRound",
    "IsRound1Or2",
    "MarketProb",
    "LastSpread",
]

WOMEN_PUBLIC_ROUTE_FEATURES = [
    "MarketProb",
    "MarketLogit",
    "MarketConfidence",
    "LastSpread",
    "AbsLastSpread",
    "AbsSeedDiff",
    "T1BetterSeed",
    "D_SeedNum",
    "D_Elo",
    "D_WinRate",
    "D_AvgMargin",
    "D_SOS",
    "D_NetRtg_z",
    "D_DefRtg_z",
    "D_DR",
    "D_Recent30EffDefRtg_z",
    "D_Recent30EffNetRtg_z",
    "D_HostLikely",
    "IsRound1Or2",
    "D_HC_PublicNETRank",
    "D_HC_PublicELORank",
    "D_HC_PublicRPIRank",
    "D_HC_PublicPredRPIRank",
]

TEXT_ROUTE_STRUCTURED_ANCHORS = {
    "M": [
        "MarketProb",
        "AbsLastSpread",
        "AbsSeedDiff",
        "D_Elo",
        "D_NetRtg_z",
        "D_Recent30EffNetRtg_z",
        "D_OppEFG",
    ],
    "W": [
        "MarketProb",
        "AbsSeedDiff",
        "D_Elo",
        "D_NetRtg_z",
        "D_DR",
        "D_Recent30EffDefRtg_z",
        "D_HostLikely",
        "IsRound1Or2",
    ],
}

MEN_RULE_COLUMNS = [
    "MarketProb",
    "LastSpread",
    "AbsLastSpread",
    "AbsSeedDiff",
    "D_Elo",
    "D_Recent30EffNetRtg_z",
    "H2HMargin",
]

WOMEN_RULE_COLUMNS = [
    "MarketProb",
    "LastSpread",
    "AbsSeedDiff",
    "D_HostLikely",
    "TourneyRound",
    "D_DR",
    "D_DefRtg_z",
]

RULE_MIN_SUPPORT = {"M": 0.03, "W": 0.02}
RULE_MAX_COUNT = {"M": 48, "W": 36}
RULE_MIN_EDGE = {"M": 0.045, "W": 0.04}

DEFAULT_TEXT_MODEL_DEV = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_MODEL_FULL = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_SUBMISSION_NAME = "submission_stage2_single_final.csv"


@dataclass(frozen=True)
class TrainConfig:
    gender: str
    years: int
    market_policy: str
    profile: str
    use_text: bool
    use_tabpfn: bool
    text_dim: int
    quick: bool = False

    @property
    def cache_tag(self) -> str:
        text_tag = "text" if self.use_text else "notext"
        tabpfn_tag = "tabpfn" if self.use_tabpfn else "notabpfn"
        quick_tag = "quick" if self.quick else "full"
        return f"{self.gender}_{self.years}y_{self.profile}_{self.market_policy}_{text_tag}_{tabpfn_tag}_{self.text_dim}d_{quick_tag}"
