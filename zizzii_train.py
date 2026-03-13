from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional
import json
import re
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependencies. Install packages from requirements-kaggle.txt before running zizzii_train.py."
    ) from exc

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None

try:
    import optuna
except ModuleNotFoundError:
    optuna = None

from zizzii_features import attach_team_ids_from_names, build_team_features, compute_elo, parse_seeds, standardize_external_team_frame


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ncaa-data"
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"
# Kaggle MSE/Brier scoring does not justify aggressive probability clipping.
# Keep only machine-scale bounds so logit transforms remain finite.
PROB_EPS = 1e-6
CLIP_LO, CLIP_HI = PROB_EPS, 1.0 - PROB_EPS
MIN_TRAIN_SEASONS = 8
CALIBRATION_MIN_ROWS = 300
MODEL_LIMIT = {"M": 4, "W": 3}
LR_C = {"M": 0.9, "W": 0.3}
MARGIN_TO_PROB_SCALE = {"M": 10.0, "W": 9.0}
MARGIN_CAUCHY_SCALE = {"M": 11.0, "W": 9.0}
CAUCHY_HESSIAN_FLOOR = 1e-6
SHRINKAGE_GRID = (0.0, 0.03, 0.06, 0.10)
TOSSUP_GATE_MODES = ("prob_only", "seed_only", "seed_and_prob")
TOSSUP_SEED_THRESHOLDS = (2, 4, 6)
TOSSUP_FAVORITE_THRESHOLDS = (0.58, 0.62, 0.66)
TOSSUP_BLEND_WEIGHTS = (0.15, 0.25, 0.35)
TOSSUP_MIN_ROWS = 180
TOSSUP_MIN_EVAL_GATE_ROWS = 50
MOE_SEED_THRESHOLDS = (3, 4, 6)
MOE_FAVORITE_THRESHOLDS = (0.60, 0.66, 0.72)
ADAPTIVE_MARKET_BASE_WEIGHTS = (0.10, 0.14, 0.18, 0.22, 0.26)
ADAPTIVE_MARKET_CLOSE_THRESHOLDS = (2, 4, 6)
ADAPTIVE_MARKET_WIDE_THRESHOLDS = (6, 8, 10)
ADAPTIVE_MARKET_CLOSE_WEIGHTS = (0.22, 0.30, 0.38, 0.46)
ADAPTIVE_MARKET_WIDE_WEIGHTS = (0.00, 0.04, 0.08, 0.12)
ADAPTIVE_MARKET_MIN_ROWS = 120
CHALK_MIN_ROWS = 260
ODDS_KEYWORDS = ("odds", "market", "vegas", "sportsbook", "spread")
SIGNAL_KEYWORDS = ("manual", "signal")
DEFAULT_EXTERNAL_CONFIG = {
    "market_blend_weight": {"M": 0.18, "W": 0.14},
    "field_market_extra_weight": {"M": 0.08, "W": 0.05},
    "manual_signal_logit_weight": {"M": 0.12, "W": 0.09},
    "external_rating_logit_weight": {"M": 0.08, "W": 0.06},
    "manual_signal_column": "Signal_ManualComposite",
    "min_market_coverage_for_training": 0.20,
    "min_market_coverage_for_residual_models": 0.85,
    "field_live_reweight_strength": {"M": 0.75, "W": 0.45},
    "field_live_reweight_min_features": 4,
    "elo_params": {
        "M": {"k": 20.0, "initial": 1500.0, "carryover": 0.75},
        "W": {"k": 20.0, "initial": 1500.0, "carryover": 0.75},
    },
    "elo_optuna": {
        "enabled": False,
        "eval_years": 7,
        "trials": {"M": 18, "W": 12},
    },
    "feature_ablation_groups": {"M": ["elo_dynamics", "adjusted_efficiency"], "W": []},
    "market_mode": {"M": "default", "W": "default"},
}
MARKET_RESIDUAL_MIN_ROWS = 250
MARKET_RESIDUAL_SELECTION_MIN_COVERAGE = 0.85
MARKET_RESIDUAL_MIN_EVAL_SEASONS = 4
MARKET_RESIDUAL_ACTIVE_START_SEASON = 2015
MARKET_RESIDUAL_OVERLAY_MODELS = ("histgb_market_resid", "xgb_market_resid", "lgbm_market_resid")
MARKET_RESIDUAL_OVERLAY_BLEND_WEIGHTS = (0.35, 0.50, 0.65, 0.80, 1.00)
MARKET_RESIDUAL_OVERLAY_MIN_EVAL_ROWS = 60
WOMEN_LOW_SIGNAL_TEAM_PREFIXES = (
    "FG3Pct",
    "RecentEffFG3Pct",
    "Recent30EffFG3Pct",
)
WOMEN_LOW_SIGNAL_MATCHUP_PREFIXES = (
    "D_FG3Pct",
    "D_RecentEffFG3Pct",
    "D_Recent30EffFG3Pct",
)
WOMEN_LOW_SIGNAL_MATCHUP_NAMES = {"AbsFG3PctDiff"}
WOMEN_LINEAR_BLEND_WEIGHTS = (0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97)
WOMEN_CHALK_TOP_SEED_MAX = (1, 2, 3, 4, 5)
WOMEN_CHALK_DOG_SEED_MIN = (9, 10, 11, 12, 13, 14, 15, 16)
WOMEN_CHALK_FLOOR_PROBS = (0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.985, 0.992, 0.997, 0.999)
WOMEN_CHALK_MAX_ROUNDS = (1, 2, 6)
WOMEN_CHALK_REQUIRE_HOST = (False, True)
WOMEN_DUAL_CHALK_TOP_SEED_MAX = (1, 2, 3)
WOMEN_DUAL_CHALK_DOG_SEED_MIN = (14, 15, 16)
WOMEN_DUAL_CHALK_MAX_ROUNDS = (1, 2)
WOMEN_DUAL_CHALK_FLOOR_PROBS = (0.97, 0.985, 0.992, 0.997, 0.999)
WOMEN_DUAL_CHALK_REQUIRE_HOST = (False, True)
WOMEN_MINIMAL_FEATURE_PRIORITY = [
    "D_SeedNum",
    "AbsSeedDiff",
    "T1BetterSeed",
    "T1_SeedNum",
    "T2_SeedNum",
    "D_Elo",
    "D_WinRate",
    "D_AvgMargin",
    "D_SOS",
    "D_NetRtg_z",
    "D_DefRtg_z",
    "D_DR",
    "D_RecentEffDR",
    "D_Recent30EffDR",
    "D_Recent30EffDefRtg_z",
    "D_Recent30EffNetRtg_z",
    "D_Last10WinRate",
    "D_Last30WinRate",
    "D_Top50Wins",
    "D_ConfNonConfWinRate",
    "D_ConfMeanElo",
    "D_ExtCompositeStrength",
]
WOMEN_ROLLBACK_FEATURE_PREFIXES = (
    "AdjOffRtg",
    "AdjDefRtg",
    "AdjNetRtg",
    "OppEFG",
    "OppTOVPct",
    "OppFTR",
    "OppORBPct",
    "NeutralAdjNetRtg",
    "RecentEffAdjOffRtg",
    "RecentEffAdjDefRtg",
    "RecentEffAdjNetRtg",
    "RecentEffOppEFG",
    "RecentEffOppTOVPct",
    "RecentEffOppFTR",
    "RecentEffOppORBPct",
    "Recent30EffAdjOffRtg",
    "Recent30EffAdjDefRtg",
    "Recent30EffAdjNetRtg",
    "Recent30EffOppEFG",
    "Recent30EffOppTOVPct",
    "Recent30EffOppFTR",
    "Recent30EffOppORBPct",
    "MomentumAdjNetRtg",
    "Momentum30AdjNetRtg",
)
WOMEN_ROLLBACK_FEATURE_NAMES = {"EloPeak", "EloPeakDrop", "EloTrend"}
FEATURE_ABLATION_GROUPS = {
    "M": {
        "adjusted_efficiency": {
            "prefixes": (
                "AdjOffRtg",
                "AdjDefRtg",
                "AdjNetRtg",
                "NeutralAdjNetRtg",
                "RecentEffAdjOffRtg",
                "RecentEffAdjDefRtg",
                "RecentEffAdjNetRtg",
                "Recent30EffAdjOffRtg",
                "Recent30EffAdjDefRtg",
                "Recent30EffAdjNetRtg",
                "MomentumAdjNetRtg",
                "Momentum30AdjNetRtg",
            ),
            "names": set(),
        },
        "elo_dynamics": {
            "prefixes": (),
            "names": {"EloPeak", "EloPeakDrop", "EloTrend"},
        },
        "opp_four_factors": {
            "prefixes": (
                "OppEFG",
                "OppTOVPct",
                "OppFTR",
                "OppORBPct",
                "RecentEffOppEFG",
                "RecentEffOppTOVPct",
                "RecentEffOppFTR",
                "RecentEffOppORBPct",
                "Recent30EffOppEFG",
                "Recent30EffOppTOVPct",
                "Recent30EffOppFTR",
                "Recent30EffOppORBPct",
            ),
            "names": set(),
        },
    },
    "W": {
        "experimental": {
            "prefixes": WOMEN_ROLLBACK_FEATURE_PREFIXES,
            "names": WOMEN_ROLLBACK_FEATURE_NAMES,
        }
    },
}
FIELD_REWEIGHT_FEATURES = [
    "AbsSeedDiff",
    "D_Elo",
    "D_EloTrend",
    "D_EloPeakDrop",
    "D_SOS",
    "D_OWP",
    "D_RPIStyle",
    "D_PageRank",
    "D_PageRankSOS",
    "D_SeedPriorExpectedWins",
    "D_PathDifficultyEarly",
    "D_PathDifficultyLate",
    "D_NetRtg_z",
    "D_AdjNetRtg_z",
    "D_Last10WinRate",
    "D_Last30WinRate",
    "D_Recent30EffNetRtg_z",
    "D_Recent30EffAdjNetRtg_z",
    "AbsTempoDiff",
    "AbsRecentNetRtgDiff",
    "StyleMismatchScore",
    "D_ExtCompositeStrength",
]

LR_CORE_FEATURES = {
    "M": [
        "Elo", "MasseyMean", "MasseyMin", "SeedNum", "WinRate", "AvgMargin", "SOS", "OWP", "RPIStyle",
        "PageRank", "PageRankSOS", "Last30WinRate", "Recent30EffNetRtg_z",
        "AdjNetRtg_z", "EloTrend", "EloPeakDrop",
        "SeedPriorExpectedWins", "PathDifficultyEarly", "PathDifficultyLate",
        "NetRtg_z", "ConfNonConfWinRate", "ConfMeanElo", "CoachTenure", "CoachMidseasonChange", "ExtCompositeStrength",
    ],
    "W": [
        "Elo", "SeedNum", "WinRate", "AvgMargin", "SOS", "NetRtg_z", "AdjNetRtg_z", "EloTrend", "EloPeakDrop",
        "ConfNonConfWinRate", "ConfMeanElo", "ExtCompositeStrength",
    ],
}

LR_PLUS_FEATURES = {
    "M": [
        "Elo", "MasseyMean", "MasseyMin", "MasseyMomentum", "SeedNum", "WinRate", "AvgMargin", "SOS", "OWP", "OOWP",
        "RPIStyle", "RPIStyleSOS", "PageRank", "PageRankSOS", "SeedPriorExpectedWins", "SeedPriorWinPct", "PathDifficultyEarly", "PathDifficultyMiddle",
        "PathDifficultyLate",
        "EloPeak", "EloTrend", "EloPeakDrop",
        "NetRtg_z", "Last10WinRate", "Last30WinRate", "Top50Wins", "CloseGameWinRate", "MomentumWinRate", "Momentum30WinRate",
        "MomentumNetRtg", "AdjNetRtg_z", "Recent30EffNetRtg_z", "Recent30EffAdjNetRtg_z", "Recent30EffDefRtg_z",
        "OppEFG", "OppFTR", "OppTOVPct", "OppORBPct", "Recent30EffORBPct", "Recent30EffTOVPct",
        "ConfNonConfWinRate", "ConfMeanElo", "CoachTenure", "CoachMidseasonChange", "ExtCompositeStrength",
    ],
    "W": [
        "Elo", "SeedNum", "WinRate", "AvgMargin", "SOS", "NetRtg_z", "AdjNetRtg_z", "EloTrend", "EloPeakDrop",
        "Last10WinRate", "Top50Wins", "CloseGameWinRate", "MomentumWinRate", "MomentumNetRtg",
        "OppEFG", "OppFTR", "ConfNonConfWinRate", "ConfMeanElo", "ExtCompositeStrength",
    ],
}
TOSSUP_SPECIALIST_FEATURES = [
    "BaseProb", "BaseFavoriteProb", "BaseUncertainty",
    "D_SeedNum", "AbsSeedDiff", "T1BetterSeed",
    "D_Elo", "D_EloTrend", "D_EloPeakDrop", "D_MasseyMean", "D_MasseyMomentum", "D_WinRate", "D_AvgMargin",
    "D_SOS", "D_OWP", "D_RPIStyle", "D_PageRank", "D_PageRankSOS", "D_SeedPriorExpectedWins",
    "D_PathDifficultyEarly", "D_PathDifficultyLate", "D_NetRtg_z", "D_AdjNetRtg_z", "D_Last10WinRate",
    "D_Last30WinRate", "D_Top50Wins", "D_CloseGameWinRate", "D_Last30CloseWinRate", "D_MomentumWinRate", "D_Momentum30WinRate",
    "D_MomentumNetRtg", "D_Momentum30NetRtg",
    "D_DefRtg", "D_DefRtg_z", "D_Recent30EffDefRtg", "D_Recent30EffDefRtg_z",
    "D_eFG", "D_FTR", "D_TOVPct", "D_ORBPct", "D_OppEFG", "D_OppFTR", "D_OppTOVPct", "D_OppORBPct", "D_Tempo", "D_FG3Pct",
    "D_RecentEffNetRtg_z", "D_RecentEffAdjNetRtg_z", "D_RecentEffeFG", "D_RecentEffFTR", "D_RecentEffTOVPct", "D_RecentEffORBPct", "D_RecentEffTempo",
    "D_Recent30EffNetRtg_z", "D_Recent30EffAdjNetRtg_z", "D_Recent30EffeFG", "D_Recent30EffFTR", "D_Recent30EffTOVPct", "D_Recent30EffORBPct", "D_Recent30EffTempo",
    "AbsTempoDiff", "AbsEFGDiff", "AbsFTRDiff", "AbsTOVPctDiff", "AbsORBPctDiff", "AbsFG3PctDiff", "StyleMismatchScore",
    "D_ExtCompositeStrength",
]
CHALK_SPECIALIST_FEATURES = [
    "BaseProb", "BaseFavoriteProb", "BaseUncertainty",
    "D_SeedNum", "AbsSeedDiff", "T1BetterSeed",
    "D_Elo", "D_EloTrend", "D_WinRate", "D_AvgMargin", "D_SOS", "D_PageRank", "D_PageRankSOS",
    "D_SeedPriorExpectedWins", "D_PathDifficultyEarly", "D_PathDifficultyLate",
    "D_OffRtg", "D_OffRtg_z", "D_NetRtg_z", "D_AdjNetRtg_z", "D_Last10WinRate", "D_Last30WinRate",
    "D_Top50Wins", "D_MomentumWinRate", "D_Momentum30WinRate",
    "D_eFG", "D_FTR", "D_OppEFG", "D_OppFTR", "D_FG3Pct", "D_Tempo",
    "D_RecentEffNetRtg_z", "D_Recent30EffNetRtg_z", "D_Recent30EffAdjNetRtg_z",
    "D_ExtCompositeStrength",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_key: str


def deep_update(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_external_config(external_dir: Optional[Path] = None) -> dict:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    config = json.loads(json.dumps(DEFAULT_EXTERNAL_CONFIG))
    if not root.exists():
        return config

    config_path = root / "external_config.json"
    if not config_path.exists():
        return config

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return config
    return deep_update(config, loaded)


def rollback_women_experimental_features(team_feats: pd.DataFrame) -> pd.DataFrame:
    if team_feats.empty:
        return team_feats

    rollback_cols = [
        column
        for column in team_feats.columns
        if column in WOMEN_ROLLBACK_FEATURE_NAMES or column.startswith(WOMEN_ROLLBACK_FEATURE_PREFIXES)
    ]
    if not rollback_cols:
        return team_feats
    return team_feats.drop(columns=sorted(set(rollback_cols)))


def resolve_feature_ablation_groups(external_config: dict, gender: str) -> list[str]:
    groups = external_config.get("feature_ablation_groups", {})
    if isinstance(groups, dict):
        selected = groups.get(gender, [])
    else:
        selected = groups
    if selected is None:
        return []
    if isinstance(selected, str):
        return [selected]
    return [str(value) for value in selected]


def apply_feature_ablations(team_feats: pd.DataFrame, gender: str, group_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if team_feats.empty or not group_names:
        return team_feats, []

    group_defs = FEATURE_ABLATION_GROUPS.get(gender, {})
    applied: list[str] = []
    drop_cols: set[str] = set()
    for group_name in group_names:
        spec = group_defs.get(group_name)
        if spec is None:
            continue
        prefixes = tuple(spec.get("prefixes", ()))
        names = set(spec.get("names", set()))
        for column in team_feats.columns:
            if column in names or column.startswith(prefixes):
                drop_cols.add(column)
        applied.append(group_name)

    if not drop_cols:
        return team_feats, []
    return team_feats.drop(columns=sorted(drop_cols)), applied


def simplify_women_feature_lists(
    feature_candidates: list[str],
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    def keep_team(column: str) -> bool:
        return not column.startswith(WOMEN_LOW_SIGNAL_TEAM_PREFIXES)

    def keep_matchup(column: str) -> bool:
        if column in WOMEN_LOW_SIGNAL_MATCHUP_NAMES:
            return False
        return not column.startswith(WOMEN_LOW_SIGNAL_MATCHUP_PREFIXES)

    feature_candidates = [column for column in feature_candidates if keep_team(column)]
    lr_core_feats = [column for column in lr_core_feats if keep_matchup(column)]
    lr_plus_feats = [column for column in lr_plus_feats if keep_matchup(column)]
    all_feats = [column for column in all_feats if keep_matchup(column)]
    return feature_candidates, lr_core_feats, lr_plus_feats, all_feats


def build_women_minimal_feature_list(df: pd.DataFrame) -> list[str]:
    return [column for column in WOMEN_MINIMAL_FEATURE_PRIORITY if column in df.columns]


def resolve_gender_override_list(external_config: dict, key: str, gender: str) -> list[str]:
    value = external_config.get(key)
    if isinstance(value, dict):
        value = value.get(gender)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def resolve_gender_override_dict(external_config: dict, key: str, gender: str) -> dict[str, object]:
    value = external_config.get(key)
    if isinstance(value, dict) and gender in value and isinstance(value.get(gender), dict):
        return dict(value[gender])
    if isinstance(value, dict) and gender not in value:
        return dict(value)
    return {}


def resolve_women_feature_simplify_enabled(external_config: dict) -> bool:
    return bool(external_config.get("women_feature_simplify_enabled", True))


def resolve_women_chalk_extremes_enabled(external_config: dict) -> bool:
    return bool(external_config.get("women_chalk_extremes_enabled", True))


def resolve_market_mode(external_config: dict, gender: str) -> str:
    market_mode = external_config.get("market_mode", {})
    if isinstance(market_mode, dict):
        return str(market_mode.get(gender, "default"))
    return str(market_mode)


def resolve_elo_params(external_config: dict, gender: str) -> dict[str, float]:
    defaults = {"k": 20.0, "initial": 1500.0, "carryover": 0.75}
    configured = external_config.get("elo_params", {}).get(gender, {})
    for key in defaults:
        if key in configured:
            defaults[key] = float(configured[key])
    return defaults


def build_elo_proxy_matchups(
    gender: str,
    compact_df: pd.DataFrame,
    seeds_raw: pd.DataFrame,
    tourney_results: pd.DataFrame,
    elo_params: dict[str, float],
) -> tuple[pd.DataFrame, list[str]]:
    team_elo = compute_elo(compact_df, **elo_params)
    seeds = parse_seeds(seeds_raw)
    proxy_team_feats = team_elo.merge(seeds[["Season", "TeamID", "SeedNum"]], on=["Season", "TeamID"], how="left")
    matchups, _, _, _, _ = build_matchup_df(tourney_results, proxy_team_feats, gender)
    proxy_features = [
        column
        for column in ["D_Elo", "D_EloPeak", "D_EloPeakDrop", "D_EloTrend", "D_SeedNum", "AbsSeedDiff"]
        if column in matchups.columns
    ]
    return matchups, proxy_features


def evaluate_elo_proxy_brier(matchups: pd.DataFrame, feature_columns: list[str], gender: str, eval_years: int) -> float:
    if matchups.empty or not feature_columns:
        return float("inf")

    seasons = sorted(matchups["Season"].unique())
    eval_seasons = seasons[-eval_years:]
    scores = []
    for season in eval_seasons:
        train_df = matchups[matchups["Season"] < season]
        test_df = matchups[matchups["Season"] == season]
        if train_df.empty or test_df.empty or len(train_df["Season"].unique()) < MIN_TRAIN_SEASONS:
            continue
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=LR_C[gender], max_iter=2000, random_state=42)),
        ])
        model.fit(train_df[feature_columns].fillna(0.0), train_df["Label"])
        pred = safe_clip(model.predict_proba(test_df[feature_columns].fillna(0.0))[:, 1])
        scores.append(brier_score_loss(test_df["Label"], pred))
    return float(np.mean(scores)) if scores else float("inf")


def optimize_elo_params(
    gender: str,
    compact_df: pd.DataFrame,
    seeds_raw: pd.DataFrame,
    tourney_results: pd.DataFrame,
    external_config: dict,
    run_id: str,
) -> tuple[dict[str, float], Optional[dict[str, object]]]:
    base_params = resolve_elo_params(external_config, gender)
    optuna_config = external_config.get("elo_optuna", {})
    if not bool(optuna_config.get("enabled", False)):
        return base_params, None
    if optuna is None:
        print(f"Optuna unavailable for {gender}; using configured Elo params.")
        return base_params, None

    trial_config = optuna_config.get("trials", {})
    if isinstance(trial_config, dict):
        trials = int(trial_config.get(gender, 12))
    else:
        trials = int(trial_config)
    eval_years = int(optuna_config.get("eval_years", 7))
    proxy_features: list[str] = []

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"elo_proxy_{gender}_{run_id}",
    )
    study.enqueue_trial({"k": base_params["k"], "carryover": base_params["carryover"]})

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal proxy_features
        params = {
            "k": float(trial.suggest_float("k", 10.0, 36.0)),
            "initial": base_params["initial"],
            "carryover": float(trial.suggest_float("carryover", 0.55, 0.92)),
        }
        matchups, proxy_features = build_elo_proxy_matchups(gender, compact_df, seeds_raw, tourney_results, params)
        return evaluate_elo_proxy_brier(matchups, proxy_features, gender, eval_years)

    study.optimize(objective, n_trials=max(trials, 1), show_progress_bar=False)
    best_params = {
        "k": float(study.best_params["k"]),
        "initial": base_params["initial"],
        "carryover": float(study.best_params["carryover"]),
    }
    _, proxy_features = build_elo_proxy_matchups(gender, compact_df, seeds_raw, tourney_results, best_params)
    summary = {
        "enabled": True,
        "eval_years": eval_years,
        "trials": len(study.trials),
        "best_params": best_params,
        "best_cv_brier": float(study.best_value),
        "proxy_features": proxy_features,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    artifact_path = RESULTS_DIR / f"elo_optuna_{gender}_{run_id}.json"
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["artifact_path"] = str(artifact_path)
    return best_params, summary


def find_external_csvs(gender: str, include_keywords: tuple[str, ...], external_dir: Optional[Path] = None) -> list[Path]:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    if not root.exists():
        return []

    paths = []
    for path in sorted(root.glob("*.csv")):
        lowered = path.name.lower()
        if not lowered.startswith(gender.lower()):
            continue
        if "template" in lowered or "example" in lowered:
            continue
        if any(keyword in lowered for keyword in include_keywords):
            paths.append(path)
    return paths


def safe_clip(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), CLIP_LO, CLIP_HI)


def safe_logit(prob: np.ndarray) -> np.ndarray:
    prob = safe_clip(prob)
    return np.log(prob / (1.0 - prob))


def apply_logit_shift(prob: np.ndarray, shift: np.ndarray | float) -> np.ndarray:
    return safe_clip(1.0 / (1.0 + np.exp(-(safe_logit(prob) + shift))))


def shrink_toward_half(prob: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if alpha <= 0:
        return safe_clip(prob)
    return safe_clip(0.5 + (safe_clip(prob) - 0.5) * (1.0 - alpha))


def american_to_prob(odds: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(odds, errors="coerce")
    negative = numeric < 0
    prob = pd.Series(np.nan, index=numeric.index, dtype=float)
    prob.loc[negative] = -numeric.loc[negative] / (-numeric.loc[negative] + 100.0)
    prob.loc[~negative] = 100.0 / (numeric.loc[~negative] + 100.0)
    return prob


def standardize_matchup_market_frame(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    if "Season" not in df.columns:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb"])

    frame = df.copy()
    if {"T1", "T2"}.issubset(frame.columns):
        pass
    elif {"Team1ID", "Team2ID"}.issubset(frame.columns):
        frame = frame.rename(columns={"Team1ID": "T1", "Team2ID": "T2"})
    else:
        left_name = next((col for col in ["Team1Name", "Team1", "AwayTeam", "LowTeamName"] if col in frame.columns), None)
        right_name = next((col for col in ["Team2Name", "Team2", "HomeTeam", "HighTeamName"] if col in frame.columns), None)
        if left_name is None or right_name is None:
            return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb"])
        frame = attach_team_ids_from_names(frame, gender, data_dir=DATA_DIR, team_col=left_name, target_col="T1")
        frame = attach_team_ids_from_names(frame, gender, data_dir=DATA_DIR, team_col=right_name, target_col="T2")

    frame["T1"] = pd.to_numeric(frame["T1"], errors="coerce")
    frame["T2"] = pd.to_numeric(frame["T2"], errors="coerce")
    frame = frame.dropna(subset=["Season", "T1", "T2"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["T1"] = frame["T1"].astype(int)
    frame["T2"] = frame["T2"].astype(int)

    prob = None
    for column in ["MarketProb", "T1WinProb", "Team1WinProb", "ImpliedProb", "NoVigProb", "Prob"]:
        if column in frame.columns:
            prob = pd.to_numeric(frame[column], errors="coerce")
            break

    if prob is None:
        left_ml = next((col for col in ["Team1Moneyline", "Moneyline1", "AwayMoneyline"] if col in frame.columns), None)
        right_ml = next((col for col in ["Team2Moneyline", "Moneyline2", "HomeMoneyline"] if col in frame.columns), None)
        if left_ml is not None and right_ml is not None:
            p1 = american_to_prob(frame[left_ml])
            p2 = american_to_prob(frame[right_ml])
            denom = (p1 + p2).replace(0, np.nan)
            prob = p1 / denom

    if prob is None:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb"])

    market = frame[["Season", "T1", "T2"]].copy()
    market["MarketProb"] = safe_clip(prob.fillna(0.5).to_numpy())

    swap_mask = market["T1"] > market["T2"]
    if swap_mask.any():
        swapped_pairs = market.loc[swap_mask, ["T1", "T2"]].copy()
        market.loc[swap_mask, "T1"] = swapped_pairs["T2"].to_numpy()
        market.loc[swap_mask, "T2"] = swapped_pairs["T1"].to_numpy()
        market.loc[swap_mask, "MarketProb"] = 1.0 - market.loc[swap_mask, "MarketProb"]

    market = market.groupby(["Season", "T1", "T2"], as_index=False)["MarketProb"].mean()
    market["MarketLogit"] = safe_logit(market["MarketProb"].to_numpy())
    market["MarketConfidence"] = np.abs(market["MarketProb"] - 0.5)
    return market


def load_matchup_market_odds(gender: str, external_dir: Optional[Path] = None) -> pd.DataFrame:
    frames = []
    for path in find_external_csvs(gender, ODDS_KEYWORDS, external_dir=external_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        current = standardize_matchup_market_frame(df, gender)
        if not current.empty:
            frames.append(current)

    if not frames:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence"])

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["Season", "T1", "T2"], as_index=False).agg(
        MarketProb=("MarketProb", "mean"),
        MarketLogit=("MarketLogit", "mean"),
        MarketConfidence=("MarketConfidence", "mean"),
    )


def market_coverage_by_season(df: pd.DataFrame) -> dict[int, float]:
    if df.empty or "MarketProb" not in df.columns or "Season" not in df.columns:
        return {}
    coverage = (
        df.groupby("Season", as_index=True)["MarketProb"]
        .apply(lambda values: float(values.notna().mean()))
        .to_dict()
    )
    return {int(season): float(value) for season, value in coverage.items()}


def market_residual_active_seasons(
    coverage_by_season: dict[int, float],
    min_coverage: float,
    start_season: int = MARKET_RESIDUAL_ACTIVE_START_SEASON,
) -> list[int]:
    return sorted(
        int(season)
        for season, coverage in coverage_by_season.items()
        if int(season) >= int(start_season) and float(coverage) >= float(min_coverage)
    )


def market_residual_active_mask(
    df: pd.DataFrame,
    active_seasons: list[int] | set[int],
    require_market_prob: bool = True,
) -> np.ndarray:
    if df.empty or "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    active_set = {int(season) for season in active_seasons}
    season_values = pd.to_numeric(df["Season"], errors="coerce").fillna(-1).astype(int)
    mask = np.asarray(season_values.isin(active_set).to_numpy(), dtype=bool).copy()
    if require_market_prob and "MarketProb" in df.columns:
        mask &= pd.to_numeric(df["MarketProb"], errors="coerce").notna().to_numpy()
    return mask


def market_residual_train_mask(
    df: pd.DataFrame,
    start_season: int = MARKET_RESIDUAL_ACTIVE_START_SEASON,
) -> np.ndarray:
    if df.empty or "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return pd.to_numeric(df["Season"], errors="coerce").fillna(-1).astype(int).ge(int(start_season)).to_numpy()


def subset_sample_weight(
    sample_weight: Optional[pd.Series | np.ndarray],
    base_index: pd.Index,
    subset_index: pd.Index,
) -> Optional[pd.Series | np.ndarray]:
    if sample_weight is None:
        return None
    if isinstance(sample_weight, pd.Series):
        return sample_weight.loc[subset_index]
    array = np.asarray(sample_weight, dtype=float)
    if len(array) != len(base_index):
        return array
    positions = pd.Index(base_index).get_indexer(subset_index)
    if (positions < 0).any():
        return array
    return array[positions]


def recent_market_residual_ready(
    matchups: pd.DataFrame,
    eval_years: int,
    min_coverage: float,
    min_eval_seasons: int = MARKET_RESIDUAL_MIN_EVAL_SEASONS,
) -> tuple[bool, dict[str, object]]:
    if matchups.empty or "MarketProb" not in matchups.columns:
        return False, {
            "coverage_by_season": {},
            "active_seasons": [],
            "eligible_eval_seasons": [],
            "matched_rows_in_eval_window": 0,
            "eval_window_seasons": [],
        }

    coverage_by_season = market_coverage_by_season(matchups)
    active_seasons = market_residual_active_seasons(coverage_by_season, min_coverage)
    eval_seasons = sorted(int(season) for season in matchups["Season"].unique())[-eval_years:]
    eligible_eval_seasons = [season for season in eval_seasons if season in active_seasons]
    matched_rows_in_eval_window = int(
        matchups.loc[matchups["Season"].isin(eligible_eval_seasons), "MarketProb"].notna().sum()
    )
    ready = len(eligible_eval_seasons) >= min(min_eval_seasons, len(eval_seasons)) and matched_rows_in_eval_window >= MARKET_RESIDUAL_MIN_ROWS
    return ready, {
        "coverage_by_season": coverage_by_season,
        "active_seasons": active_seasons,
        "eligible_eval_seasons": eligible_eval_seasons,
        "matched_rows_in_eval_window": matched_rows_in_eval_window,
        "eval_window_seasons": eval_seasons,
    }


def load_manual_signals(gender: str, external_dir: Optional[Path] = None) -> pd.DataFrame:
    frames = []
    for path in find_external_csvs(gender, SIGNAL_KEYWORDS, external_dir=external_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        current = standardize_external_team_frame(df, gender, data_dir=DATA_DIR)
        if current.empty:
            continue

        numeric_cols = [
            column for column in current.columns
            if column not in {"Season", "TeamID"} and pd.api.types.is_numeric_dtype(current[column])
        ]
        if not numeric_cols:
            continue

        rename_map = {}
        for column in numeric_cols:
            rename_map[column] = column if column.startswith("Signal_") else f"Signal_{column}"
        current = current[["Season", "TeamID"] + numeric_cols].rename(columns=rename_map)

        signal_cols = [column for column in current.columns if column.startswith("Signal_")]
        preferred = next((column for column in ["Signal_ManualComposite", "Signal_ManualNetSignal"] if column in signal_cols), None)
        if preferred is not None:
            current["Signal_ManualComposite"] = current[preferred]
        else:
            current["Signal_ManualComposite"] = current[signal_cols].sum(axis=1)
        frames.append(current)

    if not frames:
        return pd.DataFrame(columns=["Season", "TeamID", "Signal_ManualComposite"])

    combined = pd.concat(frames, ignore_index=True)
    signal_cols = [column for column in combined.columns if column.startswith("Signal_")]
    aggregated = combined.groupby(["Season", "TeamID"], as_index=False)[signal_cols].sum(min_count=1)
    if "Signal_ManualComposite" not in aggregated.columns and signal_cols:
        aggregated["Signal_ManualComposite"] = aggregated[signal_cols].sum(axis=1)
    aggregated["Signal_ManualComposite"] = aggregated["Signal_ManualComposite"].fillna(0.0)
    return aggregated


def infer_feature_candidates(team_feats: pd.DataFrame) -> list[str]:
    candidates = []
    for column in team_feats.columns:
        if column in {"Season", "TeamID"}:
            continue
        if pd.api.types.is_numeric_dtype(team_feats[column]):
            candidates.append(column)
    return sorted(candidates)


def compute_diff_features(df: pd.DataFrame, feature_list: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available = []
    for feat in feature_list:
        left = f"T1_{feat}"
        right = f"T2_{feat}"
        if left in df.columns and right in df.columns:
            name = f"D_{feat}"
            df[name] = df[left] - df[right]
            available.append(name)
    return df, available


def add_matchup_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    extras = []
    if "D_SeedNum" in df.columns:
        df["AbsSeedDiff"] = df["D_SeedNum"].abs()
        extras.append("AbsSeedDiff")
    if "T1_SeedNum" in df.columns and "T2_SeedNum" in df.columns:
        df["T1BetterSeed"] = (df["T1_SeedNum"] < df["T2_SeedNum"]).astype(int)
        extras.append("T1BetterSeed")
    if "T1_ConfAbbrev" in df.columns and "T2_ConfAbbrev" in df.columns:
        df["SameConference"] = (df["T1_ConfAbbrev"] == df["T2_ConfAbbrev"]).astype(int)
        extras.append("SameConference")

    abs_diff_map = {
        "D_Tempo": "AbsTempoDiff",
        "D_eFG": "AbsEFGDiff",
        "D_FTR": "AbsFTRDiff",
        "D_TOVPct": "AbsTOVPctDiff",
        "D_ORBPct": "AbsORBPctDiff",
        "D_FG3Pct": "AbsFG3PctDiff",
        "D_RecentEffTempo": "AbsRecentTempoDiff",
        "D_RecentEffNetRtg_z": "AbsRecentNetRtgDiff",
    }
    for diff_col, abs_col in abs_diff_map.items():
        if diff_col in df.columns:
            df[abs_col] = df[diff_col].abs()
            extras.append(abs_col)

    style_cols = [column for column in ["AbsTempoDiff", "AbsEFGDiff", "AbsFTRDiff", "AbsTOVPctDiff", "AbsORBPctDiff", "AbsFG3PctDiff"] if column in df.columns]
    if style_cols:
        df["StyleMismatchScore"] = df[style_cols].sum(axis=1)
        extras.append("StyleMismatchScore")
    return df, extras


def merge_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], float]:
    if market_df.empty:
        return df, [], 0.0

    merged = df.merge(market_df, on=["Season", "T1", "T2"], how="left")
    coverage = float(merged["MarketProb"].notna().mean()) if "MarketProb" in merged.columns else 0.0
    features = []
    for column in ["MarketProb", "MarketLogit", "MarketConfidence"]:
        if column in merged.columns:
            features.append(column)
    return merged, features, coverage


def merge_team_signal_features(df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return df

    t1 = signal_df.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2 = signal_df.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    merged = df.merge(t1, on=["Season", "T1"], how="left")
    merged = merged.merge(t2, on=["Season", "T2"], how="left")

    signal_columns = [column for column in signal_df.columns if column.startswith("Signal_")]
    for column in signal_columns:
        left = f"T1_{column}"
        right = f"T2_{column}"
        if left in merged.columns and right in merged.columns:
            merged[f"D_{column}"] = merged[left].fillna(0.0) - merged[right].fillna(0.0)
    return merged


@lru_cache(maxsize=4)
def load_women_tourney_round_map() -> pd.DataFrame:
    seeds_path = DATA_DIR / "WNCAATourneySeeds.csv"
    slots_path = DATA_DIR / "WNCAATourneySlots.csv"
    if not seeds_path.exists() or not slots_path.exists():
        return pd.DataFrame(columns=["Season", "T1", "T2", "TourneyRound"])

    try:
        seeds_raw = pd.read_csv(seeds_path)
        slots = pd.read_csv(slots_path)
    except Exception:
        return pd.DataFrame(columns=["Season", "T1", "T2", "TourneyRound"])

    if seeds_raw.empty or slots.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "TourneyRound"])

    rows: list[pd.DataFrame] = []
    valid_seasons = sorted(set(pd.to_numeric(seeds_raw["Season"], errors="coerce").dropna().astype(int)).intersection(
        set(pd.to_numeric(slots["Season"], errors="coerce").dropna().astype(int))
    ))
    for season in valid_seasons:
        season_seeds = seeds_raw[pd.to_numeric(seeds_raw["Season"], errors="coerce") == season][["Seed", "TeamID"]].dropna().copy()
        season_slots = slots[pd.to_numeric(slots["Season"], errors="coerce") == season].copy()
        if season_seeds.empty or season_slots.empty:
            continue

        seed_to_team = {
            str(seed): int(team_id)
            for seed, team_id in season_seeds[["Seed", "TeamID"]].itertuples(index=False, name=None)
        }
        children = season_slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict("index")
        memo: dict[str, set[int]] = {}

        def descendants(node: object) -> set[int]:
            key = str(node)
            if key in memo:
                return memo[key]
            if key in children:
                row = children[key]
                memo[key] = descendants(row["StrongSeed"]) | descendants(row["WeakSeed"])
                return memo[key]
            team_id = seed_to_team.get(key)
            memo[key] = {team_id} if team_id is not None else set()
            return memo[key]

        pair_rounds: dict[tuple[int, int], int] = {}
        for row in season_slots.itertuples(index=False):
            match = re.match(r"R(\d+)", str(row.Slot))
            if match is None:
                continue
            round_num = int(match.group(1))
            strong = descendants(row.StrongSeed)
            weak = descendants(row.WeakSeed)
            for left in strong:
                for right in weak:
                    if left == right:
                        continue
                    t1, t2 = sorted((int(left), int(right)))
                    key = (t1, t2)
                    if key not in pair_rounds or round_num < pair_rounds[key]:
                        pair_rounds[key] = round_num

        if pair_rounds:
            season_rows = pd.DataFrame(
                [(season, t1, t2, round_num) for (t1, t2), round_num in pair_rounds.items()],
                columns=["Season", "T1", "T2", "TourneyRound"],
            )
            rows.append(season_rows)

    if not rows:
        return pd.DataFrame(columns=["Season", "T1", "T2", "TourneyRound"])
    return pd.concat(rows, ignore_index=True)


def add_women_tourney_structure_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    structure = load_women_tourney_round_map()
    if structure.empty:
        return df, []

    merged = df.merge(structure, on=["Season", "T1", "T2"], how="left")
    extras: list[str] = []

    round_num = pd.to_numeric(merged.get("TourneyRound"), errors="coerce")
    if round_num.notna().any():
        merged["IsRound1"] = (round_num == 1).astype(int)
        merged["IsRound2"] = (round_num == 2).astype(int)
        merged["IsEarlyRound"] = round_num.isin([1, 2]).astype(int)
        extras.extend(["TourneyRound", "IsRound1", "IsRound2", "IsEarlyRound"])

    required = {"T1_SeedNum", "T2_SeedNum", "TourneyRound"}
    if required.issubset(merged.columns):
        t1_seed = pd.to_numeric(merged["T1_SeedNum"], errors="coerce")
        t2_seed = pd.to_numeric(merged["T2_SeedNum"], errors="coerce")
        early_round = round_num.isin([1, 2])
        t1_host = early_round & t1_seed.le(4) & t1_seed.lt(t2_seed)
        t2_host = early_round & t2_seed.le(4) & t2_seed.lt(t1_seed)
        merged["T1_WHostLikely"] = t1_host.astype(int)
        merged["T2_WHostLikely"] = t2_host.astype(int)
        merged["D_WHostLikely"] = merged["T1_WHostLikely"] - merged["T2_WHostLikely"]
        merged["AnyWHostLikely"] = (t1_host | t2_host).astype(int)
        extras.extend(["T1_WHostLikely", "T2_WHostLikely", "D_WHostLikely", "AnyWHostLikely"])

    return merged, extras


def load_official_field_seeds(gender: str) -> pd.DataFrame:
    path = DATA_DIR / f"{gender}NCAATourneySeeds.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "SeedNum"])
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Season", "TeamID", "SeedNum"])
    if raw.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "SeedNum"])
    seeds = parse_seeds(raw).dropna(subset=["Season", "TeamID", "SeedNum"]).copy()
    seeds["Season"] = pd.to_numeric(seeds["Season"], errors="coerce").astype(int)
    seeds["TeamID"] = pd.to_numeric(seeds["TeamID"], errors="coerce").astype(int)
    seeds["SeedNum"] = pd.to_numeric(seeds["SeedNum"], errors="coerce")
    return seeds[["Season", "TeamID", "SeedNum"]].drop_duplicates()


def build_field_pair_matchups(
    team_feats: pd.DataFrame,
    feature_candidates: list[str],
    official_seeds: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    season_seeds = official_seeds[official_seeds["Season"] == season].drop_duplicates(subset=["TeamID"])
    if len(season_seeds) < 4:
        return pd.DataFrame()

    pairs = list(combinations(sorted(season_seeds["TeamID"].astype(int).tolist()), 2))
    if not pairs:
        return pd.DataFrame()

    frame = pd.DataFrame(pairs, columns=["T1", "T2"])
    frame["Season"] = int(season)
    season_feats = team_feats[team_feats["Season"] == season].copy()
    if season_feats.empty:
        return pd.DataFrame()

    t1f = season_feats.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2f = season_feats.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    frame = frame.merge(t1f, on=["Season", "T1"], how="left")
    frame = frame.merge(t2f, on=["Season", "T2"], how="left")
    frame, _ = compute_diff_features(frame, feature_candidates)
    frame, _ = add_matchup_context_features(frame)
    if "D_Elo" in frame.columns:
        frame = frame.dropna(subset=["D_Elo"]).reset_index(drop=True)
    return frame


def field_reweight_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if column.startswith("D_"):
        values = values.abs()
    return values


def build_field_reweight_context(gender: str, team_feats: pd.DataFrame, feature_candidates: list[str]) -> dict[str, object]:
    official_seeds = load_official_field_seeds(gender)
    profiles: dict[int, dict[str, object]] = {}
    field_team_ids: dict[int, set[int]] = {}

    if official_seeds.empty:
        return {"official_seeds": official_seeds, "profiles": profiles, "field_team_ids": field_team_ids}

    for season in sorted(official_seeds["Season"].unique()):
        season_ids = set(official_seeds.loc[official_seeds["Season"] == season, "TeamID"].astype(int).tolist())
        if season_ids:
            field_team_ids[int(season)] = season_ids

        field_pairs = build_field_pair_matchups(team_feats, feature_candidates, official_seeds, int(season))
        if field_pairs.empty:
            continue

        stats = {}
        for column in FIELD_REWEIGHT_FEATURES:
            if column not in field_pairs.columns:
                continue
            values = field_reweight_series(field_pairs, column).dropna()
            if len(values) < 12:
                continue
            scale = float(values.std())
            scale = max(scale, max(abs(float(values.mean())) * 0.10, 0.15))
            stats[column] = {
                "mean": float(values.mean()),
                "scale": scale,
            }

        if stats:
            profiles[int(season)] = {
                "season": int(season),
                "pair_count": int(len(field_pairs)),
                "field_team_count": int(len(season_ids)),
                "stats": stats,
            }

    return {
        "official_seeds": official_seeds,
        "profiles": profiles,
        "field_team_ids": field_team_ids,
    }


def compute_field_reweight_weights(
    train_df: pd.DataFrame,
    target_season: int,
    field_context: Optional[dict[str, object]],
    external_config: dict,
    gender: str,
) -> tuple[Optional[pd.Series], dict[str, object]]:
    empty_info = {
        "enabled": False,
        "target_season": int(target_season),
        "used_features": [],
        "weight_mean": 1.0,
        "weight_std": 0.0,
        "weight_min": 1.0,
        "weight_max": 1.0,
        "pair_count": 0,
        "field_team_count": 0,
    }
    if not field_context:
        return None, empty_info

    profile = field_context.get("profiles", {}).get(int(target_season))
    strength = float(external_config.get("field_live_reweight_strength", {}).get(gender, 0.0))
    min_features = int(external_config.get("field_live_reweight_min_features", 4))
    if profile is None or strength <= 0 or train_df.empty:
        return None, empty_info

    vectors = []
    used_features = []
    for column, stats in profile["stats"].items():
        if column not in train_df.columns:
            continue
        series = field_reweight_series(train_df, column)
        if series.notna().sum() < 50:
            continue
        scale = max(float(stats["scale"]), 0.15)
        z = ((series - float(stats["mean"])) / scale).clip(-6.0, 6.0)
        vectors.append(z.fillna(0.0).to_numpy(dtype=float))
        used_features.append(column)

    if len(used_features) < min_features:
        return None, empty_info

    stacked = np.column_stack(vectors)
    distance = np.mean(stacked ** 2, axis=1)
    kernel = np.exp(-0.5 * distance)
    weights = 1.0 + strength * kernel
    weights = np.where(np.isfinite(weights), weights, 1.0)
    weights = weights / np.mean(weights)
    series = pd.Series(weights, index=train_df.index, dtype=float)
    info = {
        "enabled": True,
        "target_season": int(target_season),
        "used_features": used_features,
        "weight_mean": float(series.mean()),
        "weight_std": float(series.std()),
        "weight_min": float(series.min()),
        "weight_max": float(series.max()),
        "pair_count": int(profile["pair_count"]),
        "field_team_count": int(profile["field_team_count"]),
    }
    return series, info


def official_field_matchup_mask(df: pd.DataFrame, field_team_ids: set[int], season: int = 2026) -> np.ndarray:
    if not field_team_ids or any(column not in df.columns for column in ["Season", "T1", "T2"]):
        return np.zeros(len(df), dtype=bool)
    season_mask = pd.to_numeric(df["Season"], errors="coerce").fillna(-1).astype(int).to_numpy() == int(season)
    t1_mask = df["T1"].isin(field_team_ids).to_numpy()
    t2_mask = df["T2"].isin(field_team_ids).to_numpy()
    return season_mask & t1_mask & t2_mask


def build_matchup_df(tourney_results: pd.DataFrame, team_feats: pd.DataFrame, gender: str) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    feature_candidates = infer_feature_candidates(team_feats)
    games = tourney_results.copy()
    games["T1"] = games[["WTeamID", "LTeamID"]].min(axis=1)
    games["T2"] = games[["WTeamID", "LTeamID"]].max(axis=1)
    games["Label"] = (games["WTeamID"] == games["T1"]).astype(int)
    games["MarginLabel"] = np.where(
        games["WTeamID"] == games["T1"],
        games["WScore"] - games["LScore"],
        games["LScore"] - games["WScore"],
    )

    t1f = team_feats.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2f = team_feats.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    merged = games.merge(t1f, on=["Season", "T1"], how="left")
    merged = merged.merge(t2f, on=["Season", "T2"], how="left")
    merged, diff_feats = compute_diff_features(merged, feature_candidates)
    merged, extra_feats = add_matchup_context_features(merged)
    if gender == "W":
        merged, _ = add_women_tourney_structure_features(merged)

    lr_core_feats = [f"D_{feat}" for feat in LR_CORE_FEATURES[gender] if f"D_{feat}" in merged.columns]
    lr_plus_feats = [f"D_{feat}" for feat in LR_PLUS_FEATURES[gender] if f"D_{feat}" in merged.columns]
    for feat_list in [lr_core_feats, lr_plus_feats]:
        if "AbsSeedDiff" in merged.columns and "AbsSeedDiff" not in feat_list:
            feat_list.append("AbsSeedDiff")

    merged = merged.dropna(subset=["D_Elo"]).reset_index(drop=True)
    return merged, feature_candidates, lr_core_feats, lr_plus_feats, diff_feats + extra_feats


def make_lr_pipeline(gender: str) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=LR_C[gender], max_iter=2500, random_state=42)),
    ])


def make_histgb(gender: str) -> HistGradientBoostingClassifier:
    max_depth = 4 if gender == "M" else 3
    max_iter = 400 if gender == "M" else 320
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_depth=max_depth,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def make_histgb_market_resid(gender: str) -> HistGradientBoostingRegressor:
    max_depth = 4 if gender == "M" else 3
    max_iter = 360 if gender == "M" else 300
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_depth=max_depth,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def make_extratrees(gender: str) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=700 if gender == "M" else 500,
        max_depth=7 if gender == "M" else 6,
        min_samples_leaf=8 if gender == "M" else 6,
        min_samples_split=20 if gender == "M" else 16,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )


def make_xgb(gender: str):
    if xgb is None:
        return None
    max_depth = 4 if gender == "M" else 3
    n_estimators = 500 if gender == "M" else 380
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )


def make_xgb_w_minimal():
    if xgb is None:
        return None
    return xgb.XGBClassifier(
        n_estimators=280,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=8,
        gamma=0.10,
        reg_alpha=0.20,
        reg_lambda=2.50,
        max_delta_step=1,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )


def make_xgb_mse(gender: str):
    if xgb is None:
        return None
    max_depth = 4 if gender == "M" else 3
    n_estimators = 450 if gender == "M" else 340
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        tree_method="hist",
    )


def make_xgb_margin(gender: str):
    if xgb is None:
        return None
    max_depth = 4 if gender == "M" else 3
    n_estimators = 520 if gender == "M" else 380
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        tree_method="hist",
    )


def make_xgb_margin_huber(gender: str):
    if xgb is None:
        return None
    max_depth = 4 if gender == "M" else 3
    n_estimators = 560 if gender == "M" else 400
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        random_state=42,
        tree_method="hist",
    )


def make_cauchy_objective(scale: float):
    scale_sq = float(scale) ** 2

    def objective(y_true, y_pred, sample_weight=None):
        residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        denom = scale_sq + residual**2
        grad = 2.0 * residual / denom
        hess = 2.0 * (scale_sq - residual**2) / (denom**2)
        hess = np.maximum(hess, CAUCHY_HESSIAN_FLOOR)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            grad = grad * weights
            hess = hess * weights
        return grad, hess

    return objective


def make_xgb_margin_cauchy(gender: str):
    if xgb is None:
        return None
    max_depth = 4 if gender == "M" else 3
    n_estimators = 620 if gender == "M" else 420
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.03,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=4,
        gamma=0.08,
        reg_alpha=0.08,
        reg_lambda=1.2,
        objective=make_cauchy_objective(MARGIN_CAUCHY_SCALE[gender]),
        eval_metric="mae",
        random_state=42,
        tree_method="hist",
    )


def make_lgbm(gender: str):
    if lgb is None:
        return None
    n_estimators = 520 if gender == "M" else 420
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )


def make_lgbm_mse(gender: str):
    if lgb is None:
        return None
    n_estimators = 480 if gender == "M" else 380
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )


def make_catboost(gender: str):
    if CatBoostClassifier is None or gender != "M":
        return None
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=False,
        allow_writing_files=False,
        random_seed=42,
    )


def make_meta_learner() -> LogisticRegression:
    return LogisticRegression(C=0.8, max_iter=1500, random_state=42)


def is_market_residual_model(name: str) -> bool:
    return name.endswith("_market_resid")


def is_margin_model(name: str) -> bool:
    return "_margin" in name


def margin_to_prob(margin: np.ndarray, gender: str) -> np.ndarray:
    scale = float(MARGIN_TO_PROB_SCALE[gender])
    margin = np.asarray(margin, dtype=float)
    return safe_clip(1.0 / (1.0 + np.exp(-margin / scale)))


def available_model_specs(gender: str, enable_market_residual: bool = False) -> list[ModelSpec]:
    specs = [
        ModelSpec("lr_core", "lr"),
        ModelSpec("lr_plus", "lr_plus"),
        ModelSpec("et_core", "lr"),
        ModelSpec("et_plus", "lr_plus"),
        ModelSpec("histgb", "all"),
    ]
    if enable_market_residual:
        specs.append(ModelSpec("histgb_market_resid", "all"))
    if xgb is not None:
        if gender == "W":
            specs.append(ModelSpec("xgb_w_minimal", "w_minimal"))
        specs.append(ModelSpec("xgb", "all"))
        specs.append(ModelSpec("xgb_mse", "all"))
        specs.append(ModelSpec("xgb_margin", "all"))
        specs.append(ModelSpec("xgb_margin_huber", "all"))
        specs.append(ModelSpec("xgb_margin_cauchy", "all"))
        if enable_market_residual:
            specs.append(ModelSpec("xgb_market_resid", "all"))
    if lgb is not None:
        specs.append(ModelSpec("lgbm", "all"))
        specs.append(ModelSpec("lgbm_mse", "all"))
        if enable_market_residual:
            specs.append(ModelSpec("lgbm_market_resid", "all"))
    if CatBoostClassifier is not None and gender == "M":
        specs.append(ModelSpec("catboost", "all"))
    return specs


def build_model(name: str, gender: str):
    if name in {"lr_core", "lr_plus"}:
        return make_lr_pipeline(gender)
    if name in {"et_core", "et_plus"}:
        return make_extratrees(gender)
    if name == "histgb":
        return make_histgb(gender)
    if name == "histgb_market_resid":
        return make_histgb_market_resid(gender)
    if name == "xgb_w_minimal":
        return make_xgb_w_minimal()
    if name == "xgb":
        return make_xgb(gender)
    if name == "xgb_mse":
        return make_xgb_mse(gender)
    if name == "xgb_margin":
        return make_xgb_margin(gender)
    if name == "xgb_margin_huber":
        return make_xgb_margin_huber(gender)
    if name == "xgb_margin_cauchy":
        return make_xgb_margin_cauchy(gender)
    if name == "xgb_market_resid":
        return make_xgb_mse(gender)
    if name == "lgbm":
        return make_lgbm(gender)
    if name == "lgbm_mse":
        return make_lgbm_mse(gender)
    if name == "lgbm_market_resid":
        return make_lgbm_mse(gender)
    if name == "catboost":
        return make_catboost(gender)
    raise ValueError(f"Unknown model: {name}")


def fit_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    gender: str,
    sample_weight: Optional[pd.Series | np.ndarray] = None,
):
    model = build_model(name, gender)
    train_x = x_train
    train_y = y_train
    train_weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    if is_market_residual_model(name):
        if "MarketProb" not in x_train.columns:
            return None
        market_prob = pd.to_numeric(x_train["MarketProb"], errors="coerce")
        valid_mask = market_prob.notna().to_numpy()
        if valid_mask.sum() < MARKET_RESIDUAL_MIN_ROWS:
            return None
        train_x = x_train.loc[valid_mask].copy()
        train_y = pd.Series(
            y_train.loc[valid_mask].to_numpy() - market_prob.loc[valid_mask].to_numpy(),
            index=train_x.index,
        )
        if train_weights is not None:
            train_weights = train_weights[valid_mask]

    if train_weights is None:
        model.fit(train_x, train_y)
        return model

    if isinstance(model, Pipeline):
        last_step = next(reversed(model.named_steps))
        model.fit(train_x, train_y, **{f"{last_step}__sample_weight": train_weights})
        return model

    model.fit(train_x, train_y, sample_weight=train_weights)
    return model


def training_target_for_model(df: pd.DataFrame, model_name: str) -> pd.Series:
    if is_margin_model(model_name):
        return pd.to_numeric(df["MarginLabel"], errors="coerce").astype(float)
    return pd.to_numeric(df["Label"], errors="coerce").astype(float)


def predict_model(name: str, model, x_test: pd.DataFrame, gender: str) -> np.ndarray:
    if model is None:
        return np.full(len(x_test), np.nan, dtype=float)
    if is_market_residual_model(name):
        if "MarketProb" not in x_test.columns:
            return np.full(len(x_test), np.nan, dtype=float)
        market_prob = pd.to_numeric(x_test["MarketProb"], errors="coerce")
        pred = np.full(len(x_test), np.nan, dtype=float)
        valid_mask = market_prob.notna().to_numpy()
        if valid_mask.any():
            correction = np.asarray(model.predict(x_test.loc[valid_mask]), dtype=float)
            pred[valid_mask] = safe_clip(market_prob.loc[valid_mask].to_numpy() + correction)
        return pred
    if is_margin_model(name):
        return margin_to_prob(np.asarray(model.predict(x_test), dtype=float), gender)
    if hasattr(model, "predict_proba"):
        return safe_clip(model.predict_proba(x_test)[:, 1])
    return safe_clip(np.asarray(model.predict(x_test), dtype=float))


def feature_frame(
    df: pd.DataFrame,
    feature_key: str,
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    women_minimal_feats: Optional[list[str]] = None,
) -> pd.DataFrame:
    if feature_key == "lr":
        columns = lr_core_feats
    elif feature_key == "lr_plus":
        columns = lr_plus_feats
    elif feature_key == "w_minimal":
        columns = women_minimal_feats or lr_plus_feats
    else:
        columns = all_feats
    return df[columns].fillna(0.0)


def generate_base_oof(
    matchups: pd.DataFrame,
    model_specs: list[ModelSpec],
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    women_minimal_feats: Optional[list[str]],
    gender: str,
    field_context: Optional[dict[str, object]] = None,
    external_config: Optional[dict] = None,
) -> pd.DataFrame:
    seasons = sorted(matchups["Season"].unique())
    rows = []
    diag_columns = [column for column in ["AbsSeedDiff", "T1BetterSeed", "SameConference"] if column in matchups.columns]
    external_config = external_config or DEFAULT_EXTERNAL_CONFIG
    residual_min_coverage = float(external_config.get("min_market_coverage_for_residual_models", MARKET_RESIDUAL_SELECTION_MIN_COVERAGE))
    residual_active_seasons = set(market_residual_active_seasons(market_coverage_by_season(matchups), residual_min_coverage))

    for season in seasons:
        prior = [value for value in seasons if value < season]
        if len(prior) < MIN_TRAIN_SEASONS:
            continue

        train_df = matchups[matchups["Season"] < season]
        test_df = matchups[matchups["Season"] == season]
        if train_df.empty or test_df.empty:
            continue

        sample_weight, _ = compute_field_reweight_weights(train_df, season, field_context, external_config, gender)

        row = {
            "Season": test_df["Season"].values,
            "T1": test_df["T1"].values,
            "T2": test_df["T2"].values,
            "Label": test_df["Label"].values,
        }
        for column in diag_columns:
            row[f"Diag_{column}"] = test_df[column].values
        for spec in model_specs:
            train_subset = train_df
            test_subset = test_df
            sample_weight_subset = sample_weight
            if is_market_residual_model(spec.name):
                train_mask = market_residual_train_mask(train_df)
                train_subset = train_df.loc[train_mask].copy()
                sample_weight_subset = subset_sample_weight(sample_weight, train_df.index, train_subset.index)
                if int(season) not in residual_active_seasons:
                    row[f"Prob_{spec.name}"] = np.full(len(test_df), np.nan, dtype=float)
                    continue
            x_train = feature_frame(train_subset, spec.feature_key, lr_core_feats, lr_plus_feats, all_feats, women_minimal_feats)
            x_test = feature_frame(test_subset, spec.feature_key, lr_core_feats, lr_plus_feats, all_feats, women_minimal_feats)
            y_train = training_target_for_model(train_subset, spec.name)
            model = fit_model(spec.name, x_train, y_train, gender, sample_weight=sample_weight_subset)
            row[f"Prob_{spec.name}"] = predict_model(spec.name, model, x_test, gender)
        rows.append(pd.DataFrame(row))

    if not rows:
        return pd.DataFrame(columns=["Season", "Label"])
    return pd.concat(rows, ignore_index=True)


def evaluate_base_scores(oof_df: pd.DataFrame, model_names: list[str], eval_years: int) -> dict[str, float]:
    eval_seasons = sorted(oof_df["Season"].unique())[-eval_years:]
    scores = {}
    for name in model_names:
        column = f"Prob_{name}"
        season_scores = []
        for season in eval_seasons:
            fold = oof_df[(oof_df["Season"] == season) & oof_df[column].notna()]
            if is_market_residual_model(name):
                season_total = oof_df[oof_df["Season"] == season]
                season_coverage = float(season_total[column].notna().mean()) if not season_total.empty else 0.0
                if season_coverage < MARKET_RESIDUAL_SELECTION_MIN_COVERAGE:
                    continue
            if fold.empty:
                continue
            season_scores.append(brier_score_loss(fold["Label"], fold[column]))
        if is_market_residual_model(name) and len(season_scores) < min(MARKET_RESIDUAL_MIN_EVAL_SEASONS, len(eval_seasons)):
            scores[name] = np.nan
        else:
            scores[name] = float(np.mean(season_scores)) if season_scores else np.nan
    return scores


def select_model_names(base_scores: dict[str, float], gender: str) -> list[str]:
    valid = [(name, score) for name, score in base_scores.items() if not np.isnan(score)]
    valid.sort(key=lambda item: item[1])
    if not valid:
        return []

    best = valid[0][1]
    tolerance = 0.005 if gender == "M" else 0.004
    selected = [name for name, score in valid if score <= best + tolerance][: MODEL_LIMIT[gender]]
    if len(selected) < min(2, len(valid)):
        selected = [name for name, _ in valid[: MODEL_LIMIT[gender]]]
    if "lr_core" in [name for name, _ in valid] and "lr_core" not in selected:
        selected = (["lr_core"] + selected)[: MODEL_LIMIT[gender]]
    return selected


def feature_key_for_model(name: str) -> str:
    if name == "lr_core":
        return "lr"
    if name == "lr_plus":
        return "lr_plus"
    if name == "xgb_w_minimal":
        return "w_minimal"
    return "all"


def evaluate_model_set_strategy(
    oof_df: pd.DataFrame,
    model_names: list[str],
    gender: str,
    eval_years: int,
) -> Optional[dict[str, object]]:
    prob_columns = [f"Prob_{name}" for name in model_names]
    if not model_names or any(column not in oof_df.columns for column in prob_columns):
        return None

    diag_columns = [column for column in ["Diag_AbsSeedDiff", "Diag_T1BetterSeed", "Diag_SameConference"] if column in oof_df.columns]
    selected = oof_df[["Season", "T1", "T2", "Label"] + diag_columns + prob_columns].copy()
    selected_oof, _ = add_ensemble_oof(selected, model_names, gender)
    raw_column, calibration_method, shrinkage, strategy_scores = evaluate_strategy_grid(selected_oof, eval_years=eval_years)
    return {
        "model_names": model_names,
        "best_cv_brier": final_cv_score(strategy_scores),
        "raw_column": raw_column,
        "calibration_method": calibration_method,
        "shrinkage": shrinkage,
    }


def choose_women_model_set(
    oof_df: pd.DataFrame,
    default_selected: list[str],
    available_model_names: list[str],
    eval_years: int,
) -> Optional[dict[str, object]]:
    candidate_sets: list[list[str]] = []

    def add_candidate(candidate: list[str]) -> None:
        filtered = [name for name in candidate if name in available_model_names]
        if len(filtered) < 2:
            return
        if filtered not in candidate_sets:
            candidate_sets.append(filtered)

    add_candidate(default_selected)
    add_candidate(["lr_core", "et_core", "et_plus", "xgb_w_minimal"])
    add_candidate(["lr_core", "et_core", "xgb_w_minimal"])
    add_candidate(["lr_core", "xgb_w_minimal"])
    add_candidate(["lr_core", "lr_plus", "xgb_w_minimal"])

    best_choice = None
    for candidate in candidate_sets:
        result = evaluate_model_set_strategy(oof_df, candidate, "W", eval_years)
        if result is None:
            continue
        if best_choice is None or result["best_cv_brier"] < best_choice["best_cv_brier"]:
            best_choice = result
    return best_choice


def meta_feature_frame(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    prob_columns = [f"Prob_{name}" for name in model_names]
    prob_matrix = df[prob_columns].copy()
    row_mean = prob_matrix.mean(axis=1).fillna(0.5)
    prob_matrix = prob_matrix.apply(lambda col: col.fillna(row_mean))

    features = {}
    for name in model_names:
        features[f"Logit_{name}"] = safe_logit(prob_matrix[f"Prob_{name}"].to_numpy())
    features["ProbMean"] = prob_matrix.mean(axis=1).to_numpy()
    features["ProbStd"] = prob_matrix.std(axis=1).fillna(0.0).to_numpy()
    features["ProbMin"] = prob_matrix.min(axis=1).to_numpy()
    features["ProbMax"] = prob_matrix.max(axis=1).to_numpy()
    return pd.DataFrame(features, index=df.index)


def add_gender_strategy_columns(df: pd.DataFrame, model_names: list[str], gender: str) -> pd.DataFrame:
    out = df.copy()
    if gender != "W":
        return out

    lr_columns = [column for column in [f"Prob_{name}" for name in model_names if name.startswith("lr_")] if column in out.columns]
    if not lr_columns:
        return out

    linear_prob = out[lr_columns].mean(axis=1)
    other_columns = [column for column in [f"Prob_{name}" for name in model_names if not name.startswith("lr_")] if column in out.columns]
    tree_prob = out[other_columns].mean(axis=1) if other_columns else linear_prob

    out["Prob_w_linear_only"] = safe_clip(linear_prob.fillna(0.5).to_numpy())
    for weight in WOMEN_LINEAR_BLEND_WEIGHTS:
        out[f"Prob_w_lr_tilt_{int(round(weight * 100))}"] = safe_clip(
            (weight * linear_prob + (1.0 - weight) * tree_prob).fillna(0.5).to_numpy()
        )
    return out


def add_ensemble_oof(oof_df: pd.DataFrame, model_names: list[str], gender: str) -> tuple[pd.DataFrame, Optional[LogisticRegression]]:
    oof = oof_df.copy()
    oof["ProbMean"] = oof[[f"Prob_{name}" for name in model_names]].mean(axis=1)
    oof["ProbStack"] = np.nan

    seasons = sorted(oof["Season"].unique())
    for season in seasons:
        train_df = oof[oof["Season"] < season]
        test_mask = oof["Season"] == season
        if len(train_df) < CALIBRATION_MIN_ROWS:
            continue
        meta_model = make_meta_learner()
        meta_model.fit(meta_feature_frame(train_df, model_names), train_df["Label"])
        oof.loc[test_mask, "ProbStack"] = safe_clip(
            meta_model.predict_proba(meta_feature_frame(oof.loc[test_mask], model_names))[:, 1]
        )

    final_meta = None
    if len(oof) >= CALIBRATION_MIN_ROWS:
        final_meta = make_meta_learner()
        final_meta.fit(meta_feature_frame(oof, model_names), oof["Label"])

    return add_gender_strategy_columns(oof, model_names, gender), final_meta


def fit_calibrator(method: str, train_prob: np.ndarray, labels: np.ndarray):
    train_prob = safe_clip(train_prob)
    if method == "none":
        return None
    if method == "platt":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(safe_logit(train_prob).reshape(-1, 1), labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_prob, labels)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(prob: np.ndarray, method: str, calibrator) -> np.ndarray:
    prob = safe_clip(prob)
    if method == "none" or calibrator is None:
        return prob
    if method == "platt":
        return safe_clip(calibrator.predict_proba(safe_logit(prob).reshape(-1, 1))[:, 1])
    return safe_clip(calibrator.predict(prob))


def rowwise_prob_fallback(df: pd.DataFrame) -> np.ndarray:
    prob_columns = [column for column in df.columns if column.startswith("Prob_")]
    if not prob_columns:
        return np.full(len(df), 0.5, dtype=float)
    return safe_clip(df[prob_columns].mean(axis=1).fillna(0.5).to_numpy())


def resolved_strategy_probabilities(df: pd.DataFrame, column: str) -> np.ndarray:
    fallback = rowwise_prob_fallback(df)
    if column == "ProbMean":
        return fallback
    if column == "ProbStack":
        if column not in df.columns:
            return fallback
        return safe_clip(pd.to_numeric(df[column], errors="coerce").fillna(pd.Series(fallback, index=df.index)).to_numpy())

    prob = pd.to_numeric(df[column], errors="coerce")
    return safe_clip(prob.fillna(pd.Series(fallback, index=df.index)).to_numpy())


def evaluate_strategy_grid(oof_df: pd.DataFrame, eval_years: int) -> tuple[str, str, float, dict[str, float]]:
    eval_seasons = sorted(oof_df["Season"].unique())[-eval_years:]
    candidate_columns = [column for column in oof_df.columns if column.startswith("Prob_")]
    if "ProbMean" in oof_df.columns:
        candidate_columns.append("ProbMean")
    if "ProbStack" in oof_df.columns and oof_df["ProbStack"].notna().any():
        candidate_columns.append("ProbStack")

    strategy_scores = {}
    for column in candidate_columns:
        if column.startswith("Prob_"):
            model_name = column.removeprefix("Prob_")
        for method in ["none", "platt", "isotonic"]:
            fold_scores = []
            fold_predictions: list[tuple[np.ndarray, np.ndarray]] = []
            for season in eval_seasons:
                train_df = oof_df[oof_df["Season"] < season]
                test_df = oof_df[oof_df["Season"] == season]
                if test_df.empty:
                    continue
                if column.startswith("Prob_") and is_market_residual_model(model_name):
                    season_coverage = float(test_df[column].notna().mean()) if column in test_df.columns else 0.0
                    if season_coverage < MARKET_RESIDUAL_SELECTION_MIN_COVERAGE:
                        continue
                raw_train = resolved_strategy_probabilities(train_df, column)
                raw_test = resolved_strategy_probabilities(test_df, column)
                if method == "none" or len(train_df) < CALIBRATION_MIN_ROWS:
                    pred = raw_test
                else:
                    try:
                        calibrator = fit_calibrator(method, raw_train, train_df["Label"].to_numpy())
                        pred = apply_calibrator(raw_test, method, calibrator)
                    except Exception:
                        pred = raw_test
                fold_predictions.append((test_df["Label"].to_numpy(), pred))
            if column.startswith("Prob_") and is_market_residual_model(model_name):
                if len(fold_predictions) < min(MARKET_RESIDUAL_MIN_EVAL_SEASONS, len(eval_seasons)):
                    continue
            for shrinkage in SHRINKAGE_GRID:
                scores = []
                for labels, pred in fold_predictions:
                    scores.append(brier_score_loss(labels, shrink_toward_half(pred, shrinkage)))
                if scores:
                    strategy_scores[f"{column}|{method}|shrink={shrinkage:.2f}"] = float(np.mean(scores))

    best_key = min(strategy_scores, key=strategy_scores.get)
    best_column, best_method, best_shrink = best_key.split("|")
    return best_column, best_method, float(best_shrink.split("=")[1]), strategy_scores


def build_strategy_oof_predictions(
    oof_df: pd.DataFrame,
    raw_column: str,
    calibration_method: str,
    shrinkage: float,
    eval_years: Optional[int] = None,
) -> pd.DataFrame:
    seasons = sorted(oof_df["Season"].unique())
    eval_seasons = seasons if eval_years is None else seasons[-eval_years:]
    available = oof_df.copy()
    diag_columns = [column for column in available.columns if column.startswith("Diag_")]
    rows = []

    for season in eval_seasons:
        train_df = available[available["Season"] < season]
        test_df = available[available["Season"] == season].copy()
        if test_df.empty:
            continue

        raw_prob = resolved_strategy_probabilities(test_df, raw_column)
        if calibration_method == "none" or len(train_df) < CALIBRATION_MIN_ROWS:
            calibrated = raw_prob
        else:
            calibrator = fit_calibrator(
                calibration_method,
                resolved_strategy_probabilities(train_df, raw_column),
                train_df["Label"].to_numpy(),
            )
            calibrated = apply_calibrator(raw_prob, calibration_method, calibrator)
        final_prob = shrink_toward_half(calibrated, shrinkage)

        base_cols = [column for column in ["Season", "T1", "T2", "Label"] if column in test_df.columns]
        frame = test_df[base_cols + diag_columns].copy()
        frame["RawProb"] = raw_prob
        frame["FinalProb"] = final_prob
        frame["FavoriteProb"] = np.maximum(final_prob, 1.0 - final_prob)
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["Season", "Label", "RawProb", "FinalProb", "FavoriteProb"])
    return pd.concat(rows, ignore_index=True)


def replace_strategy_final_prob(strategy_df: pd.DataFrame, scored_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    keys = ["Season", "T1", "T2", "Label"]
    pred_frame = scored_df[keys].copy()
    pred_frame["FinalProb"] = safe_clip(pred)
    base = strategy_df.drop(columns=["FinalProb", "FavoriteProb"], errors="ignore")
    frame = base.merge(pred_frame, on=keys, how="left")
    frame["FavoriteProb"] = np.maximum(frame["FinalProb"], 1.0 - frame["FinalProb"])
    return frame


def make_tossup_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.35, max_iter=2500, random_state=42)),
    ])


def make_tossup_et_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=450,
        max_depth=6,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )


def make_chalk_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.55, max_iter=2500, random_state=42)),
    ])


def make_chalk_et_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        min_samples_split=18,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )


def add_base_probability_features(df: pd.DataFrame, prob: np.ndarray | pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["BaseProb"] = safe_clip(np.asarray(prob, dtype=float))
    out["BaseFavoriteProb"] = np.maximum(out["BaseProb"], 1.0 - out["BaseProb"])
    out["BaseUncertainty"] = 1.0 - 2.0 * np.abs(out["BaseProb"] - 0.5)
    return out


def specialist_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in TOSSUP_SPECIALIST_FEATURES if column in df.columns]


def chalk_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in CHALK_SPECIALIST_FEATURES if column in df.columns]


def tossup_gate_mask(df: pd.DataFrame, mode: str, seed_threshold: int, favorite_threshold: float) -> pd.Series:
    seed_mask = df["AbsSeedDiff"].fillna(np.inf) <= seed_threshold if "AbsSeedDiff" in df.columns else pd.Series(False, index=df.index)
    favorite_mask = df["BaseFavoriteProb"] <= favorite_threshold if "BaseFavoriteProb" in df.columns else pd.Series(False, index=df.index)
    if mode == "seed_only":
        return seed_mask
    if mode == "seed_and_prob":
        return seed_mask & favorite_mask
    return favorite_mask


def fit_expert_models(train_df: pd.DataFrame, feature_cols: list[str], expert_type: str) -> Optional[dict[str, object]]:
    min_rows = TOSSUP_MIN_ROWS if expert_type == "tossup" else CHALK_MIN_ROWS
    if len(train_df) < min_rows or train_df["Label"].nunique() < 2 or not feature_cols:
        return None
    x_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["Label"]
    if expert_type == "tossup":
        lr_model = make_tossup_lr_pipeline()
        et_model = make_tossup_et_model()
    else:
        lr_model = make_chalk_lr_pipeline()
        et_model = make_chalk_et_model()
    lr_model.fit(x_train, y_train)
    et_model.fit(x_train, y_train)
    return {"feature_cols": feature_cols, "lr": lr_model, "et": et_model, "expert_type": expert_type}


def predict_expert_models(bundle: dict[str, object], df: pd.DataFrame) -> np.ndarray:
    x_test = df[bundle["feature_cols"]].fillna(0.0)
    preds = [
        safe_clip(bundle["lr"].predict_proba(x_test)[:, 1]),
        safe_clip(bundle["et"].predict_proba(x_test)[:, 1]),
    ]
    return safe_clip(np.mean(np.column_stack(preds), axis=1))


def fit_tossup_specialist_models(train_df: pd.DataFrame, feature_cols: list[str]) -> Optional[dict[str, object]]:
    return fit_expert_models(train_df, feature_cols, expert_type="tossup")


def predict_tossup_specialist_models(bundle: dict[str, object], df: pd.DataFrame) -> np.ndarray:
    return predict_expert_models(bundle, df)


def available_market_residual_overlay_models(gender: str) -> list[str]:
    models = []
    for name in MARKET_RESIDUAL_OVERLAY_MODELS:
        if name.startswith("xgb") and xgb is None:
            continue
        if name.startswith("lgbm") and lgb is None:
            continue
        models.append(name)
    return models


def strategy_feature_merge(matchups: pd.DataFrame, strategy_oof: pd.DataFrame) -> pd.DataFrame:
    required = [column for column in ["Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"] if column in strategy_oof.columns]
    merged = matchups.merge(strategy_oof[required], on=["Season", "T1", "T2", "Label"], how="inner")
    return merged


def evaluate_market_residual_overlay(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    gender: str,
    eval_years: int,
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if gender != "M" or strategy_oof.empty or "MarketProb" not in matchups.columns:
        return None

    merged = strategy_feature_merge(matchups, strategy_oof)
    if merged.empty or merged["MarketProb"].notna().sum() < MARKET_RESIDUAL_MIN_ROWS:
        return None

    eval_seasons = sorted(merged["Season"].unique())[-eval_years:]
    model_names = available_market_residual_overlay_models(gender)
    if not model_names:
        return None

    grid_scores: dict[str, float] = {}
    for model_name in model_names:
        fold_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for season in eval_seasons:
            train_df = merged[merged["Season"] < season].copy()
            test_df = merged[merged["Season"] == season].copy()
            if test_df.empty:
                continue

            x_train = feature_frame(train_df, "all", lr_core_feats, lr_plus_feats, all_feats)
            x_test = feature_frame(test_df, "all", lr_core_feats, lr_plus_feats, all_feats)
            model = fit_model(model_name, x_train, train_df["Label"], gender)
            if model is None:
                continue

            overlay_prob = predict_model(model_name, model, x_test, gender)
            fold_results.append(
                (
                    test_df["Label"].to_numpy(),
                    test_df["FinalProb"].to_numpy(),
                    overlay_prob,
                )
            )

        for blend_weight in MARKET_RESIDUAL_OVERLAY_BLEND_WEIGHTS:
            scores = []
            active_rows_total = 0
            for labels, base_prob, overlay_prob in fold_results:
                pred = base_prob.copy()
                active_mask = np.isfinite(overlay_prob)
                active_rows_total += int(active_mask.sum())
                if active_mask.any():
                    pred[active_mask] = safe_clip(
                        (1.0 - blend_weight) * pred[active_mask] + blend_weight * overlay_prob[active_mask]
                    )
                scores.append(brier_score_loss(labels, pred))

            if scores and active_rows_total >= MARKET_RESIDUAL_OVERLAY_MIN_EVAL_ROWS:
                key = f"{model_name}|w={blend_weight:.2f}"
                grid_scores[key] = float(np.mean(scores))

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.00005:
        return None

    model_name, weight_part = best_key.split("|")
    blend_weight = float(weight_part.split("=")[1])
    final_x = feature_frame(merged, "all", lr_core_feats, lr_plus_feats, all_feats)
    final_model = fit_model(model_name, final_x, merged["Label"], gender)
    if final_model is None:
        return None

    return {
        "model_name": model_name,
        "feature_key": "all",
        "blend_weight": blend_weight,
        "best_cv_brier": best_score,
        "grid_scores": grid_scores,
        "model": final_model,
    }


def build_market_residual_overlay_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    overlay: dict[str, object],
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    gender: str,
) -> pd.DataFrame:
    merged = strategy_feature_merge(matchups, strategy_oof)
    if merged.empty:
        return strategy_oof

    seasons = sorted(merged["Season"].unique())
    rows = []
    for season in seasons:
        train_df = merged[merged["Season"] < season].copy()
        test_df = merged[merged["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = test_df["FinalProb"].to_numpy().copy()
        if len(train_df["Season"].unique()) >= MIN_TRAIN_SEASONS:
            x_train = feature_frame(train_df, overlay["feature_key"], lr_core_feats, lr_plus_feats, all_feats)
            x_test = feature_frame(test_df, overlay["feature_key"], lr_core_feats, lr_plus_feats, all_feats)
            model = fit_model(overlay["model_name"], x_train, train_df["Label"], gender)
            if model is not None:
                overlay_prob = predict_model(overlay["model_name"], model, x_test, gender)
                active_mask = np.isfinite(overlay_prob)
                if active_mask.any():
                    pred[active_mask] = safe_clip(
                        (1.0 - overlay["blend_weight"]) * pred[active_mask]
                        + overlay["blend_weight"] * overlay_prob[active_mask]
                    )

        frame = strategy_oof[strategy_oof["Season"] == season].copy()
        frame["FinalProb"] = pred
        frame["FavoriteProb"] = np.maximum(pred, 1.0 - pred)
        rows.append(frame)

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def apply_market_residual_overlay(
    bundle: dict[str, object],
    pred_df: pd.DataFrame,
    prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = bundle.get("market_residual_overlay")
    if bundle["gender"] != "M" or not overlay:
        return prob, np.zeros(len(pred_df), dtype=bool)

    coverage_by_season = market_coverage_by_season(pred_df)
    active_seasons = set(market_residual_active_seasons(
        coverage_by_season,
        float(bundle["external_config"].get("min_market_coverage_for_residual_models", MARKET_RESIDUAL_SELECTION_MIN_COVERAGE)),
    ))
    routing_mask = market_residual_active_mask(pred_df, active_seasons)
    if not routing_mask.any():
        return prob, np.zeros(len(pred_df), dtype=bool)

    x_test = feature_frame(pred_df, overlay["feature_key"], bundle["lr_core_feats"], bundle["lr_plus_feats"], bundle["all_feats"])
    overlay_prob = predict_model(overlay["model_name"], overlay["model"], x_test, bundle["gender"])
    active_mask = routing_mask & np.isfinite(overlay_prob)
    if not active_mask.any():
        return prob, active_mask

    adjusted = prob.copy()
    blend_weight = float(overlay["blend_weight"])
    adjusted[active_mask] = safe_clip(
        (1.0 - blend_weight) * adjusted[active_mask] + blend_weight * overlay_prob[active_mask]
    )
    return adjusted, active_mask


def evaluate_tossup_specialist(
    matchups: pd.DataFrame,
    oof_strategy: pd.DataFrame,
    eval_years: int,
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if oof_strategy.empty:
        return None

    specialist_df = matchups.merge(
        oof_strategy[[column for column in ["Season", "T1", "T2", "Label", "FinalProb", "FavoriteProb"] if column in oof_strategy.columns]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if specialist_df.empty:
        return None

    specialist_df = add_base_probability_features(specialist_df, specialist_df["FinalProb"].to_numpy())
    feature_cols = specialist_feature_columns(specialist_df)
    if not feature_cols:
        return None

    eval_seasons = sorted(specialist_df["Season"].unique())[-eval_years:]
    grid_scores: dict[str, float] = {}

    for mode in TOSSUP_GATE_MODES:
        for seed_threshold in TOSSUP_SEED_THRESHOLDS:
            for favorite_threshold in TOSSUP_FAVORITE_THRESHOLDS:
                for blend_weight in TOSSUP_BLEND_WEIGHTS:
                    fold_scores = []
                    gate_rows_total = 0
                    for season in eval_seasons:
                        train_df = specialist_df[specialist_df["Season"] < season].copy()
                        test_df = specialist_df[specialist_df["Season"] == season].copy()
                        if test_df.empty:
                            continue

                        train_gate = tossup_gate_mask(train_df, mode, seed_threshold, favorite_threshold)
                        specialist_models = fit_tossup_specialist_models(train_df.loc[train_gate], feature_cols)
                        if specialist_models is None:
                            continue

                        pred = test_df["FinalProb"].to_numpy().copy()
                        test_gate = tossup_gate_mask(test_df, mode, seed_threshold, favorite_threshold).to_numpy()
                        if test_gate.any():
                            gate_rows_total += int(test_gate.sum())
                            specialist_pred = predict_tossup_specialist_models(specialist_models, test_df.loc[test_gate])
                            pred[test_gate] = safe_clip((1.0 - blend_weight) * pred[test_gate] + blend_weight * specialist_pred)
                        fold_scores.append(brier_score_loss(test_df["Label"], pred))

                    if fold_scores and gate_rows_total >= TOSSUP_MIN_EVAL_GATE_ROWS:
                        key = f"{mode}|seed<={seed_threshold}|fav<={favorite_threshold:.2f}|w={blend_weight:.2f}"
                        grid_scores[key] = float(np.mean(fold_scores))

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.0001:
        return None

    mode, seed_part, fav_part, weight_part = best_key.split("|")
    seed_threshold = int(seed_part.split("<=")[1])
    favorite_threshold = float(fav_part.split("<=")[1])
    blend_weight = float(weight_part.split("=")[1])

    final_gate = tossup_gate_mask(specialist_df, mode, seed_threshold, favorite_threshold)
    final_models = fit_tossup_specialist_models(specialist_df.loc[final_gate], feature_cols)
    if final_models is None:
        return None

    return {
        "mode": mode,
        "seed_threshold": seed_threshold,
        "favorite_threshold": favorite_threshold,
        "blend_weight": blend_weight,
        "best_cv_brier": best_score,
        "grid_scores": grid_scores,
        "feature_cols": feature_cols,
        "models": final_models,
    }


def apply_tossup_specialist(bundle: dict[str, object], pred_df: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    specialist = bundle.get("tossup_specialist")
    if bundle["gender"] != "M" or not specialist:
        return prob

    work = add_base_probability_features(pred_df, prob)
    gate = tossup_gate_mask(work, specialist["mode"], specialist["seed_threshold"], specialist["favorite_threshold"]).to_numpy()
    if not gate.any():
        return prob

    specialist_pred = predict_tossup_specialist_models(specialist["models"], work.loc[gate])
    adjusted = prob.copy()
    weight = specialist["blend_weight"]
    adjusted[gate] = safe_clip((1.0 - weight) * adjusted[gate] + weight * specialist_pred)
    return adjusted


def evaluate_moe_routing(
    matchups: pd.DataFrame,
    oof_strategy: pd.DataFrame,
    eval_years: int,
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if oof_strategy.empty:
        return None

    moe_df = matchups.merge(
        oof_strategy[[column for column in ["Season", "T1", "T2", "Label", "FinalProb", "FavoriteProb"] if column in oof_strategy.columns]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if moe_df.empty:
        return None

    moe_df = add_base_probability_features(moe_df, moe_df["FinalProb"].to_numpy())
    toss_cols = specialist_feature_columns(moe_df)
    chalk_cols = chalk_feature_columns(moe_df)
    if not toss_cols or not chalk_cols:
        return None

    eval_seasons = sorted(moe_df["Season"].unique())[-eval_years:]
    grid_scores: dict[str, float] = {}

    for seed_threshold in MOE_SEED_THRESHOLDS:
        for favorite_threshold in MOE_FAVORITE_THRESHOLDS:
            fold_scores = []
            for season in eval_seasons:
                train_df = moe_df[moe_df["Season"] < season].copy()
                test_df = moe_df[moe_df["Season"] == season].copy()
                if test_df.empty:
                    continue

                toss_train_mask = tossup_gate_mask(train_df, "seed_and_prob", seed_threshold, favorite_threshold)
                chalk_train_mask = ~toss_train_mask
                toss_bundle = fit_expert_models(train_df.loc[toss_train_mask], toss_cols, expert_type="tossup")
                chalk_bundle = fit_expert_models(train_df.loc[chalk_train_mask], chalk_cols, expert_type="chalk")
                if toss_bundle is None or chalk_bundle is None:
                    continue

                toss_test_mask = tossup_gate_mask(test_df, "seed_and_prob", seed_threshold, favorite_threshold).to_numpy()
                pred = np.zeros(len(test_df), dtype=float)
                if toss_test_mask.any():
                    pred[toss_test_mask] = predict_expert_models(toss_bundle, test_df.loc[toss_test_mask])
                if (~toss_test_mask).any():
                    pred[~toss_test_mask] = predict_expert_models(chalk_bundle, test_df.loc[~toss_test_mask])
                fold_scores.append(brier_score_loss(test_df["Label"], safe_clip(pred)))

            if fold_scores:
                key = f"seed<={seed_threshold}|fav<={favorite_threshold:.2f}"
                grid_scores[key] = float(np.mean(fold_scores))

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.00005:
        return None

    seed_part, fav_part = best_key.split("|")
    seed_threshold = int(seed_part.split("<=")[1])
    favorite_threshold = float(fav_part.split("<=")[1])

    final_toss_mask = tossup_gate_mask(moe_df, "seed_and_prob", seed_threshold, favorite_threshold)
    toss_bundle = fit_expert_models(moe_df.loc[final_toss_mask], toss_cols, expert_type="tossup")
    chalk_bundle = fit_expert_models(moe_df.loc[~final_toss_mask], chalk_cols, expert_type="chalk")
    if toss_bundle is None or chalk_bundle is None:
        return None

    return {
        "mode": "hard_route",
        "seed_threshold": seed_threshold,
        "favorite_threshold": favorite_threshold,
        "best_cv_brier": best_score,
        "grid_scores": grid_scores,
        "tossup_feature_cols": toss_cols,
        "chalk_feature_cols": chalk_cols,
        "tossup_models": toss_bundle,
        "chalk_models": chalk_bundle,
    }


def apply_moe_routing(bundle: dict[str, object], pred_df: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    moe = bundle.get("moe_routing")
    if bundle["gender"] != "M" or not moe:
        return prob

    work = add_base_probability_features(pred_df, prob)
    toss_mask = tossup_gate_mask(work, "seed_and_prob", moe["seed_threshold"], moe["favorite_threshold"]).to_numpy()
    adjusted = np.zeros(len(work), dtype=float)
    if toss_mask.any():
        adjusted[toss_mask] = predict_expert_models(moe["tossup_models"], work.loc[toss_mask])
    if (~toss_mask).any():
        adjusted[~toss_mask] = predict_expert_models(moe["chalk_models"], work.loc[~toss_mask])
    return safe_clip(adjusted)


def build_tossup_specialist_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    specialist: dict[str, object],
) -> pd.DataFrame:
    specialist_df = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if specialist_df.empty:
        return strategy_oof

    specialist_df = add_base_probability_features(specialist_df, specialist_df["FinalProb"].to_numpy())
    rows = []
    for season in sorted(specialist_df["Season"].unique()):
        train_df = specialist_df[specialist_df["Season"] < season].copy()
        test_df = specialist_df[specialist_df["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = test_df["FinalProb"].to_numpy().copy()
        if len(train_df["Season"].unique()) >= MIN_TRAIN_SEASONS:
            train_gate = tossup_gate_mask(train_df, specialist["mode"], specialist["seed_threshold"], specialist["favorite_threshold"])
            models = fit_tossup_specialist_models(train_df.loc[train_gate], specialist["feature_cols"])
            test_gate = tossup_gate_mask(test_df, specialist["mode"], specialist["seed_threshold"], specialist["favorite_threshold"]).to_numpy()
            if models is not None and test_gate.any():
                specialist_pred = predict_tossup_specialist_models(models, test_df.loc[test_gate])
                pred[test_gate] = safe_clip(
                    (1.0 - specialist["blend_weight"]) * pred[test_gate] + specialist["blend_weight"] * specialist_pred
                )

        season_strategy = strategy_oof[strategy_oof["Season"] == season].copy()
        rows.append(replace_strategy_final_prob(season_strategy, test_df, pred))

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def build_moe_routing_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    moe: dict[str, object],
) -> pd.DataFrame:
    moe_df = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if moe_df.empty:
        return strategy_oof

    moe_df = add_base_probability_features(moe_df, moe_df["FinalProb"].to_numpy())
    rows = []
    for season in sorted(moe_df["Season"].unique()):
        train_df = moe_df[moe_df["Season"] < season].copy()
        test_df = moe_df[moe_df["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = test_df["FinalProb"].to_numpy().copy()
        if len(train_df["Season"].unique()) >= MIN_TRAIN_SEASONS:
            toss_train_mask = tossup_gate_mask(train_df, "seed_and_prob", moe["seed_threshold"], moe["favorite_threshold"])
            toss_bundle = fit_expert_models(train_df.loc[toss_train_mask], moe["tossup_feature_cols"], expert_type="tossup")
            chalk_bundle = fit_expert_models(train_df.loc[~toss_train_mask], moe["chalk_feature_cols"], expert_type="chalk")
            if toss_bundle is not None and chalk_bundle is not None:
                toss_test_mask = tossup_gate_mask(test_df, "seed_and_prob", moe["seed_threshold"], moe["favorite_threshold"]).to_numpy()
                if toss_test_mask.any():
                    pred[toss_test_mask] = predict_expert_models(toss_bundle, test_df.loc[toss_test_mask])
                if (~toss_test_mask).any():
                    pred[~toss_test_mask] = predict_expert_models(chalk_bundle, test_df.loc[~toss_test_mask])

        season_strategy = strategy_oof[strategy_oof["Season"] == season].copy()
        rows.append(replace_strategy_final_prob(season_strategy, test_df, pred))

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def adaptive_market_row_weights(df: pd.DataFrame, config: dict[str, object]) -> np.ndarray:
    row_weights = np.full(len(df), float(config["base_weight"]), dtype=float)
    if "AbsSeedDiff" not in df.columns:
        return np.clip(row_weights, 0.0, 0.70)

    abs_seed = pd.to_numeric(df["AbsSeedDiff"], errors="coerce").fillna(np.inf).to_numpy()
    row_weights[abs_seed <= int(config["close_seed_threshold"])] = float(config["close_weight"])
    row_weights[abs_seed >= int(config["wide_seed_threshold"])] = float(config["wide_weight"])
    return np.clip(row_weights, 0.0, 0.70)


def evaluate_adaptive_market_blend(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    eval_years: int,
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if strategy_oof.empty or "MarketProb" not in matchups.columns:
        return None

    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty or merged["MarketProb"].notna().sum() < ADAPTIVE_MARKET_MIN_ROWS:
        return None

    eval_seasons = sorted(merged["Season"].unique())[-eval_years:]
    grid_scores: dict[str, float] = {}
    config_by_key: dict[str, dict[str, object]] = {}

    for base_weight in ADAPTIVE_MARKET_BASE_WEIGHTS:
        for close_seed_threshold in ADAPTIVE_MARKET_CLOSE_THRESHOLDS:
            for wide_seed_threshold in ADAPTIVE_MARKET_WIDE_THRESHOLDS:
                if close_seed_threshold >= wide_seed_threshold:
                    continue
                for close_weight in ADAPTIVE_MARKET_CLOSE_WEIGHTS:
                    if close_weight < base_weight:
                        continue
                    for wide_weight in ADAPTIVE_MARKET_WIDE_WEIGHTS:
                        if wide_weight > base_weight:
                            continue
                        config = {
                            "base_weight": base_weight,
                            "close_seed_threshold": close_seed_threshold,
                            "wide_seed_threshold": wide_seed_threshold,
                            "close_weight": close_weight,
                            "wide_weight": wide_weight,
                        }
                        scores = []
                        active_rows_total = 0
                        for season in eval_seasons:
                            test_df = merged[merged["Season"] == season].copy()
                            if test_df.empty:
                                continue

                            pred = test_df["FinalProb"].to_numpy().copy()
                            row_weights = adaptive_market_row_weights(test_df, config)
                            market_mask = test_df["MarketProb"].notna().to_numpy()
                            active = market_mask & (row_weights > 0)
                            active_rows_total += int(active.sum())
                            if active.any():
                                market_prob = safe_clip(test_df.loc[active, "MarketProb"].to_numpy())
                                pred[active] = safe_clip(
                                    (1.0 - row_weights[active]) * pred[active] + row_weights[active] * market_prob
                                )
                            scores.append(brier_score_loss(test_df["Label"], pred))

                        if scores and active_rows_total >= ADAPTIVE_MARKET_MIN_ROWS:
                            key = json.dumps(config, sort_keys=True)
                            grid_scores[key] = float(np.mean(scores))
                            config_by_key[key] = config

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.00005:
        return None

    best_config = dict(config_by_key[best_key])
    best_config["best_cv_brier"] = best_score
    best_config["grid_scores"] = grid_scores
    return best_config


def build_adaptive_market_blend_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    adaptive_market_blend: dict[str, object],
) -> pd.DataFrame:
    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty:
        return strategy_oof

    rows = []
    for season in sorted(merged["Season"].unique()):
        test_df = merged[merged["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = test_df["FinalProb"].to_numpy().copy()
        row_weights = adaptive_market_row_weights(test_df, adaptive_market_blend)
        market_mask = test_df["MarketProb"].notna().to_numpy()
        active = market_mask & (row_weights > 0)
        if active.any():
            market_prob = safe_clip(test_df.loc[active, "MarketProb"].to_numpy())
            pred[active] = safe_clip((1.0 - row_weights[active]) * pred[active] + row_weights[active] * market_prob)

        season_strategy = strategy_oof[strategy_oof["Season"] == season].copy()
        rows.append(replace_strategy_final_prob(season_strategy, test_df, pred))

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def apply_women_chalk_rule_array(df: pd.DataFrame, prob: np.ndarray, config: dict[str, object]) -> np.ndarray:
    adjusted = safe_clip(prob)
    required = {"T1_SeedNum", "T2_SeedNum"}
    if not required.issubset(df.columns):
        return adjusted

    t1_seed = pd.to_numeric(df["T1_SeedNum"], errors="coerce").to_numpy()
    t2_seed = pd.to_numeric(df["T2_SeedNum"], errors="coerce").to_numpy()
    top_seed_max = int(config["top_seed_max"])
    dog_seed_min = int(config["dog_seed_min"])
    floor_prob = float(config["floor_prob"])
    max_round = int(config.get("max_round", 6))
    require_host_likely = bool(config.get("require_host_likely", False))

    active_mask = np.ones(len(adjusted), dtype=bool)
    if max_round < 6:
        if "TourneyRound" not in df.columns:
            return adjusted
        round_num = pd.to_numeric(df["TourneyRound"], errors="coerce").to_numpy()
        active_mask &= np.isfinite(round_num) & (round_num <= max_round)
    if require_host_likely:
        if "AnyWHostLikely" not in df.columns:
            return adjusted
        host_mask = pd.to_numeric(df["AnyWHostLikely"], errors="coerce").fillna(0).to_numpy().astype(bool)
        active_mask &= host_mask

    t1_chalk = active_mask & np.isfinite(t1_seed) & np.isfinite(t2_seed) & (t1_seed <= top_seed_max) & (t2_seed >= dog_seed_min)
    t2_chalk = active_mask & np.isfinite(t1_seed) & np.isfinite(t2_seed) & (t2_seed <= top_seed_max) & (t1_seed >= dog_seed_min)
    if t1_chalk.any():
        adjusted[t1_chalk] = np.maximum(adjusted[t1_chalk], floor_prob)
    if t2_chalk.any():
        adjusted[t2_chalk] = np.minimum(adjusted[t2_chalk], 1.0 - floor_prob)
    return safe_clip(adjusted)


def evaluate_women_chalk_extremes(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    eval_years: int,
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if strategy_oof.empty:
        return None

    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty or not {"T1_SeedNum", "T2_SeedNum"}.issubset(merged.columns):
        return None

    eval_seasons = sorted(merged["Season"].unique())[-eval_years:]
    grid_scores: dict[str, float] = {}
    configs: dict[str, dict[str, object]] = {}
    for top_seed_max in WOMEN_CHALK_TOP_SEED_MAX:
        for dog_seed_min in WOMEN_CHALK_DOG_SEED_MIN:
            if dog_seed_min <= top_seed_max:
                continue
            for floor_prob in WOMEN_CHALK_FLOOR_PROBS:
                for max_round in WOMEN_CHALK_MAX_ROUNDS:
                    for require_host_likely in WOMEN_CHALK_REQUIRE_HOST:
                        config = {
                            "top_seed_max": int(top_seed_max),
                            "dog_seed_min": int(dog_seed_min),
                            "floor_prob": float(floor_prob),
                            "max_round": int(max_round),
                            "require_host_likely": bool(require_host_likely),
                        }
                        scores = []
                        affected_rows = 0
                        for season in eval_seasons:
                            test_df = merged[merged["Season"] == season].copy()
                            if test_df.empty:
                                continue
                            pred = apply_women_chalk_rule_array(test_df, test_df["FinalProb"].to_numpy(), config)
                            affected_rows += int(np.sum(np.abs(pred - test_df["FinalProb"].to_numpy()) > 1e-12))
                            scores.append(brier_score_loss(test_df["Label"], pred))

                        if scores and affected_rows > 0:
                            key = json.dumps(config, sort_keys=True)
                            grid_scores[key] = float(np.mean(scores))
                            configs[key] = config

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.000001:
        return None

    best_config = dict(configs[best_key])
    best_config["best_cv_brier"] = best_score
    best_config["grid_scores"] = grid_scores
    return best_config


def build_women_chalk_extremes_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty:
        return strategy_oof

    rows = []
    for season in sorted(merged["Season"].unique()):
        test_df = merged[merged["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = apply_women_chalk_rule_array(test_df, test_df["FinalProb"].to_numpy(), config)
        season_strategy = strategy_oof[strategy_oof["Season"] == season].copy()
        rows.append(replace_strategy_final_prob(season_strategy, test_df, pred))

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def apply_women_chalk_rule_sequence(
    df: pd.DataFrame,
    prob: np.ndarray,
    configs: list[dict[str, object]],
) -> np.ndarray:
    adjusted = safe_clip(prob)
    for config in configs:
        adjusted = apply_women_chalk_rule_array(df, adjusted, config)
    return safe_clip(adjusted)


def evaluate_women_dual_chalk_extremes(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    eval_years: int,
    primary_config: dict[str, object],
    base_best_cv: float,
) -> Optional[dict[str, object]]:
    if strategy_oof.empty:
        return None

    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty or not {"T1_SeedNum", "T2_SeedNum"}.issubset(merged.columns):
        return None

    eval_seasons = sorted(merged["Season"].unique())[-eval_years:]
    grid_scores: dict[str, float] = {}
    configs: dict[str, dict[str, object]] = {}
    primary_floor = float(primary_config.get("floor_prob", 0.5))
    primary_top = int(primary_config.get("top_seed_max", 16))
    primary_dog = int(primary_config.get("dog_seed_min", 1))
    primary_round = int(primary_config.get("max_round", 6))

    for top_seed_max in WOMEN_DUAL_CHALK_TOP_SEED_MAX:
        if top_seed_max > primary_top:
            continue
        for dog_seed_min in WOMEN_DUAL_CHALK_DOG_SEED_MIN:
            if dog_seed_min < primary_dog:
                continue
            if dog_seed_min <= top_seed_max:
                continue
            for max_round in WOMEN_DUAL_CHALK_MAX_ROUNDS:
                if max_round > primary_round:
                    continue
                for floor_prob in WOMEN_DUAL_CHALK_FLOOR_PROBS:
                    if floor_prob < primary_floor:
                        continue
                    for require_host_likely in WOMEN_DUAL_CHALK_REQUIRE_HOST:
                        secondary_config = {
                            "top_seed_max": int(top_seed_max),
                            "dog_seed_min": int(dog_seed_min),
                            "floor_prob": float(floor_prob),
                            "max_round": int(max_round),
                            "require_host_likely": bool(require_host_likely),
                        }
                        scores = []
                        affected_rows = 0
                        for season in eval_seasons:
                            test_df = merged[merged["Season"] == season].copy()
                            if test_df.empty:
                                continue
                            pred = apply_women_chalk_rule_sequence(
                                test_df,
                                test_df["FinalProb"].to_numpy(),
                                [primary_config, secondary_config],
                            )
                            affected_rows += int(np.sum(np.abs(pred - test_df["FinalProb"].to_numpy()) > 1e-12))
                            scores.append(brier_score_loss(test_df["Label"], pred))

                        if scores and affected_rows > 0:
                            key = json.dumps(secondary_config, sort_keys=True)
                            grid_scores[key] = float(np.mean(scores))
                            configs[key] = secondary_config

    if not grid_scores:
        return None

    best_key = min(grid_scores, key=grid_scores.get)
    best_score = grid_scores[best_key]
    if best_score >= base_best_cv - 0.000001:
        return None

    return {
        "primary_config": dict(primary_config),
        "secondary_config": dict(configs[best_key]),
        "best_cv_brier": best_score,
        "grid_scores": grid_scores,
    }


def build_women_dual_chalk_extremes_oof(
    matchups: pd.DataFrame,
    strategy_oof: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    merged = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if merged.empty:
        return strategy_oof

    rows = []
    configs = [config["primary_config"], config["secondary_config"]]
    for season in sorted(merged["Season"].unique()):
        test_df = merged[merged["Season"] == season].copy()
        if test_df.empty:
            continue

        pred = apply_women_chalk_rule_sequence(test_df, test_df["FinalProb"].to_numpy(), configs)
        season_strategy = strategy_oof[strategy_oof["Season"] == season].copy()
        rows.append(replace_strategy_final_prob(season_strategy, test_df, pred))

    if not rows:
        return strategy_oof
    return pd.concat(rows, ignore_index=True)


def calibration_bin_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ProbType", "BinLow", "BinHigh", "Count", "PredMean", "ActualRate", "Brier", "CalibrationGap"])

    bins = np.linspace(0.0, 1.0, 11)
    reports = []
    for prob_type in ["RawProb", "FinalProb"]:
        work = df.copy()
        work["Bin"] = pd.cut(work[prob_type], bins=bins, include_lowest=True, right=True)
        grouped = work.groupby("Bin", observed=False)
        report = grouped.apply(
            lambda group: pd.Series(
                {
                    "Count": len(group),
                    "PredMean": group[prob_type].mean(),
                    "ActualRate": group["Label"].mean(),
                    "Brier": brier_score_loss(group["Label"], group[prob_type]) if len(group) else np.nan,
                }
            )
        ).reset_index()
        report["BinLow"] = report["Bin"].map(lambda interval: float(interval.left) if pd.notna(interval) else np.nan)
        report["BinHigh"] = report["Bin"].map(lambda interval: float(interval.right) if pd.notna(interval) else np.nan)
        report["CalibrationGap"] = (report["PredMean"] - report["ActualRate"]).abs()
        report["ProbType"] = prob_type
        reports.append(report[["ProbType", "BinLow", "BinHigh", "Count", "PredMean", "ActualRate", "Brier", "CalibrationGap"]])
    return pd.concat(reports, ignore_index=True)


def calibration_slice_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["SliceType", "SliceValue", "Count", "RawBrier", "FinalBrier", "FinalActualRate", "FinalPredMean"])

    reports = []

    if "Diag_AbsSeedDiff" in df.columns:
        work = df.copy()
        work["SliceValue"] = pd.cut(
            work["Diag_AbsSeedDiff"],
            bins=[-0.1, 1.0, 4.0, 8.0, np.inf],
            labels=["seed_0_1", "seed_2_4", "seed_5_8", "seed_9_plus"],
        ).astype(str)
        reports.append(("AbsSeedDiff", work))

    work = df.copy()
    work["SliceValue"] = pd.cut(
        work["FavoriteProb"],
        bins=[0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["fav_50_60", "fav_60_70", "fav_70_80", "fav_80_100"],
        include_lowest=True,
    ).astype(str)
    reports.append(("FavoriteProb", work))

    if "Diag_SameConference" in df.columns:
        work = df.copy()
        work["SliceValue"] = np.where(work["Diag_SameConference"] == 1, "same_conf", "cross_conf")
        reports.append(("Conference", work))

    outputs = []
    for slice_type, work in reports:
        grouped = work.groupby("SliceValue", observed=False)
        report = grouped.apply(
            lambda group: pd.Series(
                {
                    "Count": len(group),
                    "RawBrier": brier_score_loss(group["Label"], group["RawProb"]) if len(group) else np.nan,
                    "FinalBrier": brier_score_loss(group["Label"], group["FinalProb"]) if len(group) else np.nan,
                    "FinalActualRate": group["Label"].mean(),
                    "FinalPredMean": group["FinalProb"].mean(),
                }
            )
        ).reset_index()
        report["SliceType"] = slice_type
        outputs.append(report[["SliceType", "SliceValue", "Count", "RawBrier", "FinalBrier", "FinalActualRate", "FinalPredMean"]])

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(
        columns=["SliceType", "SliceValue", "Count", "RawBrier", "FinalBrier", "FinalActualRate", "FinalPredMean"]
    )


def write_calibration_diagnostics(
    run_id: str,
    gender: str,
    oof_df: pd.DataFrame,
    raw_column: str,
    calibration_method: str,
    shrinkage: float,
    eval_years: int,
) -> dict[str, str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    strategy_oof = build_strategy_oof_predictions(oof_df, raw_column, calibration_method, shrinkage, eval_years)
    if strategy_oof.empty:
        return {}

    oof_path = RESULTS_DIR / f"strategy_oof_{gender}_{run_id}.csv"
    bins_path = RESULTS_DIR / f"calibration_bins_{gender}_{run_id}.csv"
    slices_path = RESULTS_DIR / f"calibration_slices_{gender}_{run_id}.csv"
    summary_path = RESULTS_DIR / f"calibration_summary_{gender}_{run_id}.md"

    strategy_oof.to_csv(oof_path, index=False)
    calibration_bin_report(strategy_oof).to_csv(bins_path, index=False)
    slices = calibration_slice_report(strategy_oof)
    slices.to_csv(slices_path, index=False)

    raw_brier = brier_score_loss(strategy_oof["Label"], strategy_oof["RawProb"])
    final_brier = brier_score_loss(strategy_oof["Label"], strategy_oof["FinalProb"])
    worst_bins = calibration_bin_report(strategy_oof)
    worst_bins = worst_bins[worst_bins["ProbType"] == "FinalProb"].sort_values("CalibrationGap", ascending=False).head(3)
    worst_slices = slices.sort_values("FinalBrier", ascending=False).head(5)

    lines = [
        f"# {gender} Calibration Summary",
        "",
        f"- Raw column: `{raw_column}`",
        f"- Calibration: `{calibration_method}`",
        f"- Shrinkage: `{shrinkage:.2f}`",
        f"- Eval seasons: `{eval_years}`",
        f"- Raw Brier: `{raw_brier:.6f}`",
        f"- Final Brier: `{final_brier:.6f}`",
        "",
        "## Worst Final Calibration Bins",
    ]
    for row in worst_bins.itertuples(index=False):
        lines.append(
            f"- [{row.BinLow:.1f}, {row.BinHigh:.1f}] count={int(row.Count)} pred={row.PredMean:.3f} actual={row.ActualRate:.3f} gap={row.CalibrationGap:.3f}"
        )
    lines.append("")
    lines.append("## Highest Final Brier Slices")
    for row in worst_slices.itertuples(index=False):
        lines.append(
            f"- {row.SliceType}:{row.SliceValue} count={int(row.Count)} raw={row.RawBrier:.5f} final={row.FinalBrier:.5f}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "strategy_oof": str(oof_path),
        "bins": str(bins_path),
        "slices": str(slices_path),
        "summary": str(summary_path),
    }


def fit_full_models(
    matchups: pd.DataFrame,
    model_names: list[str],
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    women_minimal_feats: Optional[list[str]],
    gender: str,
    sample_weight: Optional[pd.Series | np.ndarray] = None,
) -> dict[str, object]:
    models = {}
    for name in model_names:
        feature_key = feature_key_for_model(name)
        train_subset = matchups
        sample_weight_subset = sample_weight
        if is_market_residual_model(name):
            train_mask = market_residual_train_mask(matchups)
            train_subset = matchups.loc[train_mask].copy()
            sample_weight_subset = subset_sample_weight(sample_weight, matchups.index, train_subset.index)
        x_train = feature_frame(train_subset, feature_key, lr_core_feats, lr_plus_feats, all_feats, women_minimal_feats)
        y_train = training_target_for_model(train_subset, name)
        models[name] = fit_model(name, x_train, y_train, gender, sample_weight=sample_weight_subset)
    return models


def final_cv_score(strategy_scores: dict[str, float]) -> float:
    values = [value for value in strategy_scores.values() if not np.isnan(value)]
    return float(min(values)) if values else np.nan


def base_probabilities(
    models: dict[str, object],
    pred_df: pd.DataFrame,
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
    women_minimal_feats: Optional[list[str]],
    gender: str,
    market_residual_active_pred_seasons: Optional[list[int]] = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=pred_df.index)
    for name, model in models.items():
        feature_key = feature_key_for_model(name)
        x_test = feature_frame(pred_df, feature_key, lr_core_feats, lr_plus_feats, all_feats, women_minimal_feats)
        pred = predict_model(name, model, x_test, gender)
        if is_market_residual_model(name):
            active_mask = market_residual_active_mask(pred_df, market_residual_active_pred_seasons or [])
            pred = np.where(active_mask, pred, np.nan)
        frame[f"Prob_{name}"] = pred
    return add_gender_strategy_columns(frame, list(models.keys()), gender)


def append_benchmark(run_id: str, bundle: dict[str, object], matchups: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "benchmarks.csv"
    row = pd.DataFrame(
        [
            {
                "RunID": run_id,
                "TimestampUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "Gender": bundle["gender"],
                "Matchups": len(matchups),
                "TeamFeatureCount": len(bundle["team_feats"].columns),
                "LRCoreFeatureCount": len(bundle["lr_core_feats"]),
                "LRPlusFeatureCount": len(bundle["lr_plus_feats"]),
                "AllFeatureCount": len(bundle["all_feats"]),
                "AvailableModels": json.dumps(bundle["available_models"]),
                "SelectedModels": json.dumps(bundle["selected_models"]),
                "SelectedRawColumn": bundle["selected_raw_column"],
                "CalibrationMethod": bundle["calibration_method"],
                "Shrinkage": bundle["shrinkage"],
                "HasMarketResidualOverlay": bool(bundle.get("market_residual_overlay")),
                "MarketResidualOverlayCV": bundle.get("market_residual_overlay_best_cv"),
                "MarketResidualOverlayConfig": json.dumps(bundle.get("market_residual_overlay_config", {}), sort_keys=True),
                "HasTossupSpecialist": bool(bundle.get("tossup_specialist")),
                "TossupSpecialistCV": bundle.get("tossup_specialist_best_cv"),
                "TossupSpecialistConfig": json.dumps(bundle.get("tossup_specialist_config", {}), sort_keys=True),
                "HasMoERouting": bool(bundle.get("moe_routing")),
                "MoERoutingCV": bundle.get("moe_routing_best_cv"),
                "MoERoutingConfig": json.dumps(bundle.get("moe_routing_config", {}), sort_keys=True),
                "HasAdaptiveMarketBlend": bool(bundle.get("adaptive_market_blend")),
                "AdaptiveMarketBlendCV": bundle.get("adaptive_market_blend_best_cv"),
                "AdaptiveMarketBlendConfig": json.dumps(bundle.get("adaptive_market_blend_config", {}), sort_keys=True),
                "HasWomenChalkExtremes": bool(bundle.get("women_chalk_extremes")),
                "WomenChalkExtremesCV": bundle.get("women_chalk_extremes_best_cv"),
                "WomenChalkExtremesConfig": json.dumps(bundle.get("women_chalk_extremes_config", {}), sort_keys=True),
                "HasWomenDualChalkExtremes": bool(bundle.get("women_dual_chalk_extremes")),
                "WomenDualChalkExtremesCV": bundle.get("women_dual_chalk_extremes_best_cv"),
                "WomenDualChalkExtremesConfig": json.dumps(bundle.get("women_dual_chalk_extremes_config", {}), sort_keys=True),
                "HasFieldLiveReweight": bool(bundle.get("field_reweight_info", {}).get("enabled")),
                "FieldLiveReweightSeason": bundle.get("field_reweight_info", {}).get("target_season"),
                "FieldLiveReweightFeatureCount": len(bundle.get("field_reweight_info", {}).get("used_features", [])),
                "FieldLiveReweightFeatures": json.dumps(bundle.get("field_reweight_info", {}).get("used_features", [])),
                "OfficialField2026Size": bundle.get("official_field_2026_size", 0),
                "MarketTrainingCoverage": bundle["market_training_coverage"],
                "MarketResidualModelsEnabled": bundle.get("market_residual_models_enabled", False),
                "HasPredictionOdds": bundle["has_prediction_odds"],
                "HasManualSignals": bundle["has_manual_signals"],
                "HasExternalComposite": bundle["has_external_composite"],
                "BaseScores": json.dumps(bundle["base_scores"], sort_keys=True),
                "StrategyScores": json.dumps(bundle["strategy_scores"], sort_keys=True),
                "BestCVBrier": bundle["best_cv_brier"],
            }
        ]
    )
    if path.exists():
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


def train_and_evaluate(
    gender: str,
    run_id: str,
    eval_years: int = 7,
    external_config_override: Optional[dict] = None,
) -> dict[str, object]:
    print("\n" + "=" * 64)
    print(f"Training {gender} model")
    print("=" * 64)

    external_config = load_external_config()
    if external_config_override:
        external_config = deep_update(external_config, external_config_override)
    tourney = pd.read_csv(DATA_DIR / f"{gender}NCAATourneyCompactResults.csv")
    compact = pd.read_csv(DATA_DIR / f"{gender}RegularSeasonCompactResults.csv")
    seeds_raw = pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")
    elo_params, elo_optuna_summary = optimize_elo_params(gender, compact, seeds_raw, tourney, external_config, run_id)
    team_feats = build_team_features(gender, elo_params=elo_params)
    if gender == "W":
        team_feats = rollback_women_experimental_features(team_feats)
    feature_ablations = resolve_feature_ablation_groups(external_config, gender)
    team_feats, applied_feature_ablations = apply_feature_ablations(team_feats, gender, feature_ablations)
    matchups, feature_candidates, lr_core_feats, lr_plus_feats, all_feats = build_matchup_df(tourney, team_feats, gender)
    if gender == "W" and resolve_women_feature_simplify_enabled(external_config):
        feature_candidates, lr_core_feats, lr_plus_feats, all_feats = simplify_women_feature_lists(
            feature_candidates,
            lr_core_feats,
            lr_plus_feats,
            all_feats,
        )
    women_minimal_feats = build_women_minimal_feature_list(matchups) if gender == "W" else []
    field_context = build_field_reweight_context(gender, team_feats, feature_candidates)
    has_live_field_2026 = bool(field_context.get("field_team_ids", {}).get(2026))
    market_mode = resolve_market_mode(external_config, gender)
    market_df = pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence"])
    if market_mode != "no_market_all":
        market_df = load_matchup_market_odds(gender)
    manual_df = load_manual_signals(gender)
    matchups, market_feats, market_coverage = merge_market_features(matchups, market_df)
    if market_feats and market_coverage >= float(external_config["min_market_coverage_for_training"]):
        all_feats = list(dict.fromkeys(all_feats + market_feats))
        for feature_list in [lr_core_feats, lr_plus_feats]:
            for column in market_feats:
                if column not in feature_list:
                    feature_list.append(column)

    residual_enabled = False
    market_residual_info = {
        "coverage_by_season": {},
        "active_seasons": [],
        "eligible_eval_seasons": [],
        "matched_rows_in_eval_window": 0,
        "eval_window_seasons": [],
    }
    if market_mode != "no_market_all" and market_feats:
        residual_enabled, market_residual_info = recent_market_residual_ready(
            matchups,
            eval_years=eval_years,
            min_coverage=float(external_config.get("min_market_coverage_for_residual_models", 0.85)),
        )
    specs = available_model_specs(gender, enable_market_residual=residual_enabled)
    model_names = [spec.name for spec in specs]
    oof_df = generate_base_oof(
        matchups,
        specs,
        lr_core_feats,
        lr_plus_feats,
        all_feats,
        women_minimal_feats,
        gender,
        field_context=field_context if has_live_field_2026 else None,
        external_config=external_config,
    )
    base_scores = evaluate_base_scores(oof_df, model_names, eval_years=eval_years)
    selected_models = select_model_names(base_scores, gender)
    forced_selected_models = [name for name in resolve_gender_override_list(external_config, "forced_selected_models", gender) if name in model_names]
    women_model_set_choice = None
    if forced_selected_models:
        selected_models = forced_selected_models
    elif gender == "W":
        women_model_set_choice = choose_women_model_set(oof_df, selected_models, model_names, eval_years)
        if women_model_set_choice is not None:
            selected_models = list(women_model_set_choice["model_names"])
    diag_columns = [column for column in ["Diag_AbsSeedDiff", "Diag_T1BetterSeed", "Diag_SameConference"] if column in oof_df.columns]
    oof_selected, final_meta = add_ensemble_oof(
        oof_df[["Season", "T1", "T2", "Label"] + diag_columns + [f"Prob_{name}" for name in selected_models]].copy(),
        selected_models,
        gender,
    )
    selected_raw_column, calibration_method, shrinkage, strategy_scores = evaluate_strategy_grid(oof_selected, eval_years=eval_years)
    forced_strategy = resolve_gender_override_dict(external_config, "forced_strategy", gender)
    if forced_strategy:
        forced_raw_column = str(forced_strategy.get("raw_column", selected_raw_column))
        if forced_raw_column in oof_selected.columns:
            selected_raw_column = forced_raw_column
        calibration_method = str(forced_strategy.get("calibration", calibration_method))
        shrinkage = float(forced_strategy.get("shrinkage", shrinkage))
        forced_strategy_oof = build_strategy_oof_predictions(
            oof_selected,
            selected_raw_column,
            calibration_method,
            shrinkage,
            eval_years=eval_years,
        )
        if not forced_strategy_oof.empty:
            strategy_scores = dict(strategy_scores)
            strategy_key = f"{selected_raw_column}|{calibration_method}|shrink={shrinkage:.2f}"
            strategy_scores[strategy_key] = float(brier_score_loss(forced_strategy_oof["Label"], forced_strategy_oof["FinalProb"]))

    calibrator = None
    if calibration_method != "none":
        calibrator = fit_calibrator(
            calibration_method,
            oof_selected[selected_raw_column].dropna().to_numpy(),
            oof_selected.loc[oof_selected[selected_raw_column].notna(), "Label"].to_numpy(),
        )

    final_train_weights, field_reweight_info = compute_field_reweight_weights(
        matchups,
        2026,
        field_context,
        external_config,
        gender,
    )
    final_models = fit_full_models(
        matchups,
        selected_models,
        lr_core_feats,
        lr_plus_feats,
        all_feats,
        women_minimal_feats,
        gender,
        sample_weight=final_train_weights,
    )
    best_cv = final_cv_score(strategy_scores)
    tossup_specialist = None
    moe_routing = None
    strategy_oof = None
    market_residual_overlay = None
    adaptive_market_blend = None
    women_chalk_extremes = None
    women_dual_chalk_extremes = None
    if gender == "M":
        strategy_oof = build_strategy_oof_predictions(oof_selected, selected_raw_column, calibration_method, shrinkage, eval_years=None)
        market_residual_overlay = evaluate_market_residual_overlay(
            matchups,
            strategy_oof,
            lr_core_feats,
            lr_plus_feats,
            all_feats,
            gender,
            eval_years=eval_years,
            base_best_cv=best_cv,
        )
        if market_residual_overlay is not None:
            best_cv = float(market_residual_overlay["best_cv_brier"])
            strategy_oof = build_market_residual_overlay_oof(
                matchups,
                strategy_oof,
                market_residual_overlay,
                lr_core_feats,
                lr_plus_feats,
                all_feats,
                gender,
            )
        post_overlay_strategy_oof = strategy_oof.copy()
        tossup_specialist = evaluate_tossup_specialist(matchups, post_overlay_strategy_oof, eval_years=eval_years, base_best_cv=best_cv)
        if tossup_specialist is not None:
            best_cv = float(tossup_specialist["best_cv_brier"])
        moe_routing = evaluate_moe_routing(matchups, post_overlay_strategy_oof, eval_years=eval_years, base_best_cv=best_cv)
        if moe_routing is not None:
            best_cv = float(moe_routing["best_cv_brier"])
            tossup_specialist = None
            strategy_oof = build_moe_routing_oof(matchups, post_overlay_strategy_oof, moe_routing)
        elif tossup_specialist is not None:
            strategy_oof = build_tossup_specialist_oof(matchups, post_overlay_strategy_oof, tossup_specialist)

        adaptive_market_blend = evaluate_adaptive_market_blend(
            matchups,
            strategy_oof,
            eval_years=eval_years,
            base_best_cv=best_cv,
        )
        if adaptive_market_blend is not None:
            best_cv = float(adaptive_market_blend["best_cv_brier"])
            strategy_oof = build_adaptive_market_blend_oof(matchups, strategy_oof, adaptive_market_blend)
    if gender == "W":
        strategy_oof = build_strategy_oof_predictions(oof_selected, selected_raw_column, calibration_method, shrinkage, eval_years=None)
        if resolve_women_chalk_extremes_enabled(external_config):
            women_chalk_extremes = evaluate_women_chalk_extremes(
                matchups,
                strategy_oof,
                eval_years=eval_years,
                base_best_cv=best_cv,
            )
            if women_chalk_extremes is not None:
                best_cv = float(women_chalk_extremes["best_cv_brier"])
                strategy_oof = build_women_chalk_extremes_oof(matchups, strategy_oof, women_chalk_extremes)
                women_dual_chalk_extremes = evaluate_women_dual_chalk_extremes(
                    matchups,
                    strategy_oof,
                    eval_years=eval_years,
                    primary_config=women_chalk_extremes,
                    base_best_cv=best_cv,
                )
                if women_dual_chalk_extremes is not None:
                    best_cv = float(women_dual_chalk_extremes["best_cv_brier"])
                    strategy_oof = build_women_dual_chalk_extremes_oof(matchups, strategy_oof, women_dual_chalk_extremes)

    print(f"Training samples: {len(matchups)}")
    print(f"Elo params: {elo_params}")
    if elo_optuna_summary is not None:
        print(
            f"Elo Optuna proxy CV: {elo_optuna_summary['best_cv_brier']:.5f} "
            f"trials={elo_optuna_summary['trials']} eval_years={elo_optuna_summary['eval_years']}"
        )
    if gender == "W":
        print("Women rollback: experimental adjusted-efficiency and Elo-trend features disabled")
    if applied_feature_ablations:
        print(f"Feature ablations: {applied_feature_ablations}")
    if market_mode != "default":
        print(f"Market mode: {market_mode}")
    if forced_selected_models:
        print(f"Forced selected models: {forced_selected_models}")
    elif women_model_set_choice is not None:
        print(
            "Women curated model set: "
            f"{women_model_set_choice['model_names']} "
            f"raw={women_model_set_choice['raw_column']} "
            f"cal={women_model_set_choice['calibration_method']} "
            f"shrink={women_model_set_choice['shrinkage']:.2f} "
            f"cv={women_model_set_choice['best_cv_brier']:.5f}"
        )
    if forced_strategy:
        print(f"Forced strategy: {forced_strategy}")
    print(f"Feature candidates ({len(feature_candidates)}): {feature_candidates}")
    print(f"LR core features ({len(lr_core_feats)}): {lr_core_feats}")
    print(f"LR plus features ({len(lr_plus_feats)}): {lr_plus_feats}")
    print(f"All matchup features ({len(all_feats)}): {all_feats}")
    if gender == "W":
        print(f"Women minimal tree features ({len(women_minimal_feats)}): {women_minimal_feats}")
    print(f"Available models: {model_names}")
    print(
        f"Field live reweight: ready_2026={has_live_field_2026} "
        f"enabled={field_reweight_info['enabled']} "
        f"target={field_reweight_info['target_season']} "
        f"features={field_reweight_info['used_features']}"
    )
    print(f"Market odds coverage: {market_coverage:.1%}")
    print(f"Market residual models enabled: {residual_enabled}")
    if market_residual_info["active_seasons"]:
        print(f"Market residual active seasons: {market_residual_info['active_seasons']}")
    if market_residual_info["eligible_eval_seasons"]:
        print(
            "Market residual eligible eval seasons: "
            f"{market_residual_info['eligible_eval_seasons']} "
            f"matched_rows={market_residual_info['matched_rows_in_eval_window']}"
        )
    print(f"Base CV Brier: {base_scores}")
    print(f"Selected models: {selected_models}")
    print(f"Best strategy: {selected_raw_column} + {calibration_method} + shrink={shrinkage:.2f} -> {best_cv:.5f}")
    if market_residual_overlay is not None:
        print(
            "Market residual overlay: "
            f"{market_residual_overlay['model_name']} "
            f"blend={market_residual_overlay['blend_weight']:.2f} "
            f"-> {market_residual_overlay['best_cv_brier']:.5f}"
        )
    if tossup_specialist is not None:
        print(
            "Tossup specialist: "
            f"{tossup_specialist['mode']} seed<={tossup_specialist['seed_threshold']} "
            f"fav<={tossup_specialist['favorite_threshold']:.2f} "
            f"blend={tossup_specialist['blend_weight']:.2f} "
            f"-> {tossup_specialist['best_cv_brier']:.5f}"
        )
    if moe_routing is not None:
        print(
            "MoE routing: "
            f"seed<={moe_routing['seed_threshold']} "
            f"fav<={moe_routing['favorite_threshold']:.2f} "
            f"-> {moe_routing['best_cv_brier']:.5f}"
        )
    if adaptive_market_blend is not None:
        print(
            "Adaptive market blend: "
            f"base={adaptive_market_blend['base_weight']:.2f} "
            f"close(seed<={adaptive_market_blend['close_seed_threshold']})={adaptive_market_blend['close_weight']:.2f} "
            f"wide(seed>={adaptive_market_blend['wide_seed_threshold']})={adaptive_market_blend['wide_weight']:.2f} "
            f"-> {adaptive_market_blend['best_cv_brier']:.5f}"
        )
    if women_chalk_extremes is not None:
        print(
            "Women chalk extremes: "
            f"top_seed<={women_chalk_extremes['top_seed_max']} "
            f"dog_seed>={women_chalk_extremes['dog_seed_min']} "
            f"max_round<={women_chalk_extremes.get('max_round', 6)} "
            f"host={bool(women_chalk_extremes.get('require_host_likely', False))} "
            f"floor={women_chalk_extremes['floor_prob']:.3f} "
            f"-> {women_chalk_extremes['best_cv_brier']:.5f}"
        )
    if women_dual_chalk_extremes is not None:
        secondary = women_dual_chalk_extremes["secondary_config"]
        print(
            "Women dual chalk: "
            f"top_seed<={secondary['top_seed_max']} "
            f"dog_seed>={secondary['dog_seed_min']} "
            f"max_round<={secondary.get('max_round', 6)} "
            f"host={bool(secondary.get('require_host_likely', False))} "
            f"floor={secondary['floor_prob']:.3f} "
            f"-> {women_dual_chalk_extremes['best_cv_brier']:.5f}"
        )

    coef_model_name = "lr_core" if "lr_core" in final_models else ("lr_plus" if "lr_plus" in final_models else None)
    coef_features = lr_core_feats if coef_model_name == "lr_core" else lr_plus_feats
    if coef_model_name is not None and coef_features:
        poly = final_models[coef_model_name].named_steps["poly"]
        lr_step = final_models[coef_model_name].named_steps["lr"]
        feature_names = poly.get_feature_names_out(coef_features)
        ranked = sorted(zip(feature_names, lr_step.coef_[0]), key=lambda item: abs(item[1]), reverse=True)
        print("Top LR coefficients:")
        for name, coef in ranked[:12]:
            print(f"  {name}: {coef:+.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    oof_selected.to_csv(RESULTS_DIR / f"oof_{gender}_{run_id}.csv", index=False)
    diagnostic_paths = (
        write_calibration_diagnostics(run_id, gender, oof_selected, selected_raw_column, calibration_method, shrinkage, eval_years)
        if gender == "M"
        else {}
    )

    bundle = {
        "gender": gender,
        "team_feats": team_feats,
        "elo_params": elo_params,
        "elo_optuna_summary": {} if elo_optuna_summary is None else elo_optuna_summary,
        "feature_ablations": applied_feature_ablations,
        "market_mode": market_mode,
        "feature_candidates": feature_candidates,
        "lr_core_feats": lr_core_feats,
        "lr_plus_feats": lr_plus_feats,
        "all_feats": all_feats,
        "women_minimal_feats": women_minimal_feats,
        "external_config": external_config,
        "available_models": model_names,
        "selected_models": selected_models,
        "women_model_set_choice": {} if women_model_set_choice is None else women_model_set_choice,
        "models": final_models,
        "meta_model": final_meta,
        "selected_raw_column": selected_raw_column,
        "calibration_method": calibration_method,
        "shrinkage": shrinkage,
        "calibrator": calibrator,
        "market_df": market_df,
        "market_training_coverage": market_coverage,
        "market_residual_models_enabled": residual_enabled,
        "market_residual_info": market_residual_info,
        "manual_df": manual_df,
        "field_context": field_context,
        "field_reweight_info": field_reweight_info,
        "official_field_2026_size": len(field_context.get("field_team_ids", {}).get(2026, set())),
        "has_live_field_2026": has_live_field_2026,
        "has_prediction_odds": bool((market_df["Season"] == 2026).any()) if not market_df.empty else False,
        "has_manual_signals": bool((manual_df["Season"] == 2026).any()) if not manual_df.empty else False,
        "has_external_composite": "ExtCompositeStrength" in team_feats.columns,
        "matchup_count": len(matchups),
        "base_scores": base_scores,
        "strategy_scores": strategy_scores,
        "best_cv_brier": best_cv,
        "diagnostic_paths": diagnostic_paths,
        "market_residual_overlay": market_residual_overlay,
        "market_residual_overlay_best_cv": None if market_residual_overlay is None else market_residual_overlay["best_cv_brier"],
        "market_residual_overlay_config": {} if market_residual_overlay is None else {
            key: market_residual_overlay[key]
            for key in ["model_name", "feature_key", "blend_weight"]
        },
        "tossup_specialist": tossup_specialist,
        "tossup_specialist_best_cv": None if tossup_specialist is None else tossup_specialist["best_cv_brier"],
        "tossup_specialist_config": {} if tossup_specialist is None else {
            key: tossup_specialist[key]
            for key in ["mode", "seed_threshold", "favorite_threshold", "blend_weight"]
        },
        "moe_routing": moe_routing,
        "moe_routing_best_cv": None if moe_routing is None else moe_routing["best_cv_brier"],
        "moe_routing_config": {} if moe_routing is None else {
            key: moe_routing[key]
            for key in ["mode", "seed_threshold", "favorite_threshold"]
        },
        "adaptive_market_blend": adaptive_market_blend,
        "adaptive_market_blend_best_cv": None if adaptive_market_blend is None else adaptive_market_blend["best_cv_brier"],
        "adaptive_market_blend_config": {} if adaptive_market_blend is None else {
            key: adaptive_market_blend[key]
            for key in ["base_weight", "close_seed_threshold", "close_weight", "wide_seed_threshold", "wide_weight"]
        },
        "women_chalk_extremes": women_chalk_extremes,
        "women_chalk_extremes_best_cv": None if women_chalk_extremes is None else women_chalk_extremes["best_cv_brier"],
        "women_chalk_extremes_config": {} if women_chalk_extremes is None else {
            key: women_chalk_extremes[key]
            for key in ["top_seed_max", "dog_seed_min", "floor_prob", "max_round", "require_host_likely"]
        },
        "women_dual_chalk_extremes": women_dual_chalk_extremes,
        "women_dual_chalk_extremes_best_cv": None if women_dual_chalk_extremes is None else women_dual_chalk_extremes["best_cv_brier"],
        "women_dual_chalk_extremes_config": {} if women_dual_chalk_extremes is None else {
            "primary_config": women_dual_chalk_extremes["primary_config"],
            "secondary_config": women_dual_chalk_extremes["secondary_config"],
        },
    }
    append_benchmark(run_id, bundle, matchups)
    if diagnostic_paths:
        print(f"Diagnostics: {diagnostic_paths['summary']}")
    return bundle


def strategy_probabilities(bundle: dict[str, object], base_prob_df: pd.DataFrame) -> np.ndarray:
    raw_column = bundle["selected_raw_column"]
    if raw_column.startswith("Prob_") and raw_column not in {"ProbMean", "ProbStack"}:
        prob = resolved_strategy_probabilities(base_prob_df, raw_column)
    elif raw_column == "ProbMean":
        prob = rowwise_prob_fallback(base_prob_df)
    elif raw_column == "ProbStack":
        meta_input = meta_feature_frame(base_prob_df[[f"Prob_{name}" for name in bundle["selected_models"]]].copy(), bundle["selected_models"])
        prob = safe_clip(bundle["meta_model"].predict_proba(meta_input)[:, 1])
    else:
        raise ValueError(f"Unknown strategy column: {raw_column}")
    return shrink_toward_half(
        apply_calibrator(prob, bundle["calibration_method"], bundle["calibrator"]),
        bundle.get("shrinkage", 0.0),
    )


def build_prediction_frame(
    sub_df: pd.DataFrame,
    team_feats: pd.DataFrame,
    feature_candidates: list[str],
    season: int,
    gender: str,
    market_df: Optional[pd.DataFrame] = None,
    signal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frame = sub_df.copy()
    ids = frame["ID"].str.split("_", expand=True)
    frame["Season"] = ids[0].astype(int)
    frame["T1"] = ids[1].astype(int)
    frame["T2"] = ids[2].astype(int)
    frame = frame[frame["Season"] == season].copy()

    if gender == "M":
        frame = frame[frame["T1"] < 2000].copy()
    else:
        frame = frame[frame["T1"] >= 3000].copy()

    season_feats = team_feats[team_feats["Season"] == season].copy()
    t1f = season_feats.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2f = season_feats.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    frame = frame.merge(t1f, on=["Season", "T1"], how="left")
    frame = frame.merge(t2f, on=["Season", "T2"], how="left")
    frame, _ = compute_diff_features(frame, feature_candidates)
    frame, _ = add_matchup_context_features(frame)
    if gender == "W":
        frame, _ = add_women_tourney_structure_features(frame)
    if market_df is not None and not market_df.empty:
        frame, _, _ = merge_market_features(frame, market_df)
    if signal_df is not None and not signal_df.empty:
        frame = merge_team_signal_features(frame, signal_df)
    return frame


def postprocess_predictions(bundle: dict[str, object], pred_df: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    adjusted = safe_clip(prob)
    config = bundle["external_config"]
    gender = bundle["gender"]
    skip_market_blend_mask = bundle.get("_skip_market_blend_mask")

    if "D_Signal_ManualComposite" in pred_df.columns:
        manual_weight = float(config["manual_signal_logit_weight"].get(gender, 0.0))
        if manual_weight > 0:
            manual_shift = manual_weight * pred_df["D_Signal_ManualComposite"].fillna(0.0).to_numpy()
            adjusted = apply_logit_shift(adjusted, manual_shift)

    if "D_ExtCompositeStrength" in pred_df.columns:
        ext_weight = float(config["external_rating_logit_weight"].get(gender, 0.0))
        if ext_weight > 0:
            ext_shift = ext_weight * pred_df["D_ExtCompositeStrength"].fillna(0.0).to_numpy()
            adjusted = apply_logit_shift(adjusted, ext_shift)

    if "MarketProb" in pred_df.columns:
        adaptive_market_blend = bundle.get("adaptive_market_blend")
        market_weight = float(config["market_blend_weight"].get(gender, 0.0))
        extra_weight = float(config.get("field_market_extra_weight", {}).get(gender, 0.0))
        market_mask = pred_df["MarketProb"].notna().to_numpy()
        if market_mask.any():
            if adaptive_market_blend:
                row_weights = adaptive_market_row_weights(pred_df, adaptive_market_blend)
            else:
                row_weights = np.full(len(pred_df), market_weight, dtype=float)
            official_field_ids = bundle.get("field_context", {}).get("field_team_ids", {}).get(2026, set())
            if extra_weight > 0 and official_field_ids:
                field_mask = official_field_matchup_mask(pred_df, official_field_ids, season=2026)
                row_weights[field_mask] = np.clip(row_weights[field_mask] + extra_weight, 0.0, 0.70)
            active = market_mask & (row_weights > 0)
            if isinstance(skip_market_blend_mask, np.ndarray) and len(skip_market_blend_mask) == len(pred_df):
                active &= ~skip_market_blend_mask
            if active.any():
                market_prob = safe_clip(pred_df.loc[active, "MarketProb"].to_numpy())
                adjusted[active] = safe_clip((1.0 - row_weights[active]) * adjusted[active] + row_weights[active] * market_prob)

    if gender == "W" and bundle.get("women_chalk_extremes"):
        adjusted = apply_women_chalk_rule_array(pred_df, adjusted, bundle["women_chalk_extremes"])
    if gender == "W" and bundle.get("women_dual_chalk_extremes"):
        adjusted = apply_women_chalk_rule_array(pred_df, adjusted, bundle["women_dual_chalk_extremes"]["secondary_config"])

    return safe_clip(adjusted)


def predict_bundle(bundle: dict[str, object], pred_df: pd.DataFrame) -> np.ndarray:
    pred_market_coverage = market_coverage_by_season(pred_df)
    pred_market_active_seasons = market_residual_active_seasons(
        pred_market_coverage,
        float(bundle["external_config"].get("min_market_coverage_for_residual_models", MARKET_RESIDUAL_SELECTION_MIN_COVERAGE)),
    )
    base_df = base_probabilities(
        bundle["models"],
        pred_df,
        bundle["lr_core_feats"],
        bundle["lr_plus_feats"],
        bundle["all_feats"],
        bundle.get("women_minimal_feats"),
        bundle["gender"],
        market_residual_active_pred_seasons=pred_market_active_seasons,
    )
    prob = strategy_probabilities(bundle, base_df)
    prob, residual_overlay_mask = apply_market_residual_overlay(bundle, pred_df, prob)
    prob = apply_moe_routing(bundle, pred_df, prob)
    prob = apply_tossup_specialist(bundle, pred_df, prob)
    bundle["_skip_market_blend_mask"] = residual_overlay_mask
    try:
        return postprocess_predictions(bundle, pred_df, prob)
    finally:
        bundle.pop("_skip_market_blend_mask", None)


def predict_for_season(sub_template_path: Path, season: int, men_bundle: dict[str, object], women_bundle: dict[str, object]) -> pd.DataFrame:
    sub = pd.read_csv(sub_template_path)
    men_frame = build_prediction_frame(
        sub,
        men_bundle["team_feats"],
        men_bundle["feature_candidates"],
        season,
        "M",
        market_df=men_bundle["market_df"],
        signal_df=men_bundle["manual_df"],
    )
    women_frame = build_prediction_frame(
        sub,
        women_bundle["team_feats"],
        women_bundle["feature_candidates"],
        season,
        "W",
        market_df=women_bundle["market_df"],
        signal_df=women_bundle["manual_df"],
    )

    men_pred = pd.DataFrame({"ID": men_frame["ID"].values, "Pred": predict_bundle(men_bundle, men_frame)})
    women_pred = pd.DataFrame({"ID": women_frame["ID"].values, "Pred": predict_bundle(women_bundle, women_frame)})
    return pd.concat([men_pred, women_pred], ignore_index=True)


def save_submission_copy(run_id: str, filename: str, df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{filename.rsplit('.', 1)[0]}_{run_id}.csv", index=False)


def combined_cv_summary(men_bundle: dict[str, object], women_bundle: dict[str, object]) -> dict[str, float]:
    men_brier = float(men_bundle["best_cv_brier"])
    women_brier = float(women_bundle["best_cv_brier"])
    men_matchups = int(men_bundle.get("matchup_count", 0))
    women_matchups = int(women_bundle.get("matchup_count", 0))

    total_matchups = men_matchups + women_matchups
    matchup_weighted = (
        (men_brier * men_matchups + women_brier * women_matchups) / total_matchups
        if total_matchups > 0
        else np.nan
    )
    return {
        "men_brier": men_brier,
        "women_brier": women_brier,
        "equal_gender_mean": (men_brier + women_brier) / 2.0,
        "historical_matchup_weighted": matchup_weighted,
        "men_matchups": men_matchups,
        "women_matchups": women_matchups,
    }


def write_combined_cv_summary(run_id: str, summary: dict[str, float]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"combined_cv_summary_{run_id}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    men_bundle = train_and_evaluate("M", run_id=run_id, eval_years=7)
    women_bundle = train_and_evaluate("W", run_id=run_id, eval_years=7)
    combined_summary = combined_cv_summary(men_bundle, women_bundle)
    combined_summary_path = write_combined_cv_summary(run_id, combined_summary)

    print("\n" + "=" * 64)
    print(f"Final CV Brier  M={men_bundle['best_cv_brier']:.5f}  W={women_bundle['best_cv_brier']:.5f}")
    print(
        "Combined CV    "
        f"equal_gender={combined_summary['equal_gender_mean']:.5f}  "
        f"matchup_weighted={combined_summary['historical_matchup_weighted']:.5f}"
    )
    print("=" * 64)

    stage1_path = DATA_DIR / "SampleSubmissionStage1.csv"
    stage2_path = DATA_DIR / "SampleSubmissionStage2.csv"

    stage1 = pd.read_csv(stage1_path)
    stage1_seasons = sorted(stage1["ID"].str.split("_").str[0].astype(int).unique())
    stage1_parts = []
    print("\nGenerating Stage 1 submission...")
    for season in stage1_seasons:
        part = predict_for_season(stage1_path, season, men_bundle, women_bundle)
        stage1_parts.append(part)
        print(f"  Season {season}: {len(part)} rows")

    stage1_sub = pd.concat(stage1_parts, ignore_index=True).sort_values("ID").reset_index(drop=True)
    stage1_sub.to_csv("submission_stage1.csv", index=False)
    save_submission_copy(run_id, "submission_stage1.csv", stage1_sub)
    print(f"Stage 1 saved: {len(stage1_sub)} rows -> submission_stage1.csv")

    print("\nGenerating Stage 2 submission...")
    stage2_sub = predict_for_season(stage2_path, 2026, men_bundle, women_bundle)
    stage2_sub = stage2_sub.sort_values("ID").reset_index(drop=True)
    stage2_sub.to_csv("submission_stage2.csv", index=False)
    save_submission_copy(run_id, "submission_stage2.csv", stage2_sub)
    print(f"Stage 2 saved: {len(stage2_sub)} rows -> submission_stage2.csv")

    print("\nSanity checks:")
    print(f"  Pred range: [{stage2_sub['Pred'].min():.4f}, {stage2_sub['Pred'].max():.4f}]")
    print(f"  Pred mean:  {stage2_sub['Pred'].mean():.4f}")
    print(f"  NaN count:  {stage2_sub['Pred'].isna().sum()}")
    print(stage2_sub.head(5).to_string(index=False))
    print(f"\nArtifacts saved under: {RESULTS_DIR}")
    print(f"Combined CV summary: {combined_summary_path}")


if __name__ == "__main__":
    main()
