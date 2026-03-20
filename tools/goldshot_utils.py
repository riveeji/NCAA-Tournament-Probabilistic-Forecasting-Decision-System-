from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"
RESULTS_DIR = ROOT / "results"


ROUND_LABELS = {
    0: "PlayIn",
    1: "RoundOf64",
    2: "RoundOf32",
    3: "Sweet16",
    4: "Elite8",
    5: "FinalFour",
    6: "Championship",
}


def safe_clip_prob(values: pd.Series | np.ndarray | float, eps: float = 1e-6) -> pd.Series | np.ndarray | float:
    if isinstance(values, pd.Series):
        return values.clip(lower=eps, upper=1.0 - eps)
    if isinstance(values, np.ndarray):
        return np.clip(values, eps, 1.0 - eps)
    return float(np.clip(float(values), eps, 1.0 - eps))


def load_team_maps(data_dir: Path | None = None) -> tuple[dict[int, str], dict[int, str]]:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    men = pd.read_csv(root / "MTeams.csv", usecols=["TeamID", "TeamName"])
    women = pd.read_csv(root / "WTeams.csv", usecols=["TeamID", "TeamName"])
    men_map = dict(zip(men["TeamID"].astype(int), men["TeamName"].astype(str)))
    women_map = dict(zip(women["TeamID"].astype(int), women["TeamName"].astype(str)))
    return men_map, women_map


def load_seed_details(gender: str, season: int, data_dir: Path | None = None) -> pd.DataFrame:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    path = root / f"{gender}NCAATourneySeeds.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "Seed", "SeedNum", "SeedRegion", "IsPlayInSeed"])
    frame = pd.read_csv(path, usecols=["Season", "Seed", "TeamID"])
    frame = frame.loc[pd.to_numeric(frame["Season"], errors="coerce").eq(int(season))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "Seed", "SeedNum", "SeedRegion", "IsPlayInSeed"])
    frame["Season"] = int(season)
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["TeamID"]).copy()
    frame["TeamID"] = frame["TeamID"].astype(int)
    frame["Seed"] = frame["Seed"].astype(str)
    frame["SeedNum"] = pd.to_numeric(frame["Seed"].str.extract(r"(\d+)")[0], errors="coerce")
    frame["SeedRegion"] = frame["Seed"].str[0]
    frame["IsPlayInSeed"] = frame["Seed"].str.contains(r"[ab]$", case=False, regex=True)
    return frame[["Season", "TeamID", "Seed", "SeedNum", "SeedRegion", "IsPlayInSeed"]].drop_duplicates()


def load_men_watchlist(season: int, results_dir: Path | None = None) -> pd.DataFrame:
    root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = root / f"availability_watchlist_{season}_men.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["TeamID"]).copy()
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def favorite_side_from_market(market_prob: float | None, spread: float | None) -> str:
    if market_prob is not None and pd.notna(market_prob):
        if float(market_prob) > 0.5:
            return "T1"
        if float(market_prob) < 0.5:
            return "T2"
    if spread is not None and pd.notna(spread):
        if float(spread) < 0:
            return "T1"
        if float(spread) > 0:
            return "T2"
    return "Unknown"


def market_and_spread_consistent(market_prob: float | None, spread: float | None) -> bool:
    if market_prob is None or spread is None or pd.isna(market_prob) or pd.isna(spread):
        return False
    market_favorite = favorite_side_from_market(float(market_prob), None)
    spread_favorite = favorite_side_from_market(None, float(spread))
    return market_favorite != "Unknown" and market_favorite == spread_favorite


def model_aligned_with_market(model_prob: float | None, market_prob: float | None) -> bool:
    if model_prob is None or market_prob is None or pd.isna(model_prob) or pd.isna(market_prob):
        return False
    if float(model_prob) == 0.5 or float(market_prob) == 0.5:
        return False
    return (float(model_prob) > 0.5) == (float(market_prob) > 0.5)


def is_extreme_game(market_prob: float | None, spread: float | None, model_prob: float | None) -> bool:
    if market_prob is not None and pd.notna(market_prob):
        if float(market_prob) >= 0.80 or float(market_prob) <= 0.20:
            return True
    if spread is not None and pd.notna(spread):
        if abs(float(spread)) >= 6.0:
            return True
    if model_prob is not None and pd.notna(model_prob):
        if float(model_prob) >= 0.80 or float(model_prob) <= 0.20:
            return True
    return False


def favorite_direction_change(current_prob: float, new_prob: float, favorite_side: str) -> bool:
    if favorite_side == "T1":
        return float(new_prob) > float(current_prob)
    if favorite_side == "T2":
        return float(new_prob) < float(current_prob)
    return False


def apply_override_guardrails(
    current_prob: float,
    target_prob: float,
    weight: float,
    *,
    market_prob: float | None = None,
    spread: float | None = None,
    model_prob: float | None = None,
    max_abs_change: float = 0.12,
) -> float:
    current = float(safe_clip_prob(current_prob))
    target = float(safe_clip_prob(target_prob))
    proposed = current + float(weight) * (target - current)
    delta = float(np.clip(proposed - current, -max_abs_change, max_abs_change))
    bounded = float(safe_clip_prob(current + delta))
    if not is_extreme_game(market_prob, spread, model_prob):
        if current > 0.5 and bounded < 0.5:
            return 0.500001
        if current < 0.5 and bounded > 0.5:
            return 0.499999
    return bounded


@lru_cache(maxsize=8)
def build_official_round_map(gender: str, season: int) -> pd.DataFrame:
    seeds_path = DATA_DIR / f"{gender}NCAATourneySeeds.csv"
    slots_path = DATA_DIR / f"{gender}NCAATourneySlots.csv"
    if not seeds_path.exists() or not slots_path.exists():
        return pd.DataFrame(columns=["Season", "T1", "T2", "OfficialMinRound", "OfficialRoundLabel"])

    seeds_raw = pd.read_csv(seeds_path)
    slots = pd.read_csv(slots_path)
    seeds_raw = seeds_raw.loc[pd.to_numeric(seeds_raw["Season"], errors="coerce").eq(int(season))].copy()
    slots = slots.loc[pd.to_numeric(slots["Season"], errors="coerce").eq(int(season))].copy()
    if seeds_raw.empty or slots.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "OfficialMinRound", "OfficialRoundLabel"])

    seed_to_team = {
        str(seed): int(team_id)
        for seed, team_id in seeds_raw[["Seed", "TeamID"]].dropna().itertuples(index=False, name=None)
    }
    children = slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict("index")
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
        memo[key] = {int(team_id)} if team_id is not None else set()
        return memo[key]

    pair_rounds: dict[tuple[int, int], int] = {}
    for row in slots.itertuples(index=False):
        slot = str(row.Slot)
        round_num: int | None = None
        match = re.match(r"^R(\d+)", slot)
        if match is not None:
            round_num = int(match.group(1))
        elif re.match(r"^[WXYZ]\d{2}$", slot):
            round_num = 0
        if round_num is None:
            continue
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

    if not pair_rounds:
        return pd.DataFrame(columns=["Season", "T1", "T2", "OfficialMinRound", "OfficialRoundLabel"])

    out = pd.DataFrame(
        [
            (int(season), int(t1), int(t2), int(round_num), ROUND_LABELS.get(int(round_num), f"Round{int(round_num)}"))
            for (t1, t2), round_num in pair_rounds.items()
        ],
        columns=["Season", "T1", "T2", "OfficialMinRound", "OfficialRoundLabel"],
    )
    out["IsPlayIn"] = out["OfficialMinRound"].eq(0)
    out["IsRound1"] = out["OfficialMinRound"].eq(1)
    out["IsRound2"] = out["OfficialMinRound"].eq(2)
    return out.sort_values(["OfficialMinRound", "T1", "T2"]).reset_index(drop=True)
