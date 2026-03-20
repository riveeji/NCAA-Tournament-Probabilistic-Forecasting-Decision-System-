from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Brier score under alternative tournament worlds by "
            "softening the current submission probabilities toward 50/50."
        )
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submission_stage2_single_final_hc.csv"),
        help="Submission CSV with ID,Pred columns.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Tournament season to simulate.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=3000,
        help="Monte Carlo simulations per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260319,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/scenario_brier_upset_bands_latest"),
        help="Output path prefix without extension.",
    )
    return parser.parse_args()


def clamp_prob(p: float) -> float:
    return min(max(float(p), 1e-9), 1.0 - 1e-9)


def base_seed(seed: str) -> str:
    return re.sub(r"[a-z]+$", "", seed.strip())


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def temperature_soften(p: float, temperature: float) -> float:
    p = clamp_prob(p)
    if temperature <= 1.0:
        return p
    logit = math.log(p / (1.0 - p))
    return sigmoid(logit / temperature)


def load_submission(path: Path) -> dict[str, float]:
    submission = pd.read_csv(path)
    if set(submission.columns) != {"ID", "Pred"}:
        raise ValueError(f"Unexpected submission columns in {path}: {submission.columns.tolist()}")
    return dict(zip(submission["ID"], submission["Pred"]))


def matchup_prob(pred_lookup: dict[str, float], season: int, team_a: int, team_b: int) -> float:
    low, high = sorted((int(team_a), int(team_b)))
    key = f"{season}_{low}_{high}"
    p = pred_lookup.get(key)
    if p is None:
        raise KeyError(f"Missing matchup probability for {key}")
    return float(p if low == team_a else 1.0 - p)


@dataclass
class ScenarioSummary:
    scenario: str
    temperature: float
    combined_mean_brier: float
    combined_p10: float
    combined_p50: float
    combined_p90: float
    men_mean_brier: float
    women_mean_brier: float
    favorite_win_rate_mean: float
    games_counted: int


class TournamentSimulator:
    def __init__(self, season: int, gender_prefix: str, pred_lookup: dict[str, float]) -> None:
        self.season = season
        self.gender_prefix = gender_prefix
        self.pred_lookup = pred_lookup
        seeds_path = Path("ncaa-data") / f"{gender_prefix}NCAATourneySeeds.csv"
        slots_path = Path("ncaa-data") / f"{gender_prefix}NCAATourneySlots.csv"

        seeds_df = pd.read_csv(seeds_path)
        slots_df = pd.read_csv(slots_path)
        self.seeds_df = seeds_df.loc[seeds_df["Season"] == season, ["Seed", "TeamID"]].copy()
        self.slots_df = slots_df.loc[
            slots_df["Season"] == season, ["Slot", "StrongSeed", "WeakSeed"]
        ].copy()
        self.slot_defs = {
            row.Slot: (row.StrongSeed, row.WeakSeed)
            for row in self.slots_df.itertuples(index=False)
        }

    def simulate(self, temperature: float, rng: np.random.Generator) -> dict[str, object]:
        game_records: list[tuple[float, int, bool]] = []

        grouped: dict[str, list[int]] = defaultdict(list)
        for row in self.seeds_df.itertuples(index=False):
            grouped[base_seed(row.Seed)].append(int(row.TeamID))

        ref_winners: dict[str, int] = {}
        slot_cache: dict[str, int] = {}

        for seed_ref, team_ids in grouped.items():
            if len(team_ids) == 1:
                ref_winners[seed_ref] = team_ids[0]
                continue

            ordered = sorted(team_ids)
            winner = ordered[0]
            for challenger in ordered[1:]:
                p_submit = matchup_prob(self.pred_lookup, self.season, winner, challenger)
                p_world = temperature_soften(p_submit, temperature)
                winner_is_first = rng.random() < p_world
                actual = 1 if winner_is_first else 0
                favorite_win = actual == (1 if p_submit >= 0.5 else 0)
                game_records.append((p_submit, actual, favorite_win))
                winner = winner if winner_is_first else challenger
            ref_winners[seed_ref] = winner

        def resolve(ref: str) -> int:
            if ref in ref_winners:
                return ref_winners[ref]
            if ref in slot_cache:
                return slot_cache[ref]
            if ref not in self.slot_defs:
                raise KeyError(f"Unknown slot or seed reference: {ref}")

            strong_ref, weak_ref = self.slot_defs[ref]
            team_a = resolve(strong_ref)
            team_b = resolve(weak_ref)
            p_submit = matchup_prob(self.pred_lookup, self.season, team_a, team_b)
            p_world = temperature_soften(p_submit, temperature)
            team_a_wins = rng.random() < p_world
            actual = 1 if team_a_wins else 0
            favorite_win = actual == (1 if p_submit >= 0.5 else 0)
            game_records.append((p_submit, actual, favorite_win))
            winner = team_a if team_a_wins else team_b
            slot_cache[ref] = winner
            return winner

        for slot in self.slots_df["Slot"]:
            resolve(str(slot))

        brier = [((p - y) ** 2) for p, y, _ in game_records]
        favorite_win_rate = [1.0 if fav else 0.0 for _, _, fav in game_records]
        return {
            "brier_mean": float(np.mean(brier)),
            "favorite_win_rate": float(np.mean(favorite_win_rate)),
            "games_counted": len(game_records),
        }


def summarize(name: str, temperature: float, combined: list[float], men: list[float], women: list[float], favorite_rates: list[float], games_counted: int) -> ScenarioSummary:
    return ScenarioSummary(
        scenario=name,
        temperature=temperature,
        combined_mean_brier=float(np.mean(combined)),
        combined_p10=float(np.quantile(combined, 0.10)),
        combined_p50=float(np.quantile(combined, 0.50)),
        combined_p90=float(np.quantile(combined, 0.90)),
        men_mean_brier=float(np.mean(men)),
        women_mean_brier=float(np.mean(women)),
        favorite_win_rate_mean=float(np.mean(favorite_rates)),
        games_counted=games_counted,
    )


def main() -> None:
    args = parse_args()
    pred_lookup = load_submission(args.submission)
    men_sim = TournamentSimulator(args.season, "M", pred_lookup)
    women_sim = TournamentSimulator(args.season, "W", pred_lookup)

    scenarios = [
        ("Light Upsets", 1.25),
        ("Moderate Upsets", 1.75),
        ("Big Chaos", 3.00),
    ]

    rng = np.random.default_rng(args.seed)
    summaries: list[ScenarioSummary] = []

    for scenario_name, temperature in scenarios:
        combined_scores: list[float] = []
        men_scores: list[float] = []
        women_scores: list[float] = []
        favorite_rates: list[float] = []
        games_counted = 0

        for _ in range(args.sims):
            men_result = men_sim.simulate(temperature, rng)
            women_result = women_sim.simulate(temperature, rng)
            games_counted = int(men_result["games_counted"]) + int(women_result["games_counted"])
            men_scores.append(float(men_result["brier_mean"]))
            women_scores.append(float(women_result["brier_mean"]))
            combined_scores.append(
                float(
                    (
                        men_result["brier_mean"] * men_result["games_counted"]
                        + women_result["brier_mean"] * women_result["games_counted"]
                    )
                    / games_counted
                )
            )
            favorite_rates.append(
                float(
                    (
                        men_result["favorite_win_rate"] * men_result["games_counted"]
                        + women_result["favorite_win_rate"] * women_result["games_counted"]
                    )
                    / games_counted
                )
            )

        summaries.append(
            summarize(
                scenario_name,
                temperature,
                combined_scores,
                men_scores,
                women_scores,
                favorite_rates,
                games_counted,
            )
        )

    rows = [
        {
            "Scenario": s.scenario,
            "Temperature": s.temperature,
            "GamesCounted": s.games_counted,
            "FavoriteWinRateMean": s.favorite_win_rate_mean,
            "CombinedMeanBrier": s.combined_mean_brier,
            "CombinedP10": s.combined_p10,
            "CombinedP50": s.combined_p50,
            "CombinedP90": s.combined_p90,
            "MenMeanBrier": s.men_mean_brier,
            "WomenMeanBrier": s.women_mean_brier,
        }
        for s in summaries
    ]
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
