from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_pairwise_predictions(path: Path) -> dict[tuple[int, int, int], float]:
    df = pd.read_csv(path, usecols=["ID", "Pred"])
    parts = df["ID"].astype(str).str.split("_", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Expected submission IDs in Season_T1_T2 format")
    seasons = pd.to_numeric(parts[0], errors="raise").astype(int)
    t1s = pd.to_numeric(parts[1], errors="raise").astype(int)
    t2s = pd.to_numeric(parts[2], errors="raise").astype(int)
    probs = pd.to_numeric(df["Pred"], errors="raise").astype(float)

    lookup: dict[tuple[int, int, int], float] = {}
    for season, t1, t2, prob in zip(seasons, t1s, t2s, probs):
        lookup[(season, t1, t2)] = float(prob)
        lookup[(season, t2, t1)] = 1.0 - float(prob)
    return lookup


def build_slot_winners(
    *,
    season: int,
    seed_to_team: dict[str, int],
    slots_df: pd.DataFrame,
    pairwise_lookup: dict[tuple[int, int, int], float],
) -> dict[str, str]:
    slot_map = {
        str(row["Slot"]): (str(row["StrongSeed"]), str(row["WeakSeed"]))
        for _, row in slots_df.iterrows()
    }
    resolved: dict[str, tuple[str, int]] = {}

    def resolve(label: str) -> tuple[str, int]:
        if label in resolved:
            return resolved[label]
        if label in seed_to_team:
            resolved[label] = (label, int(seed_to_team[label]))
            return resolved[label]
        if label not in slot_map:
            raise KeyError(f"Unknown slot/seed label: {label}")

        strong_label, weak_label = slot_map[label]
        strong_seed, strong_team = resolve(strong_label)
        weak_seed, weak_team = resolve(weak_label)
        prob = pairwise_lookup.get((season, strong_team, weak_team))
        if prob is None:
            raise KeyError(
                f"Missing pairwise probability for season={season}, teams=({strong_team}, {weak_team}) "
                f"while resolving slot {label}"
            )
        winner = (strong_seed, strong_team) if prob >= 0.5 else (weak_seed, weak_team)
        resolved[label] = winner
        return winner

    for slot in slots_df["Slot"].astype(str):
        resolve(slot)
    return {slot: seed for slot, (seed, _) in resolved.items() if slot in slot_map}


def build_submission(
    *,
    competition_dir: Path,
    pairwise_submission: Path,
    output_path: Path,
) -> pd.DataFrame:
    season = 2024
    sample_path = competition_dir / "sample_submission.csv"
    seeds_path = competition_dir / "2024_tourney_seeds.csv"
    men_slots_path = competition_dir / "MNCAATourneySlots.csv"
    women_slots_path = competition_dir / "WNCAATourneySlots.csv"

    sample = pd.read_csv(sample_path)
    seeds = pd.read_csv(seeds_path)
    men_slots = pd.read_csv(men_slots_path)
    women_slots = pd.read_csv(women_slots_path)
    pairwise_lookup = load_pairwise_predictions(pairwise_submission)

    men_seed_to_team = {
        str(row["Seed"]): int(row["TeamID"])
        for _, row in seeds.loc[seeds["Tournament"] == "M", ["Seed", "TeamID"]].iterrows()
    }
    women_seed_to_team = {
        str(row["Seed"]): int(row["TeamID"])
        for _, row in seeds.loc[seeds["Tournament"] == "W", ["Seed", "TeamID"]].iterrows()
    }

    men_slots = men_slots.loc[men_slots["Season"] == season, ["Slot", "StrongSeed", "WeakSeed"]].copy()
    women_slots = women_slots.loc[women_slots["Season"] == season, ["Slot", "StrongSeed", "WeakSeed"]].copy()

    men_winners = build_slot_winners(
        season=season,
        seed_to_team=men_seed_to_team,
        slots_df=men_slots,
        pairwise_lookup=pairwise_lookup,
    )
    women_winners = build_slot_winners(
        season=season,
        seed_to_team=women_seed_to_team,
        slots_df=women_slots,
        pairwise_lookup=pairwise_lookup,
    )

    out = sample.copy()
    teams: list[str] = []
    for _, row in out.iterrows():
        tournament = str(row["Tournament"])
        slot = str(row["Slot"])
        if tournament == "M":
            teams.append(men_winners[slot])
        elif tournament == "W":
            teams.append(women_winners[slot])
        else:
            raise ValueError(f"Unexpected tournament value: {tournament}")
    out["Team"] = teams
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2024 bracket-style submission from pairwise HC probabilities.")
    parser.add_argument(
        "--competition-dir",
        required=True,
        help="Path to the Kaggle 2024 competition input directory containing sample_submission.csv and slot/seed files.",
    )
    parser.add_argument(
        "--pairwise-submission",
        default="submission_stage2_single_final_hc_2024.csv",
        help="Path to the precomputed pairwise submission CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/submission_2024_bracket_from_hc.csv",
        help="Path for the generated bracket submission CSV.",
    )
    args = parser.parse_args()

    competition_dir = Path(args.competition_dir)
    pairwise_submission = (ROOT / args.pairwise_submission).resolve() if not Path(args.pairwise_submission).is_absolute() else Path(args.pairwise_submission)
    output_path = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    df = build_submission(
        competition_dir=competition_dir,
        pairwise_submission=pairwise_submission,
        output_path=output_path,
    )
    print(df.head())
    print(
        {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "dup_rowid": int(df["RowId"].duplicated().sum()),
            "missing_team": int(df["Team"].isna().sum()),
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
