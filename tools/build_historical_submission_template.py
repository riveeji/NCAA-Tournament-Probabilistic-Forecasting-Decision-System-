from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hc.features_structured import load_team_snapshots


def build_ids_for_gender(gender: str, season: int) -> pd.DataFrame:
    snapshots = load_team_snapshots(gender)
    snapshots["Season"] = pd.to_numeric(snapshots["Season"], errors="coerce")
    snapshots["TeamID"] = pd.to_numeric(snapshots["TeamID"], errors="coerce")
    season_teams = (
        snapshots.loc[snapshots["Season"].eq(season), "TeamID"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    rows = [{"ID": f"{season}_{t1}_{t2}", "Pred": 0.5} for t1, t2 in combinations(season_teams, 2)]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an all-teams historical submission template for a specific season.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    men = build_ids_for_gender("M", args.season)
    women = build_ids_for_gender("W", args.season)
    template = pd.concat([men, women], ignore_index=True).sort_values("ID").reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"Historical submission template written to: {output_path}")
    print(f"Rows: {len(template)}")


if __name__ == "__main__":
    main()
