from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual market supplement template for the highest-priority live market gaps."
    )
    parser.add_argument("--gender", choices=["M", "W"], required=True, help="Competition gender prefix.")
    parser.add_argument("--season", type=int, default=2026, help="Season value.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of matchups to include in the template.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional gap-report CSV path. Defaults to results/<gender>_market_gap_report_<season>.csv style lookup.",
    )
    return parser.parse_args()


def default_gap_path(gender: str, season: int) -> Path:
    if gender == "W":
        return RESULTS_DIR / f"women_market_gap_report_{season}.csv"
    return RESULTS_DIR / f"men_market_gap_report_{season}.csv"


def build_template(frame: pd.DataFrame, season: int, limit: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "Team1Name",
                "Team2Name",
                "Team1Moneyline",
                "Team2Moneyline",
                "MarketProb",
                "LastSpread",
                "Book",
                "BookCount",
                "SnapshotTime",
                "Source",
                "SourceURL",
                "Notes",
            ]
        )

    ranked = frame.sort_values(
        ["priority", "book_count", "marketprob_count", "spread_count", "action_book_count"],
        ascending=[False, True, True, True, True],
    ).head(limit)

    out = ranked[["T1", "T2", "Team1Name", "Team2Name"]].copy()
    out.insert(0, "Season", season)
    out["Team1Moneyline"] = pd.NA
    out["Team2Moneyline"] = pd.NA
    out["MarketProb"] = pd.NA
    out["LastSpread"] = pd.NA
    out["Book"] = ""
    out["BookCount"] = pd.NA
    out["SnapshotTime"] = ""
    out["Source"] = "manual_supplement"
    out["SourceURL"] = ""
    out["Notes"] = ranked.apply(
        lambda row: (
            f"priority={int(row['priority'])}; "
            f"books={int(row['book_count'])}; "
            f"marketprob_count={int(row['marketprob_count'])}; "
            f"spread_count={int(row['spread_count'])}; "
            f"action_books={int(row['action_book_count'])}"
        ),
        axis=1,
    )
    return out


def main() -> None:
    args = parse_args()
    gap_path = Path(args.input) if args.input else default_gap_path(args.gender, args.season)
    if not gap_path.exists():
        raise SystemExit(f"Gap report not found: {gap_path}")
    frame = pd.read_csv(gap_path)
    template = build_template(frame, args.season, args.limit)
    output_path = RESULTS_DIR / f"{args.gender}ManualOddsTemplate_{args.season}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"saved template -> {output_path}")
    print(f"rows={len(template)}")


if __name__ == "__main__":
    main()
