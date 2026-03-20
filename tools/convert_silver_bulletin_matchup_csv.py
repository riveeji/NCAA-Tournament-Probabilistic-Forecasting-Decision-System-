from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import attach_team_ids, canonicalize_matchups, write_unmatched_log

EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"


def _parse_snapshot_from_name(path: Path) -> pd.Timestamp:
    stem = path.stem
    match = re.search(r"(\d{8})(?:[_-]?(\d{2}))?", stem)
    if not match:
        return pd.NaT
    day = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce", utc=True)
    if pd.isna(day):
        return pd.NaT
    if match.group(2):
        return day + pd.to_timedelta(int(match.group(2)), unit="h")
    return day


def convert_matchup_projections(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = pd.read_csv(Path(path)).replace("", pd.NA)
    required = {"team_a_odds", "team_b_odds", "full_sb_name_a", "full_sb_name_b"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required Silver matchup columns: {missing}")

    out = pd.DataFrame(
        {
            "Season": int(season),
            "Team1Name": frame["full_sb_name_a"].fillna(frame.get("sb_name")),
            "Team2Name": frame["full_sb_name_b"].fillna(frame.get("rank")),
            "Team1ImpliedProb": pd.to_numeric(frame["team_a_odds"], errors="coerce"),
            "Team2ImpliedProb": pd.to_numeric(frame["team_b_odds"], errors="coerce"),
            "Team1Spread": pd.to_numeric(frame.get("team_a_spread"), errors="coerce"),
            "ProjectedTotal": pd.to_numeric(frame.get("projected_total"), errors="coerce"),
            "SilverRound": pd.to_numeric(frame.get("round"), errors="coerce"),
            "SnapshotTime": _parse_snapshot_from_name(Path(path)),
            "Source": "Silver Bulletin Matchup Projections",
        }
    )
    out = attach_team_ids(out, gender, "Team1Name", "Team2Name", "Team1ID", "Team2ID")
    audit_df = out.attrs.get("team_match_audit")
    out = canonicalize_matchups(out)
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
    out["Team1ID"] = pd.to_numeric(out["Team1ID"], errors="coerce").astype("Int64")
    out["Team2ID"] = pd.to_numeric(out["Team2ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Season", "Team1ID", "Team2ID"]).copy()
    out["Season"] = out["Season"].astype(int)
    out["Team1ID"] = out["Team1ID"].astype(int)
    out["Team2ID"] = out["Team2ID"].astype(int)
    out["SilverProb"] = pd.to_numeric(out["Team1ImpliedProb"], errors="coerce")
    out["SilverSpread"] = pd.to_numeric(out["Team1Spread"], errors="coerce")
    out["SilverProjectedTotal"] = pd.to_numeric(out["ProjectedTotal"], errors="coerce")
    out = out[
        [
            "Season",
            "Team1ID",
            "Team2ID",
            "Team1Name",
            "Team2Name",
            "SilverProb",
            "SilverSpread",
            "SilverProjectedTotal",
            "SilverRound",
            "SnapshotTime",
            "Source",
        ]
    ].drop_duplicates(subset=["Season", "Team1ID", "Team2ID"], keep="last")
    out.attrs["team_match_audit"] = audit_df
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Silver Bulletin matchup projection CSV to HC runtime format.")
    parser.add_argument("--gender", required=True, choices=("M", "W"))
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else EXTERNAL_DIR / f"{args.gender}SilverBulletinMatchupProjections_{args.season}.csv"
    frame = convert_matchup_projections(input_path, args.gender, args.season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    audit_df = frame.attrs.get("team_match_audit")
    audit_path = RESULTS_DIR / f"{output_path.stem}_unmatched.csv"
    write_unmatched_log(audit_df, audit_path)
    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "rows": int(len(frame)),
            "mapped_rows": int(len(frame)),
            "unmatched_log": str(audit_path) if audit_path.exists() else "",
        }
    )


if __name__ == "__main__":
    main()
