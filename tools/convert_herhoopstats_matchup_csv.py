from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
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


def _find_column(frame: pd.DataFrame, preferred: str | None, aliases: list[str]) -> str | None:
    if preferred and preferred in frame.columns:
        return preferred
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        found = normalized.get(alias.lower())
        if found:
            return found
    return None


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    out = pd.to_numeric(values, errors="coerce")
    too_large = out.abs() > 1.0
    if too_large.any():
        out.loc[too_large] = out.loc[too_large] / 100.0
    return out


def _derive_spread_from_scores(score1: pd.Series, score2: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(score1, errors="coerce")
    s2 = pd.to_numeric(score2, errors="coerce")
    return s2 - s1


def _derive_total_from_scores(score1: pd.Series, score2: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(score1, errors="coerce")
    s2 = pd.to_numeric(score2, errors="coerce")
    return s1 + s2


def convert_matchup_projections(
    path: str | Path,
    gender: str,
    season: int,
    *,
    team1_col: str = "",
    team2_col: str = "",
    prob_col: str = "",
    spread_col: str = "",
    total_col: str = "",
    score1_col: str = "",
    score2_col: str = "",
    round_col: str = "",
    date_col: str = "",
) -> pd.DataFrame:
    frame = pd.read_csv(Path(path)).replace("", pd.NA)
    team1_name_col = _find_column(
        frame,
        team1_col or None,
        ["Team1Name", "team1", "team_1", "away_team", "visitor", "team_a", "Team A", "Away Team", "Visitor"],
    )
    team2_name_col = _find_column(
        frame,
        team2_col or None,
        ["Team2Name", "team2", "team_2", "home_team", "home", "team_b", "Team B", "Home Team", "Home"],
    )
    if not team1_name_col or not team2_name_col:
        raise ValueError("Could not infer matchup team-name columns. Pass --team1-col and --team2-col explicitly.")

    prob_name_col = _find_column(
        frame,
        prob_col or None,
        [
            "HerHoopProb",
            "ModelProb",
            "WinProb",
            "Win Probability",
            "Predicted Win Probability",
            "team1_win_prob",
            "team_a_win_prob",
            "team_a_odds",
            "probability",
        ],
    )
    spread_name_col = _find_column(
        frame,
        spread_col or None,
        [
            "HerHoopSpread",
            "ModelSpread",
            "Spread",
            "Predicted Spread",
            "Projected Spread",
            "team1_spread",
            "team_a_spread",
            "margin",
        ],
    )
    total_name_col = _find_column(
        frame,
        total_col or None,
        ["HerHoopProjectedTotal", "ModelProjectedTotal", "ProjectedTotal", "Predicted Total", "total"],
    )
    score1_name_col = _find_column(
        frame,
        score1_col or None,
        ["Team1Score", "team1_score", "team_a_score", "Away Score", "Visitor Score", "Predicted Away Score"],
    )
    score2_name_col = _find_column(
        frame,
        score2_col or None,
        ["Team2Score", "team2_score", "team_b_score", "Home Score", "Predicted Home Score"],
    )
    round_name_col = _find_column(frame, round_col or None, ["Round", "ModelRound", "HerHoopRound"])
    date_name_col = _find_column(frame, date_col or None, ["EventDate", "Date", "GameDate"])

    out = pd.DataFrame(
        {
            "Season": int(season),
            "Team1Name": frame[team1_name_col].astype(str).str.strip(),
            "Team2Name": frame[team2_name_col].astype(str).str.strip(),
            "Team1ImpliedProb": _numeric(frame[prob_name_col]) if prob_name_col else np.nan,
            "Team1Spread": _numeric(frame[spread_name_col]) if spread_name_col else np.nan,
            "ProjectedTotal": _numeric(frame[total_name_col]) if total_name_col else np.nan,
            "HerHoopRound": pd.to_numeric(frame[round_name_col], errors="coerce") if round_name_col else np.nan,
            "EventDate": pd.to_datetime(frame[date_name_col], errors="coerce") if date_name_col else pd.NaT,
            "SnapshotTime": _parse_snapshot_from_name(Path(path)),
            "Source": "Her Hoop Stats Matchup Projections",
        }
    )

    if score1_name_col and score2_name_col:
        score1 = pd.to_numeric(frame[score1_name_col], errors="coerce")
        score2 = pd.to_numeric(frame[score2_name_col], errors="coerce")
        if out["Team1Spread"].isna().all():
            out["Team1Spread"] = _derive_spread_from_scores(score1, score2)
        if out["ProjectedTotal"].isna().all():
            out["ProjectedTotal"] = _derive_total_from_scores(score1, score2)

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
    out["HerHoopProb"] = pd.to_numeric(out["Team1ImpliedProb"], errors="coerce")
    out["HerHoopSpread"] = pd.to_numeric(out["Team1Spread"], errors="coerce")
    out["HerHoopProjectedTotal"] = pd.to_numeric(out["ProjectedTotal"], errors="coerce")
    out = out[
        [
            "Season",
            "Team1ID",
            "Team2ID",
            "Team1Name",
            "Team2Name",
            "HerHoopProb",
            "HerHoopSpread",
            "HerHoopProjectedTotal",
            "HerHoopRound",
            "EventDate",
            "SnapshotTime",
            "Source",
        ]
    ].drop_duplicates(subset=["Season", "Team1ID", "Team2ID"], keep="last")
    out.attrs["team_match_audit"] = audit_df
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Her Hoop Stats matchup CSV to HC runtime format.")
    parser.add_argument("--gender", default="W", choices=("M", "W"))
    parser.add_argument("--season", default=2026, type=int)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--team1-col", default="")
    parser.add_argument("--team2-col", default="")
    parser.add_argument("--prob-col", default="")
    parser.add_argument("--spread-col", default="")
    parser.add_argument("--total-col", default="")
    parser.add_argument("--score1-col", default="")
    parser.add_argument("--score2-col", default="")
    parser.add_argument("--round-col", default="")
    parser.add_argument("--date-col", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else EXTERNAL_DIR / f"{args.gender}HerHoopStatsMatchupProjections_{args.season}.csv"
    )
    frame = convert_matchup_projections(
        input_path,
        args.gender,
        args.season,
        team1_col=args.team1_col,
        team2_col=args.team2_col,
        prob_col=args.prob_col,
        spread_col=args.spread_col,
        total_col=args.total_col,
        score1_col=args.score1_col,
        score2_col=args.score2_col,
        round_col=args.round_col,
        date_col=args.date_col,
    )
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
            "unmatched_log": str(audit_path) if audit_path.exists() else "",
        }
    )


if __name__ == "__main__":
    main()
