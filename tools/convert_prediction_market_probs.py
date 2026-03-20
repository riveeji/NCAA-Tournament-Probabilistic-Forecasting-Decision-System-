from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Robinhood/Kalshi style prediction-market prices into HC market supplement rows."
    )
    parser.add_argument("--gender", required=True, choices=("M", "W"))
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--input", required=True, help="CSV with team names and contract prices/probabilities.")
    parser.add_argument("--output", default="", help="Optional output path.")
    return parser.parse_args()


def _first_present_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _read_scaled_pair(
    frame: pd.DataFrame,
    left_names: list[str],
    right_names: list[str],
    scale: float,
) -> tuple[pd.Series, pd.Series] | None:
    left = _first_present_column(frame, left_names)
    right = _first_present_column(frame, right_names)
    if left is None and right is None:
        return None

    left_series = pd.to_numeric(frame[left], errors="coerce") / scale if left is not None else pd.Series(np.nan, index=frame.index)
    right_series = pd.to_numeric(frame[right], errors="coerce") / scale if right is not None else pd.Series(np.nan, index=frame.index)
    return left_series, right_series


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _extract_probabilities(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pairs = [
        (
            ["Team1Prob", "T1Prob", "Prob1", "YesProb"],
            ["Team2Prob", "T2Prob", "Prob2", "NoProb"],
            1.0,
        ),
        (
            ["Team1Percent", "T1Percent", "Prob1Percent", "YesPercent"],
            ["Team2Percent", "T2Percent", "Prob2Percent", "NoPercent"],
            100.0,
        ),
        (
            ["Team1Cents", "T1Cents", "Prob1Cents", "YesCents", "YesPriceCents"],
            ["Team2Cents", "T2Cents", "Prob2Cents", "NoCents", "NoPriceCents"],
            100.0,
        ),
    ]
    for left_names, right_names, scale in pairs:
        parsed = _read_scaled_pair(frame, left_names, right_names, scale)
        if parsed is not None:
            p1, p2 = parsed
            break
    else:
        raise ValueError(
            "Missing probability columns. Provide one of: "
            "Team1Prob/Team2Prob, Team1Percent/Team2Percent, or Team1Cents/Team2Cents."
        )

    p1 = pd.to_numeric(p1, errors="coerce")
    p2 = pd.to_numeric(p2, errors="coerce")
    p1 = p1.where((p1 > 0.0) & (p1 < 1.0))
    p2 = p2.where((p2 > 0.0) & (p2 < 1.0))

    p1 = p1.fillna(1.0 - p2)
    p2 = p2.fillna(1.0 - p1)
    p1 = p1.clip(0.001, 0.999)
    p2 = p2.clip(0.001, 0.999)
    return p1, p2


def _prob_to_american(prob: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(prob, errors="coerce").clip(0.001, 0.999)
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    favorite = numeric >= 0.5
    out.loc[favorite] = -100.0 * numeric.loc[favorite] / (1.0 - numeric.loc[favorite])
    out.loc[~favorite] = 100.0 * (1.0 - numeric.loc[~favorite]) / numeric.loc[~favorite]
    return out.round().astype("Int64")


def convert_prediction_market_frame(frame: pd.DataFrame, gender: str, season: int) -> pd.DataFrame:
    frame = frame.copy().replace("", pd.NA)
    team1_col = _first_present_column(frame, ["Team1Name", "TeamAName", "YesTeam", "LeftTeam"])
    team2_col = _first_present_column(frame, ["Team2Name", "TeamBName", "NoTeam", "RightTeam"])
    if team1_col is None or team2_col is None:
        raise ValueError("Input must contain team name columns, e.g. Team1Name and Team2Name.")

    p1, p2 = _extract_probabilities(frame)

    out = pd.DataFrame(
        {
            "Season": int(season),
            "Team1Name": frame[team1_col].astype(str).str.strip(),
            "Team2Name": frame[team2_col].astype(str).str.strip(),
            "Team1Moneyline": _prob_to_american(p1),
            "Team2Moneyline": _prob_to_american(p2),
            "MarketProb": p1,
            "LastSpread": pd.to_numeric(frame.get("LastSpread"), errors="coerce"),
            "Book": _series_or_default(frame, "Book", "prediction_market").fillna("prediction_market").astype(str).str.strip(),
            "BookCount": pd.to_numeric(_series_or_default(frame, "BookCount", 1.0), errors="coerce").fillna(1.0),
            "SnapshotTime": pd.to_datetime(_series_or_default(frame, "SnapshotTime", pd.NA), errors="coerce", utc=True),
            "Source": _series_or_default(frame, "Source", "prediction_market_proxy").fillna("prediction_market_proxy").astype(str).str.strip(),
            "SourceURL": _series_or_default(frame, "SourceURL", "").fillna("").astype(str),
            "Notes": _series_or_default(frame, "Notes", "").fillna("").astype(str),
        }
    )

    if "T1" in frame.columns and "T2" in frame.columns:
        out["T1"] = pd.to_numeric(frame["T1"], errors="coerce").astype("Int64")
        out["T2"] = pd.to_numeric(frame["T2"], errors="coerce").astype("Int64")
        out = out.rename(columns={"T1": "Team1ID", "T2": "Team2ID"})
    else:
        out = attach_team_ids(out, gender, "Team1Name", "Team2Name", "Team1ID", "Team2ID")

    audit_df = out.attrs.get("team_match_audit")
    out = canonicalize_matchups(out)
    out = out.rename(columns={"Team1ID": "T1", "Team2ID": "T2"})
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
    out["T1"] = pd.to_numeric(out["T1"], errors="coerce").astype("Int64")
    out["T2"] = pd.to_numeric(out["T2"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Season", "T1", "T2"]).copy()
    out["Season"] = out["Season"].astype(int)
    out["T1"] = out["T1"].astype(int)
    out["T2"] = out["T2"].astype(int)

    keep = [
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
    out = out[keep].drop_duplicates(subset=["Season", "T1", "T2", "Book", "SnapshotTime"], keep="last")
    out.attrs["team_match_audit"] = audit_df
    return out.reset_index(drop=True)


def convert_prediction_market_file(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    return convert_prediction_market_frame(frame, gender, season)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else EXTERNAL_DIR / f"{args.gender}PredictionMarketOdds_{args.season}.csv"
    frame = convert_prediction_market_file(input_path, args.gender, args.season)
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
