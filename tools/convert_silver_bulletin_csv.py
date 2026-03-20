from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import attach_team_ids_from_names


EXTERNAL_DIR = ROOT / "external-data"


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path)).replace("", pd.NA)


def _attach_ids(frame: pd.DataFrame, gender: str) -> pd.DataFrame:
    frame = attach_team_ids_from_names(frame, gender, team_col="TeamName", target_col="TeamID")
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["Season", "TeamID", "TeamName"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def _coalesce_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["Season", "TeamID"])
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.dropna(subset=["Season", "TeamID"]).copy()
    merged["Season"] = pd.to_numeric(merged["Season"], errors="coerce").astype(int)
    merged["TeamID"] = pd.to_numeric(merged["TeamID"], errors="coerce").astype(int)
    merged = merged.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    value_columns = [column for column in merged.columns if column not in {"Season", "TeamID"}]

    def last_non_null(series: pd.Series):
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        return non_null.iloc[-1]

    aggregated = merged.groupby(["Season", "TeamID"], as_index=False)[value_columns].agg(last_non_null)
    return aggregated


def convert_history_panel(path: str | Path, gender: str) -> pd.DataFrame:
    frame = _read_csv(path)
    rename = {
        "sb_name": "TeamName",
        "season": "Season",
        "b_xelo_n": "SB_BXelo",
        "b_pppg_n": "SB_BPPPG",
        "b_ppag_n": "SB_BPPAG",
        "b_netrating_n": "SB_BNetRating",
        "lg_scoring": "SB_LeagueScoring",
    }
    available = [column for column in rename if column in frame.columns]
    frame = frame[available].rename(columns={column: rename[column] for column in available})
    frame = _attach_ids(frame, gender)
    numeric_cols = [column for column in frame.columns if column.startswith("SB_")]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = "Silver Bulletin Historical Panel"
    ordered = ["Season", "TeamID", "TeamName", "Source"] + numeric_cols
    return frame[ordered]


def convert_current_bayesian(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = _read_csv(path)
    rename = {
        "sb_name": "TeamName",
        "b_xelo_n": "SB_BXelo",
        "b_pppg_n": "SB_BPPPG",
        "b_ppag_n": "SB_BPPAG",
        "b_netrating_n": "SB_BNetRating",
        "sos": "SB_SOS",
        "current_hfa": "SB_CurrentHFA",
    }
    available = [column for column in rename if column in frame.columns]
    frame = frame[available].rename(columns={column: rename[column] for column in available})
    frame["Season"] = int(season)
    frame = _attach_ids(frame, gender)
    numeric_cols = [column for column in frame.columns if column.startswith("SB_")]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = "Silver Bulletin Current Bayesian"
    ordered = ["Season", "TeamID", "TeamName", "Source"] + numeric_cols
    return frame[ordered]


def convert_current_xelo(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = _read_csv(path)
    rename = {
        "sb_name": "TeamName",
        "xelo_n": "SB_Xelo",
        "pppg_n": "SB_PPPG",
        "ppag_n": "SB_PPAG",
        "netrating_n": "SB_NetRating",
        "sos": "SB_XeloSOS",
        "current_hfa": "SB_XeloCurrentHFA",
    }
    available = [column for column in rename if column in frame.columns]
    frame = frame[available].rename(columns={column: rename[column] for column in available})
    frame["Season"] = int(season)
    frame = _attach_ids(frame, gender)
    numeric_cols = [column for column in frame.columns if column.startswith("SB_")]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = "Silver Bulletin Current XElo"
    ordered = ["Season", "TeamID", "TeamName", "Source"] + numeric_cols
    return frame[ordered]


def convert_display_rank(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = _read_csv(path)
    if {"rank", "team", "elo", "change", "prior", "season_min", "season_max"}.issubset(frame.columns):
        rename = {
            "team": "TeamName",
            "conference": "SB_DisplayConference",
            "rank": "SB_DisplayRank",
            "elo": "SB_DisplayElo",
            "change": "SB_DisplayChange",
            "prior": "SB_DisplayPrior",
            "season_min": "SB_DisplaySeasonMin",
            "season_max": "SB_DisplaySeasonMax",
        }
    else:
        rename = {
            "Unnamed: 0": "SB_DisplayRank",
            "Team": "TeamName",
            "Conf.": "SB_DisplayConference",
            "Current Elo": "SB_DisplayElo",
            "Last": "SB_DisplayChange",
            "Season Min.": "SB_DisplaySeasonMin",
            "Season Max.": "SB_DisplaySeasonMax",
            "Home Court*": "SB_HomeCourtDisplay",
        }
    available = [column for column in rename if column in frame.columns]
    frame = frame[available].rename(columns={column: rename[column] for column in available})
    frame["Season"] = int(season)
    frame = _attach_ids(frame, gender)
    numeric_cols = [column for column in frame.columns if column.startswith("SB_") and column != "SB_DisplayConference"]
    for column in numeric_cols:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.replace(r"[^0-9+\-.]", "", regex=True)
            .replace({"": pd.NA, "+": pd.NA, "-": pd.NA})
        )
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = "Silver Bulletin Display"
    ordered = ["Season", "TeamID", "TeamName", "Source"] + [column for column in frame.columns if column.startswith("SB_")]
    return frame[ordered]


def convert_ratings_comparison(path: str | Path, gender: str, season: int) -> pd.DataFrame:
    frame = _read_csv(path)
    rename = {
        "sb_name": "TeamName",
        "cooper": "SB_Cooper",
        "pomeroy": "SB_Pomeroy",
        "composite": "SB_Composite",
        "injury_adjustment": "SB_InjuryAdjustment",
        "adjusted_composite": "SB_AdjustedComposite",
        "rank": "SB_DisplayRank",
    }
    available = [column for column in rename if column in frame.columns]
    frame = frame[available].rename(columns={column: rename[column] for column in available})
    frame["Season"] = int(season)
    frame = _attach_ids(frame, gender)
    numeric_cols = [column for column in frame.columns if column.startswith("SB_")]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = "Silver Bulletin Ratings Comparison"
    ordered = ["Season", "TeamID", "TeamName", "Source"] + numeric_cols
    return frame[ordered]


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    mapped = int(pd.to_numeric(frame["TeamID"], errors="coerce").notna().sum()) if "TeamID" in frame.columns else 0
    print({"output": str(output_path), "rows": int(len(frame)), "mapped_team_ids": mapped})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Silver Bulletin csv exports into HC-friendly team-rating CSVs.")
    parser.add_argument("--gender", required=True, choices=("M", "W"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--history-panel", default="")
    parser.add_argument("--current-bayes", default="")
    parser.add_argument("--current-xelo", default="")
    parser.add_argument("--display", nargs="*", default=[])
    parser.add_argument("--ratings-comparison", nargs="*", default=[])
    parser.add_argument("--history-output", default="")
    parser.add_argument("--current-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_frames: list[pd.DataFrame] = []
    current_frames: list[pd.DataFrame] = []

    if args.history_panel:
        history_frames.append(convert_history_panel(args.history_panel, args.gender))
    if args.current_bayes:
        current_frames.append(convert_current_bayesian(args.current_bayes, args.gender, args.season))
    if args.current_xelo:
        current_frames.append(convert_current_xelo(args.current_xelo, args.gender, args.season))
    for path in args.display:
        current_frames.append(convert_display_rank(path, args.gender, args.season))
    for path in args.ratings_comparison:
        current_frames.append(convert_ratings_comparison(path, args.gender, args.season))

    if history_frames:
        history = _coalesce_frames(history_frames)
        history_output = args.history_output or str(EXTERNAL_DIR / f"{args.gender}SilverBulletinTeamRatings_History.csv")
        write_frame(history, history_output)
    if current_frames:
        current = _coalesce_frames(current_frames)
        current_output = args.current_output or str(EXTERNAL_DIR / f"{args.gender}SilverBulletinTeamRatings_{args.season}.csv")
        write_frame(current, current_output)


if __name__ == "__main__":
    main()
