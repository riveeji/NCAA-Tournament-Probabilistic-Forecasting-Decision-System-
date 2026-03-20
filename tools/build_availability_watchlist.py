from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "ncaa-data"
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tournament-field injury/availability watchlist from normalized external reports."
    )
    parser.add_argument("--season", type=int, default=2026, help="Tournament season.")
    parser.add_argument(
        "--injuries",
        default=str(EXTERNAL_DIR / "MRotoWireInjuries_2026.csv"),
        help="Normalized injuries CSV to use.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(RESULTS_DIR / "availability_watchlist_2026_men.csv"),
        help="Detailed watchlist CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        default=str(RESULTS_DIR / "availability_watchlist_2026_men.json"),
        help="Summary JSON output path.",
    )
    return parser.parse_args()


def load_tournament_field(season: int) -> pd.DataFrame:
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")
    field = seeds.loc[pd.to_numeric(seeds["Season"], errors="coerce").eq(season), ["Season", "Seed", "TeamID"]].copy()
    field["Season"] = pd.to_numeric(field["Season"], errors="coerce").astype(int)
    field["TeamID"] = pd.to_numeric(field["TeamID"], errors="coerce").astype(int)
    field = field.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    return field


def load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    return frame


def build_watchlist(injuries: pd.DataFrame, field: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = injuries.copy()
    if frame.empty or "Season" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "Season",
                "Seed",
                "TeamID",
                "TeamName",
                "SeverityScore",
                "OutCount",
                "GTDCount",
                "PlayerCount",
                "Players",
                "Statuses",
                "Injuries",
            ]
        )
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce").fillna(season).astype(int)
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame["Severity"] = pd.to_numeric(frame["Severity"], errors="coerce").fillna(0).astype(int)
    frame["IsOut"] = pd.to_numeric(frame.get("IsOut"), errors="coerce").fillna(0).astype(int)
    frame["IsGameTimeDecision"] = pd.to_numeric(frame.get("IsGameTimeDecision"), errors="coerce").fillna(0).astype(int)
    frame = frame.dropna(subset=["TeamID"]).copy()
    frame["TeamID"] = frame["TeamID"].astype(int)
    frame = frame.merge(field, on=["Season", "TeamID"], how="inner", suffixes=("", "_Field"))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Seed",
                "TeamID",
                "TeamName",
                "SeverityScore",
                "OutCount",
                "GTDCount",
                "PlayerCount",
                "Players",
            ]
        )

    def join_players(group: pd.Series) -> str:
        values = [str(value).strip() for value in group if str(value).strip()]
        return " | ".join(sorted(dict.fromkeys(values)))

    team_level = (
        frame.groupby(["Season", "Seed", "TeamID", "TeamName_Field"], as_index=False)
        .agg(
            SeverityScore=("Severity", "sum"),
            OutCount=("IsOut", "sum"),
            GTDCount=("IsGameTimeDecision", "sum"),
            PlayerCount=("PlayerName", "count"),
            Players=("PlayerName", join_players),
            Statuses=("Status", join_players),
            Injuries=("Injury", join_players),
        )
        .rename(columns={"TeamName_Field": "TeamName"})
        .sort_values(["SeverityScore", "OutCount", "GTDCount", "Seed"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    return team_level


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    injuries = load_injuries(Path(args.injuries))
    field = load_tournament_field(args.season)
    watchlist = build_watchlist(injuries, field, args.season)

    csv_path = Path(args.output_csv)
    json_path = Path(args.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(csv_path, index=False)

    summary = {
        "season": int(args.season),
        "injury_rows": int(len(injuries)),
        "field_teams": int(len(field)),
        "watchlist_teams": int(len(watchlist)),
        "teams_with_outs": int((pd.to_numeric(watchlist.get("OutCount"), errors="coerce").fillna(0) > 0).sum()) if not watchlist.empty else 0,
        "teams_with_gtd": int((pd.to_numeric(watchlist.get("GTDCount"), errors="coerce").fillna(0) > 0).sum()) if not watchlist.empty else 0,
        "top10": watchlist.head(10).to_dict(orient="records"),
        "csv": str(csv_path),
    }
    write_json(json_path, summary)

    print(f"watchlist rows={len(watchlist)}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
