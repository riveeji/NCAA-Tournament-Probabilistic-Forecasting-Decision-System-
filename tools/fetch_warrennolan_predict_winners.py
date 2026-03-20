from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import (
    DEFAULT_FUZZY_THRESHOLD,
    attach_team_ids,
    canonicalize_matchups,
    summarize,
    write_unmatched_log,
)


PAGE_URLS = {
    "M": "https://www.warrennolan.com/basketball/{season}/predict-winners",
    "W": "https://www.warrennolan.com/basketballw/{season}/predict-winners",
}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-warrennolan"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Warren Nolan NCAA predict-winners matchup projections into normalized HC model-signal CSVs."
    )
    parser.add_argument("--gender", choices=["M", "W"], help="Competition gender prefix.")
    parser.add_argument("--all", action="store_true", help="Fetch both men's and women's projections.")
    parser.add_argument("--season", type=int, default=2026, help="Season value to write into output.")
    parser.add_argument(
        "--date-from",
        default=str(date.today()),
        help="Start date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--date-to",
        default=str(date.today() + timedelta(days=7)),
        help="End date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--type1",
        default="NCAA",
        help="Warren Nolan type1 query value, e.g. NCAA or Today, March 16.",
    )
    parser.add_argument(
        "--type2",
        default="All Games",
        help="Warren Nolan type2 query value, e.g. All Games or Top 25.",
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw HTML archives.")
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned output CSV files.")
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Minimum RapidFuzz token_set_ratio score required to auto-map a team name.",
    )
    parser.add_argument(
        "--unmatched-dir",
        default=str(AUDIT_DIR),
        help="Directory for unresolved team-name audit CSVs.",
    )
    return parser.parse_args()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("date_to must be on or after date_from")
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _parse_numeric(text: str) -> float | None:
    value = str(text or "").strip().replace("%", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_round_number(text: str) -> int | None:
    label = str(text or "").strip().lower()
    if "first four" in label:
        return 0
    if "first round" in label:
        return 1
    if "second round" in label:
        return 2
    if "sweet 16" in label:
        return 3
    if "elite 8" in label:
        return 4
    if "final four" in label:
        return 5
    if "championship" in label or "title" in label:
        return 6
    return None


def _extract_team_name(row) -> str:
    link = row.select_one("td.team-info a.blue-black")
    if link is None:
        return ""
    return link.get_text(" ", strip=True)


def _extract_predicted_score(row) -> float | None:
    score_cells = row.select("td.score")
    if not score_cells:
        return None
    return _parse_numeric(score_cells[-1].get_text(" ", strip=True))


def _extract_value_cells(row) -> list[str]:
    return [cell.get_text(" ", strip=True) for cell in row.select("td.value")]


def fetch_page_html(gender: str, season: int, game_date: date, type1: str, type2: str) -> str:
    base_url = PAGE_URLS[gender].format(season=season)
    params = {"type1": type1, "type2": type2, "date": game_date.isoformat()}
    response = requests.get(
        f"{base_url}?{urlencode(params)}",
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_predict_winners_html(html: str, gender: str, season: int, game_date: date, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for box in soup.select("div.pbox[id]"):
        row1 = box.select_one("tr.pbox__info-team1-row")
        row2 = box.select_one("tr.pbox__info-team2-row")
        if row1 is None or row2 is None:
            continue

        team1_name = _extract_team_name(row1)
        team2_name = _extract_team_name(row2)
        if not team1_name or not team2_name:
            continue

        team1_score = _extract_predicted_score(row1)
        team2_score = _extract_predicted_score(row2)
        value_cells1 = _extract_value_cells(row1)
        value_cells2 = _extract_value_cells(row2)
        projected_total = _parse_numeric(value_cells1[0]) if len(value_cells1) >= 1 else None
        prob1 = _parse_numeric(value_cells1[1]) if len(value_cells1) >= 2 else None
        prob2 = _parse_numeric(value_cells2[1]) if len(value_cells2) >= 2 else None
        conf1 = value_cells1[2] if len(value_cells1) >= 3 else ""
        conf2 = value_cells2[2] if len(value_cells2) >= 3 else ""
        round_text = box.select_one("div.pbox__footer")
        round_text = round_text.get_text(" ", strip=True) if round_text is not None else ""
        time_cell = box.select_one(".time-clock")
        local_time = time_cell.get_text(" ", strip=True) if time_cell is not None else ""

        if projected_total is None and team1_score is not None and team2_score is not None:
            projected_total = float(team1_score + team2_score)
        if prob1 is None and prob2 is not None:
            prob1 = max(0.0, min(100.0, 100.0 - prob2))
        if prob2 is None and prob1 is not None:
            prob2 = max(0.0, min(100.0, 100.0 - prob1))
        if team1_score is not None and team2_score is not None:
            team1_spread = float(-(team1_score - team2_score))
        else:
            team1_spread = None

        rows.append(
            {
                "Season": int(season),
                "EventID": str(box.get("id", "")).strip(),
                "EventDate": game_date.isoformat(),
                "EventTimeLocal": local_time,
                "Team1Name": team1_name,
                "Team2Name": team2_name,
                "Prob": (float(prob1) / 100.0) if prob1 is not None else None,
                "Spread": team1_spread,
                "ProjectedTotal": projected_total,
                "RoundText": round_text,
                "Round": _parse_round_number(round_text),
                "Team1Confidence": conf1,
                "Team2Confidence": conf2,
                "SnapshotTime": fetched_at,
                "Source": "warrennolan_predict_winners",
                "SourceURL": source_url,
                "Gender": gender,
            }
        )
    return pd.DataFrame(rows)


def prepare_output(raw: pd.DataFrame, gender: str, fuzzy_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Team1ID",
                "Team2ID",
                "Team1Name",
                "Team2Name",
                "ModelProb",
                "ModelSpread",
                "ModelProjectedTotal",
                "ModelRound",
                "ModelRoundText",
                "ModelConfidence",
                "SnapshotTime",
                "Source",
                "SourceURL",
                "EventDate",
                "EventTimeLocal",
            ]
        ), None

    frame = attach_team_ids(raw, gender, "Team1Name", "Team2Name", fuzzy_threshold=fuzzy_threshold)
    audit_df = frame.attrs.get("team_match_audit")
    frame["Prob"] = pd.to_numeric(frame.get("Prob"), errors="coerce")
    frame["Spread"] = pd.to_numeric(frame.get("Spread"), errors="coerce")
    frame["ProjectedTotal"] = pd.to_numeric(frame.get("ProjectedTotal"), errors="coerce")
    frame["Round"] = pd.to_numeric(frame.get("Round"), errors="coerce")
    frame = frame.loc[frame["Prob"].notna() | frame["Spread"].notna()].copy()
    frame = canonicalize_matchups(frame)
    prepared = frame[
        [
            "Season",
            "Team1ID",
            "Team2ID",
            "Team1Name",
            "Team2Name",
            "Prob",
            "Spread",
            "ProjectedTotal",
            "Round",
            "RoundText",
            "Team1Confidence",
            "SnapshotTime",
            "Source",
            "SourceURL",
            "EventDate",
            "EventTimeLocal",
        ]
    ].copy()
    prepared = prepared.rename(
        columns={
            "Prob": "ModelProb",
            "Spread": "ModelSpread",
            "ProjectedTotal": "ModelProjectedTotal",
            "Round": "ModelRound",
            "RoundText": "ModelRoundText",
            "Team1Confidence": "ModelConfidence",
        }
    )
    prepared = prepared.drop_duplicates(subset=["Season", "Team1ID", "Team2ID", "EventDate"], keep="last")
    return prepared, audit_df


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch(
    gender: str,
    season: int,
    date_from: date,
    date_to: date,
    type1: str,
    type2: str,
    raw_dir: Path,
    output_dir: Path,
    unmatched_dir: Path,
    fuzzy_threshold: float,
) -> None:
    output_path = output_dir / f"{gender}WarrenNolanMatchupProjections_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}WarrenNolanMatchupProjections_{season}_unmatched.csv"
    fetched_frames: list[pd.DataFrame] = []
    raw_index: list[dict[str, object]] = []

    for game_date in _date_range(date_from, date_to):
        base_url = PAGE_URLS[gender].format(season=season)
        source_url = f"{base_url}?{urlencode({'type1': type1, 'type2': type2, 'date': game_date.isoformat()})}"
        html = fetch_page_html(gender, season, game_date, type1, type2)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"{gender}_{game_date.isoformat()}_{timestamp}.html"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")
        raw_index.append({"date": game_date.isoformat(), "url": source_url, "raw_path": str(raw_path)})
        fetched_frames.append(parse_predict_winners_html(html, gender, season, game_date, source_url))

    raw = pd.concat(fetched_frames, ignore_index=True) if fetched_frames else pd.DataFrame()
    prepared, audit_df = prepare_output(raw, gender, fuzzy_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)
    write_json(
        raw_dir / f"{gender}_{season}_{date_from.isoformat()}_{date_to.isoformat()}_index.json",
        {"gender": gender, "season": season, "type1": type1, "type2": type2, "fetches": raw_index},
    )

    print(f"[{gender}] source=warrennolan_predict_winners")
    print(f"[{gender}] raw_rows={len(raw)} cleaned_rows={len(prepared)}")
    print(f"[{gender}] saved csv -> {output_path}")
    summarize(prepared.rename(columns={"Team1ID": "Team1ID", "Team2ID": "Team2ID"}))
    if unmatched_path.exists():
        unresolved = pd.read_csv(unmatched_path)
        print(f"[{gender}] unmatched_audit={unmatched_path} rows={len(unresolved)}")


def main() -> None:
    args = parse_args()
    if not args.all and not args.gender:
        raise SystemExit("Pass --gender M/W or use --all.")

    genders = ["M", "W"] if args.all else [args.gender]
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    unmatched_dir = Path(args.unmatched_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir.mkdir(parents=True, exist_ok=True)

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)

    for gender in genders:
        run_fetch(
            gender=gender,
            season=args.season,
            date_from=date_from,
            date_to=date_to,
            type1=args.type1,
            type2=args.type2,
            raw_dir=raw_dir,
            output_dir=output_dir,
            unmatched_dir=unmatched_dir,
            fuzzy_threshold=float(args.fuzzy_threshold),
        )


if __name__ == "__main__":
    main()
