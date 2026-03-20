from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import sys

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
    "M": "https://barttorvik.com/schedule.php?date={date}&conlimit=All&year={season}",
    "W": "https://barttorvik.com/ncaaw/schedule.php?date={date}&conlimit=All&year={season}",
}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-barttorvik"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
HEADERS = {"User-Agent": "Mozilla/5.0"}
LINE_RE = re.compile(
    r"^(?P<winner>.+?)\s+(?P<line>Pick|PK|[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<score1>\d+)\s*-\s*(?P<score2>\d+)\s*\((?P<prob>\d+)%\)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Bart Torvik upcoming-game matchup projections into normalized HC model-signal CSVs."
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


def fetch_page_html(gender: str, season: int, game_date: date) -> str:
    url = PAGE_URLS[gender].format(season=season, date=game_date.strftime("%Y%m%d"))
    session = requests.Session()
    session.get(url, headers=HEADERS, timeout=30)
    response = session.post(
        url,
        data={"js_test_submitted": "1"},
        headers={**HEADERS, "Referer": url},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_line_text(line_text: str, team1_name: str, team2_name: str) -> tuple[float | None, float | None, float | None]:
    clean = " ".join(str(line_text or "").split()).strip()
    match = LINE_RE.match(clean)
    if match is None:
        return None, None, None

    winner = match.group("winner").strip()
    line_text_value = match.group("line").strip().lower()
    if line_text_value in {"pick", "pk"}:
        line_value = 0.0
    else:
        line_value = abs(float(line_text_value))

    score1 = float(match.group("score1"))
    score2 = float(match.group("score2"))
    winner_prob = float(match.group("prob")) / 100.0

    if winner == team1_name:
        team1_prob = winner_prob
        team1_spread = -line_value
    elif winner == team2_name:
        team1_prob = 1.0 - winner_prob
        team1_spread = line_value
    else:
        return None, None, None

    return team1_prob, team1_spread, score1 + score2


def parse_schedule_html(html: str, gender: str, season: int, game_date: date, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue

        matchup_cell = cells[1]
        team_links = matchup_cell.select("a[href*='team.php?team=']")
        if len(team_links) < 2:
            continue
        team1_name = team_links[0].get_text(" ", strip=True)
        team2_name = team_links[1].get_text(" ", strip=True)

        line_cell = cells[2]
        line_anchor = line_cell.find("a")
        line_text = line_anchor.get_text(" ", strip=True) if line_anchor is not None else line_cell.get_text(" ", strip=True)
        team1_prob, team1_spread, projected_total = parse_line_text(line_text, team1_name, team2_name)
        if team1_prob is None:
            continue

        time_cell = cells[0]
        matchup_title = matchup_cell.get("title", "")
        event_labels = [
            span.get_text(" ", strip=True)
            for span in matchup_cell.find_all("span")
            if "mobileout" not in (span.get("class") or []) and span.get_text(" ", strip=True)
        ]
        event_label = " | ".join(label for label in event_labels if label and label not in {"vs", "at"})

        ttq_text = cells[3].get_text(" ", strip=True)
        ttq_value = pd.to_numeric(ttq_text, errors="coerce")

        rows.append(
            {
                "Season": int(season),
                "EventDate": game_date.isoformat(),
                "EventTimeLocal": time_cell.get_text(" ", strip=True),
                "Team1Name": team1_name,
                "Team2Name": team2_name,
                "ModelProb": team1_prob,
                "ModelSpread": team1_spread,
                "ModelProjectedTotal": projected_total,
                "ModelRound": pd.NA,
                "ModelRoundText": event_label,
                "ModelConfidence": float(ttq_value) if pd.notna(ttq_value) else pd.NA,
                "VenueText": matchup_title,
                "SnapshotTime": fetched_at,
                "Source": "barttorvik_schedule",
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
                "VenueText",
            ]
        ), None

    frame = attach_team_ids(raw, gender, "Team1Name", "Team2Name", fuzzy_threshold=fuzzy_threshold)
    audit_df = frame.attrs.get("team_match_audit")
    frame["ModelProb"] = pd.to_numeric(frame["ModelProb"], errors="coerce")
    frame["ModelSpread"] = pd.to_numeric(frame["ModelSpread"], errors="coerce")
    frame["ModelProjectedTotal"] = pd.to_numeric(frame["ModelProjectedTotal"], errors="coerce")
    frame["ModelRound"] = pd.to_numeric(frame["ModelRound"], errors="coerce")
    frame = frame.loc[frame["ModelProb"].notna() | frame["ModelSpread"].notna()].copy()
    frame = canonicalize_matchups(frame)
    prepared = frame[
        [
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
            "VenueText",
        ]
    ].copy()
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
    raw_dir: Path,
    output_dir: Path,
    unmatched_dir: Path,
    fuzzy_threshold: float,
) -> None:
    output_path = output_dir / f"{gender}BartTorvikMatchupProjections_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}BartTorvikMatchupProjections_{season}_unmatched.csv"
    fetched_frames: list[pd.DataFrame] = []
    raw_index: list[dict[str, object]] = []

    for game_date in _date_range(date_from, date_to):
        source_url = PAGE_URLS[gender].format(season=season, date=game_date.strftime("%Y%m%d"))
        html = fetch_page_html(gender, season, game_date)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = raw_dir / f"{gender}_{game_date.isoformat()}_{timestamp}.html"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")
        raw_index.append({"date": game_date.isoformat(), "url": source_url, "raw_path": str(raw_path)})
        fetched_frames.append(parse_schedule_html(html, gender, season, game_date, source_url))

    raw = pd.concat(fetched_frames, ignore_index=True) if fetched_frames else pd.DataFrame()
    prepared, audit_df = prepare_output(raw, gender, fuzzy_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)
    write_json(
        raw_dir / f"{gender}_{season}_{date_from.isoformat()}_{date_to.isoformat()}_index.json",
        {"gender": gender, "season": season, "fetches": raw_index},
    )

    print(f"[{gender}] source=barttorvik_schedule")
    print(f"[{gender}] raw_rows={len(raw)} cleaned_rows={len(prepared)}")
    print(f"[{gender}] saved csv -> {output_path}")
    summarize(prepared)
    if unmatched_path.exists():
        try:
            unmatched_rows = len(pd.read_csv(unmatched_path))
        except Exception:
            unmatched_rows = 0
        print(f"[{gender}] unmatched_audit={unmatched_path} rows={unmatched_rows}")


def main() -> None:
    args = parse_args()
    genders = ["M", "W"] if args.all else [args.gender]
    if not genders or genders == [None]:
        raise SystemExit("Specify --gender M/W or --all.")

    date_from = pd.to_datetime(args.date_from).date()
    date_to = pd.to_datetime(args.date_to).date()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    unmatched_dir = Path(args.unmatched_dir)

    for gender in genders:
        run_fetch(
            gender,
            args.season,
            date_from,
            date_to,
            raw_dir,
            output_dir,
            unmatched_dir,
            args.fuzzy_threshold,
        )


if __name__ == "__main__":
    main()
