from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import attach_team_ids, canonicalize_matchups, write_unmatched_log

EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"

LOGIN_URL = "https://herhoopstats.com/accounts/login/?return_url=/"
SCHEDULE_URL_TEMPLATE = "https://herhoopstats.com/stats/ncaa/schedule_date/{year}/{month}/{day}/d1/"
LOBOS_ROOT_URL = "https://herhoopstats.com/stats/lobos_look/ncaa/team/"
LOBOS_URL_TEMPLATE = "https://herhoopstats.com/stats/lobos_look/ncaa/team/{season}/d1/{team1_slug}/{team2_slug}/"
USER_AGENT = "Mozilla/5.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Her Hoop Stats subscribed women NCAA matchup projections.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--email", default=os.environ.get("HERHOOPSTATS_EMAIL", "").strip())
    parser.add_argument("--password", default=os.environ.get("HERHOOPSTATS_PASSWORD", "").strip())
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--lobos-source", default="")
    parser.add_argument("--augment-lobos", dest="augment_lobos", action="store_true")
    parser.add_argument("--no-augment-lobos", dest="augment_lobos", action="store_false")
    parser.add_argument("--output", default="")
    parser.set_defaults(augment_lobos=True)
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return " ".join(str(name).replace("\xa0", " ").split()).strip()


def normalize_numeric_text(text: str) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip()


def parse_winner_prob(text: str) -> tuple[str, float]:
    clean = normalize_name(text)
    if "(" not in clean or "%" not in clean:
        return clean, float("nan")
    winner = clean.split("(")[0].strip()
    prob_text = clean.split("(")[-1].replace(")", "").replace("%", "").strip()
    return winner, float(prob_text) / 100.0


def parse_score(text: str) -> tuple[float, float]:
    clean = normalize_name(text).replace(" - ", "|").replace("-", "|")
    parts = [part.strip() for part in clean.split("|") if part.strip()]
    if len(parts) < 2:
        return float("nan"), float("nan")
    return float(parts[0]), float(parts[1])


def parse_pct(text: str) -> float:
    clean = normalize_numeric_text(text).replace("%", "")
    if not clean:
        return float("nan")
    return float(clean) / 100.0


def parse_signed_float(text: str) -> float:
    clean = normalize_numeric_text(text).replace("+", "")
    if not clean:
        return float("nan")
    return float(clean)


def login(email: str, password: str) -> requests.Session:
    if not email or not password:
        raise RuntimeError("Her Hoop Stats credentials are required via args or env vars.")
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Referer": LOGIN_URL}
    page = session.get(LOGIN_URL, headers=headers, timeout=30)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    form = soup.find("form")
    token_input = form.find("input", {"name": "csrfmiddlewaretoken"}) if form else None
    if token_input is None:
        raise RuntimeError("Failed to locate Her Hoop Stats CSRF token.")
    payload = {
        "csrfmiddlewaretoken": token_input.get("value", ""),
        "email": email,
        "password": password,
        "return_url": "/",
    }
    response = session.post(LOGIN_URL, headers=headers, data=payload, timeout=30)
    response.raise_for_status()
    if "logged_in': 1" not in response.text and '"logged_in": 1' not in response.text and "logged_in\":1" not in response.text:
        raise RuntimeError("Her Hoop Stats login did not enter a logged-in state.")
    if "subscribed': 1" not in response.text and '"subscribed": 1' not in response.text and "subscribed\":1" not in response.text:
        raise RuntimeError("Her Hoop Stats login succeeded but subscription access is not active.")
    return session


def fetch_schedule_page(session: requests.Session, target_date: date) -> pd.DataFrame:
    url = SCHEDULE_URL_TEMPLATE.format(year=target_date.year, month=target_date.month, day=target_date.day)
    response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_=lambda value: value and "schedule_date" in value)
    if table is None or table.find("tbody") is None:
        return pd.DataFrame()

    snapshot_time = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 9:
            continue
        matchup_cell = cells[1]
        team_links = matchup_cell.find_all("a")
        if len(team_links) < 2:
            continue
        team1_name = normalize_name(team_links[0].get_text(" ", strip=True))
        team2_name = normalize_name(team_links[1].get_text(" ", strip=True))

        winner_cell = cells[-4]
        score_cell = cells[-3]
        margin_cell = cells[-2]
        total_cell = cells[-1]

        winner_text = winner_cell.get_text(" ", strip=True)
        winner_name, winner_prob = parse_winner_prob(winner_text)

        score_text = score_cell.get_text(" ", strip=True)
        score1, score2 = parse_score(score_text)
        margin = score2 - score1 if pd.notna(score1) and pd.notna(score2) else float("nan")
        total = score1 + score2 if pd.notna(score1) and pd.notna(score2) else float("nan")
        margin_text = normalize_name(margin_cell.get_text(" ", strip=True))
        if margin_text:
            try:
                parsed_margin = float(margin_text.replace("+", ""))
                if normalize_name(winner_name).lower() == normalize_name(team1_name).lower():
                    margin = -abs(parsed_margin)
                elif normalize_name(winner_name).lower() == normalize_name(team2_name).lower():
                    margin = abs(parsed_margin)
            except Exception:
                pass
        total_text = normalize_name(total_cell.get_text(" ", strip=True))
        if total_text:
            try:
                total = float(total_text)
            except Exception:
                pass

        team1_prob = float("nan")
        if pd.notna(winner_prob):
            if normalize_name(winner_name).lower() == normalize_name(team1_name).lower():
                team1_prob = float(winner_prob)
            elif normalize_name(winner_name).lower() == normalize_name(team2_name).lower():
                team1_prob = float(1.0 - winner_prob)

        rows.append(
            {
                "Season": int(target_date.year),
                "Team1Name": team1_name,
                "Team2Name": team2_name,
                "HerHoopProb": team1_prob,
                "HerHoopSpread": margin,
                "HerHoopProjectedTotal": total,
                "EventDate": target_date.isoformat(),
                "SnapshotTime": snapshot_time,
                "Source": "Her Hoop Stats Schedule Projections",
                "SourceURL": url,
            }
        )
    return pd.DataFrame(rows)


def fetch_lobos_team_slug_map(session: requests.Session) -> dict[int, dict[str, str]]:
    response = session.get(LOBOS_ROOT_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    options = [option for option in soup.find_all("option") if option.get("value")]
    if not options:
        return {}

    from zizzii_features import build_team_name_lookup, resolve_team_id

    lookup = build_team_name_lookup("W")
    slug_map: dict[int, dict[str, str]] = {}
    for option in options:
        value = str(option.get("value", "")).strip()
        raw_text = str(option.get_text(" ", strip=True)).strip()
        text = normalize_name(raw_text)
        if not value or text in {"", "team"}:
            continue
        slug_text = re.sub(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", "", value).replace("-", " ")
        team_id = None
        for candidate in [slug_text, raw_text, text]:
            team_id = resolve_team_id(candidate, lookup)
            if team_id is not None:
                break
        if team_id is None:
            continue
        slug_map[int(team_id)] = {"slug": value, "label": raw_text}
    return slug_map


def load_lobos_targets(path: Path, season: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Season", "Team1ID", "Team2ID", "Team1Name", "Team2Name"])
    frame = pd.read_csv(path)
    rename_map = {}
    if {"T1", "T2"}.issubset(frame.columns):
        rename_map.update({"T1": "Team1ID", "T2": "Team2ID"})
    frame = frame.rename(columns=rename_map)
    required = {"Team1ID", "Team2ID"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["Season", "Team1ID", "Team2ID", "Team1Name", "Team2Name"])
    if "Season" not in frame.columns:
        frame["Season"] = season
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["Team1ID"] = pd.to_numeric(frame["Team1ID"], errors="coerce")
    frame["Team2ID"] = pd.to_numeric(frame["Team2ID"], errors="coerce")
    frame = frame.dropna(subset=["Season", "Team1ID", "Team2ID"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["Team1ID"] = frame["Team1ID"].astype(int)
    frame["Team2ID"] = frame["Team2ID"].astype(int)
    frame = frame.loc[frame["Season"].eq(int(season))].copy()
    for col in ["Team1Name", "Team2Name"]:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str)
    return frame[["Season", "Team1ID", "Team2ID", "Team1Name", "Team2Name"]].drop_duplicates(
        subset=["Season", "Team1ID", "Team2ID"], keep="last"
    )


def parse_lobos_prediction_table(html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    header = next(
        (
            node
            for node in soup.find_all(["h1", "h2", "h3"])
            if normalize_name(node.get_text(" ", strip=True)).lower() == "matchup predictions"
        ),
        None,
    )
    if header is None:
        return []
    table = header.find_next("table")
    if table is None:
        return []

    rows: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 8:
            continue
        location = normalize_name(cells[0]).title()
        team1_prob = parse_pct(cells[1])
        team2_prob = parse_pct(cells[2])
        team1_points = pd.to_numeric(cells[3], errors="coerce")
        team2_points = pd.to_numeric(cells[4], errors="coerce")
        margin1 = parse_signed_float(cells[5])
        total = pd.to_numeric(cells[7], errors="coerce")
        rows.append(
            {
                "Location": location,
                "Team1Prob": team1_prob,
                "Team2Prob": team2_prob,
                "Team1Points": float(team1_points) if pd.notna(team1_points) else float("nan"),
                "Team2Points": float(team2_points) if pd.notna(team2_points) else float("nan"),
                "Team1Margin": margin1,
                "ProjectedTotal": float(total) if pd.notna(total) else float("nan"),
            }
        )
    return rows


def fetch_lobos_matchup_page(
    session: requests.Session,
    *,
    season: int,
    team1_slug: str,
    team2_slug: str,
    team1_id: int,
    team2_id: int,
    team1_name: str,
    team2_name: str,
) -> Optional[dict[str, object]]:
    url = LOBOS_URL_TEMPLATE.format(season=season, team1_slug=team1_slug, team2_slug=team2_slug)
    response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    predictions = parse_lobos_prediction_table(response.text)
    if not predictions:
        return None
    preferred = next((row for row in predictions if row["Location"] == "Neutral"), predictions[0])
    return {
        "Season": int(season),
        "Team1ID": int(team1_id),
        "Team2ID": int(team2_id),
        "Team1Name": team1_name,
        "Team2Name": team2_name,
        "HerHoopProb": preferred["Team1Prob"],
        "HerHoopSpread": -preferred["Team1Margin"] if pd.notna(preferred["Team1Margin"]) else float("nan"),
        "HerHoopProjectedTotal": preferred["ProjectedTotal"],
        "EventDate": pd.NaT,
        "SnapshotTime": datetime.now(timezone.utc).isoformat(),
        "Source": "Her Hoop Stats Lobo's Look",
        "SourceURL": url,
    }


def fetch_lobos_matchups(session: requests.Session, season: int, source_path: Path) -> pd.DataFrame:
    targets = load_lobos_targets(source_path, season)
    if targets.empty:
        return pd.DataFrame()

    slug_map = fetch_lobos_team_slug_map(session)
    rows: list[dict[str, object]] = []
    for row in targets.itertuples(index=False):
        team1_meta = slug_map.get(int(row.Team1ID))
        team2_meta = slug_map.get(int(row.Team2ID))
        if team1_meta is None or team2_meta is None:
            continue
        fetched = fetch_lobos_matchup_page(
            session,
            season=int(season),
            team1_slug=team1_meta["slug"],
            team2_slug=team2_meta["slug"],
            team1_id=int(row.Team1ID),
            team2_id=int(row.Team2ID),
            team1_name=str(row.Team1Name) or team1_meta["label"],
            team2_name=str(row.Team2Name) or team2_meta["label"],
        )
        if fetched is not None:
            rows.append(fetched)
    return pd.DataFrame(rows)


def attach_ids_and_finalize(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Team1ID",
                "Team2ID",
                "Team1Name",
                "Team2Name",
                "HerHoopProb",
                "HerHoopSpread",
                "HerHoopProjectedTotal",
                "EventDate",
                "SnapshotTime",
                "Source",
                "SourceURL",
            ]
        )
    frame = frame.copy()
    frame["Season"] = int(season)
    frame = attach_team_ids(frame, "W", "Team1Name", "Team2Name", "Team1ID", "Team2ID")
    audit_df = frame.attrs.get("team_match_audit")
    frame = canonicalize_matchups(frame)
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce").astype("Int64")
    frame["Team1ID"] = pd.to_numeric(frame["Team1ID"], errors="coerce").astype("Int64")
    frame["Team2ID"] = pd.to_numeric(frame["Team2ID"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["Season", "Team1ID", "Team2ID"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["Team1ID"] = frame["Team1ID"].astype(int)
    frame["Team2ID"] = frame["Team2ID"].astype(int)
    frame["HerHoopProb"] = pd.to_numeric(frame["HerHoopProb"], errors="coerce")
    frame["HerHoopSpread"] = pd.to_numeric(frame["HerHoopSpread"], errors="coerce")
    frame["HerHoopProjectedTotal"] = pd.to_numeric(frame["HerHoopProjectedTotal"], errors="coerce")
    frame["EventDate"] = pd.to_datetime(frame["EventDate"], errors="coerce")
    frame["SnapshotTime"] = pd.to_datetime(frame["SnapshotTime"], errors="coerce", utc=True)
    frame = frame[
        [
            "Season",
            "Team1ID",
            "Team2ID",
            "Team1Name",
            "Team2Name",
            "HerHoopProb",
            "HerHoopSpread",
            "HerHoopProjectedTotal",
            "EventDate",
            "SnapshotTime",
            "Source",
            "SourceURL",
        ]
    ].drop_duplicates(subset=["Season", "Team1ID", "Team2ID"], keep="last")
    frame.attrs["team_match_audit"] = audit_df
    return frame.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    session = login(args.email.strip(), args.password.strip())
    today = datetime.now().date()
    start_date = date.fromisoformat(args.date_from) if args.date_from else today
    end_date = date.fromisoformat(args.date_to) if args.date_to else (today + timedelta(days=7))

    frames: list[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        frame = fetch_schedule_page(session, current)
        if not frame.empty:
            frames.append(frame)
        current += timedelta(days=1)

    if args.augment_lobos:
        lobos_source = Path(args.lobos_source) if args.lobos_source else EXTERNAL_DIR / f"WMatchupOdds_{args.season}.csv"
        lobos_frame = fetch_lobos_matchups(session, args.season, lobos_source)
        if not lobos_frame.empty:
            frames.append(lobos_frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    final = attach_ids_and_finalize(combined, args.season)
    output_path = Path(args.output) if args.output else EXTERNAL_DIR / f"WHerHoopStatsMatchupProjections_{args.season}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    audit_df = final.attrs.get("team_match_audit")
    audit_path = RESULTS_DIR / f"{output_path.stem}_unmatched.csv"
    write_unmatched_log(audit_df, audit_path)

    print(
        {
            "output": str(output_path),
            "rows": int(len(final)),
            "date_from": start_date.isoformat(),
            "date_to": end_date.isoformat(),
            "unmatched_log": str(audit_path) if audit_path.exists() else "",
        }
    )


if __name__ == "__main__":
    main()
