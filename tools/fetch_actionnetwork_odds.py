from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import (
    DEFAULT_FUZZY_THRESHOLD,
    attach_team_ids,
    canonicalize_matchups,
    no_vig_prob,
    summarize,
    write_unmatched_log,
)


PAGE_URLS = {
    "M": "https://www.actionnetwork.com/ncaab/odds",
    "W": "https://www.actionnetwork.com/ncaaw/odds",
}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-actionnetwork"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current NCAA men/women odds from Action Network scoreboard pages."
    )
    parser.add_argument("--gender", choices=["M", "W"], help="Competition gender prefix.")
    parser.add_argument("--all", action="store_true", help="Fetch both men's and women's odds.")
    parser.add_argument("--season", type=int, default=2026, help="Season value to write into output.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw JSON archives.")
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


def _american_to_prob(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    neg = numeric < 0
    out.loc[neg] = -numeric.loc[neg] / (-numeric.loc[neg] + 100.0)
    out.loc[~neg] = 100.0 / (numeric.loc[~neg] + 100.0)
    return out


def fetch_page_props(url: str) -> dict[str, object]:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text)
    if match is None:
        raise RuntimeError(f"Unable to locate __NEXT_DATA__ payload for {url}")
    payload = json.loads(match.group(1))
    props = payload.get("props", {})
    page_props = props.get("pageProps", {})
    if not isinstance(page_props, dict):
        raise RuntimeError(f"Unexpected Action Network pageProps shape for {url}")
    return page_props


def choose_team_name(team: dict[str, object]) -> str:
    for field in ["display_name", "full_name", "location", "short_name"]:
        value = str(team.get(field, "")).strip()
        if value:
            return value
    return ""


def parse_moneyline(event_markets: dict[str, object]) -> tuple[float | None, float | None]:
    home_odds = None
    away_odds = None
    for item in event_markets.get("moneyline") or []:
        if item.get("is_alt_market"):
            continue
        if str(item.get("period", "event")) != "event":
            continue
        side = str(item.get("side", "")).strip().lower()
        odds = item.get("odds")
        if side == "home":
            home_odds = odds
        elif side == "away":
            away_odds = odds
    return home_odds, away_odds


def parse_spread(event_markets: dict[str, object]) -> float | None:
    home_spread = None
    away_spread = None
    for item in event_markets.get("spread") or []:
        if item.get("is_alt_market"):
            continue
        if str(item.get("period", "event")) != "event":
            continue
        side = str(item.get("side", "")).strip().lower()
        value = item.get("value")
        if side == "home":
            home_spread = value
        elif side == "away":
            away_spread = value
    if home_spread is not None:
        return float(home_spread)
    if away_spread is not None:
        return float(-float(away_spread))
    return None


def flatten_scoreboard(page_props: dict[str, object], gender: str, season: int) -> pd.DataFrame:
    scoreboard = page_props.get("scoreboardResponse") or {}
    if not isinstance(scoreboard, dict):
        return pd.DataFrame()
    book_lookup = page_props.get("allBooks") or {}
    if not isinstance(book_lookup, dict):
        book_lookup = {}
    rows: list[dict[str, object]] = []
    for game in scoreboard.get("games") or []:
        if not isinstance(game, dict):
            continue
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")
        teams = {int(team["id"]): team for team in (game.get("teams") or []) if isinstance(team, dict) and team.get("id") is not None}
        home_team = teams.get(int(home_team_id)) if home_team_id is not None else None
        away_team = teams.get(int(away_team_id)) if away_team_id is not None else None
        if home_team is None or away_team is None:
            continue
        home_name = choose_team_name(home_team)
        away_name = choose_team_name(away_team)
        if not home_name or not away_name:
            continue

        markets_by_book = game.get("markets") or {}
        if not isinstance(markets_by_book, dict):
            continue
        for book_id, payload in markets_by_book.items():
            if not isinstance(payload, dict):
                continue
            event_markets = payload.get("event") or {}
            if not isinstance(event_markets, dict):
                continue
            team1_moneyline, team2_moneyline = parse_moneyline(event_markets)
            last_spread = parse_spread(event_markets)
            if team1_moneyline is None and team2_moneyline is None and last_spread is None:
                continue
            book = book_lookup.get(str(book_id), {})
            book_name = str(book.get("source_name") or book.get("display_name") or f"book_{book_id}").strip()
            rows.append(
                {
                    "Season": season,
                    "EventID": game.get("id"),
                    "CommenceTime": game.get("start_time"),
                    "SnapshotTime": game.get("start_time"),
                    "Team1Name": home_name,
                    "Team2Name": away_name,
                    "Team1Moneyline": team1_moneyline,
                    "Team2Moneyline": team2_moneyline,
                    "LastSpread": last_spread,
                    "Book": book_name,
                    "Source": "actionnetwork_live",
                    "SourceURL": PAGE_URLS[gender],
                }
            )
    return pd.DataFrame(rows)


def prepare_output(raw: pd.DataFrame, gender: str, fuzzy_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "Team1Name",
                "Team2Name",
                "Team1Moneyline",
                "Team2Moneyline",
                "Team1ImpliedProb",
                "Team2ImpliedProb",
                "NoVigProb",
                "MarketProb",
                "Hold",
                "LastSpread",
                "Book",
                "BookCount",
                "SnapshotTime",
                "Source",
                "SourceURL",
            ]
        ), None

    frame = attach_team_ids(raw, gender, "Team1Name", "Team2Name", fuzzy_threshold=fuzzy_threshold)
    audit_df = frame.attrs.get("team_match_audit")
    frame["Team1Moneyline"] = pd.to_numeric(frame.get("Team1Moneyline"), errors="coerce")
    frame["Team2Moneyline"] = pd.to_numeric(frame.get("Team2Moneyline"), errors="coerce")
    frame["LastSpread"] = pd.to_numeric(frame.get("LastSpread"), errors="coerce")

    has_h2h = frame["Team1Moneyline"].notna() & frame["Team2Moneyline"].notna()
    frame["Team1ImpliedProb"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["Team2ImpliedProb"] = pd.Series(np.nan, index=frame.index, dtype=float)
    if has_h2h.any():
        p1 = _american_to_prob(frame.loc[has_h2h, "Team1Moneyline"])
        p2 = _american_to_prob(frame.loc[has_h2h, "Team2Moneyline"])
        frame.loc[has_h2h, "Team1ImpliedProb"] = p1.to_numpy()
        frame.loc[has_h2h, "Team2ImpliedProb"] = p2.to_numpy()
        no_vig, hold = no_vig_prob(p1, p2)
        frame["NoVigProb"] = pd.Series(np.nan, index=frame.index, dtype=float)
        frame["Hold"] = pd.Series(np.nan, index=frame.index, dtype=float)
        frame.loc[has_h2h, "NoVigProb"] = no_vig.to_numpy()
        frame.loc[has_h2h, "Hold"] = hold.to_numpy()
    else:
        frame["NoVigProb"] = pd.Series(np.nan, index=frame.index, dtype=float)
        frame["Hold"] = pd.Series(np.nan, index=frame.index, dtype=float)

    frame = frame.loc[has_h2h | frame["LastSpread"].notna()].copy()
    frame = canonicalize_matchups(frame)
    prepared = frame[
        [
            "Season",
            "Team1ID",
            "Team2ID",
            "Team1Name",
            "Team2Name",
            "Team1Moneyline",
            "Team2Moneyline",
            "Team1ImpliedProb",
            "Team2ImpliedProb",
            "NoVigProb",
            "Hold",
            "LastSpread",
            "Book",
            "SnapshotTime",
            "Source",
            "SourceURL",
        ]
    ].copy()
    prepared = prepared.rename(columns={"Team1ID": "T1", "Team2ID": "T2"})
    prepared["MarketProb"] = prepared["NoVigProb"]
    prepared["BookCount"] = 1.0
    return prepared, audit_df


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch(gender: str, season: int, raw_dir: Path, output_dir: Path, unmatched_dir: Path, fuzzy_threshold: float) -> None:
    page_props = fetch_page_props(PAGE_URLS[gender])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{gender}_{timestamp}.json"
    output_path = output_dir / f"{gender}ActionNetworkOdds_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}ActionNetworkOdds_{season}_unmatched.csv"

    write_json(
        raw_path,
        {
            "source_url": PAGE_URLS[gender],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "scoreboardResponse": page_props.get("scoreboardResponse", {}),
            "allBooks": page_props.get("allBooks", {}),
        },
    )

    raw = flatten_scoreboard(page_props, gender, season)
    prepared, audit_df = prepare_output(raw, gender, fuzzy_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)

    print(f"[{gender}] source=actionnetwork_live")
    print(f"[{gender}] raw_rows={len(raw)} cleaned_rows={len(prepared)}")
    print(f"[{gender}] saved raw -> {raw_path}")
    print(f"[{gender}] saved csv -> {output_path}")
    summarize(prepared.rename(columns={"T1": "Team1ID", "T2": "Team2ID"}))
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

    for gender in genders:
        run_fetch(
            gender=gender,
            season=args.season,
            raw_dir=raw_dir,
            output_dir=output_dir,
            unmatched_dir=unmatched_dir,
            fuzzy_threshold=float(args.fuzzy_threshold),
        )


if __name__ == "__main__":
    main()
