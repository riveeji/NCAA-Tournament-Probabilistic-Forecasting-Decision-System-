from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fetch_theoddsapi_odds import (
    EXTERNAL_DIR,
    RAW_DIR,
    SPORT_KEYS,
    flatten_events,
    prepare_kaggle_odds,
)
from tools.market_data_utils import DEFAULT_FUZZY_THRESHOLD, write_unmatched_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical NCAA tournament odds from The Odds API and write HistoricalOdds_TheOddsAPI_<season>.csv."
    )
    parser.add_argument("--season", type=int, required=True, help="Historical tournament season, for example 2025.")
    parser.add_argument("--gender", choices=["M", "W"], help="Competition gender prefix.")
    parser.add_argument("--all", action="store_true", help="Fetch both men's and women's historical odds.")
    parser.add_argument("--api-key", default="", help="The Odds API key. Defaults to THE_ODDS_API_KEY env var.")
    parser.add_argument("--regions", default="us", help="Bookmaker region list, for example us or us,us2.")
    parser.add_argument("--bookmakers", default="", help="Optional bookmaker keys, comma-separated.")
    parser.add_argument("--date-format", default="iso", choices=["iso", "unix"], help="The Odds API date format.")
    parser.add_argument("--odds-format", default="american", choices=["american", "decimal"], help="The Odds API odds format.")
    parser.add_argument(
        "--markets",
        default="h2h,spreads",
        help="Comma-separated featured markets to request from The Odds API.",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=45,
        help="Fetch each snapshot this many minutes before the known CommenceTime.",
    )
    parser.add_argument(
        "--query-granularity-minutes",
        type=int,
        default=15,
        help="Round derived query timestamps down to this many minutes to reduce duplicate requests.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on the number of historical API queries per gender for smoke/debug runs.",
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw JSON archives.")
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned historical CSV files.")
    parser.add_argument(
        "--unmatched-dir",
        default=str(EXTERNAL_DIR / "audit-logs"),
        help="Directory for unresolved team-name audit CSVs.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Minimum RapidFuzz token_set_ratio score required to auto-map a team name.",
    )
    return parser.parse_args()


def resolve_api_key(explicit: str) -> str:
    key = explicit.strip() if explicit else os.getenv("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing The Odds API key. Pass --api-key or set THE_ODDS_API_KEY.")
    return key


def fetch_historical_json(url: str) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "codex-ncaa-odds-fetcher/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"The Odds API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Unable to reach The Odds API: {exc}") from exc

    if isinstance(payload, dict):
        events = payload.get("data")
        if isinstance(events, list):
            return events, headers, payload
    if isinstance(payload, list):
        return payload, headers, {"data": payload}
    raise SystemExit("Unexpected historical The Odds API response shape.")


def build_historical_url(
    sport_key: str,
    api_key: str,
    regions: str,
    bookmakers: str,
    odds_format: str,
    date_format: str,
    markets: str,
    query_time: pd.Timestamp,
) -> str:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": query_time.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
    }
    if bookmakers.strip():
        params["bookmakers"] = bookmakers.strip()
    return f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds/?{urlencode(params)}"


def round_down_timestamp(ts: pd.Timestamp, granularity_minutes: int) -> pd.Timestamp:
    seconds = max(int(granularity_minutes), 1) * 60
    epoch = int(ts.tz_convert("UTC").timestamp())
    floored = epoch - (epoch % seconds)
    return pd.Timestamp(datetime.fromtimestamp(floored, tz=timezone.utc))


def known_commence_rows(gender: str, season: int) -> pd.DataFrame:
    path = EXTERNAL_DIR / f"{gender}HistoricalTournamentOdds.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "T1", "T2", "CommenceTime"])
    usecols = [column for column in ["Season", "T1", "T2", "CommenceTime"] if column in pd.read_csv(path, nrows=0).columns]
    frame = pd.read_csv(path, usecols=usecols)
    if "CommenceTime" not in frame.columns:
        return pd.DataFrame(columns=["Season", "T1", "T2", "CommenceTime"])
    frame = frame.loc[pd.to_numeric(frame["Season"], errors="coerce").eq(season)].copy()
    frame["CommenceTime"] = pd.to_datetime(frame["CommenceTime"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["CommenceTime"]).copy()
    return frame


def build_query_times(gender: str, season: int, lookback_minutes: int, granularity_minutes: int) -> list[pd.Timestamp]:
    commence = known_commence_rows(gender, season)
    if commence.empty:
        return []
    raw_times = commence["CommenceTime"] - timedelta(minutes=int(lookback_minutes))
    rounded = [round_down_timestamp(ts, granularity_minutes) for ts in raw_times]
    unique = sorted({ts for ts in rounded if pd.notna(ts)})
    return unique


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch(
    gender: str,
    season: int,
    api_key: str,
    regions: str,
    bookmakers: str,
    odds_format: str,
    date_format: str,
    markets: str,
    lookback_minutes: int,
    query_granularity_minutes: int,
    max_queries: int | None,
    raw_dir: Path,
    output_dir: Path,
    unmatched_dir: Path,
    fuzzy_threshold: float,
) -> None:
    query_times = build_query_times(gender, season, lookback_minutes, query_granularity_minutes)
    if not query_times:
        raise SystemExit(
            f"No usable CommenceTime rows found in external-data/{gender}HistoricalTournamentOdds.csv for season {season}."
        )
    if max_queries is not None:
        query_times = query_times[: int(max_queries)]

    sport_key = SPORT_KEYS[gender]
    all_flat = []
    last_headers: dict[str, str] = {}
    for query_time in query_times:
        url = build_historical_url(
            sport_key=sport_key,
            api_key=api_key,
            regions=regions,
            bookmakers=bookmakers,
            odds_format=odds_format,
            date_format=date_format,
            markets=markets,
            query_time=query_time,
        )
        events, headers, payload = fetch_historical_json(url)
        last_headers = headers
        raw_path = raw_dir / f"{gender}_historical_{season}_{query_time.strftime('%Y%m%dT%H%M%SZ')}.json"
        write_json(raw_path, payload)
        flat = flatten_events(events, season, markets)
        if flat.empty:
            continue
        flat["HistoricalQueryTime"] = query_time
        flat["SourceURL"] = url
        flat["Source"] = "theoddsapi_historical"
        all_flat.append(flat)

    if not all_flat:
        raise SystemExit(f"The Odds API historical fetch returned no rows for {gender} {season}.")

    flat_frame = pd.concat(all_flat, ignore_index=True)
    flat_frame = flat_frame.drop_duplicates(
        subset=[
            "Season",
            "EventID",
            "Book",
            "Team1Name",
            "Team2Name",
            "Team1Moneyline",
            "Team2Moneyline",
            "LastSpread",
            "SnapshotTime",
        ],
        keep="last",
    )
    prepared, audit_df = prepare_kaggle_odds(
        flat_frame,
        gender,
        keep_by_book=True,
        fuzzy_threshold=fuzzy_threshold,
    )
    if prepared.empty:
        raise SystemExit(f"No cleaned historical rows remained for {gender} {season}.")

    prepared["Source"] = "theoddsapi_historical"
    prepared["Season"] = int(season)
    output_path = output_dir / f"{gender}HistoricalOdds_TheOddsAPI_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}HistoricalOdds_TheOddsAPI_{season}_unmatched.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)

    print(f"[{gender}] sport={sport_key} season={season} queries={len(query_times)} rows={len(prepared)}")
    print(f"[{gender}] saved csv -> {output_path}")
    remaining = last_headers.get("x-requests-remaining")
    used = last_headers.get("x-requests-used")
    last_cost = last_headers.get("x-requests-last")
    print(f"[{gender}] quota remaining={remaining} used={used} last_cost={last_cost}")
    if unmatched_path.exists():
        unresolved = pd.read_csv(unmatched_path)
        print(f"[{gender}] unmatched_audit={unmatched_path} rows={len(unresolved)}")


def main() -> None:
    args = parse_args()
    if not args.all and not args.gender:
        raise SystemExit("Pass --gender M/W or use --all.")

    api_key = resolve_api_key(args.api_key)
    genders = ["M", "W"] if args.all else [args.gender]
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    unmatched_dir = Path(args.unmatched_dir)

    for gender in genders:
        run_fetch(
            gender=gender,
            season=int(args.season),
            api_key=api_key,
            regions=args.regions,
            bookmakers=args.bookmakers,
            odds_format=args.odds_format,
            date_format=args.date_format,
            markets=args.markets,
            lookback_minutes=int(args.lookback_minutes),
            query_granularity_minutes=int(args.query_granularity_minutes),
            max_queries=args.max_queries,
            raw_dir=raw_dir,
            output_dir=output_dir,
            unmatched_dir=unmatched_dir,
            fuzzy_threshold=float(args.fuzzy_threshold),
        )


if __name__ == "__main__":
    main()
