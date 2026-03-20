from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

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
from zizzii_features import normalize_team_name


SPORT_KEYS = {
    "M": "basketball_ncaab",
    "W": "basketball_wncaab",
}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-theoddsapi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NCAA current moneyline odds from The Odds API and write Kaggle-ready matchup files."
    )
    parser.add_argument("--gender", choices=["M", "W"], help="Competition gender prefix.")
    parser.add_argument("--all", action="store_true", help="Fetch both men's and women's odds.")
    parser.add_argument("--api-key", default="", help="The Odds API key. Defaults to THE_ODDS_API_KEY env var.")
    parser.add_argument("--regions", default="us", help="Bookmaker region list, for example us or us,us2.")
    parser.add_argument("--bookmakers", default="", help="Optional bookmaker keys, comma-separated.")
    parser.add_argument("--season", type=int, default=2026, help="Season value to write into output.")
    parser.add_argument("--date-format", default="iso", choices=["iso", "unix"], help="The Odds API date format.")
    parser.add_argument("--odds-format", default="american", choices=["american", "decimal"], help="The Odds API odds format.")
    parser.add_argument(
        "--markets",
        default="h2h,spreads",
        help="Comma-separated featured markets to request from The Odds API.",
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw JSON archives.")
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned Kaggle-ready CSV files.")
    parser.add_argument("--keep-by-book", action="store_true", help="Keep one row per book instead of averaging by matchup.")
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Minimum RapidFuzz token_set_ratio score required to auto-map a team name.",
    )
    parser.add_argument(
        "--unmatched-dir",
        default=str(EXTERNAL_DIR / "audit-logs"),
        help="Directory for unresolved team-name audit CSVs.",
    )
    return parser.parse_args()


def resolve_api_key(explicit: str) -> str:
    key = explicit.strip() if explicit else os.getenv("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing The Odds API key. Pass --api-key or set THE_ODDS_API_KEY.")
    return key


def fetch_json(url: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    request = Request(url, headers={"User-Agent": "codex-ncaa-odds-fetcher/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            if not isinstance(payload, list):
                raise RuntimeError("Unexpected The Odds API response shape: expected a list of events.")
            return payload, headers
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"The Odds API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Unable to reach The Odds API: {exc}") from exc


def _market_team_map(event: dict[str, Any]) -> tuple[str, str, dict[str, tuple[str, str]]]:
    home_team = str(event.get("home_team", "")).strip()
    away_team = str(event.get("away_team", "")).strip()
    if not home_team or not away_team:
        return "", "", {}
    desired = {
        normalize_team_name(home_team): ("Team1Name", "Team1Moneyline"),
        normalize_team_name(away_team): ("Team2Name", "Team2Moneyline"),
    }
    return home_team, away_team, desired


def normalize_moneyline_market(event: dict[str, Any], market: dict[str, Any]) -> Optional[dict[str, Any]]:
    home_team, away_team, desired = _market_team_map(event)
    if not desired:
        return None
    row = {
        "Team1Name": home_team,
        "Team2Name": away_team,
        "Team1Moneyline": None,
        "Team2Moneyline": None,
    }
    for outcome in market.get("outcomes", []):
        normalized = normalize_team_name(outcome.get("name"))
        target = desired.get(normalized)
        if target is None:
            continue
        _, price_key = target
        row[price_key] = outcome.get("price")

    if row["Team1Moneyline"] is None or row["Team2Moneyline"] is None:
        return None
    return row


def normalize_spread_market(event: dict[str, Any], market: dict[str, Any]) -> Optional[dict[str, Any]]:
    home_team, away_team, desired = _market_team_map(event)
    if not desired:
        return None
    row = {
        "Team1Name": home_team,
        "Team2Name": away_team,
        "LastSpread": None,
    }
    for outcome in market.get("outcomes", []):
        normalized = normalize_team_name(outcome.get("name"))
        target = desired.get(normalized)
        if target is None:
            continue
        team_name_key, _ = target
        if team_name_key == "Team1Name":
            row["LastSpread"] = outcome.get("point")
            break
    if row["LastSpread"] is None:
        return None
    return row


def flatten_events(events: list[dict[str, Any]], season: int, markets_requested: str) -> pd.DataFrame:
    requested = {token.strip() for token in str(markets_requested).split(",") if token.strip()}
    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id", ""))
        commence_time = str(event.get("commence_time", ""))
        sport_key = str(event.get("sport_key", ""))
        for bookmaker in event.get("bookmakers", []) or []:
            book_key = str(bookmaker.get("key", "unknown"))
            book_title = str(bookmaker.get("title", book_key))
            row = {
                "Season": season,
                "EventID": event_id,
                "SportKey": sport_key,
                "CommenceTime": commence_time,
                "SnapshotTime": str(bookmaker.get("last_update") or commence_time),
                "Book": book_key,
                "BookTitle": book_title,
                "MarketsRequested": ",".join(sorted(requested)),
                "Team1Name": str(event.get("home_team", "")).strip(),
                "Team2Name": str(event.get("away_team", "")).strip(),
                "Team1Moneyline": None,
                "Team2Moneyline": None,
                "LastSpread": None,
            }
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", "")).strip()
                if market_key == "h2h" and "h2h" in requested:
                    normalized = normalize_moneyline_market(event, market)
                    if normalized is not None:
                        row.update(normalized)
                        row["SnapshotTime"] = str(market.get("last_update") or row["SnapshotTime"])
                elif market_key == "spreads" and "spreads" in requested:
                    normalized = normalize_spread_market(event, market)
                    if normalized is not None:
                        row["LastSpread"] = normalized["LastSpread"]
                        row["SnapshotTime"] = str(market.get("last_update") or row["SnapshotTime"])
            if row["Team1Moneyline"] is None and row["Team2Moneyline"] is None and row["LastSpread"] is None:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_kaggle_odds(
    flat: pd.DataFrame,
    gender: str,
    keep_by_book: bool,
    fuzzy_threshold: float,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if flat.empty:
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
                "SnapshotTime",
            ]
        ), None

    df = flat.copy()
    df = attach_team_ids(df, gender, "Team1Name", "Team2Name", None, None, fuzzy_threshold=fuzzy_threshold)
    audit_df = df.attrs.get("team_match_audit")
    df["Team1Moneyline"] = pd.to_numeric(df.get("Team1Moneyline"), errors="coerce")
    df["Team2Moneyline"] = pd.to_numeric(df.get("Team2Moneyline"), errors="coerce")
    df["LastSpread"] = pd.to_numeric(df.get("LastSpread"), errors="coerce")

    has_h2h = df["Team1Moneyline"].notna() & df["Team2Moneyline"].notna()
    df["Team1ImpliedProb"] = pd.Series(np.nan, index=df.index, dtype=float)
    df["Team2ImpliedProb"] = pd.Series(np.nan, index=df.index, dtype=float)
    if has_h2h.any():
        favored = df.loc[has_h2h, "Team1Moneyline"] < 0
        df.loc[has_h2h, "Team1ImpliedProb"] = 100.0 / (df.loc[has_h2h, "Team1Moneyline"] + 100.0)
        df.loc[has_h2h & favored, "Team1ImpliedProb"] = -df.loc[has_h2h & favored, "Team1Moneyline"] / (
            -df.loc[has_h2h & favored, "Team1Moneyline"] + 100.0
        )
        favored = df.loc[has_h2h, "Team2Moneyline"] < 0
        df.loc[has_h2h, "Team2ImpliedProb"] = 100.0 / (df.loc[has_h2h, "Team2Moneyline"] + 100.0)
        df.loc[has_h2h & favored, "Team2ImpliedProb"] = -df.loc[has_h2h & favored, "Team2Moneyline"] / (
            -df.loc[has_h2h & favored, "Team2Moneyline"] + 100.0
        )
        no_vig, hold = no_vig_prob(df.loc[has_h2h, "Team1ImpliedProb"], df.loc[has_h2h, "Team2ImpliedProb"])
        df["NoVigProb"] = pd.Series(np.nan, index=df.index, dtype=float)
        df["Hold"] = pd.Series(np.nan, index=df.index, dtype=float)
        df.loc[has_h2h, "NoVigProb"] = no_vig.to_numpy()
        df.loc[has_h2h, "Hold"] = hold.to_numpy()
    else:
        df["NoVigProb"] = pd.Series(np.nan, index=df.index, dtype=float)
        df["Hold"] = pd.Series(np.nan, index=df.index, dtype=float)

    df = df.loc[has_h2h | df["LastSpread"].notna()].copy()

    df = canonicalize_matchups(df)
    prepared = df[
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
        ]
    ].copy()
    prepared = prepared.rename(columns={"Team1ID": "T1", "Team2ID": "T2"})
    prepared["MarketProb"] = prepared["NoVigProb"]

    if keep_by_book:
        return prepared, audit_df

    prepared = prepared.groupby(["Season", "T1", "T2"], as_index=False).agg(
        Team1Name=("Team1Name", "first"),
        Team2Name=("Team2Name", "first"),
        Team1Moneyline=("Team1Moneyline", "mean"),
        Team2Moneyline=("Team2Moneyline", "mean"),
        Team1ImpliedProb=("Team1ImpliedProb", "mean"),
        Team2ImpliedProb=("Team2ImpliedProb", "mean"),
        NoVigProb=("NoVigProb", "mean"),
        MarketProb=("MarketProb", "mean"),
        Hold=("Hold", "mean"),
        LastSpread=("LastSpread", "mean"),
        Book=("Book", lambda values: "|".join(sorted(set(map(str, values))))),
        SnapshotTime=("SnapshotTime", lambda values: max(map(str, values))),
    )
    return prepared, audit_df


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_url(sport_key: str, api_key: str, regions: str, bookmakers: str, odds_format: str, date_format: str, markets: str) -> str:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers.strip():
        params["bookmakers"] = bookmakers.strip()
    return f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?{urlencode(params)}"


def run_fetch(
    gender: str,
    api_key: str,
    regions: str,
    bookmakers: str,
    season: int,
    raw_dir: Path,
    output_dir: Path,
    odds_format: str,
    date_format: str,
    markets: str,
    keep_by_book: bool,
    fuzzy_threshold: float,
    unmatched_dir: Path,
) -> None:
    sport_key = SPORT_KEYS[gender]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{gender}_{timestamp}.json"
    output_path = output_dir / f"{gender}MatchupOdds_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}MatchupOdds_{season}_unmatched.csv"

    url = build_url(sport_key, api_key, regions, bookmakers, odds_format, date_format, markets)
    payload, headers = fetch_json(url)
    write_json(raw_path, payload)

    flat = flatten_events(payload, season, markets)
    prepared, audit_df = prepare_kaggle_odds(flat, gender, keep_by_book, fuzzy_threshold=fuzzy_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)

    print(f"[{gender}] sport={sport_key}")
    print(f"[{gender}] raw_events={len(payload)} flat_rows={len(flat)} cleaned_rows={len(prepared)}")
    print(f"[{gender}] saved raw -> {raw_path}")
    print(f"[{gender}] saved csv -> {output_path}")
    if headers:
        remaining = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        last_cost = headers.get("x-requests-last")
        print(f"[{gender}] quota remaining={remaining} used={used} last_cost={last_cost}")
    summarize(prepared.rename(columns={"T1": "Team1ID", "T2": "Team2ID"}))
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
            api_key=api_key,
            regions=args.regions,
            bookmakers=args.bookmakers,
            season=args.season,
            raw_dir=raw_dir,
            output_dir=output_dir,
            odds_format=args.odds_format,
            date_format=args.date_format,
            markets=args.markets,
            keep_by_book=args.keep_by_book,
            fuzzy_threshold=float(args.fuzzy_threshold),
            unmatched_dir=unmatched_dir,
        )


if __name__ == "__main__":
    main()
