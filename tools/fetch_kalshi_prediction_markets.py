from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import time
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert_prediction_market_probs import convert_prediction_market_frame
from tools.market_data_utils import write_unmatched_log
from zizzii_features import normalize_team_name


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_BY_GENDER = {
    "M": "KXNCAAMBGAME",
    "W": "KXNCAAWBGAME",
}
COMPETITION_BY_GENDER = {
    "M": "CBB Tournament",
    "W": "CBB Tournament (W)",
}
HEADERS = {"User-Agent": "Mozilla/5.0"}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-kalshi"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
EVENT_DATE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})")
EVENT_PAGE_LIMIT = 100
MARKET_PAGE_LIMIT = 100
SESSION = requests.Session()
DIRECT_SESSION = requests.Session()
DIRECT_SESSION.trust_env = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi NCAA game-winner prediction markets into normalized HC market-signal CSVs."
    )
    parser.add_argument("--gender", choices=["M", "W"], help="Competition gender prefix.")
    parser.add_argument("--all", action="store_true", help="Fetch both men's and women's markets.")
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
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned output CSV files.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory for raw JSON archives.")
    parser.add_argument(
        "--unmatched-dir",
        default=str(AUDIT_DIR),
        help="Directory for unresolved team-name audit CSVs.",
    )
    return parser.parse_args()


def _request_json(path: str, params: dict[str, object]) -> dict[str, object]:
    last_error: Exception | None = None
    for session in (SESSION, DIRECT_SESSION):
        for attempt in range(3):
            try:
                response = session.get(f"{BASE_URL}{path}", params=params, headers=HEADERS, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network retry path
                last_error = exc
                if attempt == 2:
                    continue
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _fetch_paginated(path: str, params: dict[str, object], list_key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = ""
    while True:
        query = dict(params)
        if cursor:
            query["cursor"] = cursor
        payload = _request_json(path, query)
        rows.extend(payload.get(list_key, []) or [])
        cursor = str(payload.get("cursor") or "").strip()
        if not cursor:
            break
    return rows


def _parse_event_date(event: dict[str, object]) -> date | None:
    ticker = str(event.get("event_ticker") or "")
    match = EVENT_DATE_RE.search(ticker)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%y%b%d").date()
        except ValueError:
            pass
    text_candidates = [
        str(event.get("sub_title") or ""),
        str(event.get("title") or ""),
    ]
    for text in text_candidates:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    return None


def _parse_matchup_title(title: str) -> tuple[str, str] | None:
    clean = " ".join(str(title or "").split()).strip()
    if " at " not in clean:
        return None
    away, home = clean.split(" at ", 1)
    away = away.strip()
    home = home.strip()
    if not away or not home:
        return None
    return away, home


def _match_side(side_name: str, target_name: str) -> bool:
    side_norm = normalize_team_name(side_name)
    target_norm = normalize_team_name(target_name)
    if not side_norm or not target_norm:
        return False
    if side_norm == target_norm:
        return True
    return side_norm in target_norm or target_norm in side_norm


def _market_probability(market: dict[str, object]) -> float | None:
    yes_bid = pd.to_numeric(pd.Series([market.get("yes_bid_dollars")]), errors="coerce").iloc[0]
    yes_ask = pd.to_numeric(pd.Series([market.get("yes_ask_dollars")]), errors="coerce").iloc[0]
    last_price = pd.to_numeric(pd.Series([market.get("last_price_dollars")]), errors="coerce").iloc[0]
    candidates = []
    if pd.notna(yes_bid):
        candidates.append(float(yes_bid))
    if pd.notna(yes_ask):
        candidates.append(float(yes_ask))
    if len(candidates) == 2:
        prob = sum(candidates) / 2.0
    elif pd.notna(last_price):
        prob = float(last_price)
    elif candidates:
        prob = candidates[0]
    else:
        return None
    return float(min(max(prob, 0.001), 0.999))


def _build_raw_rows(
    gender: str,
    season: int,
    date_from: date,
    date_to: date,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    competition = COMPETITION_BY_GENDER[gender]
    series = SERIES_BY_GENDER[gender]
    events = _fetch_paginated("/events", {"limit": EVENT_PAGE_LIMIT, "series_ticker": series}, "events")
    kept_events: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    for event in events:
        metadata = event.get("product_metadata") or {}
        if metadata.get("competition") != competition:
            continue
        event_date = _parse_event_date(event)
        if event_date is None or event_date < date_from or event_date > date_to:
            continue
        matchup = _parse_matchup_title(str(event.get("title") or ""))
        if matchup is None:
            continue

        away_name, home_name = matchup
        event_ticker = str(event.get("event_ticker") or "").strip()
        markets = _fetch_paginated("/markets", {"limit": MARKET_PAGE_LIMIT, "event_ticker": event_ticker}, "markets")
        kept_events.append(
            {
                "event_ticker": event_ticker,
                "title": event.get("title"),
                "sub_title": event.get("sub_title"),
                "last_updated_ts": event.get("last_updated_ts"),
                "market_count": len(markets),
            }
        )

        away_prob = None
        home_prob = None
        snapshot_times: list[str] = []
        market_tickers: list[str] = []
        for market in markets:
            if str(market.get("market_type") or "") != "binary":
                continue
            if str(market.get("status") or "").lower() not in {"active", "initialized", "open"}:
                continue
            side_name = str(market.get("yes_sub_title") or "").strip()
            if not side_name:
                continue
            prob = _market_probability(market)
            if prob is None:
                continue
            market_tickers.append(str(market.get("ticker") or "").strip())
            updated_time = str(market.get("updated_time") or "").strip()
            if updated_time:
                snapshot_times.append(updated_time)
            if _match_side(side_name, away_name):
                away_prob = prob
            elif _match_side(side_name, home_name):
                home_prob = prob

        if away_prob is None and home_prob is None:
            continue
        if away_prob is None and home_prob is not None:
            away_prob = 1.0 - home_prob
        if home_prob is None and away_prob is not None:
            home_prob = 1.0 - away_prob

        assert away_prob is not None and home_prob is not None
        denom = away_prob + home_prob
        if not pd.notna(denom) or denom <= 0:
            continue
        away_prob = float(min(max(away_prob / denom, 0.001), 0.999))
        home_prob = float(min(max(home_prob / denom, 0.001), 0.999))
        snapshot_time = max(snapshot_times) if snapshot_times else str(event.get("last_updated_ts") or "")

        raw_rows.append(
            {
                "Season": int(season),
                "Team1Name": away_name,
                "Team2Name": home_name,
                "Team1Prob": away_prob,
                "Team2Prob": home_prob,
                "Book": "kalshi",
                "BookCount": 1,
                "SnapshotTime": snapshot_time,
                "Source": "kalshi_prediction_market",
                "SourceURL": f"{BASE_URL}/markets?event_ticker={event_ticker}",
                "Notes": "|".join(token for token in [event_ticker, ",".join(filter(None, market_tickers))] if token),
            }
        )

    return pd.DataFrame(raw_rows), kept_events


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch(
    gender: str,
    season: int,
    date_from: date,
    date_to: date,
    output_dir: Path,
    raw_dir: Path,
    unmatched_dir: Path,
) -> None:
    output_path = output_dir / f"{gender}KalshiPredictionMarketOdds_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}KalshiPredictionMarketOdds_{season}_unmatched.csv"
    raw_frame, kept_events = _build_raw_rows(gender, season, date_from, date_to)
    prepared = convert_prediction_market_frame(raw_frame, gender, season) if not raw_frame.empty else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(prepared.attrs.get("team_match_audit"), unmatched_path)
    _write_json(
        raw_dir / f"{gender}_{season}_{date_from.isoformat()}_{date_to.isoformat()}_index.json",
        {
            "gender": gender,
            "season": season,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "events": kept_events,
            "rows": int(len(raw_frame)),
        },
    )

    print(f"[{gender}] source=kalshi_prediction_market")
    print(f"[{gender}] raw_rows={len(raw_frame)} cleaned_rows={len(prepared)}")
    print(f"[{gender}] saved csv -> {output_path}")
    if unmatched_path.exists():
        try:
            unmatched_rows = len(pd.read_csv(unmatched_path))
        except Exception:
            unmatched_rows = 0
        print(f"[{gender}] unmatched_audit={unmatched_path} rows={unmatched_rows}")


def main() -> None:
    args = parse_args()
    if not args.all and not args.gender:
        raise SystemExit("Pass --gender M/W or use --all.")

    genders = ["M", "W"] if args.all else [args.gender]
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    unmatched_dir = Path(args.unmatched_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir.mkdir(parents=True, exist_ok=True)

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)

    for gender in genders:
        run_fetch(
            gender=gender,
            season=args.season,
            date_from=date_from,
            date_to=date_to,
            output_dir=output_dir,
            raw_dir=raw_dir,
            unmatched_dir=unmatched_dir,
        )


if __name__ == "__main__":
    main()
