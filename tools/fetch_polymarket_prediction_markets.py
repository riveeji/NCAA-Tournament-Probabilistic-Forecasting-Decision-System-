from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert_prediction_market_probs import convert_prediction_market_frame
from tools.market_data_utils import write_unmatched_log
from zizzii_features import build_team_name_lookup, normalize_team_name, resolve_team_id


SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EXTERNAL_DIR = ROOT / "external-data"
RAW_DIR = EXTERNAL_DIR / "raw-polymarket"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"

SPECIAL_QUERY_ALIASES = {
    "uconn": ["Connecticut", "Connecticut Huskies"],
    "connecticut": ["Connecticut", "Connecticut Huskies"],
    "ohio st": ["Ohio State", "Ohio State Buckeyes"],
    "ohio state": ["Ohio State", "Ohio State Buckeyes"],
    "cal baptist": ["California Baptist", "California Baptist Lancers"],
    "cal baptist lancers": ["California Baptist", "California Baptist Lancers"],
    "uc san diego": ["California San Diego", "California-San Diego Tritons"],
    "uc san diego tritons": ["California San Diego", "California-San Diego Tritons"],
    "ole miss": ["Ole Miss", "Ole Miss Rebels"],
    "ole miss rebels": ["Ole Miss", "Ole Miss Rebels"],
    "fdu w": ["Fairleigh Dickinson", "Fairleigh Dickinson Knights"],
    "fdu": ["Fairleigh Dickinson", "Fairleigh Dickinson Knights"],
    "lsu": ["LSU", "LSU Tigers"],
    "miami oh": ["Miami Ohio", "Miami Ohio Redhawks"],
    "miami oh redhawks": ["Miami Ohio", "Miami Ohio Redhawks"],
    "saint mary s ca": ["Saint Mary's", "Saint Mary's Gaels"],
    "st mary s ca": ["Saint Mary's", "Saint Mary's Gaels"],
    "byu": ["BYU", "Brigham Young"],
}

MATCHUP_PATTERNS = [
    "{gender}MatchupOdds_{season}.csv",
    "{gender}ActionNetworkOdds_{season}.csv",
    "{gender}ManualOdds_{season}.csv",
    "{gender}PredictionMarketOdds_{season}.csv",
    "{gender}KalshiPredictionMarketOdds_{season}.csv",
    "{gender}SilverBulletinMatchupProjections_{season}.csv",
    "{gender}SilverBulletinMatchupProjections_*.csv",
    "{gender}BartTorvikMatchupProjections_{season}.csv",
    "{gender}WarrenNolanMatchupProjections_{season}.csv",
    "{gender}HerHoopStatsMatchupProjections_{season}.csv",
]

SESSION = requests.Session()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket NCAA game-winner markets into normalized HC market-signal CSVs."
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


def _load_tournament_field_ids(gender: str, season: int) -> set[int]:
    path = ROOT / "ncaa-data" / f"{gender}NCAATourneySeeds.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["Season", "TeamID"])
    frame = frame.loc[pd.to_numeric(frame["Season"], errors="coerce") == int(season)]
    return {int(team_id) for team_id in pd.to_numeric(frame["TeamID"], errors="coerce").dropna().astype(int)}


def _load_matchup_candidates(gender: str, season: int) -> list[dict[str, object]]:
    field_ids = _load_tournament_field_ids(gender, season)
    name_lookup = build_team_name_lookup(gender)
    candidate_map: dict[tuple[int, int], tuple[str, str, int, int, int]] = {}
    for pattern in MATCHUP_PATTERNS:
        formatted = pattern.format(gender=gender, season=season)
        for path in sorted(EXTERNAL_DIR.glob(formatted)):
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not {"Team1Name", "Team2Name"}.issubset(frame.columns):
                continue
            for row in frame.itertuples(index=False):
                team1 = str(getattr(row, "Team1Name", "")).strip()
                team2 = str(getattr(row, "Team2Name", "")).strip()
                if not team1 or not team2:
                    continue
                id1 = pd.to_numeric(pd.Series([getattr(row, "T1", getattr(row, "Team1ID", pd.NA))]), errors="coerce").iloc[0]
                id2 = pd.to_numeric(pd.Series([getattr(row, "T2", getattr(row, "Team2ID", pd.NA))]), errors="coerce").iloc[0]
                if pd.isna(id1):
                    id1 = resolve_team_id(team1, name_lookup)
                if pd.isna(id2):
                    id2 = resolve_team_id(team2, name_lookup)
                if pd.isna(id1) or pd.isna(id2):
                    continue
                id1 = int(id1)
                id2 = int(id2)
                if field_ids and (id1 not in field_ids or id2 not in field_ids):
                    continue
                key = tuple(sorted((id1, id2)))
                score = len(team1) + len(team2)
                best = candidate_map.get(key)
                if best is None or score > best[2]:
                    candidate_map[key] = (team1, team2, score, id1, id2)
    ordered = sorted(
        (
            {
                "Team1Name": value[0],
                "Team2Name": value[1],
                "T1": int(value[3]),
                "T2": int(value[4]),
            }
            for value in candidate_map.values()
        ),
        key=lambda item: (item["Team1Name"], item["Team2Name"]),
    )
    return ordered


def _query_aliases(name: str) -> list[str]:
    raw = " ".join(str(name or "").split()).strip()
    if not raw:
        return []
    normalized = normalize_team_name(raw)
    labels = [raw]
    for key, aliases in SPECIAL_QUERY_ALIASES.items():
        if normalized == key or normalized.startswith(key + " "):
            labels.extend(aliases)
    if raw not in labels:
        labels.append(raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean = " ".join(str(label).split()).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            deduped.append(clean)
    return deduped


def _build_queries(team1: str, team2: str) -> list[str]:
    labels1 = _query_aliases(team1)
    labels2 = _query_aliases(team2)
    queries: list[str] = []
    seen: set[str] = set()
    for left in labels1[:2]:
        for right in labels2[:2]:
            for a, b in ((left, right), (right, left)):
                query = f"{a} {b}".strip()
                if query.lower() not in seen:
                    seen.add(query.lower())
                    queries.append(query)
    return queries


def _parse_event_date(event: dict[str, object]) -> date | None:
    for key in ["endDate", "startDate", "createdAt"]:
        value = str(event.get(key) or "").strip()
        if not value:
            continue
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.date()
    return None


def _parse_title_sides(title: str) -> tuple[str, str] | None:
    clean = " ".join(str(title or "").split()).strip()
    clean = clean.replace(" (W)", "").replace(" (M)", "")
    for token in (" vs. ", " vs "):
        if token in clean:
            left, right = clean.split(token, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return None


def _name_variants(name: str) -> set[str]:
    variants = {normalize_team_name(name)}
    for alias in _query_aliases(name):
        variants.add(normalize_team_name(alias))
    return {token for token in variants if token}


def _match_name(left: str, right: str) -> bool:
    left_variants = _name_variants(left)
    right_variants = _name_variants(right)
    for left_value in left_variants:
        for right_value in right_variants:
            if left_value == right_value:
                return True
            if left_value in right_value or right_value in left_value:
                return True
    return False


def _parse_list_field(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _extract_market(event: dict[str, object]) -> dict[str, object] | None:
    for market in event.get("markets") or []:
        outcomes = _parse_list_field(market.get("outcomes"))
        prices = _parse_list_field(market.get("outcomePrices"))
        if len(outcomes) == 2 and len(prices) == 2:
            return market
    return None


def _event_score(event: dict[str, object], team1: str, team2: str, gender: str) -> int:
    title = str(event.get("title") or "")
    score = 0
    if gender == "W" and "(W)" in title:
        score += 4
    if gender == "M" and "(W)" not in title:
        score += 2
    if bool(event.get("active")) and not bool(event.get("closed")) and not bool(event.get("archived")):
        score += 2

    sides = _parse_title_sides(title)
    if sides is not None:
        left, right = sides
        if (_match_name(left, team1) and _match_name(right, team2)) or (_match_name(left, team2) and _match_name(right, team1)):
            score += 20

    market = _extract_market(event)
    if market is not None:
        outcomes = _parse_list_field(market.get("outcomes"))
        if len(outcomes) == 2:
            if (_match_name(outcomes[0], team1) and _match_name(outcomes[1], team2)) or (
                _match_name(outcomes[0], team2) and _match_name(outcomes[1], team1)
            ):
                score += 20
    return score


def _search_events(query: str) -> list[dict[str, object]]:
    response = SESSION.get(SEARCH_URL, params={"q": query}, headers=HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("events") or [])


def _select_event(
    team1: str,
    team2: str,
    gender: str,
    date_from: date,
    date_to: date,
) -> tuple[dict[str, object] | None, str]:
    best_event: dict[str, object] | None = None
    best_query = ""
    best_score = -1
    for query in _build_queries(team1, team2):
        try:
            events = _search_events(query)
        except Exception:
            continue
        for event in events:
            event_date = _parse_event_date(event)
            if event_date is None or event_date < date_from or event_date > date_to:
                continue
            score = _event_score(event, team1, team2, gender)
            if score > best_score:
                best_event = event
                best_query = query
                best_score = score
        if best_score >= 40:
            break
        time.sleep(0.15)
    if best_score < 20:
        return None, ""
    return best_event, best_query


def _build_raw_rows(
    gender: str,
    season: int,
    date_from: date,
    date_to: date,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    candidates = _load_matchup_candidates(gender, season)
    for candidate in candidates:
        team1 = str(candidate["Team1Name"])
        team2 = str(candidate["Team2Name"])
        event, query = _select_event(team1, team2, gender, date_from, date_to)
        if event is None:
            audit_rows.append({"Team1Name": team1, "Team2Name": team2, "query": query, "status": "unmatched"})
            continue
        market = _extract_market(event)
        if market is None:
            audit_rows.append({"Team1Name": team1, "Team2Name": team2, "query": query, "status": "missing_market"})
            continue
        outcomes = _parse_list_field(market.get("outcomes"))
        prices = [pd.to_numeric(item, errors="coerce") for item in _parse_list_field(market.get("outcomePrices"))]
        if len(outcomes) != 2 or len(prices) != 2 or not all(pd.notna(prices)):
            audit_rows.append({"Team1Name": team1, "Team2Name": team2, "query": query, "status": "missing_prices"})
            continue

        team1_prob = None
        team2_prob = None
        for outcome, price in zip(outcomes, prices):
            if _match_name(outcome, team1):
                team1_prob = float(price)
            elif _match_name(outcome, team2):
                team2_prob = float(price)
        if team1_prob is None and team2_prob is not None:
            team1_prob = 1.0 - team2_prob
        if team2_prob is None and team1_prob is not None:
            team2_prob = 1.0 - team1_prob
        if team1_prob is None or team2_prob is None:
            audit_rows.append({"Team1Name": team1, "Team2Name": team2, "query": query, "status": "side_match_failed"})
            continue

        denom = team1_prob + team2_prob
        if not pd.notna(denom) or denom <= 0:
            continue
        team1_prob = float(min(max(team1_prob / denom, 0.001), 0.999))
        team2_prob = float(min(max(team2_prob / denom, 0.001), 0.999))
        snapshot_time = str(market.get("updatedAt") or event.get("updatedAt") or "")
        raw_rows.append(
            {
                "Season": int(season),
                "T1": int(candidate["T1"]),
                "T2": int(candidate["T2"]),
                "Team1Name": team1,
                "Team2Name": team2,
                "Team1Prob": team1_prob,
                "Team2Prob": team2_prob,
                "Book": "polymarket",
                "BookCount": 1,
                "SnapshotTime": snapshot_time,
                "Source": "polymarket_prediction_market",
                "SourceURL": f"https://polymarket.com/event/{event.get('slug')}",
                "Notes": f"{event.get('slug','')}|query={query}",
            }
        )
        audit_rows.append({"Team1Name": team1, "Team2Name": team2, "query": query, "status": "matched", "slug": event.get("slug")})
    return pd.DataFrame(raw_rows), audit_rows


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
    output_path = output_dir / f"{gender}PolymarketPredictionMarketOdds_{season}.csv"
    unmatched_path = unmatched_dir / f"{gender}PolymarketPredictionMarketOdds_{season}_unmatched.csv"
    raw_frame, audit_rows = _build_raw_rows(gender, season, date_from, date_to)
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
            "rows": int(len(raw_frame)),
            "audit": audit_rows,
        },
    )

    print(f"[{gender}] source=polymarket_prediction_market")
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
