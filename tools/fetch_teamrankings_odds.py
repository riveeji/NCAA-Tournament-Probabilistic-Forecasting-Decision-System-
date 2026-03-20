from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Iterable, Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_utils import DEFAULT_FUZZY_THRESHOLD, attach_team_ids, canonicalize_matchups, summarize, write_unmatched_log


BASE_URL = "https://www.teamrankings.com"
ODDS_URL = f"{BASE_URL}/ncb/odds/"
WIN_HISTORY_URL = f"{BASE_URL}/ncb/odds-history/win/"
EXTERNAL_DIR = ROOT / "external-data"
AUDIT_DIR = EXTERNAL_DIR / "audit-logs"
HEADERS = {"User-Agent": "codex-ncaa-teamrankings/1.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch TeamRankings current NCAA spread pages and convert spreads to implied win probabilities."
    )
    parser.add_argument("--season", type=int, default=2026, help="Season value to write into output.")
    parser.add_argument(
        "--history-seasons",
        default="2020,2021,2022,2023,2024",
        help="Comma-separated TeamRankings season-filter values used to build the spread-to-win curve.",
    )
    parser.add_argument("--output-dir", default=str(EXTERNAL_DIR), help="Directory for cleaned output CSV files.")
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Minimum fuzzy match score for team mapping.",
    )
    return parser.parse_args()


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_history_seasons(raw: str) -> list[int]:
    seasons = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seasons.append(int(part))
    return seasons


def build_spread_curve(history_filters: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season_filter in history_filters:
        table = pd.read_html(f"{WIN_HISTORY_URL}?season-filter={season_filter}")[0]
        flat = pd.DataFrame(
            {
                "ClosingSpread": pd.to_numeric(table.iloc[:, 0], errors="coerce"),
                "GameCount": pd.to_numeric(table.iloc[:, 1], errors="coerce"),
                "WinPct": (
                    pd.to_numeric(
                        table.iloc[:, 3].astype(str).str.replace("%", "", regex=False),
                        errors="coerce",
                    )
                    / 100.0
                ),
            }
        ).dropna()
        frames.append(flat)

    curve = pd.concat(frames, ignore_index=True)
    curve = (
        curve.groupby("ClosingSpread", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "GameCount": float(group["GameCount"].sum()),
                    "WinPct": float(np.average(group["WinPct"], weights=group["GameCount"].clip(lower=1))),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("ClosingSpread")
    )
    return curve


def spread_to_prob(spread: float, curve: pd.DataFrame) -> float:
    spreads = curve["ClosingSpread"].to_numpy(dtype=float)
    probs = curve["WinPct"].to_numpy(dtype=float)
    spread = float(np.clip(spread, spreads.min(), spreads.max()))
    return float(np.interp(spread, spreads, probs))


def clean_matchup_name(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+\(N\)$", "", text)
    text = re.sub(r"\s+\([A-Z]{1,3}\)$", "", text)
    return text.strip()


def to_spread_url(href: str) -> str:
    path = href.strip()
    path = re.sub(r"/(spread-movement|over-under-movement)$", "", path)
    return urljoin(BASE_URL, path.rstrip("/") + "/spread-movement")


def parse_signed_number(value: object) -> Optional[float]:
    text = str(value).strip()
    match = re.search(r"([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def extract_last_spread(spread_url: str) -> Optional[float]:
    try:
        tables = pd.read_html(spread_url)
    except Exception:
        return None
    if not tables:
        return None
    value = tables[0].iloc[0, 0]
    return parse_signed_number(value)


def collect_current_spreads(curve: pd.DataFrame, season: int) -> pd.DataFrame:
    html = fetch_html(ODDS_URL)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    seen_urls: set[str] = set()

    for anchor in soup.select('a[href*="/ncaa-basketball/matchup/"]'):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href") or ""
        if " vs. " not in text or not href:
            continue

        team1_raw, team2_raw = text.split(" vs. ", 1)
        team1_name = clean_matchup_name(team1_raw)
        team2_name = clean_matchup_name(team2_raw.split(" (", 1)[0])
        spread_url = to_spread_url(href)
        if spread_url in seen_urls:
            continue
        seen_urls.add(spread_url)

        spread = extract_last_spread(spread_url)
        if spread is None:
            continue

        rows.append(
            {
                "Season": season,
                "Team1Name": team1_name,
                "Team2Name": team2_name,
                "LastSpread": spread,
                "MarketProb": spread_to_prob(spread, curve),
                "SourceURL": spread_url,
            }
        )

    return pd.DataFrame(rows)


def prepare_output(df: pd.DataFrame, season: int, fuzzy_threshold: float) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if df.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "Team1Name", "Team2Name", "LastSpread", "MarketProb", "SourceURL"]), None

    frame = attach_team_ids(df, "M", "Team1Name", "Team2Name", fuzzy_threshold=fuzzy_threshold)
    audit_df = frame.attrs.get("team_match_audit")
    frame = canonicalize_matchups(frame)
    frame = frame.rename(columns={"Team1ID": "T1", "Team2ID": "T2"})
    frame = frame.dropna(subset=["T1", "T2", "MarketProb"]).copy()
    frame["Season"] = int(season)
    frame["T1"] = frame["T1"].astype(int)
    frame["T2"] = frame["T2"].astype(int)
    frame["MarketProb"] = pd.to_numeric(frame["MarketProb"], errors="coerce").clip(0.025, 0.975)
    frame["LastSpread"] = pd.to_numeric(frame["LastSpread"], errors="coerce")
    prepared = (
        frame.groupby(["Season", "T1", "T2"], as_index=False)
        .agg(
            Team1Name=("Team1Name", "first"),
            Team2Name=("Team2Name", "first"),
            LastSpread=("LastSpread", "mean"),
            MarketProb=("MarketProb", "mean"),
            SourceURL=("SourceURL", "first"),
        )
    )
    return prepared, audit_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    curve = build_spread_curve(parse_history_seasons(args.history_seasons))
    raw = collect_current_spreads(curve, args.season)
    prepared, audit_df = prepare_output(raw, args.season, args.fuzzy_threshold)

    output_path = output_dir / f"MTeamRankingsOdds_{args.season}.csv"
    unmatched_path = AUDIT_DIR / f"MTeamRankingsOdds_{args.season}_unmatched.csv"
    prepared.to_csv(output_path, index=False)
    write_unmatched_log(audit_df, unmatched_path)

    print("[M] source=teamrankings")
    print(f"[M] raw_rows={len(raw)} cleaned_rows={len(prepared)}")
    print(f"[M] saved csv -> {output_path}")
    summarize(prepared.rename(columns={"T1": "Team1ID", "T2": "Team2ID"}))
    if unmatched_path.exists():
        unresolved = pd.read_csv(unmatched_path)
        print(f"[M] unmatched_audit={unmatched_path} rows={len(unresolved)}")


if __name__ == "__main__":
    main()
