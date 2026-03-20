from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import build_team_name_lookup, resolve_team_id


HEADERS = {"User-Agent": "Mozilla/5.0"}
EASTERN = ZoneInfo("America/New_York")
SITEMAP_TEMPLATE = "https://www.actionnetwork.com/sitemap-pt-post-{year}-{month:02d}.xml"
OUTPUT_TEMPLATE = "WHistoricalOdds_ActionNetwork_{season}.csv"
KEYWORDS = ("women", "womens", "ncaaw")
DEFAULT_MONTHS = (3, 4)
DATA_DIR = ROOT / "ncaa-data"
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape women NCAA tournament odds from Action Network article pages."
    )
    parser.add_argument("--season", type=int, action="append", required=True, help="Season year to scrape, e.g. 2023.")
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_DIR)
    return parser.parse_args()


def tourney_pairs_for_season(season: int) -> set[tuple[int, int]]:
    path = DATA_DIR / "WNCAATourneyCompactResults.csv"
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    df = df[df["Season"] == season].copy()
    if df.empty:
        return set()
    left = df[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    right = df[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    return set(zip(left, right))


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def candidate_article_urls(season: int) -> list[str]:
    urls: list[str] = []
    for month in DEFAULT_MONTHS:
        xml = fetch_text(SITEMAP_TEMPLATE.format(year=season, month=month))
        soup = BeautifulSoup(xml, "xml")
        for entry in soup.find_all("url"):
            loc = entry.loc.text.strip() if entry.loc else ""
            title_tag = entry.find("news:title")
            title = title_tag.text.strip() if title_tag else ""
            combo = f"{loc} {title}".lower()
            if any(keyword in combo for keyword in KEYWORDS):
                urls.append(loc)
    return list(dict.fromkeys(urls))


def clean_team_name(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"\s*\(W\)\s*$", "", text).strip()
    return text


def normalize_time_text(value: str) -> str:
    text = str(value).strip().upper()
    text = text.replace("A.M.", "AM").replace("P.M.", "PM")
    text = text.replace("A.M", "AM").replace("P.M", "PM")
    text = text.replace(".P.M", "PM").replace(".A.M", "AM")
    text = text.replace("ET", "").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("AM", " AM").replace("PM", " PM")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_commence_time(date_text: str, time_text: str, season: int) -> str:
    if not str(date_text).strip() or not str(time_text).strip():
        return ""
    normalized_time = normalize_time_text(time_text)
    combined = f"{date_text}, {season} {normalized_time}"
    parsed = pd.to_datetime(combined, errors="coerce")
    if pd.isna(parsed):
        return ""
    local = parsed.to_pydatetime().replace(tzinfo=EASTERN)
    return local.isoformat()


def parse_next_data(url: str) -> dict[str, object]:
    try:
        text = fetch_text(url)
    except requests.RequestException:
        return {}
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', text)
    if not match:
        return {}
    payload = json.loads(match.group(1))
    return payload.get("props", {}).get("pageProps", {})


def parse_shortcode_matchups(article_html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in re.finditer(r"\[gamematchup\s+([^\]]+)\]\[/gamematchup\]", article_html):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        away = attrs.get("awayname", "").strip()
        home = attrs.get("homename", "").strip()
        if not away or not home:
            continue
        rows.append(
            {
                "team1": away,
                "team2": home,
                "team1_spread": attrs.get("col1awaytext", "").strip(),
                "team2_spread": attrs.get("col1hometext", "").strip(),
                "team1_moneyline": attrs.get("col3awaytext", "").strip(),
                "team2_moneyline": attrs.get("col3hometext", "").strip(),
                "date": attrs.get("date", "").strip(),
                "time": attrs.get("time", "").strip(),
                "book": attrs.get("bookname", "").strip(),
            }
        )
    return rows


def parse_legacy_game_headers(article_html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    soup = BeautifulSoup(article_html, "html.parser")
    for header in soup.select("div.game-header"):
        tables = header.select("table.game-header__middle-row-table")
        if len(tables) != 2:
            continue
        center = header.select_one("div.game-header__top-row-center")
        center_parts = [item.get_text(" ", strip=True) for item in center.find_all("div")] if center else []
        team_rows = []
        for table in tables:
            team_header = table.select_one("th")
            team = team_header.get_text(" ", strip=True).replace("Odds", "").strip() if team_header else ""
            tr = table.select("tr")
            cells = tr[2].select("td") if len(tr) >= 3 else []
            spread = cells[0].find("div").get_text(" ", strip=True) if len(cells) >= 1 and cells[0].find("div") else ""
            moneyline = cells[2].get_text(" ", strip=True) if len(cells) >= 3 else ""
            team_rows.append((team, spread, moneyline))
        if len(team_rows) != 2:
            continue
        rows.append(
            {
                "team1": team_rows[0][0],
                "team2": team_rows[1][0],
                "team1_spread": team_rows[0][1],
                "team2_spread": team_rows[1][1],
                "team1_moneyline": team_rows[0][2],
                "team2_moneyline": team_rows[1][2],
                "date": center_parts[0] if len(center_parts) >= 1 else "",
                "time": center_parts[1] if len(center_parts) >= 2 else "",
                "book": "",
            }
        )
    return rows


def american_to_decimal(value: str) -> float | None:
    text = str(value).strip().replace("−", "-").upper()
    if not text or text in {"OFF", "N/A", "NA", "PK", "PICK"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric < 0:
        return round(1.0 + 100.0 / abs(numeric), 4)
    return round(1.0 + numeric / 100.0, 4)


def parse_spread(value: str) -> float | None:
    text = str(value).strip().replace("−", "-").upper()
    if not text or text in {"PK", "PICK", "OFF", "N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_rows_for_training(
    raw_rows: list[dict[str, object]],
    season: int,
    valid_pairs: set[tuple[int, int]],
    lookup: dict[str, int],
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for row in raw_rows:
        team1 = clean_team_name(str(row["team1"]))
        team2 = clean_team_name(str(row["team2"]))
        id1 = resolve_team_id(team1, lookup)
        id2 = resolve_team_id(team2, lookup)
        if id1 is None or id2 is None or id1 == id2:
            continue
        decimal1 = american_to_decimal(str(row["team1_moneyline"]))
        decimal2 = american_to_decimal(str(row["team2_moneyline"]))
        if decimal1 is None or decimal2 is None:
            continue
        implied1 = 1.0 / decimal1
        implied2 = 1.0 / decimal2
        denom = implied1 + implied2
        market_prob = implied1 / denom if denom > 0 else None

        t1_id, t2_id = int(id1), int(id2)
        t1_name, t2_name = team1, team2
        t1_decimal, t2_decimal = decimal1, decimal2
        t1_moneyline = str(row["team1_moneyline"]).strip()
        t2_moneyline = str(row["team2_moneyline"]).strip()
        spread = parse_spread(str(row["team1_spread"]))
        if t1_id > t2_id:
            t1_id, t2_id = t2_id, t1_id
            t1_name, t2_name = t2_name, t1_name
            t1_decimal, t2_decimal = t2_decimal, t1_decimal
            t1_moneyline, t2_moneyline = t2_moneyline, t1_moneyline
            spread = parse_spread(str(row["team2_spread"]))
        if (t1_id, t2_id) not in valid_pairs:
            continue

        commence_time = parse_commence_time(str(row["date"]), str(row["time"]), season)
        snapshot_time = str(row["published_at"]).strip() if str(row["published_at"]).strip() else ""
        verified_pre_tourney = 0
        if snapshot_time and commence_time:
            published_at = pd.to_datetime(snapshot_time, errors="coerce", utc=True)
            commence_at = pd.to_datetime(commence_time, errors="coerce", utc=True)
            if pd.notna(published_at) and pd.notna(commence_at) and published_at <= commence_at:
                verified_pre_tourney = 1

        out_rows.append(
            {
                "Season": season,
                "T1": t1_id,
                "T2": t2_id,
                "Team1Name": t1_name,
                "Team2Name": t2_name,
                "Team1Moneyline": t1_moneyline,
                "Team2Moneyline": t2_moneyline,
                "Team1DecimalOdds": t1_decimal,
                "Team2DecimalOdds": t2_decimal,
                "MarketProb": market_prob,
                "LastSpread": spread,
                "AbsLastSpread": abs(spread) if spread is not None else None,
                "CommenceTime": commence_time,
                "SourceURL": str(row["source_url"]),
                "SnapshotTime": snapshot_time,
                "Source": "actionnetwork_article",
                "Book": str(row["book"]).strip(),
                "BookCount": 1,
                "VerifiedPreTourney": verified_pre_tourney,
                "ArticleTitle": str(row["article_title"]).strip(),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows)
    out = out.drop_duplicates(subset=["Season", "T1", "T2", "SourceURL"], keep="last")
    return out.sort_values(["Season", "SnapshotTime", "T1", "T2"]).reset_index(drop=True)


def scrape_season(season: int) -> tuple[pd.DataFrame, dict[str, object]]:
    valid_pairs = tourney_pairs_for_season(season)
    lookup = build_team_name_lookup("W")
    raw_rows: list[dict[str, object]] = []
    candidate_urls = candidate_article_urls(season)
    for url in candidate_urls:
        page_props = parse_next_data(url)
        article = page_props.get("article", {})
        if not article:
            continue
        html = str(article.get("html", ""))
        published_at = str(article.get("published_at", ""))
        article_title = str(article.get("title", ""))
        for row in parse_shortcode_matchups(html) + parse_legacy_game_headers(html):
            row["published_at"] = published_at
            row["article_title"] = article_title
            row["source_url"] = url
            raw_rows.append(row)
    normalized = normalize_rows_for_training(raw_rows, season, valid_pairs, lookup)
    coverage = float(normalized[["T1", "T2"]].drop_duplicates().shape[0] / len(valid_pairs)) if valid_pairs else 0.0
    spread_coverage = (
        float(normalized["LastSpread"].notna().mean()) if not normalized.empty else 0.0
    )
    summary = {
        "season": season,
        "candidate_urls": len(candidate_urls),
        "raw_rows": len(raw_rows),
        "normalized_rows": int(len(normalized)),
        "unique_pairs": int(normalized[["T1", "T2"]].drop_duplicates().shape[0]) if not normalized.empty else 0,
        "tournament_pairs": int(len(valid_pairs)),
        "coverage": coverage,
        "spread_coverage": spread_coverage,
    }
    return normalized, summary


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for season in sorted(set(args.season)):
        frame, summary = scrape_season(season)
        output_path = args.output_dir / OUTPUT_TEMPLATE.format(season=season)
        frame.to_csv(output_path, index=False)
        summary["output"] = str(output_path)
        summaries.append(summary)
        print(
            f"[{season}] wrote {len(frame)} rows, coverage={summary['coverage']:.3f}, spread_coverage={summary['spread_coverage']:.3f} -> {output_path}"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = RESULTS_DIR / f"actionnetwork_women_historical_scrape_{run_id}.json"
    summary_path.write_text(json.dumps({"runs": summaries}, indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
