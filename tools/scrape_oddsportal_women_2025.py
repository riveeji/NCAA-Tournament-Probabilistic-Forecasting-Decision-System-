from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import build_team_name_lookup, resolve_team_id


BASE_URL_TEMPLATE = "https://www.oddsportal2.com/basketball/usa/ncaa-women-{start}-{end}/results/"
DATA_DIR = ROOT / "ncaa-data"


def tourney_pairs_for_season(season: int) -> set[tuple[int, int]]:
    path = DATA_DIR / "WNCAATourneyCompactResults.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    df = df[df["Season"] == season].copy()
    if df.empty:
        return set()
    left = df[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    right = df[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    return set(zip(left, right))


def parse_rows(page) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    event_rows = page.locator("div.eventRow")
    for idx in range(event_rows.count()):
        row = event_rows.nth(idx)
        try:
            date_locator = row.locator('[data-testid="date-header"]')
            titles = row.locator('[data-testid="event-participants"] a[title]')
            if date_locator.count() == 0 or titles.count() < 2:
                continue
            odds = row.locator('p[data-testid^="odd-container"]').all_inner_texts()
            if len(odds) < 2:
                continue
            href_locator = row.locator('a.next-m\\:flex')
            href = href_locator.first.get_attribute("href") if href_locator.count() else ""
            time_locator = row.locator('[data-testid="time-item"] p')
            time_text = time_locator.first.inner_text().strip() if time_locator.count() else ""
            date_texts = date_locator.first.all_inner_texts()
            date_header = date_texts[0].strip() if date_texts else ""
            rows.append(
                {
                    "dateHeader": date_header,
                    "href": href or "",
                    "team1": titles.nth(0).get_attribute("title") or "",
                    "team2": titles.nth(1).get_attribute("title") or "",
                    "odd1": odds[0].strip(),
                    "odd2": odds[1].strip(),
                    "time": time_text,
                }
            )
        except Exception:
            continue
    return rows


def parse_commence_time(date_header: str, time_str: str) -> str:
    date_part = date_header.split(" - ")[0].strip()
    if not date_part or not time_str:
        return ""
    try:
        parsed = datetime.strptime(f"{date_part} {time_str}", "%d %b %Y %H:%M")
    except ValueError:
        return ""
    return parsed.isoformat()


def default_target_url(season: int) -> str:
    start = int(season) - 1
    end = int(season)
    return BASE_URL_TEMPLATE.format(start=start, end=end)


def default_output_path(season: int) -> Path:
    return ROOT / "external-data" / f"WHistoricalOdds_OddsPortal_{season}.csv"


def scrape_results(target_url: str, max_pages: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 2400})
        page.goto(target_url, wait_until="networkidle", timeout=120000)
        pagination = page.locator('.pagination-link[data-number]')
        total_pages = int(pagination.last.get_attribute("data-number") or "1") if pagination.count() else 1
        if max_pages:
            total_pages = min(total_pages, max_pages)
        for page_no in range(1, total_pages + 1):
            if page_no > 1:
                prev_active = page.locator(".pagination-link.active").inner_text().strip()
                next_link = page.locator(".pagination-link").filter(has_text="Next").last
                next_link.click()
                try:
                    page.wait_for_function(
                        "(prev) => { const el = document.querySelector('.pagination-link.active'); return el && el.textContent.trim() !== prev; }",
                        arg=prev_active,
                        timeout=30000,
                    )
                except PlaywrightTimeoutError:
                    page.wait_for_timeout(1500)
                try:
                    page.wait_for_load_state("networkidle", timeout=30000)
                except PlaywrightTimeoutError:
                    page.wait_for_timeout(1000)
            current_rows = parse_rows(page)
            for row in current_rows:
                row["page"] = page_no
            rows.extend(current_rows)
        browser.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["dateHeader", "href", "team1", "team2", "odd1", "odd2"], keep="first")
    return frame


def normalize_for_training(frame: pd.DataFrame, season: int = 2025) -> pd.DataFrame:
    lookup = build_team_name_lookup("W")
    valid_pairs = tourney_pairs_for_season(season)
    out_rows: list[dict[str, object]] = []

    def clean_name(name: str) -> str:
        text = name.strip()
        for suffix in [" W", " Women"]:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
        return text

    for row in frame.itertuples(index=False):
        team1 = clean_name(str(row.team1))
        team2 = clean_name(str(row.team2))
        if not team1 or not team2:
            continue
        id1 = resolve_team_id(team1, lookup)
        id2 = resolve_team_id(team2, lookup)
        if id1 is None or id2 is None or id1 == id2:
            continue
        odd1 = pd.to_numeric(pd.Series([row.odd1]), errors="coerce").iloc[0]
        odd2 = pd.to_numeric(pd.Series([row.odd2]), errors="coerce").iloc[0]
        if pd.isna(odd1) or pd.isna(odd2):
            continue
        left_id, right_id = (int(id1), int(id2))
        left_name, right_name = team1, team2
        left_odd, right_odd = float(odd1), float(odd2)
        if left_id > right_id:
            left_id, right_id = right_id, left_id
            left_name, right_name = right_name, left_name
            left_odd, right_odd = right_odd, left_odd
        if (left_id, right_id) not in valid_pairs:
            continue
        href = str(row.href).strip()
        source_url = href if href.startswith("http") else f"https://www.oddsportal2.com{href}"
        out_rows.append(
            {
                "Season": season,
                "T1": left_id,
                "T2": right_id,
                "Team1Name": left_name,
                "Team2Name": right_name,
                "Team1DecimalOdds": left_odd,
                "Team2DecimalOdds": right_odd,
                "CommenceTime": parse_commence_time(str(row.dateHeader), str(row.time)),
                "SourceURL": source_url,
                "SnapshotTime": parse_commence_time(str(row.dateHeader), str(row.time)),
                "Source": "oddsportal2_playwright",
                "Page": int(row.page),
                "DateHeader": str(row.dateHeader),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows).drop_duplicates(subset=["Season", "T1", "T2"], keep="last")
    return out.sort_values(["Season", "CommenceTime", "T1", "T2"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape NCAA women OddsPortal results and normalize tournament odds.")
    parser.add_argument("--season", type=int, default=2025, help="Tournament season, e.g. 2025 means 2024-2025 site page.")
    parser.add_argument("--target-url", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-empty-overwrite",
        action="store_true",
        help="Allow replacing an existing non-empty CSV with an empty scrape result.",
    )
    args = parser.parse_args()

    season = int(args.season)
    target_url = str(args.target_url or default_target_url(season))
    output = args.output or default_output_path(season)
    raw = scrape_results(target_url=target_url, max_pages=args.max_pages)
    normalized = normalize_for_training(raw, season=season)
    existing_nonempty = output.exists() and output.stat().st_size > 2
    if normalized.empty and existing_nonempty and not args.allow_empty_overwrite:
        raise SystemExit(
            f"Scrape returned 0 normalized rows for season {season}; refusing to overwrite existing non-empty file {output}. "
            "Pass --allow-empty-overwrite to override."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    print(f"Wrote {len(normalized)} tournament odds rows to {output}")


if __name__ == "__main__":
    main()
