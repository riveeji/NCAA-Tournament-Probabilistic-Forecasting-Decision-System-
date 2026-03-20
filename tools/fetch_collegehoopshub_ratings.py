from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
import urllib3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import attach_team_ids_from_names


BASE = "https://cbb-backend-production.up.railway.app"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EXTERNAL_DIR = ROOT / "external-data"
SEASON = 2026
SLEEP_SECONDS = 0.03
MAX_RETRIES = 4
MAX_WORKERS = 12

urllib3.disable_warnings()


def get_json(path: str):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=30, verify=False)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.4 * attempt)
    raise last_exc


def fetch_team_list() -> pd.DataFrame:
    teams = get_json("/api/teams/search")
    rows = []
    for team in teams:
        metrics = team.get("metrics") or {}
        rows.append(
            {
                "Season": SEASON,
                "TeamName": team.get("school") or team.get("name", ""),
                "CHH_DisplayName": team.get("name", ""),
                "CHH_Rank": team.get("rank"),
                "CHH_KPRank": team.get("kpRank"),
                "CHH_NETRank": team.get("netRank"),
                "CHH_Wins": team.get("wins"),
                "CHH_Losses": team.get("losses"),
                "CHH_OffEff": metrics.get("offEff"),
                "CHH_DefEff": metrics.get("defEff"),
                "CHH_Tempo": metrics.get("tempo"),
                "CHH_SOS": metrics.get("sos"),
                "CHH_OffRank": metrics.get("offRank"),
                "CHH_DefRank": metrics.get("defRank"),
                "CHH_TempoRank": metrics.get("tempoRank"),
                "CHH_Conf": team.get("conference"),
                "CHH_ExternalID": team.get("id"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["CHH_WinRate"] = frame["CHH_Wins"] / (frame["CHH_Wins"] + frame["CHH_Losses"]).replace(0, np.nan)
    return frame


def summarize_schedule(games: list[dict]) -> dict[str, float]:
    if not games:
        return {}
    df = pd.DataFrame(games)
    final = df[df["status"] == "final"].copy()
    if final.empty:
        return {}

    final["won"] = final["won"].astype(int)
    final["quadrant"] = pd.to_numeric(final["quadrant"], errors="coerce")
    final["margin"] = pd.to_numeric(final["team_score"], errors="coerce") - pd.to_numeric(final["opponent_score"], errors="coerce")
    out = {
        "CHH_Q1Games": float((final["quadrant"] == 1).sum()),
        "CHH_Q1Wins": float(((final["quadrant"] == 1) & (final["won"] == 1)).sum()),
        "CHH_Q2Games": float((final["quadrant"] == 2).sum()),
        "CHH_Q2Wins": float(((final["quadrant"] == 2) & (final["won"] == 1)).sum()),
        "CHH_CloseGameRate": float((final["margin"].abs() <= 5).mean()),
        "CHH_CloseGameWinRate": float(final.loc[final["margin"].abs() <= 5, "won"].mean()) if (final["margin"].abs() <= 5).any() else np.nan,
        "CHH_NeutralGameRate": float(final["is_neutral"].mean()),
    }
    return out


def summarize_resume(payload: dict) -> dict[str, float]:
    best = pd.DataFrame(payload.get("best_wins", []))
    worst = pd.DataFrame(payload.get("worst_losses", []))
    out: dict[str, float] = {}
    if not best.empty:
        out["CHH_BestWinQualityMean"] = float(pd.to_numeric(best["quality_score"], errors="coerce").mean())
        out["CHH_BestWinQualityMax"] = float(pd.to_numeric(best["quality_score"], errors="coerce").max())
        out["CHH_BestWinQ1Count"] = float((pd.to_numeric(best["quadrant"], errors="coerce") == 1).sum())
    if not worst.empty:
        worst_quad = pd.to_numeric(worst["quadrant"], errors="coerce")
        out["CHH_WorstLossQuadrantMean"] = float(worst_quad.mean())
        out["CHH_BadLossCountQ3Q4"] = float(((worst_quad >= 3) & worst_quad.notna()).sum())
    return out


def enrich_team(frame: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    external_id = row["CHH_ExternalID"]
    out = {"Season": SEASON, "TeamName": row["TeamName"]}
    try:
        schedule = get_json(f"/api/teams/{external_id}/schedule")
        time.sleep(SLEEP_SECONDS)
        out.update(summarize_schedule(schedule.get("games", [])))
    except Exception as exc:
        out["CHH_FetchScheduleError"] = 1.0
        print(f"Schedule fetch failed for {row['TeamName']}: {exc}")

    try:
        resume = get_json(f"/api/teams/{external_id}/resume")
        time.sleep(SLEEP_SECONDS)
        out.update(summarize_resume(resume))
    except Exception as exc:
        out["CHH_FetchResumeError"] = 1.0
        print(f"Resume fetch failed for {row['TeamName']}: {exc}")
    return out


def main() -> None:
    base = fetch_team_list()
    extras = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(enrich_team, base, row) for _, row in base.iterrows()]
        for future in as_completed(futures):
            extras.append(future.result())
    extra_df = pd.DataFrame(extras)

    merged = base.merge(extra_df, on=["Season", "TeamName"], how="left")
    merged = attach_team_ids_from_names(merged, "M", team_col="TeamName", target_col="TeamID")
    merged = merged.dropna(subset=["TeamID"]).copy()
    merged["TeamID"] = merged["TeamID"].astype(int)
    merged = merged.drop(columns=["CHH_ExternalID"], errors="ignore")
    merged = merged.sort_values(["Season", "TeamID"]).reset_index(drop=True)

    EXTERNAL_DIR.mkdir(exist_ok=True)
    path = EXTERNAL_DIR / "MCollegeHoopsHubRatings.csv"
    merged.to_csv(path, index=False)
    print(f"Saved {path} rows={len(merged)}")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
