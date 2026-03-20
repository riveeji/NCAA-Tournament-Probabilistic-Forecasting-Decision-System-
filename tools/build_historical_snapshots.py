from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import HISTORICAL_TEAM_RATINGS_SAFE_COLUMNS, tourney_snapshot_cutoff_lookup

DATA_DIR = ROOT / "ncaa-data"
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"
TARGET_SEASONS = {2018, 2019, 2021, 2022, 2023, 2024, 2025}
PUBLIC_RATING_COLUMNS = frozenset(
    {
        "PublicNETRank",
        "PublicELORank",
        "PublicRPIRank",
        "PublicPredRPIRank",
        "PublicBPIRank",
        "PublicPOMRank",
        "PublicKPIRank",
        "PublicSORRank",
        "PublicELOValue",
        "PublicELORankPage",
        "PublicNETRankPage",
        "PublicAverageRank",
        "PublicTRankRank",
        "PublicWABRank",
        "PublicAvgPredRank",
        "PublicAPRank",
        "PublicCoachesRank",
    }
)


def tournament_team_ids(gender: str) -> dict[int, set[int]]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    team_map: dict[int, set[int]] = {}
    for row in df.itertuples(index=False):
        season = int(row.Season)
        team_map.setdefault(season, set()).update((int(row.WTeamID), int(row.LTeamID)))
    return team_map


def tournament_pair_count(gender: str) -> dict[int, int]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    df["T1"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["T2"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    grouped = df.groupby("Season").size()
    return {int(season): int(count) for season, count in grouped.items()}


def american_to_prob(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    prob = pd.Series(np.nan, index=numeric.index, dtype=float)
    negative = numeric < 0
    prob.loc[negative] = -numeric.loc[negative] / (-numeric.loc[negative] + 100.0)
    prob.loc[~negative] = 100.0 / (numeric.loc[~negative] + 100.0)
    return prob


def decimal_to_prob(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return 1.0 / numeric


def derive_market_prob(frame: pd.DataFrame) -> pd.Series:
    if "MarketProb" in frame.columns:
        return pd.to_numeric(frame["MarketProb"], errors="coerce")
    if {"Team1DecimalOdds", "Team2DecimalOdds"}.issubset(frame.columns):
        p1 = decimal_to_prob(frame["Team1DecimalOdds"])
        p2 = decimal_to_prob(frame["Team2DecimalOdds"])
        denom = (p1 + p2).replace(0, np.nan)
        return p1 / denom
    if {"Team1Moneyline", "Team2Moneyline"}.issubset(frame.columns):
        p1 = american_to_prob(frame["Team1Moneyline"])
        p2 = american_to_prob(frame["Team2Moneyline"])
        denom = (p1 + p2).replace(0, np.nan)
        return p1 / denom
    return pd.Series(np.nan, index=frame.index, dtype=float)


def derive_snapshot_time(frame: pd.DataFrame, cutoff_lookup: dict[int, pd.Timestamp]) -> pd.Series:
    for column in ["SnapshotTime", "CommenceTime", "GameTime", "EventTime", "Date"]:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if parsed.notna().any():
                return parsed
    out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for season, cutoff in cutoff_lookup.items():
        out.loc[pd.to_numeric(frame["Season"], errors="coerce").eq(season)] = cutoff
    return out


def derive_source(frame: pd.DataFrame, default_source: str) -> pd.Series:
    if "Source" in frame.columns:
        source = frame["Source"].fillna("").astype(str).str.strip()
        if source.ne("").any():
            return source.where(source.ne(""), default_source)
    if "SourceURL" in frame.columns:
        source = frame["SourceURL"].fillna("").astype(str).map(lambda value: urlparse(value).netloc or default_source)
        return source.where(source.ne(""), default_source)
    return pd.Series(default_source, index=frame.index, dtype=object)


def normalize_historical_ratings(gender: str) -> dict[str, object]:
    source_path = EXTERNAL_DIR / f"{gender}HistoricalTeamRatings.csv"
    target_path = EXTERNAL_DIR / f"{gender}HistoricalTeamRatingsPreTourney.csv"
    if not source_path.exists():
        return {"gender": gender, "ratings_written": False, "ratings_path": str(target_path)}

    raw = pd.read_csv(source_path)
    keep_cols = [
        column
        for column in [
            "Season",
            "TeamID",
            "TeamName",
            *sorted(HISTORICAL_TEAM_RATINGS_SAFE_COLUMNS),
            *sorted(PUBLIC_RATING_COLUMNS),
        ]
        if column in raw.columns
    ]
    frame = raw[keep_cols].copy()
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["Season", "TeamID"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)

    cutoff_lookup = tourney_snapshot_cutoff_lookup(gender, data_dir=DATA_DIR)
    frame["SnapshotDate"] = frame["Season"].map(cutoff_lookup)
    frame["Source"] = source_path.name
    frame["VerifiedPreTourney"] = frame["SnapshotDate"].notna().astype(int)
    frame = frame[frame["VerifiedPreTourney"] == 1].copy()
    frame["SnapshotDate"] = frame["SnapshotDate"].map(lambda value: value.isoformat())
    frame = frame.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")
    frame.to_csv(target_path, index=False)

    team_map = tournament_team_ids(gender)
    coverage = {}
    for season in sorted(TARGET_SEASONS):
        season_ids = team_map.get(season, set())
        covered = int(frame.loc[frame["Season"] == season, "TeamID"].isin(season_ids).sum())
        total = int(len(season_ids))
        coverage[season] = {
            "covered_teams": covered,
            "tournament_teams": total,
            "coverage": float(covered / total) if total else 0.0,
        }

    return {
        "gender": gender,
        "ratings_written": True,
        "ratings_path": str(target_path),
        "ratings_rows": int(len(frame)),
        "ratings_coverage_by_season": coverage,
    }


def normalize_historical_odds(gender: str) -> dict[str, object]:
    source_path = EXTERNAL_DIR / f"{gender}HistoricalTournamentOdds.csv"
    source_paths = sorted(EXTERNAL_DIR.glob(f"{gender}HistoricalOdds_*.csv"))
    if source_path.exists():
        source_paths.append(source_path)
    if not source_paths:
        return {"gender": gender, "odds_written": False, "odds_path": str(source_path)}

    frames = []
    for path in source_paths:
        try:
            current = pd.read_csv(path)
        except Exception:
            continue
        if current.empty:
            continue
        current["_SourceFile"] = path.name
        frames.append(current)
    if not frames:
        return {"gender": gender, "odds_written": False, "odds_path": str(source_path), "odds_rows": 0}

    frame = pd.concat(frames, ignore_index=True)
    required = {"Season", "T1", "T2"}
    if not required.issubset(frame.columns):
        return {"gender": gender, "odds_written": False, "odds_path": str(source_path), "missing_columns": sorted(required - set(frame.columns))}

    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["T1"] = pd.to_numeric(frame["T1"], errors="coerce")
    frame["T2"] = pd.to_numeric(frame["T2"], errors="coerce")
    frame = frame.dropna(subset=["Season", "T1", "T2"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["T1"] = frame["T1"].astype(int)
    frame["T2"] = frame["T2"].astype(int)

    cutoff_lookup = tourney_snapshot_cutoff_lookup(gender, data_dir=DATA_DIR)
    frame["MarketProb"] = derive_market_prob(frame)
    if "LastSpread" in frame.columns:
        frame["LastSpread"] = pd.to_numeric(frame["LastSpread"], errors="coerce")
    else:
        frame["LastSpread"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["SnapshotTime"] = derive_snapshot_time(frame, cutoff_lookup)
    frame["Source"] = derive_source(frame, source_path.name)
    if "_SourceFile" in frame.columns:
        frame["Source"] = frame["_SourceFile"].where(frame["Source"].eq(source_path.name), frame["Source"])
    frame["VerifiedPreTourney"] = 1
    frame = frame.sort_values(["Season", "T1", "T2", "SnapshotTime"]).drop_duplicates(["Season", "T1", "T2"], keep="last")
    frame["AbsLastSpread"] = frame["LastSpread"].abs()
    frame.to_csv(source_path, index=False)

    pair_counts = tournament_pair_count(gender)
    coverage = {}
    for season in sorted(TARGET_SEASONS):
        season_rows = frame[frame["Season"] == season]
        total = int(pair_counts.get(season, 0))
        coverage[season] = {
            "covered_pairs": int(len(season_rows)),
            "tournament_pairs": total,
            "coverage": float(len(season_rows) / total) if total else 0.0,
            "spread_coverage": float(season_rows["LastSpread"].notna().mean()) if not season_rows.empty else 0.0,
        }

    return {
        "gender": gender,
        "odds_written": True,
        "odds_path": str(source_path),
        "odds_rows": int(len(frame)),
        "odds_coverage_by_season": coverage,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "target_seasons": sorted(TARGET_SEASONS),
        "genders": {},
    }
    for gender in ["M", "W"]:
        rating_summary = normalize_historical_ratings(gender)
        odds_summary = normalize_historical_odds(gender)
        summary["genders"][gender] = {**rating_summary, **odds_summary}

    audit_path = RESULTS_DIR / f"historical_snapshot_audit_{summary['run_id']}.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Historical snapshot audit written to: {audit_path}")


if __name__ == "__main__":
    main()
