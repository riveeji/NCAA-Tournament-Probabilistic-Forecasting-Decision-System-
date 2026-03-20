from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ModuleNotFoundError:
    requests = None

from hc.constants import EXTERNAL_DIR, RAW_TEXT_DIR
from zizzii_features import build_team_name_lookup, resolve_team_id


@dataclass(frozen=True)
class TextDocument:
    season: int
    team_id: int
    text: str
    source_url: str
    captured_at: Optional[str]
    game_date: Optional[str]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_market_source_paths(gender: str, external_dir: Optional[Path] = None) -> list[Path]:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    preferred = root / f"{gender}HistoricalTournamentOdds.csv"
    paths = []
    if preferred.exists():
        paths.append(preferred)
    for path in sorted(root.glob(f"{gender}HistoricalOdds_*.csv")):
        if path not in paths:
            paths.append(path)
    return paths


def find_live_market_source_paths(gender: str, season: int, external_dir: Optional[Path] = None) -> list[Path]:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    candidates = [
        root / f"{gender}MatchupOdds_{season}.csv",
        root / f"{gender}ActionNetworkOdds_{season}.csv",
        root / f"{gender}ManualOdds_{season}.csv",
        root / f"{gender}PredictionMarketOdds_{season}.csv",
        root / f"{gender}KalshiPredictionMarketOdds_{season}.csv",
        root / f"{gender}PolymarketPredictionMarketOdds_{season}.csv",
    ]
    if gender == "M":
        candidates.append(root / f"MTeamRankingsOdds_{season}.csv")
    return [path for path in candidates if path.exists()]


def find_live_silver_matchup_paths(gender: str, season: int, external_dir: Optional[Path] = None) -> list[Path]:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    exact = root / f"{gender}SilverBulletinMatchupProjections_{season}.csv"
    paths: list[Path] = []
    if exact.exists():
        paths.append(exact)
    for path in sorted(root.glob(f"{gender}SilverBulletinMatchupProjections_*.csv")):
        if path not in paths:
            paths.append(path)
    return paths


def find_live_model_matchup_paths(gender: str, season: int, external_dir: Optional[Path] = None) -> list[Path]:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    patterns = [
        f"{gender}SilverBulletinMatchupProjections_{season}.csv",
        f"{gender}SilverBulletinMatchupProjections_*.csv",
        f"{gender}BartTorvikMatchupProjections_{season}.csv",
        f"{gender}BartTorvikMatchupProjections_*.csv",
        f"{gender}WarrenNolanMatchupProjections_{season}.csv",
        f"{gender}WarrenNolanMatchupProjections_*.csv",
        f"{gender}HerHoopStatsMatchupProjections_{season}.csv",
        f"{gender}HerHoopStatsMatchupProjections_*.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in paths:
                paths.append(path)
    return paths


def _american_to_prob(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    neg = numeric < 0
    out.loc[neg] = -numeric.loc[neg] / (-numeric.loc[neg] + 100.0)
    out.loc[~neg] = 100.0 / (numeric.loc[~neg] + 100.0)
    return out


def _decimal_to_prob(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").where(lambda value: value > 0)
    return 1.0 / numeric


def standardize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "LastSpread", "SnapshotTime", "Source", "BookCount"])
    df = frame.copy()
    for column in ["Season", "T1", "T2"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column for column in ["Season", "T1", "T2"] if column in df.columns]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "LastSpread", "SnapshotTime", "Source", "BookCount"])
    df["Season"] = df["Season"].astype(int)
    df["T1"] = df["T1"].astype(int)
    df["T2"] = df["T2"].astype(int)

    if "MarketProb" not in df.columns:
        df["MarketProb"] = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        df["MarketProb"] = pd.to_numeric(df["MarketProb"], errors="coerce")

    missing_market_prob = df["MarketProb"].isna()
    if missing_market_prob.any() and {"Team1DecimalOdds", "Team2DecimalOdds"}.issubset(df.columns):
        p1 = _decimal_to_prob(df["Team1DecimalOdds"])
        p2 = _decimal_to_prob(df["Team2DecimalOdds"])
        denom = (p1 + p2).replace(0, np.nan)
        df.loc[missing_market_prob, "MarketProb"] = (p1 / denom).loc[missing_market_prob]
        missing_market_prob = df["MarketProb"].isna()
    if missing_market_prob.any() and {"Team1Moneyline", "Team2Moneyline"}.issubset(df.columns):
        p1 = _american_to_prob(df["Team1Moneyline"])
        p2 = _american_to_prob(df["Team2Moneyline"])
        denom = (p1 + p2).replace(0, np.nan)
        df.loc[missing_market_prob, "MarketProb"] = (p1 / denom).loc[missing_market_prob]
        missing_market_prob = df["MarketProb"].isna()
    if missing_market_prob.any() and "NoVigProb" in df.columns:
        df.loc[missing_market_prob, "MarketProb"] = pd.to_numeric(df.loc[missing_market_prob, "NoVigProb"], errors="coerce")

    if "LastSpread" in df.columns:
        df["LastSpread"] = pd.to_numeric(df["LastSpread"], errors="coerce")
    else:
        df["LastSpread"] = pd.Series(np.nan, index=df.index, dtype=float)

    snapshot_column = next((column for column in ["SnapshotTime", "CommenceTime", "GameTime", "EventTime", "Date"] if column in df.columns), None)
    if snapshot_column is None:
        df["SnapshotTime"] = pd.NaT
    else:
        df["SnapshotTime"] = pd.to_datetime(df[snapshot_column], errors="coerce", utc=True)

    if "Source" not in df.columns:
        if "_SourceFile" in df.columns:
            df["Source"] = df["_SourceFile"].astype(str)
        else:
            df["Source"] = "unknown"
    df["Source"] = df["Source"].fillna("unknown").astype(str)

    if "BookCount" in df.columns:
        df["BookCount"] = pd.to_numeric(df["BookCount"], errors="coerce")
    else:
        df["BookCount"] = pd.Series(np.nan, index=df.index, dtype=float)
    if df["BookCount"].isna().any():
        derived = pd.Series(np.nan, index=df.index, dtype=float)
        for source_col in ["Book", "Bookmakers", "BookmakersUsed", "BookTitle"]:
            if source_col not in df.columns:
                continue
            text = df[source_col].fillna("").astype(str).str.strip()
            count = text.apply(
                lambda value: float(len({token.strip() for token in value.replace(",", "|").split("|") if token.strip()}))
                if value
                else np.nan
            )
            derived = derived.fillna(count)
        df["BookCount"] = df["BookCount"].fillna(derived)
    return df


def aggregate_market_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "MarketProb",
                "MarketLogit",
                "MarketConfidence",
                "LastSpread",
                "AbsLastSpread",
                "MarketProbMean",
                "MarketProbMedian",
                "MarketProbStd",
                "SpreadMean",
                "SpreadMedian",
                "SpreadStd",
                "BookCountMean",
                "BookCountMax",
                "BookCountTotal",
                "MarketRowCount",
                "MarketSourceCount",
                "SnapshotTime",
                "Source",
            ]
        )

    df = frame.copy()
    if "SnapshotTime" in df.columns:
        df["SnapshotTime"] = pd.to_datetime(df["SnapshotTime"], errors="coerce", utc=True)
    else:
        df["SnapshotTime"] = pd.NaT
    if "Source" not in df.columns:
        df["Source"] = "unknown"
    df["Source"] = df["Source"].fillna("unknown").astype(str)
    if "BookCount" not in df.columns:
        df["BookCount"] = np.nan
    df["BookCount"] = pd.to_numeric(df["BookCount"], errors="coerce")
    df["MarketProb"] = pd.to_numeric(df["MarketProb"], errors="coerce")
    if "LastSpread" not in df.columns:
        df["LastSpread"] = np.nan
    df["LastSpread"] = pd.to_numeric(df["LastSpread"], errors="coerce")

    grouped_rows = []
    for (season, t1, t2), group in df.groupby(["Season", "T1", "T2"], sort=False):
        ordered = group.sort_values(["SnapshotTime", "Source"], na_position="last")
        latest_prob = ordered.loc[ordered["MarketProb"].notna()]
        latest_prob_row = latest_prob.iloc[-1] if not latest_prob.empty else ordered.iloc[-1]
        latest_spread = ordered.loc[ordered["LastSpread"].notna()]
        latest_spread_row = latest_spread.iloc[-1] if not latest_spread.empty else ordered.iloc[-1]
        market_prob = pd.to_numeric(group["MarketProb"], errors="coerce")
        spread = pd.to_numeric(group["LastSpread"], errors="coerce")
        book_count = pd.to_numeric(group["BookCount"], errors="coerce")
        grouped_rows.append(
            {
                "Season": int(season),
                "T1": int(t1),
                "T2": int(t2),
                "MarketProb": float(latest_prob_row["MarketProb"]) if pd.notna(latest_prob_row["MarketProb"]) else np.nan,
                "MarketLogit": (
                    float(
                        np.log(
                            np.clip(float(latest_prob_row["MarketProb"]), 1e-6, 1.0 - 1e-6)
                            / (1.0 - np.clip(float(latest_prob_row["MarketProb"]), 1e-6, 1.0 - 1e-6))
                        )
                    )
                    if pd.notna(latest_prob_row["MarketProb"])
                    else np.nan
                ),
                "MarketConfidence": (
                    float(abs(float(latest_prob_row["MarketProb"]) - 0.5) * 2.0)
                    if pd.notna(latest_prob_row["MarketProb"])
                    else np.nan
                ),
                "LastSpread": float(latest_spread_row["LastSpread"]) if pd.notna(latest_spread_row["LastSpread"]) else np.nan,
                "AbsLastSpread": abs(float(latest_spread_row["LastSpread"])) if pd.notna(latest_spread_row["LastSpread"]) else np.nan,
                "MarketProbMean": float(market_prob.mean()) if market_prob.notna().any() else np.nan,
                "MarketProbMedian": float(market_prob.median()) if market_prob.notna().any() else np.nan,
                "MarketProbStd": float(market_prob.std(ddof=0)) if market_prob.notna().any() else np.nan,
                "SpreadMean": float(spread.mean()) if spread.notna().any() else np.nan,
                "SpreadMedian": float(spread.median()) if spread.notna().any() else np.nan,
                "SpreadStd": float(spread.std(ddof=0)) if spread.notna().any() else np.nan,
                "BookCountMean": float(book_count.mean()) if book_count.notna().any() else np.nan,
                "BookCountMax": float(book_count.max()) if book_count.notna().any() else np.nan,
                "BookCountTotal": float(book_count.sum()) if book_count.notna().any() else np.nan,
                "MarketRowCount": int(len(group)),
                "MarketSourceCount": int(group["Source"].nunique()),
                "SnapshotTime": latest_prob_row["SnapshotTime"] if pd.notna(latest_prob_row["SnapshotTime"]) else latest_spread_row["SnapshotTime"],
                "Source": str(latest_prob_row["Source"]) if pd.notna(latest_prob_row["MarketProb"]) else str(latest_spread_row["Source"]),
            }
        )
    return pd.DataFrame(grouped_rows)


def standardize_matchup_model_frame(frame: pd.DataFrame, *, source_name: str = "") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "ModelProb",
                "ModelSpread",
                "ModelProjectedTotal",
                "ModelRound",
                "ModelRoundText",
                "SnapshotTime",
                "Source",
            ]
        )
    df = frame.copy()
    rename_map = {}
    if {"Team1ID", "Team2ID"}.issubset(df.columns):
        rename_map.update({"Team1ID": "T1", "Team2ID": "T2"})
    elif {"T1", "T2"}.issubset(df.columns):
        pass
    elif {"Team1", "Team2"}.issubset(df.columns):
        rename_map.update({"Team1": "T1", "Team2": "T2"})
    if rename_map:
        df = df.rename(columns=rename_map)

    for column in ["Season", "T1", "T2"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column for column in ["Season", "T1", "T2"] if column in df.columns]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "ModelProb",
                "ModelSpread",
                "ModelProjectedTotal",
                "ModelRound",
                "ModelRoundText",
                "SnapshotTime",
                "Source",
            ]
        )
    df["Season"] = df["Season"].astype(int)
    df["T1"] = df["T1"].astype(int)
    df["T2"] = df["T2"].astype(int)

    def first_numeric(columns: list[str]) -> pd.Series:
        values = pd.Series(np.nan, index=df.index, dtype=float)
        for column in columns:
            if column not in df.columns:
                continue
            values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
        return values

    round_text_col = next(
        (column for column in ["ModelRoundText", "SilverRoundText", "WarrenRoundText", "HerHoopRoundText", "RoundText"] if column in df.columns),
        None,
    )
    snapshot_col = next(
        (column for column in ["SnapshotTime", "EventTime", "CommenceTime", "Date", "EventDate"] if column in df.columns),
        None,
    )

    out = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "T1": df["T1"].astype(int),
            "T2": df["T2"].astype(int),
            "ModelProb": first_numeric(["ModelProb", "SilverProb", "WarrenProb", "HerHoopProb", "Prob", "Team1ImpliedProb"]),
            "ModelSpread": first_numeric(["ModelSpread", "SilverSpread", "WarrenSpread", "HerHoopSpread", "Spread", "Team1Spread"]),
            "ModelProjectedTotal": first_numeric(
                ["ModelProjectedTotal", "SilverProjectedTotal", "WarrenProjectedTotal", "HerHoopProjectedTotal", "ProjectedTotal"]
            ),
            "ModelRound": first_numeric(["ModelRound", "SilverRound", "WarrenRound", "HerHoopRound", "Round"]),
            "ModelRoundText": df[round_text_col].astype(str) if round_text_col else "",
            "SnapshotTime": pd.to_datetime(df[snapshot_col], errors="coerce", utc=True) if snapshot_col else pd.NaT,
            "Source": (
                df["Source"].fillna(source_name).astype(str)
                if "Source" in df.columns
                else pd.Series(str(source_name or "unknown"), index=df.index, dtype="object")
            ),
        }
    )
    return out


def aggregate_matchup_model_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "ModelMatchupProb",
                "ModelMatchupSpread",
                "ModelProjectedTotal",
                "ModelRound",
                "ModelMatchupProbMean",
                "ModelMatchupProbMedian",
                "ModelMatchupProbStd",
                "ModelMatchupSpreadMean",
                "ModelMatchupSpreadMedian",
                "ModelMatchupSpreadStd",
                "ModelSourceCount",
                "ModelRowCount",
                "ModelSourceList",
                "SnapshotTime",
                "Source",
            ]
        )

    df = frame.copy()
    df["SnapshotTime"] = pd.to_datetime(df.get("SnapshotTime"), errors="coerce", utc=True)
    df["Source"] = df.get("Source", "unknown").fillna("unknown").astype(str)
    df["ModelProb"] = pd.to_numeric(df.get("ModelProb"), errors="coerce")
    df["ModelSpread"] = pd.to_numeric(df.get("ModelSpread"), errors="coerce")
    df["ModelProjectedTotal"] = pd.to_numeric(df.get("ModelProjectedTotal"), errors="coerce")
    df["ModelRound"] = pd.to_numeric(df.get("ModelRound"), errors="coerce")

    grouped_rows = []
    for (season, t1, t2), group in df.groupby(["Season", "T1", "T2"], sort=False):
        ordered = group.sort_values(["SnapshotTime", "Source"], na_position="last")
        last_row = ordered.iloc[-1]
        model_prob = pd.to_numeric(group["ModelProb"], errors="coerce")
        model_spread = pd.to_numeric(group["ModelSpread"], errors="coerce")
        grouped_rows.append(
            {
                "Season": int(season),
                "T1": int(t1),
                "T2": int(t2),
                "ModelMatchupProb": float(last_row["ModelProb"]) if pd.notna(last_row["ModelProb"]) else np.nan,
                "ModelMatchupSpread": float(last_row["ModelSpread"]) if pd.notna(last_row["ModelSpread"]) else np.nan,
                "ModelProjectedTotal": float(last_row["ModelProjectedTotal"]) if pd.notna(last_row["ModelProjectedTotal"]) else np.nan,
                "ModelRound": float(last_row["ModelRound"]) if pd.notna(last_row["ModelRound"]) else np.nan,
                "ModelMatchupProbMean": float(model_prob.mean()) if model_prob.notna().any() else np.nan,
                "ModelMatchupProbMedian": float(model_prob.median()) if model_prob.notna().any() else np.nan,
                "ModelMatchupProbStd": float(model_prob.std(ddof=0)) if model_prob.notna().any() else np.nan,
                "ModelMatchupSpreadMean": float(model_spread.mean()) if model_spread.notna().any() else np.nan,
                "ModelMatchupSpreadMedian": float(model_spread.median()) if model_spread.notna().any() else np.nan,
                "ModelMatchupSpreadStd": float(model_spread.std(ddof=0)) if model_spread.notna().any() else np.nan,
                "ModelSourceCount": int(group["Source"].nunique()),
                "ModelRowCount": int(len(group)),
                "ModelSourceList": "|".join(sorted(group["Source"].astype(str).unique())),
                "SnapshotTime": last_row["SnapshotTime"],
                "Source": str(last_row["Source"]),
            }
        )
    return pd.DataFrame(grouped_rows)


def flatten_theoddsapi_snapshot(path: Path, gender: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()

    lookup = build_team_name_lookup(gender)
    season_match = next((token for token in path.stem.split("_") if token.isdigit() and len(token) == 4), None)
    season = int(season_match) if season_match else None
    rows = []
    for event in payload:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            continue
        home_id = resolve_team_id(home, lookup)
        away_id = resolve_team_id(away, lookup)
        if home_id is None or away_id is None:
            continue
        t1, t2 = sorted((int(home_id), int(away_id)))
        is_home_t1 = int(home_id) == t1
        h2h_probs = []
        spreads = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])
                if key == "h2h" and len(outcomes) >= 2:
                    prices = {item.get("name"): item.get("price") for item in outcomes}
                    p_home = _american_to_prob(pd.Series([prices.get(home)])).iloc[0]
                    p_away = _american_to_prob(pd.Series([prices.get(away)])).iloc[0]
                    denom = p_home + p_away
                    if np.isfinite(denom) and denom > 0:
                        p_t1 = (p_home / denom) if is_home_t1 else (p_away / denom)
                        h2h_probs.append(float(p_t1))
                elif key == "spreads" and len(outcomes) >= 2:
                    points = {item.get("name"): item.get("point") for item in outcomes}
                    point = points.get(home if is_home_t1 else away)
                    if point is not None:
                        spreads.append(float(point))
        rows.append(
            {
                "Season": season,
                "T1": t1,
                "T2": t2,
                "Team1Name": home if is_home_t1 else away,
                "Team2Name": away if is_home_t1 else home,
                "MarketProb": float(np.nanmean(h2h_probs)) if h2h_probs else np.nan,
                "LastSpread": float(np.nanmean(spreads)) if spreads else np.nan,
                "BookCount": float(len(h2h_probs) if h2h_probs else len(spreads) if spreads else 0),
                "SnapshotTime": pd.to_datetime(event.get("commence_time"), errors="coerce", utc=True),
                "Source": path.name,
            }
        )
    return pd.DataFrame(rows)


def load_local_text_corpus(
    gender: str,
    seasons: Optional[list[int]] = None,
    raw_text_dir: Optional[Path] = None,
) -> pd.DataFrame:
    root = Path(raw_text_dir) if raw_text_dir is not None else RAW_TEXT_DIR
    if not root.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "Text", "SourceURL", "CapturedAt", "GameDate"])
    lookup = build_team_name_lookup(gender)
    season_filter = None if seasons is None else {int(season) for season in seasons}
    rows: list[dict[str, object]] = []
    for season_dir in sorted(root.glob("*")):
        if not season_dir.is_dir():
            continue
        try:
            season = int(season_dir.name)
        except ValueError:
            continue
        if season_filter is not None and season not in season_filter:
            continue
        for path in sorted(season_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    team_id = payload.get("TeamID")
                    if team_id is None:
                        team_name = payload.get("TeamName") or payload.get("School")
                        team_id = resolve_team_id(team_name, lookup) if team_name else None
                    try:
                        team_id = int(team_id)
                    except (TypeError, ValueError):
                        continue
                    text = str(payload.get("Text") or payload.get("Summary") or payload.get("Body") or "").strip()
                    if not text:
                        continue
                    rows.append(
                        {
                            "Season": season,
                            "TeamID": team_id,
                            "Text": text,
                            "SourceURL": str(payload.get("SourceURL") or ""),
                            "CapturedAt": payload.get("CapturedAt") or payload.get("FetchedAt"),
                            "GameDate": payload.get("GameDate") or payload.get("Date"),
                        }
                    )
    return pd.DataFrame(rows)


def fetch_url_text(url: str, timeout: int = 20) -> dict[str, object]:
    if requests is None:
        raise RuntimeError("requests is unavailable")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    return {
        "SourceURL": url,
        "FetchedAt": datetime.now(timezone.utc).isoformat(),
        "Text": text,
    }
