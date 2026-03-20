from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

from hc.constants import (
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_TEXT_MODEL_DEV,
    EXTERNAL_DIR,
    GUARDRAIL_YEARS,
    MARKET_POLICY_BY_GENDER,
    MARKET_POLICY_PRE_TIP_ALL,
    MARKET_POLICY_SELECTION_WEEK,
    MARKET_POLICY_SELECTION_WEEK_PLUS,
    PRIMARY_YEARS,
    RESULTS_DIR,
    TEXT_COMPONENTS,
    TEXT_DIM_CHOICES,
    TEXT_EMBED_DIR,
)
from hc.data_sources import (
    aggregate_market_consensus,
    find_market_source_paths,
    flatten_theoddsapi_snapshot,
    load_local_text_corpus,
    read_csv_if_exists,
    standardize_market_frame,
)
from zizzii_features import build_team_features, tourney_snapshot_cutoff_lookup


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_EMBED_DIR.mkdir(parents=True, exist_ok=True)


def parse_seasons(raw: Optional[str]) -> Optional[list[int]]:
    if not raw:
        return None
    seasons = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        seasons.append(int(token))
    return seasons or None


def resolve_all_tourney_seasons(gender: str) -> list[int]:
    seasons = set()
    for name in [f"{gender}NCAATourneyCompactResults.csv", f"{gender}RegularSeasonCompactResults.csv"]:
        path = DATA_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["Season"])
        seasons.update(pd.to_numeric(df["Season"], errors="coerce").dropna().astype(int).unique().tolist())
    return sorted(seasons)


def tournament_pair_counts(gender: str) -> dict[int, int]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    counts = df.groupby("Season").size()
    return {int(season): int(count) for season, count in counts.items()}


def tournament_pair_sets(gender: str) -> dict[int, set[tuple[int, int]]]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    pair_map: dict[int, set[tuple[int, int]]] = {}
    for row in df.itertuples(index=False):
        season = int(row.Season)
        pair = tuple(sorted((int(row.WTeamID), int(row.LTeamID))))
        pair_map.setdefault(season, set()).add(pair)
    return pair_map


def tournament_team_counts(gender: str) -> dict[int, int]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    rows = []
    for row in df.itertuples(index=False):
        rows.append((int(row.Season), int(row.WTeamID)))
        rows.append((int(row.Season), int(row.LTeamID)))
    frame = pd.DataFrame(rows, columns=["Season", "TeamID"]).drop_duplicates()
    counts = frame.groupby("Season")["TeamID"].nunique()
    return {int(season): int(count) for season, count in counts.items()}


def tournament_team_ids(gender: str) -> dict[int, set[int]]:
    path = DATA_DIR / f"{gender}NCAATourneyCompactResults.csv"
    df = pd.read_csv(path, usecols=["Season", "WTeamID", "LTeamID"])
    rows = []
    for row in df.itertuples(index=False):
        rows.append((int(row.Season), int(row.WTeamID)))
        rows.append((int(row.Season), int(row.LTeamID)))
    frame = pd.DataFrame(rows, columns=["Season", "TeamID"]).drop_duplicates()
    return {
        int(season): set(pd.to_numeric(group["TeamID"], errors="coerce").dropna().astype(int).tolist())
        for season, group in frame.groupby("Season", sort=True)
    }


def season_cutoff_bounds(gender: str) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    cutoff_lookup = tourney_snapshot_cutoff_lookup(gender, data_dir=DATA_DIR)
    bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for season, cutoff in cutoff_lookup.items():
        cutoff = pd.Timestamp(cutoff)
        bounds[int(season)] = (cutoff, cutoff + timedelta(days=7))
    return bounds


def cache_path(name: str, suffix: str = ".parquet") -> Path:
    return CACHE_DIR / f"{name}{suffix}"


def build_market_history(
    gender: str,
    seasons: Optional[list[int]] = None,
    market_policy: Optional[str] = None,
    include_live_raw: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dirs()
    selected_seasons = set(seasons or resolve_all_tourney_seasons(gender))
    policy = market_policy or MARKET_POLICY_BY_GENDER[gender]
    frames = []
    for path in find_market_source_paths(gender, EXTERNAL_DIR):
        current = read_csv_if_exists(path)
        if current.empty:
            continue
        current["_SourcePath"] = path.name
        frames.append(current)

    if include_live_raw:
        raw_dir = EXTERNAL_DIR / "raw-theoddsapi"
        for path in sorted(raw_dir.glob(f"{gender}_*.json")):
            current = flatten_theoddsapi_snapshot(path, gender)
            if current.empty:
                continue
            current["_SourcePath"] = path.name
            frames.append(current)

    if not frames:
        empty = pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread", "SnapshotTime", "Source"])
        return empty, {"rows": 0, "coverage_by_season": {}, "spread_coverage_by_season": {}, "market_policy": policy}

    frame = pd.concat(frames, ignore_index=True)
    frame = standardize_market_frame(frame)
    frame = frame[frame["Season"].isin(selected_seasons)].copy()
    if frame.empty:
        empty = pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread", "SnapshotTime", "Source"])
        return empty, {"rows": 0, "coverage_by_season": {}, "spread_coverage_by_season": {}, "market_policy": policy}

    bounds = season_cutoff_bounds(gender)
    if policy == MARKET_POLICY_SELECTION_WEEK:
        mask = pd.Series(False, index=frame.index, dtype=bool)
        for season, (_, upper) in bounds.items():
            season_mask = frame["Season"].eq(season)
            times = pd.to_datetime(frame["SnapshotTime"], errors="coerce", utc=True)
            mask |= season_mask & times.notna() & times.le(upper)
        frame = frame.loc[mask].copy()
    elif policy == MARKET_POLICY_SELECTION_WEEK_PLUS:
        frame["SelectionWeekOnly"] = False
        times = pd.to_datetime(frame["SnapshotTime"], errors="coerce", utc=True)
        for season, (_, upper) in bounds.items():
            season_mask = frame["Season"].eq(season)
            frame.loc[season_mask & times.notna() & times.le(upper), "SelectionWeekOnly"] = True

    frame["MarketProb"] = pd.to_numeric(frame["MarketProb"], errors="coerce")
    frame = frame.dropna(subset=["MarketProb"]).copy()
    frame = frame[(frame["MarketProb"] > 0.0) & (frame["MarketProb"] < 1.0)].copy()
    frame["LastSpread"] = pd.to_numeric(frame["LastSpread"], errors="coerce")
    frame = aggregate_market_consensus(frame)
    if not frame.empty and "AbsLastSpread" in frame.columns:
        frame["AbsLastSpread"] = pd.to_numeric(frame["AbsLastSpread"], errors="coerce")

    pair_sets = tournament_pair_sets(gender)
    coverage_by_season = {}
    spread_coverage_by_season = {}
    for season in sorted(selected_seasons):
        season_df = frame.loc[frame["Season"] == season].copy()
        actual_pairs = pair_sets.get(season, set())
        if not actual_pairs:
            coverage_by_season[season] = 0.0
            spread_coverage_by_season[season] = 0.0
            continue
        pair_hits = [
            (int(t1), int(t2)) in actual_pairs
            for t1, t2 in zip(
                pd.to_numeric(season_df["T1"], errors="coerce").fillna(-1).astype(int),
                pd.to_numeric(season_df["T2"], errors="coerce").fillna(-1).astype(int),
            )
        ]
        season_df["_ActualPair"] = pair_hits
        matched = season_df.loc[season_df["_ActualPair"]].copy()
        covered_pairs = {
            (int(row.T1), int(row.T2))
            for row in matched[["T1", "T2"]].itertuples(index=False)
        }
        coverage_by_season[season] = float(len(covered_pairs) / len(actual_pairs))
        spread_coverage_by_season[season] = float(matched["LastSpread"].notna().mean()) if not matched.empty else 0.0

    path = cache_path(f"market_history_{gender}_{policy}")
    frame.to_parquet(path, index=False)
    summary = {
        "rows": int(len(frame)),
        "path": str(path),
        "coverage_by_season": coverage_by_season,
        "spread_coverage_by_season": spread_coverage_by_season,
        "consensus_feature_columns": [
            column
            for column in [
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
            ]
            if column in frame.columns
        ],
        "market_policy": policy,
    }
    return frame, summary


def build_team_snapshots(gender: str, seasons: Optional[list[int]] = None) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dirs()
    selected_seasons = set(seasons or resolve_all_tourney_seasons(gender))
    frame = build_team_features(gender=gender, data_dir=DATA_DIR, external_dir=EXTERNAL_DIR)
    frame = frame[frame["Season"].isin(selected_seasons)].copy()
    path = cache_path(f"team_snapshots_{gender}")
    frame.to_parquet(path, index=False)
    counts = tournament_team_counts(gender)
    field_ids = tournament_team_ids(gender)
    coverage = {}
    for season in sorted(selected_seasons):
        season_df = frame.loc[frame["Season"] == season]
        total_teams = counts.get(season, 0)
        if not total_teams:
            coverage[season] = 0.0
            continue
        season_ids = set(pd.to_numeric(season_df["TeamID"], errors="coerce").dropna().astype(int).tolist())
        coverage[season] = float(len(season_ids & field_ids.get(season, set())) / total_teams)
    return frame, {"rows": int(len(frame)), "path": str(path), "coverage_by_season": coverage}


def _load_sentence_transformer(model_name: str):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _encode_texts(texts: list[str], model_name: str, text_dim: int) -> np.ndarray:
    model = _load_sentence_transformer(model_name)
    if model is not None:
        embeddings = np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True), dtype=float)
    else:
        vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        dim = max(2, min(text_dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        if dim < 2:
            dense = matrix.toarray()
            if dense.shape[1] < text_dim:
                padding = np.zeros((dense.shape[0], text_dim - dense.shape[1]), dtype=float)
                return np.hstack([dense, padding])
            return dense[:, :text_dim]
        svd = TruncatedSVD(n_components=dim, random_state=42)
        embeddings = svd.fit_transform(matrix)
    if embeddings.shape[1] < text_dim:
        padding = np.zeros((embeddings.shape[0], text_dim - embeddings.shape[1]), dtype=float)
        embeddings = np.hstack([embeddings, padding])
    elif embeddings.shape[1] > text_dim:
        embeddings = embeddings[:, :text_dim]
    return embeddings


def build_text_embeddings(
    gender: str,
    seasons: Optional[list[int]] = None,
    text_dim: int = 32,
    model_name: str = DEFAULT_TEXT_MODEL_DEV,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dirs()
    if text_dim not in TEXT_DIM_CHOICES:
        raise ValueError(f"text_dim must be one of {TEXT_DIM_CHOICES}")
    docs = load_local_text_corpus(gender, seasons=seasons)
    if docs.empty:
        empty = pd.DataFrame(columns=["Season", "TeamID", "TextDocCount"])
        path = cache_path(f"text_embeddings_{gender}_{text_dim}d")
        empty.to_parquet(path, index=False)
        return empty, {"rows": 0, "path": str(path), "docs": 0, "encoder": "none"}

    docs["GameDate"] = pd.to_datetime(docs["GameDate"], errors="coerce", utc=True)
    docs["CapturedAt"] = pd.to_datetime(docs["CapturedAt"], errors="coerce", utc=True)
    docs["SortKey"] = docs["GameDate"].fillna(docs["CapturedAt"])
    docs = docs.sort_values(["Season", "TeamID", "SortKey"]).reset_index(drop=True)
    embeddings = _encode_texts(docs["Text"].astype(str).tolist(), model_name=model_name, text_dim=text_dim)
    embed_cols = [f"DocEmb_{i:02d}" for i in range(text_dim)]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    docs = pd.concat([docs.reset_index(drop=True), embed_df], axis=1)

    rows = []
    for (season, team_id), group in docs.groupby(["Season", "TeamID"], sort=True):
        group = group.sort_values("SortKey")
        recent3 = group.tail(3)
        recent5 = group.tail(5)
        weighted = recent5.copy()
        if weighted.empty:
            continue
        weights = np.linspace(1.0, 2.0, len(weighted))
        row = {"Season": int(season), "TeamID": int(team_id), "TextDocCount": int(len(group))}
        row["TextWindowDocCount"] = int(len(recent5))
        for prefix, subset in [("Recent3", recent3), ("Recent5", recent5)]:
            values = subset[embed_cols].to_numpy(dtype=float)
            if len(values) == 0:
                mean_vec = np.zeros(text_dim, dtype=float)
            else:
                mean_vec = values.mean(axis=0)
            for idx, value in enumerate(mean_vec):
                row[f"Text{prefix}_{idx:02d}"] = float(value)
        weighted_values = weighted[embed_cols].to_numpy(dtype=float)
        weighted_mean = np.average(weighted_values, axis=0, weights=weights)
        for idx, value in enumerate(weighted_mean):
            row[f"TextWeighted5_{idx:02d}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    path = cache_path(f"text_embeddings_{gender}_{text_dim}d")
    frame.to_parquet(path, index=False)
    coverage = frame.groupby("Season")["TeamID"].nunique().to_dict()
    return frame, {
        "rows": int(len(frame)),
        "path": str(path),
        "docs": int(len(docs)),
        "encoder": model_name if SentenceTransformer is not None else "tfidf_svd",
        "coverage_by_season": {int(season): int(count) for season, count in coverage.items()},
    }


def build_all(
    seasons: Optional[list[int]] = None,
    genders: tuple[str, ...] = ("M", "W"),
    text_dim: int = 32,
    text_model: str = DEFAULT_TEXT_MODEL_DEV,
) -> dict[str, object]:
    ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_seasons = seasons or sorted(set(resolve_all_tourney_seasons("M")) | set(resolve_all_tourney_seasons("W")))
    summary = {
        "run_id": run_id,
        "seasons": selected_seasons,
        "genders": {},
    }
    for gender in genders:
        team_frame, team_summary = build_team_snapshots(gender, seasons=selected_seasons)
        market_frame, market_summary = build_market_history(
            gender,
            seasons=selected_seasons,
            market_policy=MARKET_POLICY_BY_GENDER[gender],
            include_live_raw=True,
        )
        text_frame, text_summary = build_text_embeddings(
            gender,
            seasons=selected_seasons,
            text_dim=text_dim,
            model_name=text_model,
        )
        summary["genders"][gender] = {
            "team_snapshots": team_summary,
            "market_history": market_summary,
            "text_embeddings": text_summary,
            "team_rows": int(len(team_frame)),
            "market_rows": int(len(market_frame)),
            "text_rows": int(len(text_frame)),
        }

    audit_path = RESULTS_DIR / f"hc_snapshot_audit_{run_id}.json"
    coverage_path = RESULTS_DIR / f"hc_snapshot_coverage_{run_id}.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage_payload = {
        gender: {
            "market_coverage": payload["market_history"].get("coverage_by_season", {}),
            "spread_coverage": payload["market_history"].get("spread_coverage_by_season", {}),
            "team_coverage": payload["team_snapshots"].get("coverage_by_season", {}),
            "text_coverage": payload["text_embeddings"].get("coverage_by_season", {}),
        }
        for gender, payload in summary["genders"].items()
    }
    coverage_path.write_text(json.dumps(coverage_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"audit_path": str(audit_path), "coverage_path": str(coverage_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HC snapshot caches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_all_parser = subparsers.add_parser("build-all")
    build_all_parser.add_argument("--seasons", default=None, help="Comma-separated seasons. Defaults to all available tournament seasons.")
    build_all_parser.add_argument("--genders", default="MW", help="Subset of genders to process, e.g. M, W, or MW.")
    build_all_parser.add_argument("--text-dim", type=int, default=32, choices=TEXT_DIM_CHOICES)
    build_all_parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL_DEV)

    args = parser.parse_args()
    genders = tuple(gender for gender in ["M", "W"] if gender in args.genders.upper())
    if not genders:
        raise SystemExit("No valid genders requested.")
    seasons = parse_seasons(args.seasons)
    result = build_all(seasons=seasons, genders=genders, text_dim=args.text_dim, text_model=args.text_model)
    print(f"HC snapshot audit written to: {result['audit_path']}")
    print(f"HC snapshot coverage written to: {result['coverage_path']}")


if __name__ == "__main__":
    main()
