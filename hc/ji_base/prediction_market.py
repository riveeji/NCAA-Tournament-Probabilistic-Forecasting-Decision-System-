from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "external-data"

EXPECTED_SOURCE_GROUPS = {
    "M": ("kalshi_direct", "polymarket_direct"),
    "W": ("kalshi_direct", "polymarket_direct", "prediction_market_proxy"),
}


@lru_cache(maxsize=4)
def load_historical_prediction_market_snapshots(gender: str) -> pd.DataFrame:
    path = EXTERNAL_DIR / f"{gender}HistoricalPredictionMarketSnapshots.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "Season",
                "Gender",
                "T1",
                "T2",
                "MarketProb",
                "SourceGroup",
                "SourceClass",
                "SnapshotTime",
                "SnapshotDate",
            ]
        )
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce").astype("Int64")
    frame["T1"] = pd.to_numeric(frame["T1"], errors="coerce").astype("Int64")
    frame["T2"] = pd.to_numeric(frame["T2"], errors="coerce").astype("Int64")
    frame["MarketProb"] = pd.to_numeric(frame["MarketProb"], errors="coerce")
    frame["SnapshotTime"] = pd.to_datetime(frame.get("SnapshotTime"), errors="coerce", utc=True)
    if "SnapshotDate" not in frame.columns:
        frame["SnapshotDate"] = frame["SnapshotTime"].dt.date.astype("string")
    return frame.dropna(subset=["Season", "T1", "T2", "MarketProb"]).copy()


def build_prediction_market_matchup_bundle(matchups: pd.DataFrame, gender: str) -> pd.DataFrame:
    bundle = matchups[["Season", "T1", "T2"]].copy()
    defaults = {
        "PredictionMarketProbMean": 0.5,
        "PredictionMarketProbStd": 0.0,
        "PredictionMarketSourceCount": 0,
        "PredictionMarketSnapshotCount": 0,
        "PredictionMarketCoverage": 0.0,
        "PredictionMarketDirectCoverage": 0.0,
        "PredictionMarketHasProxy": 0.0,
    }
    for column, default in defaults.items():
        bundle[column] = default

    snapshots = load_historical_prediction_market_snapshots(gender)
    if snapshots.empty:
        return bundle

    merged = bundle.merge(snapshots, on=["Season", "T1", "T2"], how="left")
    expected_total = float(len(EXPECTED_SOURCE_GROUPS.get(gender, ()))) or 1.0
    expected_direct = float(sum(1 for token in EXPECTED_SOURCE_GROUPS.get(gender, ()) if token.endswith("_direct"))) or 1.0

    grouped = (
        merged.groupby(["Season", "T1", "T2"], dropna=False)
        .agg(
            PredictionMarketProbMean=("MarketProb", "mean"),
            PredictionMarketProbStd=("MarketProb", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) or 0.0)),
            PredictionMarketSourceCount=("SourceGroup", lambda s: int(pd.Series(s).dropna().nunique())),
            PredictionMarketSnapshotCount=("SnapshotTime", lambda s: int(pd.Series(s).dropna().shape[0])),
            _DirectSourceCount=("SourceGroup", lambda s: int(pd.Series(s).dropna().astype(str).str.endswith("_direct").sum())),
            _HasProxy=("SourceClass", lambda s: float(pd.Series(s).dropna().astype(str).eq("proxy_manual").any())),
        )
        .reset_index()
    )
    grouped["PredictionMarketCoverage"] = (grouped["PredictionMarketSourceCount"] / expected_total).clip(lower=0.0, upper=1.0)
    grouped["PredictionMarketDirectCoverage"] = (grouped["_DirectSourceCount"] / expected_direct).clip(lower=0.0, upper=1.0)
    grouped["PredictionMarketHasProxy"] = grouped["_HasProxy"].astype(float)
    grouped = grouped.drop(columns=["_DirectSourceCount", "_HasProxy"])

    out = bundle.drop(columns=list(defaults.keys())).merge(grouped, on=["Season", "T1", "T2"], how="left")
    for column, default in defaults.items():
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    out["PredictionMarketSourceCount"] = out["PredictionMarketSourceCount"].astype(int)
    out["PredictionMarketSnapshotCount"] = out["PredictionMarketSnapshotCount"].astype(int)
    return out
