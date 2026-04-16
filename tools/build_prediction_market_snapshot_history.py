from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"


FILE_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "M": [
        ("MKalshiPredictionMarketOdds_*.csv", "kalshi_direct", "direct"),
        ("MPolymarketPredictionMarketOdds_*.csv", "polymarket_direct", "direct"),
    ],
    "W": [
        ("WKalshiPredictionMarketOdds_*.csv", "kalshi_direct", "direct"),
        ("WPolymarketPredictionMarketOdds_*.csv", "polymarket_direct", "direct"),
        ("WPredictionMarketOdds_*.csv", "prediction_market_proxy", "proxy_manual"),
    ],
}


def _read_prediction_market_file(
    path: Path,
    gender: str,
    source_group: str,
    source_class: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()

    required_cols = [
        "Season",
        "T1",
        "T2",
        "Team1Name",
        "Team2Name",
        "MarketProb",
        "Book",
        "Source",
        "SourceURL",
        "SnapshotTime",
        "Notes",
    ]
    for col in required_cols:
        if col not in frame.columns:
            frame[col] = pd.NA

    out = frame.copy()
    out["Gender"] = gender
    out["SourceGroup"] = source_group
    out["SourceClass"] = source_class
    out["SourceFile"] = path.name
    out["SnapshotTime"] = pd.to_datetime(out["SnapshotTime"], errors="coerce", utc=True)
    out["SnapshotDate"] = out["SnapshotTime"].dt.date.astype("string")
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
    out["T1"] = pd.to_numeric(out["T1"], errors="coerce").astype("Int64")
    out["T2"] = pd.to_numeric(out["T2"], errors="coerce").astype("Int64")
    out["MarketProb"] = pd.to_numeric(out["MarketProb"], errors="coerce")
    if "Team1Moneyline" in out.columns:
        out["Team1Moneyline"] = pd.to_numeric(out["Team1Moneyline"], errors="coerce").astype("Int64")
    else:
        out["Team1Moneyline"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
    if "Team2Moneyline" in out.columns:
        out["Team2Moneyline"] = pd.to_numeric(out["Team2Moneyline"], errors="coerce").astype("Int64")
    else:
        out["Team2Moneyline"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
    if "LastSpread" in out.columns:
        out["LastSpread"] = pd.to_numeric(out["LastSpread"], errors="coerce")
    else:
        out["LastSpread"] = pd.NA

    keep = [
        "Season",
        "Gender",
        "T1",
        "T2",
        "Team1Name",
        "Team2Name",
        "MarketProb",
        "Team1Moneyline",
        "Team2Moneyline",
        "LastSpread",
        "Book",
        "Source",
        "SourceGroup",
        "SourceClass",
        "SnapshotTime",
        "SnapshotDate",
        "SourceURL",
        "SourceFile",
        "Notes",
    ]
    out = out[keep]
    out = out.dropna(subset=["Season", "T1", "T2", "MarketProb"])
    out = out.sort_values(["Season", "SnapshotTime", "T1", "T2", "SourceGroup", "SourceFile"]).reset_index(drop=True)
    return out


def build_prediction_market_snapshot_history(external_dir: Path = EXTERNAL_DIR) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for gender, specs in FILE_PATTERNS.items():
        frames: list[pd.DataFrame] = []
        for pattern, source_group, source_class in specs:
            for path in sorted(external_dir.glob(pattern)):
                frame = _read_prediction_market_file(path, gender, source_group, source_class)
                if not frame.empty:
                    frames.append(frame)
        if frames:
            history = pd.concat(frames, ignore_index=True)
            history = history.drop_duplicates(
                subset=["Season", "T1", "T2", "SourceGroup", "SnapshotTime", "SourceFile"],
                keep="last",
            ).sort_values(["Season", "SnapshotTime", "T1", "T2", "SourceGroup"])
            histories[gender] = history.reset_index(drop=True)
        else:
            histories[gender] = pd.DataFrame()
    return histories


def build_summary(histories: dict[str, pd.DataFrame]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for gender, frame in histories.items():
        if frame.empty:
            summary[gender] = {"rows": 0}
            continue
        by_source = (
            frame.groupby(["SourceGroup", "SourceClass"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["rows", "SourceGroup"], ascending=[False, True])
        )
        by_season = (
            frame.groupby("Season", dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("Season")
        )
        summary[gender] = {
            "rows": int(len(frame)),
            "season_min": int(frame["Season"].min()),
            "season_max": int(frame["Season"].max()),
            "source_groups": sorted(frame["SourceGroup"].dropna().astype(str).unique().tolist()),
            "snapshot_time_min": frame["SnapshotTime"].min().isoformat() if frame["SnapshotTime"].notna().any() else "",
            "snapshot_time_max": frame["SnapshotTime"].max().isoformat() if frame["SnapshotTime"].notna().any() else "",
            "by_source_group": by_source.to_dict(orient="records"),
            "by_season": by_season.to_dict(orient="records"),
        }
    return summary


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    histories = build_prediction_market_snapshot_history(EXTERNAL_DIR)
    for gender, frame in histories.items():
        output_path = EXTERNAL_DIR / f"{gender}HistoricalPredictionMarketSnapshots.csv"
        frame.to_csv(output_path, index=False)
    summary = build_summary(histories)
    summary_path = RESULTS_DIR / "prediction_market_snapshot_history_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
