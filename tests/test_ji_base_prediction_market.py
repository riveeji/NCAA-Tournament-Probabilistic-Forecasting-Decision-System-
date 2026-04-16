from pathlib import Path

import pandas as pd

from hc.ji_base.prediction_market import build_prediction_market_matchup_bundle, load_historical_prediction_market_snapshots


def test_prediction_market_bundle_aggregates_probabilities_and_coverage(monkeypatch):
    snapshots = pd.DataFrame(
        [
            {
                "Season": 2026,
                "Gender": "M",
                "T1": 1104,
                "T2": 1341,
                "MarketProb": 0.92,
                "SourceGroup": "kalshi_direct",
                "SourceClass": "direct",
                "SnapshotTime": "2026-03-19T00:00:00Z",
                "SnapshotDate": "2026-03-19",
            },
            {
                "Season": 2026,
                "Gender": "M",
                "T1": 1104,
                "T2": 1341,
                "MarketProb": 0.89,
                "SourceGroup": "polymarket_direct",
                "SourceClass": "direct",
                "SnapshotTime": "2026-03-19T01:00:00Z",
                "SnapshotDate": "2026-03-19",
            },
        ]
    )
    monkeypatch.setattr("hc.ji_base.prediction_market.load_historical_prediction_market_snapshots", lambda gender: snapshots)
    matchups = pd.DataFrame([{"Season": 2026, "T1": 1104, "T2": 1341}])

    bundle = build_prediction_market_matchup_bundle(matchups, "M")
    row = bundle.iloc[0]

    assert row["PredictionMarketProbMean"] == 0.905
    assert row["PredictionMarketSourceCount"] == 2
    assert row["PredictionMarketSnapshotCount"] == 2
    assert row["PredictionMarketCoverage"] == 1.0
    assert row["PredictionMarketDirectCoverage"] == 1.0
    assert row["PredictionMarketHasProxy"] == 0.0


def test_prediction_market_bundle_handles_missing_snapshot_history(monkeypatch):
    monkeypatch.setattr(
        "hc.ji_base.prediction_market.load_historical_prediction_market_snapshots",
        lambda gender: pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb"]),
    )
    matchups = pd.DataFrame([{"Season": 2026, "T1": 3163, "T2": 3390}])

    bundle = build_prediction_market_matchup_bundle(matchups, "W")
    row = bundle.iloc[0]

    assert row["PredictionMarketProbMean"] == 0.5
    assert row["PredictionMarketSourceCount"] == 0
    assert row["PredictionMarketCoverage"] == 0.0


def test_load_historical_prediction_market_snapshots_reads_generated_csv(tmp_path: Path, monkeypatch):
    external_dir = tmp_path
    path = external_dir / "MHistoricalPredictionMarketSnapshots.csv"
    pd.DataFrame(
        [
            {
                "Season": 2026,
                "Gender": "M",
                "T1": 1104,
                "T2": 1341,
                "MarketProb": 0.91,
                "SourceGroup": "kalshi_direct",
                "SourceClass": "direct",
                "SnapshotTime": "2026-03-19T00:00:00Z",
                "SnapshotDate": "2026-03-19",
            }
        ]
    ).to_csv(path, index=False)

    monkeypatch.setattr("hc.ji_base.prediction_market.EXTERNAL_DIR", external_dir)
    load_historical_prediction_market_snapshots.cache_clear()
    frame = load_historical_prediction_market_snapshots("M")
    load_historical_prediction_market_snapshots.cache_clear()

    assert len(frame) == 1
    assert float(frame.iloc[0]["MarketProb"]) == 0.91
