from pathlib import Path

import pandas as pd

from tools.build_prediction_market_snapshot_history import build_prediction_market_snapshot_history, build_summary


def _write_snapshot_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_prediction_market_snapshot_history_combines_source_specific_and_proxy(tmp_path: Path) -> None:
    _write_snapshot_csv(
        tmp_path / "MKalshiPredictionMarketOdds_2026.csv",
        [
            {
                "Season": 2026,
                "T1": 1104,
                "T2": 1341,
                "Team1Name": "Alabama",
                "Team2Name": "Prairie View A&M",
                "MarketProb": 0.92,
                "Team1Moneyline": -1100,
                "Team2Moneyline": 1100,
                "Book": "kalshi",
                "Source": "kalshi_prediction_market",
                "SourceURL": "https://example.com/kalshi",
                "SnapshotTime": "2026-03-19T00:00:00Z",
                "Notes": "",
            }
        ],
    )
    _write_snapshot_csv(
        tmp_path / "WPredictionMarketOdds_2026.csv",
        [
            {
                "Season": 2026,
                "T1": 3158,
                "T2": 3181,
                "Team1Name": "Charleston Cougars",
                "Team2Name": "Duke Blue Devils",
                "MarketProb": 0.01,
                "Team1Moneyline": 9900,
                "Team2Moneyline": -9900,
                "Book": "robinhood_or_kalshi",
                "Source": "prediction_market_proxy",
                "SourceURL": "manual:image_prediction_market_20260317",
                "SnapshotTime": "2026-03-17T10:15:00Z",
                "Notes": "proxy",
            }
        ],
    )

    histories = build_prediction_market_snapshot_history(tmp_path)

    assert len(histories["M"]) == 1
    assert histories["M"].iloc[0]["SourceGroup"] == "kalshi_direct"
    assert histories["M"].iloc[0]["SourceClass"] == "direct"

    assert len(histories["W"]) == 1
    assert histories["W"].iloc[0]["SourceGroup"] == "prediction_market_proxy"
    assert histories["W"].iloc[0]["SourceClass"] == "proxy_manual"


def test_build_summary_reports_source_breakdown(tmp_path: Path) -> None:
    _write_snapshot_csv(
        tmp_path / "WKalshiPredictionMarketOdds_2026.csv",
        [
            {
                "Season": 2026,
                "T1": 3276,
                "T2": 3390,
                "Team1Name": "Michigan",
                "Team2Name": "Vermont",
                "MarketProb": 0.88,
                "Team1Moneyline": -733,
                "Team2Moneyline": 733,
                "Book": "kalshi",
                "Source": "kalshi_prediction_market",
                "SourceURL": "https://example.com/kalshi",
                "SnapshotTime": "2026-03-19T01:00:00Z",
                "Notes": "",
            },
            {
                "Season": 2026,
                "T1": 3390,
                "T2": 3435,
                "Team1Name": "Vermont",
                "Team2Name": "Vanderbilt",
                "MarketProb": 0.12,
                "Team1Moneyline": 733,
                "Team2Moneyline": -733,
                "Book": "kalshi",
                "Source": "kalshi_prediction_market",
                "SourceURL": "https://example.com/kalshi-2",
                "SnapshotTime": "2026-03-19T02:00:00Z",
                "Notes": "",
            },
        ],
    )
    histories = build_prediction_market_snapshot_history(tmp_path)
    summary = build_summary(histories)

    assert summary["W"]["rows"] == 2
    assert summary["W"]["season_min"] == 2026
    assert summary["W"]["by_source_group"][0]["SourceGroup"] == "kalshi_direct"
