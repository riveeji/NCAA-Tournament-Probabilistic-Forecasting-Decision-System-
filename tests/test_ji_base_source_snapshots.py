import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_source_snapshots.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_source_snapshots", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_source_snapshots_writes_manifest_and_expected_entries(monkeypatch):
    module = _load_module()
    workspace = Path.cwd()
    results_dir = workspace / ".pytest_source_snapshots"
    snapshot_dir = results_dir / "source_snapshots"
    results_dir.mkdir(exist_ok=True)
    snapshot_dir.mkdir(exist_ok=True)
    module.RESULTS = results_dir
    module.SNAPSHOT_DIR = snapshot_dir

    monkeypatch.setattr(
        module,
        "load_ji_team_features",
        lambda config: pd.DataFrame(
            [
                {
                    "Season": 2026,
                    "TeamID": 3101,
                    "Quality": 1.0,
                    "WomenCompositeQuality": 0.9,
                    "QualityWins": 0.4,
                    "OpponentQualityTournamentRank": 0.3,
                    "harry_Rating": 1.1,
                    "AvgBlkDiff": 0.2,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_direct_market_candidates",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 1, "T2": 2, "market_prob": 0.6, "source_used": "sportsbook", "gender": gender}]
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_market_matchup_bundle",
        lambda matchups, gender: pd.DataFrame(
            [
                {
                    "Season": int(matchups.iloc[0]["Season"]),
                    "T1": int(matchups.iloc[0]["T1"]),
                    "T2": int(matchups.iloc[0]["T2"]),
                    "PredictionMarketProbMean": 0.61,
                    "PredictionMarketProbStd": 0.03,
                    "PredictionMarketSourceCount": 2,
                    "PredictionMarketSnapshotCount": 4,
                    "PredictionMarketCoverage": 1.0,
                    "PredictionMarketDirectCoverage": 1.0,
                    "PredictionMarketHasProxy": 0.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.01, "confirmed_out": 1}]
        ),
    )

    manifest = module.build_source_snapshots(2026)

    assert manifest["season"] == 2026
    names = {entry["name"] for entry in manifest["snapshots"]}
    assert names == {
        "women_consensus_quality",
        "direct_matchup_market_m",
        "direct_matchup_market_w",
        "historical_prediction_market_m",
        "historical_prediction_market_w",
        "men_structured_injury",
    }
    assert (snapshot_dir / "source_snapshot_manifest_2026.json").exists()
