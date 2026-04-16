from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.gold.overlay import _load_direct_market_candidates, load_men_injury_adjustments
from hc.ji_base import JIBaseConfig
from hc.ji_base.data import load_ji_team_features
from hc.ji_base.prediction_market import build_prediction_market_matchup_bundle

RESULTS = ROOT / "results"
SNAPSHOT_DIR = RESULTS / "source_snapshots"


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coverage_stats(frame: pd.DataFrame, *, key_cols: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "non_null_rows": 0, "distinct_keys": 0}
    return {
        "rows": int(len(frame)),
        "non_null_rows": int(frame.dropna(how="all").shape[0]),
        "distinct_keys": int(frame[key_cols].drop_duplicates().shape[0]) if key_cols else int(len(frame)),
    }


def build_source_snapshots(season: int = 2026) -> dict[str, Any]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    built_at = _timestamp()

    women_features = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4"))
    women_snapshot = women_features.loc[women_features["Season"] == int(season), [
        "Season",
        "TeamID",
        "Quality",
        "WomenCompositeQuality",
        "QualityWins",
        "OpponentQualityTournamentRank",
        "harry_Rating",
        "AvgBlkDiff",
    ]].copy()
    women_path = SNAPSHOT_DIR / f"women_consensus_quality_snapshot_{season}.csv"
    women_snapshot.to_csv(women_path, index=False)

    men_market = _load_direct_market_candidates("M", season)
    women_market = _load_direct_market_candidates("W", season)
    men_market_path = SNAPSHOT_DIR / f"direct_matchup_market_m_{season}.csv"
    women_market_path = SNAPSHOT_DIR / f"direct_matchup_market_w_{season}.csv"
    men_market.to_csv(men_market_path, index=False)
    women_market.to_csv(women_market_path, index=False)

    men_prediction_market_history = build_prediction_market_matchup_bundle(
        men_market.loc[:, ["Season", "T1", "T2"]].drop_duplicates(),
        "M",
    )
    women_prediction_market_history = build_prediction_market_matchup_bundle(
        women_market.loc[:, ["Season", "T1", "T2"]].drop_duplicates(),
        "W",
    )
    men_prediction_market_history_path = SNAPSHOT_DIR / f"historical_prediction_market_m_{season}.csv"
    women_prediction_market_history_path = SNAPSHOT_DIR / f"historical_prediction_market_w_{season}.csv"
    men_prediction_market_history.to_csv(men_prediction_market_history_path, index=False)
    women_prediction_market_history.to_csv(women_prediction_market_history_path, index=False)

    injury = load_men_injury_adjustments(season)
    injury_path = SNAPSHOT_DIR / f"men_injury_snapshot_{season}.csv"
    injury.to_csv(injury_path, index=False)

    manifest = {
        "season": int(season),
        "built_at": built_at,
        "snapshots": [
            {
                "name": "women_consensus_quality",
                "layer": "core season-safe features",
                "source_name": "ji_base_women_consensus_quality",
                "path": str(women_path),
                "coverage": _coverage_stats(women_snapshot, key_cols=["Season", "TeamID"]),
            },
            {
                "name": "direct_matchup_market_m",
                "layer": "submission-only overlay sources",
                "source_name": "direct_matchup_market_m",
                "path": str(men_market_path),
                "coverage": _coverage_stats(men_market, key_cols=["Season", "T1", "T2"]),
            },
            {
                "name": "direct_matchup_market_w",
                "layer": "submission-only overlay sources",
                "source_name": "direct_matchup_market_w",
                "path": str(women_market_path),
                "coverage": _coverage_stats(women_market, key_cols=["Season", "T1", "T2"]),
            },
            {
                "name": "historical_prediction_market_m",
                "layer": "structured history builder",
                "source_name": "historical_prediction_market_m",
                "path": str(men_prediction_market_history_path),
                "coverage": _coverage_stats(men_prediction_market_history, key_cols=["Season", "T1", "T2"]),
            },
            {
                "name": "historical_prediction_market_w",
                "layer": "structured history builder",
                "source_name": "historical_prediction_market_w",
                "path": str(women_prediction_market_history_path),
                "coverage": _coverage_stats(women_prediction_market_history, key_cols=["Season", "T1", "T2"]),
            },
            {
                "name": "men_structured_injury",
                "layer": "manual / current-year only",
                "source_name": "men_structured_injury",
                "path": str(injury_path),
                "coverage": _coverage_stats(injury, key_cols=["Season", "TeamID"]),
            },
        ],
    }
    manifest_path = SNAPSHOT_DIR / f"source_snapshot_manifest_{season}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    build_source_snapshots()


if __name__ == "__main__":
    main()
