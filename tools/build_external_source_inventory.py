from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

INVENTORY_ROWS = [
    {
        "source_name": "public_consensus_ranks_m",
        "layer": "model",
        "signal_horizon": "season_long",
        "tier": "A",
        "default_enabled": True,
        "path_pattern": "Ext_PublicBPIRank|Ext_PublicPOMRank|Ext_PublicNETRank|Ext_PublicTRankRank|Ext_PublicSORRank|Ext_PublicWABRank|Ext_CHH_KPRank|Ext_CHH_NETRank",
        "notes": "Core men season-long strength/rank consensus used in gold ratings.",
    },
    {
        "source_name": "public_consensus_ranks_w",
        "layer": "model",
        "signal_horizon": "season_long",
        "tier": "A",
        "default_enabled": True,
        "path_pattern": "Ext_PublicNETRank|Ext_PublicRPIRank|Ext_PublicPredRPIRank|Ext_WN_NET|Ext_WN_ELO|Ext_WN_RPI|Ext_WN_PredRPI",
        "notes": "Core women season-long consensus ratings used in gold ratings.",
    },
    {
        "source_name": "women_polls",
        "layer": "model",
        "signal_horizon": "season_long",
        "tier": "B",
        "default_enabled": False,
        "path_pattern": "Ext_PublicAPRank|Ext_PublicCoachesRank",
        "notes": "Useful as auxiliary women signal but more redundant/noisy than core rating stack.",
    },
    {
        "source_name": "ap_poll_proxy_m",
        "layer": "model",
        "signal_horizon": "season_long",
        "tier": "B",
        "default_enabled": True,
        "path_pattern": "Ext_PublicAverageRank",
        "notes": "Currently still used by wide gold APStrength feature; should be validated separately.",
    },
    {
        "source_name": "direct_matchup_market",
        "layer": "overlay",
        "signal_horizon": "current_year_matchup",
        "tier": "A",
        "default_enabled": True,
        "path_pattern": "external-data/*MatchupOdds_*.csv|*BartTorvikMatchupProjections_*.csv|*WarrenNolanMatchupProjections_*.csv|*HerHoopStatsMatchupProjections_*.csv",
        "notes": "Highest-value submission-only source; direct matchup pricing/projections.",
    },
    {
        "source_name": "men_confirmed_injury",
        "layer": "overlay",
        "signal_horizon": "current_year_team",
        "tier": "A",
        "default_enabled": True,
        "path_pattern": "external-data/MRotoWireInjuries_*.csv",
        "notes": "Submission-only men injury adjustment; only confirmed-out/high severity should matter.",
    },
    {
        "source_name": "kalshi_futures",
        "layer": "overlay",
        "signal_horizon": "current_year_derived",
        "tier": "C",
        "default_enabled": False,
        "path_pattern": "external-data/*KalshiPredictionMarketOdds_*.csv",
        "notes": "Derived futures prior; currently excluded from default overlay.",
    },
]


def write_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_name", "layer", "signal_horizon", "tier", "default_enabled", "path_pattern", "notes"],
        )
        writer.writeheader()
        writer.writerows(INVENTORY_ROWS)


def main() -> None:
    output = RESULTS / "external_source_inventory.csv"
    if len(sys.argv) > 1:
        output = Path(sys.argv[1]).resolve()
    write_inventory(output)


if __name__ == "__main__":
    main()
