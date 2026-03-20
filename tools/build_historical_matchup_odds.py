from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hc.constants import EXTERNAL_DIR, MARKET_POLICY_BY_GENDER, RESULTS_DIR
from hc.data_build import build_market_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a historical MatchupOdds_<season>.csv from HC historical tournament odds sources."
    )
    parser.add_argument("--season", type=int, required=True, help="Historical season to export, e.g. 2025.")
    parser.add_argument("--gender", choices=["M", "W"], help="Single gender export.")
    parser.add_argument("--all", action="store_true", help="Export both men and women.")
    parser.add_argument(
        "--market-policy",
        default=None,
        help="Optional market policy override. Defaults to the HC gender default policy.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXTERNAL_DIR),
        help="Directory for exported MatchupOdds CSV files.",
    )
    return parser.parse_args()


def export_gender(season: int, gender: str, market_policy: str | None, output_dir: Path) -> dict[str, object]:
    policy = market_policy or MARKET_POLICY_BY_GENDER[gender]
    frame, summary = build_market_history(
        gender=gender,
        seasons=[season],
        market_policy=policy,
        include_live_raw=False,
    )
    frame = frame.loc[pd.to_numeric(frame["Season"], errors="coerce").eq(season)].copy()
    output_path = output_dir / f"{gender}MatchupOdds_{season}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return {
        "gender": gender,
        "season": int(season),
        "market_policy": policy,
        "rows": int(len(frame)),
        "output": str(output_path),
        "coverage_by_season": summary.get("coverage_by_season", {}),
        "spread_coverage_by_season": summary.get("spread_coverage_by_season", {}),
        "consensus_feature_columns": summary.get("consensus_feature_columns", []),
    }


def main() -> None:
    args = parse_args()
    if not args.all and not args.gender:
        raise SystemExit("Pass --gender M/W or use --all.")

    genders = ["M", "W"] if args.all else [args.gender]
    output_dir = Path(args.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "season": int(args.season),
        "exports": [export_gender(args.season, gender, args.market_policy, output_dir) for gender in genders],
    }
    summary_path = RESULTS_DIR / f"historical_matchup_odds_export_{run_id}.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Historical matchup-odds export summary written to: {summary_path}")
    for item in payload["exports"]:
        print(
            f"[{item['gender']}] rows={item['rows']} policy={item['market_policy']} output={item['output']}"
        )


if __name__ == "__main__":
    main()
