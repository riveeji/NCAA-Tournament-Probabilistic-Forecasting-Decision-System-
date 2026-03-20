from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend the final submission candidate among current/silver/goldshot/baseline.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--goldshot", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--silver-summary-json", default="")
    parser.add_argument("--goldshot-summary-json", required=True)
    parser.add_argument("--current-review-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_opposition_count(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    try:
        frame = pd.read_csv(file_path)
    except Exception:
        return 0
    if frame.empty:
        return 0
    def numeric_series(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    effective_book_count = pd.concat(
        [
            numeric_series("EffectiveBookCount"),
            numeric_series("BookCountMax"),
            numeric_series("BookCountTotal"),
            numeric_series("MarketRowCount"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    eligible = frame.loc[
        frame["ReviewEligible"].fillna(False).astype(bool)
        & frame["CloserToMarket"].fillna("").eq("Baseline")
        & pd.to_numeric(frame.get("MarketSourceCount"), errors="coerce").fillna(0.0).ge(2.0)
        & effective_book_count.fillna(0.0).ge(3.0)
        & frame["ManualReviewPriority"].fillna("").isin(["Critical", "High"])
    ]
    return int(len(eligible))


def main() -> None:
    args = parse_args()
    silver_summary = load_json(args.silver_summary_json)
    goldshot_summary = load_json(args.goldshot_summary_json)

    goldshot_auto = int(goldshot_summary.get("auto_applied_rows", 0))
    goldshot_total = int(goldshot_summary.get("total_changed_rows", 0))
    goldshot_model_supported = int(goldshot_summary.get("auto_model_supported_rows", 0))
    goldshot_round_total = (
        int(goldshot_summary.get("auto_applied_playin_rows", 0))
        + int(goldshot_summary.get("auto_applied_round1_rows", 0))
        + int(goldshot_summary.get("auto_applied_round2_rows", 0))
    )
    goldshot_market_model_share = (
        float(goldshot_model_supported) / float(goldshot_auto)
        if goldshot_auto > 0
        else 0.0
    )
    women_summary = silver_summary.get("women_live_rule_summary", {}) if isinstance(silver_summary, dict) else {}
    women_runtime_active = int(women_summary.get("women_live_rule_strong_rows", 0)) + int(
        women_summary.get("women_live_rule_extreme_rows", 0)
    )
    baseline_opposed = baseline_opposition_count(args.current_review_csv)

    recommended_label = "hc_current"
    recommended_path = str(Path(args.current).resolve())
    reasons = ["default to current HC candidate"]

    if 3 <= goldshot_total <= 12 and goldshot_round_total == goldshot_total and goldshot_market_model_share >= 0.5:
        recommended_label = "hc_goldshot"
        recommended_path = str(Path(args.goldshot).resolve())
        reasons = [
            "goldshot changed a bounded number of rows",
            "all applied rows stayed inside play-in/round-of-64/high-probability round-of-32 scope",
            "at least half of goldshot auto changes had market+model support",
        ]
    elif women_runtime_active > 0:
        recommended_label = "hc_silver_runtime"
        recommended_path = str(Path(args.silver).resolve())
        reasons = ["women runtime produced active strong/extreme rows"]
    elif baseline_opposed >= 3:
        recommended_label = "baseline"
        recommended_path = str(Path(args.baseline).resolve())
        reasons = ["market-backed review checklist shows broad opposition to current HC"]

    payload = {
        "recommended_label": recommended_label,
        "recommended_submission": recommended_path,
        "inputs": {
            "current": str(Path(args.current).resolve()),
            "silver": str(Path(args.silver).resolve()),
            "goldshot": str(Path(args.goldshot).resolve()),
            "baseline": str(Path(args.baseline).resolve()),
        },
        "signals": {
            "goldshot_auto_applied_rows": goldshot_auto,
            "goldshot_total_changed_rows": goldshot_total,
            "goldshot_market_model_share": goldshot_market_model_share,
            "women_runtime_active_rows": women_runtime_active,
            "baseline_market_backed_high_priority_rows": baseline_opposed,
        },
        "reasons": reasons,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final recommendation written to: {args.output}")


if __name__ == "__main__":
    main()
