from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.predict import build_hc_prediction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two candidate submissions and emit a diff report.")
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--label-a", default="candidate_a")
    parser.add_argument("--label-b", default="candidate_b")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--details-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = pd.read_csv(args.candidate_a)
    b = pd.read_csv(args.candidate_b)
    merged = a.merge(b, on="ID", how="inner", suffixes=("_a", "_b"))
    merged["PredDiff"] = merged["Pred_b"] - merged["Pred_a"]
    merged["AbsPredDiff"] = merged["PredDiff"].abs()
    merged = merged.sort_values("AbsPredDiff", ascending=False).reset_index(drop=True)

    women = build_hc_prediction_frame("W", args.season, 32, False, True)
    host_ids: set[str] = set()
    if not women.empty and {"ID", "D_HostLikely", "IsRound1Or2"}.issubset(women.columns):
        host_mask = (
            pd.to_numeric(women["D_HostLikely"], errors="coerce").fillna(0.0).ge(1.0)
            & pd.to_numeric(women["IsRound1Or2"], errors="coerce").fillna(0.0).ge(1.0)
        )
        host_ids = set(women.loc[host_mask, "ID"].astype(str).tolist())

    merged["WomenEarlyHost"] = merged["ID"].astype(str).isin(host_ids)
    details = merged.head(50).copy()
    details.to_csv(args.details_output, index=False)

    women_host_top = merged.loc[merged["WomenEarlyHost"]].head(50)
    summary = {
        "candidate_a": str(Path(args.candidate_a).resolve()),
        "candidate_b": str(Path(args.candidate_b).resolve()),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "rows": int(len(merged)),
        "mean_abs_diff": float(merged["AbsPredDiff"].mean()),
        "max_abs_diff": float(merged["AbsPredDiff"].max()),
        "top50_path": str(Path(args.details_output).resolve()),
        "top10": details.head(10).to_dict("records"),
        "women_early_host_top10": women_host_top.head(10).to_dict("records"),
    }
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidate diff summary written to: {args.summary_output}")
    print(f"candidate diff details written to: {args.details_output}")


if __name__ == "__main__":
    main()
