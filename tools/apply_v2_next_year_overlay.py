from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.v2.overlay import apply_current_year_overlay


def _parse_submission_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["ID"].str.split("_", expand=True)
    parsed = frame.copy()
    parsed["Season"] = pd.to_numeric(parts[0], errors="coerce")
    parsed["T1"] = pd.to_numeric(parts[1], errors="coerce")
    parsed["T2"] = pd.to_numeric(parts[2], errors="coerce")
    return parsed.dropna(subset=["Season", "T1", "T2", "Pred"]).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply current-year-only v2 overlay adjustments to a submission CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Submission-style CSV with `ID,Pred`.")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the overlaid submission CSV.")
    parser.add_argument("--season", type=int, required=True, help="Target tournament season for current-year overlays.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "v2_next_year_overlay_summary.json",
        help="JSON file describing overlay coverage and magnitude.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "results" / "v2_next_year_overlay_audit.csv",
        help="CSV file with per-row overlay audit details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = pd.read_csv(args.input)
    parsed = _parse_submission_ids(submission)

    outputs: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}
    for gender, mask in {
        "M": parsed["T1"] < 2000,
        "W": parsed["T1"] >= 3000,
    }.items():
        subset = parsed.loc[mask].copy()
        if subset.empty:
            continue
        adjusted, audit, summary = apply_current_year_overlay(subset, gender=gender, season=args.season)
        outputs.append(adjusted)
        audits.append(audit.assign(gender=gender))
        summaries[gender] = asdict(summary)

    if outputs:
        adjusted_frame = pd.concat(outputs, ignore_index=True)
        final = submission.drop(columns=["Pred"]).merge(adjusted_frame, on="ID", how="left")
        final["Pred"] = pd.to_numeric(final["Pred"], errors="coerce").fillna(0.5)
    else:
        final = submission.copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    if audits:
        audit_frame = pd.concat(audits, ignore_index=True)
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_frame.to_csv(args.audit_output, index=False)
    overlay_mean_abs_delta_m = summaries.get("M", {}).get("mean_abs_delta")
    overlay_mean_abs_delta_w = summaries.get("W", {}).get("mean_abs_delta")
    overlay_guardrail_passed = bool(
        summaries
        and all(bool(payload.get("guardrail_passed", False)) for payload in summaries.values())
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(
            {
                "season": args.season,
                "enabled": bool(outputs),
                "genders": summaries,
                "audit_path": str(args.audit_output),
                "overlay_guardrail_passed": overlay_guardrail_passed,
                "overlay_mean_abs_delta_m": overlay_mean_abs_delta_m,
                "overlay_mean_abs_delta_w": overlay_mean_abs_delta_w,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
