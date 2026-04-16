from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.gold.overlay import apply_submission_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply submission-only gold overlay to a prediction csv.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--gender", default="M", choices=["M", "W"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    adjusted, audit, summary = apply_submission_overlay(frame, gender=args.gender, season=args.season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(args.output, index=False)

    audit_path = args.audit_output or args.output.with_name(f"{args.output.stem}_audit.csv")
    summary_path = args.summary_output or args.output.with_name(f"{args.output.stem}_summary.json")
    audit.to_csv(audit_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
