from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check a Kaggle submission file against the official sample template."
    )
    parser.add_argument(
        "--submission",
        default="submission_stage2_single_final.csv",
        help="Submission CSV to validate.",
    )
    parser.add_argument(
        "--sample",
        default=str(DATA_DIR / "SampleSubmissionStage2.csv"),
        help="Official sample submission CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def warn(messages: list[str], message: str) -> None:
    messages.append(message)


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    sample_path = Path(args.sample)

    if not submission_path.exists():
        fail(f"submission file not found: {submission_path}")
    if not sample_path.exists():
        fail(f"sample file not found: {sample_path}")

    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    expected_cols = ["ID", "Pred"]
    if sub.columns.tolist() != expected_cols:
        fail(f"submission columns must be exactly {expected_cols}, got {sub.columns.tolist()}")
    if sample.columns.tolist() != expected_cols:
        fail(f"sample columns must be exactly {expected_cols}, got {sample.columns.tolist()}")

    if len(sub) != len(sample):
        fail(f"row count mismatch: submission={len(sub)} sample={len(sample)}")

    if sub["ID"].duplicated().any():
        dupes = int(sub["ID"].duplicated().sum())
        fail(f"submission contains duplicate IDs: {dupes}")

    if sample["ID"].duplicated().any():
        fail("sample submission contains duplicate IDs")

    sub_ids = set(sub["ID"])
    sample_ids = set(sample["ID"])
    if sub_ids != sample_ids:
        missing = len(sample_ids - sub_ids)
        extra = len(sub_ids - sample_ids)
        fail(f"submission IDs do not match sample IDs: missing={missing}, extra={extra}")

    warnings: list[str] = []
    if not sub["ID"].equals(sample["ID"]):
        warn(warnings, "submission ID order differs from sample order")

    pred = pd.to_numeric(sub["Pred"], errors="coerce")
    if pred.isna().any():
        fail(f"Pred contains NaN/non-numeric values: {int(pred.isna().sum())}")

    pred_np = pred.to_numpy(dtype=float)
    if not np.isfinite(pred_np).all():
        fail("Pred contains non-finite values")
    if (pred_np < 0.0).any() or (pred_np > 1.0).any():
        fail("Pred contains values outside [0, 1]")

    near_zero = int((pred_np <= 0.01).sum())
    near_one = int((pred_np >= 0.99).sum())
    exact_half = int((np.isclose(pred_np, 0.5)).sum())
    unique_count = int(pd.Series(pred_np).nunique())
    pred_std = float(np.std(pred_np))
    pred_quantiles = {
        "q001": float(np.quantile(pred_np, 0.001)),
        "q01": float(np.quantile(pred_np, 0.01)),
        "q05": float(np.quantile(pred_np, 0.05)),
        "q25": float(np.quantile(pred_np, 0.25)),
        "q50": float(np.quantile(pred_np, 0.50)),
        "q75": float(np.quantile(pred_np, 0.75)),
        "q95": float(np.quantile(pred_np, 0.95)),
        "q99": float(np.quantile(pred_np, 0.99)),
        "q999": float(np.quantile(pred_np, 0.999)),
    }
    histogram_edges = np.array([0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99, 1.0], dtype=float)
    histogram_counts, _ = np.histogram(pred_np, bins=histogram_edges)
    pred_histogram = [
        {
            "low": float(histogram_edges[idx]),
            "high": float(histogram_edges[idx + 1]),
            "count": int(histogram_counts[idx]),
        }
        for idx in range(len(histogram_counts))
    ]
    file_sha256 = hashlib.sha256(submission_path.read_bytes()).hexdigest()

    if pred_std < 0.03:
        warn(warnings, f"prediction std is low: {pred_std:.6f}")
    if unique_count < 100:
        warn(warnings, f"prediction has unusually few unique values: {unique_count}")
    if exact_half > len(sub) * 0.10:
        warn(warnings, f"more than 10% of rows are exactly 0.5: {exact_half}")

    season = pd.to_numeric(sub["ID"].str.split("_", expand=True)[0], errors="coerce")
    if season.isna().any():
        warn(warnings, "some IDs could not be parsed into season/team format")

    summary = {
        "submission": str(submission_path.resolve()),
        "sample": str(sample_path.resolve()),
        "rows": int(len(sub)),
        "pred_min": float(pred_np.min()),
        "pred_max": float(pred_np.max()),
        "pred_mean": float(pred_np.mean()),
        "pred_std": pred_std,
        "pred_quantiles": pred_quantiles,
        "pred_histogram": pred_histogram,
        "near_zero_count": near_zero,
        "near_one_count": near_one,
        "exact_half_count": exact_half,
        "unique_pred_count": unique_count,
        "sha256": file_sha256,
        "warnings": warnings,
        "status": "pass",
        "extreme_high_ids": sub.sort_values("Pred", ascending=False).head(20).to_dict("records"),
        "extreme_low_ids": sub.sort_values("Pred", ascending=True).head(20).to_dict("records"),
    }

    print("[PASS] submission sanity check passed")
    print(f"file:      {submission_path}")
    print(f"rows:      {len(sub)}")
    print(f"pred min:  {pred_np.min():.6f}")
    print(f"pred max:  {pred_np.max():.6f}")
    print(f"pred mean: {pred_np.mean():.6f}")
    print(f"pred std:  {pred_std:.6f}")
    print(f"sha256:    {file_sha256}")
    print(f"near 0:    {near_zero}")
    print(f"near 1:    {near_one}")
    print(f"exact 0.5: {exact_half}")
    print(f"unique:    {unique_count}")
    print("quantiles:")
    for key, value in pred_quantiles.items():
        print(f"  {key}: {value:.6f}")
    print("histogram:")
    for bucket in pred_histogram:
        print(f"  [{bucket['low']:.2f}, {bucket['high']:.2f}]: {bucket['count']}")
    if warnings:
        print("warnings:")
        for message in warnings:
            print(f"  - {message}")

    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary:   {output_path}")


if __name__ == "__main__":
    main()
