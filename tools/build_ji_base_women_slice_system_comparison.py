from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from hc.ji_base import JIBaseConfig, build_ji_dataset

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
JI_SLICE_CACHE = RESULTS / "ji_base_benchmark_slice_predictions.csv"
GOLD_SLICE_CACHE = RESULTS / "gold_recover_benchmark_slice_predictions.csv"

KEY_FEATURES = [
    "Delta_Seed",
    "Delta_Elo",
    "Delta_Quality",
    "QualityWins_diff",
    "OpponentQualityTournamentRank_diff",
    "AvgBlkDiff_diff",
    "Seed_x_Quality",
    "WomenCompositeQuality_diff",
]
TARGET_SLICES = [
    ("upset_bucket", "upset_gap2plus"),
    ("upset_bucket", "tossup"),
    ("seed_gap_bucket", "gap_0_1"),
    ("period_bucket", "recent"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_gap_bucket(seed_gap: float) -> str:
    value = float(abs(seed_gap))
    if value <= 1.0:
        return "gap_0_1"
    if value <= 4.0:
        return "gap_2_4"
    if value <= 8.0:
        return "gap_5_8"
    return "gap_9_plus"


def _upset_bucket(seed_gap: float, favorite_won: bool) -> str:
    if abs(float(seed_gap)) <= 1.0:
        return "tossup"
    return "favorite_win_gap2plus" if bool(favorite_won) else "upset_gap2plus"


def _build_women_frame() -> pd.DataFrame:
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    ji_config = JIBaseConfig(
        gender="W",
        model_family=snapshot["base_model_profile"],
        calibration_mode=snapshot["calibration_mode"],
        feature_profile=snapshot["feature_profile"],
        alpha_profile=snapshot["alpha_profile"],
        women_quality_profile=snapshot["women_quality_profile_w"],
    )

    ji_replay = pd.read_csv(JI_SLICE_CACHE)
    ji_replay = ji_replay.loc[ji_replay["gender"] == "W"].copy()
    ji_replay = ji_replay.rename(columns={"raw_prob": "ji_raw_prob", "calibrated_prob": "ji_calibrated_prob"})

    gold_replay = pd.read_csv(GOLD_SLICE_CACHE)
    gold_replay = gold_replay.loc[gold_replay["gender"] == "W"].copy()
    gold_replay = gold_replay.rename(columns={"raw_prob": "gold_raw_prob", "calibrated_prob": "gold_calibrated_prob"})

    dataset = build_ji_dataset(ji_config)
    existing_columns = set(ji_replay.columns)
    extra_feature_columns = [column for column in KEY_FEATURES if column in dataset.columns and column not in existing_columns]
    keep_columns = ["Season", "T1", "T2", "Label", *extra_feature_columns]
    merged = (
        ji_replay.merge(
            gold_replay[["Season", "T1", "T2", "Label", "gold_raw_prob", "gold_calibrated_prob"]],
            on=["Season", "T1", "T2", "Label"],
            how="inner",
        ).merge(dataset[keep_columns], on=["Season", "T1", "T2", "Label"], how="left")
    )

    merged["ji_calibrated_brier"] = (
        pd.to_numeric(merged["ji_calibrated_prob"], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")
    ) ** 2
    merged["gold_calibrated_brier"] = (
        pd.to_numeric(merged["gold_calibrated_prob"], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")
    ) ** 2
    merged["ji_minus_gold_brier"] = merged["ji_calibrated_brier"] - merged["gold_calibrated_brier"]
    merged["ji_better"] = merged["ji_minus_gold_brier"] < 0
    merged["seed_gap_abs"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce").abs()
    merged["favorite_is_t1"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce") <= 0
    merged["favorite_won"] = np.where(merged["favorite_is_t1"], merged["Label"] == 1, merged["Label"] == 0)
    merged["seed_gap_bucket"] = merged["seed_gap_abs"].map(_seed_gap_bucket)
    merged["upset_bucket"] = [
        _upset_bucket(seed_gap, favorite_won)
        for seed_gap, favorite_won in zip(merged["seed_gap_abs"], merged["favorite_won"], strict=False)
    ]
    latest_season = int(pd.to_numeric(merged["Season"], errors="coerce").max())
    recent_cutoff = latest_season - ji_config.recent_window + 1
    merged["period_bucket"] = np.where(
        merged["Season"] == latest_season,
        "latest",
        np.where(merged["Season"] >= recent_cutoff, "recent", "historical"),
    )
    return merged


def _corr(series: pd.Series, target: pd.Series) -> float | None:
    clean = pd.concat([pd.to_numeric(series, errors="coerce"), pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    if len(clean) < 10 or clean.iloc[:, 0].nunique() < 2:
        return None
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1]))


def _slice_summary(frame: pd.DataFrame, *, slice_type: str, slice_value: str) -> list[dict[str, Any]]:
    subset = frame.loc[frame[slice_type] == slice_value].copy()
    if subset.empty:
        return []
    ji_worse = subset.loc[subset["ji_minus_gold_brier"] > 0].copy()
    if ji_worse.empty:
        ji_worse = subset.loc[subset["ji_minus_gold_brier"] >= float(subset["ji_minus_gold_brier"].quantile(0.8))].copy()
    rows: list[dict[str, Any]] = []
    for feature in KEY_FEATURES:
        if feature not in subset.columns:
            continue
        feature_series = pd.to_numeric(subset[feature], errors="coerce")
        worse_series = pd.to_numeric(ji_worse[feature], errors="coerce")
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "rows": int(len(subset)),
                "ji_calibrated_brier": float(subset["ji_calibrated_brier"].mean()),
                "gold_calibrated_brier": float(subset["gold_calibrated_brier"].mean()),
                "ji_minus_gold_brier_mean": float(subset["ji_minus_gold_brier"].mean()),
                "ji_worse_rate": float((subset["ji_minus_gold_brier"] > 0).mean()),
                "feature": feature,
                "feature_mean": float(feature_series.mean()),
                "feature_abs_mean": float(feature_series.abs().mean()),
                "ji_worse_feature_mean": float(worse_series.mean()),
                "ji_worse_feature_abs_mean": float(worse_series.abs().mean()),
                "delta_corr": _corr(feature_series, subset["ji_minus_gold_brier"]),
            }
        )
    return rows


def _write_markdown(summary: pd.DataFrame) -> None:
    lines = [
        "# JI_base Women Slice System Comparison",
        "",
        "Compare the same women games under `JI_base` frozen baseline and `gold_recover_proxy`, then locate the feature regions where `JI_base` loses on calibrated Brier.",
        "",
    ]
    for slice_type, slice_value in TARGET_SLICES:
        subset = summary.loc[(summary["slice_type"] == slice_type) & (summary["slice_value"] == slice_value)].copy()
        if subset.empty:
            continue
        header = subset.iloc[0]
        lines.extend(
            [
                f"## {slice_type} = {slice_value}",
                "",
                f"- Rows: `{int(header['rows'])}`",
                f"- `ji_calibrated_brier`: `{header['ji_calibrated_brier']:.9f}`",
                f"- `gold_calibrated_brier`: `{header['gold_calibrated_brier']:.9f}`",
                f"- `ji_minus_gold_brier_mean`: `{header['ji_minus_gold_brier_mean']:.9f}`",
                f"- `ji_worse_rate`: `{header['ji_worse_rate']:.6f}`",
                "",
                "| Feature | Mean | JI-worse Mean | Delta Corr |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        ranked = subset.reindex(subset["delta_corr"].abs().sort_values(ascending=False).index)
        for row in ranked.to_dict(orient="records"):
            delta_corr = "" if row["delta_corr"] is None else f"{row['delta_corr']:.6f}"
            lines.append(
                f"| {row['feature']} | {row['feature_mean']:.6f} | {row['ji_worse_feature_mean']:.6f} | {delta_corr} |"
            )
        lines.append("")

    (DOCS / "JI_BASE_WOMEN_SLICE_SYSTEM_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    frame = _build_women_frame()
    frame.to_csv(RESULTS / "ji_base_women_slice_system_comparison_predictions.csv", index=False)
    rows: list[dict[str, Any]] = []
    for slice_type, slice_value in TARGET_SLICES:
        rows.extend(_slice_summary(frame, slice_type=slice_type, slice_value=slice_value))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "ji_base_women_slice_system_comparison.csv", index=False)
    (RESULTS / "ji_base_women_slice_system_comparison.json").write_text(
        json.dumps({"target_slices": TARGET_SLICES, "rows": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary)


if __name__ == "__main__":
    main()
