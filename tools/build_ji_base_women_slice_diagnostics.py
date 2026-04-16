from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from hc.ji_base import build_ji_dataset, build_working_ji_base_config, run_gender_replay

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

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
    config = build_working_ji_base_config("W")
    config.feature_profile = snapshot["feature_profile"]
    config.alpha_profile = snapshot["alpha_profile"]
    config.calibration_mode = snapshot["calibration_mode"]
    config.women_quality_profile = snapshot["women_quality_profile_w"]

    replay = run_gender_replay(config)
    dataset = build_ji_dataset(config)
    keep_columns = ["Season", "DayNum", "T1", "T2", "Label", *[column for column in KEY_FEATURES if column in dataset.columns]]
    merged = replay["predictions"].merge(dataset[keep_columns], on=["Season", "T1", "T2", "Label"], how="left")
    merged["raw_brier"] = (pd.to_numeric(merged["raw_prob"], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")) ** 2
    merged["calibrated_brier"] = (
        pd.to_numeric(merged["calibrated_prob"], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")
    ) ** 2
    merged["absolute_error"] = (pd.to_numeric(merged["calibrated_prob"], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")).abs()
    merged["seed_gap_abs"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce").abs()
    merged["favorite_is_t1"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce") <= 0
    merged["favorite_won"] = np.where(merged["favorite_is_t1"], merged["Label"] == 1, merged["Label"] == 0)
    merged["seed_gap_bucket"] = merged["seed_gap_abs"].map(_seed_gap_bucket)
    merged["upset_bucket"] = [
        _upset_bucket(seed_gap, favorite_won)
        for seed_gap, favorite_won in zip(merged["seed_gap_abs"], merged["favorite_won"], strict=False)
    ]
    latest_season = int(pd.to_numeric(merged["Season"], errors="coerce").max())
    recent_cutoff = latest_season - config.recent_window + 1
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
    high_error_cutoff = float(subset["absolute_error"].quantile(0.8))
    high_error = subset.loc[subset["absolute_error"] >= high_error_cutoff].copy()
    rows: list[dict[str, Any]] = []
    for feature in KEY_FEATURES:
        if feature not in subset.columns:
            continue
        feature_series = pd.to_numeric(subset[feature], errors="coerce")
        high_error_series = pd.to_numeric(high_error[feature], errors="coerce")
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "rows": int(len(subset)),
                "calibrated_brier": float(subset["calibrated_brier"].mean()),
                "avg_calibrated_prob": float(pd.to_numeric(subset["calibrated_prob"], errors="coerce").mean()),
                "empirical_win_rate": float(pd.to_numeric(subset["Label"], errors="coerce").mean()),
                "feature": feature,
                "feature_mean": float(feature_series.mean()),
                "feature_abs_mean": float(feature_series.abs().mean()),
                "high_error_feature_mean": float(high_error_series.mean()),
                "high_error_feature_abs_mean": float(high_error_series.abs().mean()),
                "error_corr": _corr(feature_series, subset["absolute_error"]),
                "brier_corr": _corr(feature_series, subset["calibrated_brier"]),
            }
        )
    return rows


def _write_markdown(summary: pd.DataFrame) -> None:
    lines = [
        "# JI_base Women Slice Diagnostics",
        "",
        "聚焦 women 冻结基线最弱的 slice，按关键特征拆误差相关性和高误差样本均值。",
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
                f"- Calibrated Brier: `{header['calibrated_brier']:.9f}`",
                f"- Avg calibrated prob: `{header['avg_calibrated_prob']:.6f}`",
                f"- Empirical win rate: `{header['empirical_win_rate']:.6f}`",
                "",
                "| Feature | Mean | High-error Mean | Error Corr | Brier Corr |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        ranked = subset.reindex(subset["error_corr"].abs().sort_values(ascending=False).index)
        for row in ranked.to_dict(orient="records"):
            error_corr = "" if row["error_corr"] is None else f"{row['error_corr']:.6f}"
            brier_corr = "" if row["brier_corr"] is None else f"{row['brier_corr']:.6f}"
            lines.append(
                f"| {row['feature']} | {row['feature_mean']:.6f} | {row['high_error_feature_mean']:.6f} | {error_corr} | {brier_corr} |"
            )
        lines.append("")

    (DOCS / "JI_BASE_WOMEN_SLICE_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    frame = _build_women_frame()
    rows: list[dict[str, Any]] = []
    for slice_type, slice_value in TARGET_SLICES:
        rows.extend(_slice_summary(frame, slice_type=slice_type, slice_value=slice_value))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "ji_base_women_slice_diagnostics.csv", index=False)
    (RESULTS / "ji_base_women_slice_diagnostics.json").write_text(
        json.dumps({"target_slices": TARGET_SLICES, "rows": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary)


if __name__ == "__main__":
    main()
