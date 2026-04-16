from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from hc.ji_base import JIBaseConfig, build_ji_dataset, run_gender_replay

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


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


def _favorite_seed_bucket(favorite_seed: float) -> str:
    seed = float(favorite_seed)
    if seed <= 2.0:
        return "seed_1_2"
    if seed <= 4.0:
        return "seed_3_4"
    if seed <= 8.0:
        return "seed_5_8"
    if seed <= 12.0:
        return "seed_9_12"
    return "seed_13_16"


def _upset_bucket(seed_gap: float, favorite_won: bool) -> str:
    if abs(float(seed_gap)) <= 1.0:
        return "tossup"
    return "favorite_win_gap2plus" if bool(favorite_won) else "upset_gap2plus"


def _build_frozen_config(gender: str, snapshot: dict[str, Any]) -> JIBaseConfig:
    return JIBaseConfig(
        gender=gender,  # type: ignore[arg-type]
        model_family=snapshot["base_model_profile"],
        calibration_mode=snapshot["calibration_mode"],
        feature_profile=snapshot["feature_profile"],
        alpha_profile=snapshot["alpha_profile"],
        women_quality_profile=snapshot["women_quality_profile_w"] if gender == "W" else snapshot["women_quality_profile_m"],
    )


def _prepare_slice_frame(gender: str, snapshot: dict[str, Any]) -> pd.DataFrame:
    config = _build_frozen_config(gender, snapshot)
    replay = run_gender_replay(config)
    dataset = build_ji_dataset(config)
    merged = replay["predictions"].merge(
        dataset[
            [
                "Season",
                "DayNum",
                "T1",
                "T2",
                "Delta_Seed",
                "T1_SeedNum",
                "T2_SeedNum",
            ]
        ],
        on=["Season", "T1", "T2"],
        how="left",
    )
    merged["gender"] = gender
    merged["raw_brier"] = (merged["raw_prob"] - merged["Label"]) ** 2
    merged["calibrated_brier"] = (merged["calibrated_prob"] - merged["Label"]) ** 2
    merged["seed_gap_abs"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce").abs()
    merged["favorite_seed"] = merged[["T1_SeedNum", "T2_SeedNum"]].min(axis=1)
    merged["underdog_seed"] = merged[["T1_SeedNum", "T2_SeedNum"]].max(axis=1)
    merged["favorite_is_t1"] = pd.to_numeric(merged["T1_SeedNum"], errors="coerce") <= pd.to_numeric(merged["T2_SeedNum"], errors="coerce")
    merged["favorite_won"] = np.where(merged["favorite_is_t1"], merged["Label"] == 1, merged["Label"] == 0)
    merged["seed_gap_bucket"] = merged["seed_gap_abs"].map(_seed_gap_bucket)
    merged["favorite_seed_bucket"] = merged["favorite_seed"].map(_favorite_seed_bucket)
    merged["upset_bucket"] = [
        _upset_bucket(seed_gap, favorite_won)
        for seed_gap, favorite_won in zip(merged["seed_gap_abs"], merged["favorite_won"], strict=False)
    ]
    latest_season = int(merged["Season"].max())
    recent_cutoff = latest_season - config.recent_window + 1
    merged["period_bucket"] = np.where(
        merged["Season"] == latest_season,
        "latest",
        np.where(merged["Season"] >= recent_cutoff, "recent", "historical"),
    )
    return merged


def _aggregate_slice(frame: pd.DataFrame, *, slice_type: str, slice_value: str, gender: str) -> dict[str, Any]:
    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "gender": gender,
        "rows": int(len(frame)),
        "raw_brier": float(frame["raw_brier"].mean()),
        "calibrated_brier": float(frame["calibrated_brier"].mean()),
        "favorite_win_rate": float(frame["favorite_won"].mean()) if "favorite_won" in frame.columns else None,
        "avg_seed_gap_abs": float(frame["seed_gap_abs"].mean()) if "seed_gap_abs" in frame.columns else None,
    }


def _build_slice_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gender, group in frame.groupby("gender", sort=True):
        rows.append(_aggregate_slice(group, slice_type="overall", slice_value="all", gender=str(gender)))
        for column in ("period_bucket", "seed_gap_bucket", "favorite_seed_bucket", "upset_bucket"):
            for value, slice_frame in group.groupby(column, sort=True):
                rows.append(_aggregate_slice(slice_frame, slice_type=column, slice_value=str(value), gender=str(gender)))

    rows.append(_aggregate_slice(frame, slice_type="overall", slice_value="all", gender="ALL"))
    for column in ("period_bucket", "seed_gap_bucket", "favorite_seed_bucket", "upset_bucket"):
        for value, slice_frame in frame.groupby(column, sort=True):
            rows.append(_aggregate_slice(slice_frame, slice_type=column, slice_value=str(value), gender="ALL"))
    return rows


def _write_markdown(snapshot: dict[str, Any], slices: pd.DataFrame) -> None:
    latest = slices.loc[(slices["slice_type"] == "period_bucket") & (slices["slice_value"] == "latest")]
    worst = slices.loc[(slices["gender"] != "ALL") & (slices["rows"] >= 100)].sort_values("calibrated_brier", ascending=False).head(8)

    lines = [
        "# JI_base Benchmark Slices",
        "",
        f"- Frozen candidate: `{snapshot['working_baseline_candidate']}`",
        f"- Model: `{snapshot['base_model_profile']}`",
        f"- Feature profile: `{snapshot['feature_profile']}`",
        f"- Alpha profile: `{snapshot['alpha_profile']}`",
        f"- Women quality profile: `{snapshot['women_quality_profile_w']}`",
        "",
        "## Latest / Recent",
        "",
        "| Gender | Slice | Rows | Calibrated Brier |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in latest.sort_values(["gender"]).to_dict(orient="records"):
        lines.append(f"| {row['gender']} | {row['slice_value']} | {row['rows']} | {row['calibrated_brier']:.9f} |")

    lines.extend(
        [
            "",
            "## Worst Slices",
            "",
            "| Gender | Slice Type | Slice | Rows | Calibrated Brier | Avg Seed Gap |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in worst.to_dict(orient="records"):
        lines.append(
            f"| {row['gender']} | {row['slice_type']} | {row['slice_value']} | {row['rows']} | {row['calibrated_brier']:.9f} | {row['avg_seed_gap_abs']:.3f} |"
        )

    (DOCS / "JI_BASE_BENCHMARK_SLICES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    men = _prepare_slice_frame("M", snapshot)
    women = _prepare_slice_frame("W", snapshot)
    combined = pd.concat([men, women], ignore_index=True)
    slice_rows = pd.DataFrame(_build_slice_rows(combined))

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS / "ji_base_benchmark_slice_predictions.csv", index=False)
    slice_rows.to_csv(RESULTS / "ji_base_benchmark_slices.csv", index=False)
    (RESULTS / "ji_base_benchmark_slices.json").write_text(
        json.dumps(
            {
                "snapshot": snapshot,
                "slices": slice_rows.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(snapshot, slice_rows)


if __name__ == "__main__":
    main()
