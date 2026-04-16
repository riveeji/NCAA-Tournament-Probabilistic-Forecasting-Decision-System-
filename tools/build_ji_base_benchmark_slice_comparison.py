from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from hc.gold import GoldConfig, run_gender_replay as run_gold_gender_replay
from hc.gold.data import load_gold_team_features
from hc.ji_base import JIBaseConfig, build_ji_dataset, run_gender_replay as run_ji_gender_replay

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
JI_SLICE_CACHE = RESULTS / "ji_base_benchmark_slice_predictions.csv"
GOLD_SLICE_CACHE = RESULTS / "gold_recover_benchmark_slice_predictions.csv"


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


def _finalize_slice_frame(
    frame: pd.DataFrame,
    *,
    gender: str,
    system: str,
    raw_prob_col: str,
    calibrated_prob_col: str,
) -> pd.DataFrame:
    merged = frame.copy()
    merged["gender"] = gender
    merged["system"] = system
    merged["raw_brier"] = (pd.to_numeric(merged[raw_prob_col], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")) ** 2
    merged["calibrated_brier"] = (
        pd.to_numeric(merged[calibrated_prob_col], errors="coerce") - pd.to_numeric(merged["Label"], errors="coerce")
    ) ** 2
    merged["seed_gap_abs"] = pd.to_numeric(merged["Delta_Seed"], errors="coerce").abs()
    merged["favorite_seed"] = merged[["T1_SeedNum", "T2_SeedNum"]].min(axis=1)
    merged["underdog_seed"] = merged[["T1_SeedNum", "T2_SeedNum"]].max(axis=1)
    merged["favorite_is_t1"] = pd.to_numeric(merged["T1_SeedNum"], errors="coerce") <= pd.to_numeric(
        merged["T2_SeedNum"], errors="coerce"
    )
    merged["favorite_won"] = np.where(merged["favorite_is_t1"], merged["Label"] == 1, merged["Label"] == 0)
    merged["seed_gap_bucket"] = merged["seed_gap_abs"].map(_seed_gap_bucket)
    merged["favorite_seed_bucket"] = merged["favorite_seed"].map(_favorite_seed_bucket)
    merged["upset_bucket"] = [
        _upset_bucket(seed_gap, favorite_won)
        for seed_gap, favorite_won in zip(merged["seed_gap_abs"], merged["favorite_won"], strict=False)
    ]
    latest_season = int(pd.to_numeric(merged["Season"], errors="coerce").max())
    recent_cutoff = latest_season - 5 + 1
    merged["period_bucket"] = np.where(
        merged["Season"] == latest_season,
        "latest",
        np.where(merged["Season"] >= recent_cutoff, "recent", "historical"),
    )
    return merged


def _build_frozen_ji_config(gender: str, snapshot: dict[str, Any]) -> JIBaseConfig:
    return JIBaseConfig(
        gender=gender,  # type: ignore[arg-type]
        model_family=snapshot["base_model_profile"],
        calibration_mode=snapshot["calibration_mode"],
        feature_profile=snapshot["feature_profile"],
        alpha_profile=snapshot["alpha_profile"],
        women_quality_profile=snapshot["women_quality_profile_w"] if gender == "W" else snapshot["women_quality_profile_m"],
    )


def _prepare_ji_base_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    if JI_SLICE_CACHE.exists():
        cached = pd.read_csv(JI_SLICE_CACHE)
        if "system" not in cached.columns:
            cached["system"] = "ji_base_frozen"
        return cached

    frames: list[pd.DataFrame] = []
    for gender in ("M", "W"):
        config = _build_frozen_ji_config(gender, snapshot)
        replay = run_ji_gender_replay(config)
        dataset = build_ji_dataset(config)
        merged = replay["predictions"].merge(
            dataset[["Season", "DayNum", "T1", "T2", "Delta_Seed", "T1_SeedNum", "T2_SeedNum"]],
            on=["Season", "T1", "T2"],
            how="left",
        )
        frames.append(
            _finalize_slice_frame(
                merged,
                gender=gender,
                system="ji_base_frozen",
                raw_prob_col="raw_prob",
                calibrated_prob_col="calibrated_prob",
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(JI_SLICE_CACHE, index=False)
    return combined


def _prepare_gold_frame() -> pd.DataFrame:
    if GOLD_SLICE_CACHE.exists():
        return pd.read_csv(GOLD_SLICE_CACHE)

    frames: list[pd.DataFrame] = []
    for gender in ("M", "W"):
        config = GoldConfig(
            gender=gender,  # type: ignore[arg-type]
            model_family="gold_linear",
            calibration_mode="none",
            rating_source_profile="current_default",
        )
        replay = run_gold_gender_replay(config)
        team_features = load_gold_team_features(config)[["Season", "TeamID", "SeedNum"]].copy()
        t1 = team_features.rename(columns={"TeamID": "T1", "SeedNum": "T1_SeedNum"})
        t2 = team_features.rename(columns={"TeamID": "T2", "SeedNum": "T2_SeedNum"})
        merged = (
            replay["predictions"]
            .merge(t1, on=["Season", "T1"], how="left")
            .merge(t2, on=["Season", "T2"], how="left")
        )
        merged["Delta_Seed"] = pd.to_numeric(merged["T1_SeedNum"], errors="coerce") - pd.to_numeric(
            merged["T2_SeedNum"], errors="coerce"
        )
        merged["raw_prob"] = pd.to_numeric(merged["Prob"], errors="coerce")
        merged["calibrated_prob"] = pd.to_numeric(merged["Prob"], errors="coerce")
        frames.append(
            _finalize_slice_frame(
                merged,
                gender=gender,
                system="gold_recover_proxy",
                raw_prob_col="raw_prob",
                calibrated_prob_col="calibrated_prob",
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(GOLD_SLICE_CACHE, index=False)
    return combined


def _aggregate_slice(frame: pd.DataFrame, *, system: str, slice_type: str, slice_value: str, gender: str) -> dict[str, Any]:
    return {
        "system": system,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "gender": gender,
        "rows": int(len(frame)),
        "raw_brier": float(frame["raw_brier"].mean()),
        "calibrated_brier": float(frame["calibrated_brier"].mean()),
        "favorite_win_rate": float(frame["favorite_won"].mean()),
        "avg_seed_gap_abs": float(frame["seed_gap_abs"].mean()),
    }


def _build_slice_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system, system_frame in frame.groupby("system", sort=True):
        for gender, group in system_frame.groupby("gender", sort=True):
            rows.append(_aggregate_slice(group, system=system, slice_type="overall", slice_value="all", gender=str(gender)))
            for column in ("period_bucket", "seed_gap_bucket", "favorite_seed_bucket", "upset_bucket"):
                for value, slice_frame in group.groupby(column, sort=True):
                    rows.append(
                        _aggregate_slice(
                            slice_frame,
                            system=system,
                            slice_type=column,
                            slice_value=str(value),
                            gender=str(gender),
                        )
                    )

        rows.append(_aggregate_slice(system_frame, system=system, slice_type="overall", slice_value="all", gender="ALL"))
        for column in ("period_bucket", "seed_gap_bucket", "favorite_seed_bucket", "upset_bucket"):
            for value, slice_frame in system_frame.groupby(column, sort=True):
                rows.append(
                    _aggregate_slice(
                        slice_frame,
                        system=system,
                        slice_type=column,
                        slice_value=str(value),
                        gender="ALL",
                    )
                )
    return rows


def _build_comparison_rows(slices: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        slices.pivot_table(
            index=["slice_type", "slice_value", "gender"],
            columns="system",
            values=["rows", "calibrated_brier"],
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(part) for part in column if str(part) != ""]).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    pivot["rows_min"] = pivot[["rows_gold_recover_proxy", "rows_ji_base_frozen"]].min(axis=1)
    pivot["ji_minus_gold_calibrated_brier"] = (
        pd.to_numeric(pivot["calibrated_brier_ji_base_frozen"], errors="coerce")
        - pd.to_numeric(pivot["calibrated_brier_gold_recover_proxy"], errors="coerce")
    )
    pivot["winner"] = np.where(
        pivot["ji_minus_gold_calibrated_brier"] < 0,
        "ji_base_frozen",
        np.where(pivot["ji_minus_gold_calibrated_brier"] > 0, "gold_recover_proxy", "tie"),
    )
    return pivot.sort_values(["slice_type", "gender", "slice_value"]).reset_index(drop=True)


def _write_markdown(
    snapshot: dict[str, Any],
    comparison: pd.DataFrame,
    *,
    gold_official_lb: float | None,
    old_hc_replay: float | None,
) -> None:
    meaningful = comparison.loc[comparison["rows_min"] >= 100].copy()
    ji_best = meaningful.sort_values("ji_minus_gold_calibrated_brier").head(8)
    ji_worst = meaningful.sort_values("ji_minus_gold_calibrated_brier", ascending=False).head(8)
    latest = comparison.loc[(comparison["slice_type"] == "period_bucket") & (comparison["slice_value"] == "latest")]

    lines = [
        "# JI_base Benchmark Slice Comparison",
        "",
        "## Systems",
        "",
        f"- `ji_base_frozen`: frozen baseline, official LB `{snapshot['official_lb_best_score']}`",
        f"- `gold_recover_proxy`: `gold_linear@none[current_default]` replay proxy; official LB reference for market submission is `{gold_official_lb}`",
        f"- `old_hc`: replay-only reference `{old_hc_replay}`; omitted from slice comparison because there is no aligned prediction-level artifact",
        "",
        "## Latest Slice Head-to-Head",
        "",
        "| Gender | JI_base | Gold proxy | Delta (JI - Gold) | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in latest.sort_values(["gender"]).to_dict(orient="records"):
        lines.append(
            f"| {row['gender']} | {row['calibrated_brier_ji_base_frozen']:.9f} | {row['calibrated_brier_gold_recover_proxy']:.9f} | {row['ji_minus_gold_calibrated_brier']:.9f} | {row['winner']} |"
        )

    lines.extend(
        [
            "",
            "## Strongest JI_base Slices",
            "",
            "| Gender | Slice Type | Slice | JI_base | Gold proxy | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in ji_best.to_dict(orient="records"):
        lines.append(
            f"| {row['gender']} | {row['slice_type']} | {row['slice_value']} | {row['calibrated_brier_ji_base_frozen']:.9f} | {row['calibrated_brier_gold_recover_proxy']:.9f} | {row['ji_minus_gold_calibrated_brier']:.9f} |"
        )

    lines.extend(
        [
            "",
            "## Weakest JI_base Slices",
            "",
            "| Gender | Slice Type | Slice | JI_base | Gold proxy | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in ji_worst.to_dict(orient="records"):
        lines.append(
            f"| {row['gender']} | {row['slice_type']} | {row['slice_value']} | {row['calibrated_brier_ji_base_frozen']:.9f} | {row['calibrated_brier_gold_recover_proxy']:.9f} | {row['ji_minus_gold_calibrated_brier']:.9f} |"
        )

    (DOCS / "JI_BASE_BENCHMARK_SLICE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    benchmark_report = _load_json(RESULTS / "ji_base_benchmark_report.json")
    official_lb = pd.read_csv(RESULTS / "official_lb_log.csv")
    gold_official_lb_series = pd.to_numeric(
        official_lb.loc[official_lb["submission_profile"] == "gold_recover_market", "official_lb"],
        errors="coerce",
    )
    gold_official_lb = float(gold_official_lb_series.min()) if gold_official_lb_series.notna().any() else None
    old_hc_replay = None
    for system in benchmark_report.get("systems", []):
        if system.get("system") == "old_hc":
            old_hc_replay = float(system["replay_total_cv_brier_calibrated"])
            break

    ji_base = _prepare_ji_base_frame(snapshot)
    gold = _prepare_gold_frame()
    combined = pd.concat([ji_base, gold], ignore_index=True)
    slice_rows = pd.DataFrame(_build_slice_rows(combined))
    comparison = _build_comparison_rows(slice_rows)

    slice_rows.to_csv(RESULTS / "ji_base_system_benchmark_slices.csv", index=False)
    comparison.to_csv(RESULTS / "ji_base_system_benchmark_slice_comparison.csv", index=False)
    (RESULTS / "ji_base_system_benchmark_slice_comparison.json").write_text(
        json.dumps(
            {
                "snapshot": snapshot,
                "systems": [
                    {"system": "ji_base_frozen", "official_lb": snapshot["official_lb_best_score"]},
                    {"system": "gold_recover_proxy", "official_lb_reference": gold_official_lb},
                    {"system": "old_hc", "replay_reference": old_hc_replay, "slice_comparison": "not_available"},
                ],
                "slice_rows": slice_rows.to_dict(orient="records"),
                "comparison_rows": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(snapshot, comparison, gold_official_lb=gold_official_lb, old_hc_replay=old_hc_replay)


if __name__ == "__main__":
    main()
