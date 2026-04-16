from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.gold import GoldConfig, run_gender_replay

RESULTS = ROOT / "results"
GOLD_BASELINE_VARIANTS = [
    ("gold_linear", "none", "current_default"),
    ("gold_linear", "isotonic_gender", "current_default"),
    ("gold_linear", "none", "m_ap_removed_only"),
    ("gold_linear", "isotonic_gender", "m_ap_removed_only"),
    ("gold_linear", "none", "a_tier_default"),
    ("gold_linear", "isotonic_gender", "a_tier_default"),
    ("gold_harry_lr", "none", "current_default"),
    ("gold_harry_lr", "isotonic_gender", "current_default"),
    ("gold_harry_xgb_spread", "none", "current_default"),
    ("gold_harry_xgb_spread", "isotonic_gender", "current_default"),
    ("gold_xgb_spread_light", "none", "current_default"),
    ("gold_xgb_spread_light", "isotonic_gender", "current_default"),
]
GOLD_CONTROL_VARIANTS = [
    ("gold_min_lr", "none", "current_default"),
    ("gold_min_xgb_spread", "none", "current_default"),
    ("gold_tree_control", "none", "current_default"),
    ("gold_spread_control", "none", "current_default"),
]


def resolve_variant_plan(*, include_controls: bool) -> list[tuple[str, str, str]]:
    plan = list(GOLD_BASELINE_VARIANTS)
    if include_controls:
        plan.extend(GOLD_CONTROL_VARIANTS)
    return plan


def variant_label(model_family: str, calibration_mode: str, rating_source_profile: str) -> str:
    return f"{model_family}@{calibration_mode}[{rating_source_profile}]"


def _sort_by_existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame
    return frame.sort_values(existing)


def write_replay_outputs(
    *,
    output_dir: Path,
    summary_rows: list[dict],
    combined_rows: list[dict],
    by_season_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    write_detailed_outputs: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _sort_by_existing_columns(pd.DataFrame(summary_rows), ["model_family", "calibration_mode", "gender"])
    combined_df = _sort_by_existing_columns(pd.DataFrame(combined_rows), ["equal_gender_mean_brier", "equal_gender_latest_season_brier"])
    summary_df.to_csv(output_dir / "gold_replay_summary.csv", index=False)
    combined_df.to_csv(output_dir / "gold_replay_combined.csv", index=False)
    (output_dir / "gold_replay_summary.json").write_text(
        json.dumps({"summary_rows": summary_rows, "combined": combined_rows}, indent=2),
        encoding="utf-8",
    )
    if write_detailed_outputs:
        by_season_df.to_csv(output_dir / "gold_replay_by_season.csv", index=False)
        predictions_df.to_csv(output_dir / "gold_replay_predictions.csv", index=False)


def _combined_rows(men_women_lookup: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    keys = sorted({key for _, key in men_women_lookup})
    for key in keys:
        men = men_women_lookup.get(("M", key))
        women = men_women_lookup.get(("W", key))
        if not men or not women:
            continue
        rows.append(
            {
                "variant": key,
                "model_family": men["model_family"],
                "feature_profile": f"{men['feature_profile']}|{women['feature_profile']}",
                "rating_profile": f"{men['rating_profile']}|{women['rating_profile']}",
                "rating_source_profile": f"{men['rating_source_profile']}|{women['rating_source_profile']}",
                "calibration_mode": men["calibration_mode"],
                "selection_objective": men["selection_objective"],
                "equal_gender_mean_brier": float((men["mean_brier"] + women["mean_brier"]) / 2.0),
                "latest_season": int(max(men["latest_season"], women["latest_season"])),
                "equal_gender_latest_season_brier": float((men["latest_season_brier"] + women["latest_season_brier"]) / 2.0),
                "equal_gender_recent_window_brier": float((men["recent_window_brier"] + women["recent_window_brier"]) / 2.0),
                "men_mean_brier": float(men["mean_brier"]),
                "women_mean_brier": float(women["mean_brier"]),
                "men_latest_season_brier": float(men["latest_season_brier"]),
                "women_latest_season_brier": float(women["latest_season_brier"]),
                "men_recent_window_brier": float(men["recent_window_brier"]),
                "women_recent_window_brier": float(women["recent_window_brier"]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gold-model NCAA replay baselines.")
    parser.add_argument("--include-controls", action="store_true", help="Add tree/spread control families.")
    parser.add_argument("--write-detailed-outputs", action="store_true", help="Also write by-season and prediction detail files.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS, help="Directory for generated gold replay artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_plan = resolve_variant_plan(include_controls=args.include_controls)

    summary_rows: list[dict] = []
    by_season_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    men_women_lookup: dict[tuple[str, str], dict] = {}

    for gender in ("M", "W"):
        for model_family, calibration_mode, rating_source_profile in variant_plan:
            replay = run_gender_replay(
                GoldConfig(
                    gender=gender,
                    model_family=model_family,
                    calibration_mode=calibration_mode,
                    rating_source_profile=rating_source_profile,
                )
            )
            summary_rows.append(
                {
                    "gender": gender,
                    "gender_segment": replay["gender_segment"],
                    "model_family": replay["model_family"],
                    "feature_profile": replay["feature_profile"],
                    "rating_profile": replay["rating_profile"],
                    "rating_source_profile": replay["rating_source_profile"],
                    "calibration_mode": replay["calibration_mode"],
                    "selection_objective": replay["selection_objective"],
                    "mean_brier": replay["mean_brier"],
                    "latest_season": replay["latest_season"],
                    "latest_season_brier": replay["latest_season_brier"],
                    "recent_window_brier": replay["recent_window_brier"],
                    "brier_variance": replay["brier_variance"],
                }
            )
            men_women_lookup[(gender, variant_label(model_family, calibration_mode, rating_source_profile))] = replay
            by_season_frames.append(replay["by_season"])
            prediction_frames.append(replay["predictions"])

    write_replay_outputs(
        output_dir=args.output_dir,
        summary_rows=summary_rows,
        combined_rows=_combined_rows(men_women_lookup),
        by_season_df=pd.concat(by_season_frames, ignore_index=True),
        predictions_df=pd.concat(prediction_frames, ignore_index=True),
        write_detailed_outputs=args.write_detailed_outputs,
    )


if __name__ == "__main__":
    main()
