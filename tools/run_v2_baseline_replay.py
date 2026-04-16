from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.v2 import V2Config, run_gender_replay

RESULTS = ROOT / "results"
SPREAD_BASELINE_VARIANTS = [
    ("spread", "lr", "none", "external_base_pruned", "basecal"),
    ("spread", "lr", "sportsbook", "external_base_pruned", "basecal"),
    ("spread", "lr", "none", "base", "basecal"),
]
SPREAD_PREDICTION_MARKET_VARIANTS = [
    ("spread", "lr", "sportsbook_prediction", "external_base_pruned", "basecal")
]
SPREAD_CONTROL_VARIANTS = [
    ("spread", "lr", "none", "external_base", "basecal"),
    ("spread", "lr", "none", "strength_full", "basecal"),
    ("spread", "tree", "none", "external_base_pruned", "basecal"),
    ("spread", "lr", "none", "external_base_pruned", "gendercal"),
    ("spread", "lr", "none", "external_base_pruned", "monotoniccal"),
]


def resolve_variant_plan(*, include_controls: bool, include_prediction_market: bool) -> list[tuple[str, str, str, str, str]]:
    plan = list(SPREAD_BASELINE_VARIANTS)
    if include_prediction_market:
        plan.extend(SPREAD_PREDICTION_MARKET_VARIANTS)
    if include_controls:
        plan.extend(SPREAD_CONTROL_VARIANTS)
    return plan


def learner_family(model_variant: str) -> str:
    if model_variant == "lr":
        return "linear"
    if model_variant == "tree":
        return "tree"
    return "blend"


def variant_label(route: str, model_variant: str, market_mode: str, feature_pack: str, calibration_mode: str) -> str:
    return f"{route}-{learner_family(model_variant)}:{market_mode}@{feature_pack}+{calibration_mode}"


def _sort_by_existing_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame
    return frame.sort_values(existing)


def _json_ready(summary_rows: list[dict], combined_rows: list[dict]) -> dict:
    return {
        "summary_rows": summary_rows,
        "combined": combined_rows,
    }


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
    summary_df = _sort_by_existing_columns(
        pd.DataFrame(summary_rows),
        ["route", "learner_family", "market_mode", "feature_pack", "calibration_mode", "gender_profile", "gender"],
    )
    combined_df = _sort_by_existing_columns(pd.DataFrame(combined_rows), ["equal_gender_mean_brier"])

    summary_df.to_csv(output_dir / "v2_replay_summary.csv", index=False)
    combined_df.to_csv(output_dir / "v2_replay_combined.csv", index=False)
    (output_dir / "v2_replay_summary.json").write_text(
        json.dumps(_json_ready(summary_rows, combined_rows), indent=2),
        encoding="utf-8",
    )

    if write_detailed_outputs:
        _sort_by_existing_columns(by_season_df, ["route", "market_mode", "model_variant", "gender", "season"]).to_csv(
            output_dir / "v2_replay_by_season.csv",
            index=False,
        )
        predictions_df.to_csv(output_dir / "v2_replay_predictions.csv", index=False)


def _combined_rows(men_women_lookup: dict[tuple[str, str], dict]) -> list[dict]:
    combined_rows: list[dict] = []
    keys = sorted({key for _, key in men_women_lookup.keys()})
    for key in keys:
        men = men_women_lookup.get(("M", key))
        women = men_women_lookup.get(("W", key))
        if not men or not women:
            continue
        combined_rows.append(
            {
                "variant": key,
                "route": men["route"],
                "model_variant": men["model_variant"],
                "learner_family": men["learner_family"],
                "market_mode": men["market_mode"],
                "feature_pack": men["feature_pack"],
                "calibration_mode": men["calibration_mode"],
                "gender_profile": f"{men['gender_profile']}|{women['gender_profile']}",
                "equal_gender_mean_brier": float((men["mean_brier"] + women["mean_brier"]) / 2.0),
                "latest_season": int(max(men["latest_season"], women["latest_season"])),
                "equal_gender_latest_season_brier": float((men["latest_season_brier"] + women["latest_season_brier"]) / 2.0),
                "men_mean_brier": men["mean_brier"],
                "women_mean_brier": women["mean_brier"],
                "men_latest_season_brier": men["latest_season_brier"],
                "women_latest_season_brier": women["latest_season_brier"],
                "men_recent_window_brier": men["recent_window_brier"],
                "women_recent_window_brier": women["recent_window_brier"],
                "equal_gender_recent_window_brier": float((men["recent_window_brier"] + women["recent_window_brier"]) / 2.0),
            }
        )
    return combined_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spread-first NCAA v2 replay baselines and keep only the key summary artifacts by default."
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Add probability-route control variants for the strongest default feature/calibration settings.",
    )
    parser.add_argument(
        "--include-prediction-market",
        action="store_true",
        help="Add optional sportsbook+prediction-market spread variants across all default feature/calibration settings.",
    )
    parser.add_argument(
        "--write-detailed-outputs",
        action="store_true",
        help="Also write `v2_replay_by_season.csv` and `v2_replay_predictions.csv`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS,
        help="Directory for generated v2 replay artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_plan = resolve_variant_plan(
        include_controls=args.include_controls,
        include_prediction_market=args.include_prediction_market,
    )

    summary_rows: list[dict] = []
    by_season_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    men_women_lookup: dict[tuple[str, str], dict] = {}

    for gender in ("M", "W"):
        for route, model_variant, market_mode, feature_pack, calibration_mode in variant_plan:
            cfg = V2Config(
                gender=gender,
                route=route,
                model_variant=model_variant,
                market_mode=market_mode,
                feature_pack=feature_pack,
                calibration_mode=calibration_mode,
            )
            replay = run_gender_replay(cfg)
            summary_rows.append(
                {
                    "gender": gender,
                    "route": route,
                    "model_variant": model_variant,
                    "learner_family": replay["learner_family"],
                    "market_mode": market_mode,
                    "feature_pack": replay["feature_pack"],
                    "calibration_mode": replay["calibration_mode"],
                    "gender_profile": replay["gender_profile"],
                    "mean_brier": replay["mean_brier"],
                    "latest_season": replay["latest_season"],
                    "latest_season_brier": replay["latest_season_brier"],
                    "recent_window_brier": replay["recent_window_brier"],
                    "brier_variance": replay["brier_variance"],
                    "sportsbook_coverage_mean": replay["sportsbook_coverage_mean"],
                    "prediction_market_coverage_mean": replay["prediction_market_coverage_mean"],
                }
            )
            men_women_lookup[(gender, variant_label(route, model_variant, market_mode, feature_pack, replay["calibration_mode"]))] = replay
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
