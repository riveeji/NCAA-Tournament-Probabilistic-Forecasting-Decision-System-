from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.ji_base import JIBaseConfig, run_gender_replay

RESULTS = ROOT / "results"
CURRENT_GATE_BASELINE = {
    "total_cv_brier_calibrated": 0.164156,
    "women_cv_brier_calibrated": 0.143878,
    "latest_season_equal_gender_brier": 0.127427,
    "recent_window_equal_gender_brier": 0.170846,
}
LATEST_RECENT_SLACK = 0.001
WOMEN_GATE_EPSILON = 1e-9
FOCUS_MODEL_FAMILY = "JI_lr_control"
FOCUS_CALIBRATION_MODE = "none"


def resolve_variant_plan(*, include_experimental: bool = False) -> list[tuple[str, str]]:
    plan = [
        ("JI_spread_xgb", "none"),
        ("JI_spread_xgb", "isotonic_gender"),
        ("JI_lr_control", "none"),
        ("JI_lr_control", "isotonic_gender"),
        ("JI_lgb_control", "none"),
        ("JI_lgb_control", "isotonic_gender"),
    ]
    if include_experimental:
        plan.extend(
            [
                ("JI_node_control", "none"),
                ("JI_node_control", "isotonic_gender"),
            ]
        )
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged JI_base replay.")
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include experimental control branches such as JI_node_control.",
    )
    return parser.parse_args()


def build_combined_summary(*, men: dict, women: dict) -> dict:
    return {
        "total_cv_brier_raw": float((men["cv_brier_raw"] + women["cv_brier_raw"]) / 2.0),
        "total_cv_brier_calibrated": float((men["cv_brier_calibrated"] + women["cv_brier_calibrated"]) / 2.0),
        "latest_season_equal_gender_brier": float((men["latest_season_brier"] + women["latest_season_brier"]) / 2.0),
        "recent_window_equal_gender_brier": float((men["recent_window_brier"] + women["recent_window_brier"]) / 2.0),
        "men_cv_brier_raw": float(men["cv_brier_raw"]),
        "men_cv_brier_calibrated": float(men["cv_brier_calibrated"]),
        "women_cv_brier_raw": float(women["cv_brier_raw"]),
        "women_cv_brier_calibrated": float(women["cv_brier_calibrated"]),
    }


def passes_experiment_gate(*, candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["total_cv_brier_calibrated"] < baseline["total_cv_brier_calibrated"]
        and candidate["women_cv_brier_calibrated"] <= baseline["women_cv_brier_calibrated"] + WOMEN_GATE_EPSILON
        and candidate["latest_season_equal_gender_brier"] <= baseline["latest_season_equal_gender_brier"] + LATEST_RECENT_SLACK
        and candidate["recent_window_equal_gender_brier"] <= baseline["recent_window_equal_gender_brier"] + LATEST_RECENT_SLACK
    )


def _candidate_sort_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row["total_cv_brier_calibrated"]),
        float(row["women_cv_brier_calibrated"]),
        float(row["men_cv_brier_calibrated"]),
        float(row["latest_season_equal_gender_brier"]),
        float(row["recent_window_equal_gender_brier"]),
    )


def _load_existing_stage_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = RESULTS / "ji_base_replay_summary.csv"
    combined_path = RESULTS / "ji_base_replay_combined.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    combined = pd.read_csv(combined_path) if combined_path.exists() else pd.DataFrame()
    return summary, combined


def _lookup_cached_candidate(
    *,
    candidate_name: str,
    cached_summary: pd.DataFrame,
    cached_combined: pd.DataFrame,
) -> tuple[list[dict], dict] | None:
    if cached_summary.empty or cached_combined.empty:
        return None
    summary_match = cached_summary.loc[cached_summary.get("candidate_name") == candidate_name].copy()
    combined_match = cached_combined.loc[cached_combined.get("candidate_name") == candidate_name].copy()
    if len(summary_match) < 2 or combined_match.empty:
        return None
    genders = set(summary_match.get("gender", pd.Series(dtype=str)).astype(str))
    if genders != {"M", "W"}:
        return None
    return summary_match.to_dict(orient="records"), combined_match.iloc[0].to_dict()


def _run_candidate(
    *,
    phase: str,
    candidate_name: str,
    men_config: JIBaseConfig,
    women_config: JIBaseConfig,
    summary_rows: list[dict],
    combined_rows: list[dict],
    cached_summary: pd.DataFrame | None = None,
    cached_combined: pd.DataFrame | None = None,
) -> dict:
    cached_summary = cached_summary if cached_summary is not None else pd.DataFrame()
    cached_combined = cached_combined if cached_combined is not None else pd.DataFrame()
    cached = _lookup_cached_candidate(
        candidate_name=candidate_name,
        cached_summary=cached_summary,
        cached_combined=cached_combined,
    )
    if cached is not None:
        cached_summary_rows, cached_combined_row = cached
        summary_rows.extend(cached_summary_rows)
        combined_rows.append(cached_combined_row)
        return cached_combined_row

    men = run_gender_replay(men_config)
    women = run_gender_replay(women_config)

    for replay in (men, women):
        summary_rows.append(
            {
                "phase": phase,
                "candidate_name": candidate_name,
                "gender": replay["gender"],
                "model_family": replay["model_family"],
                "feature_profile": replay["feature_profile"],
                "rating_profile": replay["rating_profile"],
                "women_quality_profile": replay["women_quality_profile"],
                "alpha_profile": replay["alpha_profile"],
                "sidecar_profile": replay["sidecar_profile"],
                "calibration_mode": replay["calibration_mode"],
                "selection_objective": replay["selection_objective"],
                "cv_brier_raw": replay["cv_brier_raw"],
                "cv_brier_calibrated": replay["cv_brier_calibrated"],
                "latest_season_brier": replay["latest_season_brier"],
                "recent_window_brier": replay["recent_window_brier"],
            }
        )

    combined = build_combined_summary(men=men, women=women)
    combined.update(
        {
            "phase": phase,
            "candidate_name": candidate_name,
            "variant": f"{men_config.model_family}@{men_config.calibration_mode}",
            "model_family": men_config.model_family,
            "calibration_mode": men_config.calibration_mode,
            "feature_profile": women_config.feature_profile,
            "women_quality_profile": women_config.women_quality_profile,
            "alpha_profile": women_config.alpha_profile,
        }
    )
    combined_rows.append(combined)
    return combined


def _build_focus_configs(**overrides: str) -> tuple[JIBaseConfig, JIBaseConfig]:
    common = {
        "model_family": FOCUS_MODEL_FAMILY,
        "calibration_mode": FOCUS_CALIBRATION_MODE,
        "feature_profile": "baseline_v1",
        "women_quality_profile": "legacy_v1",
        "alpha_profile": "core_alpha_v1",
    }
    common.update(overrides)
    men_config = JIBaseConfig(gender="M", **common)
    women_config = JIBaseConfig(gender="W", **common)
    return men_config, women_config


def main() -> None:
    args = parse_args()
    summary_rows: list[dict] = []
    combined_rows: list[dict] = []
    cached_summary, cached_combined = _load_existing_stage_results()

    for model_family, calibration_mode in resolve_variant_plan(include_experimental=args.include_experimental):
        candidate_name = f"baseline::{model_family}@{calibration_mode}"
        men_config = JIBaseConfig(gender="M", model_family=model_family, calibration_mode=calibration_mode)
        women_config = JIBaseConfig(gender="W", model_family=model_family, calibration_mode=calibration_mode)
        _run_candidate(
            phase="baseline",
            candidate_name=candidate_name,
            men_config=men_config,
            women_config=women_config,
            summary_rows=summary_rows,
            combined_rows=combined_rows,
            cached_summary=cached_summary,
            cached_combined=cached_combined,
        )

    stopped_after_phase = "baseline"

    women_quality_candidate = _run_candidate(
        phase="women_quality",
        candidate_name="women_quality::consensus_rebuild_v4",
        men_config=_build_focus_configs(women_quality_profile="legacy_v1")[0],
        women_config=_build_focus_configs(women_quality_profile="consensus_rebuild_v4")[1],
        summary_rows=summary_rows,
        combined_rows=combined_rows,
        cached_summary=cached_summary,
        cached_combined=cached_combined,
    )
    if not passes_experiment_gate(candidate=women_quality_candidate, baseline=CURRENT_GATE_BASELINE):
        stopped_after_phase = "women_quality"
    else:
        stopped_after_phase = "feature_profiles"
        feature_candidates: list[dict] = []
        feature_configs = [
            ("feature::seed_quality_interaction", "seed_quality_interaction"),
            ("feature::seed_women_consensus_interaction", "seed_women_consensus_interaction"),
            ("feature::seed_quality_plus_women_consensus", "seed_quality_plus_women_consensus"),
            ("feature::strength_blend_alt", "strength_blend_alt"),
        ]
        for candidate_name, feature_profile in feature_configs:
            men_config, women_config = _build_focus_configs(
                women_quality_profile="consensus_rebuild_v4",
                feature_profile=feature_profile,
            )
            feature_candidates.append(
                _run_candidate(
                    phase="feature_profiles",
                    candidate_name=candidate_name,
                    men_config=men_config,
                    women_config=women_config,
                    summary_rows=summary_rows,
                    combined_rows=combined_rows,
                    cached_summary=cached_summary,
                    cached_combined=cached_combined,
                )
            )

        passing_feature_candidates = [row for row in feature_candidates if passes_experiment_gate(candidate=row, baseline=women_quality_candidate)]
        if not passing_feature_candidates:
            stopped_after_phase = "feature_profiles"
        else:
            selected_feature = sorted(passing_feature_candidates, key=_candidate_sort_key)[0]
            stopped_after_phase = "alpha_profiles"
            alpha_candidates: list[dict] = []
            for alpha_profile in (
                "none",
                "harry_only",
                "quality_only",
                "quality_only_women_light",
                "quality_only_men_core_women",
                "quality_only_men_quality_blocks_women",
                "quality_only_men_harry_quality_women",
                "quality_only_men_harry_blocks_women",
                "women_blocks_only",
                "core_alpha_v1",
            ):
                men_config, women_config = _build_focus_configs(
                    women_quality_profile="consensus_rebuild_v4",
                    feature_profile=selected_feature["feature_profile"],
                    alpha_profile=alpha_profile,
                )
                alpha_candidates.append(
                    _run_candidate(
                        phase="alpha_profiles",
                        candidate_name=f"alpha::{alpha_profile}",
                        men_config=men_config,
                        women_config=women_config,
                        summary_rows=summary_rows,
                        combined_rows=combined_rows,
                        cached_summary=cached_summary,
                        cached_combined=cached_combined,
                    )
                )
            if any(passes_experiment_gate(candidate=row, baseline=selected_feature) for row in alpha_candidates):
                stopped_after_phase = "completed"

    summary_df = pd.DataFrame(summary_rows)
    combined_df = pd.DataFrame(combined_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS / "ji_base_replay_summary.csv", index=False)
    combined_df.sort_values(["phase", "total_cv_brier_calibrated", "women_cv_brier_calibrated"]).to_csv(RESULTS / "ji_base_replay_combined.csv", index=False)
    combined_df.loc[combined_df["phase"] == "women_quality"].to_csv(RESULTS / "ji_base_women_quality_comparison.csv", index=False)
    combined_df.loc[combined_df["phase"] == "feature_profiles"].to_csv(RESULTS / "ji_base_feature_comparison.csv", index=False)
    combined_df.loc[combined_df["phase"] == "alpha_profiles"].to_csv(RESULTS / "ji_base_alpha_comparison.csv", index=False)
    (RESULTS / "ji_base_replay_summary.json").write_text(
        json.dumps(
            {
                "summary_rows": summary_rows,
                "combined": combined_rows,
                "stopped_after_phase": stopped_after_phase,
                "gate_baseline": CURRENT_GATE_BASELINE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
