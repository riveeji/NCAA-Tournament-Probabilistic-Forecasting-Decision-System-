from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.ji_base import FROZEN_OVERLAY_SUBMISSION_PROFILE, build_ji_base_overlay_config
from hc.ji_base.overlay import apply_submission_overlay
from hc.ji_base.predict import parse_submission_ids

RESULTS = ROOT / "results"


def resolve_overlay_profiles() -> list[dict]:
    return [
        {
            "submission_profile": "ji_base_overlay_v1",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_priority",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_conservative_injury",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_priority",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed3",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed4",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed5",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_priority",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight070",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight060",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight050",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight040",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight030",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight020",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight025",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v2_men_player_injury_weight025",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
    ]


def resolve_overlay_profile(submission_profile: str) -> dict:
    for profile in resolve_overlay_profiles():
        if profile["submission_profile"] == submission_profile:
            return profile
    raise ValueError(f"Unknown overlay submission profile: {submission_profile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent JI_base overlay submission candidate.")
    parser.add_argument("--base-input", type=Path, default=RESULTS / "submission_stage2_ji_base.csv")
    parser.add_argument("--submission-profile", default=FROZEN_OVERLAY_SUBMISSION_PROFILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


def _load_snapshot() -> dict:
    path = RESULTS / "ji_base_baseline_snapshot.json"
    if not path.exists():
        raise FileNotFoundError("Frozen JI_base baseline snapshot is missing.")
    return json.loads(path.read_text(encoding="utf-8"))


def _default_paths(submission_profile: str) -> tuple[Path, Path, Path, Path]:
    if submission_profile == "ji_base_overlay_v1":
        return (
            RESULTS / "submission_stage2_ji_base_overlay.csv",
            RESULTS / "ji_base_overlay_audit.csv",
            RESULTS / "ji_base_overlay_summary.json",
            RESULTS / "ji_base_overlay_candidates_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_conservative_injury":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_conservative_injury.csv",
            RESULTS / "ji_base_overlay_conservative_injury_audit.csv",
            RESULTS / "ji_base_overlay_conservative_injury_summary.json",
            RESULTS / "ji_base_overlay_conservative_injury_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only.csv",
            RESULTS / "ji_base_overlay_direct_only_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_summary.json",
            RESULTS / "ji_base_overlay_direct_only_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_strict_confirmed":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only_injury_strict_confirmed.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_strict_confirmed_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_strict_confirmed_summary.json",
            RESULTS / "ji_base_overlay_direct_only_injury_strict_confirmed_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed3":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only_injury_confirmed3.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed3_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed3_summary.json",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed3_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed4":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only_injury_confirmed4.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_summary.json",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed5":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only_injury_confirmed5.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed5_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed5_summary.json",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed5_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_direct_only_injury_confirmed4_shift008.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_shift008_audit.csv",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_shift008_summary.json",
            RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_shift008_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_priority":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_priority.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_priority_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_priority_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_priority_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight070":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight070.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight070_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight070_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight070_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight060":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight060.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight060_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight060_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight060_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight050":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight050.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight050_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight050_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight050_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight040":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight040.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight040_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight040_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight040_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight030":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight030.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight030_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight030_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight030_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight020":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight020.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight020_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight020_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight020_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v1_men_best_women_direct_only_weight025":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_men_best_women_direct_only_weight025.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight025_audit.csv",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight025_summary.json",
            RESULTS / "ji_base_overlay_men_best_women_direct_only_weight025_candidate_summary.csv",
        )
    if submission_profile == "ji_base_overlay_v2_men_player_injury_weight025":
        return (
            RESULTS / "submission_stage2_ji_base_overlay_v2_men_player_injury_weight025.csv",
            RESULTS / "ji_base_overlay_v2_men_player_injury_weight025_audit.csv",
            RESULTS / "ji_base_overlay_v2_men_player_injury_weight025_summary.json",
            RESULTS / "ji_base_overlay_v2_men_player_injury_weight025_candidate_summary.csv",
        )
    raise ValueError(f"Unknown overlay submission profile for default path resolution: {submission_profile}")


def _build_gender_overlay(base_predictions: pd.DataFrame, *, gender: str, season: int, submission_profile: str):
    config = build_ji_base_overlay_config(gender, submission_profile)
    adjusted, audit, summary = apply_submission_overlay(base_predictions, gender=gender, season=season, config=config)
    audit["gender"] = gender
    summary["submission_profile"] = submission_profile
    summary["base_submission_profile"] = "ji_base_base"
    return adjusted, audit, summary, config


def main() -> None:
    args = parse_args()
    profile = resolve_overlay_profile(args.submission_profile)
    snapshot = _load_snapshot()
    output_path, audit_path, summary_path, candidate_summary_path = _default_paths(args.submission_profile)
    if args.output is not None:
        output_path = args.output
    if args.audit_output is not None:
        audit_path = args.audit_output

    if not args.base_input.exists():
        raise FileNotFoundError(f"Base submission file not found: {args.base_input}")

    base = pd.read_csv(args.base_input)
    parsed = parse_submission_ids(base)
    season = int(parsed["Season"].mode().iloc[0])

    men_mask = parsed["ID"].astype(str).str.startswith(f"{season}_1")
    men_base = parsed.loc[men_mask].copy()
    women_base = parsed.loc[~men_mask].copy()

    men_submission, men_audit, men_summary, men_config = _build_gender_overlay(
        men_base, gender="M", season=season, submission_profile=args.submission_profile
    )
    women_submission, women_audit, women_summary, women_config = _build_gender_overlay(
        women_base, gender="W", season=season, submission_profile=args.submission_profile
    )

    submission = pd.concat([men_submission, women_submission], ignore_index=True).sort_values("ID")
    audit = pd.concat([men_audit, women_audit], ignore_index=True).sort_values(["gender", "ID"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)

    candidate_summary = pd.DataFrame(
        [
            {
                "submission_profile": profile["submission_profile"],
                "base_submission_profile": profile["base_submission_profile"],
                "overlay_source_profile_m": men_config.overlay_source_profile,
                "overlay_source_profile_w": women_config.overlay_source_profile,
                "overlay_stack_m": men_config.resolved_overlay_stack(),
                "overlay_stack_w": women_config.resolved_overlay_stack(),
                "output": str(output_path),
                "audit_path": str(audit_path),
            }
        ]
    )
    candidate_summary.to_csv(candidate_summary_path, index=False)

    summary = {
        "season": season,
        "rows": int(len(submission)),
        "overlay_enabled": True,
        "submission_profile": {"M": profile["submission_profile"], "W": profile["submission_profile"]},
        "base_submission_profile": snapshot["submission_profile"],
        "base_snapshot": {
            "candidate": snapshot["working_baseline_candidate"],
            "base_model_profile": snapshot["base_model_profile"],
            "feature_profile": snapshot["feature_profile"],
            "alpha_profile": snapshot["alpha_profile"],
            "women_quality_profile_m": snapshot["women_quality_profile_m"],
            "women_quality_profile_w": snapshot["women_quality_profile_w"],
            "calibration_mode": snapshot["calibration_mode"],
        },
        "candidate_outputs": {
            args.submission_profile: str(output_path),
        },
        "candidate_summary_path": str(candidate_summary_path),
        "audit_path": str(audit_path),
        "men": men_summary,
        "women": women_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
