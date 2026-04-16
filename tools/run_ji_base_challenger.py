from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.ji_base import JIBaseConfig, build_working_ji_base_config, run_gender_replay
from tools.run_ji_base_replay import build_combined_summary, passes_experiment_gate

RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge the frozen JI_base baseline with a single candidate.")
    parser.add_argument("--candidate-name", required=True, help="Human-readable label for the challenger run.")
    parser.add_argument("--model-family", choices=["JI_spread_xgb", "JI_lr_control", "JI_lgb_control", "JI_node_control"])
    parser.add_argument("--calibration-mode", choices=["none", "isotonic_gender"])
    parser.add_argument("--isotonic-min-samples", type=int)
    parser.add_argument(
        "--feature-profile",
        choices=[
            "baseline_v1",
            "seed_quality_interaction",
            "seed_quality_interaction_women_conservative",
            "women_tossup_quality_conservative",
            "seed_women_consensus_interaction",
            "seed_quality_plus_women_consensus",
            "strength_blend_alt",
            "tossup_upset_v1",
            "lr_pruned_only_v1",
            "lr_ratings_only_v1",
            "lr_women_fix_only_v1",
            "lr_ratings_core_v2a",
            "lr_ratings_core_v2b",
            "lr_ratings_core_v2c",
            "lr_ratings_definition_v1",
            "lr_carry_elo_definition_v1",
            "lr_carry_elo_definition_confirm80",
            "lr_colley_definition_v1",
            "lr_srs_definition_v1_clip15",
            "lr_srs_definition_confirm20",
            "lr_pruned_core_v1",
            "women_slice_redesign_v1_architecture",
            "women_slice_redesign_v1_no_seed_interaction",
            "women_opp_rank_redesign_v1_architecture",
            "women_opp_rank_redesign_v1_no_seed_interaction",
            "women_qualitywins_redesign_v1_architecture",
            "women_qualitywins_redesign_v1_with_seed_interaction",
        ],
    )
    parser.add_argument(
        "--alpha-profile",
        choices=[
            "core_alpha_v1",
            "none",
            "harry_only",
            "quality_only",
            "quality_only_women_light",
            "quality_only_men_core_women",
            "quality_only_men_quality_blocks_women",
            "quality_wins_only_men_quality_blocks_women",
            "opp_rank_only_men_quality_blocks_women",
            "quality_only_men_harry_quality_women",
            "quality_only_men_harry_blocks_women",
            "women_blocks_only",
        ],
    )
    parser.add_argument(
        "--women-quality-profile-w",
        choices=[
            "legacy_v1",
            "consensus_rebuild_v2",
            "consensus_rebuild_v3",
            "consensus_rebuild_v4",
            "consensus_rebuild_v4a",
            "consensus_rebuild_v4b",
            "consensus_rebuild_v5",
            "consensus_rebuild_v6",
        ],
    )
    parser.add_argument(
        "--women-quality-profile-m",
        choices=[
            "legacy_v1",
            "consensus_rebuild_v2",
            "consensus_rebuild_v3",
            "consensus_rebuild_v4",
            "consensus_rebuild_v4a",
            "consensus_rebuild_v4b",
            "consensus_rebuild_v5",
            "consensus_rebuild_v6",
        ],
    )
    parser.add_argument(
        "--women-ranking-provider-w",
        choices=["internal_fallback", "external_consensus_v1", "external_consensus_v2", "historical_consensus_snapshots_v1"],
    )
    parser.add_argument(
        "--women-ranking-provider-m",
        choices=["internal_fallback", "external_consensus_v1", "external_consensus_v2", "historical_consensus_snapshots_v1"],
    )
    parser.add_argument("--recent-window", type=int)
    parser.add_argument("--lr-c-m", type=float)
    parser.add_argument("--lr-c-w", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sanitize_candidate_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "candidate"


def resolve_candidate_args(candidate_name: str) -> dict[str, object]:
    if candidate_name == "core::women_slice_redesign_v1_architecture":
        return {
            "feature_profile": "women_slice_redesign_v1_architecture",
            "women_quality_profile_w": "consensus_rebuild_v5",
        }
    if candidate_name == "core::women_slice_redesign_v1_no_seed_interaction":
        return {
            "feature_profile": "women_slice_redesign_v1_no_seed_interaction",
            "women_quality_profile_w": "consensus_rebuild_v5",
        }
    if candidate_name == "core::women_opp_rank_redesign_v1_architecture":
        return {
            "feature_profile": "women_opp_rank_redesign_v1_architecture",
        }
    if candidate_name == "core::women_opp_rank_redesign_v1_no_seed_interaction":
        return {
            "feature_profile": "women_opp_rank_redesign_v1_no_seed_interaction",
        }
    if candidate_name == "core::women_qualitywins_redesign_v1_architecture":
        return {
            "feature_profile": "women_qualitywins_redesign_v1_architecture",
        }
    if candidate_name == "core::women_qualitywins_redesign_v1_with_seed_interaction":
        return {
            "feature_profile": "women_qualitywins_redesign_v1_with_seed_interaction",
        }
    if candidate_name == "core::women_ranking_upstream_v1_internal_refactor":
        return {
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_w": "internal_fallback",
        }
    if candidate_name == "core::women_ranking_upstream_v1_external_consensus":
        return {
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_w": "external_consensus_v1",
        }
    if candidate_name == "core::women_ranking_upstream_v2_internal_refactor":
        return {
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_w": "internal_fallback",
        }
    if candidate_name == "core::women_ranking_upstream_v2_external_consensus":
        return {
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_w": "external_consensus_v2",
        }
    if candidate_name == "core::women_ranking_historical_snapshots_v1":
        return {
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_w": "historical_consensus_snapshots_v1",
        }
    return {}


def _load_candidate_summary(candidate_name: str) -> dict:
    combined_path = RESULTS / "ji_base_replay_combined.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        match = combined.loc[combined["candidate_name"] == candidate_name]
        if not match.empty:
            return match.sort_values(["total_cv_brier_calibrated", "women_cv_brier_calibrated"]).iloc[0].to_dict()

    challenger_path = RESULTS / f"ji_base_challenger_{sanitize_candidate_name(candidate_name)}.json"
    if challenger_path.exists():
        payload = json.loads(challenger_path.read_text(encoding="utf-8"))
        challenger = payload.get("challenger_summary")
        if isinstance(challenger, dict) and challenger:
            return {"candidate_name": candidate_name, **challenger}

    raise KeyError(f"Frozen baseline candidate '{candidate_name}' not found in replay combined summary or challenger results.")


def build_challenger_configs(args: argparse.Namespace) -> tuple[JIBaseConfig, JIBaseConfig]:
    men = build_working_ji_base_config("M")
    women = build_working_ji_base_config("W")
    lr_c_m = getattr(args, "lr_c_m", None)
    lr_c_w = getattr(args, "lr_c_w", None)
    women_ranking_provider_m = getattr(args, "women_ranking_provider_m", None)
    women_ranking_provider_w = getattr(args, "women_ranking_provider_w", None)
    defaults = resolve_candidate_args(args.candidate_name)
    if "feature_profile" in defaults and not args.feature_profile:
        args.feature_profile = str(defaults["feature_profile"])
    if "women_quality_profile_w" in defaults and not args.women_quality_profile_w:
        args.women_quality_profile_w = str(defaults["women_quality_profile_w"])
    if "women_ranking_provider_w" in defaults and not women_ranking_provider_w:
        women_ranking_provider_w = str(defaults["women_ranking_provider_w"])
    for config in (men, women):
        if args.model_family:
            config.model_family = args.model_family
        if args.calibration_mode:
            config.calibration_mode = args.calibration_mode
        if args.feature_profile:
            config.feature_profile = args.feature_profile
        if args.alpha_profile:
            config.alpha_profile = args.alpha_profile
        if args.isotonic_min_samples is not None:
            config.isotonic_min_samples = args.isotonic_min_samples
        if args.recent_window is not None:
            config.recent_window = args.recent_window
    if lr_c_m is not None:
        men.lr_c_m = lr_c_m
        women.lr_c_m = lr_c_m
    if lr_c_w is not None:
        men.lr_c_w = lr_c_w
        women.lr_c_w = lr_c_w
    if args.women_quality_profile_m:
        men.women_quality_profile = args.women_quality_profile_m
    if args.women_quality_profile_w:
        women.women_quality_profile = args.women_quality_profile_w
    if women_ranking_provider_m:
        men.women_ranking_provider = women_ranking_provider_m
    if women_ranking_provider_w:
        women.women_ranking_provider = women_ranking_provider_w
    return men, women


def load_frozen_baseline_summary() -> dict:
    snapshot_path = RESULTS / "ji_base_baseline_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError("Frozen JI_base baseline snapshot is missing.")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    baseline_candidate = snapshot.get("working_baseline_candidate")
    if not baseline_candidate:
        raise KeyError("Frozen baseline snapshot is missing working_baseline_candidate.")
    return _load_candidate_summary(str(baseline_candidate))


def run_candidate_summary(men_config: JIBaseConfig, women_config: JIBaseConfig) -> dict:
    men = run_gender_replay(men_config)
    women = run_gender_replay(women_config)
    combined = build_combined_summary(men=men, women=women)
    combined.update(
        {
            "model_family_m": men_config.model_family,
            "model_family_w": women_config.model_family,
            "calibration_mode_m": men_config.calibration_mode,
            "calibration_mode_w": women_config.calibration_mode,
            "feature_profile_m": men_config.feature_profile,
            "feature_profile_w": women_config.feature_profile,
            "alpha_profile_m": men_config.alpha_profile,
            "alpha_profile_w": women_config.alpha_profile,
            "women_quality_profile_m": men_config.women_quality_profile,
            "women_quality_profile_w": women_config.women_quality_profile,
            "women_ranking_provider_m": men_config.women_ranking_provider,
            "women_ranking_provider_w": women_config.women_ranking_provider,
            "isotonic_min_samples_m": men_config.isotonic_min_samples,
            "isotonic_min_samples_w": women_config.isotonic_min_samples,
        }
    )
    return combined


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    slug = sanitize_candidate_name(args.candidate_name)
    return RESULTS / f"ji_base_challenger_{slug}.json"


def main() -> None:
    args = parse_args()
    baseline_summary = load_frozen_baseline_summary()
    men_config, women_config = build_challenger_configs(args)
    challenger_summary = run_candidate_summary(men_config, women_config)
    passes_gate = passes_experiment_gate(candidate=challenger_summary, baseline=baseline_summary)

    payload = {
        "candidate_name": args.candidate_name,
        "baseline_candidate": baseline_summary.get("candidate_name"),
        "baseline_summary": baseline_summary,
        "challenger_summary": challenger_summary,
        "passes_gate": bool(passes_gate),
    }
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
