from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.next_arch import NextArchConfig, run_next_arch_gender_replay
from tools.run_ji_base_replay import build_combined_summary, passes_experiment_gate

RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a next-architecture replay challenger against the frozen JI_base baseline.")
    parser.add_argument(
        "--candidate-name",
        required=True,
        choices=[
            "arch::tabr_v1",
            "arch::tabr_hybrid_v1",
            "arch::tabr_feature_fusion_v1",
            "arch::pairwise_ranking_v1",
            "arch::season_encoder_transformer_v1",
            "arch::graph_static_embedding_v1",
            "arch::gender_specific_stacker_v1",
        ],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sanitize_candidate_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "candidate"


def resolve_candidate_args(candidate_name: str) -> dict[str, object]:
    if candidate_name == "arch::tabr_v1":
        return {"experiment_name": "tabr_v1"}
    if candidate_name == "arch::tabr_hybrid_v1":
        return {"experiment_name": "tabr_hybrid_v1"}
    if candidate_name == "arch::tabr_feature_fusion_v1":
        return {"experiment_name": "tabr_feature_fusion_v1"}
    if candidate_name == "arch::pairwise_ranking_v1":
        return {"experiment_name": "pairwise_ranking_v1"}
    if candidate_name == "arch::season_encoder_transformer_v1":
        return {"experiment_name": "season_encoder_transformer_v1"}
    if candidate_name == "arch::graph_static_embedding_v1":
        return {"experiment_name": "graph_static_embedding_v1"}
    if candidate_name == "arch::gender_specific_stacker_v1":
        return {"experiment_name": "gender_specific_stacker_v1"}
    return {}


def _load_candidate_summary(candidate_name: str) -> dict:
    challenger_path = RESULTS / f"ji_base_challenger_{sanitize_candidate_name(candidate_name)}.json"
    if challenger_path.exists():
        payload = json.loads(challenger_path.read_text(encoding="utf-8"))
        challenger = payload.get("challenger_summary")
        if isinstance(challenger, dict) and challenger:
            return {"candidate_name": candidate_name, **challenger}
    raise KeyError(f"Frozen baseline candidate '{candidate_name}' not found in challenger results.")


def load_frozen_baseline_summary() -> dict:
    snapshot_path = RESULTS / "ji_base_baseline_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    baseline_candidate = snapshot.get("working_baseline_candidate")
    if not baseline_candidate:
        raise KeyError("Frozen baseline snapshot is missing working_baseline_candidate.")
    return _load_candidate_summary(str(baseline_candidate))


def build_challenger_configs(candidate_name: str) -> tuple[NextArchConfig, NextArchConfig]:
    resolved = resolve_candidate_args(candidate_name)
    experiment_name = str(resolved["experiment_name"])
    return (
        NextArchConfig(gender="M", experiment_name=experiment_name),  # type: ignore[arg-type]
        NextArchConfig(gender="W", experiment_name=experiment_name),  # type: ignore[arg-type]
    )


def run_candidate_summary(men_config: NextArchConfig, women_config: NextArchConfig) -> dict:
    men = run_next_arch_gender_replay(men_config)
    women = run_next_arch_gender_replay(women_config)
    combined = build_combined_summary(men=men, women=women)
    combined.update(
        {
            "model_family_m": men_config.experiment_name,
            "model_family_w": women_config.experiment_name,
            "feature_profile_m": men_config.experiment_name,
            "feature_profile_w": women_config.experiment_name,
            "alpha_profile_m": "n/a",
            "alpha_profile_w": "n/a",
            "women_quality_profile_m": "n/a",
            "women_quality_profile_w": "n/a",
            "women_ranking_provider_m": "n/a",
            "women_ranking_provider_w": "n/a",
            "calibration_mode_m": "none",
            "calibration_mode_w": "none",
            "isotonic_min_samples_m": 0,
            "isotonic_min_samples_w": 0,
        }
    )
    return combined


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    slug = sanitize_candidate_name(args.candidate_name)
    return RESULTS / f"next_arch_challenger_{slug}.json"


def main() -> None:
    args = parse_args()
    baseline_summary = load_frozen_baseline_summary()
    men_config, women_config = build_challenger_configs(args.candidate_name)
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
