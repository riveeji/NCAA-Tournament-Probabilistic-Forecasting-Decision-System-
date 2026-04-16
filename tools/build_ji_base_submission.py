from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.ji_base import JIBaseConfig, build_ji_dataset, build_working_ji_base_config, load_ji_team_features
from hc.ji_base.predict import predict_submission

RESULTS = ROOT / "results"


def resolve_submission_profiles(
    men_config: JIBaseConfig | None = None,
    women_config: JIBaseConfig | None = None,
) -> list[dict]:
    men_config = men_config or build_working_ji_base_config("M")
    women_config = women_config or build_working_ji_base_config("W")
    return [
        {
            "submission_profile": "ji_base_base",
            "base_model_profile": men_config.model_family,
            "calibration_mode": men_config.calibration_mode,
            "feature_profile": men_config.feature_profile,
            "alpha_profile": men_config.alpha_profile,
            "women_quality_profile_m": men_config.women_quality_profile,
            "women_quality_profile_w": women_config.women_quality_profile,
            "women_ranking_provider_m": men_config.women_ranking_provider,
            "women_ranking_provider_w": women_config.women_ranking_provider,
            "apply_overlay": False,
            "overlay_stack": "none",
        }
    ]


def _load_sample_submission() -> pd.DataFrame:
    for candidate in [ROOT / "submission_stage2_single_final_hc.csv", ROOT / "submission_stage2.csv", ROOT / "submission_stage1.csv"]:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError("No submission template found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pure JI_base submission.")
    parser.add_argument("--output", type=Path, default=RESULTS / "submission_stage2_ji_base.csv")
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
        "--women-ranking-provider-w",
        choices=["internal_fallback", "external_consensus_v1", "external_consensus_v2", "historical_consensus_snapshots_v1"],
    )
    return parser.parse_args()


def is_default_baseline_run(args: argparse.Namespace) -> bool:
    return (
        args.output == RESULTS / "submission_stage2_ji_base.csv"
        and not args.women_quality_profile_w
        and not args.women_ranking_provider_w
    )


def main() -> None:
    args = parse_args()
    ids = _load_sample_submission()
    season = int(ids["ID"].astype(str).str.split("_", expand=True)[0].astype(int).mode().iloc[0])
    men_config = build_working_ji_base_config("M")
    women_config = build_working_ji_base_config("W")
    if args.women_quality_profile_w:
        women_config.women_quality_profile = args.women_quality_profile_w
    if args.women_ranking_provider_w:
        women_config.women_ranking_provider = args.women_ranking_provider_w
    profiles = resolve_submission_profiles(men_config=men_config, women_config=women_config)
    base_profile = profiles[0]
    team_features_m = load_ji_team_features(men_config)
    team_features_w = load_ji_team_features(women_config)
    train_m = build_ji_dataset(men_config)
    train_w = build_ji_dataset(women_config)

    men_ids = ids.loc[ids["ID"].astype(str).str.startswith(f"{season}_1")].copy()
    women_ids = ids.loc[~ids.index.isin(men_ids.index)].copy()
    men_submission = predict_submission(
        ids=men_ids,
        train=train_m.loc[train_m["Season"] < season],
        team_features=team_features_m.loc[team_features_m["Season"] == season],
        config=men_config,
    )
    women_submission = predict_submission(
        ids=women_ids,
        train=train_w.loc[train_w["Season"] < season],
        team_features=team_features_w.loc[team_features_w["Season"] == season],
        config=women_config,
    )
    submission = pd.concat([men_submission, women_submission], ignore_index=True).sort_values("ID")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    candidates_summary = pd.DataFrame(
        [
            {
                "submission_profile": profiles[0]["submission_profile"],
                "base_model_profile": profiles[0]["base_model_profile"],
                "overlay_stack": profiles[0]["overlay_stack"],
            }
        ]
    )
    if is_default_baseline_run(args):
        candidates_path = RESULTS / "ji_base_submission_candidates_summary.csv"
        summary_path = RESULTS / "ji_base_submission_summary.json"
    else:
        candidates_path = RESULTS / f"{args.output.stem}_candidates_summary.csv"
        summary_path = RESULTS / f"{args.output.stem}_summary.json"
    candidates_summary.to_csv(candidates_path, index=False)
    summary_path.write_text(json.dumps({"profiles": profiles, "output": str(args.output)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
