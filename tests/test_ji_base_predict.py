import importlib.util
from pathlib import Path

import pandas as pd

from hc.ji_base import JIBaseConfig, build_working_ji_base_config
from hc.ji_base.predict import build_submission_feature_frame, parse_submission_ids


def _load_build_ji_base_submission_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_submission.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_submission", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_submission_ids_extracts_season_and_team_ids():
    frame = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})
    parsed = parse_submission_ids(frame)

    assert parsed["Season"].tolist() == [2026]
    assert parsed["T1"].tolist() == [1101]
    assert parsed["T2"].tolist() == [1102]


def test_build_submission_feature_frame_emits_ji_base_core_columns():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5], "Season": [2026], "T1": [1101], "T2": [1102]})
    features = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1101, "SeedNum": 1, "Elo": 1610, "Quality": 1.1, "oeff": 115.0, "deff": 95.0, "neff": 20.0, "efg": 0.56, "tor": 0.14, "orpct": 0.33, "ftr": 0.28, "pace": 69.0},
            {"Season": 2026, "TeamID": 1102, "SeedNum": 12, "Elo": 1495, "Quality": -0.2, "oeff": 107.0, "deff": 101.0, "neff": 6.0, "efg": 0.51, "tor": 0.18, "orpct": 0.29, "ftr": 0.23, "pace": 66.0},
        ]
    )

    frame = build_submission_feature_frame(ids, features, JIBaseConfig(gender="M"))

    required = {"Delta_Seed", "Seed_sum", "Seed_prod", "Seed_gap_abs", "Delta_Elo", "Delta_Quality", "Delta_neff", "strength_blend"}
    assert required.issubset(frame.columns)


def test_build_submission_feature_frame_emits_women_consensus_interaction_when_requested():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5], "Season": [2026], "T1": [1101], "T2": [1102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 1,
                "Elo": 1610,
                "Quality": 1.1,
                "WomenCompositeQuality": 0.8,
                "oeff": 115.0,
                "deff": 95.0,
                "neff": 20.0,
                "efg": 0.56,
                "tor": 0.14,
                "orpct": 0.33,
                "ftr": 0.28,
                "pace": 69.0,
                "QualityWins": 1.2,
                "OpponentQualityTournamentRank": 0.7,
                "AvgBlkDiff": 0.4,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 12,
                "Elo": 1495,
                "Quality": -0.2,
                "WomenCompositeQuality": -0.3,
                "oeff": 107.0,
                "deff": 101.0,
                "neff": 6.0,
                "efg": 0.51,
                "tor": 0.18,
                "orpct": 0.29,
                "ftr": 0.23,
                "pace": 66.0,
                "QualityWins": -0.1,
                "OpponentQualityTournamentRank": -0.4,
                "AvgBlkDiff": -0.2,
            },
        ]
    )

    frame = build_submission_feature_frame(ids, features, JIBaseConfig(gender="W", feature_profile="seed_women_consensus_interaction"))

    assert "Seed_x_WomenConsensusQuality" in frame.columns


def test_build_submission_feature_frame_emits_both_seed_interactions_for_women_combined_profile():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5], "Season": [2026], "T1": [1101], "T2": [1102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 1,
                "Elo": 1610,
                "Quality": 1.1,
                "WomenCompositeQuality": 0.8,
                "oeff": 115.0,
                "deff": 95.0,
                "neff": 20.0,
                "efg": 0.56,
                "tor": 0.14,
                "orpct": 0.33,
                "ftr": 0.28,
                "pace": 69.0,
                "QualityWins": 1.2,
                "OpponentQualityTournamentRank": 0.7,
                "AvgBlkDiff": 0.4,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 12,
                "Elo": 1495,
                "Quality": -0.2,
                "WomenCompositeQuality": -0.3,
                "oeff": 107.0,
                "deff": 101.0,
                "neff": 6.0,
                "efg": 0.51,
                "tor": 0.18,
                "orpct": 0.29,
                "ftr": 0.23,
                "pace": 66.0,
                "QualityWins": -0.1,
                "OpponentQualityTournamentRank": -0.4,
                "AvgBlkDiff": -0.2,
            },
        ]
    )

    frame = build_submission_feature_frame(ids, features, JIBaseConfig(gender="W", feature_profile="seed_quality_plus_women_consensus"))

    assert "Seed_x_Quality" in frame.columns
    assert "Seed_x_WomenConsensusQuality" in frame.columns


def test_build_ji_base_submission_defaults_to_pure_base_submission():
    module = _load_build_ji_base_submission_module()
    profiles = module.resolve_submission_profiles()

    assert profiles == [
        {
            "submission_profile": "ji_base_base",
            "base_model_profile": "JI_lr_control",
            "calibration_mode": "none",
            "feature_profile": "lr_carry_elo_definition_v1",
            "alpha_profile": "quality_only_men_quality_blocks_women",
            "women_quality_profile_m": "legacy_v1",
            "women_quality_profile_w": "consensus_rebuild_v4",
            "women_ranking_provider_m": "internal_fallback",
            "women_ranking_provider_w": "internal_fallback",
            "apply_overlay": False,
            "overlay_stack": "none",
        }
    ]


def test_build_ji_base_submission_profiles_accept_women_upstream_override():
    module = _load_build_ji_base_submission_module()
    men_config = build_working_ji_base_config("M")
    women_config = build_working_ji_base_config("W")
    women_config.women_quality_profile = "consensus_rebuild_v6"
    women_config.women_ranking_provider = "external_consensus_v1"

    profiles = module.resolve_submission_profiles(men_config=men_config, women_config=women_config)

    assert profiles == [
        {
            "submission_profile": "ji_base_base",
            "base_model_profile": "JI_lr_control",
            "calibration_mode": "none",
            "feature_profile": "lr_carry_elo_definition_v1",
            "alpha_profile": "quality_only_men_quality_blocks_women",
            "women_quality_profile_m": "legacy_v1",
            "women_quality_profile_w": "consensus_rebuild_v6",
            "women_ranking_provider_m": "internal_fallback",
            "women_ranking_provider_w": "external_consensus_v1",
            "apply_overlay": False,
            "overlay_stack": "none",
        }
    ]
