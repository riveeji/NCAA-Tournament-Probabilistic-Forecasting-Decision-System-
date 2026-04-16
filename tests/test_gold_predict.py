import importlib.util
from pathlib import Path

import pandas as pd

from hc.gold import GoldConfig
from hc.gold.predict import build_submission_feature_frame, parse_submission_ids


def _load_build_gold_submission_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_gold_submission.py"
    spec = importlib.util.spec_from_file_location("build_gold_submission", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_submission_ids_extracts_season_and_team_ids():
    frame = pd.DataFrame({"ID": ["2026_1101_1102", "2026_2101_2102"], "Pred": [0.5, 0.5]})
    parsed = parse_submission_ids(frame)

    assert list(parsed.columns)[:4] == ["ID", "Pred", "Season", "T1"]
    assert parsed["Season"].tolist() == [2026, 2026]
    assert parsed["T1"].tolist() == [1101, 2101]
    assert parsed["T2"].tolist() == [1102, 2102]


def test_build_submission_feature_frame_emits_gold_diff_columns():
    ids = pd.DataFrame(
        {
            "ID": ["2026_1101_1102"],
            "Pred": [0.5],
            "Season": [2026],
            "T1": [1101],
            "T2": [1102],
        }
    )
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 1,
                "SeedPriorExpectedWins": 4.0,
                "CarryoverElo": 1600,
                "MoveElo": 1650,
                "GLMQuality": 1.2,
                "SRS": 11.0,
                "Colley": 0.72,
                "MasseyComposite": 0.80,
                "GoldConsensusStrength": 0.82,
                "CustomStrengthCore": 0.82,
                "CustomNetRating": 21.0,
                "OpponentQualityScore": 0.55,
                "AvgMargin": 14.0,
                "SOS": 1540,
                "Last30SOS": 1560,
                "CloseGameWinRate": 0.65,
                "OTNormalizedMargin": 12.0,
                "QualityWins": 8.0,
                "APStrength": 0.9,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 16,
                "SeedPriorExpectedWins": 0.1,
                "CarryoverElo": 1450,
                "MoveElo": 1430,
                "GLMQuality": -0.8,
                "SRS": -4.0,
                "Colley": 0.41,
                "MasseyComposite": -0.65,
                "GoldConsensusStrength": -0.71,
                "CustomStrengthCore": -0.71,
                "CustomNetRating": -7.0,
                "OpponentQualityScore": 0.10,
                "AvgMargin": -2.0,
                "SOS": 1480,
                "Last30SOS": 1470,
                "CloseGameWinRate": 0.42,
                "OTNormalizedMargin": -1.0,
                "QualityWins": 0.0,
                "APStrength": -0.5,
            },
        ]
    )

    frame = build_submission_feature_frame(ids, features, GoldConfig(gender="M"))

    for column in [
        "SeedNum_diff",
        "MoveElo_diff",
        "MasseyComposite_diff",
        "SeedAbsGap",
        "Seed_x_MasseyComposite_diff",
        "APStrength_diff",
        "SOS_diff",
    ]:
        assert column in frame.columns
    assert "GoldConsensusStrength_diff" not in frame.columns


def test_build_gold_submission_exposes_recovered_candidate_profiles():
    module = _load_build_gold_submission_module()
    profiles = module.resolve_submission_profiles()

    assert [profile["submission_profile"] for profile in profiles] == [
        "gold_recover_base",
        "gold_recover_market",
        "gold_recover_base_a_tier",
        "gold_recover_market_a_tier",
        "gold_recover_base_m_ap_removed",
        "gold_recover_market_m_ap_removed",
        "gold_recover_base_w_polls_removed",
        "gold_recover_market_w_polls_removed",
        "gold_recover_market_direct_only",
        "gold_recover_market_direct_priority",
        "gold_recover_market_injury_only",
        "gold_recover_market_injury_sharp_only",
        "gold_harry_base",
        "gold_harry_market",
        "gold_harry_market_injury",
        "gold_harry_market_injury_sharp",
        "gold_blend_60_40",
        "gold_blend_market_injury",
        "gold_blend_market_injury_sharp",
    ]
    women_profile = module.resolve_submission_profiles(gender="W")[-1]
    assert women_profile["allow_injury"] is False
    assert women_profile["allow_sharpen"] is False
    assert women_profile["overlay_stack"] == "market_injury_sharp"
    assert profiles[2]["rating_source_profile"] == "a_tier_default"
    assert profiles[4]["rating_source_profile"] == "m_ap_removed_only"
    assert profiles[6]["rating_source_profile"] == "w_polls_removed_only"
    assert profiles[9]["overlay_source_profile"] == "direct_priority"


def test_base_submission_cache_key_includes_rating_source_profile():
    module = _load_build_gold_submission_module()
    current = module.resolve_submission_profiles()[0]
    a_tier = module.resolve_submission_profiles()[2]
    men_only = module.resolve_submission_profiles()[4]

    def cache_key(profile: dict) -> tuple[str, str, str, str, str]:
        return (
            "M",
            profile["base_model_profile"],
            str(profile.get("rating_source_profile", "current_default")),
            str(profile.get("secondary_model_profile", "")),
            module.json.dumps(profile.get("blend_weights", {}), sort_keys=True),
        )

    assert cache_key(current) != cache_key(a_tier)
    assert cache_key(current) != cache_key(men_only)


def test_build_gold_submission_seeds_known_official_lb_scores(tmp_path):
    module = _load_build_gold_submission_module()
    log_path = tmp_path / "official_lb_log.csv"

    frame = module.seed_official_lb_log(log_path)

    assert {"submission_profile", "base_model_profile", "overlay_stack", "date", "official_lb", "notes"} == set(frame.columns)
    assert float(frame.loc[frame["submission_profile"] == "gold_recover_base", "official_lb"].iloc[0]) == 0.1306
    assert float(frame.loc[frame["submission_profile"] == "gold_recover_market", "official_lb"].iloc[0]) == 0.1289
    assert float(frame.loc[frame["submission_profile"] == "gold_min_market_injury_sharp", "official_lb"].iloc[0]) == 0.17


def test_build_gold_submission_defaults_to_gold_recover_market_profiles():
    module = _load_build_gold_submission_module()
    men_lookup = module._profile_lookup(module.resolve_submission_profiles(gender="M"))
    women_lookup = module._profile_lookup(module.resolve_submission_profiles(gender="W"))
    selected_profile_name = "gold_recover_market"

    assert men_lookup[selected_profile_name]["base_model_profile"] == "gold_lr_recover"
    assert women_lookup[selected_profile_name]["base_model_profile"] == "gold_lr_recover"
    assert men_lookup[selected_profile_name]["overlay_stack"] == "market_injury_sharp"
    assert women_lookup[selected_profile_name]["overlay_stack"] == "market_injury_sharp"


def test_build_submission_feature_frame_emits_harry_diff_columns():
    ids = pd.DataFrame(
        {
            "ID": ["2026_1101_1102"],
            "Pred": [0.5],
            "Season": [2026],
            "T1": [1101],
            "T2": [1102],
        }
    )
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 1,
                "SeedPriorExpectedWins": 4.0,
                "harry_Rating": 1.5,
                "OpponentQualityTournamentRank": 0.85,
                "QualityWins": 8.0,
                "AvgMargin": 14.0,
                "InjuryAdjustedStrength": 1.45,
                "AvgBlkDiff": 2.0,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 16,
                "SeedPriorExpectedWins": 0.1,
                "harry_Rating": -1.1,
                "OpponentQualityTournamentRank": 0.12,
                "QualityWins": 0.0,
                "AvgMargin": -2.0,
                "InjuryAdjustedStrength": -1.0,
                "AvgBlkDiff": -0.5,
            },
        ]
    )

    frame = build_submission_feature_frame(ids, features, GoldConfig(gender="M", feature_profile="gold_harry_m"))

    for column in [
        "harry_Rating_diff",
        "OpponentQualityTournamentRank_diff",
        "InjuryAdjustedStrength_diff",
        "Seed_x_harry_Rating_diff",
    ]:
        assert column in frame.columns
