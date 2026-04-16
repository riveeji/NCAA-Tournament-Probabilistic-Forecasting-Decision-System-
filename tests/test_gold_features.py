import pandas as pd

from hc.gold import GoldConfig
from hc.gold.data import build_gold_dataset, load_gold_team_features


def test_gold_config_exposes_source_profiles():
    config = GoldConfig(gender="M")

    assert config.rating_source_profile == "current_default"
    assert config.overlay_source_profile == "current_default"
    assert config.resolved_rating_source_profile() == "current_default"
    assert config.resolved_overlay_source_profile() == "current_default"


def test_gold_dataset_builds_recover_wide_columns_for_default_men():
    dataset = build_gold_dataset(GoldConfig(gender="M"))

    expected = {
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "MoveElo_diff",
        "GLMQuality_diff",
        "MasseyComposite_diff",
        "CustomNetRating_diff",
        "OpponentQualityScore_diff",
        "SeedAbsGap",
        "QualityWins_diff",
        "AvgMargin_diff",
        "APStrength_diff",
        "SOS_diff",
        "Last30SOS_diff",
        "CloseGameWinRate_diff",
        "OTNormalizedMargin_diff",
        "Seed_x_MasseyComposite_diff",
    }

    assert expected.issubset(dataset.columns)
    assert "GoldConsensusStrength_diff" not in dataset.columns
    assert "Label" in dataset.columns
    assert "Season" in dataset.columns
    assert "DayNum" in dataset.columns


def test_gold_dataset_builds_pruned_and_augmented_profiles_explicitly():
    pruned = build_gold_dataset(GoldConfig(gender="W", feature_profile="gold_pruned_w"))
    augmented = build_gold_dataset(GoldConfig(gender="W", feature_profile="gold_augmented_w"))

    expected = {
        "SeedNum_diff",
        "SeedAbsGap",
        "MoveElo_diff",
        "GLMQuality_diff",
        "MasseyComposite_diff",
        "CustomNetRating_diff",
        "OpponentQualityScore_diff",
        "QualityWins_diff",
        "AvgBlkDiff_diff",
    }

    assert expected.issubset(pruned.columns)
    assert "SOS_diff" not in pruned.columns
    assert "CloseGameWinRate_diff" not in pruned.columns
    assert "CustomStrengthCore_diff" in augmented.columns
    assert pruned["Label"].isin([0, 1]).all()


def test_gold_team_features_are_unique_by_season_team_and_include_consensus_strength():
    men = load_gold_team_features(GoldConfig(gender="M"))
    women = load_gold_team_features(GoldConfig(gender="W"))

    assert not men.duplicated(["Season", "TeamID"]).any()
    assert not women.duplicated(["Season", "TeamID"]).any()
    assert "GoldConsensusStrength" in men.columns
    assert "CustomStrengthCore" in men.columns
    assert "GoldConsensusStrength" in women.columns
    assert "CustomStrengthCore" in women.columns
    assert "AvgBlkDiff" in women.columns
    assert "harry_Rating" in men.columns
    assert "OpponentQualityTournamentRank" in men.columns
    assert "InjuryAdjustedStrength" in men.columns
    assert "harry_Rating" in women.columns
    assert "OpponentQualityTournamentRank" in women.columns


def test_m_ap_removed_only_keeps_women_poll_sources_intact():
    current = load_gold_team_features(GoldConfig(gender="W", rating_source_profile="current_default"))
    men_only = load_gold_team_features(GoldConfig(gender="W", rating_source_profile="m_ap_removed_only"))

    pd.testing.assert_series_equal(current["MasseyComposite"], men_only["MasseyComposite"], check_names=False)


def test_w_polls_removed_only_keeps_men_sources_intact_and_changes_women_polls():
    men_current = load_gold_team_features(GoldConfig(gender="M", rating_source_profile="current_default"))
    men_removed = load_gold_team_features(GoldConfig(gender="M", rating_source_profile="w_polls_removed_only"))
    women_current = load_gold_team_features(GoldConfig(gender="W", rating_source_profile="current_default"))
    women_removed = load_gold_team_features(GoldConfig(gender="W", rating_source_profile="w_polls_removed_only"))

    pd.testing.assert_series_equal(men_current["MasseyComposite"], men_removed["MasseyComposite"], check_names=False)
    assert not women_current["MasseyComposite"].equals(women_removed["MasseyComposite"])


def test_gold_dataset_builds_harry_profiles_with_gender_specific_columns():
    men = build_gold_dataset(
        GoldConfig(
            gender="M",
            model_family="gold_harry_xgb_spread",
            calibration_mode="isotonic_gender",
            feature_profile="gold_harry_m",
        )
    )
    women = build_gold_dataset(
        GoldConfig(
            gender="W",
            model_family="gold_harry_lr",
            calibration_mode="none",
            feature_profile="gold_harry_w",
        )
    )

    men_expected = {
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "harry_Rating_diff",
        "OpponentQualityTournamentRank_diff",
        "QualityWins_diff",
        "AvgMargin_diff",
        "InjuryAdjustedStrength_diff",
        "Seed_x_harry_Rating_diff",
    }
    women_expected = {
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "harry_Rating_diff",
        "OpponentQualityTournamentRank_diff",
        "QualityWins_diff",
        "AvgBlkDiff_diff",
    }

    assert men_expected.issubset(men.columns)
    assert women_expected.issubset(women.columns)
    assert "InjuryAdjustedStrength_diff" not in women.columns
    assert "Seed_x_harry_Rating_diff" not in women.columns
