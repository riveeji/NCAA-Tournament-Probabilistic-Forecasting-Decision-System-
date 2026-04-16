import pytest
import pandas as pd

from hc.ji_base import JIBaseConfig
from hc.ji_base import data as ji_data
from hc.ji_base.data import build_ji_dataset, build_submission_feature_frame, load_ji_team_features
from hc.ji_base.internal_ratings import compute_internal_ratings_from_games
from hc.ji_base.women_upstream import build_women_ranking_upstream


def test_women_team_features_include_required_quality_and_alpha_columns():
    features = load_ji_team_features(JIBaseConfig(gender="W"))

    required = {
        "Season",
        "TeamID",
        "SeedNum",
        "Elo",
        "Quality",
        "WomenCompositeQuality",
        "harry_Rating",
        "QualityWins",
        "OpponentQualityTournamentRank",
        "AvgBlkDiff",
    }

    assert required.issubset(features.columns)
    assert features["WomenCompositeQuality"].notna().any()


def test_ji_dataset_contains_core_delta_features():
    dataset = build_ji_dataset(JIBaseConfig(gender="M"))

    required = {
        "Season",
        "Label",
        "Margin",
        "Delta_Seed",
        "Delta_Elo",
        "Delta_Quality",
        "Delta_neff",
        "strength_blend",
    }

    assert required.issubset(dataset.columns)
    assert dataset["Label"].isin([0, 1]).all()


def test_men_quality_is_unchanged_across_women_quality_profiles():
    legacy = load_ji_team_features(JIBaseConfig(gender="M", women_quality_profile="legacy_v1"))
    rebuilt = load_ji_team_features(JIBaseConfig(gender="M", women_quality_profile="consensus_rebuild_v4"))

    pd.testing.assert_series_equal(legacy["Quality"], rebuilt["Quality"], check_names=False)


def test_women_rebuild_quality_profile_changes_composite_quality():
    legacy = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="legacy_v1"))
    rebuilt = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4"))

    assert rebuilt["WomenCompositeQuality"].notna().any()
    assert not rebuilt["WomenCompositeQuality"].equals(legacy["WomenCompositeQuality"])


def test_women_quality_profiles_v4a_v4b_change_women_outputs_without_touching_men():
    women_v4 = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4"))
    women_v4a = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4a"))
    women_v4b = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4b"))
    men_v4 = load_ji_team_features(JIBaseConfig(gender="M", women_quality_profile="consensus_rebuild_v4"))
    men_v4a = load_ji_team_features(JIBaseConfig(gender="M", women_quality_profile="consensus_rebuild_v4a"))

    assert not women_v4["Quality"].equals(women_v4a["Quality"])
    assert not women_v4["harry_Rating"].equals(women_v4b["harry_Rating"])
    pd.testing.assert_series_equal(men_v4["Quality"], men_v4a["Quality"], check_names=False)


def test_women_ranking_upstream_internal_fallback_is_non_empty_and_stable():
    profile = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 3101, "PageRank": 1.2, "RPIStyle": 0.8, "SOS": 0.4, "wpct": 0.90},
            {"Season": 2026, "TeamID": 3102, "PageRank": 0.1, "RPIStyle": 0.2, "SOS": -0.1, "wpct": 0.70},
            {"Season": 2026, "TeamID": 3103, "PageRank": -0.7, "RPIStyle": -0.3, "SOS": -0.4, "wpct": 0.55},
        ]
    )

    bundle = build_women_ranking_upstream(profile, provider="internal_fallback")

    assert {
        "Season",
        "TeamID",
        "WomenConsensusRankScore",
        "WomenConsensusRankCoverage",
        "WomenConsensusRankConfidence",
    }.issubset(bundle.columns)
    assert bundle["WomenConsensusRankScore"].notna().all()
    assert bundle["WomenConsensusRankCoverage"].eq(0.0).all()
    assert bundle["WomenConsensusRankConfidence"].eq(0.0).all()


def test_women_ranking_upstream_external_provider_explicitly_falls_back_when_sources_are_missing():
    profile = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 3101, "PageRank": 1.2, "RPIStyle": 0.8, "SOS": 0.4, "wpct": 0.90},
            {"Season": 2026, "TeamID": 3102, "PageRank": 0.1, "RPIStyle": 0.2, "SOS": -0.1, "wpct": 0.70},
        ]
    )

    internal = build_women_ranking_upstream(profile, provider="internal_fallback")
    external = build_women_ranking_upstream(profile, provider="external_consensus_v1")

    pd.testing.assert_series_equal(
        internal["WomenConsensusRankScore"],
        external["WomenConsensusRankScore"],
        check_names=False,
    )
    assert external["WomenConsensusRankCoverage"].eq(0.0).all()
    assert external["WomenConsensusRankConfidence"].eq(0.0).all()


def test_women_ranking_upstream_external_v2_uses_weighted_coverage_and_thresholded_confidence():
    profile = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 3101,
                "PageRank": 1.0,
                "RPIStyle": 0.9,
                "SOS": 0.3,
                "wpct": 0.88,
                "Ext_PublicNETRank": 4,
                "Ext_PublicPredRPIRank": 5,
            },
            {
                "Season": 2026,
                "TeamID": 3102,
                "PageRank": 0.3,
                "RPIStyle": 0.2,
                "SOS": 0.1,
                "wpct": 0.69,
                "Ext_PublicRPIRank": 18,
                "Ext_WN_NET": 21,
                "Ext_WN_ELO": 17,
                "Ext_WN_RPI": 20,
                "Ext_WN_PredRPI": 19,
            },
            {
                "Season": 2026,
                "TeamID": 3103,
                "PageRank": -0.4,
                "RPIStyle": -0.3,
                "SOS": -0.2,
                "wpct": 0.55,
            },
        ]
    )

    bundle = build_women_ranking_upstream(profile, provider="external_consensus_v2")

    row_a = bundle.loc[bundle["TeamID"] == 3101].iloc[0]
    row_b = bundle.loc[bundle["TeamID"] == 3102].iloc[0]
    row_c = bundle.loc[bundle["TeamID"] == 3103].iloc[0]

    assert row_a["WomenConsensusRankCoverage"] == 2.0 / 5.0
    assert row_a["WomenConsensusRankConfidence"] == pytest.approx(0.3)
    assert row_b["WomenConsensusRankCoverage"] == 3.0 / 5.0
    assert row_b["WomenConsensusRankConfidence"] == pytest.approx(0.7)
    assert row_c["WomenConsensusRankCoverage"] == 0.0
    assert row_c["WomenConsensusRankConfidence"] == 0.0
    assert row_a["WomenConsensusRankCoverage"] != (2.0 / 7.0)
    assert row_b["WomenConsensusRankCoverage"] != (5.0 / 7.0)


def test_women_ranking_upstream_external_v2_falls_back_when_sources_are_missing():
    profile = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 3101, "PageRank": 1.2, "RPIStyle": 0.8, "SOS": 0.4, "wpct": 0.90},
            {"Season": 2026, "TeamID": 3102, "PageRank": 0.1, "RPIStyle": 0.2, "SOS": -0.1, "wpct": 0.70},
        ]
    )

    internal = build_women_ranking_upstream(profile, provider="internal_fallback")
    external = build_women_ranking_upstream(profile, provider="external_consensus_v2")

    pd.testing.assert_series_equal(
        internal["WomenConsensusRankScore"],
        external["WomenConsensusRankScore"],
        check_names=False,
    )
    assert external["WomenConsensusRankCoverage"].eq(0.0).all()
    assert external["WomenConsensusRankConfidence"].eq(0.0).all()


def test_women_ranking_upstream_historical_snapshots_provider_emits_non_empty_signal():
    profile = pd.DataFrame(
        [
            {"Season": 2025, "TeamID": 3163, "PageRank": 1.1, "RPIStyle": 0.9, "SOS": 0.4, "wpct": 0.90},
            {"Season": 2025, "TeamID": 3390, "PageRank": 0.2, "RPIStyle": 0.1, "SOS": 0.0, "wpct": 0.70},
            {"Season": 2025, "TeamID": 3199, "PageRank": -0.4, "RPIStyle": -0.3, "SOS": -0.2, "wpct": 0.55},
        ]
    )

    bundle = build_women_ranking_upstream(profile, provider="historical_consensus_snapshots_v1")

    assert bundle["WomenConsensusRankScore"].notna().all()
    assert bundle["WomenConsensusRankCoverage"].gt(0.0).any()
    assert bundle["WomenConsensusRankConfidence"].gt(0.0).any()


def test_quality_only_women_light_scales_women_quality_alpha_without_touching_men():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Season": [2026], "T1": [1101], "T2": [1102]})
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
                "harry_Rating": 21.0,
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
                "harry_Rating": 5.5,
                "QualityWins": -0.1,
                "OpponentQualityTournamentRank": -0.4,
                "AvgBlkDiff": -0.2,
            },
        ]
    )

    from hc.ji_base.data import build_submission_feature_frame

    women_quality_only = build_submission_feature_frame(ids, features, JIBaseConfig(gender="W", alpha_profile="quality_only"))
    women_light = build_submission_feature_frame(ids, features, JIBaseConfig(gender="W", alpha_profile="quality_only_women_light"))
    men_quality_only = build_submission_feature_frame(ids, features, JIBaseConfig(gender="M", alpha_profile="quality_only"))
    men_light = build_submission_feature_frame(ids, features, JIBaseConfig(gender="M", alpha_profile="quality_only_women_light"))

    assert women_light.loc[0, "QualityWins_diff"] < women_quality_only.loc[0, "QualityWins_diff"]
    assert women_light.loc[0, "OpponentQualityTournamentRank_diff"] < women_quality_only.loc[0, "OpponentQualityTournamentRank_diff"]
    assert men_light.loc[0, "QualityWins_diff"] == men_quality_only.loc[0, "QualityWins_diff"]
    assert men_light.loc[0, "OpponentQualityTournamentRank_diff"] == men_quality_only.loc[0, "OpponentQualityTournamentRank_diff"]


def test_women_quality_profile_v6_internal_fallback_preserves_v4_outputs():
    baseline = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4"))
    refactor = load_ji_team_features(
        JIBaseConfig(
            gender="W",
            women_quality_profile="consensus_rebuild_v6",
            women_ranking_provider="internal_fallback",
        )
    )

    pd.testing.assert_series_equal(baseline["Quality"], refactor["Quality"], check_names=False)
    pd.testing.assert_series_equal(
        baseline["OpponentQualityTournamentRank"],
        refactor["OpponentQualityTournamentRank"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        baseline["WomenCompositeQuality"],
        refactor["WomenCompositeQuality"],
        check_names=False,
    )
    assert {"WomenConsensusRankScore", "WomenConsensusRankCoverage", "WomenConsensusRankConfidence"}.issubset(refactor.columns)


def test_submission_feature_frame_builds_tossup_upset_features():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Season": [2026], "T1": [1101], "T2": [1102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 5,
                "Elo": 1600,
                "Quality": 1.2,
                "WomenCompositeQuality": 0.6,
                "oeff": 115.0,
                "deff": 95.0,
                "neff": 20.0,
                "efg": 0.56,
                "tor": 0.14,
                "orpct": 0.33,
                "ftr": 0.28,
                "pace": 69.0,
                "harry_Rating": 21.0,
                "QualityWins": 1.2,
                "OpponentQualityTournamentRank": 0.7,
                "AvgBlkDiff": 0.4,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 6,
                "Elo": 1575,
                "Quality": 0.9,
                "WomenCompositeQuality": 0.4,
                "oeff": 112.0,
                "deff": 97.0,
                "neff": 15.0,
                "efg": 0.54,
                "tor": 0.16,
                "orpct": 0.31,
                "ftr": 0.24,
                "pace": 68.0,
                "harry_Rating": 16.0,
                "QualityWins": 0.9,
                "OpponentQualityTournamentRank": 0.5,
                "AvgBlkDiff": 0.2,
            },
        ]
    )

    from hc.ji_base.data import build_submission_feature_frame

    frame = build_submission_feature_frame(ids, features, JIBaseConfig(gender="M", feature_profile="tossup_upset_v1"))

    assert "CloseGameStrength" in frame.columns
    assert "UpsetPressure" in frame.columns
    assert frame.loc[0, "CloseGameStrength"] != 0.0
    assert frame.loc[0, "UpsetPressure"] != 0.0


def test_ji_dataset_cache_respects_feature_profile():
    baseline = build_ji_dataset(
        JIBaseConfig(
            gender="W",
            alpha_profile="quality_only_men_quality_blocks_women",
            women_quality_profile="consensus_rebuild_v4",
            feature_profile="seed_quality_interaction",
        )
    )
    conservative = build_ji_dataset(
        JIBaseConfig(
            gender="W",
            alpha_profile="quality_only_men_quality_blocks_women",
            women_quality_profile="consensus_rebuild_v4",
            feature_profile="seed_quality_interaction_women_conservative",
        )
    )

    diff = (baseline["Seed_x_Quality"] - conservative["Seed_x_Quality"]).abs()
    assert diff.gt(0).any()


def test_ji_dataset_cache_respects_women_ranking_provider(monkeypatch):
    calls: list[str] = []

    def _fake_results(gender):
        return pd.DataFrame(
            [{"Season": 2026, "DayNum": 1, "T1": 3101, "T2": 3102, "Label": 1, "Margin": 5.0}]
        )

    def _fake_features(config):
        calls.append(config.women_ranking_provider)
        return pd.DataFrame(
            [
                {"Season": 2026, "TeamID": 3101, "SeedNum": 1, "Elo": 1600.0, "Quality": 1.0},
                {"Season": 2026, "TeamID": 3102, "SeedNum": 2, "Elo": 1500.0, "Quality": 0.5},
            ]
        )

    def _fake_submission(ids, features, config):
        return ids.assign(Delta_Seed=-1.0, Delta_Elo=100.0, Delta_Quality=0.5)

    ji_data._cached_ji_dataset.cache_clear()
    monkeypatch.setattr(ji_data, "load_tournament_results", _fake_results)
    monkeypatch.setattr(ji_data, "load_ji_team_features", _fake_features)
    monkeypatch.setattr(ji_data, "build_submission_feature_frame", _fake_submission)

    internal = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v6", women_ranking_provider="internal_fallback")
    external = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v6", women_ranking_provider="external_consensus_v1")
    external_v2 = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v6", women_ranking_provider="external_consensus_v2")

    build_ji_dataset(internal)
    build_ji_dataset(external)
    build_ji_dataset(external_v2)

    assert calls == ["internal_fallback", "external_consensus_v1", "external_consensus_v2"]


def test_internal_ratings_builder_returns_stable_columns_and_carryover():
    games = pd.DataFrame(
        [
            {"Season": 2025, "DayNum": 1, "WTeamID": 1, "LTeamID": 2, "WScore": 70, "LScore": 60, "WLoc": "N"},
            {"Season": 2025, "DayNum": 2, "WTeamID": 2, "LTeamID": 3, "WScore": 65, "LScore": 60, "WLoc": "N"},
            {"Season": 2026, "DayNum": 1, "WTeamID": 1, "LTeamID": 3, "WScore": 75, "LScore": 65, "WLoc": "N"},
        ]
    )
    teams = pd.DataFrame(
        [
            {"Season": 2025, "TeamID": 1},
            {"Season": 2025, "TeamID": 2},
            {"Season": 2025, "TeamID": 3},
            {"Season": 2026, "TeamID": 1},
            {"Season": 2026, "TeamID": 2},
            {"Season": 2026, "TeamID": 3},
        ]
    )

    ratings = compute_internal_ratings_from_games(games=games, teams=teams)

    assert {"Season", "TeamID", "CarryElo", "Colley", "SRS"}.issubset(ratings.columns)
    assert ratings[["CarryElo", "Colley", "SRS"]].notna().all().all()
    carry_2025_team1 = float(ratings.loc[(ratings["Season"] == 2025) & (ratings["TeamID"] == 1), "CarryElo"].iloc[0])
    carry_2026_team1 = float(ratings.loc[(ratings["Season"] == 2026) & (ratings["TeamID"] == 1), "CarryElo"].iloc[0])
    assert carry_2025_team1 != 1500.0
    assert carry_2026_team1 != 1500.0
    assert carry_2026_team1 != carry_2025_team1


def test_team_features_include_internal_rating_columns():
    features = load_ji_team_features(JIBaseConfig(gender="M"))

    required = {"CarryElo", "Colley", "SRS"}
    assert required.issubset(features.columns)
    assert features[list(required)].notna().all().all()


def test_lr_pruned_core_v1_dataset_contains_new_rating_diffs_and_interaction():
    men = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_pruned_core_v1", alpha_profile="quality_only_men_quality_blocks_women"))
    women = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_pruned_core_v1", alpha_profile="quality_only_men_quality_blocks_women"))

    assert {"Delta_CarryElo", "Delta_Colley", "Delta_SRS", "Seed_x_Colley"}.issubset(men.columns)
    assert {"Delta_CarryElo", "Delta_Colley", "Delta_SRS", "Seed_x_Colley"}.issubset(women.columns)
    assert "Seed_x_Quality" in men.columns
    assert "Seed_x_Quality" in women.columns


def test_submission_feature_frame_builds_seed_x_colley_for_lr_pruned_core_v1():
    ids = pd.DataFrame({"ID": ["2026_1101_1102"], "Season": [2026], "T1": [1101], "T2": [1102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 4,
                "Elo": 1600,
                "CarryElo": 1580,
                "Colley": 0.72,
                "SRS": 12.0,
                "Quality": 1.0,
                "WomenCompositeQuality": 0.7,
                "oeff": 115.0,
                "deff": 95.0,
                "neff": 20.0,
                "efg": 0.56,
                "tor": 0.14,
                "orpct": 0.33,
                "ftr": 0.28,
                "pace": 69.0,
                "harry_Rating": 21.0,
                "QualityWins": 1.2,
                "OpponentQualityTournamentRank": 0.7,
                "AvgBlkDiff": 0.4,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 9,
                "Elo": 1520,
                "CarryElo": 1510,
                "Colley": 0.48,
                "SRS": 3.0,
                "Quality": 0.2,
                "WomenCompositeQuality": 0.1,
                "oeff": 107.0,
                "deff": 101.0,
                "neff": 6.0,
                "efg": 0.51,
                "tor": 0.18,
                "orpct": 0.29,
                "ftr": 0.23,
                "pace": 66.0,
                "harry_Rating": 5.5,
                "QualityWins": -0.1,
                "OpponentQualityTournamentRank": -0.4,
                "AvgBlkDiff": -0.2,
            },
        ]
    )

    from hc.ji_base.data import build_submission_feature_frame

    frame = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(gender="M", feature_profile="lr_pruned_core_v1", alpha_profile="quality_only_men_quality_blocks_women"),
    )

    assert "Seed_x_Colley" in frame.columns
    assert frame.loc[0, "Seed_x_Colley"] == frame.loc[0, "Delta_Seed"] * frame.loc[0, "Delta_Colley"]


def test_team_features_include_eff_srs_column():
    features = load_ji_team_features(JIBaseConfig(gender="M"))

    assert "EffSRS" in features.columns
    assert features["EffSRS"].notna().all()


def test_internal_ratings_builder_returns_stronger_carry_elo_variant():
    games = pd.DataFrame(
        [
            {"Season": 2025, "DayNum": 1, "WTeamID": 1, "LTeamID": 2, "WScore": 70, "LScore": 60, "WLoc": "N"},
            {"Season": 2025, "DayNum": 2, "WTeamID": 1, "LTeamID": 3, "WScore": 72, "LScore": 61, "WLoc": "N"},
            {"Season": 2026, "DayNum": 1, "WTeamID": 1, "LTeamID": 3, "WScore": 75, "LScore": 65, "WLoc": "N"},
        ]
    )
    teams = pd.DataFrame(
        [
            {"Season": 2025, "TeamID": 1},
            {"Season": 2025, "TeamID": 2},
            {"Season": 2025, "TeamID": 3},
            {"Season": 2026, "TeamID": 1},
            {"Season": 2026, "TeamID": 2},
            {"Season": 2026, "TeamID": 3},
        ]
    )

    ratings = compute_internal_ratings_from_games(games=games, teams=teams)

    assert "CarryElo85" in ratings.columns
    carry_2026 = float(ratings.loc[(ratings["Season"] == 2026) & (ratings["TeamID"] == 1), "CarryElo"].iloc[0])
    carry85_2026 = float(ratings.loc[(ratings["Season"] == 2026) & (ratings["TeamID"] == 1), "CarryElo85"].iloc[0])
    assert abs(carry85_2026 - 1500.0) > abs(carry_2026 - 1500.0)
    assert "CarryElo80" in ratings.columns
    carry80_2026 = float(ratings.loc[(ratings["Season"] == 2026) & (ratings["TeamID"] == 1), "CarryElo80"].iloc[0])
    assert abs(carry80_2026 - 1500.0) > abs(carry_2026 - 1500.0)
    assert abs(carry85_2026 - 1500.0) > abs(carry80_2026 - 1500.0)


def test_internal_ratings_builder_returns_conference_downweighted_colley_variant():
    games = pd.DataFrame(
        [
            {"Season": 2026, "DayNum": 1, "WTeamID": 1, "LTeamID": 2, "WScore": 70, "LScore": 60, "WLoc": "N"},
            {"Season": 2026, "DayNum": 2, "WTeamID": 1, "LTeamID": 3, "WScore": 68, "LScore": 60, "WLoc": "N"},
            {"Season": 2026, "DayNum": 3, "WTeamID": 4, "LTeamID": 2, "WScore": 66, "LScore": 61, "WLoc": "N"},
        ]
    )
    teams = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1},
            {"Season": 2026, "TeamID": 2},
            {"Season": 2026, "TeamID": 3},
            {"Season": 2026, "TeamID": 4},
        ]
    )
    conferences = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1, "ConfAbbrev": "alpha"},
            {"Season": 2026, "TeamID": 2, "ConfAbbrev": "alpha"},
            {"Season": 2026, "TeamID": 3, "ConfAbbrev": "beta"},
            {"Season": 2026, "TeamID": 4, "ConfAbbrev": "gamma"},
        ]
    )

    ratings = compute_internal_ratings_from_games(games=games, teams=teams, team_conferences=conferences)

    assert "ColleyNC" in ratings.columns
    assert ratings["ColleyNC"].notna().all()
    assert not ratings["ColleyNC"].equals(ratings["Colley"])


def test_lr_ratings_definition_v1_dataset_contains_eff_srs_diff():
    men = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_ratings_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))
    women = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_ratings_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))

    assert "Delta_EffSRS" in men.columns
    assert "Delta_EffSRS" in women.columns
    assert "Delta_SRS" in men.columns
    assert "Delta_SRS" in women.columns


def test_lr_carry_elo_definition_v1_dataset_contains_stronger_carry_diff():
    men = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_carry_elo_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))
    women = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_carry_elo_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))

    assert "Delta_CarryElo85" in men.columns
    assert "Delta_CarryElo85" in women.columns
    assert "Delta_CarryElo" in men.columns
    assert "Delta_CarryElo" in women.columns


def test_lr_carry_elo_definition_confirm80_dataset_contains_midpoint_carry_diff():
    men = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_carry_elo_definition_confirm80", alpha_profile="quality_only_men_quality_blocks_women"))
    women = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_carry_elo_definition_confirm80", alpha_profile="quality_only_men_quality_blocks_women"))

    assert "Delta_CarryElo80" in men.columns
    assert "Delta_CarryElo80" in women.columns
    assert "Delta_CarryElo85" in men.columns
    assert "Delta_CarryElo85" in women.columns


def test_lr_colley_definition_v1_dataset_contains_conference_downweighted_colley_diff():
    men = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_colley_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))
    women = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_colley_definition_v1", alpha_profile="quality_only_men_quality_blocks_women"))

    assert "Delta_ColleyNC" in men.columns
    assert "Delta_ColleyNC" in women.columns
    assert "Delta_Colley" in men.columns
    assert "Delta_Colley" in women.columns
    assert "Seed_x_ColleyNC" in men.columns
    assert "Seed_x_ColleyNC" in women.columns


def test_internal_ratings_builder_returns_clipped_srs_variants():
    games = pd.DataFrame(
        [
            {"Season": 2026, "DayNum": 1, "WTeamID": 1, "LTeamID": 2, "WScore": 95, "LScore": 60, "WLoc": "N"},
            {"Season": 2026, "DayNum": 2, "WTeamID": 3, "LTeamID": 1, "WScore": 76, "LScore": 74, "WLoc": "N"},
            {"Season": 2026, "DayNum": 3, "WTeamID": 2, "LTeamID": 3, "WScore": 80, "LScore": 65, "WLoc": "N"},
        ]
    )
    teams = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1},
            {"Season": 2026, "TeamID": 2},
            {"Season": 2026, "TeamID": 3},
        ]
    )

    ratings = compute_internal_ratings_from_games(games=games, teams=teams)

    assert {"SRSClip15", "SRSClip20"}.issubset(ratings.columns)
    assert ratings["SRSClip15"].notna().all()
    assert ratings["SRSClip20"].notna().all()
    assert not ratings["SRSClip15"].equals(ratings["SRS"])


def test_lr_srs_definition_profiles_dataset_contains_clipped_srs_diffs():
    clip15_m = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_srs_definition_v1_clip15", alpha_profile="quality_only_men_quality_blocks_women"))
    clip15_w = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_srs_definition_v1_clip15", alpha_profile="quality_only_men_quality_blocks_women"))
    clip20_m = build_ji_dataset(JIBaseConfig(gender="M", feature_profile="lr_srs_definition_confirm20", alpha_profile="quality_only_men_quality_blocks_women"))
    clip20_w = build_ji_dataset(JIBaseConfig(gender="W", feature_profile="lr_srs_definition_confirm20", alpha_profile="quality_only_men_quality_blocks_women"))

    assert "Delta_SRSClip15" in clip15_m.columns
    assert "Delta_SRSClip15" in clip15_w.columns
    assert "Delta_SRS" in clip15_m.columns
    assert "Delta_SRS" in clip15_w.columns
    assert "Delta_SRSClip20" in clip20_m.columns
    assert "Delta_SRSClip20" in clip20_w.columns


def test_women_tossup_quality_conservative_only_shrinks_small_gap_quality_features():
    ids = pd.DataFrame(
        [
            {"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102},
            {"ID": "2026_1103_1104", "Season": 2026, "T1": 1103, "T2": 1104},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 1101,
                "SeedNum": 5,
                "Elo": 1600,
                "Quality": 1.0,
                "WomenCompositeQuality": 0.7,
                "oeff": 115.0,
                "deff": 95.0,
                "neff": 20.0,
                "efg": 0.56,
                "tor": 0.14,
                "orpct": 0.33,
                "ftr": 0.28,
                "pace": 69.0,
                "harry_Rating": 21.0,
                "QualityWins": 1.1,
                "OpponentQualityTournamentRank": 0.8,
                "AvgBlkDiff": 0.4,
            },
            {
                "Season": 2026,
                "TeamID": 1102,
                "SeedNum": 6,
                "Elo": 1575,
                "Quality": 0.6,
                "WomenCompositeQuality": 0.5,
                "oeff": 112.0,
                "deff": 97.0,
                "neff": 15.0,
                "efg": 0.54,
                "tor": 0.16,
                "orpct": 0.31,
                "ftr": 0.24,
                "pace": 68.0,
                "harry_Rating": 16.0,
                "QualityWins": 0.7,
                "OpponentQualityTournamentRank": 0.5,
                "AvgBlkDiff": 0.2,
            },
            {
                "Season": 2026,
                "TeamID": 1103,
                "SeedNum": 2,
                "Elo": 1675,
                "Quality": 1.5,
                "WomenCompositeQuality": 1.1,
                "oeff": 118.0,
                "deff": 92.0,
                "neff": 26.0,
                "efg": 0.58,
                "tor": 0.13,
                "orpct": 0.35,
                "ftr": 0.30,
                "pace": 70.0,
                "harry_Rating": 26.0,
                "QualityWins": 1.4,
                "OpponentQualityTournamentRank": 1.0,
                "AvgBlkDiff": 0.5,
            },
            {
                "Season": 2026,
                "TeamID": 1104,
                "SeedNum": 11,
                "Elo": 1490,
                "Quality": -0.2,
                "WomenCompositeQuality": -0.1,
                "oeff": 106.0,
                "deff": 101.0,
                "neff": 5.0,
                "efg": 0.50,
                "tor": 0.19,
                "orpct": 0.28,
                "ftr": 0.22,
                "pace": 65.0,
                "harry_Rating": 4.0,
                "QualityWins": 0.1,
                "OpponentQualityTournamentRank": -0.2,
                "AvgBlkDiff": -0.3,
            },
        ]
    )

    from hc.ji_base.data import build_submission_feature_frame

    base = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(gender="W", feature_profile="seed_quality_interaction", alpha_profile="quality_only_men_quality_blocks_women"),
    )
    conservative = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(
            gender="W",
            feature_profile="women_tossup_quality_conservative",
            alpha_profile="quality_only_men_quality_blocks_women",
        ),
    )

    assert abs(conservative.loc[0, "QualityWins_diff"]) < abs(base.loc[0, "QualityWins_diff"])
    assert abs(conservative.loc[0, "OpponentQualityTournamentRank_diff"]) < abs(base.loc[0, "OpponentQualityTournamentRank_diff"])
    assert conservative.loc[1, "QualityWins_diff"] == base.loc[1, "QualityWins_diff"]
    assert conservative.loc[1, "OpponentQualityTournamentRank_diff"] == base.loc[1, "OpponentQualityTournamentRank_diff"]


def test_consensus_rebuild_v5_outputs_component_columns():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "TeamID": [3101, 3102],
            "SeedNum": [2, 7],
            "Elo": [1620.0, 1510.0],
            "wpct": [0.88, 0.67],
            "neff": [24.0, 11.0],
            "margin": [18.0, 7.0],
            "SOS": [4.0, 1.2],
            "RPIStyle": [0.9, 0.3],
            "PageRank": [0.8, 0.2],
            "Top50WinRate": [0.70, 0.35],
            "Top100WinRate": [0.92, 0.60],
            "ConfMeanElo": [1540.0, 1490.0],
            "AvgBlkDiff": [1.8, 0.4],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v5")

    required = {
        "WomenSeedStrength",
        "WomenQualityWinsStrength",
        "WomenOpponentTournamentStrength",
        "WomenRimProtectionStrength",
        "WomenCompositeQualityV5",
    }
    assert required.issubset(set(out.columns))
    assert out["WomenCompositeQualityV5"].notna().all()


def test_consensus_rebuild_v5_preserves_season_team_key():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026],
            "TeamID": [3101],
            "SeedNum": [3],
            "Elo": [1600.0],
            "wpct": [0.80],
            "neff": [18.0],
            "margin": [12.0],
            "SOS": [2.5],
            "RPIStyle": [0.7],
            "PageRank": [0.6],
            "Top50WinRate": [0.50],
            "Top100WinRate": [0.75],
            "ConfMeanElo": [1525.0],
            "AvgBlkDiff": [1.1],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v5")

    assert list(out[["Season", "TeamID"]].iloc[0]) == [2026, 3101]


def test_team_features_include_women_slice_redesign_component_columns():
    features = load_ji_team_features(JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v5"))

    required = {
        "WomenSeedStrength",
        "WomenQualityWinsStrength",
        "WomenOpponentTournamentStrength",
        "WomenRimProtectionStrength",
        "WomenCompositeQualityV5",
    }

    assert required.issubset(features.columns)
    assert features["WomenCompositeQualityV5"].notna().any()


def test_submission_feature_frame_builds_women_slice_redesign_columns_without_touching_men():
    ids = pd.DataFrame({"ID": ["2026_3101_3102"], "Season": [2026], "T1": [3101], "T2": [3102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 3101,
                "SeedNum": 2,
                "Elo": 1620.0,
                "CarryElo85": 1605.0,
                "Colley": 0.75,
                "SRS": 14.0,
                "Quality": 1.4,
                "WomenCompositeQuality": 0.9,
                "WomenSeedStrength": 1.2,
                "WomenQualityWinsStrength": 1.4,
                "WomenOpponentTournamentStrength": 1.1,
                "WomenRimProtectionStrength": 0.8,
                "WomenCompositeQualityV5": 1.15,
                "oeff": 116.0,
                "deff": 93.0,
                "neff": 23.0,
                "efg": 0.57,
                "tor": 0.14,
                "orpct": 0.34,
                "ftr": 0.27,
                "pace": 69.0,
                "harry_Rating": 20.0,
                "QualityWins": 1.0,
                "OpponentQualityTournamentRank": 0.8,
                "AvgBlkDiff": 0.6,
            },
            {
                "Season": 2026,
                "TeamID": 3102,
                "SeedNum": 7,
                "Elo": 1520.0,
                "CarryElo85": 1510.0,
                "Colley": 0.50,
                "SRS": 5.0,
                "Quality": 0.4,
                "WomenCompositeQuality": 0.2,
                "WomenSeedStrength": -0.7,
                "WomenQualityWinsStrength": 0.2,
                "WomenOpponentTournamentStrength": 0.1,
                "WomenRimProtectionStrength": -0.3,
                "WomenCompositeQualityV5": -0.1,
                "oeff": 108.0,
                "deff": 100.0,
                "neff": 8.0,
                "efg": 0.52,
                "tor": 0.17,
                "orpct": 0.30,
                "ftr": 0.23,
                "pace": 67.0,
                "harry_Rating": 7.0,
                "QualityWins": 0.3,
                "OpponentQualityTournamentRank": 0.1,
                "AvgBlkDiff": -0.1,
            },
        ]
    )

    women_frame = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(gender="W", feature_profile="women_slice_redesign_v1_architecture", women_quality_profile="consensus_rebuild_v5"),
    )
    men_frame = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(gender="M", feature_profile="women_slice_redesign_v1_architecture", women_quality_profile="legacy_v1"),
    )

    assert "Delta_WomenCompositeQualityV5" in women_frame.columns
    assert "Delta_WomenQualityWinsStrength" in women_frame.columns
    assert "Delta_WomenOpponentTournamentStrength" in women_frame.columns
    assert "Delta_WomenRimProtectionStrength" in women_frame.columns
    assert "Seed_x_WomenOpponentTournamentStrength" in women_frame.columns
    assert "Delta_WomenCompositeQualityV5" in men_frame.columns
    assert "Delta_WomenCompositeQualityV5" not in JIBaseConfig(
        gender="M",
        feature_profile="women_slice_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    ).resolved_model_features()


def test_women_opp_rank_redesign_outputs_v2_team_column():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "TeamID": [3101, 3102],
            "SeedNum": [2, 7],
            "Elo": [1620.0, 1510.0],
            "wpct": [0.88, 0.67],
            "neff": [24.0, 11.0],
            "margin": [18.0, 7.0],
            "SOS": [4.0, 1.2],
            "RPIStyle": [0.9, 0.3],
            "PageRank": [0.8, 0.2],
            "Top50WinRate": [0.70, 0.35],
            "Top100WinRate": [0.92, 0.60],
            "ConfMeanElo": [1540.0, 1490.0],
            "AvgBlkDiff": [1.8, 0.4],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v4")

    assert "WomenOpponentTournamentStrengthV2" in out.columns
    assert out["WomenOpponentTournamentStrengthV2"].notna().all()


def test_women_opp_rank_redesign_does_not_change_quality_wins_column_shape():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026],
            "TeamID": [3101],
            "SeedNum": [3],
            "Elo": [1600.0],
            "wpct": [0.80],
            "neff": [18.0],
            "margin": [12.0],
            "SOS": [2.5],
            "RPIStyle": [0.7],
            "PageRank": [0.6],
            "Top50WinRate": [0.50],
            "Top100WinRate": [0.75],
            "ConfMeanElo": [1525.0],
            "AvgBlkDiff": [1.1],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v4")

    assert "QualityWins" in out.columns
    assert out["QualityWins"].notna().all()


def test_submission_feature_frame_builds_women_opp_rank_redesign_columns():
    ids = pd.DataFrame({"ID": ["2026_3101_3102"], "Season": [2026], "T1": [3101], "T2": [3102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 3101,
                "SeedNum": 2,
                "Elo": 1620.0,
                "CarryElo85": 1605.0,
                "Colley": 0.75,
                "SRS": 14.0,
                "Quality": 1.4,
                "WomenOpponentTournamentStrengthV2": 1.1,
                "neff": 23.0,
                "QualityWins": 1.0,
                "AvgBlkDiff": 0.6,
            },
            {
                "Season": 2026,
                "TeamID": 3102,
                "SeedNum": 7,
                "Elo": 1520.0,
                "CarryElo85": 1510.0,
                "Colley": 0.50,
                "SRS": 5.0,
                "Quality": 0.4,
                "WomenOpponentTournamentStrengthV2": 0.1,
                "neff": 8.0,
                "QualityWins": 0.3,
                "AvgBlkDiff": -0.1,
            },
        ]
    )

    frame = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(gender="W", feature_profile="women_opp_rank_redesign_v1_architecture", women_quality_profile="consensus_rebuild_v4"),
    )

    assert "Delta_WomenOpponentTournamentStrengthV2" in frame.columns
    assert "Seed_x_WomenOpponentTournamentStrengthV2" in frame.columns


def test_men_feature_profile_does_not_consume_women_opp_rank_v2():
    men = JIBaseConfig(
        gender="M",
        feature_profile="women_opp_rank_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    )

    assert "Delta_WomenOpponentTournamentStrengthV2" not in men.resolved_model_features()


def test_women_qualitywins_redesign_outputs_v2_team_column():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "TeamID": [3101, 3102],
            "SeedNum": [2, 7],
            "Elo": [1620.0, 1510.0],
            "wpct": [0.88, 0.67],
            "neff": [24.0, 11.0],
            "margin": [18.0, 7.0],
            "SOS": [4.0, 1.2],
            "RPIStyle": [0.9, 0.3],
            "PageRank": [0.8, 0.2],
            "Top50WinRate": [0.70, 0.35],
            "Top100WinRate": [0.92, 0.60],
            "ConfMeanElo": [1540.0, 1490.0],
            "AvgBlkDiff": [1.8, 0.4],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v4")

    assert "WomenQualityWinsStrengthV2" in out.columns
    assert out["WomenQualityWinsStrengthV2"].notna().all()


def test_women_qualitywins_redesign_does_not_change_quality_wins_column_shape():
    from hc.ji_base.ratings import build_ji_ratings

    teams = pd.DataFrame(
        {
            "Season": [2026],
            "TeamID": [3101],
            "SeedNum": [3],
            "Elo": [1600.0],
            "wpct": [0.80],
            "neff": [18.0],
            "margin": [12.0],
            "SOS": [2.5],
            "RPIStyle": [0.7],
            "PageRank": [0.6],
            "Top50WinRate": [0.50],
            "Top100WinRate": [0.75],
            "ConfMeanElo": [1525.0],
            "AvgBlkDiff": [1.1],
        }
    )

    out = build_ji_ratings(teams, gender="W", women_quality_profile="consensus_rebuild_v4")

    assert "QualityWins" in out.columns
    assert out["QualityWins"].notna().all()


def test_submission_feature_frame_builds_women_qualitywins_redesign_columns():
    ids = pd.DataFrame({"ID": ["2026_3101_3102"], "Season": [2026], "T1": [3101], "T2": [3102]})
    features = pd.DataFrame(
        [
            {
                "Season": 2026,
                "TeamID": 3101,
                "SeedNum": 2,
                "Elo": 1620.0,
                "CarryElo85": 1605.0,
                "Colley": 0.75,
                "SRS": 14.0,
                "Quality": 1.4,
                "WomenQualityWinsStrengthV2": 1.1,
                "OpponentQualityTournamentRank": 0.9,
                "neff": 23.0,
                "QualityWins": 1.0,
                "AvgBlkDiff": 0.6,
            },
            {
                "Season": 2026,
                "TeamID": 3102,
                "SeedNum": 7,
                "Elo": 1520.0,
                "CarryElo85": 1510.0,
                "Colley": 0.30,
                "SRS": 5.0,
                "Quality": 0.4,
                "WomenQualityWinsStrengthV2": 0.1,
                "OpponentQualityTournamentRank": 0.2,
                "neff": 8.0,
                "QualityWins": 0.3,
                "AvgBlkDiff": -0.1,
            },
        ]
    )

    frame = build_submission_feature_frame(
        ids,
        features,
        JIBaseConfig(
            gender="W",
            feature_profile="women_qualitywins_redesign_v1_with_seed_interaction",
            women_quality_profile="consensus_rebuild_v4",
        ),
    )

    assert "Delta_WomenQualityWinsStrengthV2" in frame.columns
    assert "Seed_x_WomenQualityWinsStrengthV2" in frame.columns


def test_men_feature_profile_does_not_consume_women_qualitywins_v2():
    men = JIBaseConfig(
        gender="M",
        feature_profile="women_qualitywins_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    )

    assert "Delta_WomenQualityWinsStrengthV2" not in men.resolved_model_features()
