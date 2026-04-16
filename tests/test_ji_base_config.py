from hc.ji_base import JIBaseConfig, build_ji_base_overlay_config, build_working_ji_base_config, build_working_ji_base_overlay_config
from hc.ji_base.config import MODEL_FAMILIES, WOMEN_RANKING_PROVIDERS


def test_model_families_include_ji_node_control():
    assert "JI_node_control" in MODEL_FAMILIES


def test_quality_only_alpha_profile_uses_only_quality_alpha_columns():
    config = JIBaseConfig(gender="M", alpha_profile="quality_only")

    assert config.resolved_model_features()[-2:] == ["QualityWins_diff", "OpponentQualityTournamentRank_diff"]
    assert "harry_Rating_diff" not in config.resolved_model_features()


def test_quality_only_women_light_uses_same_feature_columns_as_quality_only():
    quality_only = JIBaseConfig(gender="W", alpha_profile="quality_only")
    women_light = JIBaseConfig(gender="W", alpha_profile="quality_only_women_light")

    assert women_light.resolved_model_features() == quality_only.resolved_model_features()


def test_quality_only_men_core_women_is_gender_asymmetric_by_design():
    men = JIBaseConfig(gender="M", alpha_profile="quality_only_men_core_women")
    women = JIBaseConfig(gender="W", alpha_profile="quality_only_men_core_women")

    assert men.resolved_model_features()[-2:] == ["QualityWins_diff", "OpponentQualityTournamentRank_diff"]
    assert "harry_Rating_diff" not in men.resolved_model_features()
    assert "harry_Rating_diff" in women.resolved_model_features()
    assert "AvgBlkDiff_diff" in women.resolved_model_features()


def test_quality_only_men_women_combo_profiles_expand_expected_columns():
    men = JIBaseConfig(gender="M", alpha_profile="quality_only_men_quality_blocks_women")
    men_quality_only = JIBaseConfig(gender="M", alpha_profile="quality_wins_only_men_quality_blocks_women")
    men_opp_only = JIBaseConfig(gender="M", alpha_profile="opp_rank_only_men_quality_blocks_women")
    women_quality_blocks = JIBaseConfig(gender="W", alpha_profile="quality_only_men_quality_blocks_women")
    women_harry_quality = JIBaseConfig(gender="W", alpha_profile="quality_only_men_harry_quality_women")
    women_harry_blocks = JIBaseConfig(gender="W", alpha_profile="quality_only_men_harry_blocks_women")

    assert men.resolved_model_features()[-2:] == ["QualityWins_diff", "OpponentQualityTournamentRank_diff"]
    assert "harry_Rating_diff" not in men.resolved_model_features()
    assert "AvgBlkDiff_diff" not in men.resolved_model_features()

    assert men_quality_only.resolved_model_features()[-1:] == ["QualityWins_diff"]
    assert "OpponentQualityTournamentRank_diff" not in men_quality_only.resolved_model_features()

    assert men_opp_only.resolved_model_features()[-1:] == ["OpponentQualityTournamentRank_diff"]
    assert "QualityWins_diff" not in men_opp_only.resolved_model_features()

    assert "QualityWins_diff" in women_quality_blocks.resolved_model_features()
    assert "OpponentQualityTournamentRank_diff" in women_quality_blocks.resolved_model_features()
    assert "AvgBlkDiff_diff" in women_quality_blocks.resolved_model_features()
    assert "harry_Rating_diff" not in women_quality_blocks.resolved_model_features()

    assert "harry_Rating_diff" in women_harry_quality.resolved_model_features()
    assert "QualityWins_diff" in women_harry_quality.resolved_model_features()
    assert "OpponentQualityTournamentRank_diff" in women_harry_quality.resolved_model_features()
    assert "AvgBlkDiff_diff" not in women_harry_quality.resolved_model_features()

    assert "harry_Rating_diff" in women_harry_blocks.resolved_model_features()
    assert "AvgBlkDiff_diff" in women_harry_blocks.resolved_model_features()
    assert "QualityWins_diff" not in women_harry_blocks.resolved_model_features()


def test_seed_women_consensus_feature_profile_adds_interaction_only_for_women():
    women = JIBaseConfig(gender="W", feature_profile="seed_women_consensus_interaction")
    men = JIBaseConfig(gender="M", feature_profile="seed_women_consensus_interaction")

    assert "Seed_x_WomenConsensusQuality" in women.resolved_model_features()
    assert "Seed_x_WomenConsensusQuality" not in men.resolved_model_features()


def test_seed_quality_plus_women_consensus_keeps_seed_quality_for_both_and_adds_consensus_only_for_women():
    women = JIBaseConfig(gender="W", feature_profile="seed_quality_plus_women_consensus")
    men = JIBaseConfig(gender="M", feature_profile="seed_quality_plus_women_consensus")

    assert "Seed_x_Quality" in women.resolved_model_features()
    assert "Seed_x_Quality" in men.resolved_model_features()
    assert "Seed_x_WomenConsensusQuality" in women.resolved_model_features()
    assert "Seed_x_WomenConsensusQuality" not in men.resolved_model_features()


def test_tossup_upset_feature_profile_adds_close_game_features():
    men = JIBaseConfig(gender="M", feature_profile="tossup_upset_v1")
    women = JIBaseConfig(gender="W", feature_profile="tossup_upset_v1")

    assert "CloseGameStrength" in men.resolved_model_features()
    assert "UpsetPressure" in men.resolved_model_features()
    assert "CloseGameStrength" in women.resolved_model_features()
    assert "UpsetPressure" in women.resolved_model_features()


def test_women_quality_profile_v5_has_distinct_rating_profile():
    women = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v5")
    men = JIBaseConfig(gender="M", women_quality_profile="consensus_rebuild_v5")

    assert women.resolved_rating_profile() == "ji_quality_elo_v5_women_consensus_more_shrunk"
    assert men.resolved_rating_profile() == "ji_quality_elo_v1"


def test_women_quality_profile_v4a_v4b_have_distinct_rating_profiles():
    women_v4a = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4a")
    women_v4b = JIBaseConfig(gender="W", women_quality_profile="consensus_rebuild_v4b")

    assert women_v4a.resolved_rating_profile() == "ji_quality_elo_v4a_women_consensus_more_conservative"
    assert women_v4b.resolved_rating_profile() == "ji_quality_elo_v4b_women_harry_more_conservative"


def test_women_quality_profile_v6_and_ranking_provider_expose_hybrid_ready_upstream():
    women = JIBaseConfig(
        gender="W",
        women_quality_profile="consensus_rebuild_v6",
        women_ranking_provider="external_consensus_v1",
    )
    working_women = build_working_ji_base_config("W")
    working_men = build_working_ji_base_config("M")

    assert "internal_fallback" in WOMEN_RANKING_PROVIDERS
    assert "external_consensus_v1" in WOMEN_RANKING_PROVIDERS
    assert "external_consensus_v2" in WOMEN_RANKING_PROVIDERS
    assert "historical_consensus_snapshots_v1" in WOMEN_RANKING_PROVIDERS
    assert women.resolved_rating_profile() == "ji_quality_elo_v6_women_upstream_consensus"
    assert working_women.women_quality_profile == "consensus_rebuild_v4"
    assert working_women.women_ranking_provider == "internal_fallback"
    assert working_men.women_ranking_provider == "internal_fallback"


def test_working_overlay_config_is_men_market_injury_and_women_market_only():
    men = build_working_ji_base_overlay_config("M")
    women = build_working_ji_base_overlay_config("W")

    assert men.overlay_source_profile == "direct_only"
    assert women.overlay_source_profile == "direct_only"
    assert men.allow_market is True
    assert women.allow_market is True
    assert men.allow_injury is True
    assert women.allow_injury is False
    assert men.injury_min_confirmed_out == 4
    assert women.direct_weight == 0.25
    assert men.resolved_overlay_stack() == "market_injury"
    assert women.resolved_overlay_stack() == "market_only"


def test_conservative_overlay_profile_only_shrinks_men_injury_cap():
    baseline_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1")
    conservative_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_conservative_injury")
    baseline_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1")
    conservative_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_conservative_injury")

    assert conservative_men.injury_cap < baseline_men.injury_cap
    assert conservative_men.overlay_source_profile == baseline_men.overlay_source_profile
    assert conservative_men.direct_weight == baseline_men.direct_weight
    assert conservative_women.injury_cap == baseline_women.injury_cap
    assert conservative_women.allow_injury is False


def test_direct_only_overlay_profile_switches_market_source_without_changing_injury_policy():
    baseline_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1")
    direct_only_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only")
    baseline_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1")
    direct_only_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only")

    assert direct_only_men.overlay_source_profile == "direct_only"
    assert direct_only_men.allow_injury is True
    assert direct_only_men.injury_cap == baseline_men.injury_cap
    assert direct_only_women.overlay_source_profile == "direct_only"
    assert direct_only_women.allow_injury is False
    assert direct_only_women.injury_cap == baseline_women.injury_cap


def test_direct_only_strict_confirmed_overlay_profile_raises_men_injury_trigger_without_affecting_women():
    baseline_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only")
    strict_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_strict_confirmed")
    baseline_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only")
    strict_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only_injury_strict_confirmed")

    assert strict_men.overlay_source_profile == "direct_only"
    assert strict_men.allow_injury is True
    assert strict_men.injury_cap == baseline_men.injury_cap
    assert strict_men.injury_min_confirmed_out == 2

    assert strict_women.overlay_source_profile == "direct_only"
    assert strict_women.allow_injury is False
    assert strict_women.injury_min_confirmed_out == baseline_women.injury_min_confirmed_out


def test_direct_only_confirmed3_overlay_profile_uses_even_stricter_men_injury_threshold():
    confirmed2_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_strict_confirmed")
    confirmed3_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed3")
    confirmed3_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only_injury_confirmed3")

    assert confirmed3_men.overlay_source_profile == "direct_only"
    assert confirmed3_men.injury_min_confirmed_out == 3
    assert confirmed3_men.injury_min_confirmed_out > confirmed2_men.injury_min_confirmed_out
    assert confirmed3_women.allow_injury is False


def test_direct_only_confirmed4_overlay_profile_uses_strictest_men_injury_threshold():
    confirmed3_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed3")
    confirmed4_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed4")
    confirmed4_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only_injury_confirmed4")

    assert confirmed4_men.overlay_source_profile == "direct_only"
    assert confirmed4_men.injury_min_confirmed_out == 4
    assert confirmed4_men.injury_min_confirmed_out > confirmed3_men.injury_min_confirmed_out
    assert confirmed4_women.allow_injury is False


def test_direct_only_confirmed5_overlay_profile_uses_more_restrictive_men_injury_threshold():
    confirmed4_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed4")
    confirmed5_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed5")
    confirmed5_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only_injury_confirmed5")

    assert confirmed5_men.overlay_source_profile == "direct_only"
    assert confirmed5_men.injury_min_confirmed_out == 5
    assert confirmed5_men.injury_min_confirmed_out > confirmed4_men.injury_min_confirmed_out
    assert confirmed5_women.allow_injury is False


def test_direct_only_confirmed4_shift008_overlay_profile_adds_abs_shift_gate():
    confirmed4_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed4")
    shift_gate_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008")
    shift_gate_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008")

    assert shift_gate_men.overlay_source_profile == "direct_only"
    assert shift_gate_men.injury_min_confirmed_out == confirmed4_men.injury_min_confirmed_out == 4
    assert shift_gate_men.injury_min_abs_shift == 0.08
    assert shift_gate_women.allow_injury is False


def test_men_best_women_direct_priority_overlay_profile_is_asymmetric_only_on_women_source():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_priority")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_priority")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_women.overlay_source_profile == "direct_priority"
    assert profile_women.allow_injury is False
    assert profile_women.direct_weight == 0.85


def test_men_best_women_direct_only_weight070_overlay_profile_only_changes_women_weight():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight070")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight070")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.70
    assert profile_women.allow_injury is False


def test_men_best_women_direct_only_weight060_overlay_profile_only_changes_women_weight_more_conservatively():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight060")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight060")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.60
    assert profile_women.allow_injury is False


def test_player_level_injury_v2_overlay_profile_keeps_frozen_women_and_switches_only_men_injury_mode():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v2_men_player_injury_weight025")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v2_men_player_injury_weight025")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_mode == "player_level_v2"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.25
    assert profile_women.allow_injury is False


def test_lr_pruned_core_v1_feature_profile_has_gender_asymmetric_fixed_feature_set():
    men = JIBaseConfig(gender="M", feature_profile="lr_pruned_core_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women = JIBaseConfig(gender="W", feature_profile="lr_pruned_core_v1", alpha_profile="quality_only_men_quality_blocks_women")

    assert men.resolved_model_features() == [
        "Delta_Seed",
        "Seed_gap_abs",
        "Delta_Elo",
        "Delta_CarryElo",
        "Delta_Quality",
        "Delta_neff",
        "Delta_Colley",
        "Delta_SRS",
        "QualityWins_diff",
        "OpponentQualityTournamentRank_diff",
        "Seed_x_Quality",
        "Seed_x_Colley",
    ]
    assert women.resolved_model_features() == [
        "Delta_Seed",
        "Seed_gap_abs",
        "Delta_Elo",
        "Delta_CarryElo",
        "Delta_Quality",
        "Delta_Colley",
        "Delta_SRS",
        "QualityWins_diff",
        "OpponentQualityTournamentRank_diff",
        "AvgBlkDiff_diff",
        "Seed_x_Colley",
    ]


def test_lr_attribution_feature_profiles_have_fixed_isolated_feature_sets():
    pruned_m = JIBaseConfig(gender="M", feature_profile="lr_pruned_only_v1", alpha_profile="quality_only_men_quality_blocks_women")
    pruned_w = JIBaseConfig(gender="W", feature_profile="lr_pruned_only_v1", alpha_profile="quality_only_men_quality_blocks_women")
    ratings_m = JIBaseConfig(gender="M", feature_profile="lr_ratings_only_v1", alpha_profile="quality_only_men_quality_blocks_women")
    ratings_w = JIBaseConfig(gender="W", feature_profile="lr_ratings_only_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women_fix_m = JIBaseConfig(gender="M", feature_profile="lr_women_fix_only_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women_fix_w = JIBaseConfig(gender="W", feature_profile="lr_women_fix_only_v1", alpha_profile="quality_only_men_quality_blocks_women")

    assert pruned_m.resolved_model_features() == [
        "Delta_Seed",
        "Seed_gap_abs",
        "Delta_Elo",
        "Delta_Quality",
        "Delta_neff",
        "QualityWins_diff",
        "OpponentQualityTournamentRank_diff",
        "Seed_x_Quality",
    ]
    assert pruned_w.resolved_model_features() == [
        "Delta_Seed",
        "Seed_gap_abs",
        "Delta_Elo",
        "Delta_Quality",
        "Delta_neff",
        "QualityWins_diff",
        "OpponentQualityTournamentRank_diff",
        "AvgBlkDiff_diff",
        "Seed_x_Quality",
    ]

    assert "Delta_CarryElo" in ratings_m.resolved_model_features()
    assert "Delta_Colley" in ratings_m.resolved_model_features()
    assert "Delta_SRS" in ratings_m.resolved_model_features()
    assert "Delta_CarryElo" in ratings_w.resolved_model_features()
    assert "Delta_Colley" in ratings_w.resolved_model_features()
    assert "Delta_SRS" in ratings_w.resolved_model_features()
    assert "Seed_x_Colley" not in ratings_w.resolved_model_features()

    assert "Seed_x_Quality" in women_fix_m.resolved_model_features()
    assert "Seed_x_Quality" not in women_fix_w.resolved_model_features()
    assert "Seed_x_Colley" in women_fix_w.resolved_model_features()


def test_lr_ratings_core_v2_profiles_isolate_single_rating_changes():
    v2a_m = JIBaseConfig(gender="M", feature_profile="lr_ratings_core_v2a", alpha_profile="quality_only_men_quality_blocks_women")
    v2a_w = JIBaseConfig(gender="W", feature_profile="lr_ratings_core_v2a", alpha_profile="quality_only_men_quality_blocks_women")
    v2b_m = JIBaseConfig(gender="M", feature_profile="lr_ratings_core_v2b", alpha_profile="quality_only_men_quality_blocks_women")
    v2b_w = JIBaseConfig(gender="W", feature_profile="lr_ratings_core_v2b", alpha_profile="quality_only_men_quality_blocks_women")
    v2c_m = JIBaseConfig(gender="M", feature_profile="lr_ratings_core_v2c", alpha_profile="quality_only_men_quality_blocks_women")
    v2c_w = JIBaseConfig(gender="W", feature_profile="lr_ratings_core_v2c", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_SRS" not in v2a_m.resolved_model_features()
    assert "Delta_SRS" not in v2a_w.resolved_model_features()
    assert "Delta_Colley" in v2a_m.resolved_model_features()
    assert "Seed_x_Colley" in v2a_w.resolved_model_features()

    assert "Delta_Colley" not in v2b_m.resolved_model_features()
    assert "Delta_Colley" not in v2b_w.resolved_model_features()
    assert "Delta_SRS" in v2b_m.resolved_model_features()
    assert "Delta_SRS" in v2b_w.resolved_model_features()
    assert "Seed_x_Colley" not in v2b_w.resolved_model_features()

    assert "Seed_x_Colley" in v2c_m.resolved_model_features()
    assert "Seed_x_Colley" not in v2c_w.resolved_model_features()
    assert "Delta_Colley" in v2c_w.resolved_model_features()
    assert "Delta_SRS" in v2c_w.resolved_model_features()


def test_lr_ratings_definition_v1_replaces_srs_with_eff_srs_only():
    men = JIBaseConfig(gender="M", feature_profile="lr_ratings_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women = JIBaseConfig(gender="W", feature_profile="lr_ratings_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_EffSRS" in men.resolved_model_features()
    assert "Delta_EffSRS" in women.resolved_model_features()
    assert "Delta_SRS" not in men.resolved_model_features()
    assert "Delta_SRS" not in women.resolved_model_features()
    assert "Seed_x_Colley" in men.resolved_model_features()
    assert "Seed_x_Colley" in women.resolved_model_features()


def test_lr_carry_elo_definition_v1_replaces_carry_elo_with_stronger_variant_only():
    men = JIBaseConfig(gender="M", feature_profile="lr_carry_elo_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women = JIBaseConfig(gender="W", feature_profile="lr_carry_elo_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_CarryElo85" in men.resolved_model_features()
    assert "Delta_CarryElo" not in men.resolved_model_features()
    assert "Delta_SRS" in men.resolved_model_features()
    assert "Delta_CarryElo85" in women.resolved_model_features()
    assert "Delta_CarryElo" not in women.resolved_model_features()
    assert "Seed_x_Colley" in women.resolved_model_features()


def test_lr_carry_elo_definition_confirm80_uses_midpoint_variant_only():
    men = JIBaseConfig(gender="M", feature_profile="lr_carry_elo_definition_confirm80", alpha_profile="quality_only_men_quality_blocks_women")
    women = JIBaseConfig(gender="W", feature_profile="lr_carry_elo_definition_confirm80", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_CarryElo80" in men.resolved_model_features()
    assert "Delta_CarryElo85" not in men.resolved_model_features()
    assert "Delta_CarryElo80" in women.resolved_model_features()
    assert "Delta_CarryElo85" not in women.resolved_model_features()


def test_lr_colley_definition_v1_replaces_colley_with_conference_downweighted_variant_only():
    men = JIBaseConfig(gender="M", feature_profile="lr_colley_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")
    women = JIBaseConfig(gender="W", feature_profile="lr_colley_definition_v1", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_ColleyNC" in men.resolved_model_features()
    assert "Delta_Colley" not in men.resolved_model_features()
    assert "Seed_x_ColleyNC" in men.resolved_model_features()
    assert "Seed_x_Colley" not in men.resolved_model_features()
    assert "Delta_CarryElo85" in men.resolved_model_features()
    assert "Delta_SRS" in men.resolved_model_features()

    assert "Delta_ColleyNC" in women.resolved_model_features()
    assert "Delta_Colley" not in women.resolved_model_features()
    assert "Seed_x_ColleyNC" in women.resolved_model_features()
    assert "Seed_x_Colley" not in women.resolved_model_features()
    assert "Delta_CarryElo85" in women.resolved_model_features()
    assert "Delta_SRS" in women.resolved_model_features()


def test_lr_srs_definition_profiles_replace_only_srs_with_clipped_variants():
    clip15_m = JIBaseConfig(gender="M", feature_profile="lr_srs_definition_v1_clip15", alpha_profile="quality_only_men_quality_blocks_women")
    clip15_w = JIBaseConfig(gender="W", feature_profile="lr_srs_definition_v1_clip15", alpha_profile="quality_only_men_quality_blocks_women")
    clip20_m = JIBaseConfig(gender="M", feature_profile="lr_srs_definition_confirm20", alpha_profile="quality_only_men_quality_blocks_women")
    clip20_w = JIBaseConfig(gender="W", feature_profile="lr_srs_definition_confirm20", alpha_profile="quality_only_men_quality_blocks_women")

    assert "Delta_SRSClip15" in clip15_m.resolved_model_features()
    assert "Delta_SRS" not in clip15_m.resolved_model_features()
    assert "Delta_CarryElo85" in clip15_m.resolved_model_features()
    assert "Delta_Colley" in clip15_m.resolved_model_features()
    assert "Delta_SRSClip15" in clip15_w.resolved_model_features()
    assert "Delta_SRS" not in clip15_w.resolved_model_features()

    assert "Delta_SRSClip20" in clip20_m.resolved_model_features()
    assert "Delta_SRS" not in clip20_m.resolved_model_features()
    assert "Delta_SRSClip20" in clip20_w.resolved_model_features()
    assert "Delta_SRS" not in clip20_w.resolved_model_features()


def test_men_best_women_direct_only_weight050_overlay_profile_only_changes_women_weight_even_more_conservatively():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight050")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight050")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.50
    assert profile_women.allow_injury is False


def test_men_best_women_direct_only_weight040_overlay_profile_only_changes_women_weight_to_040():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight040")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight040")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.40
    assert profile_women.allow_injury is False


def test_men_best_women_direct_only_weight030_overlay_profile_only_changes_women_weight_to_030():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight030")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight030")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.30
    assert profile_women.allow_injury is False


def test_men_best_women_direct_only_weight020_overlay_profile_only_changes_women_weight_to_020():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight020")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight020")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.20
    assert profile_women.allow_injury is False


def test_men_best_women_direct_only_weight025_overlay_profile_only_changes_women_weight_to_025():
    profile_men = build_ji_base_overlay_config("M", "ji_base_overlay_v1_men_best_women_direct_only_weight025")
    profile_women = build_ji_base_overlay_config("W", "ji_base_overlay_v1_men_best_women_direct_only_weight025")

    assert profile_men.overlay_source_profile == "direct_only"
    assert profile_men.injury_min_confirmed_out == 4
    assert profile_men.direct_weight == 0.85
    assert profile_women.overlay_source_profile == "direct_only"
    assert profile_women.direct_weight == 0.25
    assert profile_women.allow_injury is False


def test_lr_regularization_overrides_are_gender_specific():
    men = JIBaseConfig(gender="M", lr_c_m=0.7, lr_c_w=0.5)
    women = JIBaseConfig(gender="W", lr_c_m=0.7, lr_c_w=0.5)

    assert men.resolved_lr_c() == 0.7
    assert women.resolved_lr_c() == 0.5


def test_women_slice_redesign_v1_architecture_profile_is_women_only():
    men = JIBaseConfig(
        gender="M",
        model_family="JI_lr_control",
        feature_profile="women_slice_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    )
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_slice_redesign_v1_architecture",
        women_quality_profile="consensus_rebuild_v5",
    )

    men_cols = men.resolved_model_features()
    women_cols = women.resolved_model_features()

    assert "Delta_WomenCompositeQualityV5" not in men_cols
    assert "Delta_WomenCompositeQualityV5" in women_cols
    assert "Seed_x_WomenOpponentTournamentStrength" in women_cols
    assert "Seed_x_Quality" not in women_cols


def test_women_slice_redesign_v1_no_seed_interaction_profile_removes_seed_interaction():
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_slice_redesign_v1_no_seed_interaction",
        women_quality_profile="consensus_rebuild_v5",
    )

    women_cols = women.resolved_model_features()

    assert "Delta_WomenCompositeQualityV5" in women_cols
    assert "Seed_x_WomenOpponentTournamentStrength" not in women_cols


def test_women_opp_rank_redesign_architecture_profile_is_women_only():
    men = JIBaseConfig(
        gender="M",
        model_family="JI_lr_control",
        feature_profile="women_opp_rank_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    )
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_opp_rank_redesign_v1_architecture",
        women_quality_profile="consensus_rebuild_v4",
    )

    men_cols = men.resolved_model_features()
    women_cols = women.resolved_model_features()

    assert "Delta_WomenOpponentTournamentStrengthV2" not in men_cols
    assert "Delta_WomenOpponentTournamentStrengthV2" in women_cols
    assert "Seed_x_WomenOpponentTournamentStrengthV2" in women_cols
    assert "OpponentQualityTournamentRank_diff" not in women_cols


def test_women_opp_rank_redesign_no_seed_interaction_profile_removes_seed_term():
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_opp_rank_redesign_v1_no_seed_interaction",
        women_quality_profile="consensus_rebuild_v4",
    )

    women_cols = women.resolved_model_features()

    assert "Delta_WomenOpponentTournamentStrengthV2" in women_cols
    assert "Seed_x_WomenOpponentTournamentStrengthV2" not in women_cols


def test_women_qualitywins_redesign_architecture_profile_is_women_only():
    men = JIBaseConfig(
        gender="M",
        model_family="JI_lr_control",
        feature_profile="women_qualitywins_redesign_v1_architecture",
        women_quality_profile="legacy_v1",
    )
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_qualitywins_redesign_v1_architecture",
        women_quality_profile="consensus_rebuild_v4",
    )

    men_cols = men.resolved_model_features()
    women_cols = women.resolved_model_features()

    assert "Delta_WomenQualityWinsStrengthV2" not in men_cols
    assert "Delta_WomenQualityWinsStrengthV2" in women_cols
    assert "OpponentQualityTournamentRank_diff" in women_cols
    assert "Seed_x_WomenQualityWinsStrengthV2" not in women_cols


def test_women_qualitywins_redesign_with_seed_interaction_adds_seed_term():
    women = JIBaseConfig(
        gender="W",
        model_family="JI_lr_control",
        feature_profile="women_qualitywins_redesign_v1_with_seed_interaction",
        women_quality_profile="consensus_rebuild_v4",
    )

    women_cols = women.resolved_model_features()

    assert "Delta_WomenQualityWinsStrengthV2" in women_cols
    assert "Seed_x_WomenQualityWinsStrengthV2" in women_cols
