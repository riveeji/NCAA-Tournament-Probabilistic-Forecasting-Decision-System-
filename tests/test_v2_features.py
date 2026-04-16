from hc.v2.config import V2Config
from hc.v2.data import build_v2_dataset


def test_v2_dataset_has_diff_only_feature_columns():
    dataset = build_v2_dataset(V2Config(gender="M", route="probability", model_variant="lr", market_mode="none"))
    diff_cols = [col for col in dataset.columns if col.endswith("_diff")]
    assert diff_cols
    assert "SeedNum_diff" in diff_cols
    assert "Elo_diff" in diff_cols
    assert "AdjOffRtg_diff" in diff_cols
    assert "AdjDefRtg_diff" in diff_cols
    assert "Tempo_diff" in diff_cols
    assert "OppEFG_diff" in diff_cols
    assert "AdjNetRtg_diff" in diff_cols
    assert "Label" in dataset.columns
    assert "Margin" in dataset.columns
    assert "Season" in dataset.columns


def test_v2_dataset_builds_for_women():
    dataset = build_v2_dataset(V2Config(gender="W", route="probability", model_variant="lr", market_mode="none"))
    assert len(dataset) > 0
    assert dataset["Label"].isin([0, 1]).all()


def test_v2_dataset_margin_direction_matches_label():
    dataset = build_v2_dataset(V2Config(gender="M", route="spread", model_variant="lr", market_mode="none"))
    assert ((dataset["Label"] == 1) == (dataset["Margin"] > 0)).all()


def test_v2_dataset_feature_packs_control_efficiency_and_opponent_adjusted_columns():
    base = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="base")
    )
    efficiency = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="efficiency")
    )
    opp_adjusted = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="opp_adjusted")
    )

    assert "AdjOffRtg_diff" not in base.columns
    assert "Tempo_diff" not in base.columns
    assert "OppEFG_diff" not in base.columns

    assert "AdjOffRtg_diff" in efficiency.columns
    assert "Tempo_diff" in efficiency.columns
    assert "OppEFG_diff" not in efficiency.columns

    assert "AdjOffRtg_diff" in opp_adjusted.columns
    assert "OppEFG_diff" in opp_adjusted.columns
    assert "OppTOVPct_diff" in opp_adjusted.columns


def test_v2_dataset_strength_feature_packs_use_internal_strength_columns():
    strength_full = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="strength_full")
    )
    strength_recent = build_v2_dataset(
        V2Config(gender="W", route="spread", model_variant="lr", market_mode="none", feature_pack="strength_recent")
    )

    for column in [
        "StrengthNet_diff",
        "StrengthOff_diff",
        "StrengthDef_diff",
        "StrengthTempo_diff",
        "StrengthSOS_diff",
        "StrengthTop50_diff",
        "StrengthPath_diff",
        "StrengthMomentum_diff",
        "StrengthOffMomentum_diff",
        "StrengthDefMomentum_diff",
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "Elo_diff",
        "SOS_diff",
        "WinRate_diff",
        "AvgMargin_diff",
        "Last30WinRate_diff",
    ]:
        assert column in strength_full.columns
        assert column in strength_recent.columns

    assert "Ext_WN_ELO_diff" not in strength_full.columns
    assert "Ext_WN_ELO_diff" not in strength_recent.columns


def test_v2_dataset_external_base_feature_pack_uses_curated_external_signals():
    men = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="external_base")
    )
    women = build_v2_dataset(
        V2Config(gender="W", route="spread", model_variant="lr", market_mode="none", feature_pack="external_base")
    )

    men_expected = [
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "CloseGameWinRate_diff",
        "ExternalCompositeStrength_diff",
        "ExternalFallbackElo_diff",
        "ExternalBPIStrength_diff",
        "ExternalPOMStrength_diff",
        "ExternalNETStrength_diff",
        "ExternalELORankStrength_diff",
        "ExternalSORStrength_diff",
        "MasseyPOMStrength_diff",
        "SeedNum_x_ExternalCompositeStrength_diff",
        "CloseGameWinRate_x_AvgMargin_diff",
    ]
    women_expected = [
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "CloseGameWinRate_diff",
        "ExternalCompositeStrength_diff",
        "ExternalFallbackElo_diff",
        "ExternalNETStrength_diff",
        "ExternalRPIStrength_diff",
        "ExternalPredRPIStrength_diff",
        "ExternalELORankStrength_diff",
        "ExternalAPStrength_diff",
        "SeedNum_x_ExternalCompositeStrength_diff",
        "CloseGameWinRate_x_AvgMargin_diff",
    ]

    for column in men_expected:
        assert column in men.columns
    for column in women_expected:
        assert column in women.columns

    assert "AdjOffRtg_diff" not in men.columns
    assert "OppEFG_diff" not in men.columns
    assert "StrengthNet_diff" not in men.columns


def test_v2_dataset_external_base_pruned_uses_gender_specific_curated_subset():
    men = build_v2_dataset(
        V2Config(gender="M", route="spread", model_variant="lr", market_mode="none", feature_pack="external_base_pruned")
    )
    women = build_v2_dataset(
        V2Config(gender="W", route="spread", model_variant="lr", market_mode="none", feature_pack="external_base_pruned")
    )

    men_keep = {
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "Elo_diff",
        "AvgMargin_diff",
        "CloseGameWinRate_diff",
        "Last30SOS_diff",
        "ExternalCompositeStrength_diff",
        "ExternalFallbackElo_diff",
        "ExternalBPIStrength_diff",
        "ExternalPOMStrength_diff",
        "ExternalNETStrength_diff",
        "MasseyPOMStrength_diff",
        "MasseyMORStrength_diff",
        "SeedNum_x_ExternalCompositeStrength_diff",
        "SeedNum_x_ExternalPOMStrength_diff",
    }
    women_keep = {
        "SeedNum_diff",
        "SeedPriorExpectedWins_diff",
        "Elo_diff",
        "AvgMargin_diff",
        "CloseGameWinRate_diff",
        "ExternalCompositeStrength_diff",
        "ExternalFallbackElo_diff",
        "ExternalNETStrength_diff",
        "ExternalRPIStrength_diff",
        "ExternalPredRPIStrength_diff",
        "SeedNum_x_ExternalCompositeStrength_diff",
        "SeedNum_x_ExternalNETStrength_diff",
    }

    for column in men_keep:
        assert column in men.columns
    for column in women_keep:
        assert column in women.columns

    assert "ExternalWABStrength_diff" not in men.columns
    assert "ExternalAPStrength_diff" not in women.columns
