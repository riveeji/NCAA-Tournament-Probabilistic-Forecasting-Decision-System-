from hc.next_arch.config import NextArchConfig


def test_tabr_v1_reuses_frozen_core_feature_columns():
    men = NextArchConfig(gender="M", experiment_name="tabr_v1")
    women = NextArchConfig(gender="W", experiment_name="tabr_v1")

    assert "Delta_CarryElo85" in men.resolved_model_features()
    assert "Delta_CarryElo85" in women.resolved_model_features()
    assert "Seed_x_Quality" in men.resolved_model_features()
    assert "Seed_x_Colley" in women.resolved_model_features()


def test_tabr_hybrid_v1_adds_baseline_logit_to_frozen_core_feature_columns():
    men = NextArchConfig(gender="M", experiment_name="tabr_hybrid_v1")
    women = NextArchConfig(gender="W", experiment_name="tabr_hybrid_v1")

    assert "BaselineLogit" in men.resolved_model_features()
    assert "BaselineLogit" in women.resolved_model_features()
    assert "Delta_CarryElo85" in men.resolved_model_features()
    assert "AvgBlkDiff_diff" in women.resolved_model_features()


def test_tabr_feature_fusion_v1_adds_baseline_logit_to_frozen_core_feature_columns():
    men = NextArchConfig(gender="M", experiment_name="tabr_feature_fusion_v1")
    women = NextArchConfig(gender="W", experiment_name="tabr_feature_fusion_v1")

    assert "BaselineLogit" in men.resolved_model_features()
    assert "BaselineLogit" in women.resolved_model_features()
    assert "Seed_x_Quality" in men.resolved_model_features()
    assert "Seed_x_Colley" in women.resolved_model_features()


def test_graph_static_embedding_v1_uses_only_graph_derived_features():
    men = NextArchConfig(gender="M", experiment_name="graph_static_embedding_v1")
    women = NextArchConfig(gender="W", experiment_name="graph_static_embedding_v1")

    assert men.resolved_model_features() == ["GraphEmbCosSim", "GraphEmbL2", "Delta_GraphEmbStrength"]
    assert women.resolved_model_features() == ["GraphEmbCosSim", "GraphEmbL2", "Delta_GraphEmbStrength"]


def test_gender_specific_stacker_v1_uses_gender_specific_feature_sets():
    men = NextArchConfig(gender="M", experiment_name="gender_specific_stacker_v1")
    women = NextArchConfig(gender="W", experiment_name="gender_specific_stacker_v1")

    assert men.resolved_model_features() == ["BaselineLogit"]
    assert women.resolved_model_features() == [
        "BaselineLogit",
        "Delta_WomenConsensusRankScore",
        "WomenConsensusCoverageMean",
        "WomenConsensusConfidenceMean",
    ]
