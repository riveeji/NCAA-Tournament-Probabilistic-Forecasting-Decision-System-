from hc.next_arch.config import NextArchConfig
from hc.next_arch.replay import run_next_arch_gender_replay


def test_pairwise_ranking_v1_reuses_frozen_core_feature_columns():
    men = NextArchConfig(gender="M", experiment_name="pairwise_ranking_v1")
    women = NextArchConfig(gender="W", experiment_name="pairwise_ranking_v1")

    assert "Delta_CarryElo85" in men.resolved_model_features()
    assert "Seed_x_Quality" in men.resolved_model_features()
    assert "AvgBlkDiff_diff" in women.resolved_model_features()
    assert "Seed_x_Colley" in women.resolved_model_features()


def test_pairwise_ranking_v1_outputs_probabilities_in_unit_interval():
    replay = run_next_arch_gender_replay(NextArchConfig(gender="M", experiment_name="pairwise_ranking_v1"))

    assert replay["predictions"]["raw_prob"].between(0.0, 1.0).all()
    assert replay["predictions"]["calibrated_prob"].between(0.0, 1.0).all()
