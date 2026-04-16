from hc.next_arch.config import NextArchConfig
from hc.next_arch.data import build_next_arch_dataset
from hc.next_arch.replay import run_next_arch_gender_replay
from hc.next_arch.season_encoder import build_team_season_sequences


def test_season_encoder_transformer_v1_uses_embedding_features():
    men = NextArchConfig(gender="M", experiment_name="season_encoder_transformer_v1")
    women = NextArchConfig(gender="W", experiment_name="season_encoder_transformer_v1")

    assert men.resolved_model_features() == [
        "SeasonEmbCosSim",
        "SeasonEmbL2",
        "Delta_SeasonEmbStrength",
    ]
    assert women.resolved_model_features() == [
        "SeasonEmbCosSim",
        "SeasonEmbL2",
        "Delta_SeasonEmbStrength",
    ]


def test_build_team_season_sequences_returns_one_season_only():
    sequences = build_team_season_sequences(gender="M")

    assert {"Season", "TeamID", "SequenceLength", "GameSequence"}.issubset(sequences.columns)
    assert sequences.groupby(["Season", "TeamID"]).size().eq(1).all()


def test_season_encoder_dataset_emits_finite_embedding_features():
    dataset = build_next_arch_dataset(NextArchConfig(gender="W", experiment_name="season_encoder_transformer_v1"))

    for column in ["SeasonEmbCosSim", "SeasonEmbL2", "Delta_SeasonEmbStrength"]:
        assert dataset[column].notna().all()


def test_season_encoder_transformer_v1_outputs_probabilities_in_unit_interval():
    replay = run_next_arch_gender_replay(NextArchConfig(gender="M", experiment_name="season_encoder_transformer_v1"))

    assert replay["predictions"]["raw_prob"].between(0.0, 1.0).all()
    assert replay["predictions"]["calibrated_prob"].between(0.0, 1.0).all()
