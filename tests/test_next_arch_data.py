from hc.next_arch.config import NextArchConfig
from hc.next_arch.data import build_next_arch_dataset, load_static_graph_team_embeddings


def test_static_graph_embeddings_expose_required_team_level_columns():
    embeddings = load_static_graph_team_embeddings("M")

    assert {"Season", "TeamID", "GraphEmbStrength"}.issubset(embeddings.columns)
    graph_cols = [column for column in embeddings.columns if column.startswith("GraphEmb_")]
    assert graph_cols


def test_graph_static_embedding_dataset_exposes_required_matchup_columns_without_nan():
    dataset = build_next_arch_dataset(NextArchConfig(gender="W", experiment_name="graph_static_embedding_v1"))

    for column in ("GraphEmbCosSim", "GraphEmbL2", "Delta_GraphEmbStrength"):
        assert column in dataset.columns
        assert dataset[column].notna().all()


def test_gender_specific_stacker_v1_women_dataset_exposes_sidecar_columns_without_nan():
    dataset = build_next_arch_dataset(NextArchConfig(gender="W", experiment_name="gender_specific_stacker_v1"))

    for column in (
        "Delta_WomenConsensusRankScore",
        "WomenConsensusCoverageMean",
        "WomenConsensusConfidenceMean",
    ):
        assert column in dataset.columns
        assert dataset[column].notna().all()


def test_gender_specific_stacker_v1_men_dataset_does_not_require_women_sidecar_columns():
    dataset = build_next_arch_dataset(NextArchConfig(gender="M", experiment_name="gender_specific_stacker_v1"))

    assert "Delta_CarryElo85" in dataset.columns
    assert "Delta_WomenConsensusRankScore" not in dataset.columns
