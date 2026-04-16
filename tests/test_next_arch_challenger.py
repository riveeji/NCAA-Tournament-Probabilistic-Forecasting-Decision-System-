import importlib.util
from pathlib import Path


def _load_run_next_arch_challenger_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_next_arch_challenger.py"
    spec = importlib.util.spec_from_file_location("run_next_arch_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_candidate_args_maps_known_next_arch_candidates():
    module = _load_run_next_arch_challenger_module()

    assert module.resolve_candidate_args("arch::tabr_v1") == {"experiment_name": "tabr_v1"}
    assert module.resolve_candidate_args("arch::tabr_hybrid_v1") == {"experiment_name": "tabr_hybrid_v1"}
    assert module.resolve_candidate_args("arch::tabr_feature_fusion_v1") == {"experiment_name": "tabr_feature_fusion_v1"}
    assert module.resolve_candidate_args("arch::pairwise_ranking_v1") == {"experiment_name": "pairwise_ranking_v1"}
    assert module.resolve_candidate_args("arch::season_encoder_transformer_v1") == {"experiment_name": "season_encoder_transformer_v1"}
    assert module.resolve_candidate_args("arch::graph_static_embedding_v1") == {"experiment_name": "graph_static_embedding_v1"}
    assert module.resolve_candidate_args("arch::gender_specific_stacker_v1") == {"experiment_name": "gender_specific_stacker_v1"}


def test_sanitize_candidate_name_produces_stable_slug():
    module = _load_run_next_arch_challenger_module()

    assert module.sanitize_candidate_name("Arch::TabR V1") == "arch_tabr_v1"
