import importlib.util
from pathlib import Path
import pandas as pd


def _load_run_ji_base_replay_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_replay.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_replay", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_gate_requires_total_and_women_improvement_without_recent_regression():
    module = _load_run_ji_base_replay_module()
    baseline = {
        "total_cv_brier_calibrated": 0.164156,
        "women_cv_brier_calibrated": 0.143878,
        "latest_season_equal_gender_brier": 0.127427,
        "recent_window_equal_gender_brier": 0.170846,
    }
    improved = {
        "total_cv_brier_calibrated": 0.1639,
        "women_cv_brier_calibrated": 0.1435,
        "latest_season_equal_gender_brier": 0.1281,
        "recent_window_equal_gender_brier": 0.1714,
    }
    regressed = {
        "total_cv_brier_calibrated": 0.1639,
        "women_cv_brier_calibrated": 0.1435,
        "latest_season_equal_gender_brier": 0.1290,
        "recent_window_equal_gender_brier": 0.1725,
    }

    assert module.passes_experiment_gate(candidate=improved, baseline=baseline)
    assert not module.passes_experiment_gate(candidate=regressed, baseline=baseline)


def test_experiment_gate_allows_women_non_regression_within_tiny_numeric_epsilon():
    module = _load_run_ji_base_replay_module()
    baseline = {
        "total_cv_brier_calibrated": 0.163960452054998,
        "women_cv_brier_calibrated": 0.1437107336359529,
        "latest_season_equal_gender_brier": 0.12815626688851176,
        "recent_window_equal_gender_brier": 0.17000269115902753,
    }
    candidate = {
        "total_cv_brier_calibrated": 0.16384555876950674,
        "women_cv_brier_calibrated": 0.1437107336359529,
        "latest_season_equal_gender_brier": 0.12747509934503048,
        "recent_window_equal_gender_brier": 0.16961636184326073,
    }

    assert module.passes_experiment_gate(candidate=candidate, baseline=baseline)


def test_stage_runner_targets_women_quality_v3_candidate():
    module = _load_run_ji_base_replay_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "women_quality::consensus_rebuild_v4" in source
    assert '"feature::seed_quality_plus_women_consensus"' in source
    assert '"quality_only_women_light"' in source
    assert '"quality_only_men_core_women"' in source
    assert '"quality_only_men_quality_blocks_women"' in source
    assert '"quality_only_men_harry_quality_women"' in source
    assert '"quality_only_men_harry_blocks_women"' in source


def test_challenger_script_accepts_lr_pruned_core_v1_feature_profile():
    script = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    source = script.read_text(encoding="utf-8")

    assert '"lr_pruned_core_v1"' in source
    assert '"lr_pruned_only_v1"' in source
    assert '"lr_ratings_only_v1"' in source
    assert '"lr_women_fix_only_v1"' in source
    assert '"lr_ratings_core_v2a"' in source
    assert '"lr_ratings_core_v2b"' in source
    assert '"lr_ratings_core_v2c"' in source
    assert '"lr_ratings_definition_v1"' in source
    assert '"lr_carry_elo_definition_v1"' in source
    assert '"lr_carry_elo_definition_confirm80"' in source
    assert '"lr_colley_definition_v1"' in source
    assert '"lr_srs_definition_v1_clip15"' in source
    assert '"lr_srs_definition_confirm20"' in source
    assert '"women_slice_redesign_v1_architecture"' in source
    assert '"women_slice_redesign_v1_no_seed_interaction"' in source
    assert '"consensus_rebuild_v6"' in source
    assert '"core::women_ranking_upstream_v1_internal_refactor"' in source
    assert '"core::women_ranking_upstream_v1_external_consensus"' in source
    assert '"external_consensus_v2"' in source
    assert '"core::women_ranking_upstream_v2_internal_refactor"' in source
    assert '"core::women_ranking_upstream_v2_external_consensus"' in source
    assert '"historical_consensus_snapshots_v1"' in source
    assert '"core::women_ranking_historical_snapshots_v1"' in source


def test_challenger_script_resolves_women_slice_redesign_architecture_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_slice_redesign_v1_architecture")

    assert args["feature_profile"] == "women_slice_redesign_v1_architecture"
    assert args["women_quality_profile_w"] == "consensus_rebuild_v5"


def test_challenger_script_resolves_women_slice_redesign_no_seed_interaction_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_slice_redesign_v1_no_seed_interaction")

    assert args["feature_profile"] == "women_slice_redesign_v1_no_seed_interaction"
    assert args["women_quality_profile_w"] == "consensus_rebuild_v5"


def test_challenger_script_resolves_women_opp_rank_redesign_architecture_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_opp_rank_redesign_v1_architecture")

    assert args["feature_profile"] == "women_opp_rank_redesign_v1_architecture"


def test_challenger_script_resolves_women_opp_rank_redesign_no_seed_interaction_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_opp_rank_redesign_v1_no_seed_interaction")

    assert args["feature_profile"] == "women_opp_rank_redesign_v1_no_seed_interaction"


def test_challenger_script_resolves_women_qualitywins_redesign_architecture_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_qualitywins_redesign_v1_architecture")

    assert args["feature_profile"] == "women_qualitywins_redesign_v1_architecture"


def test_challenger_script_resolves_women_qualitywins_redesign_with_seed_interaction_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_qualitywins_redesign_v1_with_seed_interaction")

    assert args["feature_profile"] == "women_qualitywins_redesign_v1_with_seed_interaction"


def test_challenger_script_resolves_women_ranking_upstream_internal_refactor_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_ranking_upstream_v1_internal_refactor")

    assert args["women_quality_profile_w"] == "consensus_rebuild_v6"
    assert args["women_ranking_provider_w"] == "internal_fallback"


def test_challenger_script_resolves_women_ranking_upstream_external_consensus_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_ranking_upstream_v1_external_consensus")

    assert args["women_quality_profile_w"] == "consensus_rebuild_v6"
    assert args["women_ranking_provider_w"] == "external_consensus_v1"


def test_challenger_script_resolves_women_ranking_upstream_v2_internal_refactor_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_ranking_upstream_v2_internal_refactor")

    assert args["women_quality_profile_w"] == "consensus_rebuild_v6"
    assert args["women_ranking_provider_w"] == "internal_fallback"


def test_challenger_script_resolves_women_ranking_upstream_v2_external_consensus_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_ranking_upstream_v2_external_consensus")

    assert args["women_quality_profile_w"] == "consensus_rebuild_v6"
    assert args["women_ranking_provider_w"] == "external_consensus_v2"


def test_challenger_script_resolves_women_ranking_historical_snapshots_candidate():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.resolve_candidate_args("core::women_ranking_historical_snapshots_v1")

    assert args["women_quality_profile_w"] == "consensus_rebuild_v6"
    assert args["women_ranking_provider_w"] == "historical_consensus_snapshots_v1"


def test_variant_plan_excludes_ji_node_control_by_default():
    module = _load_run_ji_base_replay_module()

    assert ("JI_node_control", "none") not in module.resolve_variant_plan()
    assert ("JI_node_control", "isotonic_gender") not in module.resolve_variant_plan()


def test_variant_plan_can_include_ji_node_control_for_experimental_runs():
    module = _load_run_ji_base_replay_module()

    assert ("JI_node_control", "none") in module.resolve_variant_plan(include_experimental=True)
    assert ("JI_node_control", "isotonic_gender") in module.resolve_variant_plan(include_experimental=True)


def test_run_candidate_reuses_cached_rows_when_candidate_is_complete(monkeypatch):
    module = _load_run_ji_base_replay_module()
    summary_rows = []
    combined_rows = []
    cached_summary = pd.DataFrame(
        [
            {"candidate_name": "alpha::cached", "gender": "M", "cv_brier_calibrated": 0.18},
            {"candidate_name": "alpha::cached", "gender": "W", "cv_brier_calibrated": 0.14},
        ]
    )
    cached_combined = pd.DataFrame(
        [
            {
                "candidate_name": "alpha::cached",
                "total_cv_brier_calibrated": 0.16,
                "women_cv_brier_calibrated": 0.14,
            }
        ]
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("run_gender_replay should not be called when cache is complete")

    monkeypatch.setattr(module, "run_gender_replay", _should_not_run)

    combined = module._run_candidate(
        phase="alpha_profiles",
        candidate_name="alpha::cached",
        men_config=module.JIBaseConfig(gender="M"),
        women_config=module.JIBaseConfig(gender="W"),
        summary_rows=summary_rows,
        combined_rows=combined_rows,
        cached_summary=cached_summary,
        cached_combined=cached_combined,
    )

    assert combined["candidate_name"] == "alpha::cached"
    assert len(summary_rows) == 2
    assert len(combined_rows) == 1


def test_run_candidate_executes_when_cached_rows_are_incomplete(monkeypatch):
    module = _load_run_ji_base_replay_module()
    summary_rows = []
    combined_rows = []
    cached_summary = pd.DataFrame([{"candidate_name": "alpha::partial", "gender": "M", "cv_brier_calibrated": 0.18}])
    cached_combined = pd.DataFrame()
    calls = []

    def _fake_run(config):
        calls.append(config.gender)
        return {
            "gender": config.gender,
            "model_family": config.model_family,
            "feature_profile": config.feature_profile,
            "rating_profile": config.resolved_rating_profile(),
            "women_quality_profile": config.women_quality_profile,
            "alpha_profile": config.alpha_profile,
            "sidecar_profile": config.sidecar_profile,
            "calibration_mode": config.calibration_mode,
            "selection_objective": config.resolved_selection_objective(),
            "cv_brier_raw": 0.18 if config.gender == "M" else 0.14,
            "cv_brier_calibrated": 0.18 if config.gender == "M" else 0.14,
            "latest_season_brier": 0.13,
            "recent_window_brier": 0.17,
        }

    monkeypatch.setattr(module, "run_gender_replay", _fake_run)

    combined = module._run_candidate(
        phase="alpha_profiles",
        candidate_name="alpha::partial",
        men_config=module.JIBaseConfig(gender="M"),
        women_config=module.JIBaseConfig(gender="W"),
        summary_rows=summary_rows,
        combined_rows=combined_rows,
        cached_summary=cached_summary,
        cached_combined=cached_combined,
    )

    assert calls == ["M", "W"]
    assert combined["candidate_name"] == "alpha::partial"
    assert len(summary_rows) == 2
    assert len(combined_rows) == 1
