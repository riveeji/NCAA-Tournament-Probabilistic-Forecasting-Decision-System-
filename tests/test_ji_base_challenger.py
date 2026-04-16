import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_run_ji_base_challenger_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_challenger.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_challenger", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanitize_candidate_name_produces_stable_slug():
    module = _load_run_ji_base_challenger_module()

    assert module.sanitize_candidate_name("Women Quality V5!") == "women_quality_v5"
    assert module.sanitize_candidate_name("   ") == "candidate"


def test_build_challenger_configs_preserves_working_defaults_when_unset():
    module = _load_run_ji_base_challenger_module()
    args = Namespace(
        candidate_name="test",
        model_family=None,
        calibration_mode=None,
        isotonic_min_samples=None,
        feature_profile=None,
        alpha_profile=None,
        women_quality_profile_w=None,
        women_quality_profile_m=None,
        recent_window=None,
        output=None,
    )

    men, women = module.build_challenger_configs(args)

    assert men.model_family == "JI_lr_control"
    assert women.model_family == "JI_lr_control"
    assert men.alpha_profile == "quality_only_men_quality_blocks_women"
    assert women.alpha_profile == "quality_only_men_quality_blocks_women"
    assert men.women_quality_profile == "legacy_v1"
    assert women.women_quality_profile == "consensus_rebuild_v4"
    assert men.isotonic_min_samples == 20
    assert women.isotonic_min_samples == 20


def test_build_challenger_configs_applies_overrides_to_men_and_women():
    module = _load_run_ji_base_challenger_module()
    args = Namespace(
        candidate_name="test",
        model_family="JI_spread_xgb",
        calibration_mode="isotonic_gender",
        isotonic_min_samples=100,
        feature_profile="strength_blend_alt",
        alpha_profile="quality_only_men_core_women",
        women_quality_profile_w="consensus_rebuild_v4b",
        women_quality_profile_m="legacy_v1",
        recent_window=3,
        output=None,
    )

    men, women = module.build_challenger_configs(args)

    assert men.model_family == "JI_spread_xgb"
    assert women.model_family == "JI_spread_xgb"
    assert men.calibration_mode == "isotonic_gender"
    assert women.calibration_mode == "isotonic_gender"
    assert men.feature_profile == "strength_blend_alt"
    assert women.feature_profile == "strength_blend_alt"
    assert women.women_quality_profile == "consensus_rebuild_v4b"
    assert men.isotonic_min_samples == 100
    assert women.isotonic_min_samples == 100
    assert men.recent_window == 3
    assert women.recent_window == 3


def test_build_challenger_configs_accepts_women_conservative_feature_profile():
    module = _load_run_ji_base_challenger_module()
    args = Namespace(
        candidate_name="test",
        model_family=None,
        calibration_mode=None,
        isotonic_min_samples=None,
        feature_profile="seed_quality_interaction_women_conservative",
        alpha_profile=None,
        women_quality_profile_w=None,
        women_quality_profile_m=None,
        recent_window=None,
        output=None,
    )

    men, women = module.build_challenger_configs(args)

    assert men.feature_profile == "seed_quality_interaction_women_conservative"
    assert women.feature_profile == "seed_quality_interaction_women_conservative"


def test_build_challenger_configs_accepts_women_tossup_quality_conservative_feature_profile():
    module = _load_run_ji_base_challenger_module()
    args = Namespace(
        candidate_name="test",
        model_family=None,
        calibration_mode=None,
        isotonic_min_samples=None,
        feature_profile="women_tossup_quality_conservative",
        alpha_profile=None,
        women_quality_profile_w=None,
        women_quality_profile_m=None,
        recent_window=None,
        output=None,
    )

    men, women = module.build_challenger_configs(args)

    assert men.feature_profile == "women_tossup_quality_conservative"
    assert women.feature_profile == "women_tossup_quality_conservative"
