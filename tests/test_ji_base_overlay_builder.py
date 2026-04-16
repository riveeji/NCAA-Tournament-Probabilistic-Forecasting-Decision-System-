import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_overlay_submission.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_overlay_submission", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_overlay_profiles_is_independent_from_base_submission_builder():
    module = _load_module()
    profiles = module.resolve_overlay_profiles()

    assert profiles == [
        {
            "submission_profile": "ji_base_overlay_v1",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_priority",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_conservative_injury",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_priority",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed3",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed4",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed5",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_priority",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_priority",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight070",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight060",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight050",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight040",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight030",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight020",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight025",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
        {
            "submission_profile": "ji_base_overlay_v2_men_player_injury_weight025",
            "base_submission_profile": "ji_base_base",
            "overlay_source_profile_m": "direct_only",
            "overlay_source_profile_w": "direct_only",
            "overlay_stack_m": "market_injury",
            "overlay_stack_w": "market_only",
        },
    ]


def test_resolve_overlay_profile_supports_conservative_injury_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_conservative_injury")

    assert profile["submission_profile"] == "ji_base_overlay_v1_conservative_injury"
    assert profile["base_submission_profile"] == "ji_base_base"


def test_resolve_overlay_profile_supports_direct_only_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_direct_only_strict_confirmed_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only_injury_strict_confirmed")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_strict_confirmed"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_direct_only_confirmed3_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only_injury_confirmed3")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_confirmed3"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_direct_only_confirmed4_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only_injury_confirmed4")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_confirmed4"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_direct_only_confirmed5_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only_injury_confirmed5")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_confirmed5"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_direct_only_confirmed4_shift008_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_direct_only_injury_confirmed4_shift008")

    assert profile["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_priority_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_priority")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_priority"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_priority"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight070_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight070")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight070"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight060_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight060")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight060"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight050_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight050")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight050"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight040_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight040")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight040"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight030_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight030")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight030"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight020_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight020")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight020"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_men_best_women_direct_only_weight025_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v1_men_best_women_direct_only_weight025")

    assert profile["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight025"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"


def test_resolve_overlay_profile_supports_player_level_injury_v2_submission():
    module = _load_module()
    profile = module.resolve_overlay_profile("ji_base_overlay_v2_men_player_injury_weight025")

    assert profile["submission_profile"] == "ji_base_overlay_v2_men_player_injury_weight025"
    assert profile["overlay_source_profile_m"] == "direct_only"
    assert profile["overlay_source_profile_w"] == "direct_only"
