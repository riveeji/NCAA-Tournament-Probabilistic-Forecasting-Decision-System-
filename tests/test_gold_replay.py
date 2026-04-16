import importlib.util
from pathlib import Path

from hc.gold import GoldConfig, run_gender_replay


def _load_run_gold_replay_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_gold_replay.py"
    spec = importlib.util.spec_from_file_location("run_gold_replay", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gold_replay_outputs_required_fields():
    replay = run_gender_replay(GoldConfig(gender="M"))

    assert replay["gender"] == "M"
    assert replay["model_family"] == "gold_linear"
    assert replay["feature_profile"] == "gold_recover_wide"
    assert replay["rating_profile"] == "gold_recover_multi_rating"
    assert "mean_brier" in replay
    assert "latest_season_brier" in replay
    assert "recent_window_brier" in replay
    assert not replay["by_season"].empty
    assert not replay["predictions"].empty
    assert replay["predictions"]["Prob"].between(0.0, 1.0).all()


def test_run_gold_replay_defaults_to_recovered_variants():
    module = _load_run_gold_replay_module()
    plan = module.resolve_variant_plan(include_controls=False)

    assert plan == [
        ("gold_linear", "none", "current_default"),
        ("gold_linear", "isotonic_gender", "current_default"),
        ("gold_linear", "none", "m_ap_removed_only"),
        ("gold_linear", "isotonic_gender", "m_ap_removed_only"),
        ("gold_linear", "none", "a_tier_default"),
        ("gold_linear", "isotonic_gender", "a_tier_default"),
        ("gold_harry_lr", "none", "current_default"),
        ("gold_harry_lr", "isotonic_gender", "current_default"),
        ("gold_harry_xgb_spread", "none", "current_default"),
        ("gold_harry_xgb_spread", "isotonic_gender", "current_default"),
        ("gold_xgb_spread_light", "none", "current_default"),
        ("gold_xgb_spread_light", "isotonic_gender", "current_default"),
    ]


def test_run_gold_replay_can_add_control_families():
    module = _load_run_gold_replay_module()
    plan = module.resolve_variant_plan(include_controls=True)

    assert ("gold_min_lr", "none", "current_default") in plan
    assert ("gold_min_xgb_spread", "none", "current_default") in plan
    assert ("gold_tree_control", "none", "current_default") in plan
    assert ("gold_spread_control", "none", "current_default") in plan


def test_gold_harry_replay_runs_with_spread_and_calibration():
    replay = run_gender_replay(
        GoldConfig(
            gender="M",
            model_family="gold_harry_xgb_spread",
            calibration_mode="isotonic_gender",
            feature_profile="gold_harry_m",
        )
    )

    assert replay["model_family"] == "gold_harry_xgb_spread"
    assert replay["feature_profile"] == "gold_harry_m"
    assert replay["rating_profile"] == "harry_rating_core"
    assert replay["predictions"]["Prob"].between(0.0, 1.0).all()
