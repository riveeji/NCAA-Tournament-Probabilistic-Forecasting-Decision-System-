import importlib.util
from pathlib import Path

import pandas as pd

from hc.v2 import V2Config, run_gender_replay


def _load_run_v2_replay_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_v2_baseline_replay.py"
    spec = importlib.util.spec_from_file_location("run_v2_baseline_replay", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_gender_replay_outputs_required_fields():
    replay = run_gender_replay(V2Config(gender="M", route="probability", model_variant="lr", market_mode="none"))
    assert replay["gender"] == "M"
    assert replay["route"] == "probability"
    assert "mean_brier" in replay
    assert "recent_window_brier" in replay
    assert "brier_variance" in replay
    assert not replay["by_season"].empty
    assert not replay["predictions"].empty


def test_replay_handles_single_class_training_folds():
    replay = run_gender_replay(V2Config(gender="W", route="probability", model_variant="tree", market_mode="sportsbook"))
    assert replay["mean_brier"] >= 0.0


def test_spread_route_replay_outputs_probabilities_and_route_metadata():
    replay = run_gender_replay(V2Config(gender="M", route="spread", model_variant="lr", market_mode="none"))
    assert replay["route"] == "spread"
    assert replay["latest_season"] >= 2025
    assert replay["latest_season_brier"] >= 0.0
    assert replay["predictions"]["Prob"].between(0.0, 1.0).all()
    assert replay["predictions"]["route"].eq("spread").all()
    assert replay["by_season"]["route"].eq("spread").all()


def test_spread_route_replay_emits_feature_pack_calibration_and_learner_metadata():
    replay = run_gender_replay(
        V2Config(
            gender="M",
            route="spread",
            model_variant="tree",
            market_mode="sportsbook",
            feature_pack="efficiency",
            calibration_mode="gendercal",
        )
    )

    assert replay["learner_family"] == "tree"
    assert replay["feature_pack"] == "efficiency"
    assert replay["calibration_mode"] == "gendercal"
    assert replay["predictions"]["feature_pack"].eq("efficiency").all()
    assert replay["predictions"]["calibration_mode"].eq("gendercal").all()
    assert replay["predictions"]["learner_family"].eq("tree").all()
    assert replay["by_season"]["feature_pack"].eq("efficiency").all()
    assert replay["by_season"]["calibration_mode"].eq("gendercal").all()
    assert replay["by_season"]["learner_family"].eq("tree").all()


def test_external_base_pruned_replay_emits_gender_profile_metadata():
    replay = run_gender_replay(
        V2Config(
            gender="M",
            route="spread",
            model_variant="lr",
            market_mode="none",
            feature_pack="external_base_pruned",
            calibration_mode="basecal",
        )
    )

    assert replay["feature_pack"] == "external_base_pruned"
    assert replay["gender_profile"] == "men_external_pruned"
    assert replay["predictions"]["gender_profile"].eq("men_external_pruned").all()
    assert replay["by_season"]["gender_profile"].eq("men_external_pruned").all()


def test_spread_route_monotonic_calibration_outputs_bounded_probabilities():
    replay = run_gender_replay(
        V2Config(
            gender="W",
            route="spread",
            model_variant="lr",
            market_mode="none",
            feature_pack="opp_adjusted",
            calibration_mode="monotoniccal",
        )
    )
    assert replay["predictions"]["Prob"].between(0.0, 1.0).all()


def test_run_v2_baseline_replay_bootstraps_repo_root_before_hc_import():
    script = (Path(__file__).resolve().parents[1] / "tools" / "run_v2_baseline_replay.py").read_text(encoding="utf-8")
    root_idx = script.index("ROOT = Path(__file__).resolve().parents[1]")
    path_idx = script.index("sys.path.insert(0, str(ROOT))")
    import_idx = script.index("from hc.v2 import")
    assert root_idx < path_idx < import_idx


def test_run_v2_baseline_replay_defaults_to_minimal_baseline_variants():
    module = _load_run_v2_replay_module()
    plan = module.resolve_variant_plan(include_controls=False, include_prediction_market=False)
    assert len(plan) == 3
    assert all(route == "spread" for route, *_ in plan)
    assert {model_variant for _, model_variant, _, _, _ in plan} == {"lr"}
    assert {market_mode for _, _, market_mode, _, _ in plan} == {"none", "sportsbook"}
    assert {feature_pack for _, _, _, feature_pack, _ in plan} == {"base", "external_base_pruned"}
    assert {calibration_mode for _, _, _, _, calibration_mode in plan} == {"basecal"}
    assert ("spread", "lr", "none", "base", "basecal") in plan
    assert ("spread", "lr", "none", "external_base_pruned", "basecal") in plan
    assert ("spread", "lr", "sportsbook", "external_base_pruned", "basecal") in plan


def test_run_v2_baseline_replay_can_add_controls_and_prediction_market():
    module = _load_run_v2_replay_module()
    plan = module.resolve_variant_plan(include_controls=True, include_prediction_market=True)
    assert len(plan) == 9
    assert ("spread", "tree", "none", "external_base_pruned", "basecal") in plan
    assert ("spread", "lr", "sportsbook_prediction", "external_base_pruned", "basecal") in plan
    assert ("spread", "lr", "none", "external_base", "basecal") in plan
    assert ("spread", "lr", "none", "strength_full", "basecal") in plan
    assert ("spread", "lr", "none", "external_base_pruned", "gendercal") in plan
    assert ("spread", "lr", "none", "external_base_pruned", "monotoniccal") in plan


def test_write_replay_outputs_skips_large_detail_files_by_default(tmp_path):
    module = _load_run_v2_replay_module()
    summary_rows = [
        {
            "gender": "M",
            "route": "spread",
            "model_variant": "lr",
            "learner_family": "linear",
            "market_mode": "none",
            "feature_pack": "base",
            "calibration_mode": "basecal",
            "mean_brier": 0.1,
        }
    ]
    combined_rows = [
        {
            "variant": "spread-linear:none@base+basecal",
            "route": "spread",
            "model_variant": "lr",
            "learner_family": "linear",
            "market_mode": "none",
            "feature_pack": "base",
            "calibration_mode": "basecal",
            "equal_gender_mean_brier": 0.1,
            "latest_season": 2026,
            "equal_gender_latest_season_brier": 0.09,
        }
    ]
    by_season = pd.DataFrame(
        [
            {
                "season": 2025,
                "gender": "M",
                "route": "spread",
                "model_variant": "lr",
                "learner_family": "linear",
                "market_mode": "none",
                "feature_pack": "base",
                "calibration_mode": "basecal",
                "brier": 0.1,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "Season": 2025,
                "T1": 1,
                "T2": 2,
                "Prob": 0.6,
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "base",
                "calibration_mode": "basecal",
            }
        ]
    )

    module.write_replay_outputs(
        output_dir=tmp_path,
        summary_rows=summary_rows,
        combined_rows=combined_rows,
        by_season_df=by_season,
        predictions_df=predictions,
        write_detailed_outputs=False,
    )

    assert (tmp_path / "v2_replay_summary.csv").exists()
    assert (tmp_path / "v2_replay_combined.csv").exists()
    assert (tmp_path / "v2_replay_summary.json").exists()
    assert not (tmp_path / "v2_replay_by_season.csv").exists()
    assert not (tmp_path / "v2_replay_predictions.csv").exists()
