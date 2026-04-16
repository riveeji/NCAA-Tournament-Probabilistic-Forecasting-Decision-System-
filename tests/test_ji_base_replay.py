import importlib.util
from pathlib import Path

from hc.ji_base import JIBaseConfig, run_gender_replay


def _load_run_ji_base_replay_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_ji_base_replay.py"
    spec = importlib.util.spec_from_file_location("run_ji_base_replay", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_gender_replay_exposes_raw_and_calibrated_probabilities():
    replay = run_gender_replay(JIBaseConfig(gender="M", calibration_mode="isotonic_gender"))

    assert replay["gender"] == "M"
    assert replay["model_family"] == "JI_spread_xgb"
    assert replay["predictions"]["raw_prob"].between(0.0, 1.0).all()
    assert replay["predictions"]["calibrated_prob"].between(0.0, 1.0).all()
    assert "cv_brier_raw" in replay
    assert "cv_brier_calibrated" in replay


def test_replay_combined_summary_uses_equal_weighted_gender_brier():
    module = _load_run_ji_base_replay_module()
    men = {"cv_brier_raw": 0.1822, "cv_brier_calibrated": 0.1800, "latest_season_brier": 0.16, "recent_window_brier": 0.17}
    women = {"cv_brier_raw": 0.1358, "cv_brier_calibrated": 0.1320, "latest_season_brier": 0.12, "recent_window_brier": 0.13}

    summary = module.build_combined_summary(men=men, women=women)

    assert summary["total_cv_brier_raw"] == (men["cv_brier_raw"] + women["cv_brier_raw"]) / 2.0
    assert summary["total_cv_brier_calibrated"] == (men["cv_brier_calibrated"] + women["cv_brier_calibrated"]) / 2.0
    assert summary["latest_season_equal_gender_brier"] == (men["latest_season_brier"] + women["latest_season_brier"]) / 2.0
    assert summary["recent_window_equal_gender_brier"] == (men["recent_window_brier"] + women["recent_window_brier"]) / 2.0
