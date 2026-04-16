import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_benchmark_report", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_challenger_marks_equivalent_when_metrics_match_baseline():
    module = _load_module()
    baseline = {
        "total_cv_brier_calibrated": 0.10,
        "women_cv_brier_calibrated": 0.09,
        "latest_season_equal_gender_brier": 0.08,
        "recent_window_equal_gender_brier": 0.07,
    }
    payload = {
        "passes_gate": False,
        "challenger_summary": dict(baseline),
    }

    assert module._classify_challenger(payload, baseline, official_lb_by_candidate={}, best_official_score=None) == "equivalent"


def test_classify_challenger_marks_replay_passed_and_rejected():
    module = _load_module()
    baseline = {
        "total_cv_brier_calibrated": 0.10,
        "women_cv_brier_calibrated": 0.09,
        "latest_season_equal_gender_brier": 0.08,
        "recent_window_equal_gender_brier": 0.07,
    }
    passed = {"passes_gate": True, "challenger_summary": {}}
    rejected = {
        "passes_gate": False,
        "challenger_summary": {
            "total_cv_brier_calibrated": 0.11,
            "women_cv_brier_calibrated": 0.10,
            "latest_season_equal_gender_brier": 0.081,
            "recent_window_equal_gender_brier": 0.071,
        },
    }

    assert module._classify_challenger(passed, baseline, official_lb_by_candidate={}, best_official_score=None) == "replay_passed"
    assert module._classify_challenger(rejected, baseline, official_lb_by_candidate={}, best_official_score=None) == "rejected"


def test_classify_challenger_marks_replay_passed_but_lb_failed_when_official_check_loses():
    module = _load_module()
    baseline = {
        "total_cv_brier_calibrated": 0.10,
        "women_cv_brier_calibrated": 0.09,
        "latest_season_equal_gender_brier": 0.08,
        "recent_window_equal_gender_brier": 0.07,
    }
    payload = {
        "candidate_name": "core::women_ranking_upstream_v1_external_consensus",
        "passes_gate": True,
        "challenger_summary": {},
    }

    status = module._classify_challenger(
        payload,
        baseline,
        official_lb_by_candidate={"core::women_ranking_upstream_v1_external_consensus": 0.1231352},
        best_official_score=0.1231313,
    )

    assert status == "replay_passed_but_lb_failed"


def test_classify_challenger_marks_official_lb_passed_when_replay_passed_candidate_beats_best_score():
    module = _load_module()
    baseline = {
        "total_cv_brier_calibrated": 0.10,
        "women_cv_brier_calibrated": 0.09,
        "latest_season_equal_gender_brier": 0.08,
        "recent_window_equal_gender_brier": 0.07,
    }
    payload = {
        "candidate_name": "core::example_candidate",
        "passes_gate": True,
        "challenger_summary": {},
    }

    status = module._classify_challenger(
        payload,
        baseline,
        official_lb_by_candidate={"core::example_candidate": 0.1229},
        best_official_score=0.1231313,
    )

    assert status == "official_lb_passed"


def test_write_markdown_emits_registry_rows(tmp_path):
    module = _load_module()
    module.DOCS = tmp_path
    report = {
        "snapshot": {
            "working_baseline_candidate": "alpha::baseline",
            "base_model_profile": "JI_lr_control",
            "calibration_mode": "none",
            "feature_profile": "seed_quality_interaction",
            "alpha_profile": "quality_only_men_quality_blocks_women",
            "women_quality_profile_m": "legacy_v1",
            "women_quality_profile_w": "consensus_rebuild_v4",
            "official_lb_best_score": 0.1278438,
        },
        "frozen_baseline_summary": {
            "total_cv_brier_calibrated": 0.1638,
            "women_cv_brier_calibrated": 0.1436,
            "latest_season_equal_gender_brier": 0.1270,
            "recent_window_equal_gender_brier": 0.1695,
        },
        "systems": [
            {
                "system": "ji_base_frozen",
                "replay_total_cv_brier_calibrated": 0.1638,
                "official_lb": 0.1278438,
                "source": "frozen_baseline",
            }
        ],
    }
    registry = pd.DataFrame(
        [
            {
                "candidate_name": "challenger_a",
                "status": "rejected",
                "recommended_action": "archive_direction",
                "delta_total_cv_brier_calibrated": 0.001,
                "delta_women_cv_brier_calibrated": 0.002,
            }
        ]
    )

    module._write_markdown(report, registry)

    output = (tmp_path / "JI_BASE_BENCHMARK.md").read_text(encoding="utf-8")
    assert "challenger_a" in output
    assert "ji_base_frozen" in output


def test_best_official_lb_for_prefix_selects_lowest_matching_overlay():
    module = _load_module()
    frame = pd.DataFrame(
        [
            {"submission_profile": "ji_base_base", "official_lb": 0.1278, "date": "2026-04-12"},
            {"submission_profile": "ji_base_overlay_v1", "official_lb": 0.1274, "date": "2026-04-13"},
            {"submission_profile": "ji_base_overlay_v1_direct_only", "official_lb": 0.1273, "date": "2026-04-13"},
        ]
    )

    profile, score = module._best_official_lb_for_prefix(frame, "ji_base_overlay")

    assert profile == "ji_base_overlay_v1_direct_only"
    assert score == 0.1273


def test_official_lb_by_candidate_maps_core_submission_profiles_back_to_candidate_names():
    module = _load_module()
    frame = pd.DataFrame(
        [
            {"submission_profile": "ji_base_lr_regularization_v1_c07", "official_lb": 0.1232034, "date": "2026-04-13"},
            {"submission_profile": "ji_base_women_ranking_upstream_v1_external_consensus", "official_lb": 0.1231352, "date": "2026-04-14"},
        ]
    )

    mapping = module._official_lb_by_candidate(frame)

    assert mapping["core::lr_regularization_v1_c07"] == 0.1232034
    assert mapping["core::women_ranking_upstream_v1_external_consensus"] == 0.1231352


def test_select_best_known_submission_layer_prefers_overall_best_over_overlay_best():
    module = _load_module()
    snapshot = {
        "current_best_submission_profile": "ji_base_base",
        "current_best_submission_score": 0.1231,
        "best_overlay_submission_profile": "ji_base_overlay_v1",
        "best_overlay_submission_score": 0.1273,
    }

    best = module._select_best_known_submission_layer(snapshot)

    assert best["submission_profile"] == "ji_base_base"
    assert best["score"] == 0.1231
