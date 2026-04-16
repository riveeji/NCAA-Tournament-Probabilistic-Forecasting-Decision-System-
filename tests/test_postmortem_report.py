import importlib.util
import math
from pathlib import Path

import pandas as pd


def _load_postmortem_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_postmortem_report.py"
    spec = importlib.util.spec_from_file_location("build_postmortem_report", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_v2_metrics_works_without_detailed_by_season_file(tmp_path):
    module = _load_postmortem_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {
                "gender": "M",
                "route": "spread",
                "model_variant": "lr",
                "learner_family": "linear",
                "market_mode": "none",
                "feature_pack": "base",
                "calibration_mode": "basecal",
                "mean_brier": 0.18,
            },
            {
                "gender": "W",
                "route": "spread",
                "model_variant": "lr",
                "learner_family": "linear",
                "market_mode": "none",
                "feature_pack": "base",
                "calibration_mode": "basecal",
                "mean_brier": 0.14,
            },
        ]
    ).to_csv(results_dir / "v2_replay_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "spread-linear:none@base+basecal",
                "route": "spread",
                "model_variant": "lr",
                "learner_family": "linear",
                "market_mode": "none",
                "feature_pack": "base",
                "calibration_mode": "basecal",
                "equal_gender_mean_brier": 0.16,
                "men_mean_brier": 0.18,
                "women_mean_brier": 0.14,
            }
        ]
    ).to_csv(results_dir / "v2_replay_combined.csv", index=False)
    module.RESULTS = results_dir

    rows, seasonal_frames, combined = module._load_v2_metrics()

    assert len(rows) == 2
    assert seasonal_frames == []
    assert list(combined["variant"]) == ["spread-linear:none@base+basecal"]
    assert rows[0]["variant"] == "spread-linear:none@base+basecal"


def test_variant_metric_returns_nan_when_optional_variant_is_missing():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {"variant": "probability:none", "equal_gender_mean_brier": 0.1619},
            {"variant": "spread:sportsbook", "equal_gender_mean_brier": 0.1618},
        ]
    )

    assert module._variant_metric(combined, "probability:none") == 0.1619
    assert math.isnan(module._variant_metric(combined, "spread:sportsbook_prediction"))


def test_best_variant_can_filter_spread_strength_feature_pack_controls():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {
                "variant": "spread-linear:none@base+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "base",
                "equal_gender_mean_brier": 0.1620,
            },
            {
                "variant": "spread-linear:none@strength_full+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "strength_full",
                "equal_gender_mean_brier": 0.1615,
            },
            {
                "variant": "spread-linear:none@strength_recent+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "strength_recent",
                "equal_gender_mean_brier": 0.1617,
            },
        ]
    )

    assert module._best_variant(combined, learner_family="linear", route="spread") == "spread-linear:none@strength_full+basecal"


def test_best_variant_prefers_current_year_metric_when_available():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {
                "variant": "spread-linear:none@base+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "base",
                "equal_gender_mean_brier": 0.1614,
                "equal_gender_latest_season_brier": 0.1700,
            },
            {
                "variant": "spread-linear:none@strength_full+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "strength_full",
                "equal_gender_mean_brier": 0.1617,
                "equal_gender_latest_season_brier": 0.1680,
            },
        ]
    )

    assert module._best_variant(combined, learner_family="linear", route="spread") == "spread-linear:none@strength_full+basecal"


def test_variant_metric_and_best_variant_support_external_base_summary_fields():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {
                "variant": "spread-linear:none@base+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "base",
                "equal_gender_mean_brier": 0.1615,
            },
            {
                "variant": "spread-linear:none@external_base+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "external_base",
                "equal_gender_mean_brier": 0.1609,
            },
            {
                "variant": "spread-tree:none@external_base+basecal",
                "route": "spread",
                "learner_family": "tree",
                "feature_pack": "external_base",
                "equal_gender_mean_brier": 0.1623,
            },
        ]
    )

    assert module._best_variant(combined, learner_family="linear", route="spread", feature_pack="external_base") == (
        "spread-linear:none@external_base+basecal"
    )
    assert module._variant_metric(combined, "spread-linear:none@external_base+basecal") == 0.1609


def test_load_next_year_overlay_metadata_returns_empty_when_missing(tmp_path):
    module = _load_postmortem_module()
    module.RESULTS = tmp_path / "results"
    module.RESULTS.mkdir()

    assert module._load_next_year_overlay_metadata() == {}


def test_load_next_year_overlay_metadata_prefers_gold_submission_summary_when_present(tmp_path):
    module = _load_postmortem_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "submission_stage2_gold_summary.json").write_text(
        """
        {
          "season": 2026,
          "overlay_enabled": true,
          "submission_profile": {"M": "gold_recover_market", "W": "gold_recover_market"},
          "candidate_outputs": {"gold_recover_base": "results/submission_stage2_gold_gold_recover_base.csv"},
          "men": {"season": 2026, "rows": 10, "mean_abs_delta": 0.001, "max_abs_delta": 0.01, "overlay_submission_only_enabled": true, "injury_applied_rows": 2, "market_applied_rows": 3, "sharpen_applied_rows": 1},
          "women": {"season": 2026, "rows": 10, "mean_abs_delta": 0.0001, "max_abs_delta": 0.005, "overlay_submission_only_enabled": true, "injury_applied_rows": 0, "market_applied_rows": 2, "sharpen_applied_rows": 0}
        }
        """,
        encoding="utf-8",
    )
    module.RESULTS = results_dir

    metadata = module._load_next_year_overlay_metadata()

    assert metadata["enabled"] is True
    assert metadata["submission_profile"]["M"] == "gold_recover_market"
    assert metadata["genders"]["M"]["sharpen_applied_rows"] == 1


def test_load_next_year_overlay_metadata_surfaces_gold_recover_as_active_submission(tmp_path):
    module = _load_postmortem_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "submission_stage2_gold_summary.json").write_text(
        """
        {
          "season": 2026,
          "overlay_enabled": true,
          "submission_profile": {"M": "gold_recover_market", "W": "gold_recover_market"},
          "candidate_outputs": {},
          "men": {"season": 2026, "rows": 10, "mean_abs_delta": 0.001, "max_abs_delta": 0.01, "overlay_submission_only_enabled": true, "injury_applied_rows": 2, "market_applied_rows": 3, "sharpen_applied_rows": 1},
          "women": {"season": 2026, "rows": 10, "mean_abs_delta": 0.0001, "max_abs_delta": 0.005, "overlay_submission_only_enabled": true, "injury_applied_rows": 0, "market_applied_rows": 2, "sharpen_applied_rows": 0}
        }
        """,
        encoding="utf-8",
    )
    module.RESULTS = results_dir

    metadata = module._load_next_year_overlay_metadata()

    assert metadata["submission_profile"] == {"M": "gold_recover_market", "W": "gold_recover_market"}


def test_postmortem_best_variant_can_split_external_base_by_gender_profile():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {
                "variant": "spread-linear:none@external_base_pruned+basecal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "external_base_pruned",
                "gender_profile": "men_external_pruned",
                "equal_gender_mean_brier": 0.1606,
                "men_mean_brier": 0.1810,
                "women_mean_brier": 0.1402,
            },
            {
                "variant": "spread-linear:none@external_base_pruned+gendercal",
                "route": "spread",
                "learner_family": "linear",
                "feature_pack": "external_base_pruned",
                "gender_profile": "women_external_pruned",
                "equal_gender_mean_brier": 0.1604,
                "men_mean_brier": 0.1815,
                "women_mean_brier": 0.1393,
            },
        ]
    )

    assert module._best_variant(combined, learner_family="linear", route="spread", feature_pack="external_base_pruned") == (
        "spread-linear:none@external_base_pruned+gendercal"
    )


def test_load_gold_metrics_returns_empty_when_gold_outputs_are_missing(tmp_path):
    module = _load_postmortem_module()
    module.RESULTS = tmp_path / "results"
    module.RESULTS.mkdir()

    rows, seasonal_frames, combined = module._load_gold_metrics()

    assert rows == []
    assert seasonal_frames == []
    assert combined.empty


def test_load_official_lb_log_returns_rows_when_present(tmp_path):
    module = _load_postmortem_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {"submission_profile": "gold_recover_market", "official_lb": 0.1289},
            {"submission_profile": "gold_recover_base", "official_lb": 0.1306},
        ]
    ).to_csv(results_dir / "official_lb_log.csv", index=False)
    module.RESULTS = results_dir

    frame = module._load_official_lb_log()

    assert list(frame["submission_profile"]) == ["gold_recover_market", "gold_recover_base"]
    assert frame["official_lb"].tolist() == [0.1289, 0.1306]


def test_best_ji_base_candidate_prefers_candidate_name_over_variant():
    module = _load_postmortem_module()
    combined = pd.DataFrame(
        [
            {
                "candidate_name": "feature::seed_quality_interaction",
                "variant": "JI_lr_control@none",
                "total_cv_brier_calibrated": 0.163960,
                "women_cv_brier_calibrated": 0.143711,
            },
            {
                "candidate_name": "alpha::quality_only",
                "variant": "JI_lr_control@none",
                "total_cv_brier_calibrated": 0.163907,
                "women_cv_brier_calibrated": 0.143833,
            },
        ]
    )

    assert module._best_ji_base_candidate_label(combined) == "alpha::quality_only"


def test_external_source_inventory_script_writes_expected_tiers(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_external_source_inventory.py"
    spec = importlib.util.spec_from_file_location("build_external_source_inventory", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "external_source_inventory.csv"
    module.write_inventory(output)
    frame = pd.read_csv(output)

    assert {"source_name", "layer", "signal_horizon", "tier", "default_enabled", "notes"}.issubset(frame.columns)
    assert set(frame["tier"]) >= {"A", "B", "C"}
    assert "direct_matchup_market" in set(frame["source_name"])
