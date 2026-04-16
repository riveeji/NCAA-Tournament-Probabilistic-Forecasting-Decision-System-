import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_women_slice_diagnostics.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_women_slice_diagnostics", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slice_summary_emits_feature_rows():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "upset_bucket": ["upset_gap2plus"] * 12,
            "seed_gap_bucket": ["gap_0_1"] * 12,
            "period_bucket": ["recent"] * 12,
            "calibrated_brier": [0.2 + i * 0.01 for i in range(12)],
            "calibrated_prob": [0.3 + i * 0.01 for i in range(12)],
            "Label": [0, 1] * 6,
            "absolute_error": [0.1 + i * 0.01 for i in range(12)],
            "Delta_Quality": [float(i) for i in range(12)],
            "QualityWins_diff": [float(i) * 0.5 for i in range(12)],
            "OpponentQualityTournamentRank_diff": [float(i) * -0.25 for i in range(12)],
            "AvgBlkDiff_diff": [float(i) * 0.2 for i in range(12)],
            "Seed_x_Quality": [float(i) * 1.5 for i in range(12)],
            "WomenCompositeQuality_diff": [float(i) * 0.3 for i in range(12)],
        }
    )

    rows = module._slice_summary(frame, slice_type="upset_bucket", slice_value="upset_gap2plus")
    summary = pd.DataFrame(rows)

    assert not summary.empty
    assert "Delta_Quality" in set(summary["feature"])
    assert (summary["rows"] == 12).all()


def test_write_markdown_contains_target_slice(tmp_path):
    module = _load_module()
    module.DOCS = tmp_path
    summary = pd.DataFrame(
        [
            {
                "slice_type": "upset_bucket",
                "slice_value": "upset_gap2plus",
                "rows": 120,
                "calibrated_brier": 0.55,
                "avg_calibrated_prob": 0.44,
                "empirical_win_rate": 0.32,
                "feature": "Delta_Quality",
                "feature_mean": 0.1,
                "feature_abs_mean": 0.2,
                "high_error_feature_mean": 0.3,
                "high_error_feature_abs_mean": 0.4,
                "error_corr": 0.5,
                "brier_corr": 0.4,
            }
        ]
    )

    module._write_markdown(summary)

    output = (tmp_path / "JI_BASE_WOMEN_SLICE_DIAGNOSTICS.md").read_text(encoding="utf-8")
    assert "upset_gap2plus" in output
    assert "Delta_Quality" in output


def test_women_conservative_seed_quality_formula_shrinks_interaction():
    from hc.ji_base import JIBaseConfig
    from hc.ji_base.data import build_submission_feature_frame

    ids = pd.DataFrame([{"Season": 2025, "T1": 1, "T2": 2}])
    team_features = pd.DataFrame(
        [
            {
                "Season": 2025,
                "TeamID": 1,
                "SeedNum": 8,
                "Quality": 0.8,
                "Elo": 1600,
                "neff": 12.0,
                "QualityWins": 0.5,
                "OpponentQualityTournamentRank": 0.3,
                "AvgBlkDiff": 0.2,
            },
            {
                "Season": 2025,
                "TeamID": 2,
                "SeedNum": 5,
                "Quality": -0.2,
                "Elo": 1500,
                "neff": 6.0,
                "QualityWins": -0.1,
                "OpponentQualityTournamentRank": -0.2,
                "AvgBlkDiff": -0.1,
            },
        ]
    )

    base = build_submission_feature_frame(
        ids,
        team_features,
        JIBaseConfig(gender="W", feature_profile="seed_quality_interaction", alpha_profile="none"),
    )
    conservative = build_submission_feature_frame(
        ids,
        team_features,
        JIBaseConfig(gender="W", feature_profile="seed_quality_interaction_women_conservative", alpha_profile="none"),
    )

    assert abs(conservative.loc[0, "Seed_x_Quality"]) < abs(base.loc[0, "Seed_x_Quality"])
