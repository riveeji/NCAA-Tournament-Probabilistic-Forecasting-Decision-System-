import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_women_slice_system_comparison.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_women_slice_system_comparison", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slice_summary_emits_ji_vs_gold_delta_rows():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "upset_bucket": ["upset_gap2plus"] * 12,
            "seed_gap_bucket": ["gap_0_1"] * 12,
            "period_bucket": ["recent"] * 12,
            "ji_calibrated_brier": [0.20 + i * 0.01 for i in range(12)],
            "gold_calibrated_brier": [0.18 + i * 0.005 for i in range(12)],
            "ji_minus_gold_brier": [0.02 + i * 0.005 for i in range(12)],
            "ji_better": [False] * 9 + [True] * 3,
            "Delta_Quality": [float(i) for i in range(12)],
            "Seed_x_Quality": [float(i) * 1.5 for i in range(12)],
            "WomenCompositeQuality_diff": [float(i) * 0.2 for i in range(12)],
        }
    )

    rows = module._slice_summary(frame, slice_type="upset_bucket", slice_value="upset_gap2plus")
    summary = pd.DataFrame(rows)

    assert not summary.empty
    assert "Seed_x_Quality" in set(summary["feature"])
    assert (summary["rows"] == 12).all()
    assert (summary["ji_worse_rate"] > 0.5).all()


def test_write_markdown_contains_slice_and_delta(tmp_path):
    module = _load_module()
    module.DOCS = tmp_path
    summary = pd.DataFrame(
        [
            {
                "slice_type": "upset_bucket",
                "slice_value": "upset_gap2plus",
                "rows": 120,
                "ji_calibrated_brier": 0.31,
                "gold_calibrated_brier": 0.28,
                "ji_minus_gold_brier_mean": 0.03,
                "ji_worse_rate": 0.62,
                "feature": "Seed_x_Quality",
                "feature_mean": -0.2,
                "ji_worse_feature_mean": -0.5,
                "delta_corr": -0.6,
            }
        ]
    )

    module._write_markdown(summary)

    output = (tmp_path / "JI_BASE_WOMEN_SLICE_SYSTEM_COMPARISON.md").read_text(encoding="utf-8")
    assert "upset_gap2plus" in output
    assert "Seed_x_Quality" in output
    assert "ji_minus_gold_brier_mean" in output
