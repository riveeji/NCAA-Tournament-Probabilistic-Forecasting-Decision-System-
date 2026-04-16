import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_benchmark_slice_comparison.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_benchmark_slice_comparison", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_comparison_rows_computes_delta_and_winner():
    module = _load_module()
    slices = pd.DataFrame(
        [
            {
                "system": "ji_base_frozen",
                "slice_type": "period_bucket",
                "slice_value": "latest",
                "gender": "M",
                "rows": 100,
                "calibrated_brier": 0.10,
            },
            {
                "system": "gold_recover_proxy",
                "slice_type": "period_bucket",
                "slice_value": "latest",
                "gender": "M",
                "rows": 100,
                "calibrated_brier": 0.12,
            },
        ]
    )

    comparison = module._build_comparison_rows(slices)
    row = comparison.iloc[0]

    assert row["ji_minus_gold_calibrated_brier"] == pytest.approx(-0.02)
    assert row["winner"] == "ji_base_frozen"


def test_write_markdown_mentions_old_hc_omission(tmp_path):
    module = _load_module()
    module.DOCS = tmp_path
    snapshot = {"official_lb_best_score": 0.1278438}
    comparison = pd.DataFrame(
        [
            {
                "slice_type": "period_bucket",
                "slice_value": "latest",
                "gender": "ALL",
                "rows_min": 100,
                "calibrated_brier_ji_base_frozen": 0.12,
                "calibrated_brier_gold_recover_proxy": 0.13,
                "ji_minus_gold_calibrated_brier": -0.01,
                "winner": "ji_base_frozen",
            },
            {
                "slice_type": "upset_bucket",
                "slice_value": "upset_gap2plus",
                "gender": "W",
                "rows_min": 120,
                "calibrated_brier_ji_base_frozen": 0.30,
                "calibrated_brier_gold_recover_proxy": 0.28,
                "ji_minus_gold_calibrated_brier": 0.02,
                "winner": "gold_recover_proxy",
            },
        ]
    )

    module._write_markdown(snapshot, comparison, gold_official_lb=0.1289, old_hc_replay=0.1602)

    output = (tmp_path / "JI_BASE_BENCHMARK_SLICE_COMPARISON.md").read_text(encoding="utf-8")
    assert "old_hc" in output
    assert "omitted from slice comparison" in output
    assert "gold_recover_proxy" in output
