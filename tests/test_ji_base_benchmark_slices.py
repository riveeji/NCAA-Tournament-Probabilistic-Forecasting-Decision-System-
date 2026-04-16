import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_benchmark_slices.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_benchmark_slices", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bucket_helpers_are_stable():
    module = _load_module()

    assert module._seed_gap_bucket(0) == "gap_0_1"
    assert module._seed_gap_bucket(3) == "gap_2_4"
    assert module._seed_gap_bucket(7) == "gap_5_8"
    assert module._seed_gap_bucket(10) == "gap_9_plus"

    assert module._favorite_seed_bucket(1) == "seed_1_2"
    assert module._favorite_seed_bucket(4) == "seed_3_4"
    assert module._favorite_seed_bucket(7) == "seed_5_8"
    assert module._favorite_seed_bucket(11) == "seed_9_12"
    assert module._favorite_seed_bucket(16) == "seed_13_16"

    assert module._upset_bucket(1, True) == "tossup"
    assert module._upset_bucket(5, True) == "favorite_win_gap2plus"
    assert module._upset_bucket(5, False) == "upset_gap2plus"


def test_build_slice_rows_includes_overall_and_grouped_rows():
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "gender": "M",
                "raw_brier": 0.10,
                "calibrated_brier": 0.09,
                "favorite_won": True,
                "seed_gap_abs": 1.0,
                "period_bucket": "latest",
                "seed_gap_bucket": "gap_0_1",
                "favorite_seed_bucket": "seed_1_2",
                "upset_bucket": "tossup",
            },
            {
                "gender": "W",
                "raw_brier": 0.20,
                "calibrated_brier": 0.18,
                "favorite_won": False,
                "seed_gap_abs": 6.0,
                "period_bucket": "historical",
                "seed_gap_bucket": "gap_5_8",
                "favorite_seed_bucket": "seed_5_8",
                "upset_bucket": "upset_gap2plus",
            },
        ]
    )

    rows = pd.DataFrame(module._build_slice_rows(frame))

    assert ((rows["slice_type"] == "overall") & (rows["gender"] == "ALL")).any()
    assert ((rows["slice_type"] == "period_bucket") & (rows["slice_value"] == "latest") & (rows["gender"] == "M")).any()
    assert ((rows["slice_type"] == "upset_bucket") & (rows["slice_value"] == "upset_gap2plus") & (rows["gender"] == "W")).any()


def test_write_markdown_outputs_slice_summary(tmp_path):
    module = _load_module()
    module.DOCS = tmp_path
    snapshot = {
        "working_baseline_candidate": "alpha::baseline",
        "base_model_profile": "JI_lr_control",
        "feature_profile": "seed_quality_interaction",
        "alpha_profile": "quality_only_men_quality_blocks_women",
        "women_quality_profile_w": "consensus_rebuild_v4",
    }
    slices = pd.DataFrame(
        [
            {
                "slice_type": "period_bucket",
                "slice_value": "latest",
                "gender": "M",
                "rows": 100,
                "raw_brier": 0.1,
                "calibrated_brier": 0.09,
                "favorite_win_rate": 0.8,
                "avg_seed_gap_abs": 2.0,
            },
            {
                "slice_type": "seed_gap_bucket",
                "slice_value": "gap_9_plus",
                "gender": "W",
                "rows": 120,
                "raw_brier": 0.2,
                "calibrated_brier": 0.19,
                "favorite_win_rate": 0.7,
                "avg_seed_gap_abs": 9.5,
            },
        ]
    )

    module._write_markdown(snapshot, slices)

    output = (tmp_path / "JI_BASE_BENCHMARK_SLICES.md").read_text(encoding="utf-8")
    assert "Worst Slices" in output
    assert "gap_9_plus" in output
