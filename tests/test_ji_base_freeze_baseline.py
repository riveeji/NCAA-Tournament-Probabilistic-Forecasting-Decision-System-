import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "freeze_ji_base_baseline.py"
    spec = importlib.util.spec_from_file_location("freeze_ji_base_baseline", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_best_overall_lb_entry_prefers_lowest_across_all_submission_profiles(tmp_path):
    module = _load_module()
    module.RESULTS = tmp_path
    pd.DataFrame(
        [
            {"date": "2026-04-13", "submission_profile": "ji_base_overlay_v1", "official_lb": 0.1273, "notes": "overlay"},
            {"date": "2026-04-13", "submission_profile": "ji_base_base", "official_lb": 0.1231, "notes": "core"},
        ]
    ).to_csv(tmp_path / "official_lb_log.csv", index=False)

    best = module._load_best_overall_lb_entry()

    assert best["submission_profile"] == "ji_base_base"
    assert float(best["official_lb"]) == 0.1231
