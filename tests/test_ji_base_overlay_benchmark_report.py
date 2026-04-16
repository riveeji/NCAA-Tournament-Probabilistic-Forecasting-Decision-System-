import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_ji_base_overlay_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("build_ji_base_overlay_benchmark_report", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_overlay_status_marks_promoted_and_rejected():
    module = _load_module()

    assert module._classify_overlay_status(0.1273, 0.1273, "ji_base_overlay_v1_direct_only", "ji_base_overlay_v1_direct_only") == "promoted"
    assert module._classify_overlay_status(0.1274, 0.1273, "ji_base_overlay_v1", "ji_base_overlay_v1_direct_only") == "rejected"
    assert module._classify_overlay_status(None, 0.1273, "ji_base_overlay_v1_new", "ji_base_overlay_v1_direct_only") == "pending"


def test_build_overlay_registry_uses_best_overlay_submission_from_snapshot():
    module = _load_module()
    workspace = Path.cwd()
    results_dir = workspace / ".pytest_overlay_registry"
    results_dir.mkdir(exist_ok=True)
    module.RESULTS = results_dir
    module.DOCS = results_dir

    pd.DataFrame(
        [
            {"submission_profile": "ji_base_overlay_v1", "official_lb": 0.1273988, "date": "2026-04-13", "notes": "base overlay"},
            {"submission_profile": "ji_base_overlay_v1_direct_only", "official_lb": 0.1273545, "date": "2026-04-13", "notes": "best overlay"},
        ]
    ).to_csv(results_dir / "official_lb_log.csv", index=False)
    (results_dir / "ji_base_baseline_snapshot.json").write_text(
        """
        {
          "current_best_submission_profile": "ji_base_overlay_v1_direct_only",
          "current_best_submission_score": 0.1273545
        }
        """,
        encoding="utf-8",
    )
    for name in ("ji_base_overlay_summary.json", "ji_base_overlay_direct_only_summary.json"):
        (results_dir / name).write_text(
            """
            {
              "audit_path": "audit.csv",
              "candidate_summary_path": "summary.csv",
              "men": {
                "overlay_source_profile": "direct_priority",
                "overlay_stack": "market_injury",
                "market_applied_rows": 10,
                "injury_applied_rows": 100,
                "mean_abs_delta": 0.001
              },
              "women": {
                "overlay_source_profile": "direct_only",
                "overlay_stack": "market_only",
                "market_applied_rows": 2,
                "injury_applied_rows": 0,
                "mean_abs_delta": 0.0
              }
            }
            """,
            encoding="utf-8",
        )

    registry = module.build_overlay_registry()

    promoted = registry.loc[registry["status"] == "promoted"].iloc[0]
    assert promoted["submission_profile"] == "ji_base_overlay_v1_direct_only"


def test_build_overlay_registry_supports_strict_confirmed_best_overlay():
    module = _load_module()
    workspace = Path.cwd()
    results_dir = workspace / ".pytest_overlay_registry_strict"
    results_dir.mkdir(exist_ok=True)
    module.RESULTS = results_dir
    module.DOCS = results_dir

    pd.DataFrame(
        [
            {
                "submission_profile": "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
                "official_lb": 0.1273504,
                "date": "2026-04-13",
                "notes": "best strict confirmed overlay",
            }
        ]
    ).to_csv(results_dir / "official_lb_log.csv", index=False)
    (results_dir / "ji_base_baseline_snapshot.json").write_text(
        """
        {
          "current_best_submission_profile": "ji_base_overlay_v1_direct_only_injury_strict_confirmed",
          "current_best_submission_score": 0.1273504
        }
        """,
        encoding="utf-8",
    )
    (results_dir / "ji_base_overlay_direct_only_injury_strict_confirmed_summary.json").write_text(
        """
        {
          "audit_path": "strict_audit.csv",
          "candidate_summary_path": "strict_summary.csv",
          "men": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_injury",
            "market_applied_rows": 26,
            "injury_applied_rows": 26244,
            "mean_abs_delta": 0.001
          },
          "women": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_only",
            "market_applied_rows": 20,
            "injury_applied_rows": 0,
            "mean_abs_delta": 0.0
          }
        }
        """,
        encoding="utf-8",
    )

    registry = module.build_overlay_registry()

    promoted = registry.loc[registry["status"] == "promoted"].iloc[0]
    assert promoted["submission_profile"] == "ji_base_overlay_v1_direct_only_injury_strict_confirmed"


def test_build_overlay_registry_supports_women_weight060_best_overlay():
    module = _load_module()
    workspace = Path.cwd()
    results_dir = workspace / ".pytest_overlay_registry_weight060"
    results_dir.mkdir(exist_ok=True)
    module.RESULTS = results_dir
    module.DOCS = results_dir

    pd.DataFrame(
        [
            {
                "submission_profile": "ji_base_overlay_v1_direct_only_injury_confirmed4",
                "official_lb": 0.1272824,
                "date": "2026-04-13",
                "notes": "older best overlay",
            },
            {
                "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight060",
                "official_lb": 0.1272400,
                "date": "2026-04-13",
                "notes": "new women weight overlay",
            },
        ]
    ).to_csv(results_dir / "official_lb_log.csv", index=False)
    (results_dir / "ji_base_baseline_snapshot.json").write_text(
        """
        {
          "current_best_submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight060",
          "current_best_submission_score": 0.1272400
        }
        """,
        encoding="utf-8",
    )
    (results_dir / "ji_base_overlay_direct_only_injury_confirmed4_summary.json").write_text(
        """
        {
          "audit_path": "confirmed4_audit.csv",
          "candidate_summary_path": "confirmed4_summary.csv",
          "men": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_injury",
            "market_applied_rows": 26,
            "injury_applied_rows": 9477,
            "mean_abs_delta": 0.001
          },
          "women": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_only",
            "market_applied_rows": 20,
            "injury_applied_rows": 0,
            "mean_abs_delta": 0.0
          }
        }
        """,
        encoding="utf-8",
    )
    (results_dir / "ji_base_overlay_men_best_women_direct_only_weight060_summary.json").write_text(
        """
        {
          "audit_path": "weight060_audit.csv",
          "candidate_summary_path": "weight060_summary.csv",
          "men": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_injury",
            "market_applied_rows": 26,
            "injury_applied_rows": 9477,
            "mean_abs_delta": 0.001
          },
          "women": {
            "overlay_source_profile": "direct_only",
            "overlay_stack": "market_only",
            "market_applied_rows": 20,
            "injury_applied_rows": 0,
            "mean_abs_delta": 0.0
          }
        }
        """,
        encoding="utf-8",
    )

    report = module.build_overlay_benchmark_report()

    assert report["best_overlay_submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight060"
    assert abs(report["best_overlay_official_lb"] - 0.1272400) < 1e-12
    profiles = {row["submission_profile"] for row in report["overlay_registry"]}
    assert "ji_base_overlay_v1_men_best_women_direct_only_weight060" in profiles


def test_build_overlay_benchmark_report_separates_frozen_overlay_from_best_known_official():
    module = _load_module()
    workspace = Path.cwd()
    results_dir = workspace / ".pytest_overlay_registry_frozen_vs_best"
    results_dir.mkdir(exist_ok=True)
    module.RESULTS = results_dir
    module.DOCS = results_dir

    pd.DataFrame(
        [
            {
                "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight025",
                "official_lb": 0.1271800,
                "date": "2026-04-13",
                "notes": "frozen overlay baseline",
            },
            {
                "submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight020",
                "official_lb": 0.1271633,
                "date": "2026-04-13",
                "notes": "best-known official overlay",
            },
        ]
    ).to_csv(results_dir / "official_lb_log.csv", index=False)
    (results_dir / "ji_base_baseline_snapshot.json").write_text(
        """
        {
          "frozen_overlay_submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight025",
          "best_overlay_submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight020",
          "best_overlay_submission_score": 0.1271633,
          "current_best_submission_profile": "ji_base_overlay_v1_men_best_women_direct_only_weight020",
          "current_best_submission_score": 0.1271633
        }
        """,
        encoding="utf-8",
    )
    for name in ("ji_base_overlay_men_best_women_direct_only_weight025_summary.json", "ji_base_overlay_men_best_women_direct_only_weight020_summary.json"):
        (results_dir / name).write_text(
            """
            {
              "audit_path": "audit.csv",
              "candidate_summary_path": "summary.csv",
              "men": {
                "overlay_source_profile": "direct_only",
                "overlay_stack": "market_injury",
                "market_applied_rows": 26,
                "injury_applied_rows": 9477,
                "mean_abs_delta": 0.001
              },
              "women": {
                "overlay_source_profile": "direct_only",
                "overlay_stack": "market_only",
                "market_applied_rows": 20,
                "injury_applied_rows": 0,
                "mean_abs_delta": 0.0
              }
            }
            """,
            encoding="utf-8",
        )

    report = module.build_overlay_benchmark_report()

    assert report["frozen_overlay_submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight025"
    assert report["best_overlay_submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight020"
    assert abs(report["best_overlay_official_lb"] - 0.1271633) < 1e-12
    promoted = [row for row in report["overlay_registry"] if row["status"] == "promoted"]
    assert len(promoted) == 1
    assert promoted[0]["submission_profile"] == "ji_base_overlay_v1_men_best_women_direct_only_weight020"
