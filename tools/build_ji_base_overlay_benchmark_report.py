from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_official_lb_log() -> pd.DataFrame:
    path = RESULTS / "official_lb_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=["submission_profile", "official_lb", "date", "notes"])
    frame = pd.read_csv(path)
    frame["official_lb"] = pd.to_numeric(frame.get("official_lb"), errors="coerce")
    return frame


def _overlay_summary_paths() -> dict[str, Path]:
    return {
        "ji_base_overlay_v1": RESULTS / "ji_base_overlay_summary.json",
        "ji_base_overlay_v1_conservative_injury": RESULTS / "ji_base_overlay_conservative_injury_summary.json",
        "ji_base_overlay_v1_direct_only": RESULTS / "ji_base_overlay_direct_only_summary.json",
        "ji_base_overlay_v1_direct_only_injury_strict_confirmed": RESULTS / "ji_base_overlay_direct_only_injury_strict_confirmed_summary.json",
        "ji_base_overlay_v1_direct_only_injury_confirmed3": RESULTS / "ji_base_overlay_direct_only_injury_confirmed3_summary.json",
        "ji_base_overlay_v1_direct_only_injury_confirmed4": RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_summary.json",
        "ji_base_overlay_v1_direct_only_injury_confirmed5": RESULTS / "ji_base_overlay_direct_only_injury_confirmed5_summary.json",
        "ji_base_overlay_v1_direct_only_injury_confirmed4_shift008": RESULTS / "ji_base_overlay_direct_only_injury_confirmed4_shift008_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_priority": RESULTS / "ji_base_overlay_men_best_women_direct_priority_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight070": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight070_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight060": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight060_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight050": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight050_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight040": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight040_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight030": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight030_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight025": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight025_summary.json",
        "ji_base_overlay_v1_men_best_women_direct_only_weight020": RESULTS / "ji_base_overlay_men_best_women_direct_only_weight020_summary.json",
        "ji_base_overlay_v2_men_player_injury_weight025": RESULTS / "ji_base_overlay_v2_men_player_injury_weight025_summary.json",
    }


def _classify_overlay_status(official_lb: float | None, best_official_lb: float | None, profile: str, best_profile: str | None) -> str:
    if official_lb is None:
        return "pending"
    if best_profile == profile and best_official_lb is not None:
        return "promoted"
    if best_official_lb is not None and abs(official_lb - best_official_lb) <= 1e-12:
        return "equivalent"
    return "rejected"


def build_overlay_registry() -> pd.DataFrame:
    log = _load_official_lb_log()
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    best_profile = snapshot.get("best_overlay_submission_profile", snapshot.get("current_best_submission_profile"))
    best_score = snapshot.get("best_overlay_submission_score", snapshot.get("current_best_submission_score"))

    rows: list[dict[str, Any]] = []
    for profile, summary_path in _overlay_summary_paths().items():
        summary = _load_json(summary_path)
        if not summary:
            continue
        lb_rows = log.loc[log["submission_profile"] == profile].sort_values(["official_lb", "date"], ascending=[True, True])
        official_lb = None if lb_rows.empty else float(lb_rows.iloc[0]["official_lb"])
        notes = None if lb_rows.empty else str(lb_rows.iloc[0].get("notes", ""))
        men = summary.get("men", {})
        women = summary.get("women", {})
        rows.append(
            {
                "submission_profile": profile,
                "status": _classify_overlay_status(official_lb, best_score, profile, best_profile),
                "official_lb": official_lb,
                "official_lb_notes": notes,
                "overlay_source_profile_m": men.get("overlay_source_profile"),
                "overlay_source_profile_w": women.get("overlay_source_profile"),
                "overlay_stack_m": men.get("overlay_stack"),
                "overlay_stack_w": women.get("overlay_stack"),
                "market_applied_rows_m": men.get("market_applied_rows"),
                "market_applied_rows_w": women.get("market_applied_rows"),
                "injury_applied_rows_m": men.get("injury_applied_rows"),
                "injury_applied_rows_w": women.get("injury_applied_rows"),
                "mean_abs_delta_m": men.get("mean_abs_delta"),
                "mean_abs_delta_w": women.get("mean_abs_delta"),
                "audit_path": summary.get("audit_path"),
                "candidate_summary_path": summary.get("candidate_summary_path"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["official_lb", "submission_profile"], ascending=[True, True], na_position="last").reset_index(drop=True)
    else:
        frame = pd.DataFrame(
            columns=[
                "submission_profile",
                "status",
                "official_lb",
                "official_lb_notes",
                "overlay_source_profile_m",
                "overlay_source_profile_w",
                "overlay_stack_m",
                "overlay_stack_w",
                "market_applied_rows_m",
                "market_applied_rows_w",
                "injury_applied_rows_m",
                "injury_applied_rows_w",
                "mean_abs_delta_m",
                "mean_abs_delta_w",
                "audit_path",
                "candidate_summary_path",
            ]
        )
    return frame


def build_overlay_benchmark_report() -> dict[str, Any]:
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    registry = build_overlay_registry()
    best_row = registry.loc[registry["status"] == "promoted"].head(1) if "status" in registry.columns else registry.head(0)
    best_overlay = {} if best_row.empty else best_row.iloc[0].to_dict()
    return {
        "core_submission_profile": snapshot.get("submission_profile", "ji_base_base"),
        "frozen_overlay_submission_profile": snapshot.get("frozen_overlay_submission_profile"),
        "best_overlay_submission_profile": snapshot.get("best_overlay_submission_profile", snapshot.get("current_best_submission_profile")),
        "best_overlay_official_lb": snapshot.get("best_overlay_submission_score", snapshot.get("current_best_submission_score")),
        "overlay_registry": registry.to_dict(orient="records"),
        "best_overlay_candidate": best_overlay,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# JI_base Overlay Benchmark",
        "",
        f"- Frozen core submission profile: `{report.get('core_submission_profile')}`",
        f"- Frozen overlay submission profile: `{report.get('frozen_overlay_submission_profile')}`",
        f"- Best-known overlay submission profile: `{report.get('best_overlay_submission_profile')}`",
        f"- Best-known overlay official LB: `{report.get('best_overlay_official_lb')}`",
        "",
        "## Overlay Registry",
        "",
        "| Profile | Status | Official LB | Men source | Women source | Men injury rows |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in report.get("overlay_registry", []):
        official_lb = "" if row.get("official_lb") is None else f"{float(row['official_lb']):.7f}"
        lines.append(
            f"| {row['submission_profile']} | {row['status']} | {official_lb} | {row.get('overlay_source_profile_m')} | {row.get('overlay_source_profile_w')} | {int(row.get('injury_applied_rows_m') or 0)} |"
        )
    (DOCS / "JI_BASE_OVERLAY_BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    report = build_overlay_benchmark_report()
    registry = pd.DataFrame(report["overlay_registry"])
    registry.to_csv(RESULTS / "ji_base_overlay_registry.csv", index=False)
    (RESULTS / "ji_base_overlay_registry.json").write_text(json.dumps(report["overlay_registry"], indent=2), encoding="utf-8")
    (RESULTS / "ji_base_overlay_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report)


if __name__ == "__main__":
    main()
