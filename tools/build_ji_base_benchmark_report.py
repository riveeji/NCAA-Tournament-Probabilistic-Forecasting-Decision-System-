from __future__ import annotations

import json
import re
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


def _load_baseline_summary() -> dict[str, Any]:
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    candidate_name = snapshot.get("working_baseline_candidate")
    if not candidate_name:
        raise KeyError("Frozen baseline snapshot is missing working_baseline_candidate")
    combined_path = RESULTS / "ji_base_replay_combined.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        match = combined.loc[combined["candidate_name"] == candidate_name]
        if not match.empty:
            return match.sort_values(["total_cv_brier_calibrated", "women_cv_brier_calibrated"]).iloc[0].to_dict()

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(candidate_name).strip()).strip("_").lower() or "candidate"
    challenger_path = RESULTS / f"ji_base_challenger_{slug}.json"
    challenger_payload = _load_json(challenger_path)
    challenger = challenger_payload.get("challenger_summary")
    if isinstance(challenger, dict) and challenger:
        return challenger
    raise KeyError(f"Frozen baseline candidate '{candidate_name}' not found in ji_base replay combined or challenger outputs")


def _official_lb_by_candidate(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    mapping: dict[str, float] = {}
    core_rows = frame.loc[
        frame["submission_profile"].astype(str).str.startswith("ji_base_")
        & ~frame["submission_profile"].astype(str).str.startswith("ji_base_overlay")
        & frame["official_lb"].notna()
    ]
    for row in core_rows.sort_values(["official_lb", "date"], ascending=[True, True]).to_dict(orient="records"):
        submission_profile = str(row["submission_profile"])
        suffix = submission_profile.removeprefix("ji_base_")
        candidate_name = f"core::{suffix}"
        mapping.setdefault(candidate_name, float(row["official_lb"]))
    return mapping


def _classify_challenger(
    payload: dict[str, Any],
    baseline: dict[str, Any],
    official_lb_by_candidate: dict[str, float],
    best_official_score: float | None,
) -> str:
    if payload.get("passes_gate"):
        candidate_name = str(payload.get("candidate_name", ""))
        official_lb = official_lb_by_candidate.get(candidate_name)
        if official_lb is not None and best_official_score is not None:
            if official_lb < best_official_score - 1e-12:
                return "official_lb_passed"
            return "replay_passed_but_lb_failed"
        return "replay_passed"
    challenger = payload.get("challenger_summary", {})
    if not challenger:
        return "invalid"
    fields = (
        "total_cv_brier_calibrated",
        "women_cv_brier_calibrated",
        "latest_season_equal_gender_brier",
        "recent_window_equal_gender_brier",
    )
    if all(abs(float(challenger.get(field, 0.0)) - float(baseline.get(field, 0.0))) <= 1e-12 for field in fields):
        return "equivalent"
    return "rejected"


def _recommended_action(status: str) -> str:
    if status == "official_lb_passed":
        return "eligible_for_promotion"
    if status == "replay_passed_but_lb_failed":
        return "pause_direction"
    if status == "replay_passed":
        return "eligible_for_official_lb"
    if status == "equivalent":
        return "freeze_no_change"
    if status == "rejected":
        return "archive_direction"
    return "inspect_manually"


def _load_challenger_registry(
    baseline: dict[str, Any],
    official_lb_by_candidate: dict[str, float],
    best_official_score: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("ji_base_challenger_*.json")):
        if path.name == "ji_base_challenger_registry.json":
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        challenger = payload.get("challenger_summary", {})
        candidate_name = payload.get("candidate_name")
        status = _classify_challenger(payload, baseline, official_lb_by_candidate, best_official_score)
        rows.append(
            {
                "candidate_name": candidate_name,
                "result_file": str(path),
                "status": status,
                "recommended_action": _recommended_action(status),
                "passes_gate": bool(payload.get("passes_gate")),
                "official_lb": official_lb_by_candidate.get(str(candidate_name)),
                "model_family": challenger.get("model_family_m"),
                "calibration_mode": challenger.get("calibration_mode_w") or challenger.get("calibration_mode_m"),
                "feature_profile": challenger.get("feature_profile_w") or challenger.get("feature_profile_m"),
                "alpha_profile": challenger.get("alpha_profile_w") or challenger.get("alpha_profile_m"),
                "women_quality_profile_w": challenger.get("women_quality_profile_w"),
                "isotonic_min_samples_w": challenger.get("isotonic_min_samples_w"),
                "total_cv_brier_calibrated": challenger.get("total_cv_brier_calibrated"),
                "women_cv_brier_calibrated": challenger.get("women_cv_brier_calibrated"),
                "latest_season_equal_gender_brier": challenger.get("latest_season_equal_gender_brier"),
                "recent_window_equal_gender_brier": challenger.get("recent_window_equal_gender_brier"),
                "delta_total_cv_brier_calibrated": (
                    None
                    if challenger.get("total_cv_brier_calibrated") is None
                    else float(challenger["total_cv_brier_calibrated"]) - float(baseline["total_cv_brier_calibrated"])
                ),
                "delta_women_cv_brier_calibrated": (
                    None
                    if challenger.get("women_cv_brier_calibrated") is None
                    else float(challenger["women_cv_brier_calibrated"]) - float(baseline["women_cv_brier_calibrated"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_official_lb_frame() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "official_lb_log.csv")
    frame["official_lb"] = pd.to_numeric(frame["official_lb"], errors="coerce")
    return frame.sort_values(["official_lb", "date"], ascending=[True, True]).reset_index(drop=True)


def _best_official_lb_for_prefix(frame: pd.DataFrame, prefix: str) -> tuple[str | None, float | None]:
    if frame.empty:
        return None, None
    matches = frame.loc[
        frame["submission_profile"].astype(str).str.startswith(prefix) & frame["official_lb"].notna()
    ].sort_values(["official_lb", "date"], ascending=[True, True])
    if matches.empty:
        return None, None
    best = matches.iloc[0]
    return str(best["submission_profile"]), float(best["official_lb"])


def _select_best_known_submission_layer(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "submission_profile": snapshot.get("current_best_submission_profile", snapshot.get("best_overlay_submission_profile")),
        "score": snapshot.get("current_best_submission_score", snapshot.get("best_overlay_submission_score")),
    }


def _build_report_payload() -> dict[str, Any]:
    snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")
    postmortem = _load_json(RESULTS / "postmortem_summary.json")
    baseline = _load_baseline_summary()
    official_lb = _load_official_lb_frame()
    official_lb_by_candidate = _official_lb_by_candidate(official_lb)
    best_official_score = snapshot.get("current_best_submission_score", snapshot.get("official_lb_best_score"))
    challenger_registry = _load_challenger_registry(baseline, official_lb_by_candidate, best_official_score)

    old_hc_replay = float(baseline["total_cv_brier_calibrated"]) - float(postmortem["ji_base_vs_old_hc_delta"])
    gold_recover_replay = float(baseline["total_cv_brier_calibrated"]) - float(postmortem["ji_base_vs_gold_recover_delta"])
    gold_recover_lb = official_lb.loc[official_lb["submission_profile"] == "gold_recover_market", "official_lb"].min()
    ji_base_lb = official_lb.loc[official_lb["submission_profile"] == "ji_base_base", "official_lb"].min()
    best_overlay_profile, ji_base_overlay_lb = _best_official_lb_for_prefix(official_lb, "ji_base_overlay")

    systems = [
        {
            "system": "ji_base_frozen",
            "replay_total_cv_brier_calibrated": float(baseline["total_cv_brier_calibrated"]),
            "official_lb": float(ji_base_lb) if pd.notna(ji_base_lb) else None,
            "source": "frozen_baseline",
        },
        {
            "system": "gold_recover_market",
            "replay_total_cv_brier_calibrated": gold_recover_replay,
            "official_lb": float(gold_recover_lb) if pd.notna(gold_recover_lb) else None,
            "source": "postmortem_delta+official_lb_log",
        },
        {
            "system": "old_hc",
            "replay_total_cv_brier_calibrated": old_hc_replay,
            "official_lb": None,
            "source": "postmortem_delta",
        },
    ]

    return {
        "snapshot": snapshot,
        "core_submission_profile": snapshot.get("core_submission_profile", snapshot.get("submission_profile")),
        "best_overlay_submission_profile": snapshot.get("best_overlay_submission_profile", snapshot.get("current_best_submission_profile")),
        "frozen_overlay_submission_profile": snapshot.get("frozen_overlay_submission_profile"),
        "frozen_baseline_summary": baseline,
        "frozen_baseline_official_lb": float(ji_base_lb) if pd.notna(ji_base_lb) else None,
        "best_known_submission_layer": _select_best_known_submission_layer(snapshot),
        "overlay_official_lb": {
            "submission_profile": best_overlay_profile,
            "score": ji_base_overlay_lb,
        },
        "systems": systems,
        "challenger_registry": challenger_registry.to_dict(orient="records"),
        "challenger_counts": challenger_registry["status"].value_counts().to_dict() if not challenger_registry.empty else {},
        "best_known_official_lb": _select_best_known_submission_layer(snapshot),
    }


def _write_markdown(report: dict[str, Any], registry: pd.DataFrame) -> None:
    snapshot = report["snapshot"]
    baseline = report["frozen_baseline_summary"]
    frozen_official_lb = report.get("frozen_baseline_official_lb")
    best_known = report.get("best_known_submission_layer", {})
    lines = [
        "# JI_base Benchmark Report",
        "",
        "## Frozen Baseline",
        "",
        f"- Candidate: `{snapshot['working_baseline_candidate']}`",
        f"- Model: `{snapshot['base_model_profile']}`",
        f"- Calibration: `{snapshot['calibration_mode']}`",
        f"- Feature profile: `{snapshot['feature_profile']}`",
        f"- Alpha profile: `{snapshot['alpha_profile']}`",
        f"- Men quality: `{snapshot['women_quality_profile_m']}`",
        f"- Women quality: `{snapshot['women_quality_profile_w']}`",
        f"- Frozen baseline official LB: `{'' if frozen_official_lb is None else frozen_official_lb}`",
        f"- Frozen overlay submission: `{report.get('frozen_overlay_submission_profile')}`",
        f"- Best-known submission layer: `{best_known.get('submission_profile')}` / `{best_known.get('score')}`",
        "",
        "## Replay Benchmark",
        "",
        "| System | Total CV Brier (calibrated) | Official LB | Source |",
        "| --- | ---: | ---: | --- |",
    ]
    for system in report["systems"]:
        official_lb = "" if system["official_lb"] is None else f"{system['official_lb']:.7f}"
        lines.append(
            f"| {system['system']} | {system['replay_total_cv_brier_calibrated']:.9f} | {official_lb} | {system['source']} |"
        )

    lines.extend(
        [
            "",
            "## Frozen Baseline Metrics",
            "",
            f"- `total_cv_brier_calibrated`: `{baseline['total_cv_brier_calibrated']:.9f}`",
            f"- `women_cv_brier_calibrated`: `{baseline['women_cv_brier_calibrated']:.9f}`",
            f"- `latest_season_equal_gender_brier`: `{baseline['latest_season_equal_gender_brier']:.9f}`",
            f"- `recent_window_equal_gender_brier`: `{baseline['recent_window_equal_gender_brier']:.9f}`",
            "",
            "## Challenger Registry",
            "",
            "| Candidate | Status | Action | Official LB | Total delta | Women delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in registry.sort_values(["status", "candidate_name"]).to_dict(orient="records"):
        delta_total = row["delta_total_cv_brier_calibrated"]
        delta_women = row["delta_women_cv_brier_calibrated"]
        official_lb = "" if pd.isna(row.get("official_lb")) else f"{float(row['official_lb']):.7f}"
        lines.append(
            f"| {row['candidate_name']} | {row['status']} | {row['recommended_action']} | {official_lb} | {delta_total:.9f} | {delta_women:.9f} |"
        )

    (DOCS / "JI_BASE_BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    report = _build_report_payload()
    registry = pd.DataFrame(report["challenger_registry"])
    registry.to_csv(RESULTS / "ji_base_challenger_registry.csv", index=False)
    (RESULTS / "ji_base_challenger_registry.json").write_text(
        json.dumps(report["challenger_registry"], indent=2),
        encoding="utf-8",
    )
    (RESULTS / "ji_base_benchmark_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_markdown(report, registry)


if __name__ == "__main__":
    main()
