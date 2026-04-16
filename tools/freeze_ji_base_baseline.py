from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.ji_base import FROZEN_OVERLAY_SUBMISSION_PROFILE

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_official_lb_entry() -> dict:
    path = RESULTS / "official_lb_log.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    frame["official_lb"] = pd.to_numeric(frame["official_lb"], errors="coerce")
    frame = frame.loc[frame["submission_profile"] == "ji_base_base"].sort_values("official_lb")
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _load_best_overlay_lb_entry() -> dict:
    path = RESULTS / "official_lb_log.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    frame["official_lb"] = pd.to_numeric(frame["official_lb"], errors="coerce")
    frame = frame.loc[frame["submission_profile"].astype(str).str.startswith("ji_base_overlay")].sort_values("official_lb")
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _load_best_overall_lb_entry() -> dict:
    path = RESULTS / "official_lb_log.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    frame["official_lb"] = pd.to_numeric(frame["official_lb"], errors="coerce")
    frame = frame.loc[frame["official_lb"].notna()].sort_values(["official_lb", "date"], ascending=[True, True])
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def main() -> None:
    submission_summary = _load_json(RESULTS / "ji_base_submission_summary.json")
    postmortem = _load_json(RESULTS / "postmortem_summary.json")
    lb_entry = _load_official_lb_entry()
    overlay_lb_entry = _load_best_overlay_lb_entry()
    overall_lb_entry = _load_best_overall_lb_entry()
    profile = (submission_summary.get("profiles") or [{}])[0]

    working_baseline_candidate = f"core::{profile.get('feature_profile')}" if str(profile.get("feature_profile", "")).startswith("lr_") else postmortem.get("ji_base_best_combined_variant")

    snapshot = {
        "snapshot_date": str(date.today()),
        "core_submission_profile": profile.get("submission_profile", "ji_base_base"),
        "frozen_overlay_submission_profile": FROZEN_OVERLAY_SUBMISSION_PROFILE,
        "best_overlay_submission_profile": overlay_lb_entry.get("submission_profile"),
        "best_overlay_submission_score": overlay_lb_entry.get("official_lb"),
        "submission_profile": profile.get("submission_profile", "ji_base_base"),
        "base_model_profile": profile.get("base_model_profile"),
        "calibration_mode": profile.get("calibration_mode"),
        "feature_profile": profile.get("feature_profile"),
        "alpha_profile": profile.get("alpha_profile"),
        "women_quality_profile_m": profile.get("women_quality_profile_m"),
        "women_quality_profile_w": profile.get("women_quality_profile_w"),
        "working_baseline_candidate": working_baseline_candidate,
        "official_lb_best_score": lb_entry.get("official_lb"),
        "official_lb_date": lb_entry.get("date"),
        "official_lb_notes": lb_entry.get("notes"),
        "current_best_submission_profile": overall_lb_entry.get("submission_profile", postmortem.get("official_lb_best_submission_profile")),
        "current_best_submission_score": overall_lb_entry.get("official_lb", postmortem.get("official_lb_best_score")),
        "replay_delta_vs_old_hc": postmortem.get("ji_base_vs_old_hc_delta"),
        "replay_delta_vs_gold_recover": postmortem.get("ji_base_vs_gold_recover_delta"),
        "submission_output": submission_summary.get("output"),
        "frozen_scope": {
            "keep_as_default": [
                str(profile.get("base_model_profile")),
                str(profile.get("feature_profile")),
                str(profile.get("alpha_profile")),
                f"{profile.get('women_quality_profile_m')} men quality",
                f"{profile.get('women_quality_profile_w')} women quality",
            ],
            "exclude_from_default_replay": [
                "JI_node_control",
                "JI_tabr_control",
                "overlay",
                "current-year market/injury/futures",
            ],
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ji_base_baseline_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    markdown = "\n".join(
        [
            "# JI_base Baseline Snapshot",
            "",
            f"- Snapshot date: `{snapshot['snapshot_date']}`",
            f"- Working baseline candidate: `{snapshot['working_baseline_candidate']}`",
            f"- Submission profile: `{snapshot['submission_profile']}`",
            f"- Base model: `{snapshot['base_model_profile']}`",
            f"- Calibration: `{snapshot['calibration_mode']}`",
            f"- Feature profile: `{snapshot['feature_profile']}`",
            f"- Alpha profile: `{snapshot['alpha_profile']}`",
            f"- Men quality profile: `{snapshot['women_quality_profile_m']}`",
            f"- Women quality profile: `{snapshot['women_quality_profile_w']}`",
            "",
            "## Official LB",
            "",
            f"- Frozen core LB: `{snapshot['official_lb_best_score']}`",
            f"- Logged on: `{snapshot['official_lb_date']}`",
            f"- Notes: {snapshot['official_lb_notes']}",
            f"- Frozen overlay submission: `{snapshot['frozen_overlay_submission_profile']}`",
            f"- Best-known overlay submission: `{snapshot['best_overlay_submission_profile']}` / `{snapshot['best_overlay_submission_score']}`",
            "",
            "## Replay Position",
            "",
            f"- `ji_base_vs_old_hc_delta`: `{snapshot['replay_delta_vs_old_hc']}`",
            f"- `ji_base_vs_gold_recover_delta`: `{snapshot['replay_delta_vs_gold_recover']}`",
            "",
            "## Freeze Rules",
            "",
            "- Keep as default:",
            *[f"  - `{item}`" for item in snapshot["frozen_scope"]["keep_as_default"]],
            "- Exclude from default replay:",
            *[f"  - `{item}`" for item in snapshot["frozen_scope"]["exclude_from_default_replay"]],
            "",
            f"- Submission output: `{snapshot['submission_output']}`",
            f"- JSON snapshot: `{RESULTS / 'ji_base_baseline_snapshot.json'}`",
        ]
    )
    (DOCS / "JI_BASE_BASELINE.md").write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
