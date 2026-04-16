from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.gold import GoldConfig
from hc.gold.overlay import apply_submission_overlay
from hc.gold.predict import parse_submission_ids, predict_submission

NCAA_DATA = ROOT / "ncaa-data"
RESULTS = ROOT / "results"
OFFICIAL_LB_COLUMNS = ["submission_profile", "base_model_profile", "overlay_stack", "date", "official_lb", "notes"]
OFFICIAL_LB_SEED_ROWS = [
    {
        "submission_profile": "gold_recover_base",
        "base_model_profile": "gold_lr_recover",
        "overlay_stack": "none",
        "date": "2026-04-11",
        "official_lb": 0.1306,
        "notes": "User-reported official LB for wide gold base submission.",
    },
    {
        "submission_profile": "gold_recover_market",
        "base_model_profile": "gold_lr_recover",
        "overlay_stack": "market_injury_sharp",
        "date": "2026-04-11",
        "official_lb": 0.1289,
        "notes": "User-reported official LB for wide gold overlay submission.",
    },
    {
        "submission_profile": "gold_min_market_injury_sharp",
        "base_model_profile": "gold_min_default",
        "overlay_stack": "market_injury_sharp",
        "date": "2026-04-11",
        "official_lb": 0.17,
        "notes": "User-reported official LB for failed gold_min submission.",
    },
]


def _recover_config(gender: str, *, rating_source_profile: str = "current_default") -> GoldConfig:
    return GoldConfig(
        gender=gender,
        model_family="gold_linear",
        calibration_mode="none",
        feature_profile="gold_recover_wide",
        rating_source_profile=rating_source_profile,
    )


def _harry_lr_config(gender: str) -> GoldConfig:
    return GoldConfig(gender=gender, model_family="gold_harry_lr", calibration_mode="none")


def _harry_xgb_config(gender: str) -> GoldConfig:
    return GoldConfig(gender=gender, model_family="gold_harry_xgb_spread", calibration_mode="isotonic_gender")


def _xgb_light_config(gender: str) -> GoldConfig:
    feature_profile = "gold_pruned_m" if gender == "M" else "gold_pruned_w"
    return GoldConfig(gender=gender, model_family="gold_xgb_spread_light", calibration_mode="none", feature_profile=feature_profile)


def resolve_submission_profiles(*, gender: str | None = None) -> list[dict]:
    profiles = [
        {
            "submission_profile": "gold_recover_base",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_recover_base_a_tier",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "a_tier_default",
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_a_tier",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "a_tier_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_recover_base_m_ap_removed",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "m_ap_removed_only",
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_m_ap_removed",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "m_ap_removed_only",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_recover_base_w_polls_removed",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "w_polls_removed_only",
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_w_polls_removed",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "w_polls_removed_only",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_recover_market_direct_only",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_only",
            "overlay_source_profile": "direct_only",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_direct_priority",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_only",
            "overlay_source_profile": "direct_priority",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_injury_only",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_recover_market_injury_sharp_only",
            "base_model_profile": "gold_lr_recover",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_harry_base",
            "base_model_profile": "gold_harry_xgb_spread",
            "rating_source_profile": "current_default",
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "a_tier_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_harry_market",
            "base_model_profile": "gold_harry_xgb_spread",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_only",
            "overlay_source_profile": "a_tier_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_harry_market_injury",
            "base_model_profile": "gold_harry_xgb_spread",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury",
            "overlay_source_profile": "a_tier_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_harry_market_injury_sharp",
            "base_model_profile": "gold_harry_xgb_spread",
            "rating_source_profile": "current_default",
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "a_tier_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
        {
            "submission_profile": "gold_blend_60_40",
            "base_model_profile": "gold_blend_60_40",
            "rating_source_profile": "current_default",
            "secondary_model_profile": "gold_xgb_spread_light",
            "blend_weights": {"lr": 0.40, "xgb": 0.60},
            "apply_overlay": False,
            "overlay_stack": "none",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": False,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_blend_market_injury",
            "base_model_profile": "gold_blend_60_40",
            "rating_source_profile": "current_default",
            "secondary_model_profile": "gold_xgb_spread_light",
            "blend_weights": {"lr": 0.40, "xgb": 0.60},
            "apply_overlay": True,
            "overlay_stack": "market_injury",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": False,
        },
        {
            "submission_profile": "gold_blend_market_injury_sharp",
            "base_model_profile": "gold_blend_60_40",
            "rating_source_profile": "current_default",
            "secondary_model_profile": "gold_xgb_spread_light",
            "blend_weights": {"lr": 0.40, "xgb": 0.60},
            "apply_overlay": True,
            "overlay_stack": "market_injury_sharp",
            "overlay_source_profile": "current_default",
            "include_futures": False,
            "allow_injury": True,
            "allow_sharpen": True,
        },
    ]
    if gender == "W":
        for profile in profiles:
            profile["allow_injury"] = False
            profile["allow_sharpen"] = False
    if gender == "M":
        for profile in profiles:
            if profile["overlay_stack"] == "none":
                continue
            profile["allow_injury"] = profile["overlay_stack"] in {"market_injury", "market_injury_sharp"}
            profile["allow_sharpen"] = profile["overlay_stack"] == "market_injury_sharp"
    return profiles


def seed_official_lb_log(path: Path) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = pd.DataFrame(columns=OFFICIAL_LB_COLUMNS)
    for column in OFFICIAL_LB_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")

    for row in OFFICIAL_LB_SEED_ROWS:
        existing = frame.loc[frame["submission_profile"] == row["submission_profile"]]
        if existing.empty:
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            continue
        for column, value in row.items():
            if column == "submission_profile":
                continue
            current = existing.iloc[0][column]
            if pd.isna(current) or current == "":
                frame.loc[frame["submission_profile"] == row["submission_profile"], column] = value

    frame = frame[OFFICIAL_LB_COLUMNS].sort_values(["date", "submission_profile"], na_position="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _split_ids(ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    men_ids = ids.loc[ids["ID"].astype(str).str.contains(r"^\d+_1")].copy()
    women_ids = ids.loc[~ids.index.isin(men_ids.index)].copy()
    if men_ids.empty or women_ids.empty:
        parsed = ids["ID"].astype(str).str.split("_", expand=True)
        team1 = pd.to_numeric(parsed[1], errors="coerce")
        men_ids = ids.loc[team1 < 3000].copy()
        women_ids = ids.loc[team1 >= 3000].copy()
    return men_ids, women_ids


def _predict_base_only(ids: pd.DataFrame, config: GoldConfig) -> pd.DataFrame:
    submission, _, _ = predict_submission(ids=ids, config=config, apply_overlay=False)
    return submission


def _blend_submissions(lr_submission: pd.DataFrame, xgb_submission: pd.DataFrame, *, lr_weight: float, xgb_weight: float) -> pd.DataFrame:
    merged = lr_submission.merge(xgb_submission, on="ID", suffixes=("_lr", "_xgb"))
    return pd.DataFrame(
        {
            "ID": merged["ID"],
            "Pred": (lr_weight * merged["Pred_lr"] + xgb_weight * merged["Pred_xgb"]).clip(0.001, 0.999),
        }
    )


def _base_submission_for_profile(
    *,
    ids: pd.DataFrame,
    gender: str,
    profile: dict,
    base_cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    key = (
        gender,
        profile["base_model_profile"],
        str(profile.get("rating_source_profile", "current_default")),
        str(profile.get("secondary_model_profile", "")),
        json.dumps(profile.get("blend_weights", {}), sort_keys=True),
    )
    if key in base_cache:
        return base_cache[key].copy()

    if profile["base_model_profile"] == "gold_lr_recover":
        base = _predict_base_only(ids, _recover_config(gender, rating_source_profile=str(profile.get("rating_source_profile", "current_default"))))
    elif profile["base_model_profile"] == "gold_harry_lr":
        base = _predict_base_only(ids, _harry_lr_config(gender))
    elif profile["base_model_profile"] == "gold_harry_xgb_spread":
        base = _predict_base_only(ids, _harry_xgb_config(gender))
    elif profile["base_model_profile"] == "gold_blend_60_40":
        lr_key = (gender, "gold_lr_recover", "current_default", "", "{}")
        xgb_key = (gender, "gold_xgb_spread_light", "current_default", "", "{}")
        if lr_key not in base_cache:
            base_cache[lr_key] = _predict_base_only(ids, _recover_config(gender))
        if xgb_key not in base_cache:
            base_cache[xgb_key] = _predict_base_only(ids, _xgb_light_config(gender))
        weights = profile.get("blend_weights", {"lr": 0.4, "xgb": 0.6})
        base = _blend_submissions(
            base_cache[lr_key],
            base_cache[xgb_key],
            lr_weight=float(weights["lr"]),
            xgb_weight=float(weights["xgb"]),
        )
    else:
        raise ValueError(f"Unsupported base_model_profile: {profile['base_model_profile']}")

    base_cache[key] = base.copy()
    return base


def _apply_profile_overlay(
    *,
    ids: pd.DataFrame,
    submission: pd.DataFrame,
    gender: str,
    profile: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    parsed = parse_submission_ids(ids)[["ID", "Season", "T1", "T2"]].copy()
    season = int(parsed["Season"].mode().iloc[0])
    if not profile["apply_overlay"]:
        summary = {
            "season": season,
            "rows": int(len(submission)),
            "changed_rows": 0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "overlay_submission_only_enabled": False,
            "overlay_source_profile": profile.get("overlay_source_profile", "current_default"),
            "injury_applied_rows": 0,
            "market_applied_rows": 0,
            "sharpen_applied_rows": 0,
        }
        return submission.copy(), None, summary

    overlay_input = parsed.merge(submission, on="ID", how="left")
    adjusted, audit, summary = apply_submission_overlay(
        overlay_input[["ID", "Season", "T1", "T2", "Pred"]],
        gender=gender,
        season=season,
        overlay_source_profile=str(profile.get("overlay_source_profile", "current_default")),
        include_futures=bool(profile.get("include_futures", False)),
        allow_injury=bool(profile.get("allow_injury", False)),
        allow_sharpen=bool(profile.get("allow_sharpen", False)),
    )
    return adjusted, audit, summary


def _build_candidate_submission(
    *,
    ids: pd.DataFrame,
    gender: str,
    profile: dict,
    base_cache: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    base_submission = _base_submission_for_profile(ids=ids, gender=gender, profile=profile, base_cache=base_cache)
    adjusted, audit, summary = _apply_profile_overlay(ids=ids, submission=base_submission, gender=gender, profile=profile)
    summary = {
        **summary,
        "submission_profile": profile["submission_profile"],
        "base_model_profile": profile["base_model_profile"],
        "rating_source_profile": profile.get("rating_source_profile", "current_default"),
        "overlay_stack": profile["overlay_stack"],
        "overlay_source_profile": profile.get("overlay_source_profile", "current_default"),
    }
    return adjusted, audit, summary


def _profile_lookup(profiles: list[dict]) -> dict[str, dict]:
    return {profile["submission_profile"]: profile for profile in profiles}


def _merge_gender_frames(men_frame: pd.DataFrame | None, women_frame: pd.DataFrame | None) -> pd.DataFrame:
    return pd.concat([frame for frame in (men_frame, women_frame) if frame is not None], ignore_index=True).sort_values("ID").reset_index(drop=True)


def _candidate_summary_rows(
    *,
    candidate_outputs: dict[str, str],
    men_profiles: dict[str, dict],
    women_profiles: dict[str, dict],
    lb_log: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for submission_profile, output_path in candidate_outputs.items():
        official_score = lb_log.loc[lb_log["submission_profile"] == submission_profile, "official_lb"]
        rows.append(
            {
                "submission_profile": submission_profile,
                "base_model_profile_m": men_profiles[submission_profile]["base_model_profile"],
                "base_model_profile_w": women_profiles[submission_profile]["base_model_profile"],
                "rating_source_profile_m": men_profiles[submission_profile].get("rating_source_profile", "current_default"),
                "rating_source_profile_w": women_profiles[submission_profile].get("rating_source_profile", "current_default"),
                "overlay_stack_m": men_profiles[submission_profile]["overlay_stack"],
                "overlay_stack_w": women_profiles[submission_profile]["overlay_stack"],
                "overlay_source_profile_m": men_profiles[submission_profile].get("overlay_source_profile", "current_default"),
                "overlay_source_profile_w": women_profiles[submission_profile].get("overlay_source_profile", "current_default"),
                "output_path": output_path,
                "official_lb": float(official_score.iloc[0]) if not official_score.empty else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Stage 2 submission with the gold model.")
    parser.add_argument("--input", type=Path, default=NCAA_DATA / "SampleSubmissionStage2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "submission_stage2_gold.csv")
    parser.add_argument("--base-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--disable-overlay", action="store_true")
    parser.add_argument("--no-candidate-outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    ids = frame[["ID", "Pred"]].copy() if "Pred" in frame.columns else frame.assign(Pred=0.5)[["ID", "Pred"]]
    men_ids, women_ids = _split_ids(ids)

    men_profiles = resolve_submission_profiles(gender="M")
    women_profiles = resolve_submission_profiles(gender="W")
    men_lookup = _profile_lookup(men_profiles)
    women_lookup = _profile_lookup(women_profiles)
    selected_profile_name = "gold_recover_market" if not args.disable_overlay else "gold_recover_base"

    base_cache: dict[tuple[str, str], pd.DataFrame] = {}
    men_submission, men_audit, men_summary = _build_candidate_submission(
        ids=men_ids,
        gender="M",
        profile=men_lookup[selected_profile_name],
        base_cache=base_cache,
    )
    women_submission, women_audit, women_summary = _build_candidate_submission(
        ids=women_ids,
        gender="W",
        profile=women_lookup[selected_profile_name],
        base_cache=base_cache,
    )

    final = _merge_gender_frames(men_submission, women_submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)

    if args.base_output is not None:
        default_base_m, _, _ = _build_candidate_submission(ids=men_ids, gender="M", profile=men_lookup["gold_recover_base"], base_cache=base_cache)
        default_base_w, _, _ = _build_candidate_submission(ids=women_ids, gender="W", profile=women_lookup["gold_recover_base"], base_cache=base_cache)
        _merge_gender_frames(default_base_m, default_base_w).to_csv(args.base_output, index=False)

    candidate_outputs: dict[str, str] = {}
    audit_frames: list[pd.DataFrame] = [frame for frame in (men_audit, women_audit) if frame is not None]
    if not args.no_candidate_outputs:
        for submission_profile in men_lookup:
            men_candidate, _, _ = _build_candidate_submission(
                ids=men_ids,
                gender="M",
                profile=men_lookup[submission_profile],
                base_cache=base_cache,
            )
            women_candidate, _, _ = _build_candidate_submission(
                ids=women_ids,
                gender="W",
                profile=women_lookup[submission_profile],
                base_cache=base_cache,
            )
            candidate_path = args.output.with_name(f"{args.output.stem}_{submission_profile}.csv")
            _merge_gender_frames(men_candidate, women_candidate).to_csv(candidate_path, index=False)
            candidate_outputs[submission_profile] = str(candidate_path)

    lb_log_path = RESULTS / "official_lb_log.csv"
    lb_log = seed_official_lb_log(lb_log_path)
    candidate_summary_path = RESULTS / "gold_submission_candidates_summary.csv"
    pd.DataFrame(
        _candidate_summary_rows(
            candidate_outputs=candidate_outputs,
            men_profiles=men_lookup,
            women_profiles=women_lookup,
            lb_log=lb_log,
        )
    ).to_csv(candidate_summary_path, index=False)

    audit_path = args.audit_output or args.output.with_name(f"{args.output.stem}_audit.csv")
    summary_path = args.summary_output or args.output.with_name(f"{args.output.stem}_summary.json")
    if audit_frames:
        pd.concat(audit_frames, ignore_index=True).sort_values("ID").to_csv(audit_path, index=False)
    summary = {
        "season": 2026,
        "rows": int(len(final)),
        "overlay_enabled": not args.disable_overlay,
        "submission_profile": {"M": selected_profile_name, "W": selected_profile_name},
        "base_model_profile": {
            "M": men_lookup[selected_profile_name]["base_model_profile"],
            "W": women_lookup[selected_profile_name]["base_model_profile"],
        },
        "overlay_stack": {
            "M": men_lookup[selected_profile_name]["overlay_stack"],
            "W": women_lookup[selected_profile_name]["overlay_stack"],
        },
        "candidate_outputs": candidate_outputs,
        "candidate_summary_path": str(candidate_summary_path),
        "official_lb_log_path": str(lb_log_path),
        "men": men_summary,
        "women": women_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
