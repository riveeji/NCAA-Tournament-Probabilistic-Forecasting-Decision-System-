from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache" / "hc"


def _brier(frame: pd.DataFrame, prob_col: str) -> float:
    return float(((frame[prob_col] - frame["Label"]) ** 2).mean())


def _season_brier(frame: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    grouped = frame.groupby("Season").apply(lambda grp: ((grp[prob_col] - grp["Label"]) ** 2).mean(), include_groups=False)
    return grouped.rename("brier").reset_index()


def _load_old_current_metrics() -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    seasonal_frames: list[pd.DataFrame] = []
    mapping = {
        "M": CACHE / "final_oof_M_23y_aggressive_pre_tip_all_round_text_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
        "W": CACHE / "final_oof_W_23y_aggressive_selection_week_plus_pre_tip_text_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
    }
    for gender, path in mapping.items():
        frame = pd.read_parquet(path)
        rows.append(
            {
                "system": "hc_current",
                "gender": gender,
                "variant": "final_prob",
                "mean_brier": _brier(frame, "FinalProb"),
            }
        )
        season = _season_brier(frame, "FinalProb")
        season["system"] = "hc_current"
        season["gender"] = gender
        season["variant"] = "final_prob"
        seasonal_frames.append(season)
    return rows, seasonal_frames


def _load_old_ablation_metrics() -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    seasonal_frames: list[pd.DataFrame] = []
    mapping = {
        "M": CACHE / "base_oof_M_23y_aggressive_pre_tip_all_round_text_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
        "W": CACHE / "base_oof_W_23y_aggressive_selection_week_plus_pre_tip_text_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
    }
    candidate_cols = {
        "legacy_anchor": "Prob_legacy_anchor",
        "stats_fallback": "Prob_stats_fallback_et",
        "market_plus_stats": "Prob_market_plus_stats_lr",
        "market_only": "Prob_market_only_lr",
    }
    for gender, path in mapping.items():
        frame = pd.read_parquet(path)
        for variant, col in candidate_cols.items():
            if col not in frame.columns:
                continue
            rows.append(
                {
                    "system": "old_ablation",
                    "gender": gender,
                    "variant": variant,
                    "mean_brier": _brier(frame, col),
                }
            )
            season = _season_brier(frame, col)
            season["system"] = "old_ablation"
            season["gender"] = gender
            season["variant"] = variant
            seasonal_frames.append(season)
    return rows, seasonal_frames


def _load_runtime_variance_proxy() -> dict:
    configs = {
        "M_aggressive_5y": CACHE / "final_oof_M_5y_aggressive_pre_tip_all_round_notext_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
        "M_clean_5y": CACHE / "final_oof_M_5y_clean_selection_week_only_notext_notabpfn_32d_full_legacyv8.parquet",
        "W_aggressive_5y": CACHE / "final_oof_W_5y_aggressive_selection_week_plus_pre_tip_notext_notabpfn_32d_full_legacyv8_cachev2_fusionv3_publicv10_runtime_marketv1_ovr_off.parquet",
        "W_clean_5y": CACHE / "final_oof_W_5y_clean_selection_week_only_notext_notabpfn_32d_full_legacyv8.parquet",
    }
    scores: dict[str, float] = {}
    for key, path in configs.items():
        if path.exists():
            frame = pd.read_parquet(path)
            scores[key] = _brier(frame, "FinalProb")
    return scores


def _load_latest_goldshot_metadata() -> dict:
    summaries = sorted(RESULTS.glob("submission_stage2_single_final_hc_goldshot_summary_*.json"))
    recommendations = sorted(RESULTS.glob("final_submission_recommendation_*.json"))
    payload: dict = {}
    if summaries:
        payload["latest_goldshot_summary"] = json.loads(summaries[-1].read_text(encoding="utf-8"))
    if recommendations:
        payload["latest_recommendation"] = json.loads(recommendations[-1].read_text(encoding="utf-8"))
    return payload


def _load_next_year_overlay_metadata() -> dict:
    gold_path = RESULTS / "submission_stage2_gold_summary.json"
    if gold_path.exists():
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
        men = payload.get("men") or {}
        women = payload.get("women") or {}
        mean_abs_delta_m = men.get("mean_abs_delta")
        mean_abs_delta_w = women.get("mean_abs_delta")
        guardrail_passed = bool(
            (mean_abs_delta_m is None or float(mean_abs_delta_m) <= 0.0025)
            and (mean_abs_delta_w is None or float(mean_abs_delta_w) <= 0.0005)
        )
        return {
            "season": payload.get("season"),
            "enabled": bool(payload.get("overlay_enabled", False)),
            "submission_profile": payload.get("submission_profile", {}),
            "candidate_outputs": payload.get("candidate_outputs", {}),
            "genders": {"M": men, "W": women},
            "overlay_guardrail_passed": guardrail_passed,
            "overlay_mean_abs_delta_m": mean_abs_delta_m,
            "overlay_mean_abs_delta_w": mean_abs_delta_w,
            "audit_path": str((RESULTS / "submission_stage2_gold_audit.csv")),
        }
    path = RESULTS / "v2_next_year_overlay_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_official_lb_log() -> pd.DataFrame:
    path = RESULTS / "official_lb_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=["submission_profile", "official_lb"])
    frame = pd.read_csv(path)
    if "official_lb" in frame.columns:
        frame["official_lb"] = pd.to_numeric(frame["official_lb"], errors="coerce")
    return frame


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_external_source_inventory() -> pd.DataFrame:
    path = RESULTS / "external_source_inventory.csv"
    if not path.exists():
        return pd.DataFrame(columns=["source_name", "tier"])
    return pd.read_csv(path)


def _variant_metric(combined: pd.DataFrame, variant: str, column: str = "equal_gender_mean_brier") -> float:
    match = combined.loc[combined["variant"] == variant, column]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _route_variant_label(
    route: str | None,
    model_variant: str,
    market_mode: str,
    feature_pack: str | None = None,
    calibration_mode: str | None = None,
    learner_family: str | None = None,
) -> str:
    normalized_route = route or "probability"
    normalized_learner = learner_family
    if not normalized_learner:
        if model_variant == "lr":
            normalized_learner = "linear"
        elif model_variant == "tree":
            normalized_learner = "tree"
        else:
            normalized_learner = model_variant
    if not feature_pack or not calibration_mode:
        if model_variant == "lr":
            return f"{normalized_route}:{market_mode}"
        return f"{normalized_route}-{model_variant}:{market_mode}"
    return f"{normalized_route}-{normalized_learner}:{market_mode}@{feature_pack}+{calibration_mode}"


def _load_v2_metrics() -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    summary = pd.read_csv(RESULTS / "v2_replay_summary.csv")
    combined = pd.read_csv(RESULTS / "v2_replay_combined.csv")
    by_season_path = RESULTS / "v2_replay_by_season.csv"

    rows = [
        {
            "system": "v2",
            "gender": row["gender"],
            "variant": _route_variant_label(
                row.get("route"),
                row["model_variant"],
                row["market_mode"],
                row.get("feature_pack"),
                row.get("calibration_mode"),
                row.get("learner_family"),
            ),
            "mean_brier": float(row["mean_brier"]),
        }
        for _, row in summary.iterrows()
    ]

    seasonal_frames: list[pd.DataFrame] = []
    if by_season_path.exists():
        by_season = pd.read_csv(by_season_path)
        season_col = "Season" if "Season" in by_season.columns else "season"
        season = by_season.copy()
        season["system"] = "v2"
        season["variant"] = season.apply(
            lambda row: _route_variant_label(
                row["route"] if "route" in season.columns else None,
                row["model_variant"],
                row["market_mode"],
                row["feature_pack"] if "feature_pack" in season.columns else None,
                row["calibration_mode"] if "calibration_mode" in season.columns else None,
                row["learner_family"] if "learner_family" in season.columns else None,
            ),
            axis=1,
        )
        seasonal_frames.append(season[[season_col, "brier", "system", "gender", "variant"]].rename(columns={season_col: "season"}))

    return rows, seasonal_frames, combined


def _load_gold_metrics() -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    summary_path = RESULTS / "gold_replay_summary.csv"
    combined_path = RESULTS / "gold_replay_combined.csv"
    if not summary_path.exists() or not combined_path.exists():
        return [], [], pd.DataFrame()

    summary = pd.read_csv(summary_path)
    combined = pd.read_csv(combined_path)
    by_season_path = RESULTS / "gold_replay_by_season.csv"

    rows = [
        {
            "system": "gold",
            "gender": row["gender"],
            "variant": f"{row['model_family']}@{row['calibration_mode']}",
            "mean_brier": float(row["mean_brier"]),
        }
        for _, row in summary.iterrows()
    ]

    seasonal_frames: list[pd.DataFrame] = []
    if by_season_path.exists():
        by_season = pd.read_csv(by_season_path)
        season_col = "Season" if "Season" in by_season.columns else "season"
        season = by_season.copy()
        season["system"] = "gold"
        season["variant"] = season.apply(lambda row: f"{row['model_family']}@{row['calibration_mode']}", axis=1)
        seasonal_frames.append(season[[season_col, "brier", "system", "gender", "variant"]].rename(columns={season_col: "season"}))

    return rows, seasonal_frames, combined


def _load_ji_base_metrics() -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    summary_path = RESULTS / "ji_base_replay_summary.csv"
    combined_path = RESULTS / "ji_base_replay_combined.csv"
    if not summary_path.exists() or not combined_path.exists():
        return [], [], pd.DataFrame()

    summary = pd.read_csv(summary_path)
    combined = pd.read_csv(combined_path)
    if "variant" not in combined.columns and {"model_family", "calibration_mode"}.issubset(combined.columns):
        combined["variant"] = combined.apply(lambda row: f"{row['model_family']}@{row['calibration_mode']}", axis=1)

    challenger_rows: list[dict] = []
    for path in sorted(RESULTS.glob("ji_base_challenger_*.json")):
        if path.name == "ji_base_challenger_registry.json":
            continue
        payload = _load_json(path)
        challenger = payload.get("challenger_summary")
        if not isinstance(challenger, dict) or not challenger:
            continue
        challenger_rows.append(
            {
                "candidate_name": payload.get("candidate_name"),
                "variant": f"{challenger.get('model_family_m')}@{challenger.get('calibration_mode_m')}",
                "model_family": challenger.get("model_family_m"),
                "calibration_mode": challenger.get("calibration_mode_m"),
                "feature_profile": challenger.get("feature_profile_w") or challenger.get("feature_profile_m"),
                "women_quality_profile": challenger.get("women_quality_profile_w"),
                "alpha_profile": challenger.get("alpha_profile_w") or challenger.get("alpha_profile_m"),
                "total_cv_brier_raw": challenger.get("total_cv_brier_raw"),
                "total_cv_brier_calibrated": challenger.get("total_cv_brier_calibrated"),
                "latest_season_equal_gender_brier": challenger.get("latest_season_equal_gender_brier"),
                "recent_window_equal_gender_brier": challenger.get("recent_window_equal_gender_brier"),
                "men_cv_brier_raw": challenger.get("men_cv_brier_raw"),
                "men_cv_brier_calibrated": challenger.get("men_cv_brier_calibrated"),
                "women_cv_brier_raw": challenger.get("women_cv_brier_raw"),
                "women_cv_brier_calibrated": challenger.get("women_cv_brier_calibrated"),
            }
        )
    if challenger_rows:
        combined = pd.concat([combined, pd.DataFrame(challenger_rows)], ignore_index=True, sort=False)

    rows = [
        {
            "system": "ji_base",
            "gender": row["gender"],
            "variant": f"{row['model_family']}@{row['calibration_mode']}",
            "mean_brier": float(row["cv_brier_calibrated"]),
        }
        for _, row in summary.iterrows()
    ]
    return rows, [], combined


def _best_summary_variant(
    summary: pd.DataFrame,
    *,
    gender: str,
    feature_pack: str | None = None,
) -> str | None:
    frame = summary.loc[summary["gender"] == gender].copy()
    if feature_pack is not None and "feature_pack" in frame.columns:
        frame = frame.loc[frame["feature_pack"] == feature_pack]
    if frame.empty:
        return None
    sort_columns = [column for column in ("latest_season_brier", "recent_window_brier", "mean_brier") if column in frame.columns]
    best = frame.sort_values(sort_columns).iloc[0]
    if "variant" in frame.columns:
        return str(best["variant"])
    if {"model_family", "calibration_mode"}.issubset(frame.columns):
        return f"{best['model_family']}@{best['calibration_mode']}"
    return None


def _load_current_primary_metric() -> dict:
    bench = pd.read_csv(RESULTS / "hc_benchmarks.csv")
    latest = bench.loc[bench["years"] == 23].copy().sort_values(["gender"])
    return {row["gender"]: float(row["lb_window_4y_score"]) for _, row in latest.iterrows()}


def _fmt_metric(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _best_variant(
    combined: pd.DataFrame,
    *,
    learner_family: str | None = None,
    route: str | None = None,
    feature_pack: str | None = None,
) -> str | None:
    frame = combined
    if route is not None and "route" in frame.columns:
        frame = frame.loc[frame["route"] == route]
    if learner_family is not None and "learner_family" in frame.columns:
        frame = frame.loc[frame["learner_family"] == learner_family]
    if feature_pack is not None and "feature_pack" in frame.columns:
        frame = frame.loc[frame["feature_pack"] == feature_pack]
    if frame.empty:
        return None
    sort_column = "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in frame.columns else "equal_gender_mean_brier"
    return str(frame.sort_values(sort_column).iloc[0]["variant"])


def _best_ji_base_candidate_label(combined: pd.DataFrame) -> str | None:
    if combined.empty:
        return None
    best = combined.sort_values("total_cv_brier_calibrated").iloc[0]
    if "candidate_name" in best.index and pd.notna(best["candidate_name"]):
        return str(best["candidate_name"])
    if "variant" in best.index and pd.notna(best["variant"]):
        return str(best["variant"])
    if {"model_family", "calibration_mode"}.issubset(best.index):
        return f"{best['model_family']}@{best['calibration_mode']}"
    return "ji_base_legacy_single_run"


def _build_findings(
    ablation: pd.DataFrame,
    combined: pd.DataFrame,
    primary_metric: dict,
    runtime_proxy: dict,
    latest_goldshot: dict,
) -> str:
    best_linear = _best_variant(combined, learner_family="linear", route="spread")
    best_tree = _best_variant(combined, learner_family="tree", route="spread")
    best_strength_full = _best_variant(combined, learner_family="linear", route="spread", feature_pack="strength_full")
    best_strength_recent = _best_variant(combined, learner_family="linear", route="spread", feature_pack="strength_recent")
    best_base = _best_variant(combined, learner_family="linear", route="spread", feature_pack="base")
    best_external_base = _best_variant(combined, learner_family="linear", route="spread", feature_pack="external_base")
    best_linear_score = _variant_metric(combined, best_linear) if best_linear else float("nan")
    best_tree_score = _variant_metric(combined, best_tree) if best_tree else float("nan")
    best_strength_full_score = _variant_metric(combined, best_strength_full) if best_strength_full else float("nan")
    best_strength_recent_score = _variant_metric(combined, best_strength_recent) if best_strength_recent else float("nan")
    best_base_score = _variant_metric(combined, best_base) if best_base else float("nan")
    best_external_base_score = _variant_metric(combined, best_external_base) if best_external_base else float("nan")
    legacy_anchor_mean = float(ablation.loc[ablation["variant"] == "legacy_anchor", "mean_brier"].mean())

    lines = [
        "# Sprint 0-1 Postmortem Findings",
        "",
        "## Why the legacy system stalled around the top 30%",
        "- Legacy optimization still leaned too heavily on proxy-heavy historical leaderboard windows instead of tournament-only replay.",
        "- The old mainline stacked `legacy_anchor`, multiple market policy routes, and runtime overrides into one default path, making real signal gain hard to separate from noise.",
        "- The clean `v2` baseline is already competitive, which suggests the main miss was excess decision-layer complexity rather than a lack of raw features.",
        "",
        "## What should be removed or downgraded first",
        f"- `legacy_anchor` is the first downgrade candidate; its historical ablation mean Brier is {_fmt_metric(legacy_anchor_mean)}.",
        "- Multiple market policy routes should collapse into one explicit lightweight market-blend experiment rather than remain default mainline logic.",
        "- `goldshot` should move from default decision layer to opt-in seasonal experiment until replay evidence justifies bringing it back.",
    ]

    if runtime_proxy:
        lines.extend(
            [
                (
                    "- Runtime variance proxy: "
                    f"M aggressive={_fmt_metric(runtime_proxy.get('M_aggressive_5y', float('nan')))}; "
                    f"M clean={_fmt_metric(runtime_proxy.get('M_clean_5y', float('nan')))}; "
                    f"W aggressive={_fmt_metric(runtime_proxy.get('W_aggressive_5y', float('nan')))}; "
                    f"W clean={_fmt_metric(runtime_proxy.get('W_clean_5y', float('nan')))}."
                ),
                "- Keep runtime/extremes as an isolated experiment, not as a default layer.",
            ]
        )

    latest_summary = latest_goldshot.get("latest_goldshot_summary")
    if latest_summary:
        lines.append(
            "- Latest goldshot run changed "
            f"{latest_summary.get('total_changed_rows', 'n/a')} rows, which is seasonal evidence only and not a replay-based reason to keep it in the mainline."
        )

    lines.extend(
        [
            "",
            "## Minimum system to keep for the next stage",
            f"- Best spread-linear candidate is `{best_linear or 'n/a'}` ({_fmt_metric(best_linear_score)} equal-gender replay).",
            f"- Best spread-tree candidate is `{best_tree or 'n/a'}` ({_fmt_metric(best_tree_score)} equal-gender replay).",
            f"- Best external-base candidate is `{best_external_base or 'n/a'}` ({_fmt_metric(best_external_base_score)} equal-gender replay).",
            f"- `strength_full` best linear candidate is `{best_strength_full or 'n/a'}` ({_fmt_metric(best_strength_full_score)}); `strength_recent` best is `{best_strength_recent or 'n/a'}` ({_fmt_metric(best_strength_recent_score)}).",
            f"- Current `base` control is `{best_base or 'n/a'}` ({_fmt_metric(best_base_score)}), which is the direct benchmark for the strength rebuild.",
            "- Keep sportsbook as an outer comparison only, not a reason to reintroduce complex decision-layer routing.",
            (
                "- Current HC primary leaderboard proxy remains "
                f"M={_fmt_metric(primary_metric.get('M', float('nan')))} / "
                f"W={_fmt_metric(primary_metric.get('W', float('nan')))}, "
                "but it should be secondary to replay in the next iteration."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    current_rows, current_seasonal = _load_old_current_metrics()
    ablation_rows, ablation_seasonal = _load_old_ablation_metrics()
    v2_rows, v2_seasonal, combined = _load_v2_metrics()
    gold_rows, gold_seasonal, gold_combined = _load_gold_metrics()
    ji_rows, ji_seasonal, ji_combined = _load_ji_base_metrics()
    v2_summary = pd.read_csv(RESULTS / "v2_replay_summary.csv")
    v2_summary["variant"] = v2_summary.apply(
        lambda row: _route_variant_label(
            row.get("route"),
            row["model_variant"],
            row["market_mode"],
            row.get("feature_pack"),
            row.get("calibration_mode"),
            row.get("learner_family"),
        ),
        axis=1,
    )
    primary_metric = _load_current_primary_metric()
    runtime_proxy = _load_runtime_variance_proxy()
    latest_goldshot = _load_latest_goldshot_metadata()
    next_year_overlay = _load_next_year_overlay_metadata()
    official_lb_log = _load_official_lb_log()
    external_source_inventory = _load_external_source_inventory()

    ablation_table = pd.DataFrame(current_rows + ablation_rows + v2_rows + gold_rows + ji_rows).sort_values(["system", "gender", "mean_brier"])
    ablation_table.to_csv(RESULTS / "postmortem_ablation_table.csv", index=False)

    seasonal_frames = [*current_seasonal, *ablation_seasonal, *v2_seasonal, *gold_seasonal, *ji_seasonal]
    if seasonal_frames:
        seasonal = pd.concat(seasonal_frames, ignore_index=True)
        seasonal.to_csv(RESULTS / "postmortem_by_season.csv", index=False)

    best_sort_column = "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in combined.columns else "equal_gender_mean_brier"
    best_overall = combined.sort_values(best_sort_column).iloc[0]
    best_spread_linear_variant = _best_variant(combined, learner_family="linear", route="spread")
    best_spread_tree_variant = _best_variant(combined, learner_family="tree", route="spread")
    best_external_base_variant = _best_variant(combined, learner_family="linear", route="spread", feature_pack="external_base")
    best_external_base_pruned_variant = _best_variant(
        combined,
        learner_family="linear",
        route="spread",
        feature_pack="external_base_pruned",
    )
    if best_external_base_variant is None:
        best_external_base_variant = best_external_base_pruned_variant
    best_external_base_m_variant = _best_summary_variant(v2_summary, gender="M", feature_pack="external_base_pruned")
    best_external_base_w_variant = _best_summary_variant(v2_summary, gender="W", feature_pack="external_base_pruned")
    best_strength_full_variant = _best_variant(combined, learner_family="linear", route="spread", feature_pack="strength_full")
    best_strength_recent_variant = _best_variant(combined, learner_family="linear", route="spread", feature_pack="strength_recent")
    best_base_variant = _best_variant(combined, learner_family="linear", route="spread", feature_pack="base")
    best_feature_pack = None
    best_calibration_mode = None
    spread_only = combined.loc[combined["route"] == "spread"].copy() if "route" in combined.columns else combined.copy()
    if not spread_only.empty:
        metric_col = "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in spread_only.columns else "equal_gender_mean_brier"
        best_feature_pack = str(
            spread_only.groupby("feature_pack", as_index=False)[metric_col].mean().sort_values(metric_col).iloc[0]["feature_pack"]
        )
        best_calibration_mode = str(
            spread_only.groupby("calibration_mode", as_index=False)[metric_col].mean().sort_values(metric_col).iloc[0]["calibration_mode"]
        )
    men_best = float(best_overall["men_mean_brier"])
    women_best = float(best_overall["women_mean_brier"])
    recent_best = float(best_overall["equal_gender_recent_window_brier"])
    old_current = float(pd.DataFrame(current_rows)["mean_brier"].mean())
    best_base_score = _variant_metric(combined, best_base_variant) if best_base_variant else float("nan")
    best_external_base_score = _variant_metric(combined, best_external_base_variant) if best_external_base_variant else float("nan")
    best_external_base_pruned_score = (
        _variant_metric(combined, best_external_base_pruned_variant) if best_external_base_pruned_variant else float("nan")
    )
    best_strength_full_score = _variant_metric(combined, best_strength_full_variant) if best_strength_full_variant else float("nan")
    best_strength_recent_score = _variant_metric(combined, best_strength_recent_variant) if best_strength_recent_variant else float("nan")
    internal_candidates = [score for score in (best_base_score, best_strength_full_score, best_strength_recent_score) if not math.isnan(score)]
    best_internal_strength_score = min(internal_candidates) if internal_candidates else float("nan")
    best_tree_score = _variant_metric(combined, best_spread_tree_variant) if best_spread_tree_variant else float("nan")
    best_base_recent = None
    if best_base_variant:
        base_recent_series = combined.loc[combined["variant"] == best_base_variant, "equal_gender_recent_window_brier"]
        if not base_recent_series.empty:
            best_base_recent = float(base_recent_series.iloc[0])
    upgrade_gate = bool(
        float(best_overall["equal_gender_mean_brier"]) <= old_current
        and recent_best <= old_current
        and men_best <= float(pd.DataFrame(current_rows).loc[pd.DataFrame(current_rows)["gender"] == "M", "mean_brier"].iloc[0])
        and women_best <= float(pd.DataFrame(current_rows).loc[pd.DataFrame(current_rows)["gender"] == "W", "mean_brier"].iloc[0])
    )
    strength_rebuild_gate = bool(
        (
            (not math.isnan(best_strength_full_score) and not math.isnan(best_base_score) and best_strength_full_score <= best_base_score)
            or (not math.isnan(best_strength_recent_score) and not math.isnan(best_base_score) and best_strength_recent_score <= best_base_score)
        )
        and (
            best_base_recent is None
            or (
                (best_strength_full_variant is not None and float(combined.loc[combined["variant"] == best_strength_full_variant, "equal_gender_recent_window_brier"].iloc[0]) <= best_base_recent)
                or (best_strength_recent_variant is not None and float(combined.loc[combined["variant"] == best_strength_recent_variant, "equal_gender_recent_window_brier"].iloc[0]) <= best_base_recent)
            )
        )
    )
    gold_best_combined_variant = None if gold_combined.empty else str(
        gold_combined.sort_values(
            "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in gold_combined.columns else "equal_gender_mean_brier"
        ).iloc[0]["variant"]
    )
    gold_harry_best_combined_variant = None
    gold_best_m_variant = None
    gold_best_w_variant = None
    gold_upgrade_gate_passed = False
    gold_vs_old_hc_delta = None
    gold_vs_v2_delta = None
    if not gold_combined.empty:
        gold_harry = gold_combined.loc[gold_combined["model_family"].astype(str).str.startswith("gold_harry")].copy()
        if not gold_harry.empty:
            gold_harry_best_combined_variant = str(
                gold_harry.sort_values(
                    "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in gold_harry.columns else "equal_gender_mean_brier"
                ).iloc[0]["variant"]
            )
        gold_best = gold_combined.sort_values(
            "equal_gender_latest_season_brier" if "equal_gender_latest_season_brier" in gold_combined.columns else "equal_gender_mean_brier"
        ).iloc[0]
        gold_best_m_variant = _best_summary_variant(pd.read_csv(RESULTS / "gold_replay_summary.csv"), gender="M")
        gold_best_w_variant = _best_summary_variant(pd.read_csv(RESULTS / "gold_replay_summary.csv"), gender="W")
        gold_best_score = float(gold_best["equal_gender_mean_brier"])
        gold_vs_old_hc_delta = float(gold_best_score - old_current)
        gold_vs_v2_delta = float(gold_best_score - float(combined["equal_gender_mean_brier"].min()))
        gold_upgrade_gate_passed = bool(
            gold_best_score <= old_current
            and float(gold_best["equal_gender_latest_season_brier"]) <= float(best_overall["equal_gender_latest_season_brier"])
            and float(gold_best["equal_gender_recent_window_brier"]) <= float(best_overall["equal_gender_recent_window_brier"])
            and float(gold_best["men_mean_brier"]) <= float(pd.DataFrame(current_rows).loc[pd.DataFrame(current_rows)["gender"] == "M", "mean_brier"].iloc[0])
            and float(gold_best["women_mean_brier"]) <= float(pd.DataFrame(current_rows).loc[pd.DataFrame(current_rows)["gender"] == "W", "mean_brier"].iloc[0])
        )

    ji_base_best_combined_variant = None
    ji_base_vs_old_hc_delta = None
    ji_base_vs_gold_recover_delta = None
    ji_base_upgrade_gate_passed = False
    if not ji_combined.empty:
        ji_best = ji_combined.sort_values("total_cv_brier_calibrated").iloc[0]
        ji_base_best_combined_variant = _best_ji_base_candidate_label(ji_combined)
        ji_best_score = float(ji_best["total_cv_brier_calibrated"])
        ji_base_vs_old_hc_delta = float(ji_best_score - old_current)
        if not gold_combined.empty:
            ji_base_vs_gold_recover_delta = float(ji_best_score - float(gold_combined["equal_gender_mean_brier"].min()))
        ji_base_upgrade_gate_passed = bool(
            ji_best_score <= old_current
            and float(ji_best["latest_season_equal_gender_brier"]) <= float(best_overall["equal_gender_latest_season_brier"])
            and float(ji_best["recent_window_equal_gender_brier"]) <= float(best_overall["equal_gender_recent_window_brier"])
        )

    official_lb_best_submission_profile = None
    official_lb_best_score = None
    if not official_lb_log.empty and "official_lb" in official_lb_log.columns:
        valid_lb = official_lb_log.loc[official_lb_log["official_lb"].notna()].sort_values("official_lb")
        if not valid_lb.empty:
            official_lb_best_submission_profile = str(valid_lb.iloc[0]["submission_profile"])
            official_lb_best_score = float(valid_lb.iloc[0]["official_lb"])
    baseline_snapshot = _load_json(RESULTS / "ji_base_baseline_snapshot.json")

    summary = {
        "current_primary_metric": primary_metric,
        "systems_compared": sorted(ablation_table["system"].unique().tolist()),
        "old_current_equal_gender_mean_brier": old_current,
        "v2_best_equal_gender_mean_brier": float(combined["equal_gender_mean_brier"].min()),
        "v2_best_current_year_season": int(best_overall["latest_season"]) if "latest_season" in best_overall.index else None,
        "v2_best_current_year_equal_gender_brier": float(best_overall["equal_gender_latest_season_brier"]) if "equal_gender_latest_season_brier" in best_overall.index else None,
        "v2_best_variant": str(best_overall["variant"]),
        "best_spread_linear_variant": best_spread_linear_variant,
        "best_spread_tree_variant": best_spread_tree_variant,
        "best_external_base_variant": best_external_base_variant,
        "best_external_base_pruned_variant": best_external_base_pruned_variant,
        "best_external_base_m_variant": best_external_base_m_variant,
        "best_external_base_w_variant": best_external_base_w_variant,
        "best_strength_full_variant": best_strength_full_variant,
        "best_strength_recent_variant": best_strength_recent_variant,
        "best_calibration_mode": best_calibration_mode,
        "best_feature_pack": best_feature_pack,
        "spread_upgrade_gate_passed": upgrade_gate,
        "strength_rebuild_gate_passed": strength_rebuild_gate,
        "gold_best_combined_variant": gold_best_combined_variant,
        "gold_harry_best_combined_variant": gold_harry_best_combined_variant,
        "gold_best_m_variant": gold_best_m_variant,
        "gold_best_w_variant": gold_best_w_variant,
        "gold_vs_old_hc_delta": gold_vs_old_hc_delta,
        "gold_vs_v2_delta": gold_vs_v2_delta,
        "gold_upgrade_gate_passed": gold_upgrade_gate_passed,
        "ji_base_best_combined_variant": ji_base_best_combined_variant,
        "ji_base_vs_old_hc_delta": ji_base_vs_old_hc_delta,
        "ji_base_vs_gold_recover_delta": ji_base_vs_gold_recover_delta,
        "ji_base_upgrade_gate_passed": ji_base_upgrade_gate_passed,
        "official_lb_best_submission_profile": official_lb_best_submission_profile,
        "official_lb_best_score": official_lb_best_score,
        "frozen_overlay_submission_profile": baseline_snapshot.get("frozen_overlay_submission_profile"),
        "best_overlay_submission_profile": baseline_snapshot.get("best_overlay_submission_profile", official_lb_best_submission_profile),
        "best_overlay_submission_score": baseline_snapshot.get("best_overlay_submission_score", official_lb_best_score),
        "official_lb_log_rows": int(len(official_lb_log)),
        "overlay_submission_only_enabled": bool(next_year_overlay.get("enabled", False)),
        "linear_vs_tree_delta": (
            float(_variant_metric(combined, best_spread_linear_variant) - best_tree_score)
            if best_spread_linear_variant and not math.isnan(best_tree_score)
            else None
        ),
        "external_vs_internal_strength_delta": (
            float(best_external_base_score - best_internal_strength_score)
            if not math.isnan(best_external_base_score) and not math.isnan(best_internal_strength_score)
            else None
        ),
        "next_year_overlay_enabled": bool(next_year_overlay.get("enabled", False)),
        "overlay_guardrail_passed": bool(next_year_overlay.get("overlay_guardrail_passed", False)),
        "overlay_mean_abs_delta_m": next_year_overlay.get("overlay_mean_abs_delta_m"),
        "overlay_mean_abs_delta_w": next_year_overlay.get("overlay_mean_abs_delta_w"),
        "next_year_overlay_metadata": next_year_overlay,
        "official_lb_log_path": str(RESULTS / "official_lb_log.csv"),
        "external_source_inventory_path": str(RESULTS / "external_source_inventory.csv"),
        "external_source_tier_counts": (
            external_source_inventory.groupby("tier").size().to_dict() if not external_source_inventory.empty else {}
        ),
        "legacy_anchor_mean_brier": float(
            ablation_table.loc[ablation_table["variant"] == "legacy_anchor", "mean_brier"].mean()
        ),
        "runtime_variance_proxy": runtime_proxy,
        "latest_goldshot_metadata": latest_goldshot,
        "market_blend_gender_gap": {
            "season_dispersion_mean": float(spread_only["equal_gender_mean_brier"].std(ddof=0)) if len(spread_only) > 1 else 0.0,
            "best_variant_recent_window_brier": recent_best,
            "strength_full_vs_base_equal_gender_delta": (
                float(best_strength_full_score - best_base_score)
                if not math.isnan(best_strength_full_score) and not math.isnan(best_base_score)
                else None
            ),
            "external_base_pruned_vs_base_equal_gender_delta": (
                float(best_external_base_pruned_score - best_base_score)
                if not math.isnan(best_external_base_pruned_score) and not math.isnan(best_base_score)
                else None
            ),
            "strength_recent_vs_base_equal_gender_delta": (
                float(best_strength_recent_score - best_base_score)
                if not math.isnan(best_strength_recent_score) and not math.isnan(best_base_score)
                else None
            ),
        },
    }
    (RESULTS / "postmortem_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "postmortem_findings.md").write_text(
        _build_findings(ablation_table, combined, primary_metric, runtime_proxy, latest_goldshot),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
