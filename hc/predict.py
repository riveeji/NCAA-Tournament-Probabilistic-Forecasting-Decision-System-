from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import math
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hc.constants import DATA_DIR, DEFAULT_SUBMISSION_NAME, MARKET_POLICY_BY_PROFILE, PROFILE_AGGRESSIVE, PROFILE_CHOICES, RESULTS_DIR, TrainConfig
from hc.data_build import build_all
from hc.data_sources import (
    aggregate_matchup_model_consensus,
    aggregate_market_consensus,
    find_live_model_matchup_paths,
    find_live_market_source_paths,
    read_csv_if_exists,
    standardize_market_frame,
    standardize_matchup_model_frame,
)
from hc.features_structured import (
    augment_team_snapshots_with_public_ratings,
    build_hc_matchups,
    feature_views as build_feature_views,
    load_team_snapshots,
)
from hc.features_text import attach_text_matchup_features, load_text_embeddings
from hc.fusion import fit_meta_models, predict_meta
from hc.legacy_anchor import load_legacy_submission_anchor, merge_legacy_anchor_oof
from hc.models_routes import fit_full_models, generate_base_oof, model_specs_for_gender, predict_full_models
from hc.signals import canonicalize_matchup_signal_frame, coalesce_matchup_signal_frames
from zizzii_train import build_prediction_frame, infer_feature_candidates, safe_clip


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def combine_numeric_max(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    series_list = [numeric_series(frame, column) for column in columns if column in frame.columns]
    if not series_list:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    combined = pd.concat(series_list, axis=1)
    return combined.max(axis=1, skipna=True)


def combine_numeric_min(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    series_list = [numeric_series(frame, column) for column in columns if column in frame.columns]
    if not series_list:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    combined = pd.concat(series_list, axis=1)
    return combined.min(axis=1, skipna=True)


def apply_men_live_silver_rules(
    pred_frame: pd.DataFrame,
    base_pred: pd.DataFrame,
    final_prob: np.ndarray,
    season: int,
) -> tuple[np.ndarray, dict[str, int], pd.DataFrame]:
    adjusted = np.asarray(final_prob, dtype=float).copy()
    summary = {
        "men_live_rule_market_rows": 0,
        "men_live_rule_silver_matchup_rows": 0,
        "men_live_rule_model_matchup_rows": 0,
        "men_live_rule_confirm_rows": 0,
        "men_live_rule_extreme_rows": 0,
    }
    if season < 2026 or pred_frame.empty:
        return adjusted, summary, pd.DataFrame()
    if "MarketProb" not in pred_frame.columns:
        return adjusted, summary, pd.DataFrame()

    market_prob = numeric_series(pred_frame, "MarketProb")
    spread = numeric_series(pred_frame, "LastSpread")
    abs_seed_diff = numeric_series(pred_frame, "AbsSeedDiff")
    silver_net = combine_numeric_max(pred_frame, ["D_SB_BNetRating", "D_SB_NetRating"])
    silver_elo = combine_numeric_max(pred_frame, ["D_SB_BXelo", "D_SB_Xelo", "D_SB_DisplayElo"])
    silver_rank_edge = -combine_numeric_min(pred_frame, ["D_SB_DisplayRank"])
    silver_adjusted_composite = combine_numeric_max(pred_frame, ["D_SB_AdjustedComposite", "D_SB_Composite", "D_SB_Cooper"])
    silver_matchup_prob = numeric_series(pred_frame, "SilverMatchupProb")
    silver_matchup_spread = numeric_series(pred_frame, "SilverMatchupSpread")
    model_matchup_prob = combine_numeric_max(
        pred_frame,
        ["SilverMatchupProb", "ModelMatchupProb", "ModelMatchupProbMean", "ModelMatchupProbMedian"],
    )
    model_matchup_spread = combine_numeric_min(
        pred_frame,
        ["SilverMatchupSpread", "ModelMatchupSpread", "ModelMatchupSpreadMean", "ModelMatchupSpreadMedian"],
    )
    model_source_count = numeric_series(pred_frame, "ModelSourceCount")

    market_public = numeric_series(base_pred, "Prob_market_public_lr")
    market_plus_stats = numeric_series(base_pred, "Prob_market_plus_stats_lr")
    market_only = numeric_series(base_pred, "Prob_market_only_lr")

    market_gate = market_prob.notna() & (
        (market_prob >= 0.79)
        | (spread.notna() & (spread <= -7.5))
        | (abs_seed_diff >= 5.0)
    )
    silver_matchup_gate = model_matchup_prob.notna() & (
        (model_matchup_prob >= 0.80)
        | (model_matchup_spread.notna() & (model_matchup_spread <= -7.5))
    )
    active_gate = market_gate | silver_matchup_gate
    if not bool(active_gate.any()):
        return adjusted, summary, pd.DataFrame()

    target_floor = pd.Series(adjusted, index=pred_frame.index, dtype=float)
    target_floor = target_floor.combine(market_prob, np.fmax)
    target_floor = target_floor.combine(market_public, np.fmax)
    target_floor = target_floor.combine(market_plus_stats, np.fmax)
    target_floor = target_floor.combine(market_only, np.fmax)
    target_floor = target_floor.combine(model_matchup_prob, np.fmax)
    target_floor = target_floor.clip(upper=0.985)
    lagging = pd.Series(adjusted, index=pred_frame.index) + 0.015 < target_floor

    confirm_gate = active_gate & lagging & (
        (model_matchup_prob >= 0.82)
        | ((model_matchup_prob >= 0.76) & model_matchup_spread.notna() & (model_matchup_spread <= -6.5))
        | ((model_source_count >= 2.0) & (model_matchup_prob >= 0.74))
        | (silver_net >= 8.0)
        | (silver_elo >= 180.0)
        | (silver_rank_edge >= 25.0)
        | (silver_adjusted_composite >= 120.0)
        | ((silver_net >= 5.0) & (silver_rank_edge >= 15.0))
        | ((silver_elo >= 120.0) & (silver_rank_edge >= 18.0))
        | ((model_matchup_prob >= 0.72) & (silver_adjusted_composite >= 60.0))
    )
    extreme_gate = confirm_gate & (
        (market_prob >= 0.90)
        | (model_matchup_prob >= 0.92)
        | ((spread.notna()) & (spread <= -13.5))
        | ((model_matchup_spread.notna()) & (model_matchup_spread <= -12.5))
        | ((silver_net >= 12.0) & (abs_seed_diff >= 7.0))
    )

    if bool(confirm_gate.any()):
        confirm_floor = target_floor.clip(upper=0.94)
        idx = confirm_gate.fillna(False).to_numpy()
        adjusted[idx] = np.maximum(adjusted[idx], confirm_floor.loc[confirm_gate].to_numpy())
    if bool(extreme_gate.any()):
        extreme_floor = target_floor.clip(upper=0.975)
        idx = extreme_gate.fillna(False).to_numpy()
        adjusted[idx] = np.maximum(adjusted[idx], extreme_floor.loc[extreme_gate].to_numpy())

    summary["men_live_rule_market_rows"] = int(market_gate.fillna(False).sum())
    summary["men_live_rule_silver_matchup_rows"] = int(silver_matchup_gate.fillna(False).sum())
    summary["men_live_rule_model_matchup_rows"] = int(model_matchup_prob.notna().sum())
    summary["men_live_rule_confirm_rows"] = int(confirm_gate.fillna(False).sum())
    summary["men_live_rule_extreme_rows"] = int(extreme_gate.fillna(False).sum())
    detail = pred_frame.loc[confirm_gate.fillna(False), ["ID"]].copy()
    if not detail.empty:
        detail["RuleGroup"] = np.where(extreme_gate.loc[detail.index].fillna(False), "extreme", "confirm")
        detail["MarketProb"] = market_prob.loc[detail.index].to_numpy()
        detail["LastSpread"] = spread.loc[detail.index].to_numpy()
        detail["AbsSeedDiff"] = abs_seed_diff.loc[detail.index].to_numpy()
        detail["SilverNet"] = silver_net.loc[detail.index].to_numpy()
        detail["SilverElo"] = silver_elo.loc[detail.index].to_numpy()
        detail["SilverRankEdge"] = silver_rank_edge.loc[detail.index].to_numpy()
        detail["SilverAdjustedComposite"] = silver_adjusted_composite.loc[detail.index].to_numpy()
        detail["SilverMatchupProb"] = silver_matchup_prob.loc[detail.index].to_numpy()
        detail["SilverMatchupSpread"] = silver_matchup_spread.loc[detail.index].to_numpy()
        detail["ModelMatchupProb"] = model_matchup_prob.loc[detail.index].to_numpy()
        detail["ModelMatchupSpread"] = model_matchup_spread.loc[detail.index].to_numpy()
        detail["ModelSourceCount"] = model_source_count.loc[detail.index].to_numpy()
        detail["PreProb"] = np.asarray(final_prob, dtype=float)[detail.index]
        detail["PostProb"] = adjusted[detail.index]
    return safe_clip(adjusted), summary, detail.reset_index(drop=True)


def apply_women_live_narrow_rules(
    pred_frame: pd.DataFrame,
    base_pred: pd.DataFrame,
    final_prob: np.ndarray,
    season: int,
) -> tuple[np.ndarray, dict[str, int], pd.DataFrame]:
    adjusted = np.asarray(final_prob, dtype=float).copy()
    summary = {
        "women_live_rule_host_rows": 0,
        "women_live_rule_model_matchup_rows": 0,
        "women_live_rule_strong_rows": 0,
        "women_live_rule_extreme_rows": 0,
        "women_live_rule_silver_confirm_rows": 0,
    }
    if season < 2026 or pred_frame.empty:
        return adjusted, summary, pd.DataFrame()
    required = {"MarketProb", "D_HostLikely", "IsRound1Or2"}
    if not required.issubset(pred_frame.columns):
        return adjusted, summary, pd.DataFrame()

    market_prob = numeric_series(pred_frame, "MarketProb")
    host_likely = numeric_series(pred_frame, "D_HostLikely")
    early_round = numeric_series(pred_frame, "IsRound1Or2")
    spread = numeric_series(pred_frame, "LastSpread")
    abs_seed_diff = numeric_series(pred_frame, "AbsSeedDiff")
    silver_net = combine_numeric_max(pred_frame, ["D_SB_BNetRating", "D_SB_NetRating"])
    silver_rank_edge = -combine_numeric_min(pred_frame, ["D_SB_DisplayRank"])
    silver_host = combine_numeric_max(pred_frame, ["D_SB_CurrentHFA", "D_SB_XeloCurrentHFA", "D_SB_HomeCourtDisplay"])
    model_matchup_prob = combine_numeric_max(
        pred_frame,
        ["SilverMatchupProb", "ModelMatchupProb", "ModelMatchupProbMean", "ModelMatchupProbMedian"],
    )
    model_matchup_spread = combine_numeric_min(
        pred_frame,
        ["SilverMatchupSpread", "ModelMatchupSpread", "ModelMatchupSpreadMean", "ModelMatchupSpreadMedian"],
    )
    women_min_lr = numeric_series(base_pred, "Prob_women_min_lr")
    women_market_histgb = numeric_series(base_pred, "Prob_women_market_histgb")

    host_gate = market_prob.notna() & (host_likely >= 1.0) & (early_round >= 1.0)
    if not bool(host_gate.any()):
        return adjusted, summary, pd.DataFrame()

    target_floor = pd.Series(np.nan, index=pred_frame.index, dtype=float)
    target_floor = target_floor.combine(market_prob, np.fmax)
    target_floor = target_floor.combine(women_min_lr, np.fmax)
    target_floor = target_floor.combine(women_market_histgb, np.fmax)
    target_floor = target_floor.clip(upper=0.985)
    lagging_market = pd.Series(adjusted, index=pred_frame.index) + 0.02 < target_floor
    silver_confirm = (
        (silver_net >= 5.0)
        | (silver_rank_edge >= 18.0)
        | ((silver_host >= 8.0) & (host_likely >= 1.0))
        | (model_matchup_prob >= 0.82)
    )

    strong_gate = host_gate & lagging_market & silver_confirm & (
        (market_prob >= 0.86)
        | ((market_prob >= 0.82) & spread.notna() & (spread <= -6.5))
        | ((market_prob >= 0.78) & spread.notna() & (spread <= -10.5) & (abs_seed_diff >= 5.0))
        | ((model_matchup_prob >= 0.84) & model_matchup_spread.notna() & (model_matchup_spread <= -5.5))
    )
    extreme_gate = host_gate & lagging_market & silver_confirm & (
        (market_prob >= 0.93)
        | ((market_prob >= 0.86) & spread.notna() & (spread <= -13.5))
        | ((model_matchup_prob >= 0.90) & model_matchup_spread.notna() & (model_matchup_spread <= -10.5))
    )

    if bool(strong_gate.any()):
        strong_floor = target_floor.clip(upper=0.93)
        strong_idx = strong_gate.fillna(False).to_numpy()
        adjusted[strong_idx] = np.maximum(adjusted[strong_idx], strong_floor.loc[strong_gate].to_numpy())
    if bool(extreme_gate.any()):
        extreme_floor = target_floor.clip(upper=0.97)
        extreme_idx = extreme_gate.fillna(False).to_numpy()
        adjusted[extreme_idx] = np.maximum(adjusted[extreme_idx], extreme_floor.loc[extreme_gate].to_numpy())

    summary["women_live_rule_host_rows"] = int(host_gate.fillna(False).sum())
    summary["women_live_rule_model_matchup_rows"] = int(model_matchup_prob.notna().sum())
    summary["women_live_rule_strong_rows"] = int(strong_gate.fillna(False).sum())
    summary["women_live_rule_extreme_rows"] = int(extreme_gate.fillna(False).sum())
    summary["women_live_rule_silver_confirm_rows"] = int(silver_confirm.fillna(False).sum())
    detail = pred_frame.loc[strong_gate.fillna(False) | extreme_gate.fillna(False), ["ID"]].copy()
    if not detail.empty:
        detail["RuleGroup"] = np.where(extreme_gate.loc[detail.index].fillna(False), "extreme", "strong")
        detail["MarketProb"] = market_prob.loc[detail.index].to_numpy()
        detail["LastSpread"] = spread.loc[detail.index].to_numpy()
        detail["AbsSeedDiff"] = abs_seed_diff.loc[detail.index].to_numpy()
        detail["D_HostLikely"] = host_likely.loc[detail.index].to_numpy()
        detail["IsRound1Or2"] = early_round.loc[detail.index].to_numpy()
        detail["SilverNet"] = silver_net.loc[detail.index].to_numpy()
        detail["SilverRankEdge"] = silver_rank_edge.loc[detail.index].to_numpy()
        detail["SilverHost"] = silver_host.loc[detail.index].to_numpy()
        detail["ModelMatchupProb"] = model_matchup_prob.loc[detail.index].to_numpy()
        detail["ModelMatchupSpread"] = model_matchup_spread.loc[detail.index].to_numpy()
        detail["PreProb"] = np.asarray(final_prob, dtype=float)[detail.index]
        detail["PostProb"] = adjusted[detail.index]
    return safe_clip(adjusted), summary, detail.reset_index(drop=True)


def load_live_market_frame(gender: str, season: int) -> pd.DataFrame:
    source_paths = find_live_market_source_paths(gender, season, RESULTS_DIR.parent / "external-data")
    frames: list[pd.DataFrame] = []
    for path in source_paths:
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        if "Source" not in frame.columns:
            frame = frame.copy()
            frame["Source"] = path.name
        frames.append(standardize_market_frame(frame))
    if not frames:
        return pd.DataFrame(columns=["Season", "T1", "T2", "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread"])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["Season"].eq(season)].copy()
    if frame.empty:
        return frame
    frame = aggregate_market_consensus(frame)
    if "MarketProb" in frame.columns:
        frame["MarketLogit"] = pd.to_numeric(frame["MarketProb"], errors="coerce").apply(
            lambda value: float(math.log(min(max(value, 1e-6), 1.0 - 1e-6) / (1.0 - min(max(value, 1e-6), 1.0 - 1e-6))))
            if pd.notna(value)
            else pd.NA
        )
        frame["MarketConfidence"] = (pd.to_numeric(frame["MarketProb"], errors="coerce") - 0.5).abs() * 2.0
    return frame


def load_live_model_matchup_frame(gender: str, season: int) -> pd.DataFrame:
    source_paths = find_live_model_matchup_paths(gender, season, RESULTS_DIR.parent / "external-data")
    frames: list[pd.DataFrame] = []
    silver_frames: list[pd.DataFrame] = []
    for path in source_paths:
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        current = standardize_matchup_model_frame(frame, source_name=path.name)
        if current.empty:
            continue
        current = current[current["Season"].eq(season)].copy()
        if current.empty:
            continue
        frames.append(current)
        if "silverbulletin" in path.name.lower():
            silver_current = current.rename(
                columns={
                    "ModelProb": "SilverMatchupProb",
                    "ModelSpread": "SilverMatchupSpread",
                    "ModelProjectedTotal": "SilverProjectedTotal",
                    "ModelRound": "SilverRound",
                    "Source": "SignalSource",
                }
            )
            silver_frames.append(
                canonicalize_matchup_signal_frame(
                    silver_current[
                        [
                            "Season",
                            "T1",
                            "T2",
                            "SilverMatchupProb",
                            "SilverMatchupSpread",
                            "SilverProjectedTotal",
                            "SilverRound",
                            "SnapshotTime",
                            "SignalSource",
                        ]
                    ],
                    source=path.name,
                    priority=20,
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                "Season",
                "T1",
                "T2",
                "ModelMatchupProb",
                "ModelMatchupSpread",
                "ModelProjectedTotal",
                "ModelRound",
                "ModelMatchupProbMean",
                "ModelMatchupProbMedian",
                "ModelMatchupProbStd",
                "ModelMatchupSpreadMean",
                "ModelMatchupSpreadMedian",
                "ModelMatchupSpreadStd",
                "ModelSourceCount",
                "ModelRowCount",
                "ModelSourceList",
                "SilverMatchupProb",
                "SilverMatchupSpread",
                "SilverProjectedTotal",
                "SilverRound",
            ]
        )
    model_frame = aggregate_matchup_model_consensus(pd.concat(frames, ignore_index=True))
    if silver_frames:
        silver_frame = coalesce_matchup_signal_frames(silver_frames)
        model_frame = model_frame.merge(silver_frame, on=["Season", "T1", "T2"], how="left")
    return model_frame.reset_index(drop=True)


def build_hc_prediction_frame(
    gender: str,
    season: int,
    text_dim: int,
    include_text: bool,
    include_aggressive_public: bool,
    template_path: str | None = None,
) -> pd.DataFrame:
    team_feats = load_team_snapshots(gender)
    if season not in set(pd.to_numeric(team_feats["Season"], errors="coerce").dropna().astype(int).unique()):
        from zizzii_features import build_team_features

        team_feats = build_team_features(gender=gender)
    if include_aggressive_public:
        team_feats = augment_team_snapshots_with_public_ratings(team_feats, gender, include_silver_history=False)
    feature_candidates = infer_feature_candidates(team_feats)
    sample_submission = pd.read_csv(Path(template_path) if template_path else (DATA_DIR / "SampleSubmissionStage2.csv"))
    market_df = load_live_market_frame(gender, season)
    frame = build_prediction_frame(
        sample_submission,
        team_feats,
        feature_candidates,
        season=season,
        gender=gender,
        market_df=market_df,
        signal_df=None,
        postrule_df=None,
    )
    if include_text:
        text_df = load_text_embeddings(gender, text_dim)
        frame, _ = attach_text_matchup_features(frame, text_df)
    model_matchups = load_live_model_matchup_frame(gender, season)
    if not model_matchups.empty:
        frame = frame.merge(model_matchups, on=["Season", "T1", "T2"], how="left")
    return frame


def train_and_predict_gender(
    gender: str,
    season: int,
    market_policy: str,
    profile: str,
    use_text: bool,
    use_tabpfn: bool,
    text_dim: int,
    template_path: str | None = None,
    strict_replay: bool = False,
    runtime_rules: str = "silver",
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    build_all(genders=(gender,), text_dim=text_dim)
    use_public = profile == PROFILE_AGGRESSIVE
    train_matchups = build_hc_matchups(
        gender,
        market_policy=market_policy,
        text_dim=text_dim,
        include_text=use_text,
        profile=profile,
        include_aggressive_public=use_public,
    )
    if strict_replay:
        season_series = pd.to_numeric(train_matchups["Season"], errors="coerce")
        train_matchups = train_matchups.loc[season_series.lt(season)].copy()
        if train_matchups.empty:
            raise ValueError(
                f"Strict historical replay for gender={gender} season={season} has no prior-season training rows."
            )
    views = build_feature_views(
        train_matchups,
        gender,
        text_enabled=use_text,
        tabpfn_enabled=use_tabpfn,
        include_public_route=use_public,
    )
    specs = model_specs_for_gender(gender, views, use_tabpfn=use_tabpfn)
    base_oof = generate_base_oof(train_matchups, views, gender, specs)
    base_oof, _ = merge_legacy_anchor_oof(base_oof, gender, train_matchups)
    meta_bundle = fit_meta_models(base_oof, gender)
    full_models = fit_full_models(train_matchups, views, gender, specs)
    pred_frame = build_hc_prediction_frame(
        gender,
        season=season,
        text_dim=text_dim,
        include_text=use_text,
        include_aggressive_public=use_public,
        template_path=template_path,
    )
    base_pred = predict_full_models(pred_frame, views, gender, specs, full_models)
    if not strict_replay:
        legacy_pred = load_legacy_submission_anchor(gender, season)
        if not legacy_pred.empty:
            base_pred = base_pred.merge(legacy_pred, on="ID", how="left")
    final_prob = predict_meta(base_pred, meta_bundle, gender)
    extra_summary: dict[str, int] = {}
    runtime_detail = pd.DataFrame()
    if not strict_replay and runtime_rules != "off":
        if gender == "M":
            final_prob, extra_summary, runtime_detail = apply_men_live_silver_rules(pred_frame, base_pred, final_prob, season)
        elif gender == "W":
            final_prob, extra_summary, runtime_detail = apply_women_live_narrow_rules(pred_frame, base_pred, final_prob, season)
    return pd.DataFrame({"ID": pred_frame["ID"], "Pred": final_prob}), extra_summary, runtime_detail


def write_runtime_trigger_report(
    run_id: str,
    season: int,
    runtime_mode: str,
    men_summary: dict[str, int],
    women_summary: dict[str, int],
    men_detail: pd.DataFrame,
    women_detail: pd.DataFrame,
) -> dict[str, str]:
    payload = {
        "run_id": run_id,
        "season": int(season),
        "runtime_rules": runtime_mode,
        "armed": bool(season >= 2026 and runtime_mode != "off"),
        "men_summary": men_summary,
        "women_summary": women_summary,
        "men_trigger_rows": int(len(men_detail)),
        "women_trigger_rows": int(len(women_detail)),
    }
    json_path = RESULTS_DIR / f"hc_runtime_trigger_report_{run_id}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {"json": str(json_path)}
    if not men_detail.empty:
        men_path = RESULTS_DIR / f"hc_runtime_trigger_report_{run_id}_men.csv"
        men_detail.sort_values(["RuleGroup", "PostProb"], ascending=[True, False]).to_csv(men_path, index=False)
        outputs["men_csv"] = str(men_path)
    if not women_detail.empty:
        women_path = RESULTS_DIR / f"hc_runtime_trigger_report_{run_id}_women.csv"
        women_detail.sort_values(["RuleGroup", "PostProb"], ascending=[True, False]).to_csv(women_path, index=False)
        outputs["women_csv"] = str(women_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single HC final submission.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--market-policy", default=None)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=PROFILE_AGGRESSIVE)
    parser.add_argument("--text", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tabpfn", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--text-dim", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--template-path", default=None)
    parser.add_argument("--output", default=DEFAULT_SUBMISSION_NAME)
    parser.add_argument("--runtime-rules", choices=["silver", "off"], default="silver")
    parser.add_argument(
        "--strict-replay",
        action="store_true",
        help="For historical seasons, train only on seasons earlier than the target season and disable same-season legacy anchor.",
    )
    args = parser.parse_args()

    output_stem = Path(args.output).stem.replace(" ", "_")
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{args.runtime_rules}_{output_stem}"
    use_text = args.text != "off"
    use_tabpfn = args.tabpfn == "on"

    men, men_summary, men_detail = train_and_predict_gender(
        "M",
        args.season,
        args.market_policy or MARKET_POLICY_BY_PROFILE[args.profile]["M"],
        args.profile,
        use_text,
        use_tabpfn,
        args.text_dim,
        args.template_path,
        args.strict_replay,
        args.runtime_rules,
    )
    women, women_summary, women_detail = train_and_predict_gender(
        "W",
        args.season,
        args.market_policy or MARKET_POLICY_BY_PROFILE[args.profile]["W"],
        args.profile,
        use_text,
        use_tabpfn,
        args.text_dim,
        args.template_path,
        args.strict_replay,
        args.runtime_rules,
    )

    submission = pd.concat([men, women], ignore_index=True).sort_values("ID").reset_index(drop=True)
    output_path = Path(args.output)
    submission.to_csv(output_path, index=False)
    result = {
        "run_id": run_id,
        "season": args.season,
        "profile": args.profile,
        "strict_replay": bool(args.strict_replay),
        "runtime_rules": args.runtime_rules,
        "rows": int(len(submission)),
        "output": str(output_path),
        "women_live_rule_summary": women_summary,
        "men_live_rule_summary": men_summary,
    }
    summary_path = RESULTS_DIR / f"hc_submission_summary_{run_id}.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_report = write_runtime_trigger_report(
        run_id,
        args.season,
        args.runtime_rules,
        men_summary,
        women_summary,
        men_detail,
        women_detail,
    )
    print(f"HC submission written to: {output_path}")
    print(f"HC summary written to: {summary_path}")
    print(f"HC runtime report written to: {runtime_report['json']}")


if __name__ == "__main__":
    main()
