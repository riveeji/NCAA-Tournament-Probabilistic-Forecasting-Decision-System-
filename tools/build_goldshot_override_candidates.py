from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.predict import load_live_market_frame, load_live_model_matchup_frame
from tools.goldshot_utils import (
    DATA_DIR,
    RESULTS_DIR,
    apply_override_guardrails,
    build_official_round_map,
    favorite_direction_change,
    favorite_side_from_market,
    load_men_watchlist,
    load_seed_details,
    load_team_maps,
    market_and_spread_consistent,
    model_aligned_with_market,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatic and manual goldshot override candidates.")
    parser.add_argument("--current", required=True, help="Current HC submission candidate.")
    parser.add_argument("--baseline", required=True, help="Baseline submission candidate.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--manual-csv-output", required=True)
    parser.add_argument("--manual-json-output", required=True)
    return parser.parse_args()


def records_for_json(frame: pd.DataFrame, n: int) -> list[dict[str, object]]:
    if frame.empty:
        return []
    cleaned = frame.head(n).copy()
    cleaned = cleaned.apply(lambda column: column.map(_json_safe_scalar))
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict("records")


def _json_safe_scalar(value: object) -> object:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_submission(path: str, pred_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["ID", "Pred"]).rename(columns={"Pred": pred_col})
    ids = frame["ID"].astype(str).str.split("_", expand=True)
    frame["Season"] = pd.to_numeric(ids[0], errors="coerce")
    frame["T1"] = pd.to_numeric(ids[1], errors="coerce")
    frame["T2"] = pd.to_numeric(ids[2], errors="coerce")
    frame = frame.dropna(subset=["Season", "T1", "T2"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["T1"] = frame["T1"].astype(int)
    frame["T2"] = frame["T2"].astype(int)
    return frame


def attach_team_context(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    men_map, women_map = load_team_maps(DATA_DIR)
    men_seeds = load_seed_details("M", season, DATA_DIR)
    women_seeds = load_seed_details("W", season, DATA_DIR)
    men_watch = load_men_watchlist(season, RESULTS_DIR)
    round_map = pd.concat(
        [
            build_official_round_map("M", season),
            build_official_round_map("W", season),
        ],
        ignore_index=True,
    )

    out = frame.copy()

    def infer_gender(t1: int, t2: int) -> str:
        if t1 in men_map and t2 in men_map:
            return "M"
        if t1 in women_map and t2 in women_map:
            return "W"
        return "?"

    out["Gender"] = [infer_gender(t1, t2) for t1, t2 in zip(out["T1"], out["T2"])]
    out["Team1Name"] = out.apply(
        lambda row: men_map.get(row["T1"], "") if row["Gender"] == "M" else women_map.get(row["T1"], ""),
        axis=1,
    )
    out["Team2Name"] = out.apply(
        lambda row: men_map.get(row["T2"], "") if row["Gender"] == "M" else women_map.get(row["T2"], ""),
        axis=1,
    )

    if not men_seeds.empty:
        m1 = men_seeds.drop(columns=["Season"]).add_prefix("Team1_").rename(columns={"Team1_TeamID": "T1"})
        m2 = men_seeds.drop(columns=["Season"]).add_prefix("Team2_").rename(columns={"Team2_TeamID": "T2"})
        out = out.merge(m1, on=["T1"], how="left")
        out = out.merge(m2, on=["T2"], how="left")
    if not women_seeds.empty:
        w1 = women_seeds.drop(columns=["Season"]).add_prefix("WTeam1_").rename(columns={"WTeam1_TeamID": "T1"})
        w2 = women_seeds.drop(columns=["Season"]).add_prefix("WTeam2_").rename(columns={"WTeam2_TeamID": "T2"})
        out = out.merge(w1, on=["T1"], how="left")
        out = out.merge(w2, on=["T2"], how="left")

    out["Team1Seed"] = np.where(out["Gender"].eq("M"), out.get("Team1_Seed"), out.get("WTeam1_Seed"))
    out["Team2Seed"] = np.where(out["Gender"].eq("M"), out.get("Team2_Seed"), out.get("WTeam2_Seed"))
    out["Team1SeedNum"] = np.where(out["Gender"].eq("M"), out.get("Team1_SeedNum"), out.get("WTeam1_SeedNum"))
    out["Team2SeedNum"] = np.where(out["Gender"].eq("M"), out.get("Team2_SeedNum"), out.get("WTeam2_SeedNum"))
    out["Team1InField"] = pd.Series(out["Team1Seed"]).fillna("").astype(str).ne("").to_numpy()
    out["Team2InField"] = pd.Series(out["Team2Seed"]).fillna("").astype(str).ne("").to_numpy()
    out["BothInField"] = out["Team1InField"] & out["Team2InField"]

    if not men_watch.empty:
        watch1 = men_watch.add_prefix("Team1_").rename(columns={"Team1_TeamID": "T1"})
        watch2 = men_watch.add_prefix("Team2_").rename(columns={"Team2_TeamID": "T2"})
        out = out.merge(watch1, on="T1", how="left")
        out = out.merge(watch2, on="T2", how="left")
    else:
        for prefix in ("Team1_", "Team2_"):
            for column in ["SeverityScore", "OutCount", "GTDCount", "PlayerCount", "Players", "Statuses", "Injuries"]:
                out[f"{prefix}{column}"] = pd.NA

    out["MenInjuryRisk"] = (
        out["Gender"].eq("M")
        & (
            pd.to_numeric(out.get("Team1_SeverityScore"), errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(out.get("Team2_SeverityScore"), errors="coerce").fillna(0).gt(0)
        )
    )
    if not round_map.empty:
        out = out.merge(round_map, on=["Season", "T1", "T2"], how="left")
    else:
        out["OfficialMinRound"] = np.nan
        out["OfficialRoundLabel"] = pd.NA
        out["IsPlayIn"] = False
        out["IsRound1"] = False
        out["IsRound2"] = False
    return out


def load_market_and_model(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    for gender in ("M", "W"):
        market = load_live_market_frame(gender, season)
        if not market.empty:
            keep = [
                "Season",
                "T1",
                "T2",
                "MarketProb",
                "MarketProbMean",
                "MarketProbMedian",
                "MarketProbStd",
                "LastSpread",
                "SpreadMean",
                "SpreadMedian",
                "SpreadStd",
                "BookCountMean",
                "BookCountMax",
                "BookCountTotal",
                "MarketRowCount",
                "MarketSourceCount",
                "SnapshotTime",
                "Source",
            ]
            market_frames.append(market[[column for column in keep if column in market.columns]].copy())

        model = load_live_model_matchup_frame(gender, season)
        if not model.empty:
            keep = [
                "Season",
                "T1",
                "T2",
                "ModelMatchupProb",
                "ModelMatchupProbMean",
                "ModelMatchupProbMedian",
                "ModelMatchupProbStd",
                "ModelMatchupSpread",
                "ModelMatchupSpreadMean",
                "ModelMatchupSpreadMedian",
                "ModelMatchupSpreadStd",
                "ModelSourceCount",
                "ModelRowCount",
                "ModelSourceList",
                "SnapshotTime",
            ]
            model_frames.append(model[[column for column in keep if column in model.columns]].copy())

    market_out = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
    model_out = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    return market_out, model_out


def compute_goldshot_scope(out: pd.DataFrame) -> pd.DataFrame:
    frame = out.copy()
    team1_seed_num = pd.to_numeric(frame.get("Team1SeedNum"), errors="coerce")
    team2_seed_num = pd.to_numeric(frame.get("Team2SeedNum"), errors="coerce")
    min_seed = pd.concat([team1_seed_num, team2_seed_num], axis=1).min(axis=1, skipna=True)
    frame["SeedMin"] = min_seed
    round_num = pd.to_numeric(frame.get("OfficialMinRound"), errors="coerce")
    frame["RealGameScope"] = False
    frame.loc[round_num.eq(0) & frame["BothInField"], "RealGameScope"] = True
    frame.loc[round_num.eq(1) & frame["BothInField"], "RealGameScope"] = True
    frame.loc[
        round_num.eq(2)
        & frame["BothInField"]
        & ((min_seed <= 4.0) | frame["MenInjuryRisk"].fillna(False)),
        "RealGameScope",
    ] = True
    return frame


def build_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    current = load_submission(args.current, "HCPred")
    baseline = load_submission(args.baseline, "BaselinePred")
    merged = current.merge(baseline[["ID", "BaselinePred"]], on="ID", how="left")
    merged = merged.loc[merged["Season"].eq(int(args.season))].copy()
    merged = attach_team_context(merged, args.season)
    merged = compute_goldshot_scope(merged)

    market, model = load_market_and_model(args.season)
    if not market.empty:
        merged = merged.merge(market, on=["Season", "T1", "T2"], how="left")
    if not model.empty:
        merged = merged.merge(model, on=["Season", "T1", "T2"], how="left")

    market_prob = pd.to_numeric(merged.get("MarketProbMedian"), errors="coerce")
    spread = pd.to_numeric(merged.get("SpreadMedian"), errors="coerce")
    if spread.isna().all():
        spread = pd.to_numeric(merged.get("LastSpread"), errors="coerce")
    model_prob = pd.to_numeric(merged.get("ModelMatchupProbMedian"), errors="coerce")
    if model_prob.isna().all():
        model_prob = pd.to_numeric(merged.get("ModelMatchupProb"), errors="coerce")

    merged["MarketProbMedian"] = market_prob
    merged["SpreadMedian"] = spread
    merged["ModelMatchupProbMedian"] = model_prob

    merged["MarketAvailable"] = market_prob.notna()
    merged["EffectiveBookCount"] = pd.concat(
        [
            pd.to_numeric(merged.get("BookCountMax"), errors="coerce"),
            pd.to_numeric(merged.get("BookCountTotal"), errors="coerce"),
            pd.to_numeric(merged.get("MarketRowCount"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    merged["MarketConsensusOK"] = (
        pd.to_numeric(merged.get("MarketSourceCount"), errors="coerce").fillna(0.0).ge(2.0)
        & pd.to_numeric(merged.get("EffectiveBookCount"), errors="coerce").fillna(0.0).ge(3.0)
    )
    merged["MarketSpreadConsistent"] = [
        market_and_spread_consistent(m_prob, spr)
        for m_prob, spr in zip(market_prob, spread)
    ]
    merged["ModelAvailable"] = model_prob.notna()
    merged["ModelAligned"] = [
        model_aligned_with_market(mod_prob, m_prob)
        for mod_prob, m_prob in zip(model_prob, market_prob)
    ]
    merged["HCGapToMarketMedian"] = (pd.to_numeric(merged["HCPred"], errors="coerce") - market_prob).abs()
    merged["BaselineGapToMarketMedian"] = (pd.to_numeric(merged["BaselinePred"], errors="coerce") - market_prob).abs()
    merged["BaselineCloser"] = (
        pd.to_numeric(merged["BaselineGapToMarketMedian"], errors="coerce")
        + 0.03
        < pd.to_numeric(merged["HCGapToMarketMedian"], errors="coerce")
    )
    merged["FavoriteSide"] = [
        favorite_side_from_market(m_prob, spr)
        for m_prob, spr in zip(market_prob, spread)
    ]

    team1_sev = pd.to_numeric(merged.get("Team1_SeverityScore"), errors="coerce").fillna(0.0)
    team2_sev = pd.to_numeric(merged.get("Team2_SeverityScore"), errors="coerce").fillna(0.0)
    team1_out = pd.to_numeric(merged.get("Team1_OutCount"), errors="coerce").fillna(0.0)
    team2_out = pd.to_numeric(merged.get("Team2_OutCount"), errors="coerce").fillna(0.0)
    team1_gtd = pd.to_numeric(merged.get("Team1_GTDCount"), errors="coerce").fillna(0.0)
    team2_gtd = pd.to_numeric(merged.get("Team2_GTDCount"), errors="coerce").fillna(0.0)
    merged["FavoriteHasInjuryVeto"] = False
    fav_t1 = merged["FavoriteSide"].eq("T1")
    fav_t2 = merged["FavoriteSide"].eq("T2")
    merged.loc[
        merged["Gender"].eq("M") & fav_t1 & (team1_sev >= 4.0) & ((team1_out >= 1.0) | (team1_gtd >= 2.0)),
        "FavoriteHasInjuryVeto",
    ] = True
    merged.loc[
        merged["Gender"].eq("M") & fav_t2 & (team2_sev >= 4.0) & ((team2_out >= 1.0) | (team2_gtd >= 2.0)),
        "FavoriteHasInjuryVeto",
    ] = True

    target_prob = market_prob.copy()
    aligned_mask = merged["ModelAvailable"] & merged["ModelAligned"]
    target_prob.loc[aligned_mask] = 0.70 * market_prob.loc[aligned_mask] + 0.30 * model_prob.loc[aligned_mask]
    merged["TargetProb"] = target_prob

    auto_condition = (
        merged["RealGameScope"].fillna(False)
        & merged["MarketAvailable"].fillna(False)
        & merged["MarketConsensusOK"].fillna(False)
        & merged["MarketSpreadConsistent"].fillna(False)
        & pd.to_numeric(merged["HCGapToMarketMedian"], errors="coerce").fillna(0.0).ge(0.07)
        & (~merged["ModelAvailable"] | merged["ModelAligned"])
        & pd.to_numeric(merged["TargetProb"], errors="coerce").notna()
    )
    merged["AutoEligible"] = auto_condition

    auto_new_preds: list[float] = []
    auto_apply: list[bool] = []
    auto_reasons: list[str] = []
    model_supported = []
    for row in merged.itertuples(index=False):
        reason = ""
        if not bool(row.RealGameScope):
            reason = "OutOfScope"
        elif not bool(row.MarketAvailable):
            reason = "NoMarket"
        elif not bool(row.MarketConsensusOK):
            reason = "LowMarketConsensus"
        elif not bool(row.MarketSpreadConsistent):
            reason = "MarketSpreadConflict"
        elif not pd.notna(row.HCGapToMarketMedian) or float(row.HCGapToMarketMedian) < 0.07:
            reason = "GapTooSmall"
        elif bool(row.ModelAvailable) and not bool(row.ModelAligned):
            reason = "ModelConflict"
        elif not pd.notna(row.TargetProb):
            reason = "MissingTarget"

        target = float(row.TargetProb) if pd.notna(row.TargetProb) else np.nan
        proposed = float(row.HCPred)
        if reason == "":
            proposed = apply_override_guardrails(
                float(row.HCPred),
                target,
                0.35,
                market_prob=row.MarketProbMedian,
                spread=row.SpreadMedian,
                model_prob=row.ModelMatchupProbMedian,
            )
            if bool(row.FavoriteHasInjuryVeto) and favorite_direction_change(float(row.HCPred), proposed, str(row.FavoriteSide)):
                reason = "FavoriteInjuryVeto"

        apply_flag = reason == "" and abs(proposed - float(row.HCPred)) > 1e-12
        auto_new_preds.append(proposed)
        auto_apply.append(apply_flag)
        auto_reasons.append(reason or "AutoApply")
        model_supported.append(bool(row.ModelAvailable) and bool(row.ModelAligned))

    merged["AutoNewPred"] = auto_new_preds
    merged["AutoChangeAbs"] = (pd.to_numeric(merged["AutoNewPred"], errors="coerce") - pd.to_numeric(merged["HCPred"], errors="coerce")).abs()
    merged["AutoApply"] = auto_apply
    merged["AutoDecision"] = auto_reasons
    merged["AutoModelSupported"] = model_supported

    manual_condition = (
        merged["RealGameScope"].fillna(False)
        & merged["MarketAvailable"].fillna(False)
        & merged["MarketSpreadConsistent"].fillna(False)
        & pd.to_numeric(merged["HCGapToMarketMedian"], errors="coerce").fillna(0.0).ge(0.08)
        & (merged["BaselineCloser"].fillna(False) | merged["ModelAligned"].fillna(False))
        & (~merged["FavoriteHasInjuryVeto"].fillna(False))
    )
    merged["ManualEligible"] = manual_condition

    manual = merged.loc[merged["ManualEligible"]].copy()
    manual["StagePriority"] = pd.to_numeric(manual.get("OfficialMinRound"), errors="coerce").fillna(9.0)
    manual["ManualReviewReason"] = np.where(
        manual["BaselineCloser"].fillna(False) & manual["ModelAligned"].fillna(False),
        "BaselineAndModelAgainstHC",
        np.where(manual["BaselineCloser"].fillna(False), "BaselineCloserToMarket", "ModelAlignedAgainstHC"),
    )
    manual = manual.sort_values(
        ["StagePriority", "HCGapToMarketMedian", "MarketSourceCount", "BookCountMax"],
        ascending=[True, False, False, False],
    ).head(10).copy()
    manual["ManualApply"] = ""
    manual["ManualTargetProb"] = ""
    manual["ManualNote"] = ""

    eligible = merged.loc[merged["RealGameScope"].fillna(False)].copy()
    eligible = eligible.sort_values(
        ["AutoApply", "OfficialMinRound", "HCGapToMarketMedian", "MarketSourceCount"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)

    summary = {
        "current": str(Path(args.current).resolve()),
        "baseline": str(Path(args.baseline).resolve()),
        "season": int(args.season),
        "rows_total": int(len(merged)),
        "rows_in_scope": int(merged["RealGameScope"].fillna(False).sum()),
        "auto_eligible_rows": int(merged["AutoEligible"].fillna(False).sum()),
        "auto_applied_rows": int(merged["AutoApply"].fillna(False).sum()),
        "auto_applied_playin_rows": int((merged["AutoApply"].fillna(False) & merged["IsPlayIn"].fillna(False)).sum()),
        "auto_applied_round1_rows": int((merged["AutoApply"].fillna(False) & merged["IsRound1"].fillna(False)).sum()),
        "auto_applied_round2_rows": int((merged["AutoApply"].fillna(False) & merged["IsRound2"].fillna(False)).sum()),
        "auto_model_supported_rows": int((merged["AutoApply"].fillna(False) & merged["AutoModelSupported"].fillna(False)).sum()),
        "manual_eligible_rows": int(merged["ManualEligible"].fillna(False).sum()),
        "manual_shortlist_rows": int(len(manual)),
        "top_auto_applied": records_for_json(eligible.loc[eligible["AutoApply"].fillna(False)], 12),
        "manual_shortlist_top10": records_for_json(manual, 10),
        "csv_output": str(Path(args.csv_output).resolve()),
        "manual_csv_output": str(Path(args.manual_csv_output).resolve()),
    }
    return eligible, manual, summary


def main() -> None:
    args = parse_args()
    eligible, manual, summary = build_candidates(args)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.csv_output, index=False)
    manual.to_csv(args.manual_csv_output, index=False)
    Path(args.json_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.manual_json_output).write_text(
        json.dumps(
            {
                "season": int(args.season),
                "rows": int(len(manual)),
                "top10": records_for_json(manual, 10),
                "csv_output": str(Path(args.manual_csv_output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"goldshot candidates written to: {args.csv_output}")
    print(f"goldshot summary written to: {args.json_output}")
    print(f"manual shortlist written to: {args.manual_csv_output}")
    print(f"manual shortlist summary written to: {args.manual_json_output}")


if __name__ == "__main__":
    main()
