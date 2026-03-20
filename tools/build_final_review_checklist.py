from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import date, datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc.predict import build_hc_prediction_frame, load_live_market_frame, load_live_model_matchup_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final manual review checklist focused on market closeness, high-diff games, and injury risk."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hc", required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--top-n", type=int, default=75)
    parser.add_argument("--shortlist-n", type=int, default=20)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--shortlist-csv-output")
    parser.add_argument("--shortlist-json-output")
    return parser.parse_args()


def load_team_maps(data_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    men = pd.read_csv(data_dir / "MTeams.csv", usecols=["TeamID", "TeamName"])
    women = pd.read_csv(data_dir / "WTeams.csv", usecols=["TeamID", "TeamName"])
    men_map = dict(zip(men["TeamID"].astype(int), men["TeamName"].astype(str)))
    women_map = dict(zip(women["TeamID"].astype(int), women["TeamName"].astype(str)))
    return men_map, women_map


def load_seed_map(data_dir: Path, gender: str, season: int) -> dict[int, str]:
    path = data_dir / f"{gender}NCAATourneySeeds.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["Season", "Seed", "TeamID"])
    frame = frame.loc[pd.to_numeric(frame["Season"], errors="coerce").eq(season)].copy()
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["TeamID"])
    return dict(zip(frame["TeamID"].astype(int), frame["Seed"].astype(str)))


def load_men_watchlist(results_dir: Path, season: int) -> pd.DataFrame:
    path = results_dir / f"availability_watchlist_{season}_men.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["TeamID"]).copy()
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def build_women_host_ids(season: int) -> set[str]:
    frame = build_hc_prediction_frame("W", season, 32, False, True)
    if frame.empty or not {"ID", "D_HostLikely", "IsRound1Or2"}.issubset(frame.columns):
        return set()
    host_mask = (
        pd.to_numeric(frame["D_HostLikely"], errors="coerce").fillna(0.0).ge(1.0)
        & pd.to_numeric(frame["IsRound1Or2"], errors="coerce").fillna(0.0).ge(1.0)
    )
    return set(frame.loc[host_mask, "ID"].astype(str).tolist())


def load_market_review_frame(season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gender in ("M", "W"):
        frame = load_live_market_frame(gender, season)
        if frame.empty:
            continue
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
        present = [column for column in keep if column in frame.columns]
        frames.append(frame[present].copy())
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["SnapshotTime"] = pd.to_datetime(merged.get("SnapshotTime"), errors="coerce", utc=True)
    return merged.drop_duplicates(subset=["Season", "T1", "T2"], keep="last")


def load_model_review_frame(season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for gender in ("M", "W"):
        frame = load_live_model_matchup_frame(gender, season)
        if frame.empty:
            continue
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
        present = [column for column in keep if column in frame.columns]
        frames.append(frame[present].copy())
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["SnapshotTime"] = pd.to_datetime(merged.get("SnapshotTime"), errors="coerce", utc=True)
    return merged.drop_duplicates(subset=["Season", "T1", "T2"], keep="last")


def classify_market_closeness(frame: pd.DataFrame, tolerance: float = 0.03) -> pd.DataFrame:
    out = frame.copy()
    market_median = pd.to_numeric(out.get("MarketProbMedian"), errors="coerce")
    hc_gap = (pd.to_numeric(out["HCPred"], errors="coerce") - market_median).abs()
    baseline_gap = (pd.to_numeric(out["BaselinePred"], errors="coerce") - market_median).abs()
    out["HCGapToMarketMedian"] = hc_gap
    out["BaselineGapToMarketMedian"] = baseline_gap
    out["EffectiveBookCount"] = pd.concat(
        [
            pd.to_numeric(out.get("BookCountMax"), errors="coerce"),
            pd.to_numeric(out.get("BookCountTotal"), errors="coerce"),
            pd.to_numeric(out.get("MarketRowCount"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    out["CloserToMarket"] = "NoMarket"
    has_market = market_median.notna()
    out.loc[has_market, "CloserToMarket"] = "Tie"
    out.loc[has_market & (hc_gap + tolerance < baseline_gap), "CloserToMarket"] = "HC"
    out.loc[has_market & (baseline_gap + tolerance < hc_gap), "CloserToMarket"] = "Baseline"
    return out


def assign_review_priority(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    abs_diff = pd.to_numeric(out["AbsPredDiff"], errors="coerce").fillna(0.0)
    market_sources = pd.to_numeric(out.get("MarketSourceCount"), errors="coerce").fillna(0.0)
    book_count = pd.to_numeric(out.get("EffectiveBookCount"), errors="coerce").fillna(0.0)
    both_in_field = out["BothInField"].fillna(False)
    women_early_host = out["WomenEarlyHost"].fillna(False)
    men_injury_risk = out["MenInjuryRisk"].fillna(False)
    closer = out["CloserToMarket"].fillna("NoMarket")

    out["ReviewEligible"] = both_in_field | women_early_host | men_injury_risk
    out["ReviewReason"] = ""
    out.loc[both_in_field, "ReviewReason"] = "BothInField"
    out.loc[women_early_host & out["ReviewReason"].eq(""), "ReviewReason"] = "WomenEarlyHost"
    out.loc[women_early_host & out["ReviewReason"].ne("") & ~out["ReviewReason"].str.contains("WomenEarlyHost"), "ReviewReason"] = (
        out.loc[women_early_host & out["ReviewReason"].ne("") & ~out["ReviewReason"].str.contains("WomenEarlyHost"), "ReviewReason"]
        + "|WomenEarlyHost"
    )
    out.loc[men_injury_risk & out["ReviewReason"].eq(""), "ReviewReason"] = "MenInjuryRisk"
    out.loc[men_injury_risk & out["ReviewReason"].ne("") & ~out["ReviewReason"].str.contains("MenInjuryRisk"), "ReviewReason"] = (
        out.loc[men_injury_risk & out["ReviewReason"].ne("") & ~out["ReviewReason"].str.contains("MenInjuryRisk"), "ReviewReason"]
        + "|MenInjuryRisk"
    )

    out["RecommendedAction"] = "KeepHC"
    out.loc[closer.eq("Baseline"), "RecommendedAction"] = "CheckBaseline"
    out.loc[closer.isin(["Tie", "NoMarket"]), "RecommendedAction"] = "NeedContext"

    out["ManualReviewPriority"] = "Skip"
    critical_mask = (
        out["ReviewEligible"]
        & both_in_field
        & closer.eq("Baseline")
        & (abs_diff >= 0.08)
        & (market_sources >= 2.0)
        & (book_count >= 2.0)
    )
    high_mask = (
        out["ReviewEligible"]
        & ~critical_mask
        & (
            (both_in_field & (abs_diff >= 0.05) & (market_sources >= 1.0))
            | (men_injury_risk & (abs_diff >= 0.05))
            | (women_early_host & (abs_diff >= 0.03) & (market_sources >= 1.0))
        )
    )
    medium_mask = out["ReviewEligible"] & ~critical_mask & ~high_mask & (abs_diff >= 0.03)
    low_mask = out["ReviewEligible"] & ~critical_mask & ~high_mask & ~medium_mask

    out.loc[low_mask, "ManualReviewPriority"] = "Low"
    out.loc[medium_mask, "ManualReviewPriority"] = "Medium"
    out.loc[high_mask, "ManualReviewPriority"] = "High"
    out.loc[critical_mask, "ManualReviewPriority"] = "Critical"
    return out


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
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_submissions(args: argparse.Namespace) -> pd.DataFrame:
    baseline = pd.read_csv(args.baseline, usecols=["ID", "Pred"]).rename(columns={"Pred": "BaselinePred"})
    hc = pd.read_csv(args.hc, usecols=["ID", "Pred"]).rename(columns={"Pred": "HCPred"})
    merged = baseline.merge(hc, on="ID", how="inner")
    merged["PredDiff"] = merged["HCPred"] - merged["BaselinePred"]
    merged["AbsPredDiff"] = merged["PredDiff"].abs()
    ids = merged["ID"].astype(str).str.split("_", expand=True)
    merged["Season"] = pd.to_numeric(ids[0], errors="coerce").astype("Int64")
    merged["T1"] = pd.to_numeric(ids[1], errors="coerce").astype("Int64")
    merged["T2"] = pd.to_numeric(ids[2], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["Season", "T1", "T2"]).copy()
    merged["Season"] = merged["Season"].astype(int)
    merged["T1"] = merged["T1"].astype(int)
    merged["T2"] = merged["T2"].astype(int)
    return merged


def attach_team_context(merged: pd.DataFrame, args: argparse.Namespace, data_dir: Path, results_dir: Path) -> pd.DataFrame:
    men_map, women_map = load_team_maps(data_dir)
    men_seed = load_seed_map(data_dir, "M", args.season)
    women_seed = load_seed_map(data_dir, "W", args.season)
    watchlist = load_men_watchlist(results_dir, args.season)
    women_host_ids = build_women_host_ids(args.season)

    def infer_gender(t1: int, t2: int) -> str:
        if t1 in men_map and t2 in men_map:
            return "M"
        if t1 in women_map and t2 in women_map:
            return "W"
        return "?"

    out = merged.copy()
    out["Gender"] = [infer_gender(t1, t2) for t1, t2 in zip(out["T1"], out["T2"])]
    out["Team1Name"] = out.apply(
        lambda row: men_map.get(row["T1"]) if row["Gender"] == "M" else women_map.get(row["T1"], ""),
        axis=1,
    )
    out["Team2Name"] = out.apply(
        lambda row: men_map.get(row["T2"]) if row["Gender"] == "M" else women_map.get(row["T2"], ""),
        axis=1,
    )
    out["Team1Seed"] = out.apply(
        lambda row: men_seed.get(row["T1"], "") if row["Gender"] == "M" else women_seed.get(row["T1"], ""),
        axis=1,
    )
    out["Team2Seed"] = out.apply(
        lambda row: men_seed.get(row["T2"], "") if row["Gender"] == "M" else women_seed.get(row["T2"], ""),
        axis=1,
    )
    out["WomenEarlyHost"] = out["ID"].astype(str).isin(women_host_ids)
    out["Team1InField"] = out["Team1Seed"].fillna("").astype(str).ne("")
    out["Team2InField"] = out["Team2Seed"].fillna("").astype(str).ne("")
    out["BothInField"] = out["Team1InField"] & out["Team2InField"]

    if not watchlist.empty:
        team1_watch = watchlist.add_prefix("Team1_").rename(columns={"Team1_TeamID": "T1"})
        team2_watch = watchlist.add_prefix("Team2_").rename(columns={"Team2_TeamID": "T2"})
        out = out.merge(team1_watch, on="T1", how="left")
        out = out.merge(team2_watch, on="T2", how="left")
    else:
        for prefix in ("Team1_", "Team2_"):
            for column in ["Season", "Seed", "TeamName", "SeverityScore", "OutCount", "GTDCount", "PlayerCount", "Players", "Statuses", "Injuries"]:
                out[f"{prefix}{column}"] = pd.NA

    out["MenInjuryRisk"] = (
        out["Gender"].eq("M")
        & (
            pd.to_numeric(out["Team1_SeverityScore"], errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(out["Team2_SeverityScore"], errors="coerce").fillna(0).gt(0)
        )
    )
    out["MaxSeverityScore"] = pd.concat(
        [
            pd.to_numeric(out["Team1_SeverityScore"], errors="coerce"),
            pd.to_numeric(out["Team2_SeverityScore"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return out


def build_outputs(merged: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_review = load_market_review_frame(args.season)
    model_review = load_model_review_frame(args.season)

    out = merged.copy()
    if not market_review.empty:
        out = out.merge(market_review, on=["Season", "T1", "T2"], how="left")
    if not model_review.empty:
        out = out.merge(model_review, on=["Season", "T1", "T2"], how="left")
    out = classify_market_closeness(out)
    out = assign_review_priority(out)

    ordered = out.sort_values(
        ["AbsPredDiff", "MaxSeverityScore", "MarketSourceCount"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    shortlist = ordered.loc[ordered["ReviewEligible"]].copy()
    priority_order = pd.CategoricalDtype(["Critical", "High", "Medium", "Low", "Skip"], ordered=True)
    shortlist["ManualReviewPriority"] = shortlist["ManualReviewPriority"].astype(priority_order)
    shortlist = shortlist.sort_values(
        ["ManualReviewPriority", "AbsPredDiff", "MarketSourceCount", "MaxSeverityScore"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    shortlist = shortlist.head(args.shortlist_n).copy()
    return ordered, shortlist


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "ncaa-data"
    results_dir = ROOT / "results"

    merged = load_submissions(args)
    merged = attach_team_context(merged, args, data_dir, results_dir)
    ordered, shortlist = build_outputs(merged, args)

    top = ordered.head(args.top_n).copy()
    top.to_csv(args.csv_output, index=False)

    shortlist_csv_output = args.shortlist_csv_output or str(Path(args.csv_output).with_name(f"{Path(args.csv_output).stem}_shortlist.csv"))
    shortlist_json_output = args.shortlist_json_output or str(Path(args.json_output).with_name(f"{Path(args.json_output).stem}_shortlist.json"))
    shortlist.to_csv(shortlist_csv_output, index=False)

    summary = {
        "baseline": str(Path(args.baseline).resolve()),
        "hc": str(Path(args.hc).resolve()),
        "season": int(args.season),
        "rows_compared": int(len(ordered)),
        "top_n": int(args.top_n),
        "shortlist_n": int(args.shortlist_n),
        "mean_abs_diff": float(ordered["AbsPredDiff"].mean()),
        "max_abs_diff": float(ordered["AbsPredDiff"].max()),
        "men_injury_risk_rows_in_top_n": int(top["MenInjuryRisk"].fillna(False).sum()),
        "women_early_host_rows_in_top_n": int(top["WomenEarlyHost"].fillna(False).sum()),
        "both_in_field_rows_in_top_n": int(top["BothInField"].fillna(False).sum()),
        "eligible_rows_total": int(ordered["ReviewEligible"].fillna(False).sum()),
        "priority_counts": {
            key: int(value)
            for key, value in ordered["ManualReviewPriority"].fillna("Skip").value_counts().to_dict().items()
        },
        "top10": records_for_json(top, 10),
        "shortlist_top10": records_for_json(shortlist, 10),
        "csv_output": str(Path(args.csv_output).resolve()),
        "shortlist_csv_output": str(Path(shortlist_csv_output).resolve()),
    }
    Path(args.json_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    shortlist_summary = {
        "season": int(args.season),
        "rows": int(len(shortlist)),
        "top10": records_for_json(shortlist, 10),
        "csv_output": str(Path(shortlist_csv_output).resolve()),
    }
    Path(shortlist_json_output).write_text(json.dumps(shortlist_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"final review checklist written to: {args.csv_output}")
    print(f"summary written to: {args.json_output}")
    print(f"shortlist written to: {shortlist_csv_output}")
    print(f"shortlist summary written to: {shortlist_json_output}")


if __name__ == "__main__":
    main()
