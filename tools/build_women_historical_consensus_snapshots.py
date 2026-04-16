from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTERNAL_DIR = ROOT / "external-data"
RESULTS_DIR = ROOT / "results"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / name)


def _coerce_team_season(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce")
    out["TeamID"] = pd.to_numeric(out["TeamID"], errors="coerce")
    out = out.dropna(subset=["Season", "TeamID"]).copy()
    out["Season"] = out["Season"].astype(int)
    out["TeamID"] = out["TeamID"].astype(int)
    if "TeamName" not in out.columns:
        out["TeamName"] = ""
    out["TeamName"] = out["TeamName"].fillna("").astype(str)
    return out


def _normalize_snapshot_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")


def build_women_historical_consensus_snapshots() -> pd.DataFrame:
    hist = _coerce_team_season(_read_csv("WHistoricalTeamRatings.csv"))
    hist_pre = _coerce_team_season(_read_csv("WHistoricalTeamRatingsPreTourney.csv"))
    sb_hist = _coerce_team_season(_read_csv("WSilverBulletinTeamRatings_History.csv"))
    current = _coerce_team_season(_read_csv("WTeamRatings.csv"))

    hist_core = hist[
        [
            "Season",
            "TeamID",
            "TeamName",
            "FallbackElo",
            "FallbackSOS",
            "FallbackOWP",
            "FallbackRPIStyle",
            "FallbackRPIStyleSOS",
            "FallbackWinRate",
            "FallbackAvgMargin",
            "Games",
            "PublicAPRank",
            "PublicCoachesRank",
            "PublicNETRank",
            "PublicELORank",
            "PublicRPIRank",
            "PublicPredRPIRank",
        ]
    ].copy()
    hist_core["SnapshotType"] = "historical_panel"
    hist_core["SnapshotDate"] = pd.NA
    hist_core["VerifiedPreTourney"] = 0

    hist_pre_core = hist_pre[
        [
            "Season",
            "TeamID",
            "TeamName",
            "FallbackElo",
            "FallbackSOS",
            "FallbackOWP",
            "FallbackRPIStyle",
            "FallbackRPIStyleSOS",
            "FallbackWinRate",
            "FallbackAvgMargin",
            "Games",
            "PublicAPRank",
            "PublicCoachesRank",
            "PublicNETRank",
            "PublicELORank",
            "PublicRPIRank",
            "PublicPredRPIRank",
            "SnapshotDate",
            "VerifiedPreTourney",
        ]
    ].copy()
    hist_pre_core["SnapshotType"] = "verified_pre_tourney"
    hist_pre_core["SnapshotDate"] = _normalize_snapshot_date(hist_pre_core["SnapshotDate"])
    hist_pre_core["VerifiedPreTourney"] = pd.to_numeric(hist_pre_core["VerifiedPreTourney"], errors="coerce").fillna(0).astype(int)

    sb_core = sb_hist[
        [
            "Season",
            "TeamID",
            "TeamName",
            "SB_BXelo",
            "SB_BPPPG",
            "SB_BPPAG",
            "SB_BNetRating",
            "SB_LeagueScoring",
        ]
    ].copy()

    merged = hist_core.merge(
        hist_pre_core.drop(columns=["TeamName"], errors="ignore"),
        on=["Season", "TeamID"],
        how="outer",
        suffixes=("_hist", "_pre"),
    )

    merged["TeamName"] = merged["TeamName"].fillna("").astype(str)
    merged["SnapshotType"] = merged["SnapshotType_pre"].fillna(merged["SnapshotType_hist"]).fillna("historical_panel")
    merged["SnapshotDate"] = merged["SnapshotDate_pre"].fillna(merged["SnapshotDate_hist"])
    merged["VerifiedPreTourney"] = (
        pd.to_numeric(merged["VerifiedPreTourney_pre"], errors="coerce")
        .fillna(pd.to_numeric(merged["VerifiedPreTourney_hist"], errors="coerce"))
        .fillna(0)
        .astype(int)
    )

    for column in [
        "FallbackElo",
        "FallbackSOS",
        "FallbackOWP",
        "FallbackRPIStyle",
        "FallbackRPIStyleSOS",
        "FallbackWinRate",
        "FallbackAvgMargin",
        "Games",
        "PublicAPRank",
        "PublicCoachesRank",
        "PublicNETRank",
        "PublicELORank",
        "PublicRPIRank",
        "PublicPredRPIRank",
    ]:
        hist_col = f"{column}_hist"
        pre_col = f"{column}_pre"
        merged[column] = pd.to_numeric(merged.get(pre_col), errors="coerce").fillna(pd.to_numeric(merged.get(hist_col), errors="coerce"))

    merged = merged[
        [
            "Season",
            "TeamID",
            "TeamName",
            "SnapshotType",
            "SnapshotDate",
            "VerifiedPreTourney",
            "FallbackElo",
            "FallbackSOS",
            "FallbackOWP",
            "FallbackRPIStyle",
            "FallbackRPIStyleSOS",
            "FallbackWinRate",
            "FallbackAvgMargin",
            "Games",
            "PublicAPRank",
            "PublicCoachesRank",
            "PublicNETRank",
            "PublicELORank",
            "PublicRPIRank",
            "PublicPredRPIRank",
        ]
    ].copy()

    merged = merged.merge(sb_core, on=["Season", "TeamID", "TeamName"], how="outer")
    merged = merged.merge(
        current.rename(
            columns={
                "WN_NET": "CurrentWN_NET",
                "WN_ELO": "CurrentWN_ELO",
                "WN_RPI": "CurrentWN_RPI",
                "WN_PredRPI": "CurrentWN_PredRPI",
                "OfficialNETRank": "CurrentOfficialNETRank",
                "MarketTitleOdds": "CurrentMarketTitleOdds",
                "MarketTitleProb": "CurrentMarketTitleProb",
            }
        )[
            [
                "Season",
                "TeamID",
                "TeamName",
                "CurrentWN_NET",
                "CurrentWN_ELO",
                "CurrentWN_RPI",
                "CurrentWN_PredRPI",
                "CurrentOfficialNETRank",
                "CurrentMarketTitleOdds",
                "CurrentMarketTitleProb",
            ]
        ],
        on=["Season", "TeamID", "TeamName"],
        how="outer",
    )

    coverage_cols = [
        "PublicAPRank",
        "PublicCoachesRank",
        "PublicNETRank",
        "PublicELORank",
        "PublicRPIRank",
        "PublicPredRPIRank",
        "SB_BXelo",
        "SB_BNetRating",
        "CurrentWN_NET",
        "CurrentWN_ELO",
        "CurrentWN_RPI",
        "CurrentWN_PredRPI",
        "CurrentOfficialNETRank",
    ]
    merged["WomenConsensusSourceCount"] = merged[coverage_cols].notna().sum(axis=1)
    merged["HasVerifiedPreTourneySnapshot"] = merged["VerifiedPreTourney"].eq(1).astype(int)
    merged = merged.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    return merged


def build_summary(frame: pd.DataFrame) -> dict[str, object]:
    by_season = (
        frame.groupby("Season")
        .agg(
            teams=("TeamID", "nunique"),
            verified_pre_tourney=("HasVerifiedPreTourneySnapshot", "sum"),
            avg_source_count=("WomenConsensusSourceCount", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": int(len(frame)),
        "season_min": int(frame["Season"].min()),
        "season_max": int(frame["Season"].max()),
        "verified_pre_tourney_rows": int(frame["HasVerifiedPreTourneySnapshot"].sum()),
        "current_2026_rows": int(frame.loc[frame["Season"].eq(2026), "TeamID"].nunique()),
        "by_season": by_season.to_dict(orient="records"),
    }


def main() -> None:
    frame = build_women_historical_consensus_snapshots()
    output_path = EXTERNAL_DIR / "WHistoricalConsensusSnapshots.csv"
    frame.to_csv(output_path, index=False)
    summary = build_summary(frame)
    summary_path = RESULTS_DIR / "women_historical_consensus_snapshots_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    print(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
