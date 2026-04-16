from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import numpy as np
import pandas as pd

from .data import EXTERNAL_DATA, load_historical_sportsbook_probs, load_prediction_market_probs
from .market_blend import apply_market_experiment, clip_probs

HIGH_QUALITY_OVERLAY_SOURCES = {
    "sportsbook",
    "barttorvik",
    "warrennolan",
    "silverbulletin",
    "herhoopstats",
}
OVERLAY_SOURCE_PRIORITY = {
    "sportsbook": 0,
    "barttorvik": 1,
    "herhoopstats": 1,
    "warrennolan": 2,
    "silverbulletin": 2,
    "prediction_market": 3,
    "projection": 4,
}


def _canonicalize_overlay_source(
    frame: pd.DataFrame,
    *,
    season: int,
    prob_col: str,
    team1_col: str,
    team2_col: str,
) -> pd.DataFrame:
    required = {"Season", prob_col, team1_col, team2_col}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["Season", "T1", "T2", "overlay_prob"])

    base = frame.copy()
    base["Season"] = pd.to_numeric(base["Season"], errors="coerce")
    base[team1_col] = pd.to_numeric(base[team1_col], errors="coerce")
    base[team2_col] = pd.to_numeric(base[team2_col], errors="coerce")
    base[prob_col] = pd.to_numeric(base[prob_col], errors="coerce")
    usable = base.loc[
        (base["Season"] == int(season))
        & base[team1_col].notna()
        & base[team2_col].notna()
        & base[prob_col].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "overlay_prob"])

    t1 = usable[team1_col].astype(int)
    t2 = usable[team2_col].astype(int)
    swap = t1 > t2
    usable["T1"] = np.where(swap, t2, t1)
    usable["T2"] = np.where(swap, t1, t2)
    usable["overlay_prob"] = np.where(swap, 1.0 - usable[prob_col], usable[prob_col])
    return usable[["Season", "T1", "T2", "overlay_prob"]].drop_duplicates(["Season", "T1", "T2"], keep="last")


def _load_optional_matchup_projection(path: PathLike[str] | str, season: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {"Team1ID", "Team2ID", "ModelProb"}.issubset(frame.columns):
        return _canonicalize_overlay_source(
            frame,
            season=season,
            prob_col="ModelProb",
            team1_col="Team1ID",
            team2_col="Team2ID",
        )
    return pd.DataFrame(columns=["Season", "T1", "T2", "overlay_prob"])


def load_current_year_overlay_probs(gender: str, season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    sportsbook = load_historical_sportsbook_probs(gender)
    if not sportsbook.empty:
        frames.append(
            sportsbook.loc[sportsbook["Season"] == int(season), ["Season", "T1", "T2", "sportsbook_prob"]]
            .rename(columns={"sportsbook_prob": "overlay_prob"})
            .assign(overlay_source="sportsbook")
        )

    prediction_market = load_prediction_market_probs(gender)
    if not prediction_market.empty:
        frames.append(
            prediction_market.loc[
                prediction_market["Season"] == int(season), ["Season", "T1", "T2", "prediction_market_prob"]
            ].rename(columns={"prediction_market_prob": "overlay_prob"})
            .assign(overlay_source="prediction_market")
        )

    projection_sources = [
        ("barttorvik", EXTERNAL_DATA / f"{gender}BartTorvikMatchupProjections_{season}.csv"),
        ("warrennolan", EXTERNAL_DATA / f"{gender}WarrenNolanMatchupProjections_{season}.csv"),
        ("silverbulletin", EXTERNAL_DATA / f"{gender}SilverBulletinMatchupProjections_{season}.csv"),
        ("herhoopstats", EXTERNAL_DATA / f"{gender}HerHoopStatsMatchupProjections_{season}.csv"),
    ]
    for source_name, path in projection_sources:
        if path.exists():
            projected = _load_optional_matchup_projection(path, season)
            if not projected.empty:
                frames.append(projected.assign(overlay_source=source_name))

    if not frames:
        return pd.DataFrame(columns=["Season", "T1", "T2", "overlay_market_prob", "overlay_source"])

    combined = pd.concat(frames, ignore_index=True)
    combined["overlay_source"] = combined["overlay_source"].fillna("projection")
    combined["source_priority"] = combined["overlay_source"].map(OVERLAY_SOURCE_PRIORITY).fillna(99).astype(int)

    rows: list[dict] = []
    for (season_value, t1, t2), group in combined.groupby(["Season", "T1", "T2"], sort=False):
        best_priority = int(group["source_priority"].min())
        best = group.loc[group["source_priority"] == best_priority].copy()
        rows.append(
            {
                "Season": int(season_value),
                "T1": int(t1),
                "T2": int(t2),
                "overlay_market_prob": float(best["overlay_prob"].mean()),
                "overlay_source": str(best.iloc[0]["overlay_source"]),
            }
        )
    return pd.DataFrame(rows)


def load_current_year_injury_scores(gender: str, season: int) -> pd.DataFrame:
    path = EXTERNAL_DATA / f"{gender}RotoWireInjuries_{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "injury_score"])

    frame = pd.read_csv(path)
    if frame.empty or "TeamID" not in frame.columns:
        return pd.DataFrame(columns=["Season", "TeamID", "injury_score"])

    usable = frame.copy()
    usable["Season"] = pd.to_numeric(usable.get("Season"), errors="coerce")
    usable["TeamID"] = pd.to_numeric(usable.get("TeamID"), errors="coerce")
    usable["IsOut"] = pd.to_numeric(usable.get("IsOut"), errors="coerce").fillna(0.0)
    usable["IsGameTimeDecision"] = pd.to_numeric(usable.get("IsGameTimeDecision"), errors="coerce").fillna(0.0)
    usable["Severity"] = pd.to_numeric(usable.get("Severity"), errors="coerce").fillna(0.0)
    usable = usable.loc[(usable["Season"] == int(season)) & usable["TeamID"].notna()].copy()
    if usable.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "injury_score", "confirmed_out_score"])

    usable["injury_score"] = usable["Severity"] + usable["IsOut"] + (0.5 * usable["IsGameTimeDecision"])
    usable["confirmed_out_score"] = np.where(
        (usable["IsOut"] >= 1.0) & (usable["Severity"] >= 2.0),
        usable["Severity"],
        0.0,
    )
    return usable.groupby(["Season", "TeamID"], as_index=False)[["injury_score", "confirmed_out_score"]].sum()


def _apply_injury_shift(probabilities: pd.Series, injury_delta: pd.Series, *, scale: float, cap: float) -> pd.Series:
    clipped_base = clip_probs(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped_base / (1.0 - clipped_base))
    bounded_shift = injury_delta.clip(lower=-cap, upper=cap) * scale
    shifted = 1.0 / (1.0 + np.exp(-(logit + bounded_shift)))
    return pd.Series(shifted, index=probabilities.index)


@dataclass(slots=True)
class OverlaySummary:
    season: int
    rows: int
    market_rows: int
    injury_rows: int
    total_changed_rows: int
    mean_abs_delta: float
    max_abs_delta: float
    mean_abs_delta_m: float | None = None
    mean_abs_delta_w: float | None = None
    guardrail_passed: bool = False


def apply_current_year_overlay(
    predictions: pd.DataFrame,
    *,
    gender: str,
    season: int,
    market_weight: float = 0.2,
    bounded_pull_delta: float = 0.02,
    injury_logit_scale: float = 0.10,
    injury_cap: float = 1.5,
    clip_low: float = 0.01,
    clip_high: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, OverlaySummary]:
    base = predictions.copy()
    overlay_probs = load_current_year_overlay_probs(gender, season)
    injury_scores = load_current_year_injury_scores(gender, season)

    merged = base.merge(overlay_probs, on=["Season", "T1", "T2"], how="left")
    if not injury_scores.empty:
        t1_injury = injury_scores.rename(
            columns={
                "TeamID": "T1",
                "injury_score": "T1_injury_score",
                "confirmed_out_score": "T1_confirmed_out_score",
            }
        )
        t2_injury = injury_scores.rename(
            columns={
                "TeamID": "T2",
                "injury_score": "T2_injury_score",
                "confirmed_out_score": "T2_confirmed_out_score",
            }
        )
        merged = merged.merge(t1_injury, on=["Season", "T1"], how="left").merge(t2_injury, on=["Season", "T2"], how="left")
    else:
        merged["T1_injury_score"] = np.nan
        merged["T2_injury_score"] = np.nan
        merged["T1_confirmed_out_score"] = np.nan
        merged["T2_confirmed_out_score"] = np.nan
    if "T1_confirmed_out_score" not in merged.columns:
        merged["T1_confirmed_out_score"] = np.nan
    if "T2_confirmed_out_score" not in merged.columns:
        merged["T2_confirmed_out_score"] = np.nan

    merged["BaseProb"] = pd.to_numeric(merged["Pred"], errors="coerce")
    market_allowed = merged["overlay_source"].isin(HIGH_QUALITY_OVERLAY_SOURCES)
    merged["OverlayProb"] = merged["BaseProb"]
    if market_allowed.any():
        merged.loc[market_allowed, "OverlayProb"] = apply_market_experiment(
            merged.loc[market_allowed, "BaseProb"],
            merged.loc[market_allowed, "overlay_market_prob"],
            weight=market_weight,
            max_delta=bounded_pull_delta,
            clip_low=clip_low,
            clip_high=clip_high,
        )

    confirmed_delta = merged["T2_confirmed_out_score"].fillna(0.0) - merged["T1_confirmed_out_score"].fillna(0.0)
    injury_available = (
        (gender == "M")
        and market_allowed
        & ((merged["T1_confirmed_out_score"].fillna(0.0) > 0.0) | (merged["T2_confirmed_out_score"].fillna(0.0) > 0.0))
    )
    adjusted = merged["OverlayProb"].copy()
    if isinstance(injury_available, pd.Series) and injury_available.any():
        adjusted.loc[injury_available] = _apply_injury_shift(
            adjusted.loc[injury_available],
            confirmed_delta.loc[injury_available],
            scale=injury_logit_scale,
            cap=injury_cap,
        )

    merged["Pred"] = clip_probs(adjusted, clip_low, clip_high)
    merged["AbsDelta"] = (merged["Pred"] - merged["BaseProb"]).abs()
    merged["overlay_source"] = merged["overlay_source"].where(market_allowed, "none").fillna("none")
    merged["injury_applied"] = False
    if isinstance(injury_available, pd.Series):
        merged.loc[injury_available, "injury_applied"] = True
    guardrail_passed = bool(
        float(merged["AbsDelta"].mean()) <= 0.02
        and int((merged["AbsDelta"] > 1e-12).sum()) <= max(int(len(merged) * 0.2), 1)
    )
    summary = OverlaySummary(
        season=int(season),
        rows=int(len(merged)),
        market_rows=int(market_allowed.sum()),
        injury_rows=int(injury_available.sum()) if isinstance(injury_available, pd.Series) else 0,
        total_changed_rows=int((merged["AbsDelta"] > 1e-12).sum()),
        mean_abs_delta=float(merged["AbsDelta"].mean()),
        max_abs_delta=float(merged["AbsDelta"].max()),
        mean_abs_delta_m=float(merged["AbsDelta"].mean()) if gender == "M" else None,
        mean_abs_delta_w=float(merged["AbsDelta"].mean()) if gender == "W" else None,
        guardrail_passed=guardrail_passed,
    )
    audit = merged[
        [
            "ID",
            "BaseProb",
            "OverlayProb",
            "Pred",
            "overlay_source",
            "injury_applied",
            "AbsDelta",
        ]
    ].copy()
    return merged[["ID", "Pred"]], audit, summary
