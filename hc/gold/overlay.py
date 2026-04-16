from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA = ROOT / "external-data"

DIRECT_SOURCE_PRIORITY = {
    "vegas_direct": 0,
    "sportsbook": 0,
    "bpi_direct": 1,
    "barttorvik": 2,
    "warrennolan": 3,
}
TRUE_DIRECT_SOURCES = {"vegas_direct", "sportsbook"}
PROJECTION_SOURCES = {"bpi_direct", "barttorvik", "warrennolan"}


def _canonicalize(frame: pd.DataFrame, *, season: int, team1_col: str, team2_col: str, prob_col: str, source: str) -> pd.DataFrame:
    required = {team1_col, team2_col, prob_col}
    if "Season" not in frame.columns or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    usable = frame.copy()
    usable["Season"] = pd.to_numeric(usable["Season"], errors="coerce")
    usable[team1_col] = pd.to_numeric(usable[team1_col], errors="coerce")
    usable[team2_col] = pd.to_numeric(usable[team2_col], errors="coerce")
    usable[prob_col] = pd.to_numeric(usable[prob_col], errors="coerce")
    usable = usable.loc[
        (usable["Season"] == int(season))
        & usable[team1_col].notna()
        & usable[team2_col].notna()
        & usable[prob_col].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    t1 = usable[team1_col].astype(int)
    t2 = usable[team2_col].astype(int)
    swap = t1 > t2
    usable["T1"] = np.where(swap, t2, t1)
    usable["T2"] = np.where(swap, t1, t2)
    usable["market_prob"] = np.where(swap, 1.0 - usable[prob_col], usable[prob_col])
    usable["source_used"] = source
    return usable[["Season", "T1", "T2", "market_prob", "source_used"]].drop_duplicates(["Season", "T1", "T2"], keep="last")


def _load_direct_market_candidates(gender: str, season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    direct_sources = [
        (EXTERNAL_DATA / f"{gender}MatchupOdds_{season}.csv", "MarketProb", "T1", "T2", "sportsbook"),
        (EXTERNAL_DATA / f"{gender}BartTorvikMatchupProjections_{season}.csv", "ModelProb", "Team1ID", "Team2ID", "barttorvik"),
        (EXTERNAL_DATA / f"{gender}WarrenNolanMatchupProjections_{season}.csv", "ModelProb", "Team1ID", "Team2ID", "warrennolan"),
        (EXTERNAL_DATA / f"{gender}HerHoopStatsMatchupProjections_{season}.csv", "ModelProb", "Team1ID", "Team2ID", "bpi_direct"),
    ]
    for path, prob_col, team1_col, team2_col, source in direct_sources:
        if path.exists():
            frames.append(
                _canonicalize(
                    pd.read_csv(path),
                    season=season,
                    team1_col=team1_col,
                    team2_col=team2_col,
                    prob_col=prob_col,
                    source=source,
                )
            )
    if not frames:
        return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    return pd.concat(frames, ignore_index=True)


def load_direct_market_probs(gender: str, season: int) -> pd.DataFrame:
    combined = _load_direct_market_candidates(gender, season)
    if combined.empty:
        return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    combined["priority"] = combined["source_used"].map(DIRECT_SOURCE_PRIORITY).fillna(99).astype(int)
    rows = []
    for (season_value, t1, t2), group in combined.groupby(["Season", "T1", "T2"], sort=False):
        best_priority = int(group["priority"].min())
        best = group.loc[group["priority"] == best_priority]
        rows.append(
            {
                "Season": int(season_value),
                "T1": int(t1),
                "T2": int(t2),
                "market_prob": float(best["market_prob"].mean()),
                "source_used": str(best.iloc[0]["source_used"]),
            }
        )
    return pd.DataFrame(rows)


def load_futures_pairwise_probs(gender: str, season: int) -> pd.DataFrame:
    path = EXTERNAL_DATA / f"{gender}KalshiPredictionMarketOdds_{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    frame = pd.read_csv(path)
    return _canonicalize(frame, season=season, team1_col="T1", team2_col="T2", prob_col="MarketProb", source="kalshi_futures")


def load_men_injury_adjustments(season: int) -> pd.DataFrame:
    path = EXTERNAL_DATA / f"MRotoWireInjuries_{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out"])
    frame = pd.read_csv(path)
    if frame.empty or "TeamID" not in frame.columns:
        return pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out"])
    usable = frame.copy()
    usable["Season"] = pd.to_numeric(usable.get("Season"), errors="coerce")
    usable["TeamID"] = pd.to_numeric(usable.get("TeamID"), errors="coerce")
    usable["Severity"] = pd.to_numeric(usable.get("Severity"), errors="coerce").fillna(0.0)
    usable["IsOut"] = pd.to_numeric(usable.get("IsOut"), errors="coerce").fillna(0.0)
    usable = usable.loc[(usable["Season"] == int(season)) & usable["TeamID"].notna()].copy()
    usable["confirmed_out"] = ((usable["IsOut"] >= 1.0) & (usable["Severity"] >= 2.0)).astype(int)
    usable["injury_shift"] = np.where(usable["confirmed_out"] == 1, -0.0075 * usable["Severity"], 0.0)
    return usable.groupby(["Season", "TeamID"], as_index=False)[["injury_shift", "confirmed_out"]].sum()


def _logit(prob: pd.Series) -> pd.Series:
    clipped = prob.clip(1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _inv_logit(logit: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-logit))


def apply_submission_overlay(
    predictions: pd.DataFrame,
    *,
    gender: str,
    season: int,
    overlay_source_profile: str = "current_default",
    direct_weight: float = 0.85,
    max_delta: float = 0.025,
    injury_cap: float = 0.02,
    include_futures: bool = False,
    allow_injury: bool = True,
    allow_sharpen: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = predictions.copy()
    if "Season" not in base.columns or "T1" not in base.columns or "T2" not in base.columns:
        ids = base["ID"].astype(str).str.split("_", expand=True)
        if ids.shape[1] >= 3:
            base["Season"] = pd.to_numeric(ids[0], errors="coerce")
            base["T1"] = pd.to_numeric(ids[1], errors="coerce")
            base["T2"] = pd.to_numeric(ids[2], errors="coerce")
    base["pre_prob"] = pd.to_numeric(base["Pred"], errors="coerce")

    use_futures = include_futures and overlay_source_profile in {"current_default", "b_tier_with_futures", "c_all_sources"}
    allow_injury_effective = bool(allow_injury and overlay_source_profile in {"current_default", "a_tier_default", "b_tier_with_futures", "c_all_sources"})
    allow_sharpen_effective = bool(allow_sharpen and overlay_source_profile in {"current_default", "a_tier_default", "b_tier_with_futures", "c_all_sources"})

    if overlay_source_profile == "direct_only":
        direct = _load_direct_market_candidates(gender, season)
        if not direct.empty:
            direct = direct.loc[direct["source_used"].isin(TRUE_DIRECT_SOURCES)].copy()
            if not direct.empty:
                direct = direct.groupby(["Season", "T1", "T2"], as_index=False).agg(
                    market_prob=("market_prob", "mean"),
                    source_used=("source_used", "first"),
                )
    elif overlay_source_profile == "direct_priority":
        direct = _load_direct_market_candidates(gender, season)
        if not direct.empty:
            rows = []
            for (season_value, t1, t2), group in direct.groupby(["Season", "T1", "T2"], sort=False):
                true_direct = group.loc[group["source_used"].isin(TRUE_DIRECT_SOURCES)]
                if not true_direct.empty:
                    chosen = true_direct
                else:
                    chosen = group.loc[group["source_used"].isin(PROJECTION_SOURCES)]
                if chosen.empty:
                    continue
                rows.append(
                    {
                        "Season": int(season_value),
                        "T1": int(t1),
                        "T2": int(t2),
                        "market_prob": float(chosen["market_prob"].mean()),
                        "source_used": str(chosen.iloc[0]["source_used"]),
                    }
                )
            direct = pd.DataFrame(rows, columns=["Season", "T1", "T2", "market_prob", "source_used"])
    else:
        direct = load_direct_market_probs(gender, season)
    futures = load_futures_pairwise_probs(gender, season) if use_futures else pd.DataFrame(
        columns=["Season", "T1", "T2", "market_prob", "source_used"]
    )
    injuries = (
        load_men_injury_adjustments(season)
        if gender == "M" and allow_injury_effective
        else pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out"])
    )

    merged = base.merge(direct.rename(columns={"market_prob": "direct_prob", "source_used": "direct_source"}), on=["Season", "T1", "T2"], how="left")
    merged = merged.merge(futures.rename(columns={"market_prob": "futures_prob", "source_used": "futures_source"}), on=["Season", "T1", "T2"], how="left")

    if not injuries.empty:
        t1 = injuries.rename(columns={"TeamID": "T1", "injury_shift": "t1_shift", "confirmed_out": "t1_out"})
        t2 = injuries.rename(columns={"TeamID": "T2", "injury_shift": "t2_shift", "confirmed_out": "t2_out"})
        merged = merged.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")
    for column in ["t1_shift", "t2_shift", "t1_out", "t2_out"]:
        if column not in merged.columns:
            merged[column] = 0.0
    merged[["t1_shift", "t2_shift", "t1_out", "t2_out"]] = merged[["t1_shift", "t2_shift", "t1_out", "t2_out"]].fillna(0.0)

    base_prob = merged["pre_prob"].clip(0.001, 0.999)
    blended = base_prob.copy()
    source_used = pd.Series("none", index=merged.index, dtype=object)
    market_applied = pd.Series(False, index=merged.index)
    sharpen_applied = pd.Series(False, index=merged.index)

    direct_mask = merged["direct_prob"].notna()
    if direct_mask.any():
        delta = (pd.to_numeric(merged.loc[direct_mask, "direct_prob"], errors="coerce") - base_prob.loc[direct_mask]).clip(-max_delta, max_delta)
        blended.loc[direct_mask] = (base_prob.loc[direct_mask] + direct_weight * delta).clip(0.001, 0.999)
        source_used.loc[direct_mask] = merged.loc[direct_mask, "direct_source"].fillna("vegas_direct")
        market_applied.loc[direct_mask] = True

    futures_mask = use_futures & ~direct_mask & merged["futures_prob"].notna()
    if futures_mask.any():
        delta = (pd.to_numeric(merged.loc[futures_mask, "futures_prob"], errors="coerce") - base_prob.loc[futures_mask]).clip(-max_delta, max_delta)
        blended.loc[futures_mask] = (base_prob.loc[futures_mask] + 0.35 * delta).clip(0.001, 0.999)
        source_used.loc[futures_mask] = merged.loc[futures_mask, "futures_source"].fillna("kalshi_futures")
        market_applied.loc[futures_mask] = True

    injury_shift = (merged["t1_shift"] - merged["t2_shift"]).clip(-injury_cap, injury_cap)
    injury_applied = (gender == "M") & allow_injury_effective & ((merged["t1_out"] > 0) | (merged["t2_out"] > 0))
    post_prob = blended.copy()
    if bool(np.any(injury_applied)):
        post_prob.loc[injury_applied] = _inv_logit(_logit(blended.loc[injury_applied]) + injury_shift.loc[injury_applied]).clip(0.001, 0.999)

    if allow_sharpen_effective and gender == "M":
        direct_source_mask = market_applied & source_used.isin(["vegas_direct", "sportsbook", "bpi_direct"])
        strong_signal_mask = direct_source_mask & ((merged["direct_prob"] - base_prob).abs() >= 0.03)
        if bool(np.any(strong_signal_mask)):
            sharpen_delta = (pd.to_numeric(merged.loc[strong_signal_mask, "direct_prob"], errors="coerce") - post_prob.loc[strong_signal_mask]) * 0.15
            sharpen_delta = sharpen_delta.clip(-0.005, 0.005)
            post_prob.loc[strong_signal_mask] = (post_prob.loc[strong_signal_mask] + sharpen_delta).clip(0.001, 0.999)
            sharpen_applied.loc[strong_signal_mask] = sharpen_delta.abs() > 1e-12

    audit = pd.DataFrame(
        {
            "ID": merged["ID"],
            "pre_prob": base_prob,
            "post_prob": post_prob,
            "delta": post_prob - base_prob,
            "source_used": source_used,
            "injury_applied": injury_applied,
            "market_applied": market_applied,
            "sharpen_applied": sharpen_applied,
        }
    )
    adjusted = pd.DataFrame({"ID": merged["ID"], "Pred": post_prob})
    summary = {
        "season": int(season),
        "rows": int(len(adjusted)),
        "changed_rows": int((audit["delta"].abs() > 1e-12).sum()),
        "mean_abs_delta": float(audit["delta"].abs().mean()),
        "max_abs_delta": float(audit["delta"].abs().max()),
        "overlay_submission_only_enabled": True,
        "overlay_source_profile": overlay_source_profile,
        "injury_applied_rows": int(audit["injury_applied"].sum()),
        "market_applied_rows": int(audit["market_applied"].sum()),
        "sharpen_applied_rows": int(audit["sharpen_applied"].sum()),
        "futures_enabled": bool(use_futures),
    }
    return adjusted, audit, summary
