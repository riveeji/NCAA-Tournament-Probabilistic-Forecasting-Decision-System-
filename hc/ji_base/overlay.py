from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hc.gold.overlay import (
    _load_direct_market_candidates,
    load_direct_market_probs,
    load_men_injury_adjustments,
)

from .config import JIBaseOverlayConfig
from .predict import parse_submission_ids

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA = ROOT / "external-data"


def _logit(prob: pd.Series) -> pd.Series:
    clipped = prob.clip(1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _inv_logit(logit: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-logit))


def _load_direct_market(gender: str, season: int, overlay_source_profile: str) -> pd.DataFrame:
    if overlay_source_profile == "direct_only":
        direct = _load_direct_market_candidates(gender, season)
        if direct.empty:
            return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
        direct = direct.loc[direct["source_used"].isin({"vegas_direct", "sportsbook"})].copy()
        if direct.empty:
            return pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
        return direct.groupby(["Season", "T1", "T2"], as_index=False).agg(
            market_prob=("market_prob", "mean"),
            source_used=("source_used", "first"),
        )
    return load_direct_market_probs(gender, season)


def load_men_player_level_injury_adjustments(season: int) -> pd.DataFrame:
    path = EXTERNAL_DATA / f"MPlayerInjuryImpact_{season}.csv"
    columns = ["Season", "TeamID", "injury_shift", "confirmed_out_count", "high_impact_out_count"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    required = {
        "Season",
        "TeamID",
        "PlayerName",
        "Status",
        "ImpactScore",
        "AvailabilityWeight",
        "InjuryDeduction",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    usable = frame.copy()
    usable["Season"] = pd.to_numeric(usable["Season"], errors="coerce")
    usable["TeamID"] = pd.to_numeric(usable["TeamID"], errors="coerce")
    usable["ImpactScore"] = pd.to_numeric(usable["ImpactScore"], errors="coerce").fillna(0.0)
    usable["AvailabilityWeight"] = pd.to_numeric(usable["AvailabilityWeight"], errors="coerce").fillna(0.0)
    usable["InjuryDeduction"] = pd.to_numeric(usable["InjuryDeduction"], errors="coerce").fillna(0.0)
    usable["Status"] = usable["Status"].astype(str).str.strip()
    usable = usable.loc[(usable["Season"] == int(season)) & usable["TeamID"].notna()].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    usable["confirmed_out_count"] = (
        (usable["AvailabilityWeight"] >= 0.75)
        & usable["Status"].isin({"Out For Season", "Out"})
    ).astype(int)
    usable["high_impact_out_count"] = (
        (usable["confirmed_out_count"] == 1)
        & (usable["ImpactScore"] >= 3.0)
    ).astype(int)
    usable["injury_shift"] = -usable["InjuryDeduction"].clip(lower=0.0)

    aggregated = usable.groupby(["Season", "TeamID"], as_index=False).agg(
        injury_shift=("injury_shift", "sum"),
        confirmed_out_count=("confirmed_out_count", "sum"),
        high_impact_out_count=("high_impact_out_count", "sum"),
    )
    return aggregated


def _resolve_men_injury_adjustments(season: int, config: JIBaseOverlayConfig) -> tuple[pd.DataFrame, str]:
    columns = ["Season", "TeamID", "injury_shift", "confirmed_out_count", "high_impact_out_count"]
    if config.injury_mode == "player_level_v2":
        player_level = load_men_player_level_injury_adjustments(season)
        if not player_level.empty:
            resolved = player_level.copy()
            resolved["confirmed_out_count"] = pd.to_numeric(resolved["confirmed_out_count"], errors="coerce").fillna(0.0)
            resolved["high_impact_out_count"] = pd.to_numeric(resolved["high_impact_out_count"], errors="coerce").fillna(0.0)
            resolved["injury_shift"] = pd.to_numeric(resolved["injury_shift"], errors="coerce").fillna(0.0)
            return resolved[columns], "player_level_v2"
        fallback = load_men_injury_adjustments(season)
        if fallback.empty:
            return pd.DataFrame(columns=columns), "team_confirmed_gate_fallback"
        normalized = fallback.rename(columns={"confirmed_out": "confirmed_out_count"}).copy()
        normalized["high_impact_out_count"] = 0.0
        normalized["injury_shift"] = pd.to_numeric(normalized["injury_shift"], errors="coerce").fillna(0.0)
        normalized["confirmed_out_count"] = pd.to_numeric(normalized["confirmed_out_count"], errors="coerce").fillna(0.0)
        return normalized[columns], "team_confirmed_gate_fallback"

    team_level = load_men_injury_adjustments(season)
    if team_level.empty:
        return pd.DataFrame(columns=columns), "team_confirmed_gate"
    normalized = team_level.rename(columns={"confirmed_out": "confirmed_out_count"}).copy()
    normalized["high_impact_out_count"] = 0.0
    normalized["injury_shift"] = pd.to_numeric(normalized["injury_shift"], errors="coerce").fillna(0.0)
    normalized["confirmed_out_count"] = pd.to_numeric(normalized["confirmed_out_count"], errors="coerce").fillna(0.0)
    return normalized[columns], "team_confirmed_gate"


def apply_submission_overlay(
    predictions: pd.DataFrame,
    *,
    gender: str,
    season: int,
    config: JIBaseOverlayConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = predictions.copy()
    if {"Season", "T1", "T2"} - set(base.columns):
        base = parse_submission_ids(base)

    base["pre_prob"] = pd.to_numeric(base["Pred"], errors="coerce").clip(0.001, 0.999)
    base["Season"] = pd.to_numeric(base["Season"], errors="coerce").astype(int)
    base["T1"] = pd.to_numeric(base["T1"], errors="coerce").astype(int)
    base["T2"] = pd.to_numeric(base["T2"], errors="coerce").astype(int)

    direct = (
        _load_direct_market(gender, season, config.overlay_source_profile)
        if config.allow_market
        else pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"])
    )
    merged = base.merge(
        direct.rename(columns={"market_prob": "direct_prob", "source_used": "direct_source"}),
        on=["Season", "T1", "T2"],
        how="left",
    )

    source_used = pd.Series("none", index=merged.index, dtype=object)
    market_applied = pd.Series(False, index=merged.index)
    post_prob = merged["pre_prob"].copy()

    direct_mask = merged["direct_prob"].notna()
    if config.allow_market and direct_mask.any():
        direct_prob = pd.to_numeric(merged.loc[direct_mask, "direct_prob"], errors="coerce")
        delta = (direct_prob - merged.loc[direct_mask, "pre_prob"]).clip(-config.max_delta, config.max_delta)
        post_prob.loc[direct_mask] = (
            merged.loc[direct_mask, "pre_prob"] + config.direct_weight * delta
        ).clip(0.001, 0.999)
        source_used.loc[direct_mask] = merged.loc[direct_mask, "direct_source"].fillna("direct_market")
        market_applied.loc[direct_mask] = True

    injury_mode = "none"
    injuries = pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out_count", "high_impact_out_count"])
    if gender == "M" and config.allow_injury:
        injuries, injury_mode = _resolve_men_injury_adjustments(season, config)
    injury_applied = pd.Series(False, index=merged.index)
    if not injuries.empty:
        t1 = injuries.rename(
            columns={
                "TeamID": "T1",
                "injury_shift": "t1_shift",
                "confirmed_out_count": "t1_out",
                "high_impact_out_count": "t1_high_impact_out",
            }
        )
        t2 = injuries.rename(
            columns={
                "TeamID": "T2",
                "injury_shift": "t2_shift",
                "confirmed_out_count": "t2_out",
                "high_impact_out_count": "t2_high_impact_out",
            }
        )
        merged = merged.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")
        for column in ("t1_shift", "t2_shift", "t1_out", "t2_out", "t1_high_impact_out", "t2_high_impact_out"):
            merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(0.0)
        injury_shift = (merged["t1_shift"] - merged["t2_shift"]).clip(-config.injury_cap, config.injury_cap)
        injury_applied = (
            (merged["t1_out"] >= float(config.injury_min_confirmed_out))
            | (merged["t2_out"] >= float(config.injury_min_confirmed_out))
            | (merged["t1_high_impact_out"] > 0.0)
            | (merged["t2_high_impact_out"] > 0.0)
        ).astype(bool)
        if config.injury_min_abs_shift > 0:
            injury_applied = injury_applied & ((merged["t1_shift"] - merged["t2_shift"]).abs() >= float(config.injury_min_abs_shift))
        if bool(np.any(injury_applied)):
            post_prob.loc[injury_applied] = _inv_logit(
                _logit(post_prob.loc[injury_applied]) + injury_shift.loc[injury_applied]
            ).clip(0.001, 0.999)
    else:
        injury_shift = pd.Series(0.0, index=merged.index)

    audit = pd.DataFrame(
        {
            "ID": merged["ID"],
            "pre_prob": merged["pre_prob"],
            "post_prob": post_prob,
            "delta": post_prob - merged["pre_prob"],
            "source_used": source_used,
            "market_applied": market_applied,
            "injury_applied": injury_applied,
            "injury_mode": injury_mode,
            "injury_shift_abs": injury_shift.abs(),
        }
    )
    adjusted = pd.DataFrame({"ID": merged["ID"], "Pred": post_prob})
    summary = {
        "season": int(season),
        "rows": int(len(adjusted)),
        "changed_rows": int((audit["delta"].abs() > 1e-12).sum()),
        "mean_abs_delta": float(audit["delta"].abs().mean()),
        "max_abs_delta": float(audit["delta"].abs().max()),
        "overlay_source_profile": config.overlay_source_profile,
        "overlay_stack": config.resolved_overlay_stack(),
        "market_applied_rows": int(audit["market_applied"].sum()),
        "injury_applied_rows": int(audit["injury_applied"].sum()),
        "injury_mode": injury_mode,
    }
    return adjusted, audit, summary
