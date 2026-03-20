from __future__ import annotations

from typing import Iterable

import pandas as pd


TEAM_SIGNAL_KEYS = ["Season", "TeamID"]
MATCHUP_SIGNAL_KEYS = ["Season", "T1", "T2"]


def _last_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[-1]


def canonicalize_team_signal_frame(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    priority: int = 0,
    snapshot_col: str = "SnapshotDate",
) -> pd.DataFrame:
    if frame.empty or not set(TEAM_SIGNAL_KEYS).issubset(frame.columns):
        return pd.DataFrame(columns=TEAM_SIGNAL_KEYS)
    out = frame.copy()
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce")
    out["TeamID"] = pd.to_numeric(out["TeamID"], errors="coerce")
    out = out.dropna(subset=TEAM_SIGNAL_KEYS).copy()
    if out.empty:
        return pd.DataFrame(columns=TEAM_SIGNAL_KEYS)
    out["Season"] = out["Season"].astype(int)
    out["TeamID"] = out["TeamID"].astype(int)
    if snapshot_col in out.columns:
        out[snapshot_col] = pd.to_datetime(out[snapshot_col], errors="coerce", utc=True)
    else:
        out[snapshot_col] = pd.NaT
    if source is not None:
        out["SignalSource"] = str(source)
    elif "SignalSource" not in out.columns:
        out["SignalSource"] = "unknown"
    out["SignalPriority"] = int(priority)
    return out


def coalesce_team_signal_frames(frames: Iterable[pd.DataFrame], *, snapshot_col: str = "SnapshotDate") -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=TEAM_SIGNAL_KEYS)
    merged = pd.concat(valid, ignore_index=True, sort=False)
    merged = merged.dropna(subset=TEAM_SIGNAL_KEYS).copy()
    merged["Season"] = pd.to_numeric(merged["Season"], errors="coerce").astype(int)
    merged["TeamID"] = pd.to_numeric(merged["TeamID"], errors="coerce").astype(int)
    if snapshot_col not in merged.columns:
        merged[snapshot_col] = pd.NaT
    merged[snapshot_col] = pd.to_datetime(merged[snapshot_col], errors="coerce", utc=True)
    if "SignalPriority" not in merged.columns:
        merged["SignalPriority"] = 0
    merged["SignalPriority"] = pd.to_numeric(merged["SignalPriority"], errors="coerce").fillna(0).astype(int)
    merged = merged.sort_values(TEAM_SIGNAL_KEYS + ["SignalPriority", snapshot_col], na_position="first")
    value_columns = [column for column in merged.columns if column not in {*TEAM_SIGNAL_KEYS, "SignalPriority"}]
    aggregated = merged.groupby(TEAM_SIGNAL_KEYS, as_index=False)[value_columns].agg(_last_non_null)
    if snapshot_col in aggregated.columns:
        aggregated = aggregated.drop(columns=[snapshot_col])
    if "SignalSource" in aggregated.columns:
        aggregated = aggregated.drop(columns=["SignalSource"])
    return aggregated


def canonicalize_matchup_signal_frame(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    priority: int = 0,
    snapshot_col: str = "SnapshotTime",
) -> pd.DataFrame:
    if frame.empty or not set(MATCHUP_SIGNAL_KEYS).issubset(frame.columns):
        return pd.DataFrame(columns=MATCHUP_SIGNAL_KEYS)
    out = frame.copy()
    for column in MATCHUP_SIGNAL_KEYS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=MATCHUP_SIGNAL_KEYS).copy()
    if out.empty:
        return pd.DataFrame(columns=MATCHUP_SIGNAL_KEYS)
    out["Season"] = out["Season"].astype(int)
    out["T1"] = out["T1"].astype(int)
    out["T2"] = out["T2"].astype(int)
    if snapshot_col in out.columns:
        out[snapshot_col] = pd.to_datetime(out[snapshot_col], errors="coerce", utc=True)
    else:
        out[snapshot_col] = pd.NaT
    if source is not None:
        out["SignalSource"] = str(source)
    elif "SignalSource" not in out.columns:
        out["SignalSource"] = "unknown"
    out["SignalPriority"] = int(priority)
    return out


def coalesce_matchup_signal_frames(frames: Iterable[pd.DataFrame], *, snapshot_col: str = "SnapshotTime") -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=MATCHUP_SIGNAL_KEYS)
    merged = pd.concat(valid, ignore_index=True, sort=False)
    merged = merged.dropna(subset=MATCHUP_SIGNAL_KEYS).copy()
    for column in MATCHUP_SIGNAL_KEYS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(int)
    if snapshot_col not in merged.columns:
        merged[snapshot_col] = pd.NaT
    merged[snapshot_col] = pd.to_datetime(merged[snapshot_col], errors="coerce", utc=True)
    if "SignalPriority" not in merged.columns:
        merged["SignalPriority"] = 0
    merged["SignalPriority"] = pd.to_numeric(merged["SignalPriority"], errors="coerce").fillna(0).astype(int)
    merged = merged.sort_values(MATCHUP_SIGNAL_KEYS + ["SignalPriority", snapshot_col], na_position="first")
    value_columns = [column for column in merged.columns if column not in {*MATCHUP_SIGNAL_KEYS, "SignalPriority"}]
    aggregated = merged.groupby(MATCHUP_SIGNAL_KEYS, as_index=False)[value_columns].agg(_last_non_null)
    if snapshot_col in aggregated.columns:
        aggregated = aggregated.drop(columns=[snapshot_col])
    if "SignalSource" in aggregated.columns:
        aggregated = aggregated.drop(columns=["SignalSource"])
    return aggregated
