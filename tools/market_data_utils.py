from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ModuleNotFoundError:
    fuzz = None
    process = None

from zizzii_features import build_team_name_lookup, normalize_team_name, resolve_team_id


DEFAULT_FUZZY_THRESHOLD = 85.0


def _best_fuzzy_match(name: str, lookup: dict[str, int]) -> tuple[Optional[str], Optional[int], float]:
    if not name or not lookup:
        return None, None, 0.0
    candidates = list(lookup.keys())
    if process is not None and fuzz is not None:
        match = process.extractOne(name, candidates, scorer=fuzz.token_set_ratio)
        if match is not None:
            matched_name, score, _ = match
            return str(matched_name), int(lookup[matched_name]), float(score)

    best_name = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, name, candidate).ratio() * 100.0
        if score > best_score:
            best_name = candidate
            best_score = score
    if best_name is None:
        return None, None, 0.0
    return best_name, int(lookup[best_name]), float(best_score)


def attach_team_ids(
    df: pd.DataFrame,
    gender: str,
    team1_col: str,
    team2_col: str,
    team1_id_col: str | None = None,
    team2_id_col: str | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> pd.DataFrame:
    frame = df.copy()
    lookup = build_team_name_lookup(gender)

    audit_rows: list[dict[str, object]] = []
    cache: dict[str, Optional[int]] = {}

    def resolve_name(raw_value: object) -> Optional[int]:
        raw_name = str(raw_value).strip()
        if raw_name in cache:
            return cache[raw_name]

        normalized = normalize_team_name(raw_name)
        team_id = resolve_team_id(raw_name, lookup)
        if team_id is not None:
            cache[raw_name] = int(team_id)
            audit_rows.append(
                {
                    "raw_name": raw_name,
                    "normalized_name": normalized,
                    "matched_name": normalized,
                    "matched_id": int(team_id),
                    "score": 100.0,
                    "status": "exact",
                }
            )
            return int(team_id)

        matched_name, matched_id, score = _best_fuzzy_match(normalized, lookup)
        if matched_id is not None and score >= float(fuzzy_threshold):
            cache[raw_name] = int(matched_id)
            audit_rows.append(
                {
                    "raw_name": raw_name,
                    "normalized_name": normalized,
                    "matched_name": matched_name,
                    "matched_id": int(matched_id),
                    "score": float(score),
                    "status": "fuzzy",
                }
            )
            return int(matched_id)

        cache[raw_name] = None
        audit_rows.append(
            {
                "raw_name": raw_name,
                "normalized_name": normalized,
                "matched_name": matched_name,
                "matched_id": None,
                "score": float(score),
                "status": "unmatched",
            }
        )
        return None

    team1_target = team1_id_col or "Team1ID"
    team2_target = team2_id_col or "Team2ID"

    if team1_target not in frame.columns and team1_col in frame.columns:
        frame[team1_target] = frame[team1_col].map(resolve_name)
    if team2_target not in frame.columns and team2_col in frame.columns:
        frame[team2_target] = frame[team2_col].map(resolve_name)

    frame.attrs["team_match_audit"] = pd.DataFrame(audit_rows).drop_duplicates(subset=["raw_name"], keep="last")
    return frame


def no_vig_prob(prob_a: pd.Series, prob_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = pd.to_numeric(prob_a, errors="coerce")
    b = pd.to_numeric(prob_b, errors="coerce")
    total = a + b
    novig = a / total.replace(0, np.nan)
    novig = novig.fillna(0.5).clip(0.001, 0.999)
    hold = total - 1.0
    return novig, hold


def _swap_columns(frame: pd.DataFrame, left: str, right: str, mask: pd.Series) -> None:
    if left not in frame.columns or right not in frame.columns:
        return
    left_values = frame.loc[mask, left].copy()
    frame.loc[mask, left] = frame.loc[mask, right].to_numpy()
    frame.loc[mask, right] = left_values.to_numpy()


def canonicalize_matchups(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "Team1ID" not in frame.columns or "Team2ID" not in frame.columns:
        return frame

    frame["Team1ID"] = pd.to_numeric(frame["Team1ID"], errors="coerce")
    frame["Team2ID"] = pd.to_numeric(frame["Team2ID"], errors="coerce")
    frame = frame.dropna(subset=["Team1ID", "Team2ID"]).copy()
    frame["Team1ID"] = frame["Team1ID"].astype(int)
    frame["Team2ID"] = frame["Team2ID"].astype(int)

    swap_mask = frame["Team1ID"] > frame["Team2ID"]
    if not swap_mask.any():
        return frame

    for left, right in [
        ("Team1ID", "Team2ID"),
        ("Team1Name", "Team2Name"),
        ("Team1Moneyline", "Team2Moneyline"),
        ("Team1ImpliedProb", "Team2ImpliedProb"),
        ("Team1DecimalOdds", "Team2DecimalOdds"),
    ]:
        _swap_columns(frame, left, right, swap_mask)

    for prob_col in ["NoVigProb", "MarketProb", "Prob", "ImpliedProb", "SpreadProb", "Team1SpreadProb"]:
        if prob_col in frame.columns:
            frame.loc[swap_mask, prob_col] = 1.0 - pd.to_numeric(frame.loc[swap_mask, prob_col], errors="coerce")

    for spread_col in ["Spread", "OpenSpread", "LastSpread", "HighSpread", "LowSpread", "Team1Spread"]:
        if spread_col in frame.columns:
            frame.loc[swap_mask, spread_col] = -pd.to_numeric(frame.loc[swap_mask, spread_col], errors="coerce")

    return frame


def summarize(df: pd.DataFrame) -> None:
    team1_col = "Team1ID" if "Team1ID" in df.columns else "T1" if "T1" in df.columns else None
    team2_col = "Team2ID" if "Team2ID" in df.columns else "T2" if "T2" in df.columns else None
    if team1_col is None or team2_col is None:
        print(f"rows={len(df)}")
        return
    mapped = int(df[team1_col].notna().sum() + df[team2_col].notna().sum())
    total = int(len(df) * 2)
    print(f"Mapped team ids: {mapped}/{total}")
    unresolved = int(total - mapped)
    print(f"Rows with missing team ids: {unresolved // 2 if total else 0}")


def write_unmatched_log(audit_df: Optional[pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_df is None or audit_df.empty:
        if output_path.exists():
            output_path.unlink()
        return

    unresolved = audit_df[audit_df["status"] == "unmatched"].copy()
    if unresolved.empty:
        if output_path.exists():
            output_path.unlink()
        return
    unresolved.sort_values(["score", "raw_name"], ascending=[False, True]).to_csv(output_path, index=False)
