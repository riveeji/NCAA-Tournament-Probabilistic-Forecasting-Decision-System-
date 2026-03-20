from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hc.constants import CACHE_DIR, RESULTS_DIR, ROOT
from zizzii_train import (
    add_base_probability_features,
    build_adaptive_market_blend_oof,
    build_men_chalk_extremes_oof,
    build_men_market_extremes_oof,
    build_strategy_oof_predictions,
    build_tossup_specialist_oof,
    build_women_chalk_extremes_oof,
    build_women_dual_chalk_extremes_oof,
    build_women_market_extremes_oof,
    build_women_spread_extremes_oof,
    safe_clip,
    specialist_feature_columns,
)


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _json_value(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text in {"nan", "None", "{}"}:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _select_best_benchmark(gender: str) -> Optional[pd.Series]:
    path = RESULTS_DIR / "benchmarks.csv"
    if not path.exists():
        return None
    bench = pd.read_csv(path)
    bench = bench.loc[bench["Gender"].eq(gender)].copy()
    if bench.empty or "BestCVBrier" not in bench.columns:
        return None
    if "TimestampUTC" in bench.columns:
        bench = bench.sort_values(["TimestampUTC"], ascending=[False])
    else:
        bench = bench.sort_values(["RunID"], ascending=[False])
    return bench.iloc[0]


def _oof_path(gender: str, run_id: str) -> Path:
    return RESULTS_DIR / f"oof_{gender}_{run_id}.csv"


def _cache_path(gender: str, run_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"legacy_anchor_oof_{gender}_{run_id}.parquet"


def _maybe_apply_tossup(matchups: pd.DataFrame, strategy_oof: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if not _bool_value(row.get("HasTossupSpecialist")):
        return strategy_oof
    config = _json_value(row.get("TossupSpecialistConfig"))
    if not config:
        return strategy_oof
    specialist_df = matchups.merge(
        strategy_oof[[column for column in strategy_oof.columns if column in {"Season", "T1", "T2", "Label", "RawProb", "FinalProb", "FavoriteProb"}]],
        on=["Season", "T1", "T2", "Label"],
        how="inner",
    )
    if specialist_df.empty:
        return strategy_oof
    specialist_df = add_base_probability_features(specialist_df, specialist_df["FinalProb"].to_numpy())
    feature_cols = specialist_feature_columns(specialist_df)
    if not feature_cols:
        return strategy_oof
    config = dict(config)
    config["feature_cols"] = feature_cols
    return build_tossup_specialist_oof(matchups, strategy_oof, config)


def build_legacy_anchor_oof(
    gender: str,
    matchups: pd.DataFrame,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    row = _select_best_benchmark(gender)
    if row is None:
        empty = pd.DataFrame(columns=["Season", "T1", "T2", "Label", "Prob_legacy_anchor"])
        return empty, {"enabled": False, "reason": "no_benchmark"}

    run_id = str(row["RunID"])
    cache = _cache_path(gender, run_id)
    if cache.exists() and not force_rebuild:
        cached = pd.read_parquet(cache)
        return cached, {"enabled": True, "run_id": run_id, "source": "cache"}

    oof_path = _oof_path(gender, run_id)
    if not oof_path.exists():
        empty = pd.DataFrame(columns=["Season", "T1", "T2", "Label", "Prob_legacy_anchor"])
        return empty, {"enabled": False, "reason": "missing_oof", "run_id": run_id}

    oof_df = pd.read_csv(oof_path)
    raw_column = str(row.get("SelectedRawColumn", "ProbMean"))
    calibration_method = str(row.get("CalibrationMethod", "none"))
    shrinkage = float(pd.to_numeric(row.get("Shrinkage", 0.0), errors="coerce") or 0.0)

    strategy_oof = build_strategy_oof_predictions(
        oof_df,
        raw_column=raw_column,
        calibration_method=calibration_method,
        shrinkage=shrinkage,
        eval_years=None,
    )
    if strategy_oof.empty:
        empty = pd.DataFrame(columns=["Season", "T1", "T2", "Label", "Prob_legacy_anchor"])
        return empty, {"enabled": False, "reason": "empty_strategy", "run_id": run_id}

    if gender == "M":
        strategy_oof = _maybe_apply_tossup(matchups, strategy_oof, row)
        adaptive_cfg = _json_value(row.get("AdaptiveMarketBlendConfig"))
        if _bool_value(row.get("HasAdaptiveMarketBlend")) and adaptive_cfg and "MarketProb" in matchups.columns:
            strategy_oof = build_adaptive_market_blend_oof(matchups, strategy_oof, adaptive_cfg)
        chalk_cfg = _json_value(row.get("MenChalkExtremesConfig"))
        if _bool_value(row.get("HasMenChalkExtremes")) and chalk_cfg:
            strategy_oof = build_men_chalk_extremes_oof(matchups, strategy_oof, chalk_cfg)
        market_cfg = _json_value(row.get("MenMarketExtremesConfig"))
        if _bool_value(row.get("HasMenMarketExtremes")) and market_cfg:
            strategy_oof = build_men_market_extremes_oof(matchups, strategy_oof, market_cfg)
    else:
        chalk_cfg = _json_value(row.get("WomenChalkExtremesConfig"))
        if _bool_value(row.get("HasWomenChalkExtremes")) and chalk_cfg:
            strategy_oof = build_women_chalk_extremes_oof(matchups, strategy_oof, chalk_cfg)
        dual_cfg = _json_value(row.get("WomenDualChalkExtremesConfig"))
        if _bool_value(row.get("HasWomenDualChalkExtremes")) and dual_cfg:
            strategy_oof = build_women_dual_chalk_extremes_oof(matchups, strategy_oof, dual_cfg)
        market_cfg = _json_value(row.get("WomenMarketExtremesConfig"))
        if _bool_value(row.get("HasWomenMarketExtremes")) and market_cfg:
            strategy_oof = build_women_market_extremes_oof(matchups, strategy_oof, market_cfg)
        spread_cfg = _json_value(row.get("WomenSpreadExtremesConfig"))
        if _bool_value(row.get("HasWomenSpreadExtremes")) and spread_cfg:
            strategy_oof = build_women_spread_extremes_oof(matchups, strategy_oof, spread_cfg)

    anchor = strategy_oof[["Season", "T1", "T2", "Label", "FinalProb"]].copy()
    anchor = anchor.rename(columns={"FinalProb": "Prob_legacy_anchor"})
    anchor["Prob_legacy_anchor"] = safe_clip(pd.to_numeric(anchor["Prob_legacy_anchor"], errors="coerce").fillna(0.5).to_numpy())
    anchor.to_parquet(cache, index=False)
    return anchor, {
        "enabled": True,
        "run_id": run_id,
        "raw_column": raw_column,
        "calibration_method": calibration_method,
        "shrinkage": shrinkage,
        "source": str(oof_path),
    }


def merge_legacy_anchor_oof(
    base_oof: pd.DataFrame,
    gender: str,
    matchups: pd.DataFrame,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    anchor, summary = build_legacy_anchor_oof(gender, matchups, force_rebuild=force_rebuild)
    if anchor.empty:
        return base_oof, summary
    merged = base_oof.merge(anchor, on=["Season", "T1", "T2", "Label"], how="left")
    return merged, summary


def load_legacy_submission_anchor(gender: str, season: int) -> pd.DataFrame:
    paths = [
        ROOT / "submission_stage2_single_final.csv",
        ROOT / "submission_stage2.csv",
        ROOT / "submission_stage1.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "ID" not in frame.columns or "Pred" not in frame.columns:
            continue
        ids = frame["ID"].astype(str).str.split("_", expand=True)
        if ids.shape[1] != 3:
            continue
        frame = frame.copy()
        frame["Season"] = pd.to_numeric(ids[0], errors="coerce")
        frame["T1"] = pd.to_numeric(ids[1], errors="coerce")
        frame["T2"] = pd.to_numeric(ids[2], errors="coerce")
        frame = frame.loc[frame["Season"].eq(season)].copy()
        if gender == "M":
            frame = frame.loc[frame["T1"] < 2000].copy()
        else:
            frame = frame.loc[frame["T1"] >= 3000].copy()
        if frame.empty:
            continue
        frame["Prob_legacy_anchor"] = safe_clip(pd.to_numeric(frame["Pred"], errors="coerce").fillna(0.5).to_numpy())
        return frame[["ID", "Prob_legacy_anchor"]]
    return pd.DataFrame(columns=["ID", "Prob_legacy_anchor"])
