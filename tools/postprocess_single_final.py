from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"
RESULTS_DIR = ROOT / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_train import load_matchup_market_odds, merge_market_features, safe_clip
from tools.postprocess_women_submission_b import (
    apply_postprocess as apply_women_postprocess,
    attach_team_meta as attach_women_team_meta,
    build_team_features,
    grid_search_backtest as grid_search_women,
    load_backtest_frame as load_women_backtest_frame,
    parse_seed_meta as parse_women_seed_meta,
)


DEFAULT_WOMEN_CONFIG = {
    "host_round1": 0.015,
    "host_round2": 0.006,
    "power_intensity": 0.0,
    "tossup_boost": 0.03,
}

DEFAULT_MEN_CONFIG: dict[str, float | None] = {
    "pred_edge": None,
    "market_edge": 0.90,
    "seed_gap": 6.0,
    "blend_weight": 1.0,
    "floor_high": None,
}

DEFAULT_MEN_SLSQP_WEIGHTS = {
    "BaseProb": 0.0,
    "MenAsymProb": 0.54,
    "MarketOnlyProb": 0.46,
}

DEFAULT_WOMEN_SLSQP_WEIGHTS = {
    "BaseProb": 0.0,
    "WomenGreedyProb": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest and apply a single-final asymmetric postprocess layer."
    )
    parser.add_argument(
        "--men-strategy-oof",
        default="",
        help="Optional men strategy_oof CSV. Defaults to the latest results/strategy_oof_M_*.csv.",
    )
    parser.add_argument(
        "--women-oof",
        default="",
        help="Optional women OOF CSV. Defaults to the latest results/oof_W_*.csv.",
    )
    parser.add_argument(
        "--submission",
        default="submission_stage2.csv",
        help="Submission file to postprocess.",
    )
    parser.add_argument(
        "--output",
        default="submission_stage2_single_final.csv",
        help="Output path for the single-final candidate.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary path. Defaults to results/single_final_summary_<timestamp>.json.",
    )
    parser.add_argument(
        "--seasons",
        default="2021,2022,2023,2024,2025",
        help="Backtest seasons, comma-separated.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip OOF search and use the recommended configs directly.",
    )
    return parser.parse_args()


def latest_file(pattern: str) -> Path:
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No files matched {pattern} under {RESULTS_DIR}")
    return candidates[0]


def parse_seasons(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((y - p) ** 2))


def build_weight_map(columns: list[str], weights: np.ndarray) -> dict[str, float]:
    return {column: float(weight) for column, weight in zip(columns, weights)}


def combine_prob_columns(frame: pd.DataFrame, columns: list[str], weights: np.ndarray) -> np.ndarray:
    matrix = np.column_stack([pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) for column in columns])
    combined = matrix @ np.asarray(weights, dtype=float)
    return safe_clip(combined)


def fit_slsqp_weights(frame: pd.DataFrame, columns: list[str], label: np.ndarray, initial: dict[str, float] | None = None) -> np.ndarray:
    if len(columns) == 1:
        return np.array([1.0], dtype=float)

    matrix = np.column_stack([pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) for column in columns])
    y = np.asarray(label, dtype=float)

    if initial is None:
        x0 = np.full(len(columns), 1.0 / len(columns), dtype=float)
    else:
        x0 = np.array([float(initial.get(column, 0.0)) for column in columns], dtype=float)
        if not np.isfinite(x0).all() or x0.sum() <= 0:
            x0 = np.full(len(columns), 1.0 / len(columns), dtype=float)
        else:
            x0 = x0 / x0.sum()

    def objective(weights: np.ndarray) -> float:
        pred = safe_clip(matrix @ weights)
        return float(np.mean((y - pred) ** 2))

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(columns),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )
    if not result.success or not np.isfinite(result.x).all():
        return x0
    weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() <= 0:
        return x0
    return weights / weights.sum()


def rolling_slsqp_backtest(
    frame: pd.DataFrame,
    columns: list[str],
    label_col: str,
    seasons: list[int],
    default_weights: dict[str, float],
) -> tuple[np.ndarray, float, dict[int, dict[str, float]]]:
    work = frame.copy()
    preds = np.full(len(work), np.nan, dtype=float)
    seasonal_weights: dict[int, dict[str, float]] = {}

    for idx, season in enumerate(sorted(seasons)):
        test_mask = work["Season"] == season
        train_mask = work["Season"] < season
        if idx == 0 or train_mask.sum() == 0:
            weights = np.array([float(default_weights.get(column, 0.0)) for column in columns], dtype=float)
            if weights.sum() <= 0:
                weights = np.full(len(columns), 1.0 / len(columns), dtype=float)
            else:
                weights = weights / weights.sum()
        else:
            weights = fit_slsqp_weights(
                work.loc[train_mask, columns],
                columns,
                work.loc[train_mask, label_col].to_numpy(dtype=float),
                initial=seasonal_weights.get(sorted(seasons)[idx - 1], default_weights),
            )
        preds[test_mask] = combine_prob_columns(work.loc[test_mask], columns, weights)
        seasonal_weights[season] = build_weight_map(columns, weights)

    score = brier(work[label_col].to_numpy(dtype=float), preds)
    return preds, score, seasonal_weights


def parse_seed_meta(gender: str) -> pd.DataFrame:
    seeds = pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")
    seeds["SeedNum"] = pd.to_numeric(seeds["Seed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return seeds[["Season", "TeamID", "SeedNum"]].dropna(subset=["SeedNum"]).copy()


def attach_men_context(frame: pd.DataFrame) -> pd.DataFrame:
    market_df = load_matchup_market_odds("M")
    out = frame.copy()
    out, _, _ = merge_market_features(out, market_df)

    seed_meta = parse_seed_meta("M")
    t1_seed = seed_meta.rename(columns={"TeamID": "T1", "SeedNum": "T1_SeedNum"})
    t2_seed = seed_meta.rename(columns={"TeamID": "T2", "SeedNum": "T2_SeedNum"})
    out = out.merge(t1_seed, on=["Season", "T1"], how="left")
    out = out.merge(t2_seed, on=["Season", "T2"], how="left")

    if "Diag_AbsSeedDiff" not in out.columns:
        out["Diag_AbsSeedDiff"] = (
            pd.to_numeric(out["T1_SeedNum"], errors="coerce") - pd.to_numeric(out["T2_SeedNum"], errors="coerce")
        ).abs()
    if "Diag_T1BetterSeed" not in out.columns:
        out["Diag_T1BetterSeed"] = (
            pd.to_numeric(out["T1_SeedNum"], errors="coerce") < pd.to_numeric(out["T2_SeedNum"], errors="coerce")
        ).astype(float)
    return out


def apply_men_postprocess(prob: np.ndarray, frame: pd.DataFrame, config: dict[str, float | None]) -> np.ndarray:
    adjusted = safe_clip(prob)
    market_prob = pd.to_numeric(frame.get("MarketProb"), errors="coerce").to_numpy(dtype=float)
    abs_seed = pd.to_numeric(frame.get("Diag_AbsSeedDiff"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    t1_better = pd.to_numeric(frame.get("Diag_T1BetterSeed"), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5

    market_edge = float(config["market_edge"])
    seed_gap = float(config["seed_gap"])
    blend_weight = float(config["blend_weight"])
    pred_edge = config["pred_edge"]
    floor_high = config["floor_high"]

    high_mask = (~np.isnan(market_prob)) & (market_prob >= market_edge) & (abs_seed >= seed_gap) & t1_better
    low_mask = (~np.isnan(market_prob)) & (market_prob <= 1.0 - market_edge) & (abs_seed >= seed_gap) & (~t1_better)

    if pred_edge is not None:
        pred_edge = float(pred_edge)
        high_mask &= adjusted >= pred_edge
        low_mask &= adjusted <= 1.0 - pred_edge

    blended = safe_clip((1.0 - blend_weight) * adjusted + blend_weight * np.nan_to_num(market_prob, nan=adjusted))
    adjusted[high_mask] = np.maximum(adjusted[high_mask], blended[high_mask])
    adjusted[low_mask] = np.minimum(adjusted[low_mask], blended[low_mask])

    if floor_high is not None:
        floor_high = float(floor_high)
        adjusted[high_mask] = np.maximum(adjusted[high_mask], floor_high)
        adjusted[low_mask] = np.minimum(adjusted[low_mask], 1.0 - floor_high)

    return safe_clip(adjusted)


def load_men_backtest_frame(strategy_oof_path: Path, seasons: Iterable[int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    oof = pd.read_csv(strategy_oof_path)
    oof = oof[oof["Season"].isin(list(seasons))].copy()
    frame = attach_men_context(oof[["Season", "T1", "T2", "Diag_AbsSeedDiff", "Diag_T1BetterSeed"]].copy())
    merged = oof.merge(
        frame[["Season", "T1", "T2", "MarketProb", "Diag_AbsSeedDiff", "Diag_T1BetterSeed"]],
        on=["Season", "T1", "T2", "Diag_AbsSeedDiff", "Diag_T1BetterSeed"],
        how="left",
    )
    return merged, merged["FinalProb"].to_numpy(dtype=float), merged["Label"].to_numpy(dtype=float)


def add_men_candidate_columns(frame: pd.DataFrame, base_prob: np.ndarray, men_config: dict[str, float | None]) -> pd.DataFrame:
    out = frame.copy()
    out["BaseProb"] = safe_clip(base_prob)
    out["MenAsymProb"] = apply_men_postprocess(base_prob, out, men_config)
    market_only = pd.to_numeric(out["MarketProb"], errors="coerce")
    out["MarketOnlyProb"] = safe_clip(market_only.fillna(out["BaseProb"]).to_numpy(dtype=float))
    return out


def grid_search_men(frame: pd.DataFrame, base_prob: np.ndarray, label: np.ndarray) -> tuple[dict[str, float | None], float]:
    baseline = brier(label, base_prob)
    best_score = baseline
    best_config = dict(DEFAULT_MEN_CONFIG)

    for pred_edge, market_edge, seed_gap, blend_weight, floor_high in product(
        (None, 0.80, 0.85, 0.90),
        (0.90, 0.93, 0.95, 0.97),
        (6.0, 8.0, 10.0),
        (0.35, 0.50, 0.65, 1.0),
        (None, 0.975, 0.98, 0.985),
    ):
        config = {
            "pred_edge": pred_edge,
            "market_edge": market_edge,
            "seed_gap": seed_gap,
            "blend_weight": blend_weight,
            "floor_high": floor_high,
        }
        score = brier(label, apply_men_postprocess(base_prob, frame, config))
        if score < best_score:
            best_score = score
            best_config = config

    return best_config, best_score


def add_women_candidate_columns(frame: pd.DataFrame, base_prob: np.ndarray, women_config: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["BaseProb"] = safe_clip(base_prob)
    out["WomenGreedyProb"] = apply_women_postprocess(base_prob, out, women_config)
    return out


def apply_to_submission(
    submission_path: Path,
    output_path: Path,
    men_config: dict[str, float | None],
    women_config: dict[str, float],
    men_weights: dict[str, float],
    women_weights: dict[str, float],
) -> dict[str, int]:
    submission = pd.read_csv(submission_path)
    ids = submission["ID"].str.split("_", expand=True)
    work = submission.copy()
    work["Season"] = ids[0].astype(int)
    work["T1"] = ids[1].astype(int)
    work["T2"] = ids[2].astype(int)

    men = work[work["T1"] < 2000].copy()
    women = work[work["T1"] >= 3000].copy()

    changed = {"men_rows_changed": 0, "women_rows_changed": 0}

    if not men.empty:
        men_frame = attach_men_context(men[["Season", "T1", "T2"]].copy())
        men_frame = add_men_candidate_columns(men_frame, men["Pred"].to_numpy(dtype=float), men_config)
        men["Pred"] = combine_prob_columns(men_frame, list(men_weights), np.array([men_weights[column] for column in men_weights], dtype=float))
        changed["men_rows_changed"] = int((men["Pred"].to_numpy(dtype=float) != work.loc[work["T1"] < 2000, "Pred"].to_numpy(dtype=float)).sum())

    if not women.empty:
        women_team_feats = build_team_features("W")
        women_seed_meta = parse_women_seed_meta("W")
        women_frame = attach_women_team_meta(women[["Season", "T1", "T2"]].copy(), women_team_feats, women_seed_meta)
        women_frame = add_women_candidate_columns(women_frame, women["Pred"].to_numpy(dtype=float), women_config)
        women["Pred"] = combine_prob_columns(women_frame, list(women_weights), np.array([women_weights[column] for column in women_weights], dtype=float))
        changed["women_rows_changed"] = int((women["Pred"].to_numpy(dtype=float) != work.loc[work["T1"] >= 3000, "Pred"].to_numpy(dtype=float)).sum())

    merged = work.merge(pd.concat([men[["ID", "Pred"]], women[["ID", "Pred"]]], ignore_index=True), on="ID", how="left", suffixes=("", "_new"))
    mask = merged["Pred_new"].notna()
    merged.loc[mask, "Pred"] = merged.loc[mask, "Pred_new"]
    merged = merged[["ID", "Pred"]]
    merged.to_csv(output_path, index=False)
    return changed


def main() -> None:
    args = parse_args()
    seasons = parse_seasons(args.seasons)
    men_strategy_oof = Path(args.men_strategy_oof) if args.men_strategy_oof else latest_file("strategy_oof_M_*.csv")
    women_oof = Path(args.women_oof) if args.women_oof else latest_file("oof_W_*.csv")

    if args.skip_backtest:
        men_config = dict(DEFAULT_MEN_CONFIG)
        women_config = dict(DEFAULT_WOMEN_CONFIG)
        men_baseline = np.nan
        men_rule_best = np.nan
        men_blend_best = np.nan
        women_baseline = np.nan
        women_rule_best = np.nan
        women_blend_best = np.nan
        men_weights = dict(DEFAULT_MEN_SLSQP_WEIGHTS)
        women_weights = dict(DEFAULT_WOMEN_SLSQP_WEIGHTS)
        men_seasonal_weights = {}
        women_seasonal_weights = {}
        men_selected_mode = "slsqp"
        women_selected_mode = "rule_only"
        men_selected_recent = np.nan
        women_selected_recent = np.nan
    else:
        men_frame, men_base, men_label = load_men_backtest_frame(men_strategy_oof, seasons)
        men_baseline = brier(men_label, men_base)
        men_config, men_rule_best = grid_search_men(men_frame, men_base, men_label)
        men_frame = add_men_candidate_columns(men_frame, men_base, men_config)
        _, men_blend_best, men_seasonal_weights = rolling_slsqp_backtest(
            men_frame[["Season", "Label", "BaseProb", "MenAsymProb", "MarketOnlyProb"]].copy(),
            ["BaseProb", "MenAsymProb", "MarketOnlyProb"],
            "Label",
            seasons,
            DEFAULT_MEN_SLSQP_WEIGHTS,
        )
        men_weights = build_weight_map(
            ["BaseProb", "MenAsymProb", "MarketOnlyProb"],
            fit_slsqp_weights(men_frame[["BaseProb", "MenAsymProb", "MarketOnlyProb"]], ["BaseProb", "MenAsymProb", "MarketOnlyProb"], men_label, DEFAULT_MEN_SLSQP_WEIGHTS),
        )

        women_frame, women_base, women_label = load_women_backtest_frame(women_oof, seasons, "Prob_lr_core")
        women_baseline = brier(women_label, women_base)
        women_config, women_rule_best = grid_search_women(women_frame, women_base, women_label)
        women_frame = add_women_candidate_columns(women_frame, women_base, women_config)
        _, women_blend_best, women_seasonal_weights = rolling_slsqp_backtest(
            women_frame[["Season", "BaseProb", "WomenGreedyProb"]].assign(Label=women_label),
            ["BaseProb", "WomenGreedyProb"],
            "Label",
            seasons,
            DEFAULT_WOMEN_SLSQP_WEIGHTS,
        )
        women_weights = build_weight_map(
            ["BaseProb", "WomenGreedyProb"],
            fit_slsqp_weights(women_frame[["BaseProb", "WomenGreedyProb"]], ["BaseProb", "WomenGreedyProb"], women_label, DEFAULT_WOMEN_SLSQP_WEIGHTS),
        )

        if men_blend_best < men_rule_best:
            men_selected_mode = "slsqp"
            men_selected_recent = men_blend_best
        else:
            men_selected_mode = "rule_only"
            men_selected_recent = men_rule_best
            men_weights = {"BaseProb": 0.0, "MenAsymProb": 1.0, "MarketOnlyProb": 0.0}

        if women_blend_best < women_rule_best:
            women_selected_mode = "slsqp"
            women_selected_recent = women_blend_best
        else:
            women_selected_mode = "rule_only"
            women_selected_recent = women_rule_best
            women_weights = {"BaseProb": 0.0, "WomenGreedyProb": 1.0}

        print("Single-final backtest")
        print(f"Men strategy OOF: {men_strategy_oof}")
        print(f"Women OOF:        {women_oof}")
        print(f"Seasons:          {seasons}")
        print(f"Men baseline:     {men_baseline:.5f}")
        print(f"Men rule best:    {men_rule_best:.5f}")
        print(f"Men SLSQP best:   {men_blend_best:.5f}")
        print(f"Women baseline:   {women_baseline:.5f}")
        print(f"Women rule best:  {women_rule_best:.5f}")
        print(f"Women SLSQP best: {women_blend_best:.5f}")
        print(f"Combined baseline:{((men_baseline + women_baseline) / 2.0):.5f}")
        print(f"Combined rule:    {((men_rule_best + women_rule_best) / 2.0):.5f}")
        print(f"Combined SLSQP:   {((men_blend_best + women_blend_best) / 2.0):.5f}")
        print(f"Men selected:     {men_selected_mode} ({men_selected_recent:.5f})")
        print(f"Women selected:   {women_selected_mode} ({women_selected_recent:.5f})")
        print("Men config:")
        print(json.dumps(men_config, indent=2))
        print("Men SLSQP weights:")
        print(json.dumps(men_weights, indent=2))
        print("Women config:")
        print(json.dumps(women_config, indent=2))
        print("Women SLSQP weights:")
        print(json.dumps(women_weights, indent=2))

    changed = apply_to_submission(Path(args.submission), Path(args.output), men_config, women_config, men_weights, women_weights)
    print(f"\nSaved single-final candidate -> {args.output}")
    print(f"Rows changed: men={changed['men_rows_changed']}, women={changed['women_rows_changed']}")

    summary = {
        "men_strategy_oof": str(men_strategy_oof),
        "women_oof": str(women_oof),
        "seasons": seasons,
        "men_config": men_config,
        "women_config": women_config,
        "men_slsqp_weights": men_weights,
        "women_slsqp_weights": women_weights,
        "men_selected_mode": men_selected_mode,
        "women_selected_mode": women_selected_mode,
        "men_baseline_recent": men_baseline,
        "men_rule_best_recent": men_rule_best,
        "men_best_recent": men_blend_best,
        "men_selected_recent": men_selected_recent,
        "women_baseline_recent": women_baseline,
        "women_rule_best_recent": women_rule_best,
        "women_best_recent": women_blend_best,
        "women_selected_recent": women_selected_recent,
        "combined_baseline_recent": None if np.isnan(men_baseline) or np.isnan(women_baseline) else (men_baseline + women_baseline) / 2.0,
        "combined_rule_recent": None if np.isnan(men_rule_best) or np.isnan(women_rule_best) else (men_rule_best + women_rule_best) / 2.0,
        "combined_best_recent": None if np.isnan(men_blend_best) or np.isnan(women_blend_best) else (men_blend_best + women_blend_best) / 2.0,
        "combined_selected_recent": None if np.isnan(men_selected_recent) or np.isnan(women_selected_recent) else (men_selected_recent + women_selected_recent) / 2.0,
        "men_seasonal_slsqp_weights": men_seasonal_weights,
        "women_seasonal_slsqp_weights": women_seasonal_weights,
        "output_submission": str(Path(args.output).resolve()),
        "men_rows_changed": changed["men_rows_changed"],
        "women_rows_changed": changed["women_rows_changed"],
    }
    summary_path = Path(args.summary_output) if args.summary_output else RESULTS_DIR / f"single_final_summary_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
