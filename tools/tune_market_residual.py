from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import build_team_features
from zizzii_train import (
    DATA_DIR,
    RESULTS_DIR,
    MARKET_RESIDUAL_MIN_ROWS,
    build_matchup_df,
    feature_frame,
    load_matchup_market_odds,
    merge_market_features,
    safe_clip,
    strategy_feature_merge,
    xgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune men xgb_market_resid overlay on the current clean data."
    )
    parser.add_argument("--trials", type=int, default=60, help="Optuna trial count.")
    parser.add_argument("--timeout-sec", type=int, default=0, help="Optional Optuna timeout.")
    parser.add_argument(
        "--strategy-oof",
        default="",
        help="Optional men strategy OOF CSV. Defaults to latest results/strategy_oof_M_*.csv.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary path. Defaults to results/market_residual_tuning_<timestamp>.json.",
    )
    parser.add_argument(
        "--min-eval-folds",
        type=int,
        default=2,
        help="Minimum number of held-out seasons that must contribute valid residual evaluation.",
    )
    return parser.parse_args()


def latest_file(pattern: str) -> Path:
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No files matched {pattern} under {RESULTS_DIR}")
    return candidates[0]


def build_training_frame() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    team_feats = build_team_features("M")
    matchups, _, lr_core_feats, lr_plus_feats, all_feats = build_matchup_df(tourney, team_feats, "M")
    market_df = load_matchup_market_odds("M")
    matchups, _, _ = merge_market_features(matchups, market_df)
    return matchups, lr_core_feats, lr_plus_feats, all_feats


def build_xgb_regressor(params: dict[str, float | int]):
    if xgb is None:
        raise SystemExit("xgboost is not installed.")
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        gamma=float(params["gamma"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
    )


def evaluate_params(
    params: dict[str, float | int],
    blend_weight: float,
    min_train_market_rows: int,
    min_eval_folds: int,
    merged: pd.DataFrame,
    lr_core_feats: list[str],
    lr_plus_feats: list[str],
    all_feats: list[str],
) -> tuple[float, list[dict[str, float]]]:
    eval_seasons = sorted(merged["Season"].unique())[-5:]
    fold_metrics: list[dict[str, float]] = []

    for season in eval_seasons:
        train_df = merged[merged["Season"] < season].copy()
        test_df = merged[merged["Season"] == season].copy()
        if test_df.empty:
            continue

        x_train = feature_frame(train_df, "all", lr_core_feats, lr_plus_feats, all_feats)
        x_test = feature_frame(test_df, "all", lr_core_feats, lr_plus_feats, all_feats)

        market_train = pd.to_numeric(train_df["MarketProb"], errors="coerce")
        valid_train = market_train.notna().to_numpy()
        if int(valid_train.sum()) < int(min_train_market_rows):
            continue

        model = build_xgb_regressor(params)
        residual_target = train_df.loc[valid_train, "Label"].to_numpy(dtype=float) - market_train.loc[valid_train].to_numpy(dtype=float)
        model.fit(x_train.loc[valid_train], residual_target)

        pred = test_df["FinalProb"].to_numpy(dtype=float).copy()
        market_test = pd.to_numeric(test_df["MarketProb"], errors="coerce")
        valid_test = market_test.notna().to_numpy()
        active_rows = int(valid_test.sum())
        if active_rows > 0:
            correction = np.asarray(model.predict(x_test.loc[valid_test]), dtype=float)
            overlay_prob = safe_clip(market_test.loc[valid_test].to_numpy(dtype=float) + correction)
            pred[valid_test] = safe_clip((1.0 - blend_weight) * pred[valid_test] + blend_weight * overlay_prob)

        score = brier_score_loss(test_df["Label"], pred)
        fold_metrics.append(
            {
                "season": float(season),
                "score": float(score),
                "active_rows": float(active_rows),
            }
        )

    if not fold_metrics or len(fold_metrics) < int(min_eval_folds):
        return float("inf"), []
    active_total = sum(metric["active_rows"] for metric in fold_metrics)
    if active_total < 60:
        return float("inf"), fold_metrics
    mean_score = float(np.mean([metric["score"] for metric in fold_metrics]))
    return mean_score, fold_metrics


def main() -> None:
    args = parse_args()
    if xgb is None:
        raise SystemExit("xgboost is not installed.")

    strategy_oof_path = Path(args.strategy_oof) if args.strategy_oof else latest_file("strategy_oof_M_*.csv")
    strategy_oof = pd.read_csv(strategy_oof_path)
    matchups, lr_core_feats, lr_plus_feats, all_feats = build_training_frame()
    merged = strategy_feature_merge(matchups, strategy_oof)
    if merged.empty:
        raise SystemExit("Merged market residual frame is empty.")

    base_score = float(brier_score_loss(strategy_oof["Label"], strategy_oof["FinalProb"]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 220, 760),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.60),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 15.0, log=True),
        }
        blend_weight = trial.suggest_float("blend_weight", 0.15, 1.00)
        min_train_market_rows = trial.suggest_categorical("min_train_market_rows", [100, 120, 140, 160, 180, 200, 220, MARKET_RESIDUAL_MIN_ROWS])
        score, fold_metrics = evaluate_params(
            params,
            blend_weight,
            min_train_market_rows,
            args.min_eval_folds,
            merged,
            lr_core_feats,
            lr_plus_feats,
            all_feats,
        )
        trial.set_user_attr("fold_metrics", fold_metrics)
        return score

    study = optuna.create_study(direction="minimize", study_name="men_xgb_market_resid")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout_sec or None, show_progress_bar=False)

    best_score = float(study.best_value)
    best_params = dict(study.best_params)
    best_blend = float(best_params.pop("blend_weight"))
    best_min_train_market_rows = int(best_params.pop("min_train_market_rows"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = Path(args.summary_output) if args.summary_output else RESULTS_DIR / f"market_residual_tuning_{timestamp}.json"
    payload = {
        "strategy_oof": str(strategy_oof_path),
        "base_score": base_score,
        "best_score": best_score,
        "improvement": base_score - best_score,
        "best_blend_weight": best_blend,
        "best_min_train_market_rows": best_min_train_market_rows,
        "best_params": best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_fold_metrics": study.best_trial.user_attrs.get("fold_metrics", []),
        "trial_count": int(len(study.trials)),
        "min_eval_folds": int(args.min_eval_folds),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Market residual tuning summary")
    print(f"Strategy OOF:   {strategy_oof_path}")
    print(f"Base score:     {base_score:.5f}")
    print(f"Best score:     {best_score:.5f}")
    print(f"Improvement:    {payload['improvement']:.5f}")
    print(f"Best blend:     {best_blend:.4f}")
    print(f"Best min rows:  {best_min_train_market_rows}")
    print(f"Best params:    {json.dumps(best_params, sort_keys=True)}")
    print(f"Summary saved:  {summary_path}")


if __name__ == "__main__":
    main()
