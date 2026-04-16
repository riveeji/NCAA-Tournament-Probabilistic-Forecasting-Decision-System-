from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import V2Config
from .data import attach_market_columns, build_v2_dataset
from .market_blend import apply_market_experiment, clip_probs
from .models import fit_predict_margin, fit_predict_proba, fit_spread_calibration


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [col for col in dataset.columns if col.endswith("_diff")]


def _combine_probs(config: V2Config, lr_prob: pd.Series, tree_prob: pd.Series) -> pd.Series:
    if config.model_variant == "lr":
        return lr_prob
    if config.model_variant == "tree":
        return tree_prob
    return (lr_prob * config.lr_weight) + (tree_prob * config.tree_weight)


def _combine_margin(config: V2Config, lr_margin: pd.Series, tree_margin: pd.Series) -> pd.Series:
    if config.model_variant == "lr":
        return lr_margin
    if config.model_variant == "tree":
        return tree_margin
    return (lr_margin * config.lr_weight) + (tree_margin * config.tree_weight)


def run_gender_replay(config: V2Config) -> dict:
    dataset = attach_market_columns(build_v2_dataset(config), config.gender)
    feature_cols = _feature_columns(dataset)
    seasons = sorted(dataset["Season"].unique())
    predictions: list[pd.DataFrame] = []
    by_season_rows: list[dict] = []
    learner_family = config.resolved_learner_family()
    calibration_mode = config.resolved_calibration_mode()
    gender_profile = config.resolved_gender_profile()
    clip_low, clip_high = config.resolved_clip_bounds()
    linear_alpha = config.resolved_linear_alpha()

    for test_season in seasons:
        train_mask = dataset["Season"] != test_season
        test_mask = ~train_mask
        train = dataset.loc[train_mask]
        test = dataset.loc[test_mask]
        fold_feature_cols = [column for column in feature_cols if train[column].notna().any()]
        x_train = train[fold_feature_cols]
        x_test = test[fold_feature_cols]
        if config.route == "probability":
            y_train = train["Label"]
            lr_prob = pd.Series(fit_predict_proba("lr", x_train, y_train, x_test), index=test.index)
            tree_prob = pd.Series(fit_predict_proba("tree", x_train, y_train, x_test), index=test.index)
            base_prob = _combine_probs(config, lr_prob, tree_prob)
        elif config.route == "spread":
            y_train = train["Margin"]
            lr_margin = pd.Series(
                fit_predict_margin("lr", x_train, y_train, x_test, linear_alpha=linear_alpha),
                index=test.index,
            )
            tree_margin = pd.Series(fit_predict_margin("tree", x_train, y_train, x_test), index=test.index)
            base_margin = _combine_margin(config, lr_margin, tree_margin)
            train_lr_margin = pd.Series(
                fit_predict_margin("lr", x_train, y_train, x_train, linear_alpha=linear_alpha),
                index=train.index,
            )
            train_tree_margin = pd.Series(fit_predict_margin("tree", x_train, y_train, x_train), index=train.index)
            train_margin = _combine_margin(config, train_lr_margin, train_tree_margin)
            calibration = fit_spread_calibration(
                margin_pred=train_margin,
                labels=train["Label"],
                calibration_mode=calibration_mode,
                default_scale=config.spread_logit_scale,
            )
            base_prob = calibration.predict_proba(base_margin)
        else:
            raise ValueError(f"Unsupported route: {config.route}")
        if config.market_mode == "none":
            final_prob = clip_probs(base_prob, clip_low, clip_high)
        elif config.market_mode == "sportsbook":
            final_prob = apply_market_experiment(
                base_prob,
                test["sportsbook_prob"],
                weight=config.market_weight,
                max_delta=config.bounded_pull_delta,
                clip_low=clip_low,
                clip_high=clip_high,
            )
        elif config.market_mode == "sportsbook_prediction":
            sportsbook_blend = apply_market_experiment(
                base_prob,
                test["sportsbook_prob"],
                weight=config.market_weight,
                max_delta=config.bounded_pull_delta,
                clip_low=clip_low,
                clip_high=clip_high,
            )
            final_prob = apply_market_experiment(
                sportsbook_blend,
                test["prediction_market_prob"],
                weight=config.market_weight * 0.5,
                max_delta=config.bounded_pull_delta,
                clip_low=clip_low,
                clip_high=clip_high,
            )
        else:
            raise ValueError(f"Unsupported market mode: {config.market_mode}")

        brier = ((final_prob - test["Label"]) ** 2).mean()
        by_season_rows.append(
            {
                "route": config.route,
                "gender": config.gender,
                "season": int(test_season),
                "model_variant": config.model_variant,
                "learner_family": learner_family,
                "market_mode": config.market_mode,
                "feature_pack": config.feature_pack,
                "calibration_mode": calibration_mode,
                "gender_profile": gender_profile,
                "games": int(len(test)),
                "brier": float(brier),
                "sportsbook_coverage": float(test["sportsbook_prob"].notna().mean()),
                "prediction_market_coverage": float(test["prediction_market_prob"].notna().mean()),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "Season": test["Season"],
                    "T1": test["T1"],
                    "T2": test["T2"],
                    "Label": test["Label"],
                    "Prob": final_prob,
                    "route": config.route,
                    "learner_family": learner_family,
                    "market_mode": config.market_mode,
                    "model_variant": config.model_variant,
                    "feature_pack": config.feature_pack,
                    "calibration_mode": calibration_mode,
                    "gender_profile": gender_profile,
                }
            )
        )

    by_season = pd.DataFrame(by_season_rows)
    latest_season = int(by_season["season"].max())
    latest_season_brier = float(by_season.loc[by_season["season"] == latest_season, "brier"].iloc[-1])
    recent_cutoff = max(seasons) - config.recent_window + 1
    recent = by_season.loc[by_season["season"] >= recent_cutoff, "brier"]
    overall = {
        "config": asdict(config),
        "gender": config.gender,
        "route": config.route,
        "model_variant": config.model_variant,
        "learner_family": learner_family,
        "market_mode": config.market_mode,
        "feature_pack": config.feature_pack,
        "calibration_mode": calibration_mode,
        "gender_profile": gender_profile,
        "mean_brier": float(by_season["brier"].mean()),
        "latest_season": latest_season,
        "latest_season_brier": latest_season_brier,
        "recent_window_brier": float(recent.mean()) if not recent.empty else float("nan"),
        "brier_variance": float(by_season["brier"].var(ddof=0)),
        "sportsbook_coverage_mean": float(by_season["sportsbook_coverage"].mean()),
        "prediction_market_coverage_mean": float(by_season["prediction_market_coverage"].mean()),
        "by_season": by_season,
        "predictions": pd.concat(predictions, ignore_index=True),
    }
    return overall
