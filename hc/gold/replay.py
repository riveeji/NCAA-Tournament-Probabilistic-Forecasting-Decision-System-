from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict

from .config import GoldConfig
from .data import build_gold_dataset
from .models import fit_gender_calibrator, fit_predict_raw


def _fit_outer_predictions(config: GoldConfig, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    target_kind = (
        "margin"
        if config.model_family in {"gold_harry_xgb_spread", "gold_min_xgb_spread", "gold_xgb_spread_light", "gold_spread_control"}
        else "label"
    )
    target = train["Margin"] if target_kind == "margin" else train["Label"]
    raw = fit_predict_raw(config, train[feature_cols], target, test[feature_cols], target_kind=target_kind)
    return pd.Series(raw, index=test.index, dtype=float)


def _fit_training_calibrator(config: GoldConfig, train: pd.DataFrame, feature_cols: list[str]):
    seasons = sorted(train["Season"].unique())
    if len(seasons) < 2:
        raw = _fit_outer_predictions(config, train, train, feature_cols)
        return fit_gender_calibrator(probabilities=raw, labels=train["Label"], calibration_mode=config.calibration_mode)

    oof_parts: list[pd.Series] = []
    label_parts: list[pd.Series] = []
    for season in seasons:
        inner_train = train.loc[train["Season"] != season]
        inner_valid = train.loc[train["Season"] == season]
        if inner_train.empty or inner_valid.empty:
            continue
        raw = _fit_outer_predictions(config, inner_train, inner_valid, feature_cols)
        oof_parts.append(raw)
        label_parts.append(inner_valid["Label"])

    if not oof_parts:
        raw = _fit_outer_predictions(config, train, train, feature_cols)
        return fit_gender_calibrator(probabilities=raw, labels=train["Label"], calibration_mode=config.calibration_mode)

    return fit_gender_calibrator(
        probabilities=pd.concat(oof_parts).sort_index(),
        labels=pd.concat(label_parts).sort_index(),
        calibration_mode=config.calibration_mode,
    )


def run_gender_replay(config: GoldConfig) -> dict:
    dataset = build_gold_dataset(config)
    feature_cols = [column for column in config.resolved_model_features() if column in dataset.columns]
    seasons = sorted(dataset["Season"].unique())
    predictions: list[pd.DataFrame] = []
    by_season_rows: list[dict] = []
    clip_low, clip_high = config.resolved_clip_bounds()
    feature_profile = config.resolved_feature_profile()
    rating_profile = config.resolved_rating_profile()
    selection_objective = config.resolved_selection_objective()
    rating_source_profile = config.resolved_rating_source_profile()

    for test_season in seasons:
        train = dataset.loc[dataset["Season"] != test_season].copy()
        test = dataset.loc[dataset["Season"] == test_season].copy()
        fold_features = [column for column in feature_cols if train[column].notna().any()]
        calibrator = _fit_training_calibrator(config, train, fold_features)
        raw_prob = _fit_outer_predictions(config, train, test, fold_features)
        final_prob = pd.Series(np.clip(calibrator.predict(raw_prob), clip_low, clip_high), index=test.index, dtype=float)

        brier = float(((final_prob - test["Label"]) ** 2).mean())
        by_season_rows.append(
            {
                "season": int(test_season),
                "gender": config.gender,
                "gender_segment": "men" if config.gender == "M" else "women",
                "model_family": config.model_family,
                "feature_profile": feature_profile,
                "rating_profile": rating_profile,
                "rating_source_profile": rating_source_profile,
                "calibration_mode": config.calibration_mode,
                "selection_objective": selection_objective,
                "games": int(len(test)),
                "brier": brier,
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
                    "gender": config.gender,
                    "gender_segment": "men" if config.gender == "M" else "women",
                    "model_family": config.model_family,
                    "feature_profile": feature_profile,
                    "rating_profile": rating_profile,
                    "rating_source_profile": rating_source_profile,
                    "calibration_mode": config.calibration_mode,
                    "selection_objective": selection_objective,
                }
            )
        )

    by_season = pd.DataFrame(by_season_rows)
    latest_season = int(by_season["season"].max())
    recent_cutoff = latest_season - config.recent_window + 1
    recent = by_season.loc[by_season["season"] >= recent_cutoff, "brier"]
    return {
        "config": asdict(config),
        "gender": config.gender,
        "gender_segment": "men" if config.gender == "M" else "women",
        "model_family": config.model_family,
        "feature_profile": feature_profile,
        "rating_profile": rating_profile,
        "rating_source_profile": rating_source_profile,
        "calibration_mode": config.calibration_mode,
        "selection_objective": selection_objective,
        "mean_brier": float(by_season["brier"].mean()),
        "latest_season": latest_season,
        "latest_season_brier": float(by_season.loc[by_season["season"] == latest_season, "brier"].iloc[-1]),
        "recent_window_brier": float(recent.mean()) if not recent.empty else float("nan"),
        "brier_variance": float(by_season["brier"].var(ddof=0)),
        "by_season": by_season,
        "predictions": pd.concat(predictions, ignore_index=True),
    }
