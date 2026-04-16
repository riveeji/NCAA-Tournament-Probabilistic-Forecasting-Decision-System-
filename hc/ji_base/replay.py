from __future__ import annotations

import numpy as np
import pandas as pd

from .config import JIBaseConfig
from .data import build_ji_dataset
from .models import fit_gender_calibrator, fit_predict_raw


def _fit_outer_raw_probability(config: JIBaseConfig, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    target = train["Margin"] if config.model_family == "JI_spread_xgb" else train["Label"]
    raw = fit_predict_raw(config, train[feature_cols], target, test[feature_cols])
    return pd.Series(raw, index=test.index, dtype=float)


def _fit_training_calibrator(config: JIBaseConfig, train: pd.DataFrame, feature_cols: list[str]):
    seasons = sorted(train["Season"].unique())
    if len(seasons) < 2:
        raw = _fit_outer_raw_probability(config, train, train, feature_cols)
        return fit_gender_calibrator(
            probabilities=raw,
            labels=train["Label"],
            calibration_mode=config.calibration_mode,
            isotonic_min_samples=config.isotonic_min_samples,
        )

    oof_parts: list[pd.Series] = []
    label_parts: list[pd.Series] = []
    for season in seasons:
        inner_train = train.loc[train["Season"] != season]
        inner_valid = train.loc[train["Season"] == season]
        if inner_train.empty or inner_valid.empty:
            continue
        raw = _fit_outer_raw_probability(config, inner_train, inner_valid, feature_cols)
        oof_parts.append(raw)
        label_parts.append(inner_valid["Label"])

    raw = pd.concat(oof_parts).sort_index() if oof_parts else _fit_outer_raw_probability(config, train, train, feature_cols)
    labels = pd.concat(label_parts).sort_index() if label_parts else train["Label"]
    return fit_gender_calibrator(
        probabilities=raw,
        labels=labels,
        calibration_mode=config.calibration_mode,
        isotonic_min_samples=config.isotonic_min_samples,
    )


def run_gender_replay(config: JIBaseConfig) -> dict:
    dataset = build_ji_dataset(config)
    feature_cols = [column for column in config.resolved_model_features() if column in dataset.columns]
    seasons = sorted(dataset["Season"].unique())
    predictions: list[pd.DataFrame] = []
    by_season_rows: list[dict] = []
    clip_low, clip_high = config.resolved_clip_bounds()

    for test_season in seasons:
        train = dataset.loc[dataset["Season"] != test_season].copy()
        test = dataset.loc[dataset["Season"] == test_season].copy()
        fold_features = [column for column in feature_cols if train[column].notna().any()]
        calibrator = _fit_training_calibrator(config, train, fold_features)
        raw_prob = _fit_outer_raw_probability(config, train, test, fold_features)
        calibrated_prob = pd.Series(np.clip(calibrator.predict(raw_prob), clip_low, clip_high), index=test.index, dtype=float)
        raw_brier = float(((raw_prob - test["Label"]) ** 2).mean())
        calibrated_brier = float(((calibrated_prob - test["Label"]) ** 2).mean())
        by_season_rows.append(
            {
                "season": int(test_season),
                "gender": config.gender,
                "model_family": config.model_family,
                "feature_profile": config.resolved_feature_profile(),
                "rating_profile": config.resolved_rating_profile(),
                "women_quality_profile": config.women_quality_profile,
                "alpha_profile": config.alpha_profile,
                "sidecar_profile": config.sidecar_profile,
                "calibration_mode": config.calibration_mode,
                "selection_objective": config.resolved_selection_objective(),
                "raw_brier": raw_brier,
                "calibrated_brier": calibrated_brier,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "Season": test["Season"],
                    "T1": test["T1"],
                    "T2": test["T2"],
                    "Label": test["Label"],
                    "raw_prob": np.clip(raw_prob, 0.0, 1.0),
                    "calibrated_prob": calibrated_prob,
                    "gender": config.gender,
                }
            )
        )

    by_season = pd.DataFrame(by_season_rows)
    latest_season = int(by_season["season"].max())
    recent_cutoff = latest_season - config.recent_window + 1
    recent = by_season.loc[by_season["season"] >= recent_cutoff]
    return {
        "gender": config.gender,
        "model_family": config.model_family,
        "feature_profile": config.resolved_feature_profile(),
        "rating_profile": config.resolved_rating_profile(),
        "women_quality_profile": config.women_quality_profile,
        "alpha_profile": config.alpha_profile,
        "sidecar_profile": config.sidecar_profile,
        "calibration_mode": config.calibration_mode,
        "selection_objective": config.resolved_selection_objective(),
        "isotonic_min_samples": int(config.isotonic_min_samples),
        "cv_brier_raw": float(by_season["raw_brier"].mean()),
        "cv_brier_calibrated": float(by_season["calibrated_brier"].mean()),
        "latest_season_brier": float(by_season.loc[by_season["season"] == latest_season, "calibrated_brier"].iloc[-1]),
        "recent_window_brier": float(recent["calibrated_brier"].mean()),
        "by_season": by_season,
        "predictions": pd.concat(predictions, ignore_index=True),
    }
