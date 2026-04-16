from __future__ import annotations

import numpy as np
import pandas as pd

from hc.ji_base.models import fit_predict_raw

from .config import NextArchConfig
from .data import build_next_arch_dataset
from .models import fit_predict_next_arch_raw


def _to_logit(probabilities: pd.Series | np.ndarray) -> pd.Series:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)
    return pd.Series(np.log(probs / (1.0 - probs)), dtype=float)


def _build_hybrid_baseline_logits(config: NextArchConfig, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    base_config = config.base_config()
    base_feature_cols = [column for column in base_config.resolved_model_features() if column in train.columns]
    train_oof = pd.Series(index=train.index, dtype=float)
    for season in sorted(train["Season"].unique()):
        inner_train = train.loc[train["Season"] != season]
        inner_valid = train.loc[train["Season"] == season]
        fold_features = [column for column in base_feature_cols if inner_train[column].notna().any()]
        raw_prob = fit_predict_raw(base_config, inner_train[fold_features], inner_train["Label"], inner_valid[fold_features])
        train_oof.loc[inner_valid.index] = _to_logit(raw_prob).to_numpy()

    outer_features = [column for column in base_feature_cols if train[column].notna().any()]
    test_prob = fit_predict_raw(base_config, train[outer_features], train["Label"], test[outer_features])
    test_logit = _to_logit(test_prob)
    test_logit.index = test.index
    return train_oof.fillna(0.0), test_logit


def run_next_arch_gender_replay(config: NextArchConfig) -> dict:
    dataset = build_next_arch_dataset(config)
    requested_feature_cols = list(config.resolved_model_features())
    seasons = sorted(dataset["Season"].unique())
    predictions: list[pd.DataFrame] = []
    by_season_rows: list[dict] = []
    clip_low, clip_high = config.resolved_clip_bounds()

    for test_season in seasons:
        train = dataset.loc[dataset["Season"] != test_season].copy()
        test = dataset.loc[dataset["Season"] == test_season].copy()
        if config.experiment_name in {"tabr_hybrid_v1", "tabr_feature_fusion_v1", "gender_specific_stacker_v1"}:
            train_baseline_logit, test_baseline_logit = _build_hybrid_baseline_logits(config, train, test)
            train["BaselineLogit"] = train_baseline_logit
            test["BaselineLogit"] = test_baseline_logit
        fold_features = [column for column in requested_feature_cols if column in train.columns and train[column].notna().any()]
        raw_prob = fit_predict_next_arch_raw(config, train[fold_features], train["Label"], test[fold_features])
        calibrated_prob = pd.Series(np.clip(raw_prob, clip_low, clip_high), index=test.index, dtype=float)
        raw_brier = float(((raw_prob - test["Label"]) ** 2).mean())
        calibrated_brier = float(((calibrated_prob - test["Label"]) ** 2).mean())
        by_season_rows.append(
            {
                "season": int(test_season),
                "gender": config.gender,
                "model_family": config.experiment_name,
                "feature_profile": config.experiment_name,
                "rating_profile": "n/a",
                "women_quality_profile": "n/a",
                "alpha_profile": "n/a",
                "sidecar_profile": "none",
                "calibration_mode": "none",
                "selection_objective": "total_cv_brier_calibrated",
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
        "model_family": config.experiment_name,
        "feature_profile": config.experiment_name,
        "rating_profile": "n/a",
        "women_quality_profile": "n/a",
        "alpha_profile": "n/a",
        "sidecar_profile": "none",
        "calibration_mode": "none",
        "selection_objective": "total_cv_brier_calibrated",
        "isotonic_min_samples": 0,
        "cv_brier_raw": float(by_season["raw_brier"].mean()),
        "cv_brier_calibrated": float(by_season["calibrated_brier"].mean()),
        "latest_season_brier": float(by_season.loc[by_season["season"] == latest_season, "calibrated_brier"].iloc[-1]),
        "recent_window_brier": float(recent["calibrated_brier"].mean()),
        "by_season": by_season,
        "predictions": pd.concat(predictions, ignore_index=True),
    }
