from __future__ import annotations

import numpy as np
import pandas as pd

from .config import JIBaseConfig
from .data import build_submission_feature_frame
from .models import fit_gender_calibrator, fit_predict_raw


def parse_submission_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.copy()
    ids = parsed["ID"].astype(str).str.split("_", expand=True)
    parsed["Season"] = pd.to_numeric(ids[0], errors="coerce").astype(int)
    parsed["T1"] = pd.to_numeric(ids[1], errors="coerce").astype(int)
    parsed["T2"] = pd.to_numeric(ids[2], errors="coerce").astype(int)
    return parsed


def _fit_full_calibrator(config: JIBaseConfig, train: pd.DataFrame, feature_cols: list[str]):
    seasons = sorted(train["Season"].unique())
    oof_parts: list[pd.Series] = []
    label_parts: list[pd.Series] = []
    for season in seasons:
        inner_train = train.loc[train["Season"] != season]
        inner_valid = train.loc[train["Season"] == season]
        if inner_train.empty or inner_valid.empty:
            continue
        target = inner_train["Margin"] if config.model_family == "JI_spread_xgb" else inner_train["Label"]
        raw = fit_predict_raw(config, inner_train[feature_cols], target, inner_valid[feature_cols])
        oof_parts.append(pd.Series(raw, index=inner_valid.index, dtype=float))
        label_parts.append(inner_valid["Label"])
    raw = pd.concat(oof_parts).sort_index() if oof_parts else pd.Series(dtype=float)
    labels = pd.concat(label_parts).sort_index() if label_parts else pd.Series(dtype=float)
    return fit_gender_calibrator(probabilities=raw, labels=labels, calibration_mode=config.calibration_mode)


def predict_submission(*, ids: pd.DataFrame, train: pd.DataFrame, team_features: pd.DataFrame, config: JIBaseConfig) -> pd.DataFrame:
    parsed = parse_submission_ids(ids)
    frame = build_submission_feature_frame(parsed, team_features, config)
    feature_cols = [column for column in config.resolved_model_features() if column in frame.columns and train[column].notna().any()]
    calibrator = _fit_full_calibrator(config, train, feature_cols)
    target = train["Margin"] if config.model_family == "JI_spread_xgb" else train["Label"]
    raw = fit_predict_raw(config, train[feature_cols], target, frame[feature_cols])
    clip_low, clip_high = config.resolved_clip_bounds()
    calibrated = np.clip(calibrator.predict(raw), clip_low, clip_high)
    return pd.DataFrame({"ID": frame["ID"], "Pred": calibrated})
