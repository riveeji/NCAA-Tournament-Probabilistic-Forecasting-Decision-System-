from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GoldConfig
from .data import build_gold_dataset, load_gold_team_features
from .models import fit_gender_calibrator, fit_predict_raw
from .overlay import apply_submission_overlay


def parse_submission_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.copy()
    ids = parsed["ID"].astype(str).str.split("_", expand=True)
    parsed["Season"] = pd.to_numeric(ids[0], errors="coerce").astype(int)
    parsed["T1"] = pd.to_numeric(ids[1], errors="coerce").astype(int)
    parsed["T2"] = pd.to_numeric(ids[2], errors="coerce").astype(int)
    return parsed


def build_submission_feature_frame(ids: pd.DataFrame, team_features: pd.DataFrame, config: GoldConfig) -> pd.DataFrame:
    team_feature_names = list(dict.fromkeys(config.resolved_candidate_features()))
    t1 = team_features.rename(columns={"TeamID": "T1", **{col: f"T1_{col}" for col in team_feature_names if col in team_features.columns}})
    t2 = team_features.rename(columns={"TeamID": "T2", **{col: f"T2_{col}" for col in team_feature_names if col in team_features.columns}})
    merged = ids.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")

    engineered_columns: list[str] = []
    for feature in team_feature_names:
        t1_col = f"T1_{feature}"
        t2_col = f"T2_{feature}"
        if t1_col in merged.columns and t2_col in merged.columns:
            diff_col = f"{feature}_diff"
            merged[diff_col] = merged[t1_col] - merged[t2_col]
            engineered_columns.append(diff_col)

    if "T1_SeedNum" in merged.columns and "T2_SeedNum" in merged.columns:
        merged["SeedAbsGap"] = (merged["T1_SeedNum"] - merged["T2_SeedNum"]).abs()
        merged["SeedPairProduct"] = merged["T1_SeedNum"] * merged["T2_SeedNum"]
        engineered_columns.extend(["SeedAbsGap", "SeedPairProduct"])

    for interaction_name, left_feature, right_feature in config.resolved_interactions():
        t1_left = f"T1_{left_feature}"
        t1_right = f"T1_{right_feature}"
        t2_left = f"T2_{left_feature}"
        t2_right = f"T2_{right_feature}"
        if all(column in merged.columns for column in (t1_left, t1_right, t2_left, t2_right)):
            diff_col = f"{interaction_name}_diff"
            merged[diff_col] = (merged[t1_left] * merged[t1_right]) - (merged[t2_left] * merged[t2_right])
            engineered_columns.append(diff_col)

    return merged[["ID", "Pred", "Season", "T1", "T2", *engineered_columns]].copy()


def _fit_full_calibrator(config: GoldConfig, train: pd.DataFrame, feature_cols: list[str]):
    spread_families = {"gold_harry_xgb_spread", "gold_min_xgb_spread", "gold_xgb_spread_light", "gold_spread_control"}
    raw = pd.Series(
        fit_predict_raw(
            config,
            train[feature_cols],
            train["Margin"] if config.model_family in spread_families else train["Label"],
            train[feature_cols],
            target_kind="margin" if config.model_family in spread_families else "label",
        ),
        index=train.index,
        dtype=float,
    )
    return fit_gender_calibrator(probabilities=raw, labels=train["Label"], calibration_mode=config.calibration_mode)


def predict_submission(
    *,
    ids: pd.DataFrame,
    config: GoldConfig,
    apply_overlay: bool = True,
    include_futures: bool = False,
    allow_injury: bool = True,
    allow_sharpen: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict | None]:
    parsed = parse_submission_ids(ids)
    season = int(parsed["Season"].mode().iloc[0])
    train = build_gold_dataset(config)
    train = train.loc[train["Season"] < season].copy()
    team_features = load_gold_team_features(config)
    team_features = team_features.loc[team_features["Season"] == season].copy()
    feature_frame = build_submission_feature_frame(parsed, team_features, config)
    feature_cols = [column for column in config.resolved_model_features() if column in feature_frame.columns and train[column].notna().any()]
    calibrator = _fit_full_calibrator(config, train, feature_cols)
    spread_families = {"gold_harry_xgb_spread", "gold_min_xgb_spread", "gold_xgb_spread_light", "gold_spread_control"}
    raw_prob = pd.Series(
        fit_predict_raw(
            config,
            train[feature_cols],
            train["Margin"] if config.model_family in spread_families else train["Label"],
            feature_frame[feature_cols],
            target_kind="margin" if config.model_family in spread_families else "label",
        ),
        index=feature_frame.index,
        dtype=float,
    )
    clip_low, clip_high = config.resolved_clip_bounds()
    base_prob = np.clip(calibrator.predict(raw_prob), clip_low, clip_high)
    base_submission = pd.DataFrame({"ID": feature_frame["ID"], "Pred": base_prob})

    if not apply_overlay:
        return base_submission, None, None

    adjusted, audit, summary = apply_submission_overlay(
        pd.concat([feature_frame[["ID", "Season", "T1", "T2"]], pd.DataFrame({"Pred": base_prob}, index=feature_frame.index)], axis=1),
        gender=config.gender,
        season=season,
        include_futures=include_futures,
        allow_injury=allow_injury,
        allow_sharpen=allow_sharpen,
    )
    return adjusted, audit, summary
