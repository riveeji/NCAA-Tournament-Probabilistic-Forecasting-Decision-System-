from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from .config import GoldConfig


def build_linear_pipeline(config: GoldConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=config.resolved_lr_c(), max_iter=2000, solver="lbfgs")),
        ]
    )


def build_tree_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, random_state=42)),
        ]
    )


def build_spread_pipeline(config: GoldConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=1.0 if config.gender == "M" else 2.0)),
        ]
    )


def build_min_spread_pipeline(config: GoldConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=80 if config.gender == "M" else 60,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    min_child_weight=2.0,
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgb_light_spread_pipeline(config: GoldConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=110 if config.gender == "M" else 85,
                    max_depth=2,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_lambda=3.0,
                    min_child_weight=3.0,
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_harry_spread_pipeline(config: GoldConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=160 if config.gender == "M" else 120,
                    max_depth=2,
                    learning_rate=0.045,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_lambda=3.5,
                    min_child_weight=2.0,
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                    n_jobs=1,
                ),
            ),
        ]
    )


def _build_model(config: GoldConfig) -> Pipeline:
    if config.model_family in {"gold_min_lr", "gold_linear", "gold_harry_lr"}:
        return build_linear_pipeline(config)
    if config.model_family == "gold_tree_control":
        return build_tree_pipeline()
    raise ValueError(f"Unsupported probability model family: {config.model_family}")


def fit_predict_raw(
    config: GoldConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    target_kind: str = "label",
) -> np.ndarray:
    if target_kind == "label":
        if y_train.nunique() < 2:
            fallback = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0]))
            fallback.fit(np.zeros((len(y_train), 1)), y_train)
            return fallback.predict_proba(np.zeros((len(x_test), 1)))[:, 1]
        model = _build_model(config)
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    if config.model_family == "gold_harry_xgb_spread":
        spread_model = build_harry_spread_pipeline(config)
    elif config.model_family == "gold_min_xgb_spread":
        spread_model = build_min_spread_pipeline(config)
    elif config.model_family == "gold_xgb_spread_light":
        spread_model = build_xgb_light_spread_pipeline(config)
    else:
        spread_model = build_spread_pipeline(config)
    spread_model.fit(x_train, y_train)
    margin = spread_model.predict(x_test)
    scale = max(float(pd.Series(y_train).std(ddof=0)), 1.0)
    return 1.0 / (1.0 + np.exp(-(margin / scale)))


class IdentityCalibrator:
    def predict(self, probabilities: pd.Series | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


@dataclass(slots=True)
class IsotonicGenderCalibrator:
    model: IsotonicRegression

    def predict(self, probabilities: pd.Series | np.ndarray) -> np.ndarray:
        raw = np.asarray(probabilities, dtype=float)
        return np.clip(self.model.predict(raw), 0.0, 1.0)


@dataclass(slots=True)
class MonotonicSplineGenderCalibrator:
    interpolator: PchipInterpolator

    def predict(self, probabilities: pd.Series | np.ndarray) -> np.ndarray:
        raw = np.asarray(probabilities, dtype=float)
        return np.clip(np.asarray(self.interpolator(np.clip(raw, 0.0, 1.0)), dtype=float), 0.0, 1.0)


def fit_gender_calibrator(
    *,
    probabilities: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    calibration_mode: str,
) -> IdentityCalibrator | IsotonicGenderCalibrator | MonotonicSplineGenderCalibrator:
    raw = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    usable = np.isfinite(raw) & np.isfinite(y)
    raw = np.clip(raw[usable], 0.0, 1.0)
    y = y[usable]

    if calibration_mode == "none" or len(raw) < 20 or len(np.unique(y)) < 2:
        return IdentityCalibrator()

    if calibration_mode == "isotonic_gender":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw, y)
        return IsotonicGenderCalibrator(model=model)

    if calibration_mode == "monotonic_spline_gender":
        bins = np.unique(np.quantile(raw, np.linspace(0.0, 1.0, 17)))
        if len(bins) < 3:
            return IdentityCalibrator()
        x_vals: list[float] = []
        y_vals: list[float] = []
        for lower, upper in zip(bins[:-1], bins[1:], strict=False):
            mask = (raw >= lower) & (raw <= upper if upper == bins[-1] else raw < upper)
            if not mask.any():
                continue
            x_vals.append(float(raw[mask].mean()))
            y_vals.append(float(y[mask].mean()))
        if len(x_vals) < 3:
            return IdentityCalibrator()
        monotone_y = np.maximum.accumulate(np.clip(y_vals, 0.0, 1.0))
        spline = PchipInterpolator(np.asarray(x_vals, dtype=float), np.asarray(monotone_y, dtype=float), extrapolate=True)
        return MonotonicSplineGenderCalibrator(interpolator=spline)

    raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")
