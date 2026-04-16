from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def build_tree_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)),
        ]
    )


def build_lr_regression_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def build_tree_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)),
        ]
    )


def fit_predict_proba(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if y_train.nunique() < 2:
        fallback = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0]))
        fallback.fit(np.zeros((len(y_train), 1)), y_train)
        return fallback.predict_proba(np.zeros((len(x_test), 1)))[:, 1]
    if model_name == "lr":
        model = build_lr_pipeline()
    elif model_name == "tree":
        model = build_tree_pipeline()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def fit_predict_margin(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    linear_alpha: float = 1.0,
) -> np.ndarray:
    if y_train.nunique() < 2:
        return np.full(len(x_test), float(y_train.iloc[0]))
    if model_name == "lr":
        model = build_lr_regression_pipeline(alpha=linear_alpha)
    elif model_name == "tree":
        model = build_tree_regression_pipeline()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    model.fit(x_train, y_train)
    return model.predict(x_test)


def margin_to_probability(margin: pd.Series | np.ndarray, scale: float) -> pd.Series:
    series = pd.Series(margin, copy=False, dtype=float)
    clipped_scale = max(float(scale), 1e-6)
    logits = series / clipped_scale
    return 1.0 / (1.0 + np.exp(-logits))


def optimize_logit_scale(
    margin: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    default_scale: float,
) -> float:
    margin_series = pd.Series(margin, copy=False, dtype=float)
    label_series = pd.Series(labels, copy=False, dtype=float)
    candidates = sorted({float(default_scale), *np.linspace(4.0, 16.0, 49).tolist()})
    best_scale = max(float(default_scale), 1e-6)
    best_brier = float("inf")
    for scale in candidates:
        probs = margin_to_probability(margin_series, scale)
        brier = float(((probs - label_series) ** 2).mean())
        if brier < best_brier:
            best_brier = brier
            best_scale = scale
    return best_scale


@dataclass(slots=True)
class SpreadCalibrationModel:
    mode: str
    scale: float
    isotonic: IsotonicRegression | None = None

    def predict_proba(self, margin: pd.Series | np.ndarray) -> pd.Series:
        base_prob = margin_to_probability(margin, self.scale)
        if self.isotonic is None:
            return base_prob
        calibrated = self.isotonic.predict(base_prob.to_numpy())
        return pd.Series(np.clip(calibrated, 0.0, 1.0), index=base_prob.index)


def fit_spread_calibration(
    *,
    margin_pred: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    calibration_mode: str,
    default_scale: float,
) -> SpreadCalibrationModel:
    if calibration_mode == "basecal":
        return SpreadCalibrationModel(mode=calibration_mode, scale=max(float(default_scale), 1e-6))

    fitted_scale = optimize_logit_scale(margin_pred, labels, default_scale=default_scale)
    if calibration_mode == "gendercal":
        return SpreadCalibrationModel(mode=calibration_mode, scale=fitted_scale)
    if calibration_mode == "monotoniccal":
        raw_prob = margin_to_probability(margin_pred, fitted_scale)
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(raw_prob.to_numpy(), pd.Series(labels, copy=False, dtype=float).to_numpy())
        return SpreadCalibrationModel(mode=calibration_mode, scale=fitted_scale, isotonic=isotonic)
    raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")
