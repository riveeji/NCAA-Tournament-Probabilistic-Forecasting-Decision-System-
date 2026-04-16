from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from .config import JIBaseConfig

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class MarginProbabilityMapper:
    residual_scale: float

    @classmethod
    def fit(cls, *, predicted_margin: np.ndarray, actual_margin: np.ndarray) -> "MarginProbabilityMapper":
        residuals = np.asarray(actual_margin, dtype=float) - np.asarray(predicted_margin, dtype=float)
        scale = float(np.nanstd(residuals, ddof=0))
        if not np.isfinite(scale) or scale < 1.0:
            scale = 1.0
        return cls(residual_scale=scale)

    def predict(self, predicted_margin: np.ndarray | pd.Series) -> np.ndarray:
        values = np.asarray(predicted_margin, dtype=float) / max(self.residual_scale, 1e-6)
        probs = 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))
        return np.clip(probs.astype(float), 0.0, 1.0)


class IdentityCalibrator:
    def predict(self, probabilities: pd.Series | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


class NeuralObliviousDecisionEnsemble(nn.Module):  # pragma: no cover - exercised through fit_predict_raw
    def __init__(self, input_dim: int, *, num_trees: int = 4, depth: int = 2):
        super().__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.feature_logits = nn.Parameter(torch.zeros(num_trees, depth, input_dim))
        self.thresholds = nn.Parameter(torch.zeros(num_trees, depth))
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, depth))
        self.leaf_logits = nn.Parameter(torch.zeros(num_trees, 2**depth))
        leaf_bits = torch.tensor(
            [[(leaf_index >> bit) & 1 for bit in range(depth)] for leaf_index in range(2**depth)],
            dtype=torch.float32,
        )
        self.register_buffer("leaf_bits", leaf_bits)
        nn.init.normal_(self.feature_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.thresholds, mean=0.0, std=0.02)
        nn.init.constant_(self.log_temperatures, 0.0)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selectors = torch.softmax(self.feature_logits, dim=-1)
        selected = torch.einsum("bf,tdf->btd", x, selectors)
        temperatures = torch.exp(self.log_temperatures).clamp(min=0.1, max=10.0)
        gates = torch.sigmoid((selected - self.thresholds.unsqueeze(0)) * temperatures.unsqueeze(0))
        leaf_bits = self.leaf_bits.unsqueeze(0).unsqueeze(0)
        path_prob = torch.where(leaf_bits > 0.5, gates.unsqueeze(2), 1.0 - gates.unsqueeze(2)).prod(dim=-1)
        tree_logits = (path_prob * self.leaf_logits.unsqueeze(0)).sum(dim=-1)
        return tree_logits.mean(dim=1)


def build_spread_xgb_pipeline(config: JIBaseConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=220 if config.gender == "M" else 180,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_lambda=3.0,
                    min_child_weight=2.0,
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_lr_control_pipeline(config: JIBaseConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=config.resolved_lr_c(), max_iter=2000, solver="lbfgs")),
        ]
    )


def build_lgb_control_pipeline() -> Pipeline:
    if LGBMClassifier is None:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, random_state=42)),
            ]
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=220,
                    learning_rate=0.04,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def _fit_predict_node_control(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if torch is None:
        raise ImportError("JI_node_control requires torch to be installed.")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train_np = scaler.fit_transform(imputer.fit_transform(x_train)).astype(np.float32)
    x_test_np = scaler.transform(imputer.transform(x_test)).astype(np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    torch.manual_seed(42)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)

    x_train_tensor = torch.as_tensor(x_train_np)
    y_train_tensor = torch.as_tensor(y_train_np)
    x_test_tensor = torch.as_tensor(x_test_np)

    model = NeuralObliviousDecisionEnsemble(x_train_tensor.shape[1], num_trees=4, depth=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    best_loss = float("inf")
    epochs_without_improvement = 0
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_tensor)
        loss = loss_fn(logits, y_train_tensor)
        loss.backward()
        optimizer.step()
        current_loss = float(loss.detach().cpu().item())
        if current_loss + 1e-4 < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 6:
                break

    model.eval()
    with torch.no_grad():
        logits = model(x_test_tensor)
        return torch.sigmoid(logits).cpu().numpy().astype(float)


def fit_predict_raw(config: JIBaseConfig, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if config.model_family == "JI_spread_xgb":
        model = build_spread_xgb_pipeline(config)
        model.fit(x_train, y_train)
        mapper = MarginProbabilityMapper.fit(predicted_margin=model.predict(x_train), actual_margin=y_train.to_numpy())
        return mapper.predict(model.predict(x_test))
    if config.model_family == "JI_lr_control":
        model = build_lr_control_pipeline(config)
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    if config.model_family == "JI_node_control":
        return _fit_predict_node_control(x_train, y_train, x_test)
    if LGBMClassifier is None:
        model = build_lgb_control_pipeline()
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train_imputed, y_train)
    return model.predict_proba(x_test_imputed)[:, 1]


def fit_gender_calibrator(
    *,
    probabilities: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    calibration_mode: str,
    isotonic_min_samples: int = 20,
):
    raw = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    usable = np.isfinite(raw) & np.isfinite(y)
    raw = np.clip(raw[usable], 0.0, 1.0)
    y = y[usable]
    if calibration_mode == "none" or len(raw) < isotonic_min_samples or len(np.unique(y)) < 2:
        return IdentityCalibrator()
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(raw, y)
    return model
