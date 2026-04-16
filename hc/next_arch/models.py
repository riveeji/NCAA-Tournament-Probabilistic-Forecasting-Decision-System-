from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import NextArchConfig

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class _TabRStyleModel(nn.Module):  # pragma: no cover - exercised through fit function
    def __init__(self, input_dim: int, d_model: int = 16, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class _PairwiseMLP(nn.Module):  # pragma: no cover - exercised through fit function
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _preprocess_numeric(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_np = imputer.fit_transform(x_train)
    test_np = imputer.transform(x_test)
    train_np = scaler.fit_transform(train_np).astype(np.float32)
    test_np = scaler.transform(test_np).astype(np.float32)
    return train_np, test_np


def _fit_predict_tabr_v1(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if torch is None:
        raise ImportError("TabR-style experiment requires torch to be installed.")

    x_train_np, x_test_np = _preprocess_numeric(x_train, x_test)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    torch.manual_seed(42)

    train_count = len(x_train_np)
    if train_count < 12:
        val_idx = np.arange(train_count)
        fit_idx = np.arange(train_count)
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_count)
        val_size = max(1, int(round(train_count * 0.15)))
        val_idx = np.sort(indices[:val_size])
        fit_idx = np.sort(indices[val_size:])

    fit_x = torch.as_tensor(x_train_np[fit_idx], dtype=torch.float32)
    fit_y = torch.as_tensor(y_train_np[fit_idx], dtype=torch.float32)
    val_x = torch.as_tensor(x_train_np[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_train_np[val_idx], dtype=torch.float32)
    test_x = torch.as_tensor(x_test_np, dtype=torch.float32)

    model = _TabRStyleModel(input_dim=x_train_np.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(80):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_logits = model(fit_x)
        fit_loss = loss_fn(fit_logits, fit_y)
        fit_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(loss_fn(val_logits, val_y).cpu().item())
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x)
        return torch.sigmoid(test_logits).cpu().numpy().astype(float)


def _fit_predict_pairwise_ranking_v1(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if torch is None:
        raise ImportError("Pairwise ranking experiment requires torch to be installed.")

    x_train_np, x_test_np = _preprocess_numeric(x_train, x_test)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    torch.manual_seed(42)

    train_count = len(x_train_np)
    if train_count < 12:
        val_idx = np.arange(train_count)
        fit_idx = np.arange(train_count)
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_count)
        val_size = max(1, int(round(train_count * 0.15)))
        val_idx = np.sort(indices[:val_size])
        fit_idx = np.sort(indices[val_size:])

    fit_x = torch.as_tensor(x_train_np[fit_idx], dtype=torch.float32)
    fit_y = torch.as_tensor(y_train_np[fit_idx], dtype=torch.float32)
    val_x = torch.as_tensor(x_train_np[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_train_np[val_idx], dtype=torch.float32)
    test_x = torch.as_tensor(x_test_np, dtype=torch.float32)

    model = _PairwiseMLP(input_dim=x_train_np.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(120):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_logits = model(fit_x)
        fit_loss = loss_fn(fit_logits, fit_y)
        fit_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(loss_fn(val_logits, val_y).cpu().item())
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x)
        return torch.sigmoid(test_logits).cpu().numpy().astype(float)


def _fit_predict_tabr_hybrid_v1(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if torch is None:
        raise ImportError("TabR-style hybrid experiment requires torch to be installed.")

    feature_columns = [column for column in x_train.columns if column != "BaselineLogit"]
    x_train_np, x_test_np = _preprocess_numeric(x_train[feature_columns], x_test[feature_columns])
    baseline_train = np.asarray(pd.to_numeric(x_train["BaselineLogit"], errors="coerce").fillna(0.0), dtype=np.float32)
    baseline_test = np.asarray(pd.to_numeric(x_test["BaselineLogit"], errors="coerce").fillna(0.0), dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    torch.manual_seed(42)

    train_count = len(x_train_np)
    if train_count < 12:
        val_idx = np.arange(train_count)
        fit_idx = np.arange(train_count)
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_count)
        val_size = max(1, int(round(train_count * 0.15)))
        val_idx = np.sort(indices[:val_size])
        fit_idx = np.sort(indices[val_size:])

    fit_x = torch.as_tensor(x_train_np[fit_idx], dtype=torch.float32)
    fit_base = torch.as_tensor(baseline_train[fit_idx], dtype=torch.float32)
    fit_y = torch.as_tensor(y_train_np[fit_idx], dtype=torch.float32)
    val_x = torch.as_tensor(x_train_np[val_idx], dtype=torch.float32)
    val_base = torch.as_tensor(baseline_train[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_train_np[val_idx], dtype=torch.float32)
    test_x = torch.as_tensor(x_test_np, dtype=torch.float32)
    test_base = torch.as_tensor(baseline_test, dtype=torch.float32)

    model = _TabRStyleModel(input_dim=x_train_np.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(80):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_delta = model(fit_x)
        fit_loss = loss_fn(fit_base + fit_delta, fit_y)
        fit_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_delta = model(val_x)
            val_loss = float(loss_fn(val_base + val_delta, val_y).cpu().item())
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_delta = model(test_x)
        return torch.sigmoid(test_base + test_delta).cpu().numpy().astype(float)


def _fit_predict_tabr_feature_fusion_v1(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if torch is None:
        raise ImportError("TabR-style feature-fusion experiment requires torch to be installed.")

    x_train_np, x_test_np = _preprocess_numeric(x_train, x_test)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    torch.manual_seed(42)

    train_count = len(x_train_np)
    if train_count < 12:
        val_idx = np.arange(train_count)
        fit_idx = np.arange(train_count)
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_count)
        val_size = max(1, int(round(train_count * 0.15)))
        val_idx = np.sort(indices[:val_size])
        fit_idx = np.sort(indices[val_size:])

    fit_x = torch.as_tensor(x_train_np[fit_idx], dtype=torch.float32)
    fit_y = torch.as_tensor(y_train_np[fit_idx], dtype=torch.float32)
    val_x = torch.as_tensor(x_train_np[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_train_np[val_idx], dtype=torch.float32)
    test_x = torch.as_tensor(x_test_np, dtype=torch.float32)

    model = _TabRStyleModel(input_dim=x_train_np.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(80):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_logits = model(fit_x)
        fit_loss = loss_fn(fit_logits, fit_y)
        fit_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(loss_fn(val_logits, val_y).cpu().item())
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x)
        return torch.sigmoid(test_logits).cpu().numpy().astype(float)


def _fit_predict_graph_lr(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def _fit_predict_gender_specific_stacker_v1(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=0.7, max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def fit_predict_next_arch_raw(config: NextArchConfig, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    if config.experiment_name == "tabr_v1":
        return _fit_predict_tabr_v1(x_train, y_train, x_test)
    if config.experiment_name == "pairwise_ranking_v1":
        return _fit_predict_pairwise_ranking_v1(x_train, y_train, x_test)
    if config.experiment_name == "season_encoder_transformer_v1":
        return _fit_predict_graph_lr(x_train, y_train, x_test)
    if config.experiment_name == "tabr_hybrid_v1":
        return _fit_predict_tabr_hybrid_v1(x_train, y_train, x_test)
    if config.experiment_name == "tabr_feature_fusion_v1":
        return _fit_predict_tabr_feature_fusion_v1(x_train, y_train, x_test)
    if config.experiment_name == "graph_static_embedding_v1":
        return _fit_predict_graph_lr(x_train, y_train, x_test)
    if config.experiment_name == "gender_specific_stacker_v1":
        return _fit_predict_gender_specific_stacker_v1(x_train, y_train, x_test)
    raise KeyError(f"Unknown next-arch experiment: {config.experiment_name}")
