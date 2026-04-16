from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pandas as pd

from hc.ji_base.data import NCAA_DATA

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


SEASON_EMB_DIM = 16
SEASON_TOKEN_COLS = ["ResultSign", "MarginClip20", "LocHomeFlag", "LocAwayFlag", "DayNorm"]


def _load_regular_season_compact(gender: str) -> pd.DataFrame:
    frame = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonCompactResults.csv")
    frame["Season"] = frame["Season"].astype(int)
    frame["DayNum"] = frame["DayNum"].astype(int)
    return frame


def _expand_team_view(compact: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": compact["Season"],
            "DayNum": compact["DayNum"],
            "TeamID": compact["WTeamID"],
            "OppTeamID": compact["LTeamID"],
            "Margin": compact["WScore"] - compact["LScore"],
            "Loc": compact["WLoc"],
            "ResultSign": 1.0,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": compact["Season"],
            "DayNum": compact["DayNum"],
            "TeamID": compact["LTeamID"],
            "OppTeamID": compact["WTeamID"],
            "Margin": compact["LScore"] - compact["WScore"],
            "Loc": compact["WLoc"].map({"H": "A", "A": "H"}).fillna("N"),
            "ResultSign": -1.0,
        }
    )
    return pd.concat([winners, losers], ignore_index=True)


def build_team_season_sequences(gender: str) -> pd.DataFrame:
    compact = _load_regular_season_compact(gender)
    team_view = _expand_team_view(compact)
    max_day_by_season = team_view.groupby("Season")["DayNum"].transform("max").replace(0, 1)
    team_view["MarginClip20"] = team_view["Margin"].clip(-20, 20) / 20.0
    team_view["LocHomeFlag"] = (team_view["Loc"] == "H").astype(float)
    team_view["LocAwayFlag"] = (team_view["Loc"] == "A").astype(float)
    team_view["DayNorm"] = team_view["DayNum"] / max_day_by_season

    rows: list[dict[str, object]] = []
    grouped = team_view.sort_values(["Season", "TeamID", "DayNum"]).groupby(["Season", "TeamID"], sort=True)
    for (season, team_id), group in grouped:
        sequence = group[SEASON_TOKEN_COLS].to_numpy(dtype=np.float32)
        rows.append(
            {
                "Season": int(season),
                "TeamID": int(team_id),
                "SequenceLength": int(len(sequence)),
                "GameSequence": sequence,
            }
        )
    return pd.DataFrame(rows)


class _SeasonSequenceEncoder(nn.Module):  # pragma: no cover - exercised through fit function
    def __init__(self, token_dim: int = 5, d_model: int = SEASON_EMB_DIM, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        valid_mask = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return self.out(pooled)


def _pad_sequences(sequence_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = sequence_df["SequenceLength"].to_numpy(dtype=int)
    max_len = int(lengths.max()) if len(lengths) else 1
    batch = np.zeros((len(sequence_df), max_len, len(SEASON_TOKEN_COLS)), dtype=np.float32)
    mask = np.ones((len(sequence_df), max_len), dtype=bool)
    targets = np.zeros(len(sequence_df), dtype=np.float32)

    for idx, row in enumerate(sequence_df.itertuples(index=False)):
        sequence = np.asarray(row.GameSequence, dtype=np.float32)
        seq_len = int(row.SequenceLength)
        batch[idx, :seq_len, :] = sequence
        mask[idx, :seq_len] = False
        targets[idx] = float(np.tanh(np.mean(sequence[:, 0] + sequence[:, 1]))) if seq_len > 0 else 0.0

    return batch, mask, targets


@lru_cache(maxsize=4)
def fit_team_season_encoder(gender: str) -> pd.DataFrame:
    if torch is None:
        raise ImportError("Season encoder experiment requires torch to be installed.")

    sequence_df = build_team_season_sequences(gender)
    if sequence_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "SeasonEmbStrength", "SeasonEmbVector"])

    x_np, mask_np, targets_np = _pad_sequences(sequence_df)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    torch.manual_seed(42)

    train_count = len(sequence_df)
    if train_count < 12:
        val_idx = np.arange(train_count)
        fit_idx = np.arange(train_count)
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_count)
        val_size = max(1, int(round(train_count * 0.15)))
        val_idx = np.sort(indices[:val_size])
        fit_idx = np.sort(indices[val_size:])

    fit_x = torch.as_tensor(x_np[fit_idx], dtype=torch.float32)
    fit_mask = torch.as_tensor(mask_np[fit_idx], dtype=torch.bool)
    fit_y = torch.as_tensor(targets_np[fit_idx], dtype=torch.float32)
    val_x = torch.as_tensor(x_np[val_idx], dtype=torch.float32)
    val_mask = torch.as_tensor(mask_np[val_idx], dtype=torch.bool)
    val_y = torch.as_tensor(targets_np[val_idx], dtype=torch.float32)
    all_x = torch.as_tensor(x_np, dtype=torch.float32)
    all_mask = torch.as_tensor(mask_np, dtype=torch.bool)

    model = _SeasonSequenceEncoder()
    head = nn.Linear(SEASON_EMB_DIM, 1)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=0.004, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = {
        "model": copy.deepcopy(model.state_dict()),
        "head": copy.deepcopy(head.state_dict()),
    }
    best_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(60):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        fit_emb = model(fit_x, fit_mask)
        fit_pred = head(fit_emb).squeeze(-1)
        fit_loss = loss_fn(fit_pred, fit_y)
        fit_loss.backward()
        optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            val_emb = model(val_x, val_mask)
            val_pred = head(val_emb).squeeze(-1)
            val_loss = float(loss_fn(val_pred, val_y).cpu().item())
        if val_loss + 1e-4 < best_loss:
            best_loss = val_loss
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                break

    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        embeddings = model(all_x, all_mask).cpu().numpy().astype(float)

    result = sequence_df[["Season", "TeamID"]].copy()
    result["SeasonEmbStrength"] = embeddings.mean(axis=1)
    result["SeasonEmbVector"] = [embeddings[idx] for idx in range(len(result))]
    return result


def _safe_cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_season_encoder_matchup_features(base_dataset: pd.DataFrame, gender: str) -> pd.DataFrame:
    embeddings = fit_team_season_encoder(gender)
    t1 = embeddings.rename(
        columns={
            "TeamID": "T1",
            "SeasonEmbVector": "T1_SeasonEmbVector",
            "SeasonEmbStrength": "T1_SeasonEmbStrength",
        }
    )
    t2 = embeddings.rename(
        columns={
            "TeamID": "T2",
            "SeasonEmbVector": "T2_SeasonEmbVector",
            "SeasonEmbStrength": "T2_SeasonEmbStrength",
        }
    )
    merged = base_dataset[["Season", "DayNum", "T1", "T2", "Label", "Margin"]].merge(
        t1[["Season", "T1", "T1_SeasonEmbVector", "T1_SeasonEmbStrength"]],
        on=["Season", "T1"],
        how="left",
    )
    merged = merged.merge(
        t2[["Season", "T2", "T2_SeasonEmbVector", "T2_SeasonEmbStrength"]],
        on=["Season", "T2"],
        how="left",
    )

    cos_sim: list[float] = []
    l2_values: list[float] = []
    for row in merged.itertuples(index=False):
        left = getattr(row, "T1_SeasonEmbVector")
        right = getattr(row, "T2_SeasonEmbVector")
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            cos_sim.append(_safe_cos(left, right))
            l2_values.append(float(np.linalg.norm(left - right)))
        else:
            cos_sim.append(0.0)
            l2_values.append(0.0)

    merged["SeasonEmbCosSim"] = cos_sim
    merged["SeasonEmbL2"] = l2_values
    merged["Delta_SeasonEmbStrength"] = (
        pd.to_numeric(merged.get("T1_SeasonEmbStrength"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("T2_SeasonEmbStrength"), errors="coerce").fillna(0.0)
    )
    ordered = [
        "Season",
        "DayNum",
        "T1",
        "T2",
        "Label",
        "Margin",
        "SeasonEmbCosSim",
        "SeasonEmbL2",
        "Delta_SeasonEmbStrength",
    ]
    return merged[ordered].sort_values(["Season", "DayNum", "T1", "T2"]).reset_index(drop=True)
