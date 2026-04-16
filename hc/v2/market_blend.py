from __future__ import annotations

import numpy as np
import pandas as pd


def clip_probs(probabilities: pd.Series | np.ndarray, low: float, high: float) -> pd.Series:
    series = pd.Series(probabilities, copy=False, dtype=float)
    return series.clip(lower=low, upper=high)


def fixed_weight_blend(
    model_prob: pd.Series | np.ndarray,
    market_prob: pd.Series | np.ndarray,
    *,
    model_weight: float,
    market_weight: float,
) -> pd.Series:
    model = pd.Series(model_prob, copy=False, dtype=float)
    market = pd.Series(market_prob, copy=False, dtype=float)
    return (model * model_weight) + (market * market_weight)


def bounded_pull(
    model_prob: pd.Series | np.ndarray,
    market_prob: pd.Series | np.ndarray,
    *,
    max_delta: float,
) -> pd.Series:
    model = pd.Series(model_prob, copy=False, dtype=float)
    market = pd.Series(market_prob, copy=False, dtype=float)
    delta = (market - model).clip(lower=-max_delta, upper=max_delta)
    return model + delta


def apply_market_experiment(
    model_prob: pd.Series,
    market_prob: pd.Series | None,
    *,
    weight: float,
    max_delta: float,
    clip_low: float,
    clip_high: float,
) -> pd.Series:
    if market_prob is None:
        return clip_probs(model_prob, clip_low, clip_high)
    aligned_market = pd.Series(market_prob, index=model_prob.index, dtype=float)
    result = model_prob.copy()
    available = aligned_market.notna()
    if available.any():
        result.loc[available] = fixed_weight_blend(
            model_prob.loc[available],
            bounded_pull(model_prob.loc[available], aligned_market.loc[available], max_delta=max_delta),
            model_weight=1.0 - weight,
            market_weight=weight,
        )
    return clip_probs(result, clip_low, clip_high)
