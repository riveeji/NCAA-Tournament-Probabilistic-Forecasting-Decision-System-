import pandas as pd

from hc.v2.market_blend import apply_market_experiment, bounded_pull, clip_probs


def test_clip_probs_respects_bounds():
    clipped = clip_probs(pd.Series([0.0, 0.5, 1.0]), 0.02, 0.98)
    assert clipped.tolist() == [0.02, 0.5, 0.98]


def test_bounded_pull_caps_market_distance():
    result = bounded_pull(pd.Series([0.2]), pd.Series([0.8]), max_delta=0.05)
    assert result.iloc[0] == 0.25


def test_apply_market_experiment_uses_available_market_only():
    model = pd.Series([0.4, 0.6], index=[0, 1])
    market = pd.Series([0.9, None], index=[0, 1])
    result = apply_market_experiment(model, market, weight=0.2, max_delta=0.1, clip_low=0.02, clip_high=0.98)
    assert round(result.iloc[0], 4) == 0.42
    assert round(result.iloc[1], 4) == 0.6
