import pandas as pd

from hc.v2 import overlay as overlay_module


def test_apply_current_year_overlay_uses_market_and_injury_inputs(monkeypatch):
    predictions = pd.DataFrame(
        [
            {"ID": "2027_1101_1102", "Season": 2027, "T1": 1101, "T2": 1102, "Pred": 0.60},
            {"ID": "2027_3101_3102", "Season": 2027, "T1": 3101, "T2": 3102, "Pred": 0.40},
        ]
    )

    monkeypatch.setattr(
        overlay_module,
        "load_current_year_overlay_probs",
        lambda gender, season: pd.DataFrame(
            [
                {"Season": season, "T1": 1101, "T2": 1102, "overlay_market_prob": 0.70, "overlay_source": "sportsbook"},
                {"Season": season, "T1": 3101, "T2": 3102, "overlay_market_prob": 0.35, "overlay_source": "projection"},
            ]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_current_year_injury_scores",
        lambda gender, season: pd.DataFrame(
            [
                {"Season": season, "TeamID": 1101, "injury_score": 2.0, "confirmed_out_score": 2.0},
                {"Season": season, "TeamID": 1102, "injury_score": 0.0},
                {"Season": season, "TeamID": 3102, "injury_score": 1.0},
            ]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_current_year_overlay(predictions, gender="M", season=2027)

    assert list(adjusted.columns) == ["ID", "Pred"]
    assert {"ID", "BaseProb", "Pred", "overlay_source", "injury_applied", "AbsDelta"}.issubset(audit.columns)
    assert summary.rows == 2
    assert summary.market_rows == 1
    assert summary.injury_rows == 1
    assert summary.total_changed_rows == 1
    assert adjusted["Pred"].between(0.01, 0.99).all()
    assert adjusted.loc[adjusted["ID"] == "2027_1101_1102", "Pred"].iloc[0] < 0.60
    assert summary.mean_abs_delta < 0.03
