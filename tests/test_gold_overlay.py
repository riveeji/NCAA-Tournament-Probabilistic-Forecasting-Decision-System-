import pandas as pd

from hc.gold import overlay as overlay_module


def test_gold_overlay_outputs_audit_fields_and_respects_source_priority(monkeypatch):
    predictions = pd.DataFrame(
        [
            {"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60},
            {"ID": "2026_2101_2102", "Season": 2026, "T1": 2101, "T2": 2102, "Pred": 0.45},
        ]
    )

    monkeypatch.setattr(
        overlay_module,
        "load_direct_market_probs",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 1101, "T2": 1102, "market_prob": 0.68, "source_used": "vegas_direct"}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_futures_pairwise_probs",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 2101, "T2": 2102, "market_prob": 0.49, "source_used": "kalshi_futures"}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.015, "confirmed_out": 1}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(predictions, gender="M", season=2026)

    assert {
        "pre_prob",
        "post_prob",
        "delta",
        "source_used",
        "injury_applied",
        "market_applied",
        "sharpen_applied",
    }.issubset(audit.columns)
    assert adjusted["Pred"].between(0.0, 1.0).all()
    assert summary["overlay_submission_only_enabled"] is True
    assert summary["rows"] == 2
    assert summary["changed_rows"] >= 1
    row_2 = audit.loc[audit["ID"] == "2026_2101_2102"].iloc[0]
    assert bool(row_2["market_applied"]) is False


def test_gold_overlay_keeps_women_market_only_without_injury_or_sharpen(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "load_direct_market_probs",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.56, "source_used": "bpi_direct"}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_futures_pairwise_probs",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.70, "source_used": "kalshi_futures"}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(predictions, gender="W", season=2026)

    assert adjusted["Pred"].between(0.0, 1.0).all()
    assert summary["injury_applied_rows"] == 0
    assert bool(audit.iloc[0]["market_applied"]) is True
    assert bool(audit.iloc[0]["injury_applied"]) is False
    assert bool(audit.iloc[0]["sharpen_applied"]) is False


def test_gold_overlay_a_tier_profile_disables_futures_even_when_enabled(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "load_direct_market_probs",
        lambda gender, season: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_futures_pairwise_probs",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.70, "source_used": "kalshi_futures"}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        include_futures=True,
        overlay_source_profile="a_tier_default",
    )

    assert adjusted["Pred"].tolist() == [0.52]
    assert bool(audit.iloc[0]["market_applied"]) is False
    assert summary["overlay_source_profile"] == "a_tier_default"


def test_gold_overlay_direct_only_requires_true_direct_market(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market_candidates",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.61, "source_used": "barttorvik"}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        overlay_source_profile="direct_only",
    )

    assert adjusted["Pred"].tolist() == [0.52]
    assert bool(audit.iloc[0]["market_applied"]) is False
    assert summary["overlay_source_profile"] == "direct_only"


def test_gold_overlay_direct_priority_falls_back_to_projection_when_true_direct_missing(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market_candidates",
        lambda gender, season: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.61, "source_used": "barttorvik"}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        overlay_source_profile="direct_priority",
    )

    assert adjusted["Pred"].iloc[0] != 0.52
    assert bool(audit.iloc[0]["market_applied"]) is True
    assert audit.iloc[0]["source_used"] == "barttorvik"
    assert summary["overlay_source_profile"] == "direct_priority"
