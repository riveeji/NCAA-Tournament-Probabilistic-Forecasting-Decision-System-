import pandas as pd

from hc.ji_base import JIBaseOverlayConfig
from hc.ji_base import overlay as overlay_module


def test_ji_base_overlay_outputs_expected_audit_fields(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(
            [{"Season": season, "T1": 1101, "T2": 1102, "market_prob": 0.68, "source_used": "sportsbook"}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.015, "confirmed_out": 1}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=JIBaseOverlayConfig(gender="M"),
    )

    assert {
        "ID",
        "pre_prob",
        "post_prob",
        "delta",
        "source_used",
        "market_applied",
        "injury_applied",
        "injury_mode",
        "injury_shift_abs",
    } == set(audit.columns)
    assert adjusted["Pred"].between(0.0, 1.0).all()
    assert summary["market_applied_rows"] == 1
    assert summary["injury_applied_rows"] == 1
    assert summary["injury_mode"] == "team_confirmed_gate"
    assert "sharpen_applied_rows" not in summary
    assert "futures_enabled" not in summary


def test_ji_base_overlay_keeps_women_market_only_without_injury(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(
            [{"Season": season, "T1": 3101, "T2": 3102, "market_prob": 0.56, "source_used": "barttorvik"}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 3101, "injury_shift": -0.015, "confirmed_out": 1}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="W",
        season=2026,
        config=JIBaseOverlayConfig(gender="W", allow_injury=False),
    )

    assert adjusted["Pred"].between(0.0, 1.0).all()
    assert summary["market_applied_rows"] == 1
    assert summary["injury_applied_rows"] == 0
    assert bool(audit.iloc[0]["market_applied"]) is True
    assert bool(audit.iloc[0]["injury_applied"]) is False


def test_conservative_injury_profile_reduces_men_overlay_shift(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.05, "confirmed_out": 1}]
        ),
    )

    baseline_adjusted, _, _ = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(gender="M", injury_cap=0.02),
    )
    conservative_adjusted, _, _ = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(gender="M", injury_cap=0.01),
    )

    baseline_prob = float(baseline_adjusted.iloc[0]["Pred"])
    conservative_prob = float(conservative_adjusted.iloc[0]["Pred"])

    assert conservative_prob > baseline_prob
    assert abs(conservative_prob - 0.60) < abs(baseline_prob - 0.60)


def test_strict_confirmed_injury_gate_requires_higher_confirmed_out_threshold(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )

    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.05, "confirmed_out": 1}]
        ),
    )
    no_apply_adjusted, no_apply_audit, no_apply_summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_min_confirmed_out=2,
        ),
    )

    assert float(no_apply_adjusted.iloc[0]["Pred"]) == 0.60
    assert bool(no_apply_audit.iloc[0]["injury_applied"]) is False
    assert no_apply_summary["injury_applied_rows"] == 0

    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.05, "confirmed_out": 2}]
        ),
    )
    apply_adjusted, apply_audit, apply_summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_min_confirmed_out=2,
        ),
    )

    assert float(apply_adjusted.iloc[0]["Pred"]) != 0.60
    assert bool(apply_audit.iloc[0]["injury_applied"]) is True
    assert apply_summary["injury_applied_rows"] == 1


def test_abs_shift_gate_blocks_small_shift_even_when_confirmed_out_threshold_is_met(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.06, "confirmed_out": 4}]
        ),
    )

    blocked_adjusted, blocked_audit, blocked_summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_min_confirmed_out=4,
            injury_min_abs_shift=0.08,
        ),
    )

    assert float(blocked_adjusted.iloc[0]["Pred"]) == 0.60
    assert bool(blocked_audit.iloc[0]["injury_applied"]) is False
    assert blocked_summary["injury_applied_rows"] == 0

    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.09, "confirmed_out": 4}]
        ),
    )

    applied_adjusted, applied_audit, applied_summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_min_confirmed_out=4,
            injury_min_abs_shift=0.08,
        ),
    )

    assert float(applied_adjusted.iloc[0]["Pred"]) != 0.60
    assert bool(applied_audit.iloc[0]["injury_applied"]) is True
    assert applied_summary["injury_applied_rows"] == 1


def test_player_level_injury_v2_falls_back_to_team_level_when_player_table_missing(monkeypatch):
    predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_player_level_injury_adjustments",
        lambda season: pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out_count", "high_impact_out_count"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.05, "confirmed_out": 4}]
        ),
    )

    adjusted, audit, summary = overlay_module.apply_submission_overlay(
        predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_mode="player_level_v2",
            injury_min_confirmed_out=4,
        ),
    )

    assert float(adjusted.iloc[0]["Pred"]) != 0.60
    assert bool(audit.iloc[0]["injury_applied"]) is True
    assert audit.iloc[0]["injury_mode"] == "team_confirmed_gate_fallback"
    assert audit.iloc[0]["injury_shift_abs"] > 0.0
    assert summary["injury_mode"] == "team_confirmed_gate_fallback"


def test_player_level_injury_v2_uses_player_table_and_keeps_women_unchanged(monkeypatch):
    men_predictions = pd.DataFrame(
        [{"ID": "2026_1101_1102", "Season": 2026, "T1": 1101, "T2": 1102, "Pred": 0.60}]
    )
    women_predictions = pd.DataFrame(
        [{"ID": "2026_3101_3102", "Season": 2026, "T1": 3101, "T2": 3102, "Pred": 0.52}]
    )

    monkeypatch.setattr(
        overlay_module,
        "_load_direct_market",
        lambda gender, season, profile: pd.DataFrame(columns=["Season", "T1", "T2", "market_prob", "source_used"]),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_player_level_injury_adjustments",
        lambda season: pd.DataFrame(
            [{"Season": season, "TeamID": 1101, "injury_shift": -0.12, "confirmed_out_count": 1, "high_impact_out_count": 1}]
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "load_men_injury_adjustments",
        lambda season: pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out"]),
    )

    men_adjusted, men_audit, men_summary = overlay_module.apply_submission_overlay(
        men_predictions,
        gender="M",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="M",
            overlay_source_profile="direct_only",
            injury_mode="player_level_v2",
            injury_min_confirmed_out=4,
        ),
    )
    women_adjusted, women_audit, women_summary = overlay_module.apply_submission_overlay(
        women_predictions,
        gender="W",
        season=2026,
        config=overlay_module.JIBaseOverlayConfig(
            gender="W",
            overlay_source_profile="direct_only",
            allow_injury=False,
            injury_mode="player_level_v2",
        ),
    )

    assert float(men_adjusted.iloc[0]["Pred"]) != 0.60
    assert bool(men_audit.iloc[0]["injury_applied"]) is True
    assert men_audit.iloc[0]["injury_mode"] == "player_level_v2"
    assert men_summary["injury_mode"] == "player_level_v2"

    assert float(women_adjusted.iloc[0]["Pred"]) == 0.52
    assert bool(women_audit.iloc[0]["injury_applied"]) is False
    assert women_audit.iloc[0]["injury_mode"] == "none"
    assert women_summary["injury_applied_rows"] == 0
