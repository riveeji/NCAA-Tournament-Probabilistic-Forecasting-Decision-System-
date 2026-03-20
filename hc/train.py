from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from hc.constants import (
    BENCHMARKS_PATH,
    CACHE_DIR,
    CACHE_SCHEMA_VERSION,
    ENABLE_HISTORICAL_SEASON_OVERRIDES,
    FUSION_LOGIC_VERSION,
    HISTORICAL_OVERRIDE_POLICY_VERSION,
    HOLDOUT_YEARS,
    LB_PROXY_SEASONS,
    MARKET_POLICY_BY_PROFILE,
    MARKET_POLICY_VERSION,
    PROFILE_AGGRESSIVE,
    PROFILE_CHOICES,
    PRIMARY_YEARS,
    PUBLIC_ROUTE_VERSION,
    RESULTS_DIR,
    TrainConfig,
)
from hc.data_build import build_all
from hc.features_structured import build_hc_matchups, feature_views as build_feature_views
from hc.fusion import fit_meta_models, generate_final_oof
from hc.legacy_anchor import merge_legacy_anchor_oof
from hc.models_routes import base_oof_cache_path, fit_full_models, generate_base_oof, model_specs_for_gender


def resolve_eval_seasons(frame: pd.DataFrame, years: int) -> list[int]:
    seasons = sorted(pd.to_numeric(frame["Season"], errors="coerce").dropna().astype(int).unique().tolist())
    return seasons[-years:]


def score_oof(final_oof: pd.DataFrame, years: int) -> dict[str, object]:
    eval_seasons = resolve_eval_seasons(final_oof, years)
    season_scores = {}
    for season in eval_seasons:
        fold = final_oof.loc[final_oof["Season"] == season]
        if fold.empty:
            continue
        season_scores[int(season)] = float(brier_score_loss(fold["Label"], fold["FinalProb"]))
    score = float(np.mean(list(season_scores.values()))) if season_scores else float("nan")
    return {"score": score, "eval_seasons": eval_seasons, "season_scores": season_scores}


def score_specific_seasons(final_oof: pd.DataFrame, seasons: list[int] | tuple[int, ...], label: str = "eval_seasons") -> dict[str, object]:
    season_scores = {}
    available_seasons = sorted(pd.to_numeric(final_oof["Season"], errors="coerce").dropna().astype(int).unique().tolist())
    eval_seasons = [int(season) for season in seasons if int(season) in available_seasons]
    for season in eval_seasons:
        fold = final_oof.loc[final_oof["Season"] == season]
        if fold.empty:
            continue
        season_scores[int(season)] = float(brier_score_loss(fold["Label"], fold["FinalProb"]))
    score = float(np.mean(list(season_scores.values()))) if season_scores else float("nan")
    return {"score": score, label: eval_seasons, "season_scores": season_scores}


def holdout_score(final_oof: pd.DataFrame, holdout_years: int = HOLDOUT_YEARS) -> dict[str, object]:
    seasons = resolve_eval_seasons(final_oof, holdout_years)
    season_scores = {}
    for season in seasons:
        fold = final_oof.loc[final_oof["Season"] == season]
        if fold.empty:
            continue
        season_scores[int(season)] = float(brier_score_loss(fold["Label"], fold["FinalProb"]))
    score = float(np.mean(list(season_scores.values()))) if season_scores else float("nan")
    return {"score": score, "holdout_seasons": seasons, "season_scores": season_scores}


def latest_year_score(final_oof: pd.DataFrame) -> dict[str, object]:
    latest = resolve_eval_seasons(final_oof, 1)
    season_scores = {}
    for season in latest:
        fold = final_oof.loc[final_oof["Season"] == season]
        if fold.empty:
            continue
        season_scores[int(season)] = float(brier_score_loss(fold["Label"], fold["FinalProb"]))
    score = float(np.mean(list(season_scores.values()))) if season_scores else float("nan")
    return {"score": score, "eval_seasons": latest, "season_scores": season_scores}


def lb_proxy_score(final_oof: pd.DataFrame) -> dict[str, object]:
    return score_specific_seasons(final_oof, LB_PROXY_SEASONS, label="lb_proxy_seasons")


def _cache_file(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def model_cache_tag(config: TrainConfig) -> str:
    return "_".join(
        [
            config.cache_tag,
            "legacyv8",
            CACHE_SCHEMA_VERSION,
            FUSION_LOGIC_VERSION,
            PUBLIC_ROUTE_VERSION,
            MARKET_POLICY_VERSION,
            HISTORICAL_OVERRIDE_POLICY_VERSION,
        ]
    )


def run_gender_cv(config: TrainConfig, force_rebuild: bool = False) -> dict[str, object]:
    build_all(genders=(config.gender,), text_dim=config.text_dim)
    matchups = build_hc_matchups(
        config.gender,
        market_policy=config.market_policy,
        text_dim=config.text_dim,
        include_text=config.use_text,
        profile=config.profile,
        include_aggressive_public=(config.profile == PROFILE_AGGRESSIVE),
        force_rebuild=force_rebuild,
    )
    views = build_feature_views(
        matchups,
        config.gender,
        text_enabled=config.use_text,
        tabpfn_enabled=config.use_tabpfn,
        include_public_route=(config.profile == PROFILE_AGGRESSIVE),
    )
    specs = model_specs_for_gender(config.gender, views, use_tabpfn=config.use_tabpfn)

    cache_tag = model_cache_tag(config)
    base_path = _cache_file(base_oof_cache_path(cache_tag))
    if base_path.exists() and not force_rebuild:
        base_oof = pd.read_parquet(base_path)
    else:
        base_oof = generate_base_oof(matchups, views, config.gender, specs)
    if "Prob_legacy_anchor" not in base_oof.columns:
        base_oof, legacy_summary = merge_legacy_anchor_oof(base_oof, config.gender, matchups, force_rebuild=force_rebuild)
        if legacy_summary.get("enabled"):
            base_oof.to_parquet(base_path, index=False)
    else:
        legacy_summary = {"enabled": True, "source": "base_cache"}

    final_path = _cache_file(f"final_oof_{cache_tag}.parquet")
    if final_path.exists() and not force_rebuild:
        final_oof = pd.read_parquet(final_path)
        fusion_summary = {}
    else:
        final_oof, fusion_summary = generate_final_oof(base_oof, config.gender)
        final_oof.to_parquet(final_path, index=False)

    lb_window = lb_proxy_score(final_oof)
    holdout = holdout_score(final_oof, HOLDOUT_YEARS)
    latest_holdout = latest_year_score(final_oof)
    summary = {
        "gender": config.gender,
        "years": config.years,
        "market_policy": config.market_policy,
        "profile": config.profile,
        "text_enabled": config.use_text,
        "tabpfn_enabled": config.use_tabpfn,
        "text_dim": config.text_dim,
        "rows": int(len(matchups)),
        "base_oof_path": str(base_path),
        "final_oof_path": str(final_path),
        "available_models": [spec.name for spec in specs],
        "feature_views": {key: len(value) for key, value in views.items()},
        "legacy_anchor": legacy_summary,
        "primary_metric": "lb_window_4y",
        "primary": lb_window,
        "lb_window_4y": lb_window,
        "latest_holdout": latest_holdout,
        "holdout_2y": holdout,
        "reliability_policy": {
            "historical_season_overrides_enabled": ENABLE_HISTORICAL_SEASON_OVERRIDES,
            "cache_tag": cache_tag,
        },
        "fusion": fusion_summary,
    }
    return summary


def append_benchmark_row(run_id: str, summary: dict[str, object]) -> None:
    BENCHMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "gender": summary["gender"],
        "years": summary["years"],
        "market_policy": summary["market_policy"],
        "profile": summary["profile"],
        "text_enabled": summary["text_enabled"],
        "tabpfn_enabled": summary["tabpfn_enabled"],
        "text_dim": summary["text_dim"],
        "primary_metric": summary.get("primary_metric", "lb_window_4y"),
        "primary_score": summary["primary"]["score"],
        "lb_window_4y_score": summary["lb_window_4y"]["score"],
        "latest_holdout_score": summary["latest_holdout"]["score"],
        "holdout_score": summary["holdout_2y"]["score"],
        "holdout_2y_score": summary["holdout_2y"]["score"],
        "rows": summary["rows"],
    }
    frame = pd.DataFrame([row])
    if BENCHMARKS_PATH.exists():
        existing = pd.read_csv(BENCHMARKS_PATH)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(BENCHMARKS_PATH, index=False)


def run_combined_cv(
    years: int,
    genders: tuple[str, ...],
    market_policy: Optional[str],
    profile: str,
    use_text: bool,
    use_tabpfn: bool,
    text_dim: int,
    force_rebuild: bool = False,
) -> dict[str, object]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summaries = {}
    for gender in genders:
        config = TrainConfig(
            gender=gender,
            years=years,
            market_policy=market_policy or MARKET_POLICY_BY_PROFILE[profile][gender],
            profile=profile,
            use_text=use_text,
            use_tabpfn=use_tabpfn,
            text_dim=text_dim,
            quick=(years <= 3),
        )
        summaries[gender] = run_gender_cv(config, force_rebuild=force_rebuild)
        append_benchmark_row(run_id, summaries[gender])

    combined = {
        "run_id": run_id,
        "mode": "cv",
        "years": years,
        "primary_metric": "lb_window_4y_equal_gender_mean",
        "genders": summaries,
    }
    if set(genders) == {"M", "W"}:
        men = summaries["M"]["lb_window_4y"]["score"]
        women = summaries["W"]["lb_window_4y"]["score"]
        combined["equal_gender_mean"] = float((men + women) / 2.0)
        combined["latest_holdout_equal_gender_mean"] = float(
            (summaries["M"]["latest_holdout"]["score"] + summaries["W"]["latest_holdout"]["score"]) / 2.0
        )
        combined["lb_window_4y_equal_gender_mean"] = float(
            (summaries["M"]["lb_window_4y"]["score"] + summaries["W"]["lb_window_4y"]["score"]) / 2.0
        )
        combined["holdout_2y_equal_gender_mean"] = float(
            (summaries["M"]["holdout_2y"]["score"] + summaries["W"]["holdout_2y"]["score"]) / 2.0
        )
        combined["metric_contract"] = {
            "primary_metric": "lb_window_4y_equal_gender_mean",
            "lb_proxy_seasons": list(LB_PROXY_SEASONS),
            "secondary_metrics": [
                "latest_holdout_equal_gender_mean",
                "holdout_2y_equal_gender_mean",
            ],
        }

    summary_path = RESULTS_DIR / f"hc_combined_cv_summary_{run_id}.json"
    summary_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    combined["summary_path"] = str(summary_path)
    return combined


def run_holdout(
    genders: tuple[str, ...],
    market_policy: Optional[str],
    profile: str,
    use_text: bool,
    use_tabpfn: bool,
    text_dim: int,
    force_rebuild: bool = False,
) -> dict[str, object]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summaries = {}
    for gender in genders:
        config = TrainConfig(
            gender=gender,
            years=PRIMARY_YEARS,
            market_policy=market_policy or MARKET_POLICY_BY_PROFILE[profile][gender],
            profile=profile,
            use_text=use_text,
            use_tabpfn=use_tabpfn,
            text_dim=text_dim,
            quick=False,
        )
        summary = run_gender_cv(config, force_rebuild=force_rebuild)
        summaries[gender] = summary["holdout_2y"]
    result = {"run_id": run_id, "mode": "holdout", "genders": summaries}
    if set(genders) == {"M", "W"}:
        result["equal_gender_mean"] = float((summaries["M"]["score"] + summaries["W"]["score"]) / 2.0)
    path = RESULTS_DIR / f"hc_holdout_summary_{run_id}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["summary_path"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HC parallel system.")
    parser.add_argument("--mode", choices=["cv", "holdout"], default="cv")
    parser.add_argument("--years", type=int, default=PRIMARY_YEARS)
    parser.add_argument("--genders", default="MW", help="Subset of genders to run, e.g. M, W, or MW.")
    parser.add_argument("--market-policy", default=None, help="Override market policy for both genders.")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=PROFILE_AGGRESSIVE)
    parser.add_argument("--text", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tabpfn", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--text-dim", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Shortcut for 3-year development runs.")
    args = parser.parse_args()

    genders = tuple(gender for gender in ["M", "W"] if gender in args.genders.upper())
    if not genders:
        raise SystemExit("No valid genders requested.")
    years = 3 if args.quick else args.years
    use_text = args.text != "off"
    use_tabpfn = args.tabpfn == "on"

    if args.mode == "holdout":
        result = run_holdout(
            genders=genders,
            market_policy=args.market_policy,
            profile=args.profile,
            use_text=use_text,
            use_tabpfn=use_tabpfn,
            text_dim=args.text_dim,
            force_rebuild=args.force_rebuild,
        )
        print(f"HC holdout summary written to: {result['summary_path']}")
        return

    result = run_combined_cv(
        years=years,
            genders=genders,
            market_policy=args.market_policy,
            profile=args.profile,
            use_text=use_text,
            use_tabpfn=use_tabpfn,
            text_dim=args.text_dim,
        force_rebuild=args.force_rebuild,
    )
    print(f"HC CV summary written to: {result['summary_path']}")


if __name__ == "__main__":
    main()
