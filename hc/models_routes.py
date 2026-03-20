from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None

try:
    from tabpfn import TabPFNClassifier
except ModuleNotFoundError:
    TabPFNClassifier = None

from hc.constants import MARKET_COVERAGE_THRESHOLD, MARKET_ROUTE_MIN_ROWS, MIN_TRAIN_SEASONS
from zizzii_train import margin_to_prob, safe_clip


@dataclass(frozen=True)
class HCModelSpec:
    name: str
    feature_key: str
    route_group: str
    task: str = "clf"


def resolve_n_jobs() -> int:
    raw = str(os.environ.get("ZIZZII_TREE_N_JOBS", "-1")).strip()
    try:
        value = int(raw)
    except ValueError:
        return -1
    return -1 if value == 0 else value


def make_lr(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=float(c), max_iter=2500, random_state=42)),
        ]
    )


def make_histgb(max_depth: int = 4, max_iter: int = 320) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_depth=max_depth,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def make_et(max_depth: Optional[int] = None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=420,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=resolve_n_jobs(),
    )


def make_xgb_classifier(max_depth: int = 4, n_estimators: int = 360):
    if xgb is None:
        return None
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.82,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )


def make_xgb_margin(max_depth: int = 4, n_estimators: int = 380):
    if xgb is None:
        return None
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.82,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:pseudohubererror",
        random_state=42,
        tree_method="hist",
    )


def make_catboost(iterations: int = 320):
    if CatBoostClassifier is None:
        return None
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.04,
        depth=5,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )


def make_tabpfn():
    if TabPFNClassifier is None:
        return None
    try:
        return TabPFNClassifier(device="cpu")
    except Exception:
        return None


def model_specs_for_gender(gender: str, feature_views: dict[str, list[str]], use_tabpfn: bool) -> list[HCModelSpec]:
    specs: list[HCModelSpec] = []
    if gender == "M":
        if feature_views.get("market_only"):
            specs.extend(
                [
                    HCModelSpec("market_only_lr", "market_only", "market"),
                    HCModelSpec("market_only_histgb", "market_only", "market"),
                ]
            )
        if feature_views.get("market_public"):
            specs.extend(
                [
                    HCModelSpec("market_public_lr", "market_public", "market"),
                ]
            )
        if feature_views.get("market_plus_structured"):
            specs.extend(
                [
                    HCModelSpec("market_plus_stats_lr", "market_plus_structured", "market"),
                    HCModelSpec("market_plus_stats_histgb", "market_plus_structured", "market"),
                    HCModelSpec("spread_margin_xgb", "market_plus_structured", "market", task="margin"),
                ]
            )
        if feature_views.get("stats_fallback"):
            specs.extend(
                [
                    HCModelSpec("stats_fallback_et", "stats_fallback", "stats"),
                    HCModelSpec("stats_fallback_margin", "stats_fallback", "stats", task="margin"),
                ]
            )
    else:
        if feature_views.get("women_minimal"):
            specs.extend(
                [
                    HCModelSpec("women_min_lr", "women_minimal", "stats"),
                    HCModelSpec("women_min_et", "women_minimal", "stats"),
                    HCModelSpec("women_min_xgb", "women_minimal", "stats"),
                ]
            )
        if feature_views.get("women_market"):
            specs.extend(
                [
                    HCModelSpec("women_market_lr", "women_market", "market"),
                    HCModelSpec("women_market_histgb", "women_market", "market"),
                ]
            )
        if feature_views.get("women_public"):
            specs.extend(
                [
                    HCModelSpec("women_public_lr", "women_public", "market"),
                    HCModelSpec("women_public_histgb", "women_public", "market"),
                ]
            )

    if feature_views.get("text_fusion"):
        specs.extend(
            [
                HCModelSpec(f"{gender.lower()}_text_histgb", "text_fusion", "text"),
                HCModelSpec(f"{gender.lower()}_text_catboost", "text_fusion", "text"),
            ]
        )
    if use_tabpfn and feature_views.get("tabpfn"):
        specs.append(HCModelSpec(f"{gender.lower()}_tabpfn", "tabpfn", "tabpfn"))
    return specs


def build_model(spec: HCModelSpec, gender: str):
    if spec.name.endswith("_lr"):
        c = 0.5 if spec.route_group == "market" else 0.8
        if gender == "W":
            c = 0.35 if spec.route_group == "market" else 0.25
        return make_lr(c=c)
    if spec.name.endswith("_histgb"):
        return make_histgb(max_depth=4 if gender == "M" else 3, max_iter=320 if gender == "M" else 260)
    if spec.name.endswith("_et"):
        return make_et(max_depth=9 if gender == "M" else 7)
    if spec.name.endswith("_xgb"):
        return make_xgb_classifier(max_depth=4 if gender == "M" else 3, n_estimators=340 if gender == "M" else 280)
    if spec.task == "margin":
        return make_xgb_margin(max_depth=4 if gender == "M" else 3, n_estimators=360 if gender == "M" else 280)
    if "catboost" in spec.name:
        return make_catboost(iterations=300 if gender == "M" else 240)
    if "tabpfn" in spec.name:
        return make_tabpfn()
    raise ValueError(f"Unhandled HC model spec: {spec}")


def feature_frame(df: pd.DataFrame, feature_views: dict[str, list[str]], feature_key: str) -> pd.DataFrame:
    columns = feature_views.get(feature_key, [])
    if not columns:
        return pd.DataFrame(index=df.index)
    available = [column for column in columns if column in df.columns]
    missing = [column for column in columns if column not in df.columns]
    frame = df[available].copy()
    for column in missing:
        frame[column] = 0.0
    return frame[columns].fillna(0.0)


def column_as_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _restrict_for_route(df: pd.DataFrame, spec: HCModelSpec, gender: str) -> tuple[pd.DataFrame, np.ndarray]:
    valid = np.ones(len(df), dtype=bool)
    if spec.route_group == "market":
        valid = column_as_series(df, "MarketProb").notna().to_numpy()
    elif spec.route_group == "text":
        text_total = column_as_series(df, "TextDocCountTotal", default=0.0).fillna(0.0)
        valid = text_total.to_numpy() >= 2
    return df.loc[valid].copy(), valid


def fit_model(spec: HCModelSpec, x_train: pd.DataFrame, y_train: pd.Series, gender: str):
    model = build_model(spec, gender)
    if model is None or x_train.empty:
        return None
    try:
        model.fit(x_train, y_train)
    except Exception:
        return None
    return model


def predict_model(spec: HCModelSpec, model, x_test: pd.DataFrame, gender: str) -> np.ndarray:
    if model is None:
        return np.full(len(x_test), np.nan, dtype=float)
    if spec.task == "margin":
        try:
            margin = np.asarray(model.predict(x_test), dtype=float)
        except Exception:
            return np.full(len(x_test), np.nan, dtype=float)
        return margin_to_prob(margin, gender)
    try:
        if hasattr(model, "predict_proba"):
            return safe_clip(model.predict_proba(x_test)[:, 1])
        return safe_clip(np.asarray(model.predict(x_test), dtype=float))
    except Exception:
        return np.full(len(x_test), np.nan, dtype=float)


def base_oof_cache_path(cache_tag: str) -> str:
    return f"base_oof_{cache_tag}.parquet"


def generate_base_oof(
    matchups: pd.DataFrame,
    feature_views: dict[str, list[str]],
    gender: str,
    specs: list[HCModelSpec],
) -> pd.DataFrame:
    seasons = sorted(pd.to_numeric(matchups["Season"], errors="coerce").dropna().astype(int).unique().tolist())
    coverage_by_season = {
        season: float(matchups.loc[matchups["Season"] == season, "MarketProb"].notna().mean()) if "MarketProb" in matchups.columns else 0.0
        for season in seasons
    }
    rows: list[pd.DataFrame] = []
    passthrough_cols = [
        column for column in [
            "Season", "T1", "T2", "Label", "MarginLabel",
            "MarketProb", "MarketLogit", "MarketConfidence", "LastSpread", "AbsLastSpread",
            "AbsSeedDiff", "T1BetterSeed", "D_HostLikely", "TourneyRound", "IsRound1Or2",
        ]
        if column in matchups.columns
    ]

    for season in seasons:
        prior = [value for value in seasons if value < season]
        if len(prior) < MIN_TRAIN_SEASONS:
            continue
        train_df = matchups.loc[matchups["Season"] < season].copy()
        test_df = matchups.loc[matchups["Season"] == season].copy()
        if train_df.empty or test_df.empty:
            continue

        row = {column: test_df[column].values for column in passthrough_cols}
        for spec in specs:
            route_threshold = MARKET_COVERAGE_THRESHOLD[gender]
            if spec.route_group == "market" and coverage_by_season.get(season, 0.0) < route_threshold:
                row[f"Prob_{spec.name}"] = np.full(len(test_df), np.nan, dtype=float)
                continue
            train_subset, valid_mask = _restrict_for_route(train_df, spec, gender)
            if len(train_subset) < MARKET_ROUTE_MIN_ROWS[gender] and spec.route_group in {"market", "text", "tabpfn"}:
                row[f"Prob_{spec.name}"] = np.full(len(test_df), np.nan, dtype=float)
                continue
            x_train = feature_frame(train_subset, feature_views, spec.feature_key)
            y_train = train_subset["MarginLabel"] if spec.task == "margin" else train_subset["Label"]
            if spec.route_group == "tabpfn" and x_train.shape[1] > 32:
                x_train = x_train.iloc[:, :32]
            model = fit_model(spec, x_train, y_train, gender)
            test_subset = test_df.copy()
            pred = np.full(len(test_subset), np.nan, dtype=float)
            active_test = np.ones(len(test_subset), dtype=bool)
            if spec.route_group == "market":
                active_test = pd.to_numeric(test_subset.get("MarketProb"), errors="coerce").notna().to_numpy()
            elif spec.route_group == "text":
                active_test = pd.to_numeric(test_subset.get("TextDocCountTotal"), errors="coerce").fillna(0.0).to_numpy() >= 2
            if active_test.any():
                x_test = feature_frame(test_subset.loc[active_test], feature_views, spec.feature_key)
                if spec.route_group == "tabpfn" and x_test.shape[1] > 32:
                    x_test = x_test.iloc[:, :32]
                pred[active_test] = predict_model(spec, model, x_test, gender)
            row[f"Prob_{spec.name}"] = pred
        rows.append(pd.DataFrame(row))

    if not rows:
        return pd.DataFrame(columns=["Season", "Label"])
    return pd.concat(rows, ignore_index=True)


def fit_full_models(matchups: pd.DataFrame, feature_views: dict[str, list[str]], gender: str, specs: list[HCModelSpec]) -> dict[str, object]:
    models = {}
    for spec in specs:
        train_subset, _ = _restrict_for_route(matchups, spec, gender)
        if spec.route_group in {"market", "text", "tabpfn"} and len(train_subset) < MARKET_ROUTE_MIN_ROWS[gender]:
            models[spec.name] = None
            continue
        x_train = feature_frame(train_subset, feature_views, spec.feature_key)
        if spec.route_group == "tabpfn" and x_train.shape[1] > 32:
            x_train = x_train.iloc[:, :32]
        y_train = train_subset["MarginLabel"] if spec.task == "margin" else train_subset["Label"]
        models[spec.name] = fit_model(spec, x_train, y_train, gender)
    return models


def predict_full_models(
    pred_df: pd.DataFrame,
    feature_views: dict[str, list[str]],
    gender: str,
    specs: list[HCModelSpec],
    models: dict[str, object],
) -> pd.DataFrame:
    passthrough_cols = [
        "ID",
        "Season",
        "MarketProb",
        "MarketConfidence",
        "LastSpread",
        "AbsLastSpread",
        "AbsSeedDiff",
        "T1BetterSeed",
        "D_HostLikely",
        "TourneyRound",
        "IsRound1Or2",
    ]
    keep_cols = [column for column in passthrough_cols if column in pred_df.columns]
    out = pred_df.loc[:, keep_cols].copy()
    for spec in specs:
        pred = np.full(len(pred_df), np.nan, dtype=float)
        active = np.ones(len(pred_df), dtype=bool)
        if spec.route_group == "market":
            active = column_as_series(pred_df, "MarketProb").notna().to_numpy()
        elif spec.route_group == "text":
            active = column_as_series(pred_df, "TextDocCountTotal", default=0.0).fillna(0.0).to_numpy() >= 2
        if active.any():
            x_test = feature_frame(pred_df.loc[active], feature_views, spec.feature_key)
            if spec.route_group == "tabpfn" and x_test.shape[1] > 32:
                x_test = x_test.iloc[:, :32]
            pred[active] = predict_model(spec, models.get(spec.name), x_test, gender)
        out[f"Prob_{spec.name}"] = pred
    return out
