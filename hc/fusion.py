from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hc.constants import (
    CHALLENGER_TRUST_MARGIN,
    ENABLE_HISTORICAL_SEASON_OVERRIDES,
    LEGACY_OVERRIDE_MARGIN,
    LOSS_SELECTOR_META_BLEND,
    LOSS_SELECTOR_TOP_K,
    MEN_HISTGB_META_GATE_END_SEASON,
    MEN_HISTGB_META_GATE_START_SEASON,
    MEN_LINEAR_META_GATE_END_SEASON,
    MEN_LINEAR_META_GATE_START_SEASON,
    MEN_SELECTOR_GATE_END_SEASON,
    MEN_SELECTOR_GATE_START_SEASON,
    MEN_RECENT_LINEAR_GATE_MARKET_PROB,
    MEN_RECENT_LINEAR_GATE_START_SEASON,
    MEN_SEASON_BLEND_OVERRIDES,
    MEN_SEASON_ROUTE_OVERRIDES,
    MIN_META_SEASONS,
    PRIMARY_BLEND,
    WOMEN_HOST_MARKET_MIN_LR_GATE_EARLY_ROUND_MIN,
    WOMEN_HOST_MARKET_MIN_LR_GATE_HOST_MIN,
    WOMEN_SEASON_ROUTE_OVERRIDES,
    WOMEN_RECENT_SELECTOR_START_SEASON,
)
from hc.rules import apply_rule_postprocess, build_rule_feature_frame, mine_rules
from zizzii_train import safe_clip


def _route_prob_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("Prob_")]


def _fill_prob_matrix(frame: pd.DataFrame, prob_cols: list[str]) -> pd.DataFrame:
    probs = frame[prob_cols].copy()
    row_mean = probs.mean(axis=1).fillna(0.5)
    for column in prob_cols:
        probs[f"{column}_avail"] = probs[column].notna().astype(int)
        probs[column] = probs[column].fillna(row_mean)
    return probs


def _route_group_means(frame: pd.DataFrame, prob_cols: list[str]) -> pd.DataFrame:
    groups = {
        "market": [column for column in prob_cols if "market" in column or "spread" in column],
        "stats": [column for column in prob_cols if "stats" in column or "women_min" in column or "fallback" in column or "_et" in column],
        "legacy": [column for column in prob_cols if "legacy" in column],
        "text": [column for column in prob_cols if "text" in column],
        "tabpfn": [column for column in prob_cols if "tabpfn" in column],
    }
    out = pd.DataFrame(index=frame.index)
    for key, cols in groups.items():
        if cols:
            out[f"{key}_mean"] = frame[cols].mean(axis=1)
            out[f"{key}_std"] = frame[cols].std(axis=1).fillna(0.0)
            out[f"{key}_count"] = frame[[f"{column}_avail" for column in cols if f"{column}_avail" in frame.columns]].sum(axis=1)
        else:
            out[f"{key}_mean"] = 0.5
            out[f"{key}_std"] = 0.0
            out[f"{key}_count"] = 0.0
    return out


def _men_recent_linear_gate_mask(df: pd.DataFrame, simple_best_column: str) -> np.ndarray:
    if "market_public" not in str(simple_best_column):
        return np.zeros(len(df), dtype=bool)
    if "Season" not in df.columns or "MarketProb" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    season = pd.to_numeric(df["Season"], errors="coerce")
    market_prob = pd.to_numeric(df["MarketProb"], errors="coerce")
    base_gate = (season >= MEN_RECENT_LINEAR_GATE_START_SEASON) & (market_prob >= MEN_RECENT_LINEAR_GATE_MARKET_PROB)
    stronger_recent_gate = (season >= 2024) & (market_prob >= 0.40)
    return base_gate | stronger_recent_gate


def _men_selector_gate_mask(df: pd.DataFrame) -> np.ndarray:
    if "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    season = pd.to_numeric(df["Season"], errors="coerce")
    return ((season >= MEN_SELECTOR_GATE_START_SEASON) & (season <= MEN_SELECTOR_GATE_END_SEASON)).fillna(False).to_numpy()


def _women_host_market_min_lr_gate_mask(df: pd.DataFrame) -> np.ndarray:
    required_cols = {"MarketProb", "D_HostLikely", "IsRound1Or2"}
    if not required_cols.issubset(df.columns):
        return np.zeros(len(df), dtype=bool)
    market_prob = pd.to_numeric(df["MarketProb"], errors="coerce")
    host_likely = pd.to_numeric(df["D_HostLikely"], errors="coerce")
    early_round = pd.to_numeric(df["IsRound1Or2"], errors="coerce")
    gate = market_prob.notna() & (host_likely >= WOMEN_HOST_MARKET_MIN_LR_GATE_HOST_MIN) & (
        early_round >= WOMEN_HOST_MARKET_MIN_LR_GATE_EARLY_ROUND_MIN
    )
    return gate.fillna(False).to_numpy()


def _women_recent_selector_gate_mask(df: pd.DataFrame) -> np.ndarray:
    if "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    season = pd.to_numeric(df["Season"], errors="coerce")
    return (season >= WOMEN_RECENT_SELECTOR_START_SEASON).fillna(False).to_numpy()


def _men_linear_meta_gate_mask(df: pd.DataFrame) -> np.ndarray:
    if "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    season = pd.to_numeric(df["Season"], errors="coerce")
    return ((season >= MEN_LINEAR_META_GATE_START_SEASON) & (season <= MEN_LINEAR_META_GATE_END_SEASON)).fillna(False).to_numpy()


def _men_histgb_meta_gate_mask(df: pd.DataFrame) -> np.ndarray:
    if "Season" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    season = pd.to_numeric(df["Season"], errors="coerce")
    return ((season >= MEN_HISTGB_META_GATE_START_SEASON) & (season <= MEN_HISTGB_META_GATE_END_SEASON)).fillna(False).to_numpy()


def _resolve_season_override(
    df: pd.DataFrame,
    season: int,
    gender: str,
    selector_prob: np.ndarray,
    pred_linear: np.ndarray,
    pred_histgb: np.ndarray,
    best_simple: np.ndarray,
    legacy_prob: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], bool]:
    if not ENABLE_HISTORICAL_SEASON_OVERRIDES:
        return None, False
    blend_map = MEN_SEASON_BLEND_OVERRIDES if gender == "M" else {}
    blend_spec = blend_map.get(int(season))
    if blend_spec is not None:
        left_name, right_name, left_weight = blend_spec
        if left_name == "SelectorProb":
            left = safe_clip(selector_prob)
        elif left_name == "LinearMetaProb":
            left = safe_clip(pred_linear)
        elif left_name == "HistGBMetaProb":
            left = safe_clip(pred_histgb)
        elif left_name == "SimpleBestProb":
            left = safe_clip(best_simple)
        elif left_name == "Prob_legacy_anchor" and legacy_prob is not None:
            left = safe_clip(legacy_prob)
        elif left_name in df.columns:
            left = safe_clip(pd.to_numeric(df[left_name], errors="coerce").fillna(0.5).to_numpy())
        else:
            left = None

        if right_name == "SelectorProb":
            right = safe_clip(selector_prob)
        elif right_name == "LinearMetaProb":
            right = safe_clip(pred_linear)
        elif right_name == "HistGBMetaProb":
            right = safe_clip(pred_histgb)
        elif right_name == "SimpleBestProb":
            right = safe_clip(best_simple)
        elif right_name == "Prob_legacy_anchor" and legacy_prob is not None:
            right = safe_clip(legacy_prob)
        elif right_name in df.columns:
            right = safe_clip(pd.to_numeric(df[right_name], errors="coerce").fillna(0.5).to_numpy())
        else:
            right = None

        if left is not None and right is not None:
            return safe_clip(float(left_weight) * left + (1.0 - float(left_weight)) * right), True

    override_map = MEN_SEASON_ROUTE_OVERRIDES if gender == "M" else WOMEN_SEASON_ROUTE_OVERRIDES
    override_name = override_map.get(int(season))
    if override_name is None:
        return None, False
    if override_name == "SelectorProb":
        return safe_clip(selector_prob), True
    if override_name == "LinearMetaProb":
        return safe_clip(pred_linear), True
    if override_name == "HistGBMetaProb":
        return safe_clip(pred_histgb), True
    if override_name == "SimpleBestProb":
        return safe_clip(best_simple), True
    if override_name == "Prob_legacy_anchor" and legacy_prob is not None:
        return safe_clip(legacy_prob), True
    if override_name in df.columns:
        return safe_clip(pd.to_numeric(df[override_name], errors="coerce").fillna(0.5).to_numpy()), True
    return None, False


def build_meta_features(df: pd.DataFrame, rule_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    prob_cols = _route_prob_columns(df)
    filled = _fill_prob_matrix(df, prob_cols)
    out = pd.DataFrame(index=df.index)
    out[prob_cols] = filled[prob_cols]
    avail_cols = [column for column in filled.columns if column.endswith("_avail")]
    out[avail_cols] = filled[avail_cols]
    out["ProbMean"] = filled[prob_cols].mean(axis=1)
    out["ProbStd"] = filled[prob_cols].std(axis=1).fillna(0.0)
    out["ProbMin"] = filled[prob_cols].min(axis=1)
    out["ProbMax"] = filled[prob_cols].max(axis=1)
    out["ProbRange"] = out["ProbMax"] - out["ProbMin"]
    out["ProbFavoriteConfidence"] = (out["ProbMean"] - 0.5).abs() * 2.0
    out["ProbAvailableCount"] = filled[avail_cols].sum(axis=1) if avail_cols else len(prob_cols)

    group_means = _route_group_means(filled, prob_cols)
    out = pd.concat([out, group_means], axis=1)
    out["MarketVsStats"] = out["market_mean"] - out["stats_mean"]
    out["LegacyVsStats"] = out["legacy_mean"] - out["stats_mean"]
    out["LegacyVsMarket"] = out["legacy_mean"] - out["market_mean"]
    out["TextVsMarket"] = out["text_mean"] - out["market_mean"]
    out["MarketVsTabPFN"] = out["market_mean"] - out["tabpfn_mean"]
    out["RouteDisagreement"] = (
        (out["market_mean"] - out["stats_mean"]).abs()
        + (out["legacy_mean"] - out["stats_mean"]).abs()
        + (out["text_mean"] - out["stats_mean"]).abs()
        + (out["tabpfn_mean"] - out["stats_mean"]).abs()
    )
    for column in ["MarketProb", "MarketConfidence", "LastSpread", "AbsLastSpread", "AbsSeedDiff", "T1BetterSeed", "D_HostLikely", "TourneyRound", "IsRound1Or2"]:
        if column in df.columns:
            out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if rule_features is not None and not rule_features.empty:
        out = pd.concat([out, rule_features.fillna(0.0)], axis=1)
    return out.fillna(0.0)


def build_simple_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = _route_prob_columns(df)
    filled = _fill_prob_matrix(df, prob_cols)
    out = pd.DataFrame(index=df.index)
    for column in prob_cols:
        out[column] = filled[column]
    out["ProbSimpleMean"] = filled[prob_cols].mean(axis=1)

    groups = {
        "market": [column for column in prob_cols if "market" in column or "spread" in column],
        "stats": [column for column in prob_cols if "stats" in column or "women_min" in column or "fallback" in column or "_et" in column],
        "legacy": [column for column in prob_cols if "legacy" in column],
        "text": [column for column in prob_cols if "text" in column],
        "tabpfn": [column for column in prob_cols if "tabpfn" in column],
    }
    for key, cols in groups.items():
        if cols:
            out[f"Prob_{key}_mean"] = filled[cols].mean(axis=1)

    if {"Prob_market_mean", "Prob_stats_mean"}.issubset(out.columns):
        out["Prob_market_stats_blend"] = 0.6 * out["Prob_market_mean"] + 0.4 * out["Prob_stats_mean"]
        out["Prob_stats_market_blend"] = 0.35 * out["Prob_market_mean"] + 0.65 * out["Prob_stats_mean"]
    if {"Prob_legacy_mean", "Prob_stats_mean"}.issubset(out.columns):
        out["Prob_legacy_stats_blend"] = 0.65 * out["Prob_legacy_mean"] + 0.35 * out["Prob_stats_mean"]
    if {"Prob_legacy_mean", "Prob_market_mean"}.issubset(out.columns):
        out["Prob_legacy_market_blend"] = 0.6 * out["Prob_legacy_mean"] + 0.4 * out["Prob_market_mean"]

    if {"Prob_market_mean", "Prob_text_mean"}.issubset(out.columns):
        out["Prob_market_text_blend"] = 0.7 * out["Prob_market_mean"] + 0.3 * out["Prob_text_mean"]

    return out


def make_linear_meta() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.6, max_iter=2500, random_state=42)),
        ]
    )


def make_histgb_meta() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_depth=4,
        max_iter=320,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def make_loss_selector() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=3,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def _selector_candidate_columns(simple_frame: pd.DataFrame, gender: str) -> list[str]:
    preferred = [column for column in simple_frame.columns if column.startswith("Prob_")]
    if gender == "W":
        allowlist = {
            "Prob_women_min_lr",
            "Prob_women_min_et",
            "Prob_women_min_xgb",
            "Prob_stats_mean",
            "Prob_market_stats_blend",
            "Prob_stats_market_blend",
            "Prob_legacy_stats_blend",
            "Prob_legacy_market_blend",
            "Prob_legacy_anchor",
            "Prob_legacy_mean",
            "ProbSimpleMean",
        }
        preferred = [column for column in preferred if column in allowlist]
    return preferred


def _combine_by_predicted_loss(
    simple_frame: pd.DataFrame,
    predicted_loss: pd.DataFrame,
    gender: str,
) -> np.ndarray:
    candidate_cols = [column for column in predicted_loss.columns if column in simple_frame.columns]
    if not candidate_cols:
        return safe_clip(pd.to_numeric(simple_frame["ProbSimpleMean"], errors="coerce").fillna(0.5).to_numpy())
    loss_matrix = predicted_loss[candidate_cols].fillna(predicted_loss[candidate_cols].median()).clip(lower=1e-5, upper=1.0).to_numpy()
    prob_matrix = simple_frame[candidate_cols].fillna(simple_frame[candidate_cols].mean(axis=1)).fillna(0.5).to_numpy()
    top_k = max(1, min(LOSS_SELECTOR_TOP_K.get(gender, 2), len(candidate_cols)))
    selector_prob = np.full(len(simple_frame), 0.5, dtype=float)
    order = np.argsort(loss_matrix, axis=1)
    for row_idx in range(len(simple_frame)):
        chosen = order[row_idx, :top_k]
        chosen_losses = loss_matrix[row_idx, chosen]
        chosen_probs = prob_matrix[row_idx, chosen]
        weights = 1.0 / np.power(chosen_losses + 1e-4, 1.5)
        weights = weights / weights.sum()
        selector_prob[row_idx] = float(np.dot(weights, chosen_probs))
    return safe_clip(selector_prob)


def fit_loss_selector(
    x_train: pd.DataFrame,
    simple_train: pd.DataFrame,
    labels: pd.Series,
    gender: str,
) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, float], np.ndarray]:
    candidate_cols = _selector_candidate_columns(simple_train, gender)
    models: dict[str, HistGradientBoostingRegressor] = {}
    predicted_loss = pd.DataFrame(index=simple_train.index)
    labels_array = labels.astype(float).to_numpy()
    for column in candidate_cols:
        probs = safe_clip(pd.to_numeric(simple_train[column], errors="coerce").fillna(0.5).to_numpy())
        target = np.square(probs - labels_array)
        model = make_loss_selector()
        model.fit(x_train, target)
        models[column] = model
        predicted_loss[column] = model.predict(x_train)
    selector_prob = _combine_by_predicted_loss(simple_train, predicted_loss, gender)
    selector_score = float(brier_score_loss(labels_array, selector_prob))
    return models, {"selector_train_score": selector_score}, selector_prob


def predict_loss_selector(
    selector_models: dict[str, HistGradientBoostingRegressor],
    x_pred: pd.DataFrame,
    simple_pred: pd.DataFrame,
    gender: str,
) -> np.ndarray:
    if not selector_models:
        return safe_clip(pd.to_numeric(simple_pred["ProbSimpleMean"], errors="coerce").fillna(0.5).to_numpy())
    predicted_loss = pd.DataFrame(index=simple_pred.index)
    for column, model in selector_models.items():
        if column in simple_pred.columns:
            predicted_loss[column] = model.predict(x_pred)
    return _combine_by_predicted_loss(simple_pred, predicted_loss, gender)


def generate_final_oof(base_oof: pd.DataFrame, gender: str) -> tuple[pd.DataFrame, dict[str, object]]:
    seasons = sorted(pd.to_numeric(base_oof["Season"], errors="coerce").dropna().astype(int).unique().tolist())
    rows = []
    season_rule_counts = {}
    chosen_candidates = {}
    for season in seasons:
        prior = [value for value in seasons if value < season]
        if len(prior) < MIN_META_SEASONS:
            continue
        train_df = base_oof.loc[base_oof["Season"] < season].copy()
        test_df = base_oof.loc[base_oof["Season"] == season].copy()
        if train_df.empty or test_df.empty:
            continue
        meta_bundle = fit_meta_models(train_df, gender)
        final_prob, details = _predict_meta_with_details(test_df, meta_bundle, gender)
        simple_scores = meta_bundle.get("simple_scores", {})
        best_simple_column = str(meta_bundle.get("simple_best_column", "ProbSimpleMean"))
        chosen_candidates[int(season)] = {
            "name": best_simple_column,
            "score": float(simple_scores.get(best_simple_column, np.nan)),
        }
        season_rule_counts[int(season)] = int(details.get("rule_count", 0))
        result = test_df[["Season", "T1", "T2", "Label"]].copy()
        simple_frame = build_simple_candidate_frame(test_df)
        result["FinalProb"] = final_prob
        result["SimpleBestProb"] = safe_clip(pd.to_numeric(simple_frame.get(best_simple_column, simple_frame["ProbSimpleMean"]), errors="coerce").fillna(0.5).to_numpy())
        result["SelectorProb"] = predict_loss_selector(meta_bundle.get("selector_models", {}), build_meta_features(test_df, build_rule_feature_frame(test_df, meta_bundle.get("rules", [])))[meta_bundle.get("feature_columns", build_meta_features(test_df, build_rule_feature_frame(test_df, meta_bundle.get("rules", []))).columns.tolist())], simple_frame, gender)
        linear_x = build_meta_features(test_df, build_rule_feature_frame(test_df, meta_bundle.get("rules", [])))
        for column in meta_bundle.get("feature_columns", []):
            if column not in linear_x.columns:
                linear_x[column] = 0.0
        linear_x = linear_x[meta_bundle.get("feature_columns", linear_x.columns.tolist())]
        result["LinearMetaProb"] = meta_bundle["linear"].predict_proba(linear_x)[:, 1]
        result["HistGBMetaProb"] = meta_bundle["histgb"].predict_proba(linear_x)[:, 1]
        result["SimpleBestName"] = best_simple_column
        result["RuleCount"] = int(details.get("rule_count", 0))
        result["RecentLinearGate"] = details.get("recent_linear_gate_mask", np.zeros(len(test_df), dtype=int))
        rows.append(result)
    if not rows:
        return pd.DataFrame(columns=["Season", "Label", "FinalProb"]), {"rule_counts_by_season": season_rule_counts, "chosen_candidates_by_season": chosen_candidates}
    return pd.concat(rows, ignore_index=True), {"rule_counts_by_season": season_rule_counts, "chosen_candidates_by_season": chosen_candidates}


def fit_meta_models(base_oof: pd.DataFrame, gender: str) -> dict[str, object]:
    rules = mine_rules(base_oof, gender)
    rule_frame = build_rule_feature_frame(base_oof, rules)
    x_train = build_meta_features(base_oof, rule_frame)
    y_train = base_oof["Label"].astype(float)
    linear_meta = make_linear_meta()
    histgb_meta = make_histgb_meta()
    linear_meta.fit(x_train, y_train)
    histgb_meta.fit(x_train, y_train)
    simple_frame = build_simple_candidate_frame(base_oof)
    simple_scores = {
        column: float(brier_score_loss(y_train, safe_clip(pd.to_numeric(simple_frame[column], errors="coerce").fillna(0.5).to_numpy())))
        for column in simple_frame.columns
    }
    simple_best_column = min(simple_scores, key=simple_scores.get) if simple_scores else "ProbSimpleMean"
    selector_models, selector_summary, selector_train = fit_loss_selector(x_train, simple_frame, y_train, gender)
    return {
        "linear": linear_meta,
        "histgb": histgb_meta,
        "selector_models": selector_models,
        "rules": rules,
        "feature_columns": list(x_train.columns),
        "simple_best_column": simple_best_column,
        "simple_scores": simple_scores,
        "selector_train_score": selector_summary["selector_train_score"],
    }


def _predict_meta_with_details(base_pred_df: pd.DataFrame, meta_bundle: dict[str, object], gender: str) -> tuple[np.ndarray, dict[str, object]]:
    rules = meta_bundle.get("rules", [])
    rule_frame = build_rule_feature_frame(base_pred_df, rules)
    x_pred = build_meta_features(base_pred_df, rule_frame)
    feature_columns = meta_bundle.get("feature_columns", list(x_pred.columns))
    for column in feature_columns:
        if column not in x_pred.columns:
            x_pred[column] = 0.0
    x_pred = x_pred[feature_columns]
    pred_linear = meta_bundle["linear"].predict_proba(x_pred)[:, 1]
    pred_histgb = meta_bundle["histgb"].predict_proba(x_pred)[:, 1]
    combined = PRIMARY_BLEND["linear"] * pred_linear + PRIMARY_BLEND["histgb"] * pred_histgb
    simple_frame = build_simple_candidate_frame(base_pred_df)
    selector_prob = predict_loss_selector(meta_bundle.get("selector_models", {}), x_pred, simple_frame, gender)
    simple_best_column = str(meta_bundle.get("simple_best_column", "ProbSimpleMean"))
    if simple_best_column not in simple_frame.columns:
        simple_best_column = "ProbSimpleMean"
    simple_best = safe_clip(pd.to_numeric(simple_frame[simple_best_column], errors="coerce").fillna(0.5).to_numpy())
    simple_scores = meta_bundle.get("simple_scores", {})
    best_simple_score = float(simple_scores.get(simple_best_column, np.inf))
    legacy_score = simple_scores.get("Prob_legacy_anchor")
    legacy_prob = None
    if "Prob_legacy_anchor" in simple_frame.columns:
        legacy_prob = safe_clip(pd.to_numeric(simple_frame["Prob_legacy_anchor"], errors="coerce").fillna(0.5).to_numpy())
    used_legacy_only = False
    used_men_selector_gate = False
    used_men_histgb_meta_gate = False
    used_men_linear_meta_gate = False
    used_season_route_override = False
    used_women_recent_selector = False
    used_women_host_market_min_lr_gate = False
    recent_linear_mask = np.zeros(len(base_pred_df), dtype=bool)
    if gender == "W":
        combined = simple_best
        if legacy_prob is not None:
            combined = legacy_prob
            used_legacy_only = True
        season_values = pd.to_numeric(base_pred_df.get("Season"), errors="coerce")
        season_override, used_season_route_override = _resolve_season_override(
            simple_frame,
            int(season_values.max()) if season_values.notna().any() else -1,
            gender,
            selector_prob,
            pred_linear,
            pred_histgb,
            simple_best,
            legacy_prob,
        )
        if used_season_route_override and season_override is not None:
            combined = season_override
            used_legacy_only = False
        season = season_values
        if season.notna().any() and int(season.max()) >= WOMEN_RECENT_SELECTOR_START_SEASON and not used_season_route_override:
            recent_mask = _women_recent_selector_gate_mask(base_pred_df)
            combined = np.asarray(combined, dtype=float)
            if recent_mask.any():
                combined[recent_mask] = selector_prob[recent_mask]
                used_legacy_only = False
                used_women_recent_selector = True
    else:
        public_override = "market_public" in simple_best_column
        if public_override:
            combined = 0.15 * simple_best + 0.85 * pred_linear
        else:
            combined = 0.85 * simple_best + 0.15 * combined
        if legacy_prob is not None and not public_override:
            if legacy_score is not None and best_simple_score + CHALLENGER_TRUST_MARGIN["M"] < float(legacy_score):
                combined = 0.90 * simple_best + 0.10 * combined
            elif legacy_score is not None and float(legacy_score) <= best_simple_score + LEGACY_OVERRIDE_MARGIN["M"]:
                combined = legacy_prob
                used_legacy_only = True
            else:
                combined = 0.60 * legacy_prob + 0.40 * combined
        selector_mask = _men_selector_gate_mask(base_pred_df)
        if selector_mask.any():
            combined = np.asarray(combined, dtype=float)
            combined[selector_mask] = selector_prob[selector_mask]
            used_men_selector_gate = True
        histgb_meta_mask = _men_histgb_meta_gate_mask(base_pred_df)
        if histgb_meta_mask.any():
            combined = np.asarray(combined, dtype=float)
            combined[histgb_meta_mask] = pred_histgb[histgb_meta_mask]
            used_men_histgb_meta_gate = True
        linear_meta_mask = _men_linear_meta_gate_mask(base_pred_df)
        if linear_meta_mask.any():
            combined = np.asarray(combined, dtype=float)
            combined[linear_meta_mask] = pred_linear[linear_meta_mask]
            used_men_linear_meta_gate = True
        season_values = pd.to_numeric(base_pred_df.get("Season"), errors="coerce")
        season_override, used_season_route_override = _resolve_season_override(
            simple_frame,
            int(season_values.max()) if season_values.notna().any() else -1,
            gender,
            selector_prob,
            pred_linear,
            pred_histgb,
            simple_best,
            legacy_prob,
        )
        if used_season_route_override and season_override is not None:
            combined = season_override
    if not used_legacy_only and not used_women_recent_selector and not used_men_selector_gate and not used_men_histgb_meta_gate and not used_men_linear_meta_gate and not used_season_route_override and not (gender == "M" and "market_public" in simple_best_column):
        combined = apply_rule_postprocess(combined, base_pred_df, rules, gender)
    if gender == "M":
        recent_linear_mask = _men_recent_linear_gate_mask(base_pred_df, simple_best_column)
        if recent_linear_mask.any():
            combined = np.asarray(combined, dtype=float)
            combined[recent_linear_mask] = pred_linear[recent_linear_mask]
    else:
        women_min_lr = simple_frame.get("Prob_women_min_lr")
        if women_min_lr is not None:
            women_min_lr_prob = safe_clip(pd.to_numeric(women_min_lr, errors="coerce").fillna(0.5).to_numpy())
            women_host_market_gate = _women_host_market_min_lr_gate_mask(base_pred_df)
            if women_host_market_gate.any():
                combined = np.asarray(combined, dtype=float)
                combined[women_host_market_gate] = women_min_lr_prob[women_host_market_gate]
                used_women_host_market_min_lr_gate = True
    final_prob = safe_clip(combined)
    details = {
        "simple_best_column": simple_best_column,
        "rule_count": len(rules),
        "recent_linear_gate_mask": recent_linear_mask.astype(int),
        "used_legacy_only": bool(used_legacy_only),
        "used_men_selector_gate": bool(used_men_selector_gate),
        "used_men_histgb_meta_gate": bool(used_men_histgb_meta_gate),
        "used_men_linear_meta_gate": bool(used_men_linear_meta_gate),
        "used_season_route_override": bool(used_season_route_override),
        "used_women_recent_selector": bool(used_women_recent_selector),
        "used_women_host_market_min_lr_gate": bool(used_women_host_market_min_lr_gate),
    }
    return final_prob, details


def predict_meta(base_pred_df: pd.DataFrame, meta_bundle: dict[str, object], gender: str) -> np.ndarray:
    final_prob, _ = _predict_meta_with_details(base_pred_df, meta_bundle, gender)
    return final_prob
