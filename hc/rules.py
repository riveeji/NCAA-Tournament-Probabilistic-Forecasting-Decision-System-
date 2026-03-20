from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from hc.constants import MEN_RULE_COLUMNS, RULE_MAX_COUNT, RULE_MIN_EDGE, RULE_MIN_SUPPORT, WOMEN_RULE_COLUMNS


@dataclass(frozen=True)
class Rule:
    name: str
    gender: str
    columns: tuple[str, ...]
    thresholds: tuple[float, ...]
    operators: tuple[str, ...]
    direction: int
    support: float
    target_mean: float
    edge: float
    floor_prob: float


def _threshold_grid(column: str, gender: str) -> list[float]:
    fixed = {
        "MarketProb": [0.80, 0.85, 0.90, 0.94, 0.97, 0.985],
        "LastSpread": [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 30.0],
        "AbsLastSpread": [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 30.0],
        "AbsSeedDiff": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "D_HostLikely": [0.5],
        "TourneyRound": [1.0, 2.0, 3.0],
        "IsRound1Or2": [0.5],
        "D_DR": [0.3, 0.6, 1.0],
        "D_DefRtg_z": [0.3, 0.6, 1.0],
        "D_Elo": [35.0, 60.0, 90.0, 120.0],
        "D_Recent30EffNetRtg_z": [0.4, 0.8, 1.2],
        "H2HMargin": [2.0, 4.0, 8.0],
    }
    return fixed.get(column, [])


def _apply_rule_mask(df: pd.DataFrame, columns: tuple[str, ...], operators: tuple[str, ...], thresholds: tuple[float, ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for column, op, threshold in zip(columns, operators, thresholds):
        raw_values = df.get(column)
        if raw_values is None or np.isscalar(raw_values):
            values = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            values = pd.to_numeric(raw_values, errors="coerce")
        if op == ">=":
            mask &= values.ge(threshold)
        elif op == "<=":
            mask &= values.le(threshold)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask.fillna(False)


def mine_rules(train_df: pd.DataFrame, gender: str) -> list[Rule]:
    candidate_columns = MEN_RULE_COLUMNS if gender == "M" else WOMEN_RULE_COLUMNS
    usable = [column for column in candidate_columns if column in train_df.columns]
    if not usable or "Label" not in train_df.columns:
        return []
    global_mean = float(pd.to_numeric(train_df["Label"], errors="coerce").mean())
    min_support = RULE_MIN_SUPPORT[gender]
    min_edge = RULE_MIN_EDGE[gender]
    candidates: list[Rule] = []

    for column in usable:
        for threshold in _threshold_grid(column, gender):
            for op in (">=", "<="):
                mask = _apply_rule_mask(train_df, (column,), (op,), (threshold,))
                support = float(mask.mean())
                if support < min_support or support > 0.90:
                    continue
                target_mean = float(train_df.loc[mask, "Label"].mean())
                edge = abs(target_mean - global_mean)
                if not np.isfinite(edge) or edge < min_edge:
                    continue
                direction = 1 if target_mean >= global_mean else -1
                floor_prob = float(np.clip(target_mean, 0.01, 0.999))
                candidates.append(
                    Rule(
                        name=f"{column}_{op}_{threshold:g}",
                        gender=gender,
                        columns=(column,),
                        thresholds=(float(threshold),),
                        operators=(op,),
                        direction=direction,
                        support=support,
                        target_mean=target_mean,
                        edge=edge,
                        floor_prob=floor_prob,
                    )
                )

    combo_bases = [column for column in usable if column in {"MarketProb", "AbsSeedDiff", "D_HostLikely", "TourneyRound", "LastSpread", "AbsLastSpread", "D_DR", "D_DefRtg_z", "D_Elo"}]
    for left, right in combinations(combo_bases, 2):
        left_thresholds = _threshold_grid(left, gender)[:3]
        right_thresholds = _threshold_grid(right, gender)[:3]
        for left_threshold in left_thresholds:
            for right_threshold in right_thresholds:
                for left_op, right_op in ((">=", ">="), (">=", "<=")):
                    mask = _apply_rule_mask(train_df, (left, right), (left_op, right_op), (left_threshold, right_threshold))
                    support = float(mask.mean())
                    if support < min_support or support > 0.80:
                        continue
                    target_mean = float(train_df.loc[mask, "Label"].mean())
                    edge = abs(target_mean - global_mean)
                    if not np.isfinite(edge) or edge < min_edge:
                        continue
                    direction = 1 if target_mean >= global_mean else -1
                    candidates.append(
                        Rule(
                            name=f"{left}_{left_op}_{left_threshold:g}__{right}_{right_op}_{right_threshold:g}",
                            gender=gender,
                            columns=(left, right),
                            thresholds=(float(left_threshold), float(right_threshold)),
                            operators=(left_op, right_op),
                            direction=direction,
                            support=support,
                            target_mean=target_mean,
                            edge=edge,
                            floor_prob=float(np.clip(target_mean, 0.01, 0.999)),
                        )
                    )

    candidates.sort(key=lambda rule: (rule.edge * np.sqrt(rule.support)), reverse=True)
    deduped: list[Rule] = []
    seen = set()
    for rule in candidates:
        key = (rule.columns, rule.operators, rule.thresholds, rule.direction)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
        if len(deduped) >= RULE_MAX_COUNT[gender]:
            break
    return deduped


def build_rule_feature_frame(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    pos_count = np.zeros(len(df), dtype=float)
    neg_count = np.zeros(len(df), dtype=float)
    pos_edge = np.zeros(len(df), dtype=float)
    neg_edge = np.zeros(len(df), dtype=float)
    for idx, rule in enumerate(rules):
        mask = _apply_rule_mask(df, rule.columns, rule.operators, rule.thresholds).astype(float)
        name = f"Rule_{idx:02d}"
        out[name] = mask * rule.direction
        if rule.direction > 0:
            pos_count += mask.to_numpy()
            pos_edge = np.maximum(pos_edge, mask.to_numpy() * rule.edge)
        else:
            neg_count += mask.to_numpy()
            neg_edge = np.maximum(neg_edge, mask.to_numpy() * rule.edge)
    out["RulePositiveCount"] = pos_count
    out["RuleNegativeCount"] = neg_count
    out["RulePositiveEdgeMax"] = pos_edge
    out["RuleNegativeEdgeMax"] = neg_edge
    out["RuleNetCount"] = pos_count - neg_count
    out["RuleNetEdge"] = pos_edge - neg_edge
    return out


def apply_rule_postprocess(prob: np.ndarray, df: pd.DataFrame, rules: list[Rule], gender: str) -> np.ndarray:
    adjusted = np.asarray(prob, dtype=float).copy()
    for rule in rules[: min(8, len(rules))]:
        if rule.edge < RULE_MIN_EDGE[gender] + 0.02:
            continue
        mask = _apply_rule_mask(df, rule.columns, rule.operators, rule.thresholds).to_numpy()
        if not mask.any():
            continue
        if rule.direction > 0 and rule.floor_prob >= 0.80:
            adjusted[mask] = np.maximum(adjusted[mask], min(0.999, rule.floor_prob))
        elif rule.direction < 0 and rule.floor_prob <= 0.20:
            adjusted[mask] = np.minimum(adjusted[mask], max(0.001, rule.floor_prob))
    return adjusted
