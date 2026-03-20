from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"
RESULTS_DIR = ROOT / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import build_team_features
from zizzii_train import safe_clip


HOST_POD_MAP = {
    1: {"round1": {16}, "round2": {8, 9}},
    2: {"round1": {15}, "round2": {7, 10}},
    3: {"round1": {14}, "round2": {6, 11}},
    4: {"round1": {13}, "round2": {5, 12}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest and apply a women-only greedy postprocess layer."
    )
    parser.add_argument(
        "--oof-path",
        default="",
        help="Optional women OOF CSV. Defaults to the latest results/oof_W_*.csv.",
    )
    parser.add_argument(
        "--submission",
        default="submission_stage2.csv",
        help="Submission file to postprocess for 2026 inference.",
    )
    parser.add_argument(
        "--output",
        default="submission_stage2_women_greedy.csv",
        help="Output submission path for the greedy women ticket.",
    )
    parser.add_argument(
        "--seasons",
        default="2021,2022,2023,2024,2025",
        help="Backtest seasons, comma-separated.",
    )
    parser.add_argument(
        "--prob-column",
        default="Prob_lr_core",
        help="OOF probability column to treat as the women baseline.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip OOF grid search and use the recommended config directly.",
    )
    return parser.parse_args()


def latest_file(pattern: str) -> Path:
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No files matched {pattern} under {RESULTS_DIR}")
    return candidates[0]


def parse_seed_meta(gender: str = "W") -> pd.DataFrame:
    seeds = pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")
    seeds["SeedRegion"] = seeds["Seed"].astype(str).str[0]
    seeds["SeedNum"] = pd.to_numeric(seeds["Seed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return seeds[["Season", "TeamID", "SeedRegion", "SeedNum"]].dropna(subset=["SeedNum"]).copy()


def attach_team_meta(frame: pd.DataFrame, team_feats: pd.DataFrame, seed_meta: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "Season",
        "TeamID",
        "Recent30EffTOVPct",
        "RecentEffTOVPct",
        "TOVPct",
        "Recent30EffAst",
        "Recent30EffTO",
        "RecentEffAst",
        "RecentEffTO",
        "Ast",
        "TO",
    ]
    feat = team_feats[[col for col in keep_cols if col in team_feats.columns]].copy()
    t1 = feat.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    t2 = feat.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    out = frame.merge(t1, on=["Season", "T1"], how="left").merge(t2, on=["Season", "T2"], how="left")

    s1 = seed_meta.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"})
    s2 = seed_meta.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"})
    out = out.merge(s1, on=["Season", "T1"], how="left").merge(s2, on=["Season", "T2"], how="left")

    def best_tov(prefix: str) -> pd.Series:
        for column in [f"{prefix}Recent30EffTOVPct", f"{prefix}RecentEffTOVPct", f"{prefix}TOVPct"]:
            if column in out.columns:
                return pd.to_numeric(out[column], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    def ast_to_ratio(prefix: str) -> pd.Series:
        candidate_pairs = [
            (f"{prefix}Recent30EffAst", f"{prefix}Recent30EffTO"),
            (f"{prefix}RecentEffAst", f"{prefix}RecentEffTO"),
            (f"{prefix}Ast", f"{prefix}TO"),
        ]
        for ast_col, to_col in candidate_pairs:
            if ast_col in out.columns and to_col in out.columns:
                ast = pd.to_numeric(out[ast_col], errors="coerce")
                to = pd.to_numeric(out[to_col], errors="coerce").replace(0, np.nan)
                return ast / to
        return pd.Series(np.nan, index=out.index, dtype=float)

    out["T1_BestTOVPct"] = best_tov("T1_")
    out["T2_BestTOVPct"] = best_tov("T2_")
    out["D_BestTOVPct"] = out["T1_BestTOVPct"] - out["T2_BestTOVPct"]

    out["T1_AstTo"] = ast_to_ratio("T1_")
    out["T2_AstTo"] = ast_to_ratio("T2_")
    out["D_AstTo"] = out["T1_AstTo"] - out["T2_AstTo"]
    return out


def is_host_matchup(row: pd.Series, side: str) -> tuple[bool, str]:
    host_seed = row[f"{side}_SeedNum"]
    opp_seed = row["T2_SeedNum"] if side == "T1" else row["T1_SeedNum"]
    host_region = row[f"{side}_SeedRegion"]
    opp_region = row["T2_SeedRegion"] if side == "T1" else row["T1_SeedRegion"]

    if pd.isna(host_seed) or pd.isna(opp_seed) or pd.isna(host_region) or pd.isna(opp_region):
        return False, ""
    host_seed = int(host_seed)
    opp_seed = int(opp_seed)
    if host_seed not in HOST_POD_MAP:
        return False, ""
    if str(host_region) != str(opp_region):
        return False, ""

    pod = HOST_POD_MAP[host_seed]
    if opp_seed in pod["round1"]:
        return True, "round1"
    if opp_seed in pod["round2"]:
        return True, "round2"
    return False, ""


def apply_host_boost(prob: np.ndarray, frame: pd.DataFrame, round1_boost: float, round2_boost: float) -> np.ndarray:
    adjusted = np.asarray(prob, dtype=float).copy()
    for idx, row in frame.iterrows():
        t1_host, t1_round = is_host_matchup(row, "T1")
        t2_host, t2_round = is_host_matchup(row, "T2")
        if t1_host and not t2_host:
            adjusted[idx] += round1_boost if t1_round == "round1" else round2_boost
        elif t2_host and not t1_host:
            adjusted[idx] -= round1_boost if t2_round == "round1" else round2_boost
    return safe_clip(adjusted)


def extreme_push(prob: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0:
        return safe_clip(prob)
    clipped = safe_clip(prob)
    high_target = np.interp(clipped, [0.85, 0.90, 0.95, 0.975], [0.92, 0.96, 0.99, 0.995])
    high_mask = clipped >= 0.85
    low_ref = 1.0 - clipped
    low_target = np.interp(low_ref, [0.85, 0.90, 0.95, 0.975], [0.92, 0.96, 0.99, 0.995])
    low_mask = clipped <= 0.15

    adjusted = clipped.copy()
    adjusted[high_mask] = clipped[high_mask] + intensity * (high_target[high_mask] - clipped[high_mask])
    adjusted[low_mask] = clipped[low_mask] - intensity * ((1.0 - low_target[low_mask]) - clipped[low_mask])
    return safe_clip(adjusted)


def apply_tossup_control(prob: np.ndarray, frame: pd.DataFrame, tossup_boost: float) -> np.ndarray:
    if tossup_boost <= 0:
        return safe_clip(prob)

    adjusted = np.asarray(prob, dtype=float).copy()
    mask = (adjusted >= 0.45) & (adjusted <= 0.55)
    if not mask.any():
        return safe_clip(adjusted)

    tov_edge = -pd.to_numeric(frame["D_BestTOVPct"], errors="coerce").fillna(0.0).to_numpy()
    ast_edge = pd.to_numeric(frame["D_AstTo"], errors="coerce").fillna(0.0).to_numpy()
    edge = np.tanh((tov_edge / 0.03) * 0.75 + ast_edge * 0.25)
    adjusted[mask] = adjusted[mask] + tossup_boost * edge[mask]
    return safe_clip(adjusted)


def apply_postprocess(prob: np.ndarray, frame: pd.DataFrame, config: dict[str, float]) -> np.ndarray:
    adjusted = safe_clip(prob)
    adjusted = apply_host_boost(
        adjusted,
        frame,
        round1_boost=float(config["host_round1"]),
        round2_boost=float(config["host_round2"]),
    )
    adjusted = extreme_push(adjusted, float(config["power_intensity"]))
    adjusted = apply_tossup_control(adjusted, frame, float(config["tossup_boost"]))
    return safe_clip(adjusted)


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((y - p) ** 2))


def parse_seasons(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def grid_search_backtest(frame: pd.DataFrame, base_prob: np.ndarray, label: np.ndarray) -> tuple[dict[str, float], float]:
    baseline = brier(label, base_prob)
    best_score = baseline
    best_config = {"host_round1": 0.0, "host_round2": 0.0, "power_intensity": 0.0, "tossup_boost": 0.0}

    for host_r1, host_r2, power_intensity, tossup_boost in product(
        (0.0, 0.006, 0.010, 0.015),
        (0.0, 0.003, 0.006, 0.010),
        (0.0, 0.35, 0.70, 1.0),
        (0.0, 0.010, 0.020, 0.030),
    ):
        config = {
            "host_round1": host_r1,
            "host_round2": host_r2,
            "power_intensity": power_intensity,
            "tossup_boost": tossup_boost,
        }
        score = brier(label, apply_postprocess(base_prob, frame, config))
        if score < best_score:
            best_score = score
            best_config = config

    return best_config, best_score


def load_backtest_frame(oof_path: Path, seasons: Iterable[int], prob_column: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    oof = pd.read_csv(oof_path)
    oof = oof[oof["Season"].isin(list(seasons))].copy()
    if prob_column not in oof.columns:
        raise SystemExit(f"{prob_column} not found in {oof_path}")

    team_feats = build_team_features("W")
    seed_meta = parse_seed_meta("W")
    frame = attach_team_meta(oof[["Season", "T1", "T2"]].copy(), team_feats, seed_meta)
    return frame, oof[prob_column].to_numpy(dtype=float), oof["Label"].to_numpy(dtype=float)


def apply_to_submission(submission_path: Path, output_path: Path, config: dict[str, float]) -> None:
    submission = pd.read_csv(submission_path)
    ids = submission["ID"].str.split("_", expand=True)
    submission = submission.copy()
    submission["Season"] = ids[0].astype(int)
    submission["T1"] = ids[1].astype(int)
    submission["T2"] = ids[2].astype(int)
    women = submission[submission["T1"] >= 3000].copy()
    if women.empty:
        raise SystemExit("No women rows found in submission.")

    team_feats = build_team_features("W")
    seed_meta = parse_seed_meta("W")
    frame = attach_team_meta(women[["Season", "T1", "T2"]].copy(), team_feats, seed_meta)
    women["Pred"] = apply_postprocess(women["Pred"].to_numpy(dtype=float), frame, config)

    merged = submission.merge(women[["ID", "Pred"]], on="ID", how="left", suffixes=("", "_women"))
    mask = merged["Pred_women"].notna()
    merged.loc[mask, "Pred"] = merged.loc[mask, "Pred_women"]
    merged = merged.drop(columns=["Pred_women"])
    merged.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    oof_path = Path(args.oof_path) if args.oof_path else latest_file("oof_W_*.csv")
    seasons = parse_seasons(args.seasons)

    if args.skip_backtest:
        best_config = {"host_round1": 0.015, "host_round2": 0.006, "power_intensity": 0.0, "tossup_boost": 0.03}
        best_score = np.nan
        baseline = np.nan
    else:
        frame, base_prob, label = load_backtest_frame(oof_path, seasons, args.prob_column)
        baseline = brier(label, base_prob)
        best_config, best_score = grid_search_backtest(frame, base_prob, label)
        print("Women greedy backtest")
        print(f"OOF file:    {oof_path}")
        print(f"Seasons:     {seasons}")
        print(f"Baseline:    {baseline:.5f}")
        print(f"Best score:  {best_score:.5f}")
        print(f"Improvement: {baseline - best_score:+.5f}")
        print("Best config:")
        print(json.dumps(best_config, indent=2))

    apply_to_submission(Path(args.submission), Path(args.output), best_config)
    print(f"\nSaved women-greedy submission -> {args.output}")


if __name__ == "__main__":
    main()
