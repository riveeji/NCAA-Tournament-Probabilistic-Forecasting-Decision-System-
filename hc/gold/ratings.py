from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from zizzii_features import build_team_features

ROOT = Path(__file__).resolve().parents[2]
NCAA_DATA = ROOT / "ncaa-data"
EXTERNAL_DATA = ROOT / "external-data"


def _read_compact(gender: str) -> pd.DataFrame:
    frame = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonCompactResults.csv")
    frame["Season"] = frame["Season"].astype(int)
    frame["DayNum"] = pd.to_numeric(frame["DayNum"], errors="coerce").fillna(0).astype(int)
    return frame.sort_values(["Season", "DayNum"]).reset_index(drop=True)


def _read_detailed(gender: str) -> pd.DataFrame:
    frame = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonDetailedResults.csv")
    frame["Season"] = frame["Season"].astype(int)
    frame["DayNum"] = pd.to_numeric(frame["DayNum"], errors="coerce").fillna(0).astype(int)
    return frame.sort_values(["Season", "DayNum"]).reset_index(drop=True)


def _season_percentile_strength(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series(numeric)
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=numeric.index)
    if higher_is_better:
        rank = numeric.rank(method="average", ascending=True, pct=True)
    else:
        rank = numeric.rank(method="average", ascending=False, pct=True)
    return (rank - 0.5) * 2.0


def _weighted_strength_consensus(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    numerators: list[pd.Series] = []
    denominators: list[pd.Series] = []
    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        strength = frame.groupby("Season")[column].transform(lambda s: _season_percentile_strength(s, higher_is_better=True))
        numerators.append(strength.fillna(0.0) * float(weight))
        denominators.append(strength.notna().astype(float) * float(weight))
    if not numerators:
        return pd.Series(np.nan, index=frame.index)
    numerator = pd.concat(numerators, axis=1).sum(axis=1, skipna=True)
    denominator = pd.concat(denominators, axis=1).sum(axis=1, skipna=True).replace(0.0, np.nan)
    return numerator / denominator


def _load_men_injury_adjustments() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(EXTERNAL_DATA.glob("MRotoWireInjuries_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "TeamID" not in frame.columns:
            continue
        season = pd.to_numeric(frame.get("Season"), errors="coerce")
        team = pd.to_numeric(frame.get("TeamID"), errors="coerce")
        severity = pd.to_numeric(frame.get("Severity"), errors="coerce").fillna(0.0)
        is_out = pd.to_numeric(frame.get("IsOut"), errors="coerce").fillna(0.0)
        usable = pd.DataFrame({"Season": season, "TeamID": team, "Severity": severity, "IsOut": is_out}).dropna(subset=["Season", "TeamID"])
        if usable.empty:
            continue
        usable["Season"] = usable["Season"].astype(int)
        usable["TeamID"] = usable["TeamID"].astype(int)
        usable["confirmed_out"] = ((usable["IsOut"] >= 1.0) & (usable["Severity"] >= 2.0)).astype(int)
        usable["injury_shift"] = np.where(usable["confirmed_out"] == 1, -0.04 * usable["Severity"], 0.0)
        grouped = usable.groupby(["Season", "TeamID"], as_index=False)[["injury_shift", "confirmed_out"]].sum()
        rows.extend(grouped.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=["Season", "TeamID", "injury_shift", "confirmed_out"])
    return pd.DataFrame(rows)


def _compute_harry_tournament_rank(ratings: pd.DataFrame) -> pd.Series:
    seed_strength = ratings.groupby("Season")["SeedNum"].transform(lambda s: _season_percentile_strength(s, higher_is_better=False))
    opp_quality_strength = ratings.groupby("Season")["OpponentQualityScore"].transform(
        lambda s: _season_percentile_strength(s, higher_is_better=True)
    )
    quality_wins_strength = ratings.groupby("Season")["QualityWins"].transform(
        lambda s: _season_percentile_strength(s, higher_is_better=True)
    )
    numerator = (
        seed_strength.fillna(0.0) * 0.20
        + opp_quality_strength.fillna(0.0) * 0.50
        + quality_wins_strength.fillna(0.0) * 0.30
    )
    denominator = (
        seed_strength.notna().astype(float) * 0.20
        + opp_quality_strength.notna().astype(float) * 0.50
        + quality_wins_strength.notna().astype(float) * 0.30
    ).replace(0.0, np.nan)
    return numerator / denominator


def _expected_prob(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _compute_elo_variant(
    compact: pd.DataFrame,
    *,
    initial: float = 1500.0,
    carryover: float = 0.75,
    k: float = 20.0,
    home_advantage: float = 100.0,
    mov_scaling: bool,
) -> pd.DataFrame:
    ratings: dict[int, float] = {}
    rows: list[dict] = []

    for season in sorted(compact["Season"].unique()):
        season_df = compact.loc[compact["Season"] == season].sort_values("DayNum")
        season_teams = pd.unique(pd.concat([season_df["WTeamID"], season_df["LTeamID"]], ignore_index=True))
        for team_id in list(ratings):
            ratings[team_id] = initial + carryover * (ratings[team_id] - initial)
        for team_id in season_teams:
            ratings.setdefault(int(team_id), initial)

        for row in season_df.itertuples(index=False):
            winner = int(row.WTeamID)
            loser = int(row.LTeamID)
            winner_rating = ratings[winner]
            loser_rating = ratings[loser]
            location = getattr(row, "WLoc", "N")
            if location == "H":
                winner_rating += home_advantage
            elif location == "A":
                loser_rating += home_advantage

            expected = _expected_prob(winner_rating, loser_rating)
            margin = max(float(row.WScore) - float(row.LScore), 1.0)
            scale = 1.0
            if mov_scaling:
                rating_gap = abs(winner_rating - loser_rating)
                scale = np.log1p(margin) * (2.2 / (0.001 * rating_gap + 2.2))
            delta = k * scale * (1.0 - expected)
            ratings[winner] += delta
            ratings[loser] -= delta

        for team_id in season_teams:
            rows.append(
                {
                    "Season": int(season),
                    "TeamID": int(team_id),
                    "rating": float(ratings[int(team_id)]),
                }
            )

    return pd.DataFrame(rows)


def _compute_colley(compact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season, season_df in compact.groupby("Season", sort=True):
        teams = sorted(pd.unique(pd.concat([season_df["WTeamID"], season_df["LTeamID"]], ignore_index=True)).tolist())
        if not teams:
            continue
        team_index = {int(team_id): idx for idx, team_id in enumerate(teams)}
        n = len(teams)
        matrix = np.zeros((n, n), dtype=float)
        b = np.ones(n, dtype=float)
        wins = np.zeros(n, dtype=float)
        losses = np.zeros(n, dtype=float)

        for game in season_df.itertuples(index=False):
            winner = team_index[int(game.WTeamID)]
            loser = team_index[int(game.LTeamID)]
            wins[winner] += 1.0
            losses[loser] += 1.0
            matrix[winner, winner] += 1.0
            matrix[loser, loser] += 1.0
            matrix[winner, loser] -= 1.0
            matrix[loser, winner] -= 1.0

        matrix += 2.0 * np.eye(n)
        b += 0.5 * (wins - losses)
        ratings = np.linalg.solve(matrix, b)
        for team_id, value in zip(teams, ratings, strict=False):
            rows.append({"Season": int(season), "TeamID": int(team_id), "Colley": float(value)})
    return pd.DataFrame(rows)


def _compute_srs(compact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season, season_df in compact.groupby("Season", sort=True):
        teams = sorted(pd.unique(pd.concat([season_df["WTeamID"], season_df["LTeamID"]], ignore_index=True)).tolist())
        if not teams:
            continue
        team_index = {int(team_id): idx for idx, team_id in enumerate(teams)}
        n = len(teams)
        games = np.zeros((n, n), dtype=float)
        margins = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=float)

        for game in season_df.itertuples(index=False):
            winner = team_index[int(game.WTeamID)]
            loser = team_index[int(game.LTeamID)]
            margin = float(game.WScore) - float(game.LScore)
            games[winner, loser] += 1.0
            games[loser, winner] += 1.0
            margins[winner] += margin
            margins[loser] -= margin
            counts[winner] += 1.0
            counts[loser] += 1.0

        avg_margin = np.divide(margins, np.maximum(counts, 1.0))
        matrix = np.eye(n, dtype=float)
        for idx in range(n):
            if counts[idx] > 0:
                matrix[idx] -= games[idx] / counts[idx]
        matrix[-1] = 1.0
        rhs = avg_margin
        rhs[-1] = 0.0
        ratings = np.linalg.solve(matrix, rhs)
        for team_id, value in zip(teams, ratings, strict=False):
            rows.append({"Season": int(season), "TeamID": int(team_id), "SRS": float(value)})
    return pd.DataFrame(rows)


def _compute_glm_quality(compact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season, season_df in compact.groupby("Season", sort=True):
        teams = sorted(pd.unique(pd.concat([season_df["WTeamID"], season_df["LTeamID"]], ignore_index=True)).tolist())
        if len(teams) < 2:
            continue
        team_index = {int(team_id): idx for idx, team_id in enumerate(teams)}
        game_count = len(season_df)
        row_count = game_count * 2
        row_idx = np.repeat(np.arange(row_count), 2)
        col_idx = np.empty(row_count * 2, dtype=int)
        data = np.tile(np.array([1.0, -1.0]), row_count)
        labels = np.empty(row_count, dtype=int)

        for idx, game in enumerate(season_df.itertuples(index=False)):
            winner = team_index[int(game.WTeamID)]
            loser = team_index[int(game.LTeamID)]
            pos_row = 2 * idx
            neg_row = pos_row + 1
            col_idx[4 * idx] = winner
            col_idx[4 * idx + 1] = loser
            col_idx[4 * idx + 2] = loser
            col_idx[4 * idx + 3] = winner
            labels[pos_row] = 1
            labels[neg_row] = 0

        design = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(row_count, len(teams)))
        model = LogisticRegression(
            C=0.075,
            fit_intercept=False,
            penalty="l2",
            solver="liblinear",
            max_iter=400,
        )
        model.fit(design, labels)
        coefs = model.coef_[0]
        for team_id, coef in zip(teams, coefs, strict=False):
            rows.append({"Season": int(season), "TeamID": int(team_id), "GLMQuality": float(coef)})
    return pd.DataFrame(rows)


def _build_detailed_team_games(detailed: pd.DataFrame) -> pd.DataFrame:
    common = ["Season", "DayNum", "NumOT"]
    winner = detailed[
        common
        + ["WTeamID", "WScore", "LTeamID", "LScore", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
           "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]
    ].copy()
    winner.columns = [
        "Season", "DayNum", "NumOT",
        "TeamID", "Score", "OppID", "OppScore", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
        "OppFGM", "OppFGA", "OppFGM3", "OppFGA3", "OppFTM", "OppFTA", "OppOR", "OppDR", "OppAst", "OppTO", "OppStl", "OppBlk", "OppPF",
    ]
    winner["Win"] = 1

    loser = detailed[
        common
        + ["LTeamID", "LScore", "WTeamID", "WScore", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
           "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    ].copy()
    loser.columns = winner.columns[:-1]
    loser["Win"] = 0

    games = pd.concat([winner, loser], ignore_index=True)
    games["Margin"] = games["Score"] - games["OppScore"]
    games["Poss"] = 0.5 * (
        (games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"])
        + (games["OppFGA"] - games["OppOR"] + games["OppTO"] + 0.475 * games["OppFTA"])
    )
    games["Poss"] = games["Poss"].replace(0, np.nan)
    games["OffEff"] = 100.0 * games["Score"] / games["Poss"]
    games["DefEff"] = 100.0 * games["OppScore"] / games["Poss"]
    games["NetEff"] = games["OffEff"] - games["DefEff"]
    games["OTNormalizedMargin"] = games["Margin"] / (1.0 + 0.125 * pd.to_numeric(games["NumOT"], errors="coerce").fillna(0.0))
    return games


def _trimmed_mean(series: pd.Series, trim: float = 0.02) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if clean.empty:
        return float("nan")
    k = int(np.floor(len(clean) * trim))
    if k > 0 and (2 * k) < len(clean):
        clean = clean.iloc[k:-k]
    return float(clean.mean())


def _compute_custom_net_and_ot_margin(detailed: pd.DataFrame) -> pd.DataFrame:
    games = _build_detailed_team_games(detailed)
    grouped = games.groupby(["Season", "TeamID"], sort=True)
    rows = []
    for (season, team_id), frame in grouped:
        rows.append(
            {
                "Season": int(season),
                "TeamID": int(team_id),
                "CustomNetRating": _trimmed_mean(frame["NetEff"]),
                "OTNormalizedMargin": _trimmed_mean(frame["OTNormalizedMargin"], trim=0.0),
            }
        )
    return pd.DataFrame(rows)


def _compute_quality_rows(compact: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    comp_map = composite.rename(columns={"TeamID": "OppID"})
    winner = compact[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()
    winner.columns = ["Season", "DayNum", "TeamID", "OppID", "Score", "OppScore"]
    winner["Win"] = 1
    loser = compact[["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore"]].copy()
    loser.columns = ["Season", "DayNum", "TeamID", "OppID", "Score", "OppScore"]
    loser["Win"] = 0
    all_games = pd.concat([winner, loser], ignore_index=True)
    all_games = all_games.merge(comp_map, on=["Season", "OppID"], how="left")
    all_games["OppCompositeStrength"] = pd.to_numeric(all_games["CompositeStrength"], errors="coerce")
    season_top = all_games.groupby("Season")["OppCompositeStrength"].transform(lambda s: s.quantile(0.75))
    all_games["QualityWin"] = ((all_games["Win"] == 1) & (all_games["OppCompositeStrength"] >= season_top)).astype(int)
    grouped = all_games.groupby(["Season", "TeamID"], sort=True)
    rows = []
    for (season, team_id), frame in grouped:
        wins = frame.loc[frame["Win"] == 1]
        rows.append(
            {
                "Season": int(season),
                "TeamID": int(team_id),
                "OpponentQualityScore": float(wins["OppCompositeStrength"].mean()) if not wins.empty else 0.0,
                "QualityWins": float(frame["QualityWin"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _compute_massey_composite(base: pd.DataFrame, gender: str, rating_source_profile: str) -> pd.Series:
    if gender == "M":
        rank_columns = [
            "Rank_MOR",
            "Rank_POM",
            "Rank_KPK",
            "Rank_NET",
            "Ext_PublicBPIRank",
            "Ext_PublicPOMRank",
            "Ext_PublicNETRank",
            "Ext_PublicTRankRank",
            "Ext_PublicSORRank",
            "Ext_PublicWABRank",
            "Ext_CHH_KPRank",
            "Ext_CHH_NETRank",
        ]
        if rating_source_profile == "b_tier_plus_polls":
            rank_columns = rank_columns + ["Ext_PublicAverageRank"]
        elif rating_source_profile == "c_all_external":
            rank_columns = rank_columns + ["Ext_PublicAverageRank"]
    else:
        a_tier = [
            "Ext_PublicNETRank",
            "Ext_PublicRPIRank",
            "Ext_PublicPredRPIRank",
            "Ext_WN_NET",
            "Ext_WN_ELO",
            "Ext_WN_RPI",
            "Ext_WN_PredRPI",
        ]
        if rating_source_profile in {"current_default", "m_ap_removed_only", "b_tier_plus_polls", "c_all_external"}:
            rank_columns = a_tier + ["Ext_PublicAPRank", "Ext_PublicCoachesRank"]
        else:
            rank_columns = a_tier

    parts: list[pd.Series] = []
    for column in rank_columns:
        if column in base.columns:
            parts.append(base.groupby("Season")[column].transform(lambda s: _season_percentile_strength(s, higher_is_better=False)))
    if not parts:
        return pd.Series(np.nan, index=base.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


@lru_cache(maxsize=16)
def build_gold_ratings(gender: str, rating_source_profile: str = "current_default") -> pd.DataFrame:
    compact = _read_compact(gender)
    detailed = _read_detailed(gender)
    base = build_team_features(gender, include_external=True).copy()
    base["Season"] = base["Season"].astype(int)
    base["TeamID"] = base["TeamID"].astype(int)

    carryover_elo = _compute_elo_variant(compact, mov_scaling=False).rename(columns={"rating": "CarryoverElo"})
    move_elo = _compute_elo_variant(compact, mov_scaling=True).rename(columns={"rating": "MoveElo"})
    colley = _compute_colley(compact)
    srs = _compute_srs(compact)
    glm_quality = _compute_glm_quality(compact)
    custom = _compute_custom_net_and_ot_margin(detailed)

    ratings = base[["Season", "TeamID", "SeedNum", "SeedPriorExpectedWins", "AvgMargin", "SOS", "Last30SOS", "CloseGameWinRate"]].copy()
    ratings["AvgBlkDiff"] = pd.to_numeric(base["Blk"], errors="coerce") if "Blk" in base.columns else np.nan
    if gender == "M":
        ap_series = base["Ext_PublicAverageRank"] if "Ext_PublicAverageRank" in base.columns else pd.Series(np.nan, index=base.index)
    else:
        ap_series = base["Ext_PublicAPRank"] if "Ext_PublicAPRank" in base.columns else pd.Series(np.nan, index=base.index)
    ratings["APStrength"] = _season_percentile_strength(ap_series, higher_is_better=False).reindex(base.index)
    ratings = ratings.merge(carryover_elo, on=["Season", "TeamID"], how="left")
    ratings = ratings.merge(move_elo, on=["Season", "TeamID"], how="left")
    ratings = ratings.merge(colley, on=["Season", "TeamID"], how="left")
    ratings = ratings.merge(srs, on=["Season", "TeamID"], how="left")
    ratings = ratings.merge(glm_quality, on=["Season", "TeamID"], how="left")
    ratings = ratings.merge(custom, on=["Season", "TeamID"], how="left")
    ratings["MasseyComposite"] = _compute_massey_composite(base, gender, rating_source_profile)

    composite_parts = []
    for column in ["MoveElo", "GLMQuality", "SRS", "Colley", "MasseyComposite", "CustomNetRating"]:
        if column in ratings.columns:
            composite_parts.append(ratings.groupby("Season")[column].transform(lambda s: _season_percentile_strength(s, higher_is_better=True)))
    ratings["CompositeStrength"] = pd.concat(composite_parts, axis=1).mean(axis=1, skipna=True) if composite_parts else np.nan
    quality = _compute_quality_rows(compact, ratings[["Season", "TeamID", "CompositeStrength"]])
    ratings = ratings.merge(quality, on=["Season", "TeamID"], how="left")
    ratings["OpponentQualityTournamentRank"] = _compute_harry_tournament_rank(ratings)
    ratings["GoldConsensusStrength"] = _weighted_strength_consensus(
        ratings,
        {
            "MasseyComposite": 0.30,
            "GLMQuality": 0.25,
            "MoveElo": 0.20,
            "CustomNetRating": 0.15,
            "OpponentQualityScore": 0.10,
        },
    )
    ratings["harry_Rating"] = _weighted_strength_consensus(
        ratings,
        {
            "MoveElo": 0.22,
            "GLMQuality": 0.22,
            "MasseyComposite": 0.22,
            "CustomNetRating": 0.14,
            "OpponentQualityTournamentRank": 0.12,
            "QualityWins": 0.08,
        },
    )
    if gender == "M":
        injuries = _load_men_injury_adjustments()
        ratings = ratings.merge(injuries, on=["Season", "TeamID"], how="left")
        ratings["injury_shift"] = pd.to_numeric(ratings.get("injury_shift"), errors="coerce").fillna(0.0)
        ratings["InjuryAdjustedStrength"] = ratings["harry_Rating"].fillna(0.0) + ratings["injury_shift"]
    else:
        ratings["confirmed_out"] = 0.0
        ratings["injury_shift"] = 0.0
        ratings["InjuryAdjustedStrength"] = ratings["harry_Rating"]
    ratings["CustomStrengthCore"] = ratings["GoldConsensusStrength"]
    for column in ratings.columns:
        if column not in {"Season", "TeamID"}:
            ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    return ratings
