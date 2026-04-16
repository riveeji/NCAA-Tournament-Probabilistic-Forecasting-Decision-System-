from __future__ import annotations

from functools import lru_cache
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NCAA_DATA = ROOT / "ncaa-data"


def _safe_team_frame(teams: pd.DataFrame) -> pd.DataFrame:
    frame = teams[["Season", "TeamID"]].copy()
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce").astype(int)
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce").astype(int)
    return frame.drop_duplicates(["Season", "TeamID"]).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    required = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    if not required.issubset(games.columns):
        raise KeyError(f"Missing columns for internal ratings: {sorted(required - set(games.columns))}")
    frame = games[list(required)].copy()
    for col in ("Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["WLoc"] = frame["WLoc"].astype(str).fillna("N")
    frame = frame.dropna(subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"])
    for col in ("Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"):
        frame[col] = frame[col].astype(int)
    return frame.sort_values(["Season", "DayNum"]).reset_index(drop=True)


def _prepare_team_conferences(team_conferences: pd.DataFrame | None) -> pd.DataFrame:
    if team_conferences is None:
        return pd.DataFrame(columns=["Season", "TeamID", "ConfAbbrev"])
    required = {"Season", "TeamID", "ConfAbbrev"}
    if not required.issubset(team_conferences.columns):
        return pd.DataFrame(columns=["Season", "TeamID", "ConfAbbrev"])
    frame = team_conferences[["Season", "TeamID", "ConfAbbrev"]].copy()
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame["ConfAbbrev"] = frame["ConfAbbrev"].astype(str).str.strip()
    frame = frame.dropna(subset=["Season", "TeamID"])
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame.drop_duplicates(["Season", "TeamID"]).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def _prepare_detailed_games(games: pd.DataFrame | None) -> pd.DataFrame:
    if games is None:
        return pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "EffMargin"])
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WFGA",
        "WOR",
        "WTO",
        "WFTA",
        "LFGA",
        "LOR",
        "LTO",
        "LFTA",
    }
    if not required.issubset(games.columns):
        return pd.DataFrame(columns=["Season", "DayNum", "WTeamID", "LTeamID", "EffMargin"])
    frame = games[list(required)].copy()
    numeric_cols = [col for col in frame.columns if col not in {"Season", "DayNum", "WTeamID", "LTeamID"}]
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna()
    for col in ("Season", "DayNum", "WTeamID", "LTeamID"):
        frame[col] = frame[col].astype(int)
    w_poss = frame["WFGA"] - frame["WOR"] + frame["WTO"] + 0.475 * frame["WFTA"]
    l_poss = frame["LFGA"] - frame["LOR"] + frame["LTO"] + 0.475 * frame["LFTA"]
    w_poss = w_poss.replace(0, np.nan)
    l_poss = l_poss.replace(0, np.nan)
    eff_margin = (100.0 * frame["WScore"] / w_poss) - (100.0 * frame["LScore"] / l_poss)
    prepared = frame[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    prepared["EffMargin"] = pd.to_numeric(eff_margin, errors="coerce").fillna(0.0)
    return prepared.sort_values(["Season", "DayNum"]).reset_index(drop=True)


def _compute_carry_elo(games: pd.DataFrame, teams: pd.DataFrame, *, k: float = 20.0, home_adv: float = 100.0, mean_reversion: float = 0.75) -> pd.DataFrame:
    elo: dict[int, float] = {}
    rows: list[dict[str, float | int]] = []

    for season in sorted(teams["Season"].unique()):
        season_teams = sorted(teams.loc[teams["Season"] == season, "TeamID"].unique())
        for team in list(elo):
            elo[team] = elo[team] * mean_reversion + 1500.0 * (1.0 - mean_reversion)
        for team in season_teams:
            elo.setdefault(int(team), 1500.0)

        season_games = games.loc[games["Season"] == season]
        for _, game in season_games.iterrows():
            w = int(game["WTeamID"])
            l = int(game["LTeamID"])
            elo.setdefault(w, 1500.0)
            elo.setdefault(l, 1500.0)
            w_elo = elo[w]
            l_elo = elo[l]
            if game["WLoc"] == "H":
                w_elo += home_adv
            elif game["WLoc"] == "A":
                l_elo += home_adv
            w_exp = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))
            mov = max(int(game["WScore"]) - int(game["LScore"]), 1)
            mov_mult = log(mov + 1.0) * (2.2 / (((w_elo - l_elo) * 0.001) + 2.2))
            delta = k * mov_mult * (1.0 - w_exp)
            elo[w] += delta
            elo[l] -= delta

        for team in season_teams:
            rows.append({"Season": int(season), "TeamID": int(team), "CarryElo": float(elo[int(team)])})

    return pd.DataFrame(rows)


def _compute_carry_elo85(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    carry = _compute_carry_elo(games, teams, mean_reversion=0.85)
    return carry.rename(columns={"CarryElo": "CarryElo85"})


def _compute_carry_elo80(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    carry = _compute_carry_elo(games, teams, mean_reversion=0.80)
    return carry.rename(columns={"CarryElo": "CarryElo80"})


def _compute_colley(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for season in sorted(teams["Season"].unique()):
        season_teams = sorted(teams.loc[teams["Season"] == season, "TeamID"].unique())
        if not season_teams:
            continue
        index = {team: idx for idx, team in enumerate(season_teams)}
        n = len(season_teams)
        c = np.zeros((n, n), dtype=float)
        wins = np.zeros(n, dtype=float)
        losses = np.zeros(n, dtype=float)
        for _, game in games.loc[games["Season"] == season].iterrows():
            w = int(game["WTeamID"])
            l = int(game["LTeamID"])
            if w not in index or l not in index:
                continue
            iw = index[w]
            il = index[l]
            c[iw, iw] += 1.0
            c[il, il] += 1.0
            c[iw, il] -= 1.0
            c[il, iw] -= 1.0
            wins[iw] += 1.0
            losses[il] += 1.0
        colley = c + 2.0 * np.eye(n)
        b = 1.0 + (wins - losses) / 2.0
        try:
            ratings = np.linalg.solve(colley, b)
        except np.linalg.LinAlgError:
            ratings = np.full(n, 0.5, dtype=float)
        for team in season_teams:
            rows.append({"Season": int(season), "TeamID": int(team), "Colley": float(ratings[index[team]])})
    return pd.DataFrame(rows)


def _compute_colley_nc(games: pd.DataFrame, teams: pd.DataFrame, team_conferences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    conf_frame = _prepare_team_conferences(team_conferences)
    conf_lookup = {
        (int(row["Season"]), int(row["TeamID"])): str(row["ConfAbbrev"])
        for _, row in conf_frame.iterrows()
    }
    for season in sorted(teams["Season"].unique()):
        season_teams = sorted(teams.loc[teams["Season"] == season, "TeamID"].unique())
        if not season_teams:
            continue
        index = {team: idx for idx, team in enumerate(season_teams)}
        n = len(season_teams)
        c = np.zeros((n, n), dtype=float)
        wins = np.zeros(n, dtype=float)
        losses = np.zeros(n, dtype=float)
        for _, game in games.loc[games["Season"] == season].iterrows():
            w = int(game["WTeamID"])
            l = int(game["LTeamID"])
            if w not in index or l not in index:
                continue
            w_conf = conf_lookup.get((int(season), w))
            l_conf = conf_lookup.get((int(season), l))
            weight = 0.75 if w_conf and l_conf and w_conf == l_conf else 1.0
            iw = index[w]
            il = index[l]
            c[iw, iw] += weight
            c[il, il] += weight
            c[iw, il] -= weight
            c[il, iw] -= weight
            wins[iw] += weight
            losses[il] += weight
        colley = c + 2.0 * np.eye(n)
        b = 1.0 + (wins - losses) / 2.0
        try:
            ratings = np.linalg.solve(colley, b)
        except np.linalg.LinAlgError:
            ratings = np.full(n, 0.5, dtype=float)
        for team in season_teams:
            rows.append({"Season": int(season), "TeamID": int(team), "ColleyNC": float(ratings[index[team]])})
    return pd.DataFrame(rows)


def _compute_srs(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    margin_clip: float | None = None,
    output_column: str = "SRS",
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for season in sorted(teams["Season"].unique()):
        season_teams = sorted(teams.loc[teams["Season"] == season, "TeamID"].unique())
        if not season_teams:
            continue
        margins: dict[int, list[float]] = {team: [] for team in season_teams}
        opponents: dict[int, list[int]] = {team: [] for team in season_teams}
        for _, game in games.loc[games["Season"] == season].iterrows():
            w = int(game["WTeamID"])
            l = int(game["LTeamID"])
            if w not in margins or l not in margins:
                continue
            margin = float(int(game["WScore"]) - int(game["LScore"]))
            if margin_clip is not None:
                margin = float(np.clip(margin, -margin_clip, margin_clip))
            margins[w].append(margin)
            opponents[w].append(l)
            margins[l].append(-margin)
            opponents[l].append(w)

        ratings = {team: 0.0 for team in season_teams}
        for _ in range(max_iter):
            updated: dict[int, float] = {}
            for team in season_teams:
                if not margins[team]:
                    updated[team] = 0.0
                    continue
                avg_margin = float(np.mean(margins[team]))
                opp_rating = float(np.mean([ratings[opp] for opp in opponents[team]]))
                updated[team] = avg_margin + opp_rating
            mean_rating = float(np.mean(list(updated.values()))) if updated else 0.0
            updated = {team: value - mean_rating for team, value in updated.items()}
            max_delta = max(abs(updated[team] - ratings[team]) for team in season_teams) if season_teams else 0.0
            ratings = updated
            if max_delta <= tol:
                break
        for team in season_teams:
            rows.append({"Season": int(season), "TeamID": int(team), output_column: float(ratings[team])})
    return pd.DataFrame(rows)


def _compute_eff_srs(detailed_games: pd.DataFrame, teams: pd.DataFrame, *, tol: float = 1e-6, max_iter: int = 200) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if detailed_games.empty:
        for _, row in teams.iterrows():
            rows.append({"Season": int(row["Season"]), "TeamID": int(row["TeamID"]), "EffSRS": 0.0})
        return pd.DataFrame(rows)

    for season in sorted(teams["Season"].unique()):
        season_teams = sorted(teams.loc[teams["Season"] == season, "TeamID"].unique())
        if not season_teams:
            continue
        margins: dict[int, list[float]] = {team: [] for team in season_teams}
        opponents: dict[int, list[int]] = {team: [] for team in season_teams}
        for _, game in detailed_games.loc[detailed_games["Season"] == season].iterrows():
            w = int(game["WTeamID"])
            l = int(game["LTeamID"])
            if w not in margins or l not in margins:
                continue
            margin = float(game["EffMargin"])
            margins[w].append(margin)
            opponents[w].append(l)
            margins[l].append(-margin)
            opponents[l].append(w)

        ratings = {team: 0.0 for team in season_teams}
        for _ in range(max_iter):
            updated: dict[int, float] = {}
            for team in season_teams:
                if not margins[team]:
                    updated[team] = 0.0
                    continue
                avg_margin = float(np.mean(margins[team]))
                opp_rating = float(np.mean([ratings[opp] for opp in opponents[team]]))
                updated[team] = avg_margin + opp_rating
            mean_rating = float(np.mean(list(updated.values()))) if updated else 0.0
            updated = {team: value - mean_rating for team, value in updated.items()}
            max_delta = max(abs(updated[team] - ratings[team]) for team in season_teams) if season_teams else 0.0
            ratings = updated
            if max_delta <= tol:
                break
        for team in season_teams:
            rows.append({"Season": int(season), "TeamID": int(team), "EffSRS": float(ratings[team])})
    return pd.DataFrame(rows)


def compute_internal_ratings_from_games(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    detailed_games: pd.DataFrame | None = None,
    team_conferences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    team_frame = _safe_team_frame(teams)
    game_frame = _prepare_games(games)
    detailed_frame = _prepare_detailed_games(detailed_games)
    conference_frame = _prepare_team_conferences(team_conferences)

    carry_elo = _compute_carry_elo(game_frame, team_frame)
    carry_elo80 = _compute_carry_elo80(game_frame, team_frame)
    carry_elo85 = _compute_carry_elo85(game_frame, team_frame)
    colley = _compute_colley(game_frame, team_frame)
    colley_nc = _compute_colley_nc(game_frame, team_frame, conference_frame)
    srs = _compute_srs(game_frame, team_frame)
    srs_clip15 = _compute_srs(game_frame, team_frame, margin_clip=15.0, output_column="SRSClip15")
    srs_clip20 = _compute_srs(game_frame, team_frame, margin_clip=20.0, output_column="SRSClip20")
    eff_srs = _compute_eff_srs(detailed_frame, team_frame)

    merged = team_frame.merge(carry_elo, on=["Season", "TeamID"], how="left")
    merged = merged.merge(carry_elo80, on=["Season", "TeamID"], how="left")
    merged = merged.merge(carry_elo85, on=["Season", "TeamID"], how="left")
    merged = merged.merge(colley, on=["Season", "TeamID"], how="left")
    merged = merged.merge(colley_nc, on=["Season", "TeamID"], how="left")
    merged = merged.merge(srs, on=["Season", "TeamID"], how="left")
    merged = merged.merge(srs_clip15, on=["Season", "TeamID"], how="left")
    merged = merged.merge(srs_clip20, on=["Season", "TeamID"], how="left")
    merged = merged.merge(eff_srs, on=["Season", "TeamID"], how="left")
    merged["CarryElo"] = pd.to_numeric(merged["CarryElo"], errors="coerce").fillna(1500.0)
    merged["CarryElo80"] = pd.to_numeric(merged["CarryElo80"], errors="coerce").fillna(1500.0)
    merged["CarryElo85"] = pd.to_numeric(merged["CarryElo85"], errors="coerce").fillna(1500.0)
    merged["Colley"] = pd.to_numeric(merged["Colley"], errors="coerce").fillna(0.5)
    merged["ColleyNC"] = pd.to_numeric(merged["ColleyNC"], errors="coerce").fillna(0.5)
    merged["SRS"] = pd.to_numeric(merged["SRS"], errors="coerce").fillna(0.0)
    merged["SRSClip15"] = pd.to_numeric(merged["SRSClip15"], errors="coerce").fillna(0.0)
    merged["SRSClip20"] = pd.to_numeric(merged["SRSClip20"], errors="coerce").fillna(0.0)
    merged["EffSRS"] = pd.to_numeric(merged["EffSRS"], errors="coerce").fillna(0.0)
    return merged


@lru_cache(maxsize=4)
def _cached_internal_ratings(gender: str) -> pd.DataFrame:
    compact = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonCompactResults.csv")
    detailed = pd.read_csv(NCAA_DATA / f"{gender}RegularSeasonDetailedResults.csv")
    conferences = pd.read_csv(NCAA_DATA / f"{gender}TeamConferences.csv")
    teams = pd.DataFrame(
        pd.unique(
            pd.concat(
                [
                    compact[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
                    compact[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
                ],
                ignore_index=True,
            ).to_records(index=False)
        ),
        columns=["Season", "TeamID"],
    )
    return compute_internal_ratings_from_games(games=compact, teams=teams, detailed_games=detailed, team_conferences=conferences)


def load_internal_ratings(gender: str, teams: pd.DataFrame) -> pd.DataFrame:
    requested = _safe_team_frame(teams)
    ratings = _cached_internal_ratings(gender)
    merged = requested.merge(ratings, on=["Season", "TeamID"], how="left")
    merged["CarryElo"] = pd.to_numeric(merged["CarryElo"], errors="coerce").fillna(1500.0)
    merged["CarryElo80"] = pd.to_numeric(merged["CarryElo80"], errors="coerce").fillna(1500.0)
    merged["CarryElo85"] = pd.to_numeric(merged["CarryElo85"], errors="coerce").fillna(1500.0)
    merged["Colley"] = pd.to_numeric(merged["Colley"], errors="coerce").fillna(0.5)
    merged["ColleyNC"] = pd.to_numeric(merged["ColleyNC"], errors="coerce").fillna(0.5)
    merged["SRS"] = pd.to_numeric(merged["SRS"], errors="coerce").fillna(0.0)
    merged["SRSClip15"] = pd.to_numeric(merged["SRSClip15"], errors="coerce").fillna(0.0)
    merged["SRSClip20"] = pd.to_numeric(merged["SRSClip20"], errors="coerce").fillna(0.0)
    merged["EffSRS"] = pd.to_numeric(merged["EffSRS"], errors="coerce").fillna(0.0)
    return merged
