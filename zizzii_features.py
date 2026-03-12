from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "ncaa-data"
EXTERNAL_DIR = Path(__file__).resolve().parent / "external-data"
POLL_SYSTEMS = frozenset({"AP", "USA", "DES"})
EXTERNAL_SKIP_TOKENS = ("manual", "signal", "odds", "market", "template", "example")
EXTERNAL_RATING_NEGATIVE_TOKENS = ("adjd", "defrtg", "deff", "defense", "allowed")
POSSESSION_FTA_COEF = 0.475
COMMON_TEAM_ALIASES = {
    "ar pine bluff": "ark pine bluff",
    "app state": "appalachian st",
    "byu": "brigham young",
    "fau": "fl atlantic",
    "lsu": "louisiana state",
    "lmu ca": "loy marymount",
    "loyola chi": "loyola chicago",
    "miami": "miami fl",
    "niu": "n illinois",
    "ole miss": "mississippi",
    "presbyterian college": "presbyterian",
    "queens": "queens nc",
    "saint francis": "st francis pa",
    "saint mary s college": "st mary s ca",
    "saint thomas": "st thomas mn",
    "seattle university": "seattle",
    "southeast missouri": "se missouri st",
    "southern ind": "southern indiana",
    "ualbany": "suny albany",
    "uiw": "incarnate word",
    "uconn": "connecticut",
    "ucf": "central florida",
    "unc": "north carolina",
    "uta": "ut arlington",
    "vcu": "virginia commonwealth",
    "west ga": "west georgia",
}
PATH_ROUND_GROUPS = {
    "Early": (1, 2),
    "Middle": (3, 4),
    "Late": (5, 6),
}


def competition_path(filename: str, data_dir: Optional[Path] = None) -> Path:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    return root / filename


def read_competition_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_csv(competition_path(filename, data_dir))


def normalize_team_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\(\d+\)", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = COMMON_TEAM_ALIASES.get(text, text)
    return text


def build_team_name_lookup(gender: str, data_dir: Optional[Path] = None) -> dict[str, int]:
    spellings_path = competition_path(f"{gender}TeamSpellings.csv", data_dir)
    teams_path = competition_path(f"{gender}Teams.csv", data_dir)

    lookup: dict[str, int] = {}
    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        for row in teams.itertuples(index=False):
            lookup[normalize_team_name(row.TeamName)] = int(row.TeamID)

    if spellings_path.exists():
        spellings = pd.read_csv(spellings_path)
        for row in spellings.itertuples(index=False):
            lookup[normalize_team_name(row.TeamNameSpelling)] = int(row.TeamID)
    return lookup


def resolve_team_id(name: object, lookup: dict[str, int]) -> Optional[int]:
    normalized = normalize_team_name(name)
    exact = lookup.get(normalized)
    if exact is not None:
        return exact

    candidates = [
        (alias, team_id)
        for alias, team_id in lookup.items()
        if normalized.startswith(alias + " ") or normalized == alias
    ]
    if candidates:
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        return int(candidates[0][1])
    return None


def attach_team_ids_from_names(
    df: pd.DataFrame,
    gender: str,
    data_dir: Optional[Path] = None,
    team_col: str = "TeamName",
    target_col: str = "TeamID",
) -> pd.DataFrame:
    if target_col in df.columns or team_col not in df.columns:
        return df

    lookup = build_team_name_lookup(gender, data_dir)
    mapped = df.copy()
    mapped[target_col] = mapped[team_col].map(lambda value: resolve_team_id(value, lookup))
    return mapped


def standardize_external_team_frame(df: pd.DataFrame, gender: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    if "Season" not in df.columns:
        return pd.DataFrame(columns=["Season", "TeamID"])

    frame = df.copy()
    if "TeamID" not in frame.columns:
        team_name_column = next((col for col in ["TeamName", "School", "Team", "TeamNameSpelling"] if col in frame.columns), None)
        if team_name_column is None:
            return pd.DataFrame(columns=["Season", "TeamID"])
        frame = attach_team_ids_from_names(frame, gender, data_dir=data_dir, team_col=team_name_column)

    if "TeamID" not in frame.columns:
        return pd.DataFrame(columns=["Season", "TeamID"])

    frame["TeamID"] = pd.to_numeric(frame["TeamID"], errors="coerce")
    frame = frame.dropna(subset=["Season", "TeamID"]).copy()
    frame["Season"] = frame["Season"].astype(int)
    frame["TeamID"] = frame["TeamID"].astype(int)
    return frame


def should_skip_external_file(path: Path) -> bool:
    lowered = path.name.lower()
    return any(token in lowered for token in EXTERNAL_SKIP_TOKENS)


def season_zscore(df: pd.DataFrame, column: str) -> pd.Series:
    mean_ = df.groupby("Season")[column].transform("mean")
    std_ = df.groupby("Season")[column].transform("std").clip(lower=0.1)
    return (df[column] - mean_) / std_


def build_compact_game_log(compact_df: pd.DataFrame) -> pd.DataFrame:
    df = compact_df.copy()
    has_wloc = "WLoc" in df.columns

    winners = df[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()
    winners.columns = ["Season", "DayNum", "TeamID", "OppID", "Pts", "OppPts"]
    winners["Win"] = 1.0

    losers = df[["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore"]].copy()
    losers.columns = ["Season", "DayNum", "TeamID", "OppID", "Pts", "OppPts"]
    losers["Win"] = 0.0

    if has_wloc:
        winners["IsHome"] = (df["WLoc"] == "H").astype(int)
        losers["IsHome"] = (df["WLoc"] == "A").astype(int)

    games = pd.concat([winners, losers], ignore_index=True)
    games["Margin"] = games["Pts"] - games["OppPts"]
    return games


def build_detailed_game_log(detailed_df: pd.DataFrame) -> pd.DataFrame:
    df = detailed_df.copy()
    has_wloc = "WLoc" in df.columns

    def poss(fga, oreb, tov, fta):
        return fga - oreb + tov + POSSESSION_FTA_COEF * fta

    df["WPoss"] = poss(df["WFGA"], df["WOR"], df["WTO"], df["WFTA"])
    df["LPoss"] = poss(df["LFGA"], df["LOR"], df["LTO"], df["LFTA"])

    winners = df[
        [
            "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WPoss", "LPoss",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO",
            "WStl", "WBlk", "LFGM", "LFGA", "LFGM3", "LFTA", "LOR", "LDR", "LTO",
        ]
    ].copy()
    winners.columns = [
        "Season", "DayNum", "TeamID", "OppID", "Pts", "OppPts", "Poss", "OppPoss",
        "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk",
        "OppFGM", "OppFGA", "OppFGM3", "OppFTA", "OppOR", "OppDR", "OppTO",
    ]
    winners["Win"] = 1

    losers = df[
        [
            "Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore", "LPoss", "WPoss",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO",
            "LStl", "LBlk", "WFGM", "WFGA", "WFGM3", "WFTA", "WOR", "WDR", "WTO",
        ]
    ].copy()
    losers.columns = winners.columns[:-1]
    losers["Win"] = 0

    if has_wloc:
        winners["IsHome"] = (df["WLoc"] == "H").astype(int)
        losers["IsHome"] = (df["WLoc"] == "A").astype(int)

    games = pd.concat([winners, losers], ignore_index=True)
    games["Poss"] = games["Poss"].clip(lower=1)
    games["OppPoss"] = games["OppPoss"].clip(lower=1)
    games["OffRtg"] = 100.0 * games["Pts"] / games["Poss"]
    games["DefRtg"] = 100.0 * games["OppPts"] / games["OppPoss"]
    games["FGPct"] = games["FGM"] / games["FGA"].clip(lower=1)
    games["FG3Pct"] = games["FGM3"] / games["FGA3"].clip(lower=1)
    games["FTPct"] = games["FTM"] / games["FTA"].clip(lower=1)
    games["Margin"] = games["Pts"] - games["OppPts"]
    games["eFG"] = (games["FGM"] + 0.5 * games["FGM3"]) / games["FGA"].clip(lower=1)
    games["TOVPct"] = games["TO"] / games["Poss"].clip(lower=1)
    games["FTR"] = games["FTA"] / games["FGA"].clip(lower=1)
    games["ORBPct"] = games["OR"] / (games["OR"] + games["OppDR"]).clip(lower=1)
    games["OppEFG"] = (games["OppFGM"] + 0.5 * games["OppFGM3"]) / games["OppFGA"].clip(lower=1)
    games["OppTOVPct"] = games["OppTO"] / games["OppPoss"].clip(lower=1)
    games["OppFTR"] = games["OppFTA"] / games["OppFGA"].clip(lower=1)
    games["OppORBPct"] = games["OppOR"] / (games["OppOR"] + games["DR"]).clip(lower=1)
    return add_adjusted_efficiency_features(games)


def add_adjusted_efficiency_features(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games

    season_ratings = games.groupby(["Season", "TeamID"]).agg(
        TeamSeasonOffRtg=("OffRtg", "mean"),
        TeamSeasonDefRtg=("DefRtg", "mean"),
    ).reset_index()
    season_ratings["LeagueAvgOffRtg"] = season_ratings.groupby("Season")["TeamSeasonOffRtg"].transform("mean")
    season_ratings["LeagueAvgDefRtg"] = season_ratings.groupby("Season")["TeamSeasonDefRtg"].transform("mean")

    opponent_strength = season_ratings.rename(
        columns={
            "TeamID": "OppID",
            "TeamSeasonOffRtg": "OppSeasonOffRtg",
            "TeamSeasonDefRtg": "OppSeasonDefRtg",
        }
    )
    enriched = games.merge(opponent_strength, on=["Season", "OppID"], how="left")
    enriched["AdjOffRtg"] = enriched["OffRtg"] + (enriched["LeagueAvgDefRtg"] - enriched["OppSeasonDefRtg"])
    enriched["AdjDefRtg"] = enriched["DefRtg"] + (enriched["LeagueAvgOffRtg"] - enriched["OppSeasonOffRtg"])
    enriched["AdjNetRtg"] = enriched["AdjOffRtg"] - enriched["AdjDefRtg"]
    return enriched


def compute_elo(
    results_df: pd.DataFrame,
    k: float = 20.0,
    initial: float = 1500.0,
    carryover: float = 0.75,
) -> pd.DataFrame:
    results = results_df.sort_values(["Season", "DayNum"]).copy()
    seasons = sorted(results["Season"].unique())
    elo = {}
    records = []

    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for season in seasons:
        season_df = results[results["Season"] == season]
        season_teams = sorted(set(season_df["WTeamID"]).union(set(season_df["LTeamID"])))
        season_rows: list[dict[str, float | int]] = []

        for team_id in list(elo.keys()):
            elo[team_id] = initial + carryover * (elo[team_id] - initial)
        for team_id in season_teams:
            elo.setdefault(int(team_id), initial)
        starting_elo = {int(team_id): float(elo[int(team_id)]) for team_id in season_teams}
        peak_elo = dict(starting_elo)

        season_day_values = pd.to_numeric(season_df["DayNum"], errors="coerce").dropna()
        half_day = float(season_day_values.median()) if not season_day_values.empty else 0.0

        for row in season_df.itertuples(index=False):
            winner = int(row.WTeamID)
            loser = int(row.LTeamID)
            ew = elo[winner]
            el = elo[loser]
            exp_w = expected(ew, el)
            margin = abs(float(row.WScore) - float(row.LScore))
            elo_gap = abs(ew - el)
            mov_mult = np.log(margin + 1.0) * (2.2 / (elo_gap * 0.001 + 2.2))
            k_adj = k * mov_mult
            elo[winner] = ew + k_adj * (1.0 - exp_w)
            elo[loser] = el + k_adj * (0.0 - (1.0 - exp_w))
            peak_elo[winner] = max(peak_elo[winner], float(elo[winner]))
            peak_elo[loser] = max(peak_elo[loser], float(elo[loser]))

            day_num = float(row.DayNum)
            season_rows.append({"TeamID": winner, "DayNum": day_num, "PostGameElo": float(elo[winner])})
            season_rows.append({"TeamID": loser, "DayNum": day_num, "PostGameElo": float(elo[loser])})

        if season_rows:
            elo_path = pd.DataFrame(season_rows)
            early_means = elo_path[elo_path["DayNum"] <= half_day].groupby("TeamID")["PostGameElo"].mean().to_dict()
            late_means = elo_path[elo_path["DayNum"] > half_day].groupby("TeamID")["PostGameElo"].mean().to_dict()
        else:
            early_means = {}
            late_means = {}

        for team_id in season_teams:
            team_key = int(team_id)
            final_elo = float(elo[team_key])
            peak = float(peak_elo.get(team_key, final_elo))
            early = float(early_means.get(team_key, starting_elo[team_key]))
            late = float(late_means.get(team_key, final_elo))
            records.append(
                {
                    "Season": season,
                    "TeamID": team_key,
                    "Elo": final_elo,
                    "EloPeak": peak,
                    "EloPeakDrop": peak - final_elo,
                    "EloTrend": late - early,
                }
            )

    return pd.DataFrame(records)


def aggregate_efficiency_rows(games: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    agg_spec = {
        "Games": ("Win", "count"),
        "WinRate": ("Win", "mean"),
        "OffRtg": ("OffRtg", "mean"),
        "DefRtg": ("DefRtg", "mean"),
        "Tempo": ("Poss", "mean"),
        "FGPct": ("FGPct", "mean"),
        "FG3Pct": ("FG3Pct", "mean"),
        "FTPct": ("FTPct", "mean"),
        "OR": ("OR", "mean"),
        "DR": ("DR", "mean"),
        "Ast": ("Ast", "mean"),
        "TO": ("TO", "mean"),
        "Stl": ("Stl", "mean"),
        "Blk": ("Blk", "mean"),
        "AvgMargin": ("Margin", "mean"),
        "eFG": ("eFG", "mean"),
        "TOVPct": ("TOVPct", "mean"),
        "FTR": ("FTR", "mean"),
        "ORBPct": ("ORBPct", "mean"),
    }
    for column in ["AdjOffRtg", "AdjDefRtg", "OppEFG", "OppTOVPct", "OppFTR", "OppORBPct"]:
        if column in games.columns:
            agg_spec[column] = (column, "mean")

    agg = games.groupby(["Season", "TeamID"]).agg(**agg_spec).reset_index()
    agg["NetRtg"] = agg["OffRtg"] - agg["DefRtg"]
    if "AdjOffRtg" in agg.columns and "AdjDefRtg" in agg.columns:
        agg["AdjNetRtg"] = agg["AdjOffRtg"] - agg["AdjDefRtg"]

    for column in ["OffRtg", "DefRtg", "NetRtg", "AdjOffRtg", "AdjDefRtg", "AdjNetRtg"]:
        if column not in agg.columns:
            continue
        season_mean = agg.groupby("Season")[column].transform("mean")
        season_std = agg.groupby("Season")[column].transform("std").clip(lower=0.1)
        agg[f"{column}_z"] = (agg[column] - season_mean) / season_std

    if prefix:
        rename_map = {col: f"{prefix}{col}" for col in agg.columns if col not in {"Season", "TeamID"}}
        agg = agg.rename(columns=rename_map)
    return agg


def resolve_detailed_games(detailed_df: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if games is not None:
        return games.copy()
    if detailed_df is None or detailed_df.empty:
        return pd.DataFrame()
    return build_detailed_game_log(detailed_df)


def compute_efficiency(detailed_df: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    games = resolve_detailed_games(detailed_df, games)
    if games.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    agg = aggregate_efficiency_rows(games)

    if "IsHome" in games.columns:
        neutral = games[games["IsHome"] == 0].copy()
        if not neutral.empty:
            neutral_spec = {
                "NeutralOffRtg": ("OffRtg", "mean"),
                "NeutralDefRtg": ("DefRtg", "mean"),
            }
            if "AdjOffRtg" in neutral.columns and "AdjDefRtg" in neutral.columns:
                neutral_spec["NeutralAdjOffRtg"] = ("AdjOffRtg", "mean")
                neutral_spec["NeutralAdjDefRtg"] = ("AdjDefRtg", "mean")

            neutral_agg = neutral.groupby(["Season", "TeamID"]).agg(**neutral_spec).reset_index()
            neutral_agg["NeutralNetRtg"] = neutral_agg["NeutralOffRtg"] - neutral_agg["NeutralDefRtg"]
            if "NeutralAdjOffRtg" in neutral_agg.columns and "NeutralAdjDefRtg" in neutral_agg.columns:
                neutral_agg["NeutralAdjNetRtg"] = neutral_agg["NeutralAdjOffRtg"] - neutral_agg["NeutralAdjDefRtg"]

            merge_cols = ["Season", "TeamID"]
            for column in ["NeutralNetRtg", "NeutralAdjNetRtg"]:
                if column not in neutral_agg.columns:
                    continue
                mean_ = neutral_agg.groupby("Season")[column].transform("mean")
                std_ = neutral_agg.groupby("Season")[column].transform("std").clip(lower=0.1)
                neutral_agg[f"{column}_z"] = (neutral_agg[column] - mean_) / std_
                merge_cols.append(f"{column}_z")
            agg = agg.merge(neutral_agg[merge_cols], on=["Season", "TeamID"], how="left")

    return agg


def compute_recent_efficiency(
    detailed_df: Optional[pd.DataFrame] = None,
    n_games: int = 10,
    games: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    games = resolve_detailed_games(detailed_df, games)
    if games.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    games = games.sort_values(["Season", "TeamID", "DayNum"])
    recent = games.groupby(["Season", "TeamID"]).tail(n_games).copy()
    return aggregate_efficiency_rows(recent, prefix="RecentEff")


def compute_recent_efficiency_window(
    detailed_df: Optional[pd.DataFrame] = None,
    lookback_days: int = 30,
    prefix: str = "Recent30Eff",
    games: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    games = resolve_detailed_games(detailed_df, games)
    if games.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    games = games.sort_values(["Season", "TeamID", "DayNum"])
    season_last_day = games.groupby("Season")["DayNum"].transform("max")
    recent = games[games["DayNum"] >= (season_last_day - lookback_days)].copy()
    if recent.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])
    return aggregate_efficiency_rows(recent, prefix=prefix)


def compute_winrate_compact(compact_df: pd.DataFrame) -> pd.DataFrame:
    games = build_compact_game_log(compact_df)
    return games.groupby(["Season", "TeamID"]).agg(
        WinRate=("Win", "mean"),
        AvgMargin=("Margin", "mean"),
        Games=("Win", "count"),
    ).reset_index()


def compute_recent_form(compact_df: pd.DataFrame, n_games: int = 15) -> pd.DataFrame:
    games = build_compact_game_log(compact_df).sort_values(["Season", "TeamID", "DayNum"])
    recent = games.groupby(["Season", "TeamID"]).tail(n_games).copy()

    def ewm_agg(group: pd.DataFrame) -> pd.Series:
        size = len(group)
        weights = np.exp(-0.14 * np.arange(size - 1, -1, -1))
        weights /= weights.sum()
        return pd.Series({
            "RecentWinRate": float(np.dot(group["Win"].values, weights)),
            "RecentMargin": float(np.dot(group["Margin"].values, weights)),
        })

    return recent.groupby(["Season", "TeamID"]).apply(ewm_agg).reset_index()


def compute_recent_compact_features(compact_df: pd.DataFrame, elo_df: pd.DataFrame, n_games: int = 10, close_margin: int = 5) -> pd.DataFrame:
    games = build_compact_game_log(compact_df).sort_values(["Season", "TeamID", "DayNum"])
    opp_quality = elo_df.copy()
    opp_quality["OppEloRank"] = opp_quality.groupby("Season")["Elo"].rank(method="first", ascending=False)
    opp_quality = opp_quality.rename(columns={"TeamID": "OppID", "Elo": "OppElo"})
    games = games.merge(opp_quality[["Season", "OppID", "OppElo", "OppEloRank"]], on=["Season", "OppID"], how="left")

    recent = games.groupby(["Season", "TeamID"]).tail(n_games).copy()
    recent_agg = recent.groupby(["Season", "TeamID"]).agg(
        Last10WinRate=("Win", "mean"),
        Last10Margin=("Margin", "mean"),
        Last10SOS=("OppElo", "mean"),
    ).reset_index()

    close_games = games[games["Margin"].abs() <= close_margin].copy()
    close_agg = close_games.groupby(["Season", "TeamID"]).agg(
        CloseGameWinRate=("Win", "mean"),
        CloseGameCount=("Win", "count"),
        CloseGameMargin=("Margin", "mean"),
    ).reset_index()

    games["VsTop50"] = (games["OppEloRank"] <= 50).astype(int)
    games["VsTop100"] = (games["OppEloRank"] <= 100).astype(int)
    games["Top50Win"] = ((games["Win"] == 1) & (games["VsTop50"] == 1)).astype(int)
    games["Top100Win"] = ((games["Win"] == 1) & (games["VsTop100"] == 1)).astype(int)

    quality = games.groupby(["Season", "TeamID"]).agg(
        Top50Games=("VsTop50", "sum"),
        Top50Wins=("Top50Win", "sum"),
        Top100Games=("VsTop100", "sum"),
        Top100Wins=("Top100Win", "sum"),
    ).reset_index()
    quality["Top50WinRate"] = quality["Top50Wins"] / quality["Top50Games"].replace(0, np.nan)
    quality["Top100WinRate"] = quality["Top100Wins"] / quality["Top100Games"].replace(0, np.nan)

    return recent_agg.merge(close_agg, on=["Season", "TeamID"], how="left").merge(quality, on=["Season", "TeamID"], how="left")


def compute_recent_compact_window_features(compact_df: pd.DataFrame, elo_df: pd.DataFrame, lookback_days: int = 30, close_margin: int = 5) -> pd.DataFrame:
    games = build_compact_game_log(compact_df).sort_values(["Season", "TeamID", "DayNum"])
    if games.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])

    opp_quality = elo_df.copy()
    opp_quality["OppEloRank"] = opp_quality.groupby("Season")["Elo"].rank(method="first", ascending=False)
    opp_quality = opp_quality.rename(columns={"TeamID": "OppID", "Elo": "OppElo"})
    games = games.merge(opp_quality[["Season", "OppID", "OppElo", "OppEloRank"]], on=["Season", "OppID"], how="left")
    season_last_day = games.groupby("Season")["DayNum"].transform("max")
    recent = games[games["DayNum"] >= (season_last_day - lookback_days)].copy()
    if recent.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])

    recent_agg = recent.groupby(["Season", "TeamID"]).agg(
        Last30WinRate=("Win", "mean"),
        Last30Margin=("Margin", "mean"),
        Last30SOS=("OppElo", "mean"),
    ).reset_index()

    close_games = recent[recent["Margin"].abs() <= close_margin].copy()
    close_agg = close_games.groupby(["Season", "TeamID"]).agg(
        Last30CloseWinRate=("Win", "mean"),
        Last30CloseCount=("Win", "count"),
    ).reset_index()

    recent["Last30VsTop50"] = (recent["OppEloRank"] <= 50).astype(int)
    recent["Last30Top50Win"] = ((recent["Win"] == 1) & (recent["Last30VsTop50"] == 1)).astype(int)
    quality = recent.groupby(["Season", "TeamID"]).agg(
        Last30Top50Games=("Last30VsTop50", "sum"),
        Last30Top50Wins=("Last30Top50Win", "sum"),
    ).reset_index()
    quality["Last30Top50WinRate"] = quality["Last30Top50Wins"] / quality["Last30Top50Games"].replace(0, np.nan)

    return recent_agg.merge(close_agg, on=["Season", "TeamID"], how="left").merge(quality, on=["Season", "TeamID"], how="left")


def compute_massey_features(massey_df: pd.DataFrame, key_systems=("KPK", "POM", "NET", "MOR"), max_day: int = 133) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Season", "TeamID", "MasseyMean", "MasseyStd", "MasseyMin"] + [f"Rank_{system}" for system in key_systems])
    if massey_df.empty:
        return empty

    df = massey_df[massey_df["RankingDayNum"] <= max_day].copy()
    if df.empty:
        return empty
    df = df[~df["SystemName"].isin(POLL_SYSTEMS)].copy()
    if df.empty:
        return empty

    latest_idx = df.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
    latest = df.loc[latest_idx].copy()
    latest["Pct"] = latest.groupby(["Season", "SystemName"])["OrdinalRank"].rank(pct=True)
    consensus = latest.groupby(["Season", "TeamID"])["Pct"].agg(MasseyMean="mean", MasseyStd="std", MasseyMin="min").reset_index()
    consensus["MasseyStd"] = consensus["MasseyStd"].fillna(0.0)

    key_df = latest[latest["SystemName"].isin(key_systems)].copy()
    if key_df.empty:
        return consensus

    pivot = key_df.pivot_table(index=["Season", "TeamID"], columns="SystemName", values="OrdinalRank").reset_index()
    pivot.columns.name = None
    rename_map = {system: f"Rank_{system}" for system in key_systems if system in pivot.columns}
    return consensus.merge(pivot.rename(columns=rename_map), on=["Season", "TeamID"], how="left")


def compute_massey_momentum(massey_df: pd.DataFrame, latest_day: int = 133, lookback_days: int = 14) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Season", "TeamID", "MasseyMomentum"])
    if massey_df.empty:
        return empty

    df = massey_df[~massey_df["SystemName"].isin(POLL_SYSTEMS)].copy()
    if df.empty:
        return empty

    latest_cut = df[df["RankingDayNum"] <= latest_day].copy()
    previous_cut = df[df["RankingDayNum"] <= latest_day - lookback_days].copy()
    if latest_cut.empty or previous_cut.empty:
        return empty

    latest_idx = latest_cut.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
    previous_idx = previous_cut.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
    latest = latest_cut.loc[latest_idx].copy()
    previous = previous_cut.loc[previous_idx].copy()

    latest["PctNow"] = latest.groupby(["Season", "SystemName"])["OrdinalRank"].rank(pct=True)
    previous["PctPrev"] = previous.groupby(["Season", "SystemName"])["OrdinalRank"].rank(pct=True)

    latest = latest[["Season", "TeamID", "SystemName", "PctNow"]]
    previous = previous[["Season", "TeamID", "SystemName", "PctPrev"]]
    joined = latest.merge(previous, on=["Season", "TeamID", "SystemName"], how="inner")
    if joined.empty:
        return empty

    joined["MasseyMomentum"] = joined["PctPrev"] - joined["PctNow"]
    return joined.groupby(["Season", "TeamID"])["MasseyMomentum"].mean().reset_index()


def compute_sos(compact_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    games = build_compact_game_log(compact_df)[["Season", "TeamID", "OppID"]]
    opp_elo = elo_df.rename(columns={"TeamID": "OppID", "Elo": "OppElo"})
    games = games.merge(opp_elo, on=["Season", "OppID"], how="left")
    return games.groupby(["Season", "TeamID"])["OppElo"].mean().reset_index().rename(columns={"OppElo": "SOS"})


def compute_rpi_style_metrics(compact_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Season", "TeamID", "OWP", "OOWP", "RPIStyle", "RPIStyleSOS"]
    if compact_df.empty:
        return pd.DataFrame(columns=columns)

    games = build_compact_game_log(compact_df)[["Season", "TeamID", "OppID", "Win"]].copy()
    team_totals = games.groupby(["Season", "TeamID"]).agg(TeamWins=("Win", "sum"), TeamGames=("Win", "size")).reset_index()
    team_totals["WinRate"] = team_totals["TeamWins"] / team_totals["TeamGames"].clip(lower=1)

    head_to_head = games.groupby(["Season", "TeamID", "OppID"]).agg(VsGames=("Win", "size"), VsWins=("Win", "sum")).reset_index()
    opp_totals = team_totals.rename(columns={"TeamID": "OppID", "TeamWins": "OppWins", "TeamGames": "OppGames"})
    opp_vs_team = head_to_head.rename(
        columns={"TeamID": "OppID", "OppID": "TeamID", "VsGames": "OppVsTeamGames", "VsWins": "OppVsTeamWins"}
    )

    owp_games = games.merge(opp_totals, on=["Season", "OppID"], how="left")
    owp_games = owp_games.merge(opp_vs_team, on=["Season", "OppID", "TeamID"], how="left")
    owp_games["OppVsTeamGames"] = owp_games["OppVsTeamGames"].fillna(0.0)
    owp_games["OppVsTeamWins"] = owp_games["OppVsTeamWins"].fillna(0.0)
    owp_games["OppGamesExcl"] = owp_games["OppGames"] - owp_games["OppVsTeamGames"]
    owp_games["OppWinsExcl"] = owp_games["OppWins"] - owp_games["OppVsTeamWins"]
    owp_games["OppWinPctExcl"] = owp_games["OppWinsExcl"] / owp_games["OppGamesExcl"].replace(0, np.nan)
    owp = owp_games.groupby(["Season", "TeamID"])["OppWinPctExcl"].mean().reset_index(name="OWP")

    oowp_games = games.merge(owp.rename(columns={"TeamID": "OppID", "OWP": "OppOWP"}), on=["Season", "OppID"], how="left")
    oowp = oowp_games.groupby(["Season", "TeamID"])["OppOWP"].mean().reset_index(name="OOWP")

    rpi = team_totals[["Season", "TeamID", "WinRate"]].merge(owp, on=["Season", "TeamID"], how="left")
    rpi = rpi.merge(oowp, on=["Season", "TeamID"], how="left")
    rpi["OWP"] = rpi["OWP"].fillna(rpi["WinRate"])
    rpi["OOWP"] = rpi["OOWP"].fillna(rpi["OWP"])
    rpi["RPIStyle"] = 0.25 * rpi["WinRate"] + 0.50 * rpi["OWP"] + 0.25 * rpi["OOWP"]
    rpi["RPIStyleSOS"] = 0.50 * rpi["OWP"] + 0.50 * rpi["OOWP"]
    return rpi[columns]


def compute_graph_strength(compact_df: pd.DataFrame, damping: float = 0.85, max_iter: int = 80) -> pd.DataFrame:
    columns = ["Season", "TeamID", "PageRank", "PageRank_z", "PageRankSOS"]
    if compact_df.empty:
        return pd.DataFrame(columns=columns)

    records = []
    has_wloc = "WLoc" in compact_df.columns
    for season, season_df in compact_df.groupby("Season", sort=True):
        teams = np.array(sorted(set(season_df["WTeamID"]).union(set(season_df["LTeamID"]))), dtype=int)
        if len(teams) == 0:
            continue

        edges = season_df[["WTeamID", "LTeamID", "WScore", "LScore"]].copy()
        edges["Weight"] = 1.0 + (edges["WScore"] - edges["LScore"]).clip(lower=0).astype(float).clip(upper=30.0) / 20.0
        if has_wloc:
            edges["Weight"] += np.where(season_df["WLoc"] == "A", 0.15, 0.0)
            edges["Weight"] += np.where(season_df["WLoc"] == "N", 0.05, 0.0)
        edges = edges.groupby(["LTeamID", "WTeamID"], as_index=False)["Weight"].sum()

        team_to_idx = {team_id: idx for idx, team_id in enumerate(teams)}
        src = edges["LTeamID"].map(team_to_idx).to_numpy(dtype=int)
        dst = edges["WTeamID"].map(team_to_idx).to_numpy(dtype=int)
        weight = edges["Weight"].to_numpy(dtype=float)
        out_weight = np.bincount(src, weights=weight, minlength=len(teams))
        transition = weight / np.maximum(out_weight[src], 1e-12)

        rank = np.full(len(teams), 1.0 / len(teams), dtype=float)
        dangling = out_weight == 0
        for _ in range(max_iter):
            contrib = np.zeros(len(teams), dtype=float)
            np.add.at(contrib, dst, rank[src] * transition)
            dangling_mass = rank[dangling].sum() / len(teams)
            updated = (1.0 - damping) / len(teams) + damping * (contrib + dangling_mass)
            if np.max(np.abs(updated - rank)) < 1e-12:
                rank = updated
                break
            rank = updated

        page_rank = pd.DataFrame({"Season": int(season), "TeamID": teams, "PageRank": rank})
        mean_ = page_rank["PageRank"].mean()
        std_ = max(page_rank["PageRank"].std(ddof=0), 1e-12)
        page_rank["PageRank_z"] = (page_rank["PageRank"] - mean_) / std_

        games = build_compact_game_log(season_df)[["Season", "TeamID", "OppID"]].copy()
        opp_rank = page_rank.rename(columns={"TeamID": "OppID", "PageRank": "OppPageRank"})
        games = games.merge(opp_rank[["Season", "OppID", "OppPageRank"]], on=["Season", "OppID"], how="left")
        page_rank_sos = games.groupby(["Season", "TeamID"])["OppPageRank"].mean().reset_index(name="PageRankSOS")
        records.append(page_rank.merge(page_rank_sos, on=["Season", "TeamID"], how="left"))

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True)[columns]


def compute_conf_tourney(conf_tourney_df: pd.DataFrame) -> pd.DataFrame:
    if conf_tourney_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "ConfTourneyWins"])
    return conf_tourney_df.groupby(["Season", "WTeamID"]).size().reset_index(name="ConfTourneyWins").rename(columns={"WTeamID": "TeamID"})


def compute_conference_strength(compact_df: pd.DataFrame, team_conf_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Season", "TeamID", "ConfAbbrev", "ConfNonConfWinRate", "ConfNonConfAvgMargin", "ConfNonConfGames", "ConfMeanElo", "ConfTeamCount"]
    if compact_df.empty or team_conf_df.empty:
        return pd.DataFrame(columns=columns)

    conf_lookup = team_conf_df[["Season", "TeamID", "ConfAbbrev"]].drop_duplicates()
    games = build_compact_game_log(compact_df)
    games = games.merge(conf_lookup.rename(columns={"ConfAbbrev": "TeamConf"}), on=["Season", "TeamID"], how="left")
    games = games.merge(conf_lookup.rename(columns={"TeamID": "OppID", "ConfAbbrev": "OppConf"}), on=["Season", "OppID"], how="left")
    games = games.dropna(subset=["TeamConf", "OppConf"])
    non_conf = games[games["TeamConf"] != games["OppConf"]].copy()
    if non_conf.empty:
        return pd.DataFrame(columns=columns)

    conf_perf = non_conf.groupby(["Season", "TeamConf"]).agg(
        ConfNonConfWinRate=("Win", "mean"),
        ConfNonConfAvgMargin=("Margin", "mean"),
        ConfNonConfGames=("Win", "size"),
    ).reset_index().rename(columns={"TeamConf": "ConfAbbrev"})

    conf_elo = conf_lookup.merge(elo_df, on=["Season", "TeamID"], how="left").groupby(["Season", "ConfAbbrev"]).agg(
        ConfMeanElo=("Elo", "mean"),
        ConfTeamCount=("TeamID", "nunique"),
    ).reset_index()

    conf_stats = conf_lookup.merge(conf_perf, on=["Season", "ConfAbbrev"], how="left")
    conf_stats = conf_stats.merge(conf_elo, on=["Season", "ConfAbbrev"], how="left")
    return conf_stats[columns]


def compute_coach_features(coaches_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Season", "TeamID", "CoachTenure", "CoachChange", "CoachMidseasonChange", "CoachCount"]
    if coaches_df.empty:
        return pd.DataFrame(columns=columns)

    df = coaches_df.copy()
    df["CoachName"] = df["CoachName"].astype(str)
    df["SpanDays"] = (df["LastDayNum"] - df["FirstDayNum"] + 1).clip(lower=1)
    season_summary = df.groupby(["Season", "TeamID"]).agg(CoachCount=("CoachName", "nunique")).reset_index()
    season_summary["CoachMidseasonChange"] = (season_summary["CoachCount"] > 1).astype(int)

    coach_spans = df.groupby(["Season", "TeamID", "CoachName"])["SpanDays"].sum().reset_index().sort_values(
        ["Season", "TeamID", "SpanDays", "CoachName"], ascending=[True, True, False, True]
    )
    primary = coach_spans.groupby(["Season", "TeamID"]).head(1).copy().sort_values(["TeamID", "Season"])

    rows = []
    previous = {}
    for row in primary.itertuples(index=False):
        team_id = int(row.TeamID)
        season = int(row.Season)
        coach_name = row.CoachName
        prev = previous.get(team_id)
        if prev is not None and prev["CoachName"] == coach_name and prev["Season"] == season - 1:
            tenure = prev["CoachTenure"] + 1
        else:
            tenure = 1
        coach_change = 1 if prev is not None and prev["CoachName"] != coach_name else 0
        previous[team_id] = {"Season": season, "CoachName": coach_name, "CoachTenure": tenure}
        rows.append({"Season": season, "TeamID": team_id, "CoachTenure": tenure, "CoachChange": coach_change})

    return pd.DataFrame(rows).merge(season_summary, on=["Season", "TeamID"], how="left")[columns]


def load_external_team_features(gender: str, external_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> pd.DataFrame:
    root = Path(external_dir) if external_dir is not None else EXTERNAL_DIR
    if not root.exists():
        return pd.DataFrame(columns=["Season", "TeamID"])

    candidates = []
    for pattern in [f"{gender}*.csv", f"{gender}_*.csv", f"{gender.lower()}*.csv"]:
        candidates.extend(sorted(root.glob(pattern)))

    unique_paths = []
    seen = set()
    for path in candidates:
        if path.name not in seen:
            seen.add(path.name)
            unique_paths.append(path)

    merged = None
    for path in unique_paths:
        if should_skip_external_file(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = standardize_external_team_frame(df, gender, data_dir=data_dir)
        if df.empty:
            continue

        keep = ["Season", "TeamID"]
        rename_map = {}
        for column in df.columns:
            if column in {"Season", "TeamID"}:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            keep.append(column)
            rename_map[column] = column if column.startswith("Ext_") else f"Ext_{column}"

        if len(keep) == 2:
            continue

        current = df[keep].rename(columns=rename_map)
        merged = current if merged is None else merged.merge(current, on=["Season", "TeamID"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["Season", "TeamID"])

    numeric_ext = [col for col in merged.columns if col.startswith("Ext_") and pd.api.types.is_numeric_dtype(merged[col])]
    strength_parts = []
    for column in numeric_ext:
        z_col = f"{column}_z"
        merged[z_col] = season_zscore(merged, column)
        direction = -1.0 if any(token in column.lower() for token in EXTERNAL_RATING_NEGATIVE_TOKENS) else 1.0
        strength_parts.append(direction * merged[z_col])

    if strength_parts:
        merged["ExtCompositeStrength"] = np.nanmean(np.column_stack(strength_parts), axis=1)
    return merged


def parse_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    seeds = seeds_df.copy()
    seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(float)
    return seeds[["Season", "TeamID", "SeedNum"]]


def compute_seed_priors(tourney_df: pd.DataFrame, seeds_df: pd.DataFrame, last_season: int) -> pd.DataFrame:
    columns = ["Season", "SeedBucket", "SeedPriorExpectedWins", "SeedPriorWinPct"]
    if tourney_df.empty or seeds_df.empty:
        return pd.DataFrame(columns=columns)

    seeds = parse_seeds(seeds_df).dropna(subset=["SeedNum"]).copy()
    seeds["SeedBucket"] = seeds["SeedNum"].round().astype(int).clip(lower=1, upper=16)
    seasons = list(range(int(seeds["Season"].min()), int(last_season) + 1))
    grid = pd.MultiIndex.from_product([seasons, range(1, 17)], names=["Season", "SeedBucket"]).to_frame(index=False)

    participants = pd.concat(
        [
            tourney_df[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
            tourney_df[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
        ],
        ignore_index=True,
    )
    wins = tourney_df[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})

    seed_teams = seeds.groupby(["Season", "SeedBucket"]).size().reset_index(name="SeedTeamCount")
    seed_games = participants.merge(seeds[["Season", "TeamID", "SeedBucket"]], on=["Season", "TeamID"], how="inner")
    seed_games = seed_games.groupby(["Season", "SeedBucket"]).size().reset_index(name="SeedGames")
    seed_wins = wins.merge(seeds[["Season", "TeamID", "SeedBucket"]], on=["Season", "TeamID"], how="inner")
    seed_wins = seed_wins.groupby(["Season", "SeedBucket"]).size().reset_index(name="SeedWins")

    priors = grid.merge(seed_teams, on=["Season", "SeedBucket"], how="left")
    priors = priors.merge(seed_games, on=["Season", "SeedBucket"], how="left")
    priors = priors.merge(seed_wins, on=["Season", "SeedBucket"], how="left")
    priors[["SeedTeamCount", "SeedGames", "SeedWins"]] = priors[["SeedTeamCount", "SeedGames", "SeedWins"]].fillna(0.0)
    priors = priors.sort_values(["SeedBucket", "Season"]).reset_index(drop=True)

    for column in ["SeedTeamCount", "SeedGames", "SeedWins"]:
        cumulative = priors.groupby("SeedBucket")[column].cumsum()
        priors[f"Prev{column}"] = cumulative.groupby(priors["SeedBucket"]).shift(fill_value=0.0)

    priors["SeedPriorExpectedWins"] = priors["PrevSeedWins"] / priors["PrevSeedTeamCount"].replace(0, np.nan)
    priors["SeedPriorWinPct"] = priors["PrevSeedWins"] / priors["PrevSeedGames"].replace(0, np.nan)
    priors["SeedPriorExpectedWins"] = priors["SeedPriorExpectedWins"].fillna(0.0)
    priors["SeedPriorWinPct"] = priors["SeedPriorWinPct"].fillna(0.0)
    return priors[columns]


def build_generic_seed_opponent_map(data_dir: Optional[Path] = None) -> dict[int, dict[int, tuple[int, ...]]]:
    slots = read_competition_csv("MNCAATourneySlots.csv", data_dir)
    if slots.empty:
        return {}

    latest_season = int(slots["Season"].max())
    current = slots[slots["Season"] == latest_season].copy()
    children = current.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict("index")
    memo: dict[str, set[int]] = {}

    def descendants(node: object) -> set[int]:
        key = str(node)
        if key in memo:
            return memo[key]
        if key in children:
            row = children[key]
            memo[key] = descendants(row["StrongSeed"]) | descendants(row["WeakSeed"])
            return memo[key]
        match = re.search(r"(\d+)", key)
        memo[key] = {int(match.group(1))} if match is not None else set()
        return memo[key]

    opponent_map: dict[int, dict[int, set[int]]] = {}
    for row in current.itertuples(index=False):
        match = re.match(r"R(\d+)", str(row.Slot))
        if match is None:
            continue
        round_num = int(match.group(1))
        strong = descendants(row.StrongSeed)
        weak = descendants(row.WeakSeed)
        for seed in strong:
            opponent_map.setdefault(seed, {}).setdefault(round_num, set()).update(weak)
        for seed in weak:
            opponent_map.setdefault(seed, {}).setdefault(round_num, set()).update(strong)

    return {
        seed: {round_num: tuple(sorted(values)) for round_num, values in rounds.items()}
        for seed, rounds in opponent_map.items()
    }


def compute_seed_path_features(base_df: pd.DataFrame, data_dir: Optional[Path] = None) -> pd.DataFrame:
    columns = ["Season", "TeamID"]
    round_columns = [f"PathDifficultyR{round_num}" for round_num in range(1, 7)] + [
        f"PathSeedMeanR{round_num}" for round_num in range(1, 7)
    ]
    group_columns = []
    for label in PATH_ROUND_GROUPS:
        group_columns.extend([f"PathDifficulty{label}", f"PathSeedMean{label}"])
    if base_df.empty or "SeedNum" not in base_df.columns:
        return pd.DataFrame(columns=columns + round_columns + group_columns)

    opponent_map = build_generic_seed_opponent_map(data_dir)
    df = base_df[["Season", "TeamID", "SeedNum", "Elo", "WinRate"]].copy()
    if "NetRtg_z" in base_df.columns:
        df["NetRtg_z"] = base_df["NetRtg_z"]
    if "ExtCompositeStrength" in base_df.columns:
        df["ExtCompositeStrength"] = base_df["ExtCompositeStrength"]

    strength_parts = [season_zscore(df, "Elo").fillna(0.0), season_zscore(df, "WinRate").fillna(0.0)]
    if "NetRtg_z" in df.columns:
        strength_parts.append(df["NetRtg_z"].fillna(0.0))
    if "ExtCompositeStrength" in df.columns:
        strength_parts.append(df["ExtCompositeStrength"].fillna(0.0))

    df["SeedBucket"] = pd.to_numeric(df["SeedNum"], errors="coerce").fillna(17.0).round().clip(lower=1, upper=17).astype(int)
    strength_matrix = np.column_stack([series.to_numpy() for series in strength_parts])
    df["SeedPathStrength"] = np.nanmean(strength_matrix, axis=1)

    seeded = df[df["SeedBucket"] <= 16].copy()
    season_seed_strength = (
        seeded.groupby(["Season", "SeedBucket"])["SeedPathStrength"].mean().reset_index().pivot(index="Season", columns="SeedBucket", values="SeedPathStrength")
    )
    season_seed_mean = seeded.groupby("Season")["SeedPathStrength"].mean().to_dict()

    rows = []
    for row in df.itertuples(index=False):
        record = {"Season": int(row.Season), "TeamID": int(row.TeamID)}
        seed_bucket = int(row.SeedBucket)
        for round_num in range(1, 7):
            opponent_buckets = opponent_map.get(seed_bucket, {}).get(round_num, ())
            record[f"PathSeedMeanR{round_num}"] = float(np.mean(opponent_buckets)) if opponent_buckets else np.nan
            if not opponent_buckets or row.Season not in season_seed_strength.index:
                record[f"PathDifficultyR{round_num}"] = np.nan
                continue

            strengths = season_seed_strength.loc[row.Season, list(opponent_buckets)].dropna()
            if strengths.empty:
                record[f"PathDifficultyR{round_num}"] = season_seed_mean.get(int(row.Season), np.nan)
            else:
                record[f"PathDifficultyR{round_num}"] = float(strengths.mean())

        for label, rounds in PATH_ROUND_GROUPS.items():
            round_values = [record.get(f"PathDifficultyR{round_num}") for round_num in rounds]
            seed_values = [record.get(f"PathSeedMeanR{round_num}") for round_num in rounds]
            round_values = [value for value in round_values if pd.notna(value)]
            seed_values = [value for value in seed_values if pd.notna(value)]
            record[f"PathDifficulty{label}"] = float(np.mean(round_values)) if round_values else np.nan
            record[f"PathSeedMean{label}"] = float(np.mean(seed_values)) if seed_values else np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def impute_seeds(base_df: pd.DataFrame, gender: str) -> pd.DataFrame:
    df = base_df.copy()
    missing = df["SeedNum"].isna()
    if not missing.any():
        return df

    for season in sorted(df.loc[missing, "Season"].unique()):
        season_mask = df["Season"] == season
        missing_mask = season_mask & df["SeedNum"].isna()
        if gender == "M" and "MasseyMean" in df.columns and df.loc[season_mask, "MasseyMean"].notna().any():
            ranking = df.loc[season_mask, "MasseyMean"].rank(method="first", ascending=True)
        else:
            ranking = df.loc[season_mask, "Elo"].rank(method="first", ascending=False)
        imputed = np.where(ranking <= 64, np.ceil(ranking / 4.0).clip(upper=16), 17.0)
        df.loc[season_mask, "_ImputedSeed"] = imputed
        df.loc[missing_mask, "SeedNum"] = df.loc[missing_mask, "_ImputedSeed"]

    return df.drop(columns=["_ImputedSeed"], errors="ignore")


def build_team_features(
    gender: str = "M",
    data_dir: Optional[Path] = None,
    external_dir: Optional[Path] = None,
    elo_params: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    prefix = gender
    compact = read_competition_csv(f"{prefix}RegularSeasonCompactResults.csv", data_dir)
    seeds_raw = read_competition_csv(f"{prefix}NCAATourneySeeds.csv", data_dir)
    team_conf = read_competition_csv(f"{prefix}TeamConferences.csv", data_dir)
    tourney_results = read_competition_csv(f"{prefix}NCAATourneyCompactResults.csv", data_dir)

    elo_config = {"k": 20.0, "initial": 1500.0, "carryover": 0.75}
    if elo_params:
        for key in elo_config:
            if key in elo_params:
                elo_config[key] = float(elo_params[key])

    elo_df = compute_elo(compact, **elo_config)
    winrate_df = compute_winrate_compact(compact)
    recent_df = compute_recent_form(compact)
    trend_df = compute_recent_compact_features(compact, elo_df)
    trend30_df = compute_recent_compact_window_features(compact, elo_df)
    sos_df = compute_sos(compact, elo_df)
    rpi_df = compute_rpi_style_metrics(compact)
    page_rank_df = compute_graph_strength(compact)
    conf_strength_df = compute_conference_strength(compact, team_conf, elo_df)

    detailed_path = competition_path(f"{prefix}RegularSeasonDetailedResults.csv", data_dir)
    detailed_df = pd.read_csv(detailed_path) if detailed_path.exists() else pd.DataFrame()
    detailed_games = build_detailed_game_log(detailed_df) if not detailed_df.empty else pd.DataFrame()
    eff_df = compute_efficiency(games=detailed_games) if not detailed_games.empty else pd.DataFrame()
    recent_eff_df = compute_recent_efficiency(games=detailed_games) if not detailed_games.empty else pd.DataFrame()
    recent30_eff_df = compute_recent_efficiency_window(games=detailed_games) if not detailed_games.empty else pd.DataFrame()

    conf_tourney_path = competition_path(f"{prefix}ConferenceTourneyGames.csv", data_dir)
    conf_wins_df = compute_conf_tourney(pd.read_csv(conf_tourney_path)) if conf_tourney_path.exists() else pd.DataFrame(columns=["Season", "TeamID", "ConfTourneyWins"])

    if gender == "M":
        massey_path = competition_path("MMasseyOrdinals.csv", data_dir)
        if massey_path.exists():
            massey_df = pd.read_csv(massey_path)
            rank_df = compute_massey_features(massey_df)
            massey_momentum_df = compute_massey_momentum(massey_df)
        else:
            rank_df = pd.DataFrame()
            massey_momentum_df = pd.DataFrame(columns=["Season", "TeamID", "MasseyMomentum"])
        coach_path = competition_path("MTeamCoaches.csv", data_dir)
        coach_df = compute_coach_features(pd.read_csv(coach_path)) if coach_path.exists() else pd.DataFrame()
    else:
        rank_df = pd.DataFrame(columns=["Season", "TeamID", "MasseyMean", "MasseyStd", "MasseyMin"])
        massey_momentum_df = pd.DataFrame(columns=["Season", "TeamID", "MasseyMomentum"])
        coach_df = pd.DataFrame(columns=["Season", "TeamID", "CoachTenure", "CoachChange", "CoachMidseasonChange", "CoachCount"])

    base = elo_df.merge(winrate_df, on=["Season", "TeamID"], how="left")
    base = base.merge(recent_df, on=["Season", "TeamID"], how="left")
    base = base.merge(trend_df, on=["Season", "TeamID"], how="left")
    base = base.merge(trend30_df, on=["Season", "TeamID"], how="left")
    base = base.merge(sos_df, on=["Season", "TeamID"], how="left")
    base = base.merge(rpi_df, on=["Season", "TeamID"], how="left")
    base = base.merge(page_rank_df, on=["Season", "TeamID"], how="left")
    base = base.merge(team_conf[["Season", "TeamID", "ConfAbbrev"]].drop_duplicates(), on=["Season", "TeamID"], how="left")
    base = base.merge(conf_strength_df, on=["Season", "TeamID", "ConfAbbrev"], how="left")
    base = base.merge(conf_wins_df, on=["Season", "TeamID"], how="left")

    if not eff_df.empty:
        eff_cols = [
            "Season", "TeamID", "OffRtg", "DefRtg", "NetRtg", "Tempo", "FGPct", "FG3Pct", "FTPct", "OR", "DR",
            "Ast", "TO", "Stl", "Blk", "OffRtg_z", "DefRtg_z", "NetRtg_z",
            "AdjOffRtg", "AdjDefRtg", "AdjNetRtg", "AdjOffRtg_z", "AdjDefRtg_z", "AdjNetRtg_z",
            "eFG", "TOVPct", "FTR", "ORBPct", "OppEFG", "OppTOVPct", "OppFTR", "OppORBPct",
            "NeutralNetRtg_z", "NeutralAdjNetRtg_z",
        ]
        base = base.merge(eff_df[[col for col in eff_cols if col in eff_df.columns]], on=["Season", "TeamID"], how="left")
    if not recent_eff_df.empty:
        base = base.merge(recent_eff_df, on=["Season", "TeamID"], how="left")
    if not recent30_eff_df.empty:
        base = base.merge(recent30_eff_df, on=["Season", "TeamID"], how="left")
    if not rank_df.empty:
        base = base.merge(rank_df, on=["Season", "TeamID"], how="left")
    if not massey_momentum_df.empty:
        base = base.merge(massey_momentum_df, on=["Season", "TeamID"], how="left")
    if not coach_df.empty:
        base = base.merge(coach_df, on=["Season", "TeamID"], how="left")

    external_df = load_external_team_features(gender, external_dir, data_dir=data_dir)
    if not external_df.empty:
        base = base.merge(external_df, on=["Season", "TeamID"], how="left")

    base = base.merge(parse_seeds(seeds_raw), on=["Season", "TeamID"], how="left")
    base["SeedImputed"] = base["SeedNum"].isna().astype(int)
    base = impute_seeds(base, gender)
    base["SeedBucket"] = pd.to_numeric(base["SeedNum"], errors="coerce").fillna(17.0).round().clip(lower=1, upper=17).astype(int)

    seed_priors = compute_seed_priors(tourney_results, seeds_raw, last_season=int(compact["Season"].max()))
    base = base.merge(seed_priors, on=["Season", "SeedBucket"], how="left")
    path_df = compute_seed_path_features(base, data_dir=data_dir)
    base = base.merge(path_df, on=["Season", "TeamID"], how="left")

    base["MomentumWinRate"] = base["Last10WinRate"] - base["WinRate"]
    base["MomentumMargin"] = base["Last10Margin"] - base["AvgMargin"]
    if "Last30WinRate" in base.columns:
        base["Momentum30WinRate"] = base["Last30WinRate"] - base["WinRate"]
    if "Last30Margin" in base.columns:
        base["Momentum30Margin"] = base["Last30Margin"] - base["AvgMargin"]
    if "RecentEffNetRtg" in base.columns and "NetRtg" in base.columns:
        base["MomentumNetRtg"] = base["RecentEffNetRtg"] - base["NetRtg"]
        base["MomentumNetRtg_z"] = base["RecentEffNetRtg_z"] - base["NetRtg_z"]
    if "RecentEffAdjNetRtg" in base.columns and "AdjNetRtg" in base.columns:
        base["MomentumAdjNetRtg"] = base["RecentEffAdjNetRtg"] - base["AdjNetRtg"]
        base["MomentumAdjNetRtg_z"] = base["RecentEffAdjNetRtg_z"] - base["AdjNetRtg_z"]
    if "Recent30EffNetRtg" in base.columns and "NetRtg" in base.columns:
        base["Momentum30NetRtg"] = base["Recent30EffNetRtg"] - base["NetRtg"]
        base["Momentum30NetRtg_z"] = base["Recent30EffNetRtg_z"] - base["NetRtg_z"]
    if "Recent30EffAdjNetRtg" in base.columns and "AdjNetRtg" in base.columns:
        base["Momentum30AdjNetRtg"] = base["Recent30EffAdjNetRtg"] - base["AdjNetRtg"]
        base["Momentum30AdjNetRtg_z"] = base["Recent30EffAdjNetRtg_z"] - base["AdjNetRtg_z"]

    fill_zero = [
        "ConfTourneyWins", "CloseGameCount", "Top50Games", "Top50Wins", "Top100Games", "Top100Wins",
        "SeedImputed", "CoachChange", "CoachMidseasonChange", "SeedPriorExpectedWins", "SeedPriorWinPct",
        "Last30CloseCount", "Last30Top50Games", "Last30Top50Wins",
    ]
    for column in fill_zero:
        if column in base.columns:
            base[column] = base[column].fillna(0)

    fill_one = ["CoachTenure", "CoachCount"]
    for column in fill_one:
        if column in base.columns:
            base[column] = base[column].fillna(1)

    fill_neutral = [
        "CloseGameWinRate", "Top50WinRate", "Top100WinRate", "MasseyMomentum", "OWP", "OOWP", "RPIStyle", "RPIStyleSOS",
        "Last30CloseWinRate", "Last30Top50WinRate", "PageRankSOS",
        "PathDifficultyR1", "PathDifficultyR2", "PathDifficultyR3", "PathDifficultyR4", "PathDifficultyR5", "PathDifficultyR6",
        "PathDifficultyEarly", "PathDifficultyMiddle", "PathDifficultyLate",
        "PathSeedMeanR1", "PathSeedMeanR2", "PathSeedMeanR3", "PathSeedMeanR4", "PathSeedMeanR5", "PathSeedMeanR6",
        "PathSeedMeanEarly", "PathSeedMeanMiddle", "PathSeedMeanLate", "PageRank", "PageRank_z",
    ]
    for column in fill_neutral:
        if column in base.columns:
            base[column] = base[column].fillna(0)

    return base


if __name__ == "__main__":
    men = build_team_features("M")
    women = build_team_features("W")
    print("Men:", men.shape, men["Season"].min(), men["Season"].max())
    print("Women:", women.shape, women["Season"].min(), women["Season"].max())
