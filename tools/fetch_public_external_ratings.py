from __future__ import annotations

import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zizzii_features import attach_team_ids_from_names


HEADERS = {"User-Agent": "Mozilla/5.0"}
EXTERNAL_DIR = ROOT / "external-data"
SEASON = 2026

MEN_FUTURES = {
    "Duke": 425,
    "Houston": 550,
    "Florida": 800,
    "Auburn": 800,
    "Alabama": 850,
    "Tennessee": 1800,
    "Michigan State": 1800,
    "Michigan": 2500,
    "Iowa State": 3000,
    "Kentucky": 3000,
}

WOMEN_FUTURES = {
    "UConn": 150,
    "South Carolina": 270,
    "UCLA": 500,
    "USC": 1400,
    "Texas": 2500,
    "Notre Dame": 3000,
    "TCU": 4000,
    "LSU": 5000,
}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def read_first_table(url: str) -> pd.DataFrame:
    html = fetch_html(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise ValueError(f"No tables found for {url}")
    return tables[0]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part).strip() for part in column if str(part).strip() and "Unnamed" not in str(part)).strip("_")
            for column in frame.columns
        ]
    else:
        frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def american_to_prob(value: float) -> float:
    odds = float(value)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def fetch_warrennolan_compare(gender: str) -> pd.DataFrame:
    path = "basketball" if gender == "M" else "basketballw"
    table = read_first_table(f"https://www.warrennolan.com/{path}/{SEASON}/compare-rankings")
    table = flatten_columns(table)
    table["Season"] = SEASON
    keep = ["Season", "Team", "NET", "ELO"]
    rename_map = {
        "Team": "TeamName",
        "NET": "WN_NET",
        "ELO": "WN_ELO",
    }
    optional = []
    for source_col, target_col in {
        "BPI": "WN_BPI",
        "POM": "WN_POM",
        "T-Rank": "WN_TRank",
        "KPI": "WN_KPI",
        "SOR": "WN_SOR",
        "WAB": "WN_WAB",
        "Average Rank": "WN_AverageRank",
        "Avg. Pred. Rank": "WN_AvgPredRank",
        "RPI": "WN_RPI",
        "Pred. RPI": "WN_PredRPI",
        "Predicted RPI": "WN_PredRPI",
    }.items():
        if source_col in table.columns:
            keep.append(source_col)
            rename_map[source_col] = target_col
            optional.append(target_col)
    frame = table[keep].rename(columns=rename_map)
    numeric_cols = [col for col in frame.columns if col not in {"Season", "TeamName"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return frame


def fetch_warrennolan_polls(gender: str) -> pd.DataFrame:
    path = "basketball" if gender == "M" else "basketballw"
    html = fetch_html(f"https://www.warrennolan.com/{path}/{SEASON}/polls")
    tables = [flatten_columns(table) for table in pd.read_html(io.StringIO(html))]

    outputs = []
    for table in tables:
        if "AP Poll_Rank" in table.columns and "AP Poll_Team" in table.columns:
            ap = table.rename(columns={"AP Poll_Team": "TeamName", "AP Poll_Rank": "APRank"})
            ap["Season"] = SEASON
            outputs.append(ap[["Season", "TeamName", "APRank"]])
        if "Coaches Poll_Rank" in table.columns and "Coaches Poll_Team" in table.columns:
            coaches = table.rename(columns={"Coaches Poll_Team": "TeamName", "Coaches Poll_Rank": "CoachesRank"})
            coaches["Season"] = SEASON
            outputs.append(coaches[["Season", "TeamName", "CoachesRank"]])

    if not outputs:
        return pd.DataFrame(columns=["Season", "TeamName", "APRank", "CoachesRank"])

    merged = outputs[0]
    for frame in outputs[1:]:
        merged = merged.merge(frame, on=["Season", "TeamName"], how="outer")
    return merged


def fetch_official_ncaa_net(gender: str) -> pd.DataFrame:
    slug = "basketball-men/d1/ncaa-mens-basketball-net-rankings" if gender == "M" else "basketball-women/d1/ncaa-womens-basketball-net-rankings"
    table = read_first_table(f"https://www.ncaa.com/rankings/{slug}")
    table = flatten_columns(table)
    frame = table[["Rank", "School"]].copy()
    frame["Season"] = SEASON
    frame = frame.rename(columns={"School": "TeamName", "Rank": "OfficialNETRank"})
    frame["OfficialNETRank"] = pd.to_numeric(frame["OfficialNETRank"], errors="coerce")
    return frame[["Season", "TeamName", "OfficialNETRank"]]


def build_futures_frame(futures: dict[str, int]) -> pd.DataFrame:
    rows = []
    for team_name, american_odds in futures.items():
        rows.append(
            {
                "Season": SEASON,
                "TeamName": team_name,
                "MarketTitleOdds": american_odds,
                "MarketTitleProb": american_to_prob(american_odds),
            }
        )
    return pd.DataFrame(rows)


def attach_and_collapse_team_ids(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    frame = attach_team_ids_from_names(df, gender, team_col="TeamName", target_col="TeamID")
    frame = frame.dropna(subset=["TeamID"]).copy()
    frame["TeamID"] = frame["TeamID"].astype(int)

    numeric_cols = [
        column for column in frame.columns
        if column not in {"Season", "TeamID", "TeamName"} and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = frame.groupby(["Season", "TeamID"], as_index=False)[numeric_cols].mean() if numeric_cols else frame[["Season", "TeamID"]].drop_duplicates()
    names = frame.groupby(["Season", "TeamID"], as_index=False)["TeamName"].first()
    return names.merge(grouped, on=["Season", "TeamID"], how="left")


def build_public_team_ratings(gender: str) -> pd.DataFrame:
    compare_df = attach_and_collapse_team_ids(fetch_warrennolan_compare(gender), gender)
    polls_df = attach_and_collapse_team_ids(fetch_warrennolan_polls(gender), gender)
    official_net_df = attach_and_collapse_team_ids(fetch_official_ncaa_net(gender), gender)
    futures_df = attach_and_collapse_team_ids(build_futures_frame(MEN_FUTURES if gender == "M" else WOMEN_FUTURES), gender)

    name_lookup = pd.concat(
        [
            compare_df[["Season", "TeamID", "TeamName"]],
            polls_df[["Season", "TeamID", "TeamName"]],
            official_net_df[["Season", "TeamID", "TeamName"]],
            futures_df[["Season", "TeamID", "TeamName"]],
        ],
        ignore_index=True,
    ).dropna(subset=["TeamID"]).drop_duplicates(subset=["Season", "TeamID"])

    merged = compare_df.drop(columns=["TeamName"], errors="ignore").merge(
        polls_df.drop(columns=["TeamName"], errors="ignore"), on=["Season", "TeamID"], how="outer"
    )
    merged = merged.merge(official_net_df.drop(columns=["TeamName"], errors="ignore"), on=["Season", "TeamID"], how="outer")
    merged = merged.merge(futures_df.drop(columns=["TeamName"], errors="ignore"), on=["Season", "TeamID"], how="left")
    merged = merged.merge(name_lookup, on=["Season", "TeamID"], how="left")
    merged = merged.sort_values(["Season", "TeamID"]).reset_index(drop=True)

    ordered = ["Season", "TeamID", "TeamName"] + [col for col in merged.columns if col not in {"Season", "TeamID", "TeamName"}]
    return merged[ordered]


def save_frame(frame: pd.DataFrame, output_name: str) -> None:
    EXTERNAL_DIR.mkdir(exist_ok=True)
    output_path = EXTERNAL_DIR / output_name
    frame.to_csv(output_path, index=False)
    missing = int(frame["TeamID"].isna().sum()) if "TeamID" in frame.columns else -1
    print(f"Saved {output_path} rows={len(frame)} missing_teamid={missing}")


def main() -> None:
    men = build_public_team_ratings("M")
    women = build_public_team_ratings("W")
    save_frame(men, "MTeamRatings.csv")
    save_frame(women, "WTeamRatings.csv")


if __name__ == "__main__":
    main()
