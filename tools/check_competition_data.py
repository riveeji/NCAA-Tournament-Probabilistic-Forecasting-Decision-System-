from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ncaa-data"

EXPECTED_FILES = [
    "Cities.csv", "Conferences.csv", "MConferenceTourneyGames.csv",
    "MGameCities.csv", "MMasseyOrdinals.csv", "MNCAATourneyCompactResults.csv",
    "MNCAATourneyDetailedResults.csv", "MNCAATourneySeedRoundSlots.csv",
    "MNCAATourneySeeds.csv", "MNCAATourneySlots.csv",
    "MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv",
    "MSeasons.csv", "MSecondaryTourneyCompactResults.csv",
    "MSecondaryTourneyTeams.csv", "MTeamCoaches.csv", "MTeamConferences.csv",
    "MTeams.csv", "MTeamSpellings.csv", "SampleSubmissionStage1.csv",
    "SampleSubmissionStage2.csv", "WConferenceTourneyGames.csv",
    "WGameCities.csv", "WNCAATourneyCompactResults.csv",
    "WNCAATourneyDetailedResults.csv", "WNCAATourneySeeds.csv",
    "WNCAATourneySlots.csv", "WRegularSeasonCompactResults.csv",
    "WRegularSeasonDetailedResults.csv", "WSeasons.csv",
    "WSecondaryTourneyCompactResults.csv", "WSecondaryTourneyTeams.csv",
    "WTeamConferences.csv", "WTeams.csv", "WTeamSpellings.csv",
]


def first_last_value(path: Path, column_name: str):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        first = None
        last = None
        for row in reader:
            value = row[column_name]
            if first is None:
                first = value
            last = value
    return first or "", last or ""


def parse_submission_counts(path: Path):
    total = 0
    men = 0
    women = 0
    seasons = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            season, t1, _ = row["ID"].split("_")
            seasons.add(season)
            if int(t1) < 2000:
                men += 1
            elif int(t1) >= 3000:
                women += 1
    return total, men, women, sorted(seasons)


def main() -> None:
    actual_files = sorted(path.name for path in DATA_DIR.glob("*.csv"))
    missing = sorted(set(EXPECTED_FILES) - set(actual_files))
    extra = sorted(set(actual_files) - set(EXPECTED_FILES))

    print("March Machine Learning Mania 2026 data check")
    print(f"Data directory: {DATA_DIR}")
    print()
    print(f"Expected CSV files: {len(EXPECTED_FILES)}")
    print(f"Actual CSV files:   {len(actual_files)}")
    print(f"Missing files:      {missing if missing else 'None'}")
    print(f"Extra files:        {extra if extra else 'None'}")
    print()

    m_seasons = first_last_value(DATA_DIR / "MSeasons.csv", "Season")
    w_seasons = first_last_value(DATA_DIR / "WSeasons.csv", "Season")
    m_tourney = first_last_value(DATA_DIR / "MNCAATourneyCompactResults.csv", "Season")
    w_tourney = first_last_value(DATA_DIR / "WNCAATourneyCompactResults.csv", "Season")
    m_seeds = first_last_value(DATA_DIR / "MNCAATourneySeeds.csv", "Season")
    w_seeds = first_last_value(DATA_DIR / "WNCAATourneySeeds.csv", "Season")

    print("Season coverage")
    print(f"  MSeasons:                    {m_seasons[0]} -> {m_seasons[1]}")
    print(f"  WSeasons:                    {w_seasons[0]} -> {w_seasons[1]}")
    print(f"  MNCAATourneyCompactResults:  {m_tourney[0]} -> {m_tourney[1]}")
    print(f"  WNCAATourneyCompactResults:  {w_tourney[0]} -> {w_tourney[1]}")
    print(f"  MNCAATourneySeeds:           {m_seeds[0]} -> {m_seeds[1]}")
    print(f"  WNCAATourneySeeds:           {w_seeds[0]} -> {w_seeds[1]}")
    print()

    stage1_total, stage1_men, stage1_women, stage1_seasons = parse_submission_counts(DATA_DIR / "SampleSubmissionStage1.csv")
    stage2_total, stage2_men, stage2_women, stage2_seasons = parse_submission_counts(DATA_DIR / "SampleSubmissionStage2.csv")

    print("Submission templates")
    print(f"  Stage 1 rows: {stage1_total}  men={stage1_men}  women={stage1_women}  seasons={stage1_seasons}")
    print(f"  Stage 2 rows: {stage2_total}  men={stage2_men}  women={stage2_women}  seasons={stage2_seasons}")
    print()

    print("Update reminders")
    if m_seeds[1] != "2026":
        print("  - Men's 2026 seeds are not present yet. Re-download before final submission.")
    if w_seeds[1] != "2026":
        print("  - Women's 2026 seeds are not present yet. Re-download before final submission.")
    if missing:
        print("  - Some official CSV files are missing.")
    else:
        print("  - Official CSV set looks complete.")
    print("  - Final competition deadline is March 19, 2026 16:00 UTC.")


if __name__ == "__main__":
    main()
