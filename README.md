# Kaggle NCAA March Mania 2026

This repository is now reduced to a single-submission workflow for `March Machine Learning Mania 2026`.

Current best-known local score:

- men: `0.19215`
- women: `0.13740`
- `equal_gender`: `0.16478`

Reference:

- [combined_cv_summary_20260312T040220Z.json](</j:/ide-workspace/kaggle-ncaa prediction/results/combined_cv_summary_20260312T040220Z.json>)

Recommended upload file:

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

## Core Files

- [zizzii_features.py](</j:/ide-workspace/kaggle-ncaa prediction/zizzii_features.py>)
- [zizzii_train.py](</j:/ide-workspace/kaggle-ncaa prediction/zizzii_train.py>)
- [tools/postprocess_single_final.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/postprocess_single_final.py>)
- [tools/postprocess_women_submission_b.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/postprocess_women_submission_b.py>)
- [tools/check_competition_data.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/check_competition_data.py>)
- [tools/check_submission_sanity.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/check_submission_sanity.py>)
- [tools/run_selection_sunday_pipeline.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/run_selection_sunday_pipeline.py>)

Supporting refresh utilities kept for live data:

- [tools/fetch_public_external_ratings.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_public_external_ratings.py>)
- [tools/fetch_collegehoopshub_ratings.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_collegehoopshub_ratings.py>)
- [tools/fetch_theoddsapi_odds.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_theoddsapi_odds.py>)
- [tools/fetch_teamrankings_odds.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_teamrankings_odds.py>)

Data folders:

- [ncaa-data](</j:/ide-workspace/kaggle-ncaa prediction/ncaa-data>)
- [external-data](</j:/ide-workspace/kaggle-ncaa prediction/external-data>)
- [results](</j:/ide-workspace/kaggle-ncaa prediction/results>)

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-kaggle.txt
python -m playwright install chromium
```

## Normal Workflow

1. Check official Kaggle data.

```powershell
python tools\check_competition_data.py
```

2. Train the main pipeline.

```powershell
python zizzii_train.py
```

3. Build the final one-file submission.

```powershell
python tools\postprocess_single_final.py
```

4. Run sanity checks.

```powershell
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```

5. Upload:

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

Notes:

- [submission_stage2.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2.csv>) is the raw model output.
- Only [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>) is the recommended final upload.

## Selection Sunday Workflow

When `2026` seeds and live odds are available:

```powershell
$env:THE_ODDS_API_KEY="your_key"
python tools\run_selection_sunday_pipeline.py
python tools\postprocess_single_final.py
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```

This refreshes:

- official data validation
- public external ratings
- CollegeHoopsHub ratings
- current The Odds API matchup odds
- current TeamRankings spread-derived men odds
- training outputs

## External Data

Current training already reads the kept CSVs under [external-data](</j:/ide-workspace/kaggle-ncaa prediction/external-data>), including:

- men historical odds
- men and women current matchup odds
- current team ratings
- men manual signals

Format details:

- [external-data/README.md](</j:/ide-workspace/kaggle-ncaa prediction/external-data/README.md>)

## Chinese Docs

- [README.zh-CN.md](</j:/ide-workspace/kaggle-ncaa prediction/README.zh-CN.md>)
- [SELECTION_SUNDAY_CHECKLIST.zh-CN.md](</j:/ide-workspace/kaggle-ncaa prediction/SELECTION_SUNDAY_CHECKLIST.zh-CN.md>)
