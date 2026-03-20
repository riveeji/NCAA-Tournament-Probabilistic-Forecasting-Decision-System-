# NCAA Tournament Probabilistic Forecasting & Decision System

An end-to-end forecasting, decision, and submission pipeline for Kaggle NCAA March Machine Learning Mania.

This project is not a single notebook or a one-off model export. It is a full tournament forecasting system that covers:

- multi-source data ingestion
- structured team-level and matchup-level signal construction
- probabilistic modeling and meta fusion
- runtime market-aware adjustments
- candidate submission generation and recommendation
- sanity checks, hashing, and release artifacts

## Highlights

- Supports both **men's and women's** NCAA tournament forecasting
- Generates probabilities for the full `132,133` matchup submission space
- Integrates sportsbook odds, prediction markets, external ratings, matchup models, and injury/availability signals
- Includes historical CV, replay-oriented evaluation, and scenario-based Brier simulations
- Provides a bounded final decision layer (`goldshot`) for high-leverage real matchups

## Tech Stack

- Python
- Pandas / NumPy
- scikit-learn
- XGBoost / CatBoost / TabPFN (optional)
- Optuna
- requests / BeautifulSoup / Playwright
- rapidfuzz

## Modeling Approach

- Elo-style and efficiency-based structured features
- strength of schedule, recent form, seed gap, host/site features
- market-implied probability and spread features
- multi-route modeling and stacking / meta fusion
- bounded post-processing driven by market + model consensus
- Brier-score-based validation, replay checks, and scenario simulation

## Data Sources

The repo is designed to work with a mix of:

- official NCAA data
- sportsbook odds
- prediction markets
- external ratings and matchup projections
- manual supplements

Examples used in the pipeline include The Odds API, Silver Bulletin, Bart Torvik, Warren Nolan, Her Hoop Stats, Kalshi, and Polymarket.

## Repository Structure

- `hc/`: forecasting core, loaders, training, prediction logic
- `tools/`: ingestion scripts, pipeline runners, evaluation, release helpers
- `data/`: intermediate project data
- `results/`: generated reports and release artifacts
- `external-data/`: pulled and standardized third-party signals

## Quick Start

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-hc.txt
```

Run the Selection Sunday pipeline:

```powershell
.\.venv\Scripts\python.exe tools\run_selection_sunday_pipeline.py
```

## Notes

- The public GitHub version excludes large local data folders and result dumps by default.
- Figures shown in the README are generated from tracked artifacts via `tools/build_readme_figures.py`.
- The Chinese homepage README is in [README.md](README.md).

## License

Released under the [MIT License](LICENSE).
