# NCAA Tournament Probabilistic Forecasting & Decision System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Project-0f766e)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/riveeji/NCAA-Tournament-Probabilistic-Forecasting-Decision-System-)

![Project Banner](docs/assets/project-banner.svg)

An end-to-end forecasting, decision, and submission pipeline for Kaggle NCAA March Machine Learning Mania.

This project is not a single notebook or a one-off model export. It is a full tournament forecasting system that covers:

- multi-source data ingestion
- structured team-level and matchup-level signal construction
- probabilistic modeling and meta fusion
- runtime market-aware adjustments
- candidate submission generation and recommendation
- sanity checks, hashing, and release artifacts

## Why This Project Is Interesting

This repository is stronger than a typical competition notebook because it combines:

- a full submission-space probability engine
- market-aware decision logic close to lock time
- cross-source ingestion from messy real-world feeds
- replay / CV / scenario-based evaluation
- a release workflow with recommendation, sanity checks, and artifact tracking

## Why Star This Repo

- It is a full **submission-ready forecasting system**, not just a training notebook
- It combines **sportsbook odds, prediction markets, external matchup models, and structured team signals**
- It supports both **men's and women's** tournament forecasting in one pipeline
- It includes **decision-layer logic, release validation, and reproducible artifacts**
- It is useful both as a Kaggle competition repo and as a practical ML systems portfolio project

If this repo is useful for your Kaggle workflow, sports analytics learning, or ML systems portfolio, consider giving it a star.

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

## Recruiting-Oriented Summary

This project demonstrates:

- end-to-end machine learning system design
- probabilistic forecasting and calibration
- multi-source data engineering under unstable external inputs
- practical decision-layer design instead of model-only experimentation
- reproducible release workflows for high-pressure submission windows

## Notes

- The public GitHub version excludes large local data folders and result dumps by default.
- Figures shown in the README are generated from tracked artifacts via `tools/build_readme_figures.py`.
- The Chinese homepage README is in [README.md](README.md).
- Chinese interview notes are in [docs/INTERVIEW_NOTES.zh-CN.md](docs/INTERVIEW_NOTES.zh-CN.md).
- GitHub growth and sharing notes are in [docs/GITHUB_GROWTH_PLAYBOOK.zh-CN.md](docs/GITHUB_GROWTH_PLAYBOOK.zh-CN.md).

## License

Released under the [MIT License](LICENSE).
