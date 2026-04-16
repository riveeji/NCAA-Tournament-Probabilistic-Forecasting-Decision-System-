# NCAA Tournament Probabilistic Forecasting & Decision System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Frozen%20Core%20%2B%20Research%20Lane-0f766e)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/riveeji/NCAA-Tournament-Probabilistic-Forecasting-Decision-System-)

![Project Banner](docs/assets/project-banner.svg)

An end-to-end forecasting, decision, and submission system for Kaggle NCAA March Machine Learning Mania.

This repository is not a single notebook and not a one-off model export. It is a full competition engineering stack covering:

- multi-source ingestion
- team-level and matchup-level feature construction
- men / women split modeling paths
- replay, benchmark, registry, and official leaderboard sanity checks
- submission building, validation, and postmortem documentation

Chinese homepage: [README.md](README.md)

## Current Production State

The production lane is currently frozen at `JI_base`:

- frozen core: `core::lr_carry_elo_definition_v1`
- model family: `JI_lr_control`
- feature profile: `lr_carry_elo_definition_v1`
- alpha profile: `quality_only_men_quality_blocks_women`
- women quality profile: `consensus_rebuild_v4`
- best official LB: `0.1231313`

Reference documents:

- [docs/JI_BASE_PHASE_STATUS.md](docs/JI_BASE_PHASE_STATUS.md)
- [docs/JI_NEXT_ARCH_PHASE1.md](docs/JI_NEXT_ARCH_PHASE1.md)

The current system-level conclusion is straightforward:

- the production core is already strong
- most pure architecture gains have been exhausted
- the next meaningful upside is more likely to come from richer signals than from replacing the core model

## Figures

### System architecture

![System Architecture](docs/assets/system-architecture.svg)

### Replay / CV trend

![Historical CV Brier Trend](docs/assets/cv-trend.svg)

### Upset stress test

![Upset Scenario Brier](docs/assets/upset-scenarios.svg)

## What This Repository Actually Builds

Typical competition repos stop at “train model, export submission.” This project is built closer to a production forecasting system:

1. Data layer  
   Aggregates official NCAA data, public ratings, matchup projections, odds, prediction markets, and availability-style signals.
2. Feature layer  
   Standardizes team-season, team-game, and matchup views into reusable structured features.
3. Model layer  
   Keeps a stable production core while allowing challengers, upstream providers, sidecars, and next-arch experiments to run in isolation.
4. Governance layer  
   Uses replay gates, registries, benchmark reports, and official LB sanity checks before promoting changes.
5. Release layer  
   Produces candidate submissions, summaries, manifests, hashes, and written postmortems.

## Core Capabilities

- supports both men's and women's tournament forecasting
- targets the full Kaggle `132,133` matchup submission space
- keeps men / women as first-class separate paths
- supports replay, benchmark, slice diagnostics, and promotion governance
- integrates multiple external signal families:
  - odds
  - prediction markets
  - public ratings
  - matchup projections
  - women historical consensus snapshots
- separates:
  - `hc/ji_base` for stable production
  - `hc/next_arch` for replay-first research

## Main Findings So Far

### Promoted structural upgrades

- `lr_pruned_core_v1`
- `lr_carry_elo_definition_v1`

### Rejected or paused directions

- `Colley` conference downweight
- `SRS` clipping
- women internal-only feature reshaping
- `TabR`
- pairwise-only neural heads
- graph-only architectures
- standalone and hybrid transformer families
- narrow gender-specific stacker v1

### Current read

- the main bottleneck is no longer the core head/backbone
- richer sidecars and stronger upstream signals are the most plausible next step
- those directions need stronger historical and current-year data coverage to be meaningful

## Repository Layout

```text
.
|-- hc/
|   |-- ji_base/            # frozen production core
|   |-- next_arch/          # replay-first new architecture experiments
|   |-- gold/               # historical / baseline support code
|   `-- v2/                 # v2 rebuild / replay lane
|-- tools/                  # ingestion, builders, benchmark, snapshots, submission tooling
|-- tests/                  # production and research tests
|-- docs/                   # phase reports, specs, plans, benchmark docs
|   `-- assets/             # README figures
|-- results/                # tracked summaries only; most generated files stay local
|-- zizzii_features.py      # legacy-compatible baseline feature logic
|-- README.md
`-- README.en.md
```

## Key Documents

- [docs/JI_BASE_PHASE_STATUS.md](docs/JI_BASE_PHASE_STATUS.md)  
  production baseline, frozen scope, promoted vs paused directions
- [docs/JI_NEXT_ARCH_PHASE1.md](docs/JI_NEXT_ARCH_PHASE1.md)  
  replay results for `TabR`, `pairwise`, `graph`, `transformer`, and stacker experiments
- [docs/JI_BASE_BENCHMARK.md](docs/JI_BASE_BENCHMARK.md)  
  core challenger benchmark summary
- [docs/JI_BASE_WOMEN_SLICE_DIAGNOSTICS.md](docs/JI_BASE_WOMEN_SLICE_DIAGNOSTICS.md)  
  women slice diagnosis
- [docs/INTERVIEW_NOTES.zh-CN.md](docs/INTERVIEW_NOTES.zh-CN.md)  
  recruiting/interview-oriented project explanation

## Quick Start

### 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-hc.txt
```

### 2. Run replay / postmortem

```powershell
.\.venv\Scripts\python.exe tools\run_v2_baseline_replay.py
.\.venv\Scripts\python.exe tools\build_postmortem_report.py
```

### 3. Run a `JI_base` challenger

```powershell
.\.venv\Scripts\python.exe tools\run_ji_base_challenger.py --candidate-name core::women_ranking_historical_snapshots_v1
```

### 4. Run a next-arch challenger

```powershell
.\.venv\Scripts\python.exe tools\run_next_arch_challenger.py --candidate-name arch::gender_specific_stacker_v1
```

## Data and Reproducibility Notes

The public GitHub repository intentionally does **not** commit:

- `external-data/`
- `ncaa-data/`
- most generated `results/`
- local submission CSVs

That is deliberate:

- data volume is large
- some sources are time-sensitive or usage-constrained
- many files are local caches or one-off generated artifacts

This repository is therefore designed to preserve:

- source code
- workflow
- tracked summaries
- phase conclusions

not a full raw-data dump.

## Why This Repo Is Worth Studying

- it is a full competition system, not a single model
- men / women split design is built into the architecture
- it uses replay gates and official sanity checks instead of leaderboard-only iteration
- it cleanly separates production and research lanes
- it is useful as a case study in:
  - sports analytics engineering
  - probabilistic forecasting
  - calibration-aware evaluation
  - experiment governance
  - competition release workflow design

## Where This Goes Next

The most defensible next step is not another backbone swap. It is:

- keeping the current production core frozen
- improving richer sidecars and upstream signals
- reopening, closer to next season:
  - men player-level injury / value sidecars
  - market / prediction-market sidecars
  - stronger women upstreams
  - gender-specific stacking once those sidecars are actually strong

## License

Released under the [MIT License](LICENSE).
