# Current Baseline

Baseline frozen on 2026-03-12 after:

- removing aggressive probability clipping for Brier scoring
- rolling back men `elo_dynamics` and `adjusted_efficiency`
- keeping women on the rollback-safe feature set
- adding `xgb_margin` as a spread regressor with smooth logistic probability conversion

Primary reference artifact:

- `results/combined_cv_summary_20260312T115339Z.json`

Current best local CV:

- Men Brier: `0.18801117917279417`
- Women Brier: `0.13902863035611718`
- Equal-gender mean: `0.16351990476445566`
- Historical matchup-weighted: `0.1684614264256453`
