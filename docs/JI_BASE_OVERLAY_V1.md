# JI_base Overlay v1

`JI_base_overlay` v1 is a submission-only layer built on top of the frozen `JI_base` base submission.

## Scope

- Direct matchup market only
- Men structured injury only
- Full audit and summary outputs
- No futures
- No sharpen

## Inputs

- Base submission: `results/submission_stage2_ji_base.csv`
- Frozen baseline snapshot: `results/ji_base_baseline_snapshot.json`
- Direct market sources from the existing `gold` overlay data loaders
- Men injury file from the existing `gold` overlay data loader

## Outputs

- Overlay submission: `results/submission_stage2_ji_base_overlay.csv`
- Overlay audit: `results/ji_base_overlay_audit.csv`
- Overlay summary: `results/ji_base_overlay_summary.json`
- Overlay candidate summary: `results/ji_base_overlay_candidates_summary.csv`

## Default behavior

- Men: `direct_priority` market + injury
- Women: `direct_priority` market only
- Base submission builder stays unchanged
- Overlay results are produced by `tools/build_ji_base_overlay_submission.py`

## Governance

- Replay does not promote overlay
- Official LB decides whether overlay is worth keeping
- Even if overlay wins LB, it remains submission-only and does not replace the frozen `JI_base` core
