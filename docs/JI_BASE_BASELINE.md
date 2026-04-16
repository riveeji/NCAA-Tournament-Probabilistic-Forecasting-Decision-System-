# JI_base Baseline Snapshot

- Snapshot date: `2026-04-13`
- Working baseline candidate: `core::lr_carry_elo_definition_v1`
- Submission profile: `ji_base_base`
- Base model: `JI_lr_control`
- Calibration: `none`
- Feature profile: `lr_carry_elo_definition_v1`
- Alpha profile: `quality_only_men_quality_blocks_women`
- Men quality profile: `legacy_v1`
- Women quality profile: `consensus_rebuild_v4`

## Official LB

- Frozen core LB: `0.1231313`
- Logged on: `2026-04-13`
- Notes: lr_carry_elo_definition_v1 frozen core baseline
- Frozen overlay submission: `ji_base_overlay_v1_men_best_women_direct_only_weight025`
- Best-known overlay submission: `ji_base_overlay_v1_men_best_women_direct_only_weight020` / `0.1271633`

## Replay Position

- `ji_base_vs_old_hc_delta`: `0.0006084874757513736`
- `ji_base_vs_gold_recover_delta`: `-0.0007832043918782616`

## Freeze Rules

- Keep as default:
  - `JI_lr_control`
  - `lr_carry_elo_definition_v1`
  - `quality_only_men_quality_blocks_women`
  - `legacy_v1 men quality`
  - `consensus_rebuild_v4 women quality`
- Exclude from default replay:
  - `JI_node_control`
  - `JI_tabr_control`
  - `overlay`
  - `current-year market/injury/futures`

- Submission output: `J:\ide-workspace\kaggle-ncaa prediction\results\submission_stage2_ji_base_carryelo85.csv`
- JSON snapshot: `J:\ide-workspace\kaggle-ncaa prediction\results\ji_base_baseline_snapshot.json`
