# JI_base Benchmark Report

## Frozen Baseline

- Candidate: `core::lr_carry_elo_definition_v1`
- Model: `JI_lr_control`
- Calibration: `none`
- Feature profile: `lr_carry_elo_definition_v1`
- Alpha profile: `quality_only_men_quality_blocks_women`
- Men quality: `legacy_v1`
- Women quality: `consensus_rebuild_v4`
- Frozen baseline official LB: `0.1231313`
- Frozen overlay submission: `ji_base_overlay_v1_men_best_women_direct_only_weight025`
- Best-known submission layer: `ji_base_base` / `0.1231313`

## Replay Benchmark

| System | Total CV Brier (calibrated) | Official LB | Source |
| --- | ---: | ---: | --- |
| ji_base_frozen | 0.160925279 | 0.1231313 | frozen_baseline |
| gold_recover_market | 0.161708484 | 0.1289000 | postmortem_delta+official_lb_log |
| old_hc | 0.160316792 |  | postmortem_delta |

## Frozen Baseline Metrics

- `total_cv_brier_calibrated`: `0.160925279`
- `women_cv_brier_calibrated`: `0.139254270`
- `latest_season_equal_gender_brier`: `0.124136830`
- `recent_window_equal_gender_brier`: `0.168369183`

## Challenger Registry

| Candidate | Status | Action | Official LB | Total delta | Women delta |
| --- | --- | --- | ---: | ---: | ---: |
| core::women_ranking_upstream_v1_internal_refactor | equivalent | freeze_no_change |  | 0.000000000 | 0.000000000 |
| core::women_ranking_upstream_v2_internal_refactor | equivalent | freeze_no_change |  | 0.000000000 | 0.000000000 |
| calibration isotonic gender min100 | rejected | archive_direction |  | 0.003967011 | 0.005923741 |
| calibration isotonic gender min50 | rejected | archive_direction |  | 0.003967011 | 0.005923741 |
| calibration none rerun | rejected | archive_direction |  | 0.002899157 | 0.004414220 |
| core::lr_colley_definition_v1 | rejected | archive_direction |  | 0.000020388 | 0.000044615 |
| core::lr_pruned_only_v1 | rejected | archive_direction |  | 0.002130740 | 0.002949127 |
| core::lr_ratings_core_v2a | rejected | archive_direction |  | 0.002221764 | 0.002813375 |
| core::lr_ratings_core_v2b | rejected | archive_direction |  | -0.000041510 | 0.000050579 |
| core::lr_ratings_core_v2c | rejected | archive_direction |  | 0.000124870 | 0.000179284 |
| core::lr_ratings_definition_v1 | rejected | archive_direction |  | 0.002101648 | 0.002537374 |
| core::lr_ratings_only_v1 | rejected | archive_direction |  | 0.000816044 | 0.001601226 |
| core::lr_srs_definition_confirm20 | rejected | archive_direction |  | 0.000385290 | 0.000691688 |
| core::lr_srs_definition_v1_clip15 | rejected | archive_direction |  | 0.001071438 | 0.001545072 |
| core::lr_women_fix_only_v1 | rejected | archive_direction |  | 0.002909845 | 0.004435596 |
| core::seed_quality_interaction_regression_check | rejected | archive_direction |  | 0.002899157 | 0.004414220 |
| core::women_opp_rank_redesign_v1_architecture | rejected | archive_direction |  | 0.000124894 | 0.000249789 |
| core::women_qualitywins_redesign_v1_architecture | rejected | archive_direction |  | 0.000001281 | 0.000002563 |
| core::women_slice_redesign_v1_architecture | rejected | archive_direction |  | 0.000083145 | 0.000166291 |
| feature tossup upset v1 | rejected | archive_direction |  | 0.003297284 | 0.004755539 |
| quality_only_men_core_women challenge | rejected | archive_direction |  | 0.002920279 | 0.004456464 |
| women quality v4a conservative | rejected | archive_direction |  | 0.002903599 | 0.004423104 |
| women quality v4b conservative | rejected | archive_direction |  | 0.002899157 | 0.004414220 |
| women seed quality conservative | rejected | archive_direction |  | 0.002973356 | 0.004562617 |
| core::lr_carry_elo_definition_confirm80 | replay_passed | eligible_for_official_lb |  | 0.000015869 | 0.000001650 |
| core::lr_carry_elo_definition_v1 | replay_passed | eligible_for_official_lb |  | 0.000000000 | 0.000000000 |
| core::lr_pruned_core_v1 | replay_passed | eligible_for_official_lb |  | 0.000051012 | 0.000031567 |
| core::women_ranking_upstream_v2_external_consensus | replay_passed | eligible_for_official_lb |  | -0.000013158 | -0.000026317 |
| women tossup quality conservative | replay_passed | eligible_for_official_lb |  | 0.002850589 | 0.004317083 |
| core::lr_regularization_confirm_m10_w05 | replay_passed_but_lb_failed | pause_direction | 0.1232169 | -0.000041057 | -0.000082114 |
| core::lr_regularization_v1_c07 | replay_passed_but_lb_failed | pause_direction | 0.1232034 | -0.000035798 | -0.000038680 |
| core::women_ranking_upstream_v1_external_consensus | replay_passed_but_lb_failed | pause_direction | 0.1231352 | -0.000011269 | -0.000022537 |
