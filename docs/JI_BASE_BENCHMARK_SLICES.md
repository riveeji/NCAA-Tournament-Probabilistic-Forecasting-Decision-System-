# JI_base Benchmark Slices

- Frozen candidate: `alpha::quality_only_men_quality_blocks_women`
- Model: `JI_lr_control`
- Feature profile: `seed_quality_interaction`
- Alpha profile: `quality_only_men_quality_blocks_women`
- Women quality profile: `consensus_rebuild_v4`

## Latest / Recent

| Gender | Slice | Rows | Calibrated Brier |
| --- | --- | ---: | ---: |
| ALL | latest | 134 | 0.126990891 |
| M | latest | 67 | 0.142475556 |
| W | latest | 67 | 0.111506226 |

## Worst Slices

| Gender | Slice Type | Slice | Rows | Calibrated Brier | Avg Seed Gap |
| --- | --- | --- | ---: | ---: | ---: |
| W | upset_bucket | upset_gap2plus | 214 | 0.555447805 | 4.827 |
| M | upset_bucket | upset_gap2plus | 519 | 0.477684037 | 5.838 |
| M | seed_gap_bucket | gap_0_1 | 511 | 0.238838377 | 0.810 |
| M | upset_bucket | tossup | 511 | 0.238838377 | 0.810 |
| W | seed_gap_bucket | gap_0_1 | 376 | 0.237370032 | 0.880 |
| W | upset_bucket | tossup | 376 | 0.237370032 | 0.880 |
| M | favorite_seed_bucket | seed_5_8 | 721 | 0.236292585 | 4.194 |
| M | seed_gap_bucket | gap_2_4 | 526 | 0.223792500 | 3.148 |
