# JI_base Women Slice System Comparison

Compare the same women games under `JI_base` frozen baseline and `gold_recover_proxy`, then locate the feature regions where `JI_base` loses on calibrated Brier.

## upset_bucket = upset_gap2plus

- Rows: `214`
- `ji_calibrated_brier`: `0.555447805`
- `gold_calibrated_brier`: `0.537427800`
- `ji_minus_gold_brier_mean`: `0.018020005`
- `ji_worse_rate`: `0.570093`

| Feature | Mean | JI-worse Mean | Delta Corr |
| --- | ---: | ---: | ---: |
| Delta_Seed | 0.742991 | 0.278689 | -0.117971 |
| Seed_x_Quality | -1.444575 | -1.588440 | -0.073623 |
| Delta_Elo | -3.380129 | -9.188717 | -0.068028 |
| Delta_Quality | 0.007016 | -0.009878 | -0.066884 |
| OpponentQualityTournamentRank_diff | -0.083457 | -0.052815 | 0.054882 |
| QualityWins_diff | -0.044103 | -0.057922 | -0.046513 |
| WomenCompositeQuality_diff | -0.027962 | -0.016463 | 0.043032 |
| AvgBlkDiff_diff | -0.041046 | -0.044064 | -0.026249 |

## upset_bucket = tossup

- Rows: `376`
- `ji_calibrated_brier`: `0.237370032`
- `gold_calibrated_brier`: `0.226091840`
- `ji_minus_gold_brier_mean`: `0.011278192`
- `ji_worse_rate`: `0.593085`

| Feature | Mean | JI-worse Mean | Delta Corr |
| --- | ---: | ---: | ---: |
| QualityWins_diff | 0.014624 | 0.016201 | 0.060958 |
| AvgBlkDiff_diff | -0.028263 | -0.015844 | -0.047084 |
| Delta_Elo | 14.453692 | 17.687481 | 0.036306 |
| WomenCompositeQuality_diff | -0.005563 | 0.004221 | 0.015965 |
| OpponentQualityTournamentRank_diff | -0.021226 | -0.015143 | 0.014318 |
| Seed_x_Quality | -0.150533 | -0.156774 | 0.005439 |
| Delta_Quality | 0.063406 | 0.070709 | 0.001974 |
| Delta_Seed | 0.039894 | 0.022422 | -0.001529 |

## seed_gap_bucket = gap_0_1

- Rows: `376`
- `ji_calibrated_brier`: `0.237370032`
- `gold_calibrated_brier`: `0.226091840`
- `ji_minus_gold_brier_mean`: `0.011278192`
- `ji_worse_rate`: `0.593085`

| Feature | Mean | JI-worse Mean | Delta Corr |
| --- | ---: | ---: | ---: |
| QualityWins_diff | 0.014624 | 0.016201 | 0.060958 |
| AvgBlkDiff_diff | -0.028263 | -0.015844 | -0.047084 |
| Delta_Elo | 14.453692 | 17.687481 | 0.036306 |
| WomenCompositeQuality_diff | -0.005563 | 0.004221 | 0.015965 |
| OpponentQualityTournamentRank_diff | -0.021226 | -0.015143 | 0.014318 |
| Seed_x_Quality | -0.150533 | -0.156774 | 0.005439 |
| Delta_Quality | 0.063406 | 0.070709 | 0.001974 |
| Delta_Seed | 0.039894 | 0.022422 | -0.001529 |

## period_bucket = recent

- Rows: `264`
- `ji_calibrated_brier`: `0.152361986`
- `gold_calibrated_brier`: `0.147021531`
- `ji_minus_gold_brier_mean`: `0.005340455`
- `ji_worse_rate`: `0.537879`

| Feature | Mean | JI-worse Mean | Delta Corr |
| --- | ---: | ---: | ---: |
| Seed_x_Quality | -4.526618 | -3.130133 | 0.076621 |
| Delta_Quality | -0.094342 | 0.035739 | -0.027721 |
| Delta_Elo | -27.174081 | 6.863606 | -0.016598 |
| WomenCompositeQuality_diff | -0.082233 | -0.015353 | -0.013320 |
| AvgBlkDiff_diff | -0.374229 | -0.126507 | 0.012283 |
| QualityWins_diff | -0.139390 | -0.035346 | 0.004327 |
| Delta_Seed | 1.094697 | 0.232394 | 0.004322 |
| OpponentQualityTournamentRank_diff | -0.130673 | -0.039666 | -0.003450 |
