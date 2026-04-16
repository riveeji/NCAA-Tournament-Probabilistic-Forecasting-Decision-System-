# JI_base Women Slice Diagnostics

聚焦 women 冻结基线最弱的 slice，按关键特征拆误差相关性和高误差样本均值。

## upset_bucket = upset_gap2plus

- Rows: `214`
- Calibrated Brier: `0.555447805`
- Avg calibrated prob: `0.501073`
- Empirical win rate: `0.574766`

| Feature | Mean | High-error Mean | Error Corr | Brier Corr |
| --- | ---: | ---: | ---: | ---: |
| Seed_x_Quality | -1.444575 | -3.618560 | -0.663021 | -0.683239 |
| Delta_Seed | 0.742991 | -1.023256 | -0.218545 | -0.220329 |
| WomenCompositeQuality_diff | -0.027962 | 0.089404 | 0.175753 | 0.185741 |
| OpponentQualityTournamentRank_diff | -0.083457 | 0.074156 | 0.153885 | 0.161170 |
| Delta_Elo | -3.380129 | 47.842245 | 0.096212 | 0.117146 |
| Delta_Quality | 0.007016 | 0.135341 | 0.054113 | 0.073419 |
| QualityWins_diff | -0.044103 | 0.030092 | 0.051150 | 0.056773 |
| AvgBlkDiff_diff | -0.041046 | 0.017988 | -0.013687 | -0.006979 |

## upset_bucket = tossup

- Rows: `376`
- Calibrated Brier: `0.237370032`
- Avg calibrated prob: `0.548593`
- Empirical win rate: `0.539894`

| Feature | Mean | High-error Mean | Error Corr | Brier Corr |
| --- | ---: | ---: | ---: | ---: |
| Seed_x_Quality | -0.150533 | -0.161524 | 0.249941 | 0.135732 |
| Delta_Quality | 0.063406 | 0.074658 | -0.184329 | -0.105647 |
| AvgBlkDiff_diff | -0.028263 | -0.166309 | -0.152864 | -0.123699 |
| Delta_Elo | 14.453692 | 25.409839 | -0.130115 | -0.053300 |
| Delta_Seed | 0.039894 | -0.157895 | 0.073228 | 0.026923 |
| WomenCompositeQuality_diff | -0.005563 | 0.019800 | -0.069743 | -0.004425 |
| OpponentQualityTournamentRank_diff | -0.021226 | 0.035930 | 0.053012 | 0.085769 |
| QualityWins_diff | 0.014624 | 0.061193 | 0.026311 | 0.053994 |

## seed_gap_bucket = gap_0_1

- Rows: `376`
- Calibrated Brier: `0.237370032`
- Avg calibrated prob: `0.548593`
- Empirical win rate: `0.539894`

| Feature | Mean | High-error Mean | Error Corr | Brier Corr |
| --- | ---: | ---: | ---: | ---: |
| Seed_x_Quality | -0.150533 | -0.161524 | 0.249941 | 0.135732 |
| Delta_Quality | 0.063406 | 0.074658 | -0.184329 | -0.105647 |
| AvgBlkDiff_diff | -0.028263 | -0.166309 | -0.152864 | -0.123699 |
| Delta_Elo | 14.453692 | 25.409839 | -0.130115 | -0.053300 |
| Delta_Seed | 0.039894 | -0.157895 | 0.073228 | 0.026923 |
| WomenCompositeQuality_diff | -0.005563 | 0.019800 | -0.069743 | -0.004425 |
| OpponentQualityTournamentRank_diff | -0.021226 | 0.035930 | 0.053012 | 0.085769 |
| QualityWins_diff | 0.014624 | 0.061193 | 0.026311 | 0.053994 |

## period_bucket = recent

- Rows: `264`
- Calibrated Brier: `0.152361986`
- Avg calibrated prob: `0.475277`
- Empirical win rate: `0.477273`

| Feature | Mean | High-error Mean | Error Corr | Brier Corr |
| --- | ---: | ---: | ---: | ---: |
| Seed_x_Quality | -4.526618 | -0.857472 | 0.513439 | 0.333386 |
| OpponentQualityTournamentRank_diff | -0.130673 | 0.002475 | 0.120276 | 0.098936 |
| WomenCompositeQuality_diff | -0.082233 | -0.026629 | 0.118225 | 0.091240 |
| QualityWins_diff | -0.139390 | -0.013496 | 0.110903 | 0.085314 |
| Delta_Seed | 1.094697 | 0.415094 | -0.089460 | -0.075863 |
| Delta_Elo | -27.174081 | -9.725383 | 0.088170 | 0.056207 |
| Delta_Quality | -0.094342 | -0.066860 | 0.061363 | 0.029638 |
| AvgBlkDiff_diff | -0.374229 | -0.435567 | 0.023030 | -0.001600 |
