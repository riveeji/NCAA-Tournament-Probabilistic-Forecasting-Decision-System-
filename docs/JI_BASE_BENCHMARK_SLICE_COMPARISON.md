# JI_base Benchmark Slice Comparison

## Systems

- `ji_base_frozen`: frozen baseline, official LB `0.1278438`
- `gold_recover_proxy`: `gold_linear@none[current_default]` replay proxy; official LB reference for market submission is `0.1289`
- `old_hc`: replay-only reference `0.16027528186298307`; omitted from slice comparison because there is no aligned prediction-level artifact

## Latest Slice Head-to-Head

| Gender | JI_base | Gold proxy | Delta (JI - Gold) | Winner |
| --- | ---: | ---: | ---: | --- |
| ALL | 0.126990891 | 0.125762074 | 0.001228817 | gold_recover_proxy |
| M | 0.142475556 | 0.142578449 | -0.000102893 | ji_base_frozen |
| W | 0.111506226 | 0.108945699 | 0.002560526 | gold_recover_proxy |

## Strongest JI_base Slices

| Gender | Slice Type | Slice | JI_base | Gold proxy | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| M | period_bucket | recent | 0.208051102 | 0.211964230 | -0.003913128 |
| M | seed_gap_bucket | gap_0_1 | 0.238838377 | 0.242254926 | -0.003416549 |
| M | upset_bucket | tossup | 0.238838377 | 0.242254926 | -0.003416549 |
| M | seed_gap_bucket | gap_2_4 | 0.223792500 | 0.225909781 | -0.002117281 |
| W | upset_bucket | favorite_win_gap2plus | 0.034218516 | 0.035519740 | -0.001301224 |
| M | favorite_seed_bucket | seed_1_2 | 0.143702365 | 0.144531934 | -0.000829569 |
| ALL | upset_bucket | favorite_win_gap2plus | 0.053862446 | 0.054403486 | -0.000541040 |
| W | seed_gap_bucket | gap_9_plus | 0.016563408 | 0.016941931 | -0.000378523 |

## Weakest JI_base Slices

| Gender | Slice Type | Slice | JI_base | Gold proxy | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| W | upset_bucket | upset_gap2plus | 0.555447805 | 0.537427800 | 0.018020005 |
| W | seed_gap_bucket | gap_0_1 | 0.237370032 | 0.226091840 | 0.011278192 |
| W | upset_bucket | tossup | 0.237370032 | 0.226091840 | 0.011278192 |
| ALL | upset_bucket | upset_gap2plus | 0.500387238 | 0.491439616 | 0.008947621 |
| W | favorite_seed_bucket | seed_5_8 | 0.219189429 | 0.213826387 | 0.005363042 |
| W | period_bucket | recent | 0.152361986 | 0.147021531 | 0.005340455 |
| M | upset_bucket | upset_gap2plus | 0.477684037 | 0.472477244 | 0.005206793 |
| M | seed_gap_bucket | gap_5_8 | 0.200582738 | 0.195958465 | 0.004624273 |
