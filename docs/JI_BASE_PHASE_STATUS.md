# JI_base Phase Status

## Current Frozen Core

- model family: `JI_lr_control`
- feature profile: `lr_carry_elo_definition_v1`
- alpha profile: `quality_only_men_quality_blocks_women`
- women quality profile: `consensus_rebuild_v4`
- official LB: `0.1231313`

## Promoted Structural Upgrades

- `lr_pruned_core_v1`
- `lr_carry_elo_definition_v1`

## Rejected Or Frozen Directions

- `Colley` conference downweight
- `SRS` clipping (`clip15`, `clip20`)
- women internal-only `consensus/composite` rebuild
- women internal-only `OpponentQualityTournamentRank` rebuild
- women internal-only `QualityWins` rebuild
- replay-passed `LR` regularization variants that lost the official LB sanity check
- `women_ranking_upstream_v1_external_consensus`: replay-passed, official LB `0.1231352`, did not beat frozen core `0.1231313`; paused without promotion

## Current Read

- The internal-only core is mature and currently near its local ceiling.
- The remaining high-value women work is upstream input architecture, not more feature reshaping.
- Overlay is paused. It is not part of the current improvement loop.
- Hybrid-Ready women upstream is viable as architecture, but `v1` is not strong enough yet to replace the frozen baseline.

## Next Phase Focus

1. Add a Hybrid-Ready women ranking upstream provider with `internal_fallback` and `external_consensus_v1`.
2. Validate `consensus_rebuild_v6` through staged challengers without changing the frozen baseline.
3. Only consider promotion if replay improves and a 2026 official LB sanity check beats `0.1231313`.
