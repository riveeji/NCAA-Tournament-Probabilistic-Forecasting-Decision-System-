# JI Next-Arch Phase 1

## Scope

Production lane remains frozen:

- `core::lr_carry_elo_definition_v1`
- `JI_lr_control`
- `feature_profile = lr_carry_elo_definition_v1`
- `alpha_profile = quality_only_men_quality_blocks_women`
- best official LB: `0.1231313`

This phase only validates two replay-first experimental architectures:

- `arch::tabr_v1`
- `arch::tabr_hybrid_v1`
- `arch::tabr_feature_fusion_v1`
- `arch::pairwise_ranking_v1`
- `arch::season_encoder_transformer_v1`
- `arch::graph_static_embedding_v1`

Neither experiment mutates `JI_base` governance or baseline files.

## Implemented

Experimental lane:

- `hc/next_arch/config.py`
- `hc/next_arch/data.py`
- `hc/next_arch/models.py`
- `hc/next_arch/replay.py`
- `tools/run_next_arch_challenger.py`

Implemented behaviors:

- `TabR-style` minimal tabular transformer using frozen-core replay features.
- `TabR hybrid` residual model using frozen-core fold-safe baseline logits.
- `TabR feature fusion` model using frozen-core features plus fold-safe baseline logits as direct fused inputs.
- `Pairwise ranking` MLP head using the same frozen-core matchup diff features, but replacing the LR head with a small neural pairwise scorer.
- `Season encoder transformer` that converts regular-season compact game sequences into team embeddings, then uses three embedding-derived matchup features with a lightweight LR head.
- Season-safe static graph embedding using regular-season compact results only.
- Independent replay challenger outputs under `results/next_arch_challenger_*.json`.

## Results

### Frozen baseline

- `total_cv_brier_calibrated = 0.1609253`
- `women_cv_brier_calibrated = 0.1392543`
- `latest = 0.1241368`
- `recent = 0.1683692`

### `arch::tabr_v1`

- result file:
  - `results/next_arch_challenger_arch_tabr_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.1623529`
  - `women_cv_brier_calibrated = 0.1416918`
  - `latest = 0.1305380`
  - `recent = 0.1712313`
- verdict:
  - `fails_gate`

Interpretation:

- replay quality is materially worse than the frozen LR core
- this minimal TabR-style architecture is not competitive enough to justify immediate integration work

### `arch::tabr_hybrid_v1`

- result file:
  - `results/next_arch_challenger_arch_tabr_hybrid_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.1611370`
  - `women_cv_brier_calibrated = 0.1394923`
  - `latest = 0.1242577`
  - `recent = 0.1686229`
- verdict:
  - `fails_gate`

Interpretation:

- residual fusion is much stronger than pure `TabR`, so the “learn residual on top of frozen LR” idea is directionally better
- but it still does not beat the frozen core, and women remains slightly worse
- this is not strong enough for promotion or official LB follow-up

### `arch::tabr_feature_fusion_v1`

- result file:
  - `results/next_arch_challenger_arch_tabr_feature_fusion_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.1619156`
  - `women_cv_brier_calibrated = 0.1402849`
  - `latest = 0.1297100`
  - `recent = 0.1693841`
- verdict:
  - `fails_gate`

Interpretation:

- direct feature fusion is better than pure `TabR`
- but it is weaker than the residual hybrid and still clearly below the frozen core
- the stronger direction inside this family is residual correction, not direct replacement or direct fusion

### `arch::graph_static_embedding_v1`

- result file:
  - `results/next_arch_challenger_arch_graph_static_embedding_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.2134609`
  - `women_cv_brier_calibrated = 0.2069369`
  - `latest = 0.2160685`
  - `recent = 0.2251253`
- verdict:
  - `fails_gate`

Interpretation:

- graph representation by itself does not carry enough predictive signal in this minimal static form
- current evidence does not support opening a pure graph-only production candidate

### `arch::pairwise_ranking_v1`

- result file:
  - `results/next_arch_challenger_arch_pairwise_ranking_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.1662422`
  - `women_cv_brier_calibrated = 0.1428253`
  - `latest = 0.1285555`
  - `recent = 0.1735323`
- verdict:
  - `fails_gate`

Interpretation:

- replacing the frozen LR head with a small neural pairwise scorer is not competitive on the current tournament-sized replay regime
- the task-aligned pairwise framing alone does not overcome the small-sample instability of the neural head
- current evidence does not support promoting pairwise-only neural scoring as the next production candidate

### `arch::season_encoder_transformer_v1`

- result file:
  - `results/next_arch_challenger_arch_season_encoder_transformer_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.2134901`
  - `women_cv_brier_calibrated = 0.2060402`
  - `latest = 0.2195426`
  - `recent = 0.2254197`
- verdict:
  - `fails_gate`

Interpretation:

- this minimal season-sequence representation probe does not recover enough signal to compete with the frozen engineered ratings stack
- in v1 form, the learned team embedding is materially weaker than the current hand-built season summaries
- current evidence does not support opening a heavier transformer branch until the representation target or hybrid strategy changes

### `arch::gender_specific_stacker_v1`

- result file:
  - `results/next_arch_challenger_arch_gender_specific_stacker_v1.json`
- replay:
  - `total_cv_brier_calibrated = 0.1610393`
  - `men_cv_brier_calibrated = 0.1825567`
  - `women_cv_brier_calibrated = 0.1395220`
  - `latest = 0.1319524`
  - `recent = 0.1699159`
- verdict:
  - `fails_gate`

Interpretation:

- a narrow gender-specific stacker with only `BaselineLogit` for men and `historical consensus` sidecar features for women is not competitive
- women-side sidecar information does not become stronger merely by moving from direct upstream replacement into a simple stacker position
- current evidence does not support promoting a narrow `women-only sidecar stacker` before adding materially stronger sidecar signals

## Phase Conclusion

Phase 1 answers two questions:

1. Is minimal `TabR-style` tabular architecture already stronger than the frozen LR core?
   - No.
2. Does `TabR` become competitive once it learns residual corrections on top of the frozen LR baseline?
   - Not yet, but residual hybrid is materially better than pure `TabR`.
3. Is direct `TabR` feature fusion stronger than residual hybrid?
   - No. Residual hybrid remains the better of the two hybrid variants tested so far.
4. Does static season-safe graph representation alone show enough signal to justify deeper graph work?
   - Not in graph-only form.
5. Does a task-aligned neural pairwise head outperform the frozen LR head when using the same matchup diff features?
   - No.
6. Does a minimal transformer-based team-season representation contain enough standalone signal to justify deeper season-encoder work?
   - Not in this narrow v1 form.
7. Does a narrow gender-specific stacker improve on the frozen core when only women receives historical-consensus sidecar inputs?
   - No.

Current recommendation:

- do not promote either architecture
- do not promote the residual hybrid either
- do not promote the feature-fusion hybrid either
- do not promote the pairwise-only neural head either
- do not promote the season-encoder transformer either
- do not promote the narrow gender-specific stacker either
- keep production lane frozen
- if graph work continues later, skip further graph-only baselines and move directly to a richer hybrid design
- if tabular deep learning continues later, continue only through residual-style hybrids, not pure replacement models or direct feature-fusion variants
- if pairwise work continues later, it should move toward a stronger hybrid or team-encoder setup, not a standalone neural scorer on the same frozen-core features
- if transformer work continues later, it should be a hybrid that explicitly borrows frozen-core structure, not another standalone representation probe
- if stacker work continues later, it should only resume after adding materially stronger sidecars on both signal quality and history coverage, not by retuning this narrow women-only stacker

## Suggested Next Step

If next-arch work continues, the most defensible next experiment is:

- a materially stronger sidecar/teacher stacker with broader historical signal coverage, not this narrow women-only stacker

Not recommended as the immediate next step:

- more graph-only static variants
- official LB submission from these phase-1 experiments
- retuning `gender_specific_stacker_v1` without adding stronger sidecar inputs
