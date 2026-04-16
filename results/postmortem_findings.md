# Sprint 0-1 Postmortem Findings

## Why the legacy system stalled around the top 30%
- Legacy optimization still leaned too heavily on proxy-heavy historical leaderboard windows instead of tournament-only replay.
- The old mainline stacked `legacy_anchor`, multiple market policy routes, and runtime overrides into one default path, making real signal gain hard to separate from noise.
- The clean `v2` baseline is already competitive, which suggests the main miss was excess decision-layer complexity rather than a lack of raw features.

## What should be removed or downgraded first
- `legacy_anchor` is the first downgrade candidate; its historical ablation mean Brier is 0.1653.
- Multiple market policy routes should collapse into one explicit lightweight market-blend experiment rather than remain default mainline logic.
- `goldshot` should move from default decision layer to opt-in seasonal experiment until replay evidence justifies bringing it back.
- Runtime variance proxy: M aggressive=0.1832; M clean=0.1864; W aggressive=0.1383; W clean=0.1390.
- Keep runtime/extremes as an isolated experiment, not as a default layer.
- Latest goldshot run changed 1 rows, which is seasonal evidence only and not a replay-based reason to keep it in the mainline.

## Minimum system to keep for the next stage
- Best spread-linear candidate is `spread-linear:none@base+basecal` (0.1615 equal-gender replay).
- Best spread-tree candidate is `n/a` (n/a equal-gender replay).
- Best external-base candidate is `n/a` (n/a equal-gender replay).
- `strength_full` best linear candidate is `n/a` (n/a); `strength_recent` best is `n/a` (n/a).
- Current `base` control is `spread-linear:none@base+basecal` (0.1615), which is the direct benchmark for the strength rebuild.
- Keep sportsbook as an outer comparison only, not a reason to reintroduce complex decision-layer routing.
- Current HC primary leaderboard proxy remains M=0.1593 / W=0.1366, but it should be secondary to replay in the next iteration.
