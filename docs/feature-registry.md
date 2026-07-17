# Feature Registry (Rule 11 & 22)

Google's Rules of ML #11: *give feature columns owners and documentation.* #22:
*clean up features you no longer use.* This registry is the single place that
records what every feature means, who owns it, and its lifecycle status.

In a real org this would live in a feature-store catalog. Here it's a doc, but the
discipline is the same: **no unowned, undocumented feature ships.**

## Conventions
- **Owner** -- the team/person accountable (here: the phase that defines it).
- **Source** -- directly observed, derived, or learned.
- **Point-in-time** -- yes if computed as-of the event time (no future leakage).
- **Status** -- `active`, `experimental`, or `retired`.

## Registry

| Feature | Owner (phase) | Source | Point-in-time | Status | Description |
|---|---|---|---|---|---|
| `item_popularity` | phase0 | derived (signal-weighted counts) | yes (train window) | active | Weighted interaction count per item; the heuristic's backbone. |
| `user_category_affinity` | phase0/phase2 | derived | yes | active | Distribution of a user's past interactions over item categories. |
| `user_id_embedding` | phase1 | learned | n/a | active | Latent user vector (two-tower). Cold-start users fall back to popularity. |
| `item_id_embedding` | phase1 | learned | n/a | active | Latent item vector; only items above the min-interaction threshold get one (Rule 21). |
| `user_interaction_count_Nd` | phase2 | derived | yes | active | Rolling count of a user's positive signals in the last N days. |
| `item_signal_rate_Nd` | phase2 | derived | yes | active | Rolling positive-signal rate per item. |
| `user_x_category_affinity` | phase2 | cross (Rule 20) | yes | active | Interaction of user affinity x item category -- the personalizing signal. |
| `covisitation_score` | phase7 | derived | yes | active | Session co-occurrence weight ("seen X -> then Y"). |
| `propensity` (shown-prob) | phase8 (Part II) | logged policy | yes | experimental | P(item shown \| context) from the logging policy; enables IPS/DR. |

## Retirement log
*(Nothing retired yet. When a feature is removed, record it here with the date and
reason -- Rule 22.)*

## Adding a feature (checklist)
1. Add a row here with an owner and description.
2. Tag the defining function's docstring with the relevant `Rule N:`.
3. Ensure it's computed **point-in-time** (as-of event time) if derived.
4. If it feeds the ranker, make sure the *same* code path serves it (Rule 32).
