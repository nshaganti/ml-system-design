"""
Interaction signal taxonomy (domain-neutral)
=============================================
Google's Rule 7 says: encode domain knowledge as features -- in ONE place.

Recommender datasets speak different verbs: view/click/watch, cart/save/add,
purchase/like/long-watch/share. Rather than sprinkle "purchase" and "add_to_cart"
through every phase (which quietly welds the codebase to e-commerce), we normalize
every dataset -- inside its data_sources adapter -- to a small, ordered set of
canonical *signal levels*. The rest of the pipeline is then domain-agnostic.

Levels (increasing intent):
    WEAK    an exposure with minimal intent  (impression / view / scroll-past)
    MEDIUM  an engagement                    (click / dwell / add-to-list / save)
    STRONG  the target action                (purchase / like / long-watch / share)

Every adapter maps its native events into these three. A dataset that only has one
kind of positive (e.g. purchases) simply maps it to STRONG.
"""

from __future__ import annotations

WEAK = "weak"
MEDIUM = "medium"
STRONG = "strong"

ALL_SIGNALS: tuple[str, ...] = (WEAK, MEDIUM, STRONG)

# Weights for the heuristic popularity score. A target action counts more than an
# engagement, which counts more than a mere exposure (Rule 7).
SIGNAL_WEIGHTS: dict[str, float] = {WEAK: 1.0, MEDIUM: 2.0, STRONG: 3.0}

# Signals treated as POSITIVE labels for supervised training / retrieval vocab.
# Engagement-or-stronger; a mere exposure (WEAK) is not a positive. This is the
# domain-neutral version of "cart + purchase, not views" (Rule 17: prefer directly
# observed intent; don't drown the signal in exposures).
POSITIVE_SIGNALS: tuple[str, ...] = (MEDIUM, STRONG)

# The single strongest signal -- used where we want "the user definitively
# consumed this" (e.g. exclude-already-consumed in the heuristic).
TARGET_SIGNAL = STRONG


def is_positive(signal: str) -> bool:
    """True if a signal counts as a positive training label."""
    return signal in POSITIVE_SIGNALS
