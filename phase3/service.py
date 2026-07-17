"""
Phase 3 -- The Recommendation Service (the 100ms request path)
================================================================
Everything from Phases 1-2 assembled into ONE request handler. This is the code
that runs when a user hits the homepage, and it must return in ~100ms.

The flow (design doc, Phase 3):

    1. Fetch user features            (~5ms)   Feature store online lookup
    2. Candidate generation (Stage 1) (~10ms)  Popularity / two-tower ANN
    3. Batch fetch item features      (~15ms)  Feature store online lookup
    4. Rank candidates (Stage 2)      (~30ms)  LR ranker
    5. Apply business rules           (~2ms)   Drop OOS, already-seen
    6. Return top-N + async feature log

Three production concepts this module makes concrete:

- **Graceful degradation (Rule 10).** If the ranker errors or overruns, we don't
  500 the user -- we fall back to the candidate order. Production ML systems
  degrade, they don't crash.
- **Inference feature logging (Rule 29).** We log the EXACT features passed to
  the model, keyed by request_id. The next model version trains by joining user
  actions to THIS log -- not to the current feature store -- which eliminates
  training-serving skew by construction.
- **Latency budgeting.** Every stage is timed. You cannot optimize a 100ms SLA
  you don't measure per-stage.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import polars as pl

from candidate_generator import CandidateGenerator


@dataclass
class RecommendationRequest:
    user_id: str
    n: int = 20
    page_type: str = "homepage"        # context feature (Rule 20)
    device: str = "desktop"
    candidate_pool: int = 500          # Stage 1 breadth


@dataclass
class RecommendationResponse:
    request_id: str
    user_id: str
    items: list[str]
    model_version: str
    fallback_used: bool
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(self.latency_ms.values()), 2)


class _Stopwatch:
    """Context manager that records elapsed ms into a dict under `label`."""

    def __init__(self, sink: dict, label: str):
        self.sink, self.label = sink, label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.label] = round((time.perf_counter() - self._t0) * 1000, 3)


class RecommendationService:
    """
    Wires a candidate generator + feature store + ranker into one handler.

    ranker: any object with .rank(candidates_df, item_col, n) -> list[str]
            (Phase 2's LRRanker fits exactly). If None, the service returns the
            candidate order -- a valid, always-available baseline.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        feature_store,                       # PointInTimeFeatureStore
        ranker=None,                         # LRRanker or None
        model_version: str = "lr_ranker_v1",
        out_of_stock: set[str] | None = None,
        latency_budget_ms: float = 100.0,
    ):
        self.cg = candidate_generator
        self.store = feature_store
        self.ranker = ranker
        self.model_version = model_version
        self.out_of_stock = out_of_stock or set()
        self.latency_budget_ms = latency_budget_ms
        # Rule 29: the inference feature log. In production this is an async write
        # to Kafka/Iceberg; here it's an in-memory list you can inspect.
        self.feature_log: list[dict] = []

    def recommend(
        self,
        request: RecommendationRequest,
        already_seen: set[str] | None = None,
    ) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        latency: dict[str, float] = {}
        already_seen = already_seen or set()
        fallback_used = False

        # --- Stage 1: candidate generation --------------------------------
        with _Stopwatch(latency, "candidate_generation"):
            candidates = self.cg.generate(request.user_id, n=request.candidate_pool)

        # --- Fetch features for all candidates (online store) -------------
        with _Stopwatch(latency, "feature_fetch"):
            cand_df = pl.DataFrame({"item_id": candidates}).with_columns(
                pl.lit(request.user_id).alias("user_id")
            )
            features = self.store.get_online_features_batch(cand_df)

        # --- Stage 2: ranking ---------------------------------------------
        with _Stopwatch(latency, "ranking"):
            ranked, fallback_used = self._rank(features, candidates, request.n * 3)

        # --- Business rules (Rule 7 / Rule 10) ----------------------------
        with _Stopwatch(latency, "business_rules"):
            final = [
                item for item in ranked
                if item not in self.out_of_stock and item not in already_seen
            ][: request.n]

        # --- Async inference feature log (Rule 29) ------------------------
        with _Stopwatch(latency, "feature_log"):
            self._log_features(request_id, request, features, final)

        resp = RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            items=final,
            model_version=self.model_version if not fallback_used else "fallback",
            fallback_used=fallback_used,
            latency_ms=latency,
        )
        return resp

    # ------------------------------------------------------------- internals

    def _rank(self, features: pl.DataFrame, candidates: list[str], n: int):
        """Run the ranker; degrade gracefully to candidate order on any failure."""
        if self.ranker is None:
            return candidates[:n], True
        try:
            return self.ranker.rank(features, item_col="item_id", n=n), False
        except Exception as e:  # noqa: BLE001 -- never let ranking 500 the user
            print(f"[service] ranker failed ({e}); falling back to candidate order")
            return candidates[:n], True

    def _log_features(self, request_id, request, features, served_items):
        """
        Persist the exact features used for the SERVED items (Rule 29). Joining
        tomorrow's clicks to this log gives skew-free training data.
        """
        served = set(served_items)
        rows = features.filter(pl.col("item_id").is_in(list(served)))
        for r in rows.iter_rows(named=True):
            self.feature_log.append({
                "request_id": request_id,
                "user_id": request.user_id,
                "item_id": r["item_id"],
                "page_type": request.page_type,
                "device": request.device,
                "features": {
                    "item_pop": r.get("item_pop"),
                    "user_pop": r.get("user_pop"),
                    "user_cat_affinity": r.get("user_cat_affinity"),
                },
            })

    def within_budget(self, resp: RecommendationResponse) -> bool:
        return resp.total_latency_ms <= self.latency_budget_ms
