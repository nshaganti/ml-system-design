"""
Phase 2 -- Logistic Regression Ranker (Stage 2)
=================================================
The two-tower model (Phase 1) turns 50M items into ~500 candidates. This ranker
reorders those candidates into the final top-20 by predicting
P(strong interaction | user, item) from feature-store features.

Why logistic regression first, not XGBoost or a DNN? (Rules 4, 14)
  When the ranker makes a mistake, LR tells you WHY -- inspect the weights.
  The lift gap vs a tree model is small at this stage; the debuggability gap is
  enormous. Earn complexity later.

This module is intentionally model-light: feature engineering and point-in-time
correctness (the feature_store) are where the production value lives.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from feature_store import FEATURE_COLUMNS


def to_matrix(
    df: pl.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    log_columns: list[str] | None = None,
) -> np.ndarray:
    """
    Turn feature columns into a float matrix for sklearn.

    log1p on the popularity counts: they're heavy-tailed, and a linear model
    handles log-scaled magnitudes far better than raw counts (Rule 20-ish --
    transform features to a shape the model can use).

    log_columns: which columns to log1p-transform. Defaults to ALL feature_columns
    (the Phase 2 behaviour). Pass an explicit list when some features are NOT
    heavy-tailed counts -- e.g. a similarity score that can be negative, where
    log1p is nonsensical. Anything not in log_columns is passed through raw.
    """
    log_set = set(feature_columns if log_columns is None else log_columns)
    cols = []
    for c in feature_columns:
        v = df[c].to_numpy().astype(np.float64)
        cols.append(np.log1p(v) if c in log_set else v)
    return np.column_stack(cols)


class LRRanker:
    """
    Thin wrapper around sklearn LogisticRegression with a polars-friendly API.

    fit()   : train on a labelled feature DataFrame (label col = 1 strong / 0 not)
    rank()  : given candidate rows with features, return them sorted by score
    weights : inspect learned coefficients (the whole point of choosing LR)
    """

    def __init__(self, feature_columns: list[str] = FEATURE_COLUMNS, C: float = 1.0,
                 log_columns: list[str] | None = None):
        self.feature_columns = feature_columns
        self.log_columns = log_columns   # None => log every column (Phase 2 default)
        self.model = LogisticRegression(C=C, max_iter=1000)
        self._fitted = False

    def fit(self, training_df: pl.DataFrame, label_col: str = "label") -> "LRRanker":
        if label_col not in training_df.columns:
            raise ValueError(f"training_df missing label column {label_col!r}")
        X = to_matrix(training_df, self.feature_columns, self.log_columns)
        y = training_df[label_col].to_numpy().astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Need both positive and negative labels to train LR.")
        self.model.fit(X, y)
        self._fitted = True
        print(
            f"[lr_ranker] Fitted on {len(y):,} rows "
            f"({int(y.sum()):,} positives) | weights={self.weights}"
        )
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        self._require_fitted()
        X = to_matrix(df, self.feature_columns, self.log_columns)
        return self.model.predict_proba(X)[:, 1]

    def rank(self, candidates_df: pl.DataFrame, item_col: str = "item_id", n: int = 20) -> list[str]:
        """Return the top-n item IDs ordered by predicted score (descending)."""
        self._require_fitted()
        scores = self.predict_proba(candidates_df)
        scored = candidates_df.with_columns(pl.Series("score", scores))
        return (
            scored.sort("score", descending=True)
            .head(n)[item_col]
            .to_list()
        )

    @property
    def weights(self) -> dict:
        if not self._fitted:
            return {}
        coefs = self.model.coef_[0]
        return {c: round(float(w), 4) for c, w in zip(self.feature_columns, coefs)}

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before scoring/ranking.")
