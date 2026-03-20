"""
NLP Feature Extraction Pipeline.

Responsibilities:
  1. Vectorise free-text columns (website_description, recent_campaigns)
     with TF-IDF.
  2. Reduce the high-dimensional TF-IDF matrix with UMAP (or PCA fallback).
  3. Combine the reduced text features with scaled numeric features into a
     single feature matrix ready for clustering.

The pipeline is intentionally stateful — fit() stores the learned
transformers so the same pipeline can later transform new data without
re-fitting (useful for a future scoring API).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric columns that get z-score scaled before merging with text features.
# These must exist in the input DataFrame.
# ---------------------------------------------------------------------------
NUMERIC_COLS: list[str] = [
    "annual_revenue",
    "employee_count",
    "social_presence_score",
    "current_ad_spend_estimate",
]

# ---------------------------------------------------------------------------
# Text columns that get TF-IDF vectorised and then dimensionally reduced.
# ---------------------------------------------------------------------------
TEXT_COLS: list[str] = [
    "website_description",
    "recent_campaigns",
]


class TextFeatureProcessor:
    """
    End-to-end feature processor that converts a prospects DataFrame into
    a dense numeric matrix suitable for clustering.

    Parameters
    ----------
    n_text_components : int
        Number of dimensions to keep after reducing each TF-IDF matrix.
        A lower value trades information for speed and interpretability.
    max_tfidf_features : int
        Maximum vocabulary size for each TF-IDF vectoriser.
    use_umap : bool
        If True, use UMAP for dimensionality reduction; otherwise fall
        back to PCA.  UMAP preserves local structure better but requires
        the ``umap-learn`` package.
    random_state : int
        Seed for reproducibility across all stochastic components.
    """

    def __init__(
        self,
        n_text_components: int = 10,
        max_tfidf_features: int = 500,
        use_umap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_text_components = n_text_components
        self.max_tfidf_features = max_tfidf_features
        self.use_umap = use_umap
        self.random_state = random_state

        # These are populated during fit()
        self._vectorizers: dict[str, TfidfVectorizer] = {}
        self._reducers: dict[str, object] = {}
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reducer(self) -> object:
        """
        Instantiate the appropriate dimensionality-reduction model.

        Tries UMAP first (better at preserving local neighbourhood
        structure, which matters for clustering).  Falls back to PCA
        if umap-learn is unavailable — PCA is always in scikit-learn.
        """
        if self.use_umap:
            try:
                import umap  # type: ignore[import-untyped]

                logger.info("Using UMAP for dimensionality reduction")
                return umap.UMAP(
                    n_components=self.n_text_components,
                    random_state=self.random_state,
                    # metric="cosine" works well with TF-IDF vectors
                    metric="cosine",
                    n_neighbors=15,
                    min_dist=0.1,
                )
            except ImportError:
                logger.warning(
                    "umap-learn not installed — falling back to PCA"
                )

        # Fallback: PCA (always available via scikit-learn)
        from sklearn.decomposition import PCA

        logger.info("Using PCA for dimensionality reduction")
        return PCA(
            n_components=self.n_text_components,
            random_state=self.random_state,
        )

    def _vectorize_text(
        self, series: pd.Series, col_name: str, fit: bool = True
    ) -> np.ndarray:
        """
        Convert a text Series into a TF-IDF sparse matrix, then reduce
        its dimensionality.

        Parameters
        ----------
        series : pd.Series
            The raw text column.
        col_name : str
            Column name — used as the key for storing the fitted
            vectoriser and reducer so they can be reused at transform time.
        fit : bool
            If True, fit new transformers; if False, reuse previously
            fitted ones (for transform-only calls).

        Returns
        -------
        np.ndarray
            Dense matrix of shape (n_samples, n_text_components).
        """
        # Fill missing text with empty string to avoid NaN propagation
        clean_text = series.fillna("").astype(str)

        if fit:
            # --- fit a new TF-IDF vectoriser for this column -------------
            vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                stop_words="english",
                ngram_range=(1, 2),   # unigrams + bigrams for richer signal
                min_df=2,             # ignore terms appearing in < 2 docs
                max_df=0.95,          # ignore terms appearing in > 95% docs
            )
            tfidf_matrix = vectorizer.fit_transform(clean_text)
            self._vectorizers[col_name] = vectorizer
            logger.info(
                "TF-IDF [%s]: vocabulary size = %d, matrix shape = %s",
                col_name,
                len(vectorizer.vocabulary_),
                tfidf_matrix.shape,
            )

            # --- fit a dimensionality reducer ----------------------------
            # Need dense input for UMAP; PCA can handle sparse but we
            # convert for consistency
            dense = tfidf_matrix.toarray()

            # Clamp n_components to the smaller of (n_samples, n_features)
            # to avoid errors when the dataset is tiny
            max_components = min(dense.shape[0], dense.shape[1])
            actual_components = min(self.n_text_components, max_components)

            reducer = self._build_reducer()
            # Override n_components if we had to clamp
            if hasattr(reducer, "n_components"):
                reducer.n_components = actual_components  # type: ignore[attr-defined]

            reduced = reducer.fit_transform(dense)
            self._reducers[col_name] = reducer
            logger.info(
                "Reduced [%s]: %s -> %s",
                col_name,
                dense.shape,
                reduced.shape,
            )
        else:
            # --- transform only (reuse fitted objects) -------------------
            vectorizer = self._vectorizers[col_name]
            tfidf_matrix = vectorizer.transform(clean_text)
            dense = tfidf_matrix.toarray()
            reducer = self._reducers[col_name]
            reduced = reducer.transform(dense)

        return reduced

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit all transformers on *df* and return the combined feature matrix.

        The output matrix has columns laid out as:
            [text_features_col1 | text_features_col2 | scaled_numeric_features]

        Parameters
        ----------
        df : pd.DataFrame
            Prospect data containing TEXT_COLS and NUMERIC_COLS.

        Returns
        -------
        np.ndarray
            Dense feature matrix of shape
            (n_samples, 2 * n_text_components + len(NUMERIC_COLS)).
        """
        parts: list[np.ndarray] = []

        # --- text features -----------------------------------------------
        for col in TEXT_COLS:
            reduced = self._vectorize_text(df[col], col, fit=True)
            parts.append(reduced)

        # --- numeric features (z-score scaled) ---------------------------
        numeric = df[NUMERIC_COLS].fillna(0).values.astype(float)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(numeric)
        parts.append(scaled)
        logger.info(
            "Numeric features: %d columns, scaled with StandardScaler",
            scaled.shape[1],
        )

        self._is_fitted = True
        combined = np.hstack(parts)
        logger.info("Combined feature matrix shape: %s", combined.shape)
        return combined

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the previously fitted transformers.

        Raises RuntimeError if fit_transform() has not been called first.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Processor has not been fitted. Call fit_transform() first."
            )

        parts: list[np.ndarray] = []

        for col in TEXT_COLS:
            reduced = self._vectorize_text(df[col], col, fit=False)
            parts.append(reduced)

        numeric = df[NUMERIC_COLS].fillna(0).values.astype(float)
        scaled = self._scaler.transform(numeric)  # type: ignore[union-attr]
        parts.append(scaled)

        return np.hstack(parts)

    def get_2d_projection(self, df: pd.DataFrame) -> np.ndarray:
        """
        Produce a 2-D UMAP (or PCA) projection of the combined feature
        matrix for visualisation purposes.

        This fits a *separate* 2-component reducer on the full feature
        matrix — it is intentionally independent of the n_text_components
        reducers used for clustering.

        Parameters
        ----------
        df : pd.DataFrame
            The same DataFrame passed to fit_transform() (or a subset).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2).
        """
        # Get the full feature matrix first
        if self._is_fitted:
            features = self.transform(df)
        else:
            features = self.fit_transform(df)

        # Build a 2-component reducer for visualisation
        if self.use_umap:
            try:
                import umap  # type: ignore[import-untyped]

                vis_reducer = umap.UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    metric="euclidean",
                    n_neighbors=15,
                    min_dist=0.1,
                )
            except ImportError:
                from sklearn.decomposition import PCA

                vis_reducer = PCA(
                    n_components=2,
                    random_state=self.random_state,
                )
        else:
            from sklearn.decomposition import PCA

            vis_reducer = PCA(
                n_components=2,
                random_state=self.random_state,
            )

        projection = vis_reducer.fit_transform(features)
        logger.info("2-D projection shape: %s", projection.shape)
        return projection
