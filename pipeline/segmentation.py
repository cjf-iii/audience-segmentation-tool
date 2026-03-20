"""
Clustering Engine for Audience Segmentation.

Responsibilities:
  1. Run K-Means with the elbow method to find optimal k (or accept a
     user-specified k).
  2. Run DBSCAN as a comparison algorithm.
  3. Generate rich cluster profiles: mean numeric stats, dominant
     industries, common campaign themes, and auto-generated segment names.
  4. Produce outreach strategy recommendations per segment.

Design notes:
  - The elbow method evaluates inertia (within-cluster sum of squares)
    across a range of k values and picks the "knee" using the kneed library
    heuristic.  If kneed is unavailable, a simple second-derivative approach
    is used.
  - Segment naming uses a rule-based approach that inspects the dominant
    industry and the median revenue/ad-spend of each cluster to compose
    descriptive labels like "High-Spend Entertainment Brands".
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for the elbow search range
# ---------------------------------------------------------------------------
MIN_K = 2
MAX_K = 15


# ---------------------------------------------------------------------------
# Revenue-tier labels used in segment naming
# ---------------------------------------------------------------------------
def _revenue_tier(median_revenue: float) -> str:
    """
    Classify a cluster's median revenue into a human-readable tier.

    These thresholds are calibrated to the synthetic data ranges
    defined in generate_data.py (roughly $500K – $2B).
    """
    if median_revenue >= 200_000_000:
        return "Enterprise"
    elif median_revenue >= 50_000_000:
        return "High-Spend"
    elif median_revenue >= 10_000_000:
        return "Mid-Market"
    elif median_revenue >= 2_000_000:
        return "Growing"
    else:
        return "Emerging"


def _ad_spend_qualifier(median_ad_spend: float) -> str:
    """
    Return an adjective describing the cluster's advertising intensity.
    """
    if median_ad_spend >= 3_000_000:
        return "Heavy-Advertiser"
    elif median_ad_spend >= 1_000_000:
        return "Active-Advertiser"
    elif median_ad_spend >= 300_000:
        return "Moderate-Advertiser"
    else:
        return "Light-Advertiser"


# ---------------------------------------------------------------------------
# Outreach strategy templates — keyed by revenue tier + social engagement
# ---------------------------------------------------------------------------
OUTREACH_TEMPLATES: dict[str, str] = {
    "high_revenue_high_social": (
        "Lead with data-driven ROI case studies from comparable enterprise "
        "accounts. Propose multi-channel packages combining CTV, programmatic "
        "display, and branded content. Emphasise audience measurement and "
        "attribution capabilities. Offer a pilot program with guaranteed "
        "impressions to reduce perceived risk."
    ),
    "high_revenue_low_social": (
        "Position offerings as a way to modernise their media mix and reach "
        "audiences they are currently missing on digital channels. Provide "
        "competitive benchmarking data showing peer brands' digital spend. "
        "Suggest starting with search and display before scaling to CTV."
    ),
    "mid_revenue_high_social": (
        "Highlight cost-efficient digital channels (programmatic, paid social, "
        "podcast audio) that punch above their weight for mid-market budgets. "
        "Offer flexible commitment terms and performance-based pricing. Lean "
        "into their strong social presence by proposing influencer amplification."
    ),
    "mid_revenue_low_social": (
        "Focus on awareness-building campaigns to establish brand presence "
        "before driving performance. Recommend a blended approach of content "
        "marketing, native advertising, and targeted display. Provide creative "
        "services support to help them develop compelling assets."
    ),
    "low_revenue_high_social": (
        "Propose performance-oriented packages with low minimums and clear KPIs. "
        "Leverage their social engagement by suggesting paid amplification of "
        "existing organic content. Offer self-serve platform access with "
        "dedicated account support to grow the relationship."
    ),
    "low_revenue_low_social": (
        "Start with educational outreach — share industry insights and "
        "advertising benchmarks to build trust. Suggest a small test campaign "
        "on a single channel to demonstrate value. Nurture this segment with "
        "content marketing and quarterly check-ins."
    ),
}


def find_optimal_k(
    features: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    random_state: int = 42,
) -> tuple[int, list[float]]:
    """
    Use the elbow method to find the optimal number of clusters.

    Fits K-Means for each k in [min_k, max_k] and records inertia.
    Then applies a second-derivative heuristic (or the ``kneed``
    library if installed) to identify the elbow point.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features).
    min_k, max_k : int
        Search range for k.
    random_state : int
        Seed for K-Means.

    Returns
    -------
    optimal_k : int
        The selected number of clusters.
    inertias : list[float]
        Inertia values for each k tested (useful for plotting).
    """
    # Cap max_k at n_samples to avoid degenerate fits
    max_k = min(max_k, features.shape[0] - 1)
    k_range = range(min_k, max_k + 1)

    inertias: list[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(features)
        inertias.append(km.inertia_)
        logger.debug("k=%d  inertia=%.2f", k, km.inertia_)

    # --- try the kneed library first (more robust) -----------------------
    try:
        from kneed import KneeLocator  # type: ignore[import-untyped]

        kneedle = KneeLocator(
            list(k_range), inertias, curve="convex", direction="decreasing"
        )
        if kneedle.knee is not None:
            optimal_k = int(kneedle.knee)
            logger.info("Kneed library selected k=%d", optimal_k)
            return optimal_k, inertias
    except ImportError:
        logger.debug("kneed not installed — using second-derivative method")

    # --- fallback: second derivative (finite differences) ----------------
    if len(inertias) < 3:
        optimal_k = min_k
    else:
        # Second derivative of the inertia curve
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        # The elbow is where the second derivative is maximised
        # (largest deceleration of inertia decrease)
        optimal_k = int(np.argmax(diffs2)) + min_k + 1

    # Clamp to search range
    optimal_k = max(min_k, min(optimal_k, max_k))
    logger.info("Second-derivative method selected k=%d", optimal_k)
    return optimal_k, inertias


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """
    Fit K-Means and return cluster labels + the fitted model.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    n_clusters : int
        Number of clusters.
    random_state : int
        Seed.

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each sample (0-indexed).
    model : KMeans
        The fitted K-Means estimator (useful for centroid inspection).
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = model.fit_predict(features)
    sil = silhouette_score(features, labels) if n_clusters > 1 else 0.0
    logger.info(
        "K-Means (k=%d): silhouette=%.3f, inertia=%.1f",
        n_clusters,
        sil,
        model.inertia_,
    )
    return labels, model


def run_dbscan(
    features: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Run DBSCAN as a comparison clustering algorithm.

    DBSCAN is density-based and does not require specifying k upfront.
    It can also identify noise points (label = -1) which K-Means cannot.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    eps : float
        Maximum distance between two samples in the same neighbourhood.
    min_samples : int
        Minimum number of samples in a neighbourhood to form a core point.

    Returns
    -------
    labels : np.ndarray
        Cluster assignments (-1 = noise).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    logger.info(
        "DBSCAN (eps=%.2f, min_samples=%d): %d clusters, %d noise points",
        eps,
        min_samples,
        n_clusters,
        n_noise,
    )
    return labels


def _extract_campaign_themes(texts: pd.Series, top_n: int = 3) -> list[str]:
    """
    Extract the most common meaningful bigrams from campaign text to
    characterise a cluster's marketing activity.

    Uses a simple word-frequency approach rather than a full TF-IDF
    model — we just need descriptive phrases, not features.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if texts.empty:
        return []

    # Fit a bigram counter on the cluster's campaign texts
    vec = CountVectorizer(
        ngram_range=(2, 2),
        stop_words="english",
        max_features=50,
    )
    try:
        counts = vec.fit_transform(texts.fillna("").astype(str))
    except ValueError:
        # Happens if all texts are empty / stop words only
        return []

    # Sum counts across documents, pick top_n bigrams
    freq = counts.sum(axis=0).A1  # dense 1-D array
    vocab = vec.get_feature_names_out()
    top_idx = freq.argsort()[-top_n:][::-1]
    return [str(vocab[i]) for i in top_idx]


def _generate_segment_name(profile: dict[str, Any]) -> str:
    """
    Auto-generate a descriptive segment name from the cluster profile.

    Naming convention:  "{Revenue Tier} {Dominant Industry} {Qualifier}"

    Examples:
        "High-Spend Entertainment Brands"
        "Emerging Fintech Disruptors"
        "Mid-Market Healthcare Innovators"
    """
    tier = _revenue_tier(profile["median_revenue"])
    industry = profile["dominant_industry"]

    # Pick a qualifier based on social presence and ad spend
    social = profile["median_social_score"]
    ad_spend = profile["median_ad_spend"]

    if social >= 70 and ad_spend >= 1_000_000:
        qualifier = "Power Players"
    elif social >= 70:
        qualifier = "Socially Active"
    elif ad_spend >= 1_000_000:
        qualifier = "Big Spenders"
    elif profile["median_employees"] >= 1_000:
        qualifier = "Established Players"
    elif profile["median_revenue"] >= 10_000_000:
        qualifier = "Growth Leaders"
    else:
        qualifier = "Emerging Brands"

    return f"{tier} {industry} {qualifier}"


def _generate_outreach_strategy(profile: dict[str, Any]) -> str:
    """
    Select and return an outreach strategy recommendation based on
    the cluster's revenue and social engagement profile.
    """
    rev = profile["median_revenue"]
    social = profile["median_social_score"]

    # Classify into one of six quadrants
    if rev >= 50_000_000:
        rev_key = "high_revenue"
    elif rev >= 5_000_000:
        rev_key = "mid_revenue"
    else:
        rev_key = "low_revenue"

    social_key = "high_social" if social >= 55 else "low_social"
    key = f"{rev_key}_{social_key}"

    return OUTREACH_TEMPLATES.get(key, OUTREACH_TEMPLATES["mid_revenue_high_social"])


def generate_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    algorithm_name: str = "kmeans",
) -> list[dict[str, Any]]:
    """
    Build a rich profile dictionary for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original prospect data (must contain industry, revenue, etc.).
    labels : np.ndarray
        Cluster assignments aligned to df's index.
    algorithm_name : str
        Name of the clustering algorithm (for the profile metadata).

    Returns
    -------
    list[dict]
        One profile dict per cluster, each containing:
        - cluster_id, segment_name, algorithm
        - size, percentage
        - median and mean numeric stats
        - dominant_industry, industry_distribution
        - campaign_themes
        - outreach_strategy
    """
    df = df.copy()
    df["cluster"] = labels

    profiles: list[dict[str, Any]] = []
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        # Skip DBSCAN noise label
        if cluster_id == -1:
            continue

        mask = df["cluster"] == cluster_id
        subset = df[mask]
        n = len(subset)

        # --- numeric stats -----------------------------------------------
        median_rev = float(subset["annual_revenue"].median())
        mean_rev = float(subset["annual_revenue"].mean())
        median_emp = float(subset["employee_count"].median())
        median_social = float(subset["social_presence_score"].median())
        median_ad = float(subset["current_ad_spend_estimate"].median())
        mean_ad = float(subset["current_ad_spend_estimate"].mean())

        # --- industry distribution ---------------------------------------
        industry_counts = Counter(subset["industry"].tolist())
        dominant_industry = industry_counts.most_common(1)[0][0]
        industry_dist = {
            k: round(v / n * 100, 1)
            for k, v in industry_counts.most_common()
        }

        # --- campaign themes (bigram extraction) -------------------------
        themes = _extract_campaign_themes(subset["recent_campaigns"])

        # --- build the profile dict before naming (naming uses it) -------
        profile: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "algorithm": algorithm_name,
            "size": n,
            "percentage": round(n / len(df) * 100, 1),
            "median_revenue": median_rev,
            "mean_revenue": mean_rev,
            "median_employees": median_emp,
            "median_social_score": median_social,
            "median_ad_spend": median_ad,
            "mean_ad_spend": mean_ad,
            "dominant_industry": dominant_industry,
            "industry_distribution": industry_dist,
            "campaign_themes": themes,
            "top_demographics": (
                subset["target_demographics"]
                .value_counts()
                .head(3)
                .index.tolist()
            ),
        }

        # --- auto-generated name and outreach strategy -------------------
        profile["segment_name"] = _generate_segment_name(profile)
        profile["outreach_strategy"] = _generate_outreach_strategy(profile)

        profiles.append(profile)
        logger.info(
            "Cluster %d (%s): %d companies — %s",
            cluster_id,
            algorithm_name,
            n,
            profile["segment_name"],
        )

    return profiles
