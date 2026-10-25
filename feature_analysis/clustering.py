"""Unsupervised exploration of the feature space."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def perform_clustering(features_df, feature_cols, n_clusters=3):
    """Remove multivariate outliers and partition the remaining samples.

    The procedure follows the analysis in the paper: an isolation forest with a
    0.5 contamination rate discards half of the samples, the survivors are
    standardised and partitioned with k-means (k = 3).

    Parameters
    ----------
    features_df : pandas.DataFrame
        Feature table for a single movement (one row per segment).
    feature_cols : list of str
        Columns entering the clustering.
    n_clusters : int
        Number of k-means clusters.

    Returns
    -------
    pandas.DataFrame
        The retained rows with an additional ``cluster_label`` column, or an
        empty DataFrame when too few samples survive outlier removal.
    """
    features = features_df[feature_cols]

    iso_forest = IsolationForest(contamination=0.5, random_state=42)
    inlier_mask = iso_forest.fit_predict(features) == 1
    cleaned_df = features_df[inlier_mask].copy()

    if len(cleaned_df) < n_clusters:
        return pd.DataFrame()

    scaled = StandardScaler().fit_transform(cleaned_df[feature_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cleaned_df["cluster_label"] = kmeans.fit_predict(scaled)

    return cleaned_df
