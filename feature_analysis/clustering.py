import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def perform_clustering(features_df, feature_cols, n_clusters=3):
    X = features_df[feature_cols]
    
    isf = IsolationForest(contamination=0.5, random_state=42)
    mask = isf.fit_predict(X) == 1
    cleaned_df = features_df[mask].copy()
    
    if len(cleaned_df) < n_clusters: return pd.DataFrame()

    X_scaled = StandardScaler().fit_transform(cleaned_df[feature_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cleaned_df['cluster_label'] = kmeans.fit_predict(X_scaled)
    
    return cleaned_df