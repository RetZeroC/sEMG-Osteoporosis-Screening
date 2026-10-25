"""Heat maps summarising the feature set.

Both figures are written to the current working directory as vector graphics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_correlation_heatmap(features_df, feature_cols, action_name):
    """Pearson correlation matrix of the selected features."""
    corr_matrix = features_df[feature_cols].corr()

    plt.figure(figsize=(33, 28))
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                     annot_kws={"size": 22, "fontname": "Arial", "weight": "bold"})

    plt.xticks(rotation=45, ha="right", fontsize=26, fontweight="bold", fontname="Arial")
    plt.yticks(rotation=0, fontsize=26, fontweight="bold", fontname="Arial")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=24)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(f"{action_name}_feature_correlation.svg", format="svg", bbox_inches="tight")
    plt.close()


def plot_mean_heatmap(features_df, feature_cols, action_name, label_col="label"):
    """Per-level mean of each feature, standardised across levels."""
    means = features_df.groupby(label_col)[feature_cols].mean()
    means_scaled = StandardScaler().fit_transform(means)
    means_df = pd.DataFrame(means_scaled, index=[f"Level {i}" for i in means.index],
                            columns=means.columns)

    plt.figure(figsize=(60, 8))
    ax = sns.heatmap(means_df, annot=True, fmt=".2f", cmap="Blues",
                     annot_kws={"size": 22, "fontname": "Arial", "weight": "bold"})

    plt.xticks(rotation=45, ha="right", fontsize=24, fontweight="bold", fontname="Arial")
    plt.yticks(rotation=0, fontsize=24, fontweight="bold", fontname="Arial")
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=24)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(f"{action_name}_mean_heatmap.svg", format="svg", bbox_inches="tight")
    plt.close()
