"""Two- and three-dimensional PCA projections of the feature space."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LABEL_FONT = {"fontname": "Arial", "size": 44, "weight": "bold"}
TICK_FONT_SIZE = 34


def plot_pca_visualization(features_df, feature_cols, action_name, label_col="label",
                           sample_size=500):
    """Write ``pca_2d_<action>.svg`` and ``pca_3d_<action>.svg``.

    The first two principal components are shown as a density contour plus
    scatter; the third component is added in the 3-D panel. At most
    ``sample_size`` segments per level are drawn.
    """
    scaled = StandardScaler().fit_transform(features_df[feature_cols])
    pcs = PCA(n_components=3, random_state=42).fit_transform(scaled)

    pca_df = pd.DataFrame(data=pcs, columns=["PC1", "PC2", "PC3"], index=features_df.index)
    pca_df[label_col] = features_df[label_col]

    levels = sorted(pca_df[label_col].unique())
    palette = sns.color_palette("viridis", n_colors=len(levels))

    # --- 2-D projection ----------------------------------------------------
    fig2d, ax2d = plt.subplots(figsize=(22, 20))
    for index, level in enumerate(levels):
        level_data = pca_df[pca_df[label_col] == level]
        level_sample = level_data.sample(n=min(len(level_data), sample_size), random_state=42)

        sns.kdeplot(data=level_sample, x="PC1", y="PC2", fill=True, alpha=0.3,
                    color=palette[index], levels=5, ax=ax2d)
        ax2d.scatter(level_sample["PC1"], level_sample["PC2"],
                     c=[palette[index]] * len(level_sample), edgecolor="k",
                     linewidth=0.5, s=150, label=f"Level {level}")

    ax2d.set_xlabel("Principal Component 1", fontdict=LABEL_FONT, labelpad=25)
    ax2d.set_ylabel("Principal Component 2", fontdict=LABEL_FONT, labelpad=25)
    ax2d.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE, pad=18, width=4)

    for tick_label in ax2d.get_xticklabels() + ax2d.get_yticklabels():
        tick_label.set_fontname("Arial")
        tick_label.set_fontweight("bold")
    for spine in ax2d.spines.values():
        spine.set_linewidth(4)

    legend = ax2d.legend(fontsize=34, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig2d.tight_layout()
    fig2d.savefig(f"pca_2d_{action_name}.svg", format="svg")
    plt.close(fig2d)

    # --- 3-D projection ----------------------------------------------------
    # NOTE: ``groupby(...).apply`` no longer keeps the grouping column on
    # pandas >= 3.0, hence the explicit per-group sampling.
    sampled_3d = pd.concat(
        [group.sample(n=min(len(group), sample_size), random_state=42)
         for _, group in pca_df.groupby(label_col)],
        ignore_index=True,
    )

    fig3d = plt.figure(figsize=(22, 20))
    ax3d = fig3d.add_subplot(111, projection="3d")
    scatter = ax3d.scatter(sampled_3d["PC1"], sampled_3d["PC2"], sampled_3d["PC3"],
                           c=sampled_3d[label_col], cmap="viridis", s=150, edgecolor="k")

    axis_font = {"fontname": "Arial", "size": 28, "weight": "bold"}
    ax3d.set_xlabel("Principal Component 1", fontdict=axis_font, labelpad=30)
    ax3d.set_ylabel("Principal Component 2", fontdict=axis_font, labelpad=30)
    ax3d.set_zlabel("Principal Component 3", fontdict=axis_font, labelpad=30)
    ax3d.tick_params(axis="both", labelsize=TICK_FONT_SIZE, pad=2)

    legend_3d = ax3d.legend(*scatter.legend_elements(), fontsize=28)
    for text in legend_3d.get_texts():
        text.set_fontweight("bold")

    fig3d.tight_layout()
    fig3d.savefig(f"pca_3d_{action_name}.svg", format="svg")
    plt.close(fig3d)
