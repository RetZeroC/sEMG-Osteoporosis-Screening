# 2_visualization/plot_pca.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_visualization(features_df, feature_cols, action_name, label_col='label', sample_size=500):
   
    X = features_df[feature_cols]
    y = features_df[label_col]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2', 'PC3'], index=features_df.index)
    pca_df[label_col] = y


    LABEL_FONT = {'fontname': 'Arial', 'size': 44, 'weight': 'bold'}
    TICK_FONT_PROPS = {'family': 'Arial', 'size': 34, 'weight': 'bold'}
    palette = sns.color_palette("viridis", n_colors=len(pca_df[label_col].unique()))


    fig2d, ax2d = plt.subplots(figsize=(22, 20))
    for i, class_val in enumerate(sorted(pca_df[label_col].unique())):
        cluster_data = pca_df[pca_df[label_col] == class_val]
        cluster_sampled = cluster_data.sample(n=min(len(cluster_data), sample_size), random_state=42)
        
        sns.kdeplot(data=cluster_sampled, x='PC1', y='PC2', fill=True, alpha=0.3, color=palette[i], levels=5, ax=ax2d)
        ax2d.scatter(cluster_sampled['PC1'], cluster_sampled['PC2'], c=[palette[i]]*len(cluster_sampled),
                     edgecolor='k', linewidth=0.5, s=150, label=f'Level {class_val}')

    ax2d.set_xlabel("Principal Component 1", fontdict=LABEL_FONT, labelpad=25)
    ax2d.set_ylabel("Principal Component 2", fontdict=LABEL_FONT, labelpad=25)
    ax2d.tick_params(axis='both', which='major', labelsize=TICK_FONT_PROPS['size'], pad=18, width=4)
    
    for label in ax2d.get_xticklabels() + ax2d.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
    for spine in ax2d.spines.values():
        spine.set_linewidth(4)

    legend = ax2d.legend(fontsize=34, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    for text in legend.get_texts(): text.set_fontweight('bold')
    
    fig2d.tight_layout()
    fig2d.savefig(f"pca_2d_{action_name}.svg", format="svg")
    plt.close(fig2d)

    fig3d = plt.figure(figsize=(22, 20))
    ax3d = fig3d.add_subplot(111, projection='3d')
    pca_df_3d_sampled = pca_df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size), random_state=42)
    )
    scatter = ax3d.scatter(pca_df_3d_sampled['PC1'], pca_df_3d_sampled['PC2'], pca_df_3d_sampled['PC3'],
                           c=pca_df_3d_sampled[label_col], cmap='viridis', s=150, edgecolor='k')

    ax3d.set_xlabel('Principal Component 1', fontdict={'fontname': 'Arial', 'size': 28, 'weight': 'bold'}, labelpad=30)
    ax3d.set_ylabel('Principal Component 2', fontdict={'fontname': 'Arial', 'size': 28, 'weight': 'bold'}, labelpad=30)
    ax3d.set_zlabel('Principal Component 3', fontdict={'fontname': 'Arial', 'size': 28, 'weight': 'bold'}, labelpad=30)
    ax3d.tick_params(axis='both', labelsize=TICK_FONT_PROPS['size'], pad=2)

    legend_3d = ax3d.legend(*scatter.legend_elements(), fontsize=28)
    for text in legend_3d.get_texts(): text.set_fontweight('bold')

    fig3d.tight_layout()
    fig3d.savefig(f"pca_3d_{action_name}.svg", format="svg")
    plt.close(fig3d)
   