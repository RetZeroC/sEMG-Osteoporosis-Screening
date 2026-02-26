# 2_visualization/plot_distribution.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

def plot_dual_mode_distribution(bending_df, sitstand_df, feature_cols, label_col='label', plot_type='violin'):
   
    
    combined_df = pd.concat([bending_df.assign(action_mode='bending'), 
                             sitstand_df.assign(action_mode='sit-stand')], ignore_index=True)
    
    scaled_features = StandardScaler().fit_transform(combined_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=combined_df.index)
    scaled_df[label_col] = combined_df[label_col].astype(str)
    
    scaled_bending = scaled_df[combined_df['action_mode'] == 'bending']
    scaled_sitstand = scaled_df[combined_df['action_mode'] == 'sit-stand']

    num_features = len(feature_cols)
    sns.set_style('white')
    fig, axes = plt.subplots(nrows=2, ncols=num_features, figsize=(3 * num_features, 12), sharey=True)

    TITLE_FONT = {'fontname': 'Arial', 'size': 24, 'weight': 'bold'}
    LABEL_FONT = {'fontname': 'Arial', 'size': 20, 'weight': 'bold'}
    TICK_FONT = {'family': 'Arial', 'size': 18, 'weight': 'bold'}

    unique_classes = sorted(scaled_df[label_col].unique())
    palette = sns.color_palette("viridis", len(unique_classes))
    color_map = dict(zip(unique_classes, palette))

    for i, feature in enumerate(feature_cols):
        ax_bend, ax_sit = axes[0, i], axes[1, i]
        ax_bend.set_facecolor('white')
        ax_sit.set_facecolor('white')

        for ax, df in zip([ax_bend, ax_sit], [scaled_bending, scaled_sitstand]):
            if plot_type == 'violin':
                sns.violinplot(data=df, x=label_col, y=feature, ax=ax, palette=color_map, inner='box', cut=0)
            else:
                sns.boxplot(data=df, x=label_col, y=feature, ax=ax, palette=color_map, showfliers=False)
            
            ax.set_xlabel('Level', fontdict=LABEL_FONT) if ax == ax_sit else ax.set_xlabel('')
            ax.set_ylabel(feature if i == 0 else '', fontdict=LABEL_FONT)
            ax.tick_params(axis='both', labelsize=TICK_FONT['size'], width=2)
            ax.grid(False)
            for spine in ax.spines.values(): spine.set_linewidth(2)

        ax_bend.set_title(feature, fontdict=TITLE_FONT)

    plt.tight_layout()
    fig.savefig(f"dual_distribution_{plot_type}.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
