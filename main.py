"""Full analysis pipeline.

Run from the repository root, with the recordings in ``data/`` and the clinical
table ``metadata_clinical_diagnosis.csv`` in place. Figures are written to
``output_plots/`` and the cross-validated classification report is printed.
"""

import os

from deep_learning.train import train_eval_model
from feature_analysis.clustering import perform_clustering
from feature_analysis.data_loader import load_emg_data, process_emg_data
from feature_analysis.extractor import extract_features
from label_mapper import merge_real_labels
from visualization.plot_distribution import plot_dual_mode_distribution
from visualization.plot_heatmap import plot_correlation_heatmap, plot_mean_heatmap
from visualization.plot_pca import plot_pca_visualization

DATA_ROOT = "data"
METADATA_PATH = "metadata_clinical_diagnosis.csv"
TRAIN_EPOCHS = 100
N_FOLDS = 5
MOVEMENTS = (("bending", "Bending"), ("sit-stand", "Sit-Stand"))

# Columns of the feature table that are not features.
NON_FEATURE_COLUMNS = ("action_type", "subject_id", "channel_id", "sample_id", "label")


def plot_feature_visualizations(bending_features, sitstand_features, feature_cols):
    """Write all figures into ``output_plots/``.

    The plotting helpers save to the current working directory, so the working
    directory is switched for the duration of the calls.
    """
    os.makedirs("output_plots", exist_ok=True)
    os.chdir("output_plots")
    try:
        if not bending_features.empty:
            plot_correlation_heatmap(bending_features, feature_cols, "Bending")
            plot_mean_heatmap(bending_features, feature_cols, "Bending", label_col="label")
            plot_pca_visualization(bending_features, feature_cols, "Bending", label_col="label")

        if not sitstand_features.empty:
            plot_pca_visualization(sitstand_features, feature_cols, "SitStand",
                                   label_col="label")

        if not bending_features.empty and not sitstand_features.empty:
            plot_dual_mode_distribution(bending_features, sitstand_features, feature_cols,
                                        label_col="label", plot_type="violin")
    finally:
        os.chdir("..")


def main():
    raw_data = load_emg_data(DATA_ROOT)
    emg_samples = process_emg_data(raw_data)

    labelled_samples = merge_real_labels(emg_samples, METADATA_PATH,
                                         subject_col="subject_id",
                                         label_col="diagnosis_level")
    if not labelled_samples:
        print("No segment could be matched to the clinical metadata; aborting.")
        return

    features_df = extract_features(labelled_samples)
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]

    # Clustering is run per movement, as the two tasks have distinct dynamics.
    bending_features = perform_clustering(
        features_df[features_df["action_type"] == "bending"].copy(), feature_cols)
    sitstand_features = perform_clustering(
        features_df[features_df["action_type"] == "sit-stand"].copy(), feature_cols)

    plot_feature_visualizations(bending_features, sitstand_features, feature_cols)

    for action_type, title in MOVEMENTS:
        samples = [s for s in labelled_samples if s["action_type"] == action_type]
        if not samples:
            continue
        print(f"\n=== Training {title} 1D-CNN model (epochs: {TRAIN_EPOCHS}) ===")
        train_eval_model(samples, epochs=TRAIN_EPOCHS, folds=N_FOLDS)

    print("\n>>> Full pipeline completed.")


if __name__ == "__main__":
    main()
