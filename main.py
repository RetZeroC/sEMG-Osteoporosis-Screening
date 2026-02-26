import pandas as pd
import os


from label_mapper import merge_real_labels
from feature_analysis.data_loader import load_emg_data, process_emg_data
from feature_analysis.extractor import extract_features
from feature_analysis.clustering import perform_clustering
from visualization.plot_pca import plot_pca_visualization
from visualization.plot_heatmap import plot_correlation_heatmap, plot_mean_heatmap
from visualization.plot_distribution import plot_dual_mode_distribution
from deep_learning.dataset import balance_and_augment_data
from deep_learning.train import train_eval_model

def main():
    # Configure paths 
    DATA_ROOT = 'data'
    METADATA_PATH = 'metadata_clinical_diagnosis.csv'
    
    raw_data = load_emg_data(DATA_ROOT)
    emg_samples = process_emg_data(raw_data) 
    
    labeled_samples = merge_real_labels(emg_samples, METADATA_PATH, subject_col='subject_id', label_col='diagnosis_level')
    
    if not labeled_samples:
        print("Fatal Error: Could not find any data with real labels. Terminating program.")
        return

    bending_samples = [s for s in labeled_samples if s['action_type'] == 'bending']
    sitstand_samples = [s for s in labeled_samples if s['action_type'] == 'sit-stand']

    features_df = extract_features(labeled_samples)
    
    bending_features = features_df[features_df['action_type'] == 'bending'].copy()
    sitstand_features = features_df[features_df['action_type'] == 'sit-stand'].copy()
    
    feature_cols = [c for c in features_df.columns if c not in ['action_type', 'subject_id', 'channel_id', 'sample_id', 'label']]

    bending_features = perform_clustering(bending_features, feature_cols)
    sitstand_features = perform_clustering(sitstand_features, feature_cols)


    os.makedirs('output_plots', exist_ok=True)
    os.chdir('output_plots')

    # Here we use the real 'label' for plotting to evaluate feature performance across different diagnostic levels
    if not bending_features.empty:
        plot_correlation_heatmap(bending_features, feature_cols, 'Bending')
        plot_mean_heatmap(bending_features, feature_cols, 'Bending', label_col='label')
        plot_pca_visualization(bending_features, feature_cols, 'Bending', label_col='label')

    if not sitstand_features.empty:
        plot_pca_visualization(sitstand_features, feature_cols, 'SitStand', label_col='label')

    if not bending_features.empty and not sitstand_features.empty:
        plot_dual_mode_distribution(bending_features, sitstand_features, feature_cols, label_col='label', plot_type='violin')

    os.chdir('..') 


    bending_train_data = bending_samples
    sitstand_train_data = sitstand_samples
    

    TRAIN_EPOCHS = 100
    

    if bending_train_data:
        print(f"\n=== Training Bending 1D-CNN Model (Epochs: {TRAIN_EPOCHS}) ===")
        train_eval_model(bending_train_data, epochs=TRAIN_EPOCHS, folds=5)
        
    if sitstand_train_data:
        print(f"\n=== Training Sit-Stand 1D-CNN Model (Epochs: {TRAIN_EPOCHS}) ===")
        train_eval_model(sitstand_train_data, epochs=TRAIN_EPOCHS, folds=5)

    print("\n>>> Full pipeline execution completed!")


if __name__ == "__main__":
    main()