"""Run the analysis pipeline on the synthetic data in ``demo/data``.

The steps mirror ``main.py`` (load, segment, label, feature extraction,
clustering, figures) but take explicit paths, so the repository's own ``data/``
directory is left untouched and no participant data are required.

    python demo/run_demo.py                 # features, clustering, figures
    python demo/run_demo.py --with-dl       # additionally train the 1-D CNN (needs torch)
    python demo/run_demo.py --output-dir out_demo

Output is written to ``demo/output`` unless ``--output-dir`` is given.
"""

import argparse
import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from feature_analysis.clustering import perform_clustering  # noqa: E402
from feature_analysis.data_loader import load_emg_data, process_emg_data  # noqa: E402
from feature_analysis.extractor import extract_features  # noqa: E402
from label_mapper import merge_real_labels  # noqa: E402

NON_FEATURE_COLUMNS = ("action_type", "subject_id", "channel_id", "sample_id",
                       "label", "cluster_label")
MOVEMENTS = ("bending", "sit-stand")


def pretty(path):
    """Show ``path`` relative to the repository root when possible."""
    path = os.path.abspath(path)
    try:
        relative = os.path.relpath(path, REPO_ROOT)
    except ValueError:              # different drive
        return path
    return path if relative.startswith("..") else relative.replace(os.sep, "/")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo on synthetic sEMG data.")
    parser.add_argument("--data-root", default=os.path.join(DEMO_DIR, "data"),
                        help="folder with <action>/<subject>/<channel>.csv")
    parser.add_argument("--metadata",
                        default=os.path.join(DEMO_DIR, "metadata_clinical_diagnosis.csv"),
                        help="csv with subject_id and diagnosis_level columns")
    parser.add_argument("--output-dir", default=os.path.join(DEMO_DIR, "output"))
    parser.add_argument("--sampling-rate", type=int, default=2000)
    parser.add_argument("--with-dl", action="store_true",
                        help="also train the 1-D CNN (requires torch)")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("sEMG osteoporosis screening - demo pipeline (synthetic data)")
    print(f"data     : {pretty(args.data_root)}")
    print(f"metadata : {pretty(args.metadata)}")
    print(f"output   : {pretty(out_dir)}")
    print("=" * 68)

    # ------------------------------------------------------------------ 1/6
    print("\n[1/6] load raw sEMG files")
    raw = load_emg_data(args.data_root)
    n_files = sum(len(recordings) for recordings in raw.values())
    for action, recordings in raw.items():
        print(f"      {action:<10s} {len(recordings):>3d} file(s)")
    print(f"      total      {n_files:>3d} file(s)")

    # ------------------------------------------------------------------ 2/6
    print("\n[2/6] segment activity bursts (RMS threshold, 0.2 s window)")
    segments = process_emg_data(raw, sampling_rate=args.sampling_rate)
    print(f"      {len(segments)} segment(s) kept "
          f"({len(segments) / max(n_files, 1):.1f} per file)")
    if not segments:
        print("      no segment found - check the RMS thresholds / data layout")
        return 1

    # ------------------------------------------------------------------ 3/6
    print("\n[3/6] attach diagnosis labels from the metadata csv")
    labelled = merge_real_labels(segments, args.metadata, subject_col="subject_id",
                                 label_col="diagnosis_level")
    print(f"      {len(labelled)} labelled segment(s), "
          f"{len(segments) - len(labelled)} dropped (unknown subject)")
    if not labelled:
        print("      no labelled sample - check subject_id values in the metadata")
        return 1
    label_counts = pd.Series([s["label"] for s in labelled]).value_counts().sort_index()
    print("      labels:", ", ".join(f"level {k}: {v}" for k, v in label_counts.items()))

    # ------------------------------------------------------------------ 4/6
    print("\n[4/6] extract time / frequency / wavelet features")
    features_df = extract_features(labelled, sampling_rate=args.sampling_rate)
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]
    print(f"      {features_df.shape[0]} sample(s) x {len(feature_cols)} feature(s)")
    features_df.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    by_movement = {movement: features_df[features_df["action_type"] == movement].copy()
                   for movement in MOVEMENTS}
    print(f"      bending: {len(by_movement['bending'])} sample(s), "
          f"sit-stand: {len(by_movement['sit-stand'])} sample(s)")

    # ------------------------------------------------------------------ 5/6
    print("\n[5/6] clustering (IsolationForest + KMeans, k=3)")
    clustered = {}
    for movement in MOVEMENTS:
        clustered[movement] = perform_clustering(by_movement[movement], feature_cols)
        frame = clustered[movement]
        if frame.empty:
            print(f"      {movement:<9s}: not enough samples after outlier removal")
            continue
        table = pd.crosstab(frame["cluster_label"], frame["label"])
        print(f"      {movement:<9s}: {len(frame)} sample(s) kept -> cluster x level")
        for line in table.to_string().splitlines():
            print(f"         {line}")
        frame.to_csv(os.path.join(out_dir, f"clusters_{movement.replace('-', '_')}.csv"),
                     index=False)

    # ------------------------------------------------------------------ 6/6
    print("\n[6/6] figures (svg)")
    from visualization.plot_distribution import plot_dual_mode_distribution
    from visualization.plot_heatmap import plot_correlation_heatmap, plot_mean_heatmap
    from visualization.plot_pca import plot_pca_visualization

    bending, sitstand = clustered["bending"], clustered["sit-stand"]
    cwd = os.getcwd()
    os.chdir(out_dir)                 # the plotting helpers write into the cwd
    try:
        if not bending.empty:
            plot_correlation_heatmap(bending, feature_cols, "Bending")
            plot_mean_heatmap(bending, feature_cols, "Bending", label_col="label")
            plot_pca_visualization(bending, feature_cols, "Bending", label_col="label")
        if not sitstand.empty:
            plot_pca_visualization(sitstand, feature_cols, "SitStand", label_col="label")
        if not bending.empty and not sitstand.empty:
            plot_dual_mode_distribution(bending, sitstand, feature_cols,
                                        label_col="label", plot_type="violin")
    finally:
        os.chdir(cwd)

    for name in sorted(f for f in os.listdir(out_dir) if f.endswith(".svg")):
        print(f"      {pretty(os.path.join(out_dir, name))}")

    # --------------------------------------------------------- optional DL
    if args.with_dl:
        from deep_learning.train import train_eval_model

        samples = [s for s in labelled if s["action_type"] == "bending"]
        if not samples:
            print("\n[+] 1-D CNN skipped: no 'bending' sample to train on")
        else:
            # Five folds require at least five subjects and five samples per
            # class; the demo has three subjects, so it degrades to
            # leave-one-subject-out.
            n_subjects = len({s["subject_id"] for s in samples})
            n_min_class = int(pd.Series([s["label"] for s in samples]).value_counts().min())
            folds = max(2, min(5, n_subjects, n_min_class))
            print(f"\n[+] 1-D CNN: {len(samples)} sample(s), {folds} fold(s), "
                  f"{args.epochs} epoch(s)")
            train_eval_model(samples, epochs=args.epochs, folds=folds)
    else:
        print("\n[+] 1D-CNN stage skipped (needs torch: pip install torch, then --with-dl)")

    print("\n>>> demo pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
