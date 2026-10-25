# Supporting Information

**Eutectic-induced Polymerization Enables Skin-integrated and Robust Bioelectrode for High-Fidelity sEMG and AI-assisted Osteoporosis Screening**

This repository contains the analysis code used in the manuscript. Surface electromyography
(sEMG) recordings acquired with the bioelectrode are segmented into muscle-activity bursts,
described by 23 time-, frequency- and time-frequency-domain features, explored by
unsupervised clustering and principal component analysis (PCA), and classified into three
diagnosis levels by a one-dimensional convolutional neural network (1D-CNN).

---

## 1. Repository contents

| Path | Description |
| --- | --- |
| `main.py` | Full pipeline: segmentation, feature extraction, clustering, figures, 1D-CNN |
| `label_mapper.py` | Attaches the clinical diagnosis level to each segment |
| `feature_analysis/data_loader.py` | Reads recordings and segments muscle activity |
| `feature_analysis/extractor.py` | 23-dimensional feature extraction |
| `feature_analysis/clustering.py` | Isolation forest followed by k-means |
| `visualization/plot_heatmap.py` | Feature-correlation and per-level mean heat maps |
| `visualization/plot_pca.py` | 2D and 3D PCA projections |
| `visualization/plot_distribution.py` | Feature distributions per level and movement |
| `deep_learning/dataset.py` | Dataset, sequence padding, optional augmentation |
| `deep_learning/model.py` | 1D-CNN (`EMGNet`) and focal loss |
| `deep_learning/train.py` | Subject-wise cross-validated training and evaluation |
| `demo/` | Synthetic demo dataset, data generator and demo driver |
| `.gitignore` | Keeps generated figures and caches out of version control |
| `LICENSE` | MIT licence |

## 2. Software requirements

The code was developed and tested with Python 3.10–3.12 (3.12 used for validation).

| Package | Version used | Role |
| --- | --- | --- |
| `numpy` | 2.3.5 | Numerical arrays |
| `pandas` | 3.0.1 | Recording and feature tables |
| `scipy` | 1.18.1 | Welch spectrum, statistical moments |
| `PyWavelets` | 1.8 | Wavelet-packet decomposition |
| `scikit-learn` | 1.9.1 | Clustering, scaling, cross-validation, metrics |
| `matplotlib` | 3.11.2 | Figure rendering |
| `seaborn` | 0.13.2 | Statistical plotting |
| `torch` | ≥ 2.0 | 1D-CNN; required by `main.py` and by `--with-dl` only |

```bash
# Analysis, clustering and figures
pip install numpy pandas scipy PyWavelets scikit-learn matplotlib seaborn

# Optional: 1D-CNN training (CPU build is sufficient)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

A CUDA build of PyTorch may be used instead; the training code selects the device
automatically.

## 3. Data format

Recordings are organised by movement task, subject and channel:

```
data/
├── bending/                    # movement task; the folder name is the action_type
│   ├── S01/
│   │   ├── ch1.csv             # channel_id is the file name without extension
│   │   └── ch2.csv
│   └── S02/ ...
└── sit-stand/
    └── S01/ ...
```

Each CSV file holds one continuous recording and must contain a column named `EMG_Clean`
(the filtered signal); additional columns are ignored. One file may contain several
contractions — segmentation is performed automatically (Section 7).

The clinical labels are supplied by `metadata_clinical_diagnosis.csv` in the repository
root:

| Column | Content |
| --- | --- |
| `subject_id` | Subject identifier, identical to the subject folder name |
| `diagnosis_level` | Diagnosis level as an integer, 0 (normal), 1 (osteopenia) or 2 (osteoporosis) |

Segments whose subject does not appear in this table are discarded. The sampling rate is
assumed to be 2000 Hz throughout.

## 4. Synthetic demo dataset

`demo/data` contains 12 synthetic recordings — three simulated subjects × two movement
tasks × two channels, 8.5 s (17 000 samples) each, 1.6 MB in total. No participant data
are included or redistributed. The signals are produced by `demo/make_demo_data.py` from
the superposition of random-phase sinusoids within a burst envelope plus additive noise;
the frequency band and amplitude depend on the simulated diagnosis level
(`CLASS_MODEL` in that script). The model reproduces the qualitative trend used to make the
demo meaningful and is not a physiological model, so results obtained on this data have no
clinical interpretation.

```bash
python demo/make_demo_data.py            # regenerate demo/data and the metadata table
python demo/make_demo_data.py --check    # verify file layout, columns and lengths
```

## 5. Usage

```bash
# Demo on the synthetic dataset (CPU, approximately 1 min)
python demo/run_demo.py

# Same, including the 1D-CNN stage (requires PyTorch; 3 folds, 10 epochs)
python demo/run_demo.py --with-dl

# Full pipeline on the study data (run from the repository root)
python main.py
```

`demo/run_demo.py` may be called from any directory and writes to `demo/output`
(`--output-dir` overrides this). `main.py` reads `data/` and
`metadata_clinical_diagnosis.csv` relative to the working directory, writes the figures to
`output_plots/`, and then trains one 1D-CNN per movement task with 100 epochs and 5 folds;
the paths, epoch count and fold count are defined at the top of the script.

With `--with-dl`, the number of folds is reduced to
`min(5, number of subjects, samples in the smallest class)`; the demo therefore runs
leave-one-subject-out validation with three folds. In the full pipeline the fold count is
fixed at five, which requires at least five subjects and at least five segments per class.

The console output of the demo is reproduced in `demo/expected_output.txt`; the first and
last steps are:

```
[1/6] load raw sEMG files
      bending      6 file(s)
      sit-stand    6 file(s)
      total       12 file(s)
[2/6] segment activity bursts (RMS threshold, 0.2 s window)
      24 segment(s) kept (2.0 per file)
...
[6/6] figures (svg)
      demo/output/pca_2d_Bending.svg
      ...
>>> demo pipeline finished
```

## 6. Output files

| File | Content |
| --- | --- |
| `<action>_feature_correlation.svg` | Pearson correlation matrix of the 23 features |
| `<action>_mean_heatmap.svg` | Per-level mean of each feature, standardised across levels |
| `pca_2d_<action>.svg` | PC1–PC2 projection, with density contours per level |
| `pca_3d_<action>.svg` | PC1–PC3 projection |
| `dual_distribution_<type>.svg` | Feature distributions, bending (top) and sit-stand (bottom) |
| `features.csv` | Segment metadata and features (demo driver only) |
| `clusters_<action>.csv` | Retained segments and their cluster assignment (demo driver only) |
| `classification_report` (console) | Precision, recall and F1 per level over all folds |

`<action>` is `Bending` or `SitStand` for the figures and `bending` or `sit-stand` for the
CSV files.

## 7. Analysis parameters

| Stage | Parameter | Value |
| --- | --- | --- |
| Segmentation | Sliding RMS window | 0.2 s (400 samples at 2 kHz) |
| | Onset / offset threshold | mean + 0.9 σ / mean − 0.03 σ of the RMS envelope |
| | Minimum duration below offset | 2000 samples |
| | Accepted burst length | 3500–10 000 samples |
| | Trailing samples excluded | 1500 |
| | Minimum segment length | 3000 samples |
| | Artefact clipping | samples beyond ±30 σ set to the mean |
| Features | Time domain | MAV, RMS, IEMG, ZC, WL, VAR, SSC, DAMV, peak-to-peak, kurtosis, skewness |
| | Frequency domain | total power, mean, median and peak frequency |
| | Time–frequency domain | energy of eight wavelet-packet bands (`db4`, level 3) |
| Clustering | Outlier removal | isolation forest, contamination 0.5 |
| | Partitioning | k-means, k = 3, standardised features |
| Classification | Network | three convolutional blocks (32/64/128 filters) and a 128–64–3 head |
| | Objective | focal loss, γ = 2, class weights inversely proportional to frequency |
| | Optimisation | Adam, learning rate 10⁻³, batch size 32, 100 epochs |
| | Validation | stratified grouped 5-fold cross-validation, grouped by subject |

Random seeds are fixed at 42 for the fold split, k-means, PCA and the isolation forest.
PyTorch training is not seeded, so repeated runs may differ slightly; the demo is intended
to verify that the pipeline executes, not to reproduce the reported performance.

## 8. Data and code availability

The sEMG recordings analysed in the manuscript are not publicly available because they
contain participant information; requests should be addressed to the corresponding author.
The synthetic dataset in `demo/data` and the code in this repository are provided so that
the analysis can be executed and inspected. The code is intended for research use only and
is not a medical device.

## 9. Licence

Released under the MIT Licence (Copyright © 2026 RetZeroC); see `LICENSE`.
