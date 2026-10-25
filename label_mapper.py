"""Attach clinical diagnosis labels to segmented sEMG samples."""

import pandas as pd


def merge_real_labels(emg_samples, metadata_path, subject_col="subject_id",
                      label_col="diagnosis_level"):
    """Return the samples whose subject appears in the clinical metadata table.

    Parameters
    ----------
    emg_samples : list of dict
        Segments produced by :func:`feature_analysis.data_loader.process_emg_data`.
        Each entry carries a ``subject_id`` key.
    metadata_path : str
        CSV file with at least ``subject_col`` and ``label_col``.
    subject_col : str
        Column holding the subject identifier (matched against the subject
        directory name used in the data tree).
    label_col : str
        Column holding the integer diagnosis level (0, 1 or 2).

    Returns
    -------
    list of dict
        The subset of ``emg_samples`` that could be matched, each with an
        integer ``label`` entry added. Samples from unlisted subjects are
        discarded.
    """
    try:
        metadata_df = pd.read_csv(metadata_path)
        metadata_df[subject_col] = metadata_df[subject_col].astype(str)
        label_map = dict(zip(metadata_df[subject_col], metadata_df[label_col]))
    except Exception:
        return []

    labelled_samples = []
    for sample in emg_samples:
        subject_id = str(sample["subject_id"])
        if subject_id in label_map:
            sample["label"] = int(label_map[subject_id])
            labelled_samples.append(sample)
    return labelled_samples
