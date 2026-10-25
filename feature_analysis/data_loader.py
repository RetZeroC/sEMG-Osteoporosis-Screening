"""Read raw sEMG recordings and cut them into muscle-activity segments.

The data tree is organised as ``<root>/<action_type>/<subject_id>/<channel_id>.csv``;
each CSV must contain a column named ``EMG_Clean`` with the filtered signal.
"""

import glob
import os

import numpy as np
import pandas as pd


def load_emg_data(root_path):
    """Load every recording below ``root_path``.

    Returns
    -------
    dict
        ``{action_type: [recording, ...]}`` where each recording is a dict with
        keys ``data`` (pandas DataFrame), ``action_type``, ``subject_id`` and
        ``channel_id``.
    """
    all_data = {}

    action_types = sorted(
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    )

    for action_type in action_types:
        action_path = os.path.join(root_path, action_type)
        csv_files = sorted(glob.glob(os.path.join(action_path, "**", "*.csv"), recursive=True))

        emg_data_list = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            emg_data_list.append({
                "data": df,
                "action_type": action_type,
                "subject_id": os.path.basename(os.path.dirname(csv_file)),
                "channel_id": os.path.basename(csv_file).split(".")[0],
            })
        all_data[action_type] = emg_data_list

    return all_data


def _segment_indices(rms, high_thresh, low_thresh):
    """Locate bursts in an RMS envelope by hysteresis thresholding.

    A segment opens when the envelope exceeds ``high_thresh`` and closes once it
    has stayed below ``low_thresh`` for more than 2000 samples. Segments shorter
    than 3500 or longer than 10000 samples are rejected; the trailing 1500
    samples (the relaxation phase) are trimmed from the returned end index.
    """
    segments = []
    is_segmenting = False
    seg_start, seg_gap = 0, 0

    for j, value in enumerate(rms):
        if not is_segmenting and value > high_thresh:
            is_segmenting = True
            seg_start = seg_gap = j
        elif is_segmenting and value > low_thresh:
            seg_gap = j
        elif is_segmenting and value < low_thresh:
            if j - seg_gap > 2000:
                is_segmenting = False
                if 3500 < j - seg_start < 10000:
                    segments.append((seg_start, j - 1500))

    return segments


def process_emg_data(loaded_data, sampling_rate=2000, window_size=0.2,
                     high_factor=0.9, low_factor=-0.03, min_length=3000):
    """Segment all recordings and return the individual bursts.

    Parameters
    ----------
    loaded_data : dict
        Output of :func:`load_emg_data`.
    sampling_rate : int
        Sampling rate of the recordings, in Hz.
    window_size : float
        Sliding RMS window, in seconds.
    high_factor, low_factor : float
        Onset / offset thresholds expressed in standard deviations of the RMS
        envelope, relative to its mean.
    min_length : int
        Segments shorter than this number of samples are discarded.

    Returns
    -------
    list of dict
        One entry per kept segment, with keys ``data``, ``action_type``,
        ``subject_id``, ``channel_id`` and ``sample_id``.
    """
    all_segments = []
    sample_counter = 0
    clean_factor = 30
    window_len = int(window_size * sampling_rate)

    all_files = [info for files in loaded_data.values() for info in files]

    for file_info in all_files:
        emg_clean = file_info["data"]["EMG_Clean"].values.copy()

        # Clip impulsive artefacts (>30 sigma) to the signal mean.
        mean_val, std_val = np.mean(emg_clean), np.std(emg_clean)
        outliers = np.where(np.abs(emg_clean - mean_val) > clean_factor * std_val)[0]
        if len(outliers) > 0:
            emg_clean[outliers] = mean_val

        if len(emg_clean) < window_len:
            continue

        rms = np.sqrt(np.convolve(emg_clean ** 2, np.ones(window_len) / window_len, mode="valid"))
        rms_mean, rms_std = np.mean(rms), np.std(rms)
        high_thresh = rms_mean + high_factor * rms_std
        low_thresh = rms_mean + low_factor * rms_std

        for start, end in _segment_indices(rms, high_thresh, low_thresh):
            # The RMS envelope is shorter than the signal by window_len samples.
            start_idx = int(start * (len(emg_clean) / len(rms)))
            end_idx = int(end * (len(emg_clean) / len(rms))) + window_len
            segment = emg_clean[start_idx:end_idx]

            if len(segment) >= min_length:
                all_segments.append({
                    "data": pd.DataFrame({"EMG_Clean": segment}),
                    "action_type": file_info["action_type"],
                    "subject_id": file_info["subject_id"],
                    "channel_id": file_info["channel_id"],
                    "sample_id": (f"{file_info['action_type']}_{file_info['subject_id']}"
                                  f"_{file_info['channel_id']}_{sample_counter}"),
                })
                sample_counter += 1

    return all_segments
