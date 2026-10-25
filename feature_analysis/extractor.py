"""Feature extraction from segmented sEMG signals.

Twenty-three descriptors are computed per segment: eleven in the time domain,
four in the frequency domain and eight wavelet-packet band energies.
"""

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew

try:  # SciPy >= 1.14 renamed simps -> simpson
    from scipy.integrate import simpson as _integrate
except ImportError:  # SciPy < 1.14
    from scipy.integrate import simps as _integrate

SEGMENT_KEYS = ("action_type", "subject_id", "channel_id", "sample_id")


def _time_domain_features(signal):
    """Amplitude- and complexity-based descriptors of the raw segment."""
    return {
        "MAV": np.mean(np.abs(signal)),
        "RMS": np.sqrt(np.mean(signal ** 2)),
        "IEMG": np.sum(np.abs(signal)),
        "ZC": np.sum(np.diff(np.sign(signal)) != 0),
        "WL": np.sum(np.abs(np.diff(signal))),
        "VAR": np.var(signal),
        "SSC": np.sum(np.diff(np.sign(np.diff(signal))) != 0),
        "DAMV": np.mean(np.abs(np.diff(signal))),
        "PeakToPeak": np.max(signal) - np.min(signal),
        "Kurtosis": kurtosis(signal),
        "Skewness": skew(signal),
    }


def _frequency_domain_features(signal, sampling_rate):
    """Power-spectrum descriptors (median/mean/peak frequency, total power)."""
    nperseg = min(int(sampling_rate / 8), len(signal))
    freqs, psd = welch(signal, sampling_rate, nperseg=nperseg)

    total_power = _integrate(psd, freqs)
    features = {
        "TotalPower": total_power,
        "MNF": _integrate(freqs * psd, freqs) / total_power if total_power != 0 else 0,
    }

    power_sum = np.sum(psd)
    if power_sum != 0:
        median_idx = np.where(np.cumsum(psd) >= power_sum * 0.5)[0][0]
        features["MDF"] = freqs[median_idx]
        features["PKF"] = freqs[np.argmax(psd)]
    else:
        features["MDF"] = features["PKF"] = 0

    return features


def _wavelet_features(signal, wavelet, level):
    """Energy of the ``2 ** level`` wavelet-packet bands at the finest level."""
    features = {}
    try:
        packet = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric",
                                    maxlevel=level)
        nodes = packet.get_level(level, "natural")
        for k, node in enumerate(nodes):
            features[f"WPE_L{level}_E{k}"] = np.sum(node.data ** 2)
    except ValueError:
        for k in range(2 ** level):
            features[f"WPE_L{level}_E{k}"] = 0
    return features


def extract_features(emg_samples, sampling_rate=2000, wavelet="db4", level=3):
    """Compute the 23-dimensional feature table for a list of segments.

    Parameters
    ----------
    emg_samples : list of dict
        Labelled segments produced by the pipeline; each holds a ``data``
        DataFrame with an ``EMG_Clean`` column.
    sampling_rate : int
        Sampling rate of the signal, in Hz.
    wavelet : str
        Mother wavelet used for the wavelet-packet decomposition.
    level : int
        Decomposition level; eight bands are obtained at level 3.

    Returns
    -------
    pandas.DataFrame
        One row per segment: the segment metadata (including ``label`` when
        present) followed by the feature columns. Rows containing non-finite
        values are dropped.
    """
    features_list = []

    for sample in emg_samples:
        signal = sample["data"]["EMG_Clean"].values
        if len(signal) <= 1:
            continue

        row = {key: sample[key] for key in SEGMENT_KEYS}
        if "label" in sample:
            row["label"] = sample["label"]

        row.update(_time_domain_features(signal))
        row.update(_frequency_domain_features(signal, sampling_rate))
        row.update(_wavelet_features(signal, wavelet, level))
        features_list.append(row)

    features_df = pd.DataFrame(features_list)
    return features_df.replace([np.inf, -np.inf], np.nan).dropna()
