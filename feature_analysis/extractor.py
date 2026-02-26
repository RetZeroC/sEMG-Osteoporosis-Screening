import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simps
from scipy.stats import kurtosis, skew
import pywt

def extract_features(emg_samples, sampling_rate=2000, wavelet='db4', level=3):
    features_list = []
    
    for sample in emg_samples:
        emg_data = sample['data']['EMG_Clean'].values
        if len(emg_data) <= 1: continue

        feat = {k: sample[k] for k in ['action_type', 'subject_id', 'channel_id', 'sample_id']}
        
        # Time Domain
        feat['MAV'] = np.mean(np.abs(emg_data))
        feat['RMS'] = np.sqrt(np.mean(emg_data**2))
        feat['IEMG'] = np.sum(np.abs(emg_data))
        feat['ZC'] = np.sum(np.diff(np.sign(emg_data)) != 0)
        feat['WL'] = np.sum(np.abs(np.diff(emg_data)))
        feat['VAR'] = np.var(emg_data)
        feat['SSC'] = np.sum(np.diff(np.sign(np.diff(emg_data))) != 0)
        feat['DAMV'] = np.mean(np.abs(np.diff(emg_data)))
        feat['PeakToPeak'] = np.max(emg_data) - np.min(emg_data)
        feat['Kurtosis'] = kurtosis(emg_data)
        feat['Skewness'] = skew(emg_data)

        # Frequency Domain
        nperseg = min(int(sampling_rate / 8), len(emg_data))
        f, Pxx = welch(emg_data, sampling_rate, nperseg=nperseg)
        total_power = simps(Pxx, f)
        
        feat['TotalPower'] = total_power
        feat['MNF'] = simps(f * Pxx, f) / total_power if total_power != 0 else 0
        
        power_sum = np.sum(Pxx)
        if power_sum != 0:
            median_idx = np.where(np.cumsum(Pxx) >= power_sum * 0.5)[0][0]
            feat['MDF'] = f[median_idx]
            feat['PKF'] = f[np.argmax(Pxx)]
        else:
            feat['MDF'] = feat['PKF'] = 0

        # Time-Frequency Domain (Wavelet)
        try:
            wp = pywt.WaveletPacket(data=emg_data, wavelet=wavelet, mode='symmetric', maxlevel=level)
            for k, node in enumerate(wp.get_level(level, 'natural')):
                feat[f'WPE_L{level}_E{k}'] = np.sum(node.data**2)
        except ValueError:
            for k in range(2**level): feat[f'WPE_L{level}_E{k}'] = 0

        features_list.append(feat)

    df = pd.DataFrame(features_list).replace([np.inf, -np.inf], np.nan).dropna()
    return df