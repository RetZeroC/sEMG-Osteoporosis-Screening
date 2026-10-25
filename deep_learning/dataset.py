"""Dataset and collation utilities for the 1-D convolutional network."""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def augment_signal(signal, sigma=0.01, shift_max=20, mask_ratio=0.05):
    """Return an augmented copy of ``signal``.

    Gaussian noise, a random circular shift and a short zeroed window are
    applied; these operations preserve the spectral content of the segment.
    """
    augmented = signal.copy()
    augmented += np.random.normal(0, sigma, size=augmented.shape)
    augmented = np.roll(augmented, np.random.randint(-shift_max, shift_max))

    mask_len = int(len(augmented) * mask_ratio)
    start = np.random.randint(0, len(augmented) - mask_len)
    augmented[start:start + mask_len] = 0
    return augmented


def balance_and_augment_data(data_list, target_label, augment_factor):
    """Append ``augment_factor`` synthetic copies of the ``target_label`` samples.

    Provided for class-imbalance experiments; the pipeline reported in the paper
    does not enable it by default.
    """
    augmented = []
    for sample in data_list:
        if sample["label"] == target_label:
            for _ in range(augment_factor):
                aug_sample = sample.copy()
                aug_sample["data"] = sample["data"].copy()
                aug_sample["data"]["EMG_Clean"] = augment_signal(
                    sample["data"]["EMG_Clean"].values
                )
                augmented.append(aug_sample)
    return data_list + augmented


class EMGDataset(Dataset):
    """Per-segment dataset; signals are standardised and left unpadded."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        emg_signal = sample["data"]["EMG_Clean"].values.astype(np.float32)

        mean_val = np.mean(emg_signal)
        std_val = np.std(emg_signal)
        if std_val > 1e-8:
            emg_signal = (emg_signal - mean_val) / std_val
        else:
            emg_signal = emg_signal - mean_val

        label = sample["label"]
        return torch.tensor(emg_signal), torch.tensor(label, dtype=torch.long)


def pad_collate(batch):
    """Pad variable-length segments to the longest one in the batch."""
    signals, labels = zip(*batch)
    padded = pad_sequence(signals, batch_first=True, padding_value=0.0).unsqueeze(1)
    return padded, torch.stack(labels)
