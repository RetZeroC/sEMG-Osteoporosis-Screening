import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def augment_signal(signal, sigma=0.01, shift_max=20, mask_ratio=0.05):
    aug = signal.copy()
    aug += np.random.normal(0, sigma, size=aug.shape)
    aug = np.roll(aug, np.random.randint(-shift_max, shift_max))
    
    mask_len = int(len(aug) * mask_ratio)
    start = np.random.randint(0, len(aug) - mask_len)
    aug[start:start + mask_len] = 0
    return aug

def balance_and_augment_data(data_list, target_label, augment_factor):

    augmented = []
    for sample in data_list:
        if sample['label'] == target_label:
            for _ in range(augment_factor):
                aug_sample = sample.copy()
                aug_sample['data'] = sample['data'].copy()
                aug_sample['data']['EMG_Clean'] = augment_signal(sample['data']['EMG_Clean'].values)
                augmented.append(aug_sample)
    return data_list + augmented

class EMGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        emg_signal = sample['data']['EMG_Clean'].values.astype(np.float32)
        
        mean_val = np.mean(emg_signal)
        std_val = np.std(emg_signal)
        if std_val > 1e-8:
            emg_signal = (emg_signal - mean_val) / std_val
        else:
            emg_signal = emg_signal - mean_val
            
        label = sample['label'] 
        return torch.tensor(emg_signal), torch.tensor(label, dtype=torch.long)

def pad_collate(batch):
    signals, labels = zip(*batch)
    padded = pad_sequence(signals, batch_first=True, padding_value=0.0).unsqueeze(1)
    return padded, torch.stack(labels)