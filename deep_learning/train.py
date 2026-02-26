import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from deep_learning.dataset import EMGDataset, pad_collate
from deep_learning.model import EMGNet, FocalLoss

def train_eval_model(data_list, epochs=20, folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_all = [sample['label'] for sample in data_list] 

    groups = [sample['subject_id'] for sample in data_list]
    
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    all_preds, all_labels = [], []


    for fold, (train_idx, val_idx) in enumerate(sgkf.split(data_list, y_all, groups=groups)):
        print(f"\n--- Fold {fold+1}/{folds} ---")
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(EMGDataset(train_data), batch_size=32, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(EMGDataset(val_data), batch_size=32, shuffle=False, collate_fn=pad_collate)

        model = EMGNet().to(device)
        
        train_labels = [s['label'] for s in train_data]
        class_counts = np.bincount(train_labels, minlength=3)
        alpha = torch.tensor(len(train_labels) / (3 * (class_counts + 1e-5)), dtype=torch.float32).to(device)
        
        criterion = FocalLoss(alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()


        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

    print("\n>>> Final Cross-Validation Report <<<")
    print(classification_report(all_labels, all_preds))
    return all_labels, all_preds