"""Subject-wise cross-validated training and evaluation of :class:`EMGNet`."""

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from deep_learning.dataset import EMGDataset, pad_collate
from deep_learning.model import EMGNet, FocalLoss


def train_eval_model(data_list, epochs=20, folds=5):
    """Train and evaluate the 1-D CNN with grouped, stratified k-fold CV.

    Folds are split on ``subject_id`` so that no subject contributes segments to
    both the training and the validation side of a fold.

    Parameters
    ----------
    data_list : list of dict
        Labelled segments (``data``, ``label`` and ``subject_id`` keys).
    epochs : int
        Number of training epochs per fold.
    folds : int
        Number of cross-validation folds.

    Returns
    -------
    tuple of (list, list)
        Ground-truth labels and predictions accumulated over all folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_all = [sample["label"] for sample in data_list]
    groups = [sample["subject_id"] for sample in data_list]

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    all_preds, all_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data_list, y_all, groups=groups)):
        print(f"\n--- Fold {fold + 1}/{folds} ---")
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        train_loader = DataLoader(EMGDataset(train_data), batch_size=32, shuffle=True,
                                  collate_fn=pad_collate)
        val_loader = DataLoader(EMGDataset(val_data), batch_size=32, shuffle=False,
                                collate_fn=pad_collate)

        model = EMGNet().to(device)

        # Class weights are set inversely proportional to the fold's frequencies.
        train_labels = [sample["label"] for sample in train_data]
        class_counts = np.bincount(train_labels, minlength=3)
        alpha = torch.tensor(len(train_labels) / (3 * (class_counts + 1e-5)),
                             dtype=torch.float32).to(device)

        criterion = FocalLoss(alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(epochs):
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
