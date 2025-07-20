import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class RIRHDF5Dataset(Dataset):
    def __init__(self, rir_path: str = 'data/rir_dataset.h5', metrics_path: str = 'data/rir_metrics.h5', target_keys=None, normalize_rir=True, normalize_targets=True):
        self.rir_h5 = h5py.File(rir_path, 'r')
        self.metrics_h5 = h5py.File(metrics_path, 'r')
        self.rirs = self.rir_h5['rirs']
        self.target_keys = target_keys or ['rt60', 'edt', 'c50', 'd50']
        self.normalize_rir = normalize_rir
        self.normalize_targets = normalize_targets

        # Load all targets at once for normalization
        self.targets_raw = np.stack([self.metrics_h5[k][:] for k in self.target_keys], axis=1)

        if self.normalize_targets:
            self.target_mean = self.targets_raw.mean(axis=0)
            self.target_std = self.targets_raw.std(axis=0)
            self.targets = (self.targets_raw - self.target_mean) / (self.target_std + 1e-12)
        else:
            self.targets = self.targets_raw

    def __len__(self):
        return len(self.rirs)

    def __getitem__(self, idx):
        rir = self.rirs[idx]
        if self.normalize_rir:
            energy = np.sqrt(np.sum(rir**2)) + 1e-12
            rir = rir / energy

        rir_tensor = torch.tensor(rir, dtype=torch.float32)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

        return rir_tensor, target_tensor

    def close(self):
        self.rir_h5.close()
        self.metrics_h5.close()


def denormalize(preds: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_tensor = torch.from_numpy(mean).to(preds.device).type_as(preds)
    std_tensor = torch.from_numpy(std).to(preds.device).type_as(preds)
    return preds * std_tensor + mean_tensor