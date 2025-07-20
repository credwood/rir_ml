import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class RIRHDF5Dataset(Dataset):
    def __init__(
        self,
        rir_path: str = 'data/rir_dataset.h5',
        metrics_path: str = 'data/rir_metrics.h5',
        target_keys=None,
        normalize_targets=False,
        target_mean=None,
        target_std=None,
        subset_indices=None,
        normalize_rir=False  # ← added this missing argument
    ):
        self.rir_h5 = h5py.File(rir_path, 'r')
        self.metrics_h5 = h5py.File(metrics_path, 'r')
        self.rirs = self.rir_h5['rirs']
        self.target_keys = target_keys or ['rt60', 'edt', 'c50', 'd50']
        self.normalize_targets = normalize_targets
        self.normalize_rir = normalize_rir

        # Load all targets at once for normalization
        raw_targets = []
        for key in self.target_keys:
            values = self.metrics_h5[key][:]
            if key == 'c50':
                values = 10 ** (values / 10)  # convert dB to linear
            raw_targets.append(values)
        self.targets_raw = np.stack([self.metrics_h5[k][:] for k in self.target_keys], axis=1)

        # Handle subset if provided
        if subset_indices is not None:
            subset_indices = np.sort(np.array(subset_indices))
            self.rirs = self.rirs[subset_indices]
            self.targets = self.targets_raw[subset_indices]
        else:
            self.targets = self.targets_raw

        # Normalize targets if requested
        if normalize_targets:
            assert target_mean is not None and target_std is not None, "Must provide mean and std for normalization"
            self.target_mean = target_mean
            self.target_std = target_std
            self.targets = (self.targets - target_mean) / (target_std + 1e-12)
        else:
            self.target_mean = None
            self.target_std = None

        # Optional: ensure matching shapes
        assert self.rirs.shape[0] == self.targets.shape[0], "Mismatch between RIR and target count"

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

    def __del__(self):
        self.close()


def denormalize(preds: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_tensor = torch.from_numpy(mean).to(preds.device).type_as(preds)
    std_tensor = torch.from_numpy(std).to(preds.device).type_as(preds)
    return preds * std_tensor + mean_tensor

def convert_to_db(values: torch.Tensor, indices: list) -> torch.Tensor:
    """
    Converts specified columns from linear to dB in-place.
    `indices` is a list of indices for C50, D50 (e.g., [2, 3])
    """
    db_values = values.clone()
    db_values[:, indices] = 10 * torch.log10(db_values[:, indices] + 1e-12)
    return db_values

