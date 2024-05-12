import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
from hyperparams import HyperParams
import matplotlib.pyplot as plt
from typing import Callable
# import warnings
# warnings.filterwarnings("ignore")

def uniform_sampler():
    return lambda: np.random.randint(1, 99)

def random_d4_transform(x_batch, y_batch):
    batch_size = x_batch.shape[0]

    # horizontal flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, :, ::-1]
    y_batch[mask] = y_batch[mask, :, ::-1]

    # vertical flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, ::-1]
    y_batch[mask] = y_batch[mask, ::-1]

    # 90* rotation
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = np.swapaxes(x_batch[mask], 1, 2)
    y_batch[mask] = np.swapaxes(y_batch[mask], 1, 2)

    return x_batch, y_batch


class TODataset(Dataset):
    def __init__(self, path: str, iter_sampler: Callable, transforms: Callable=None) -> None:
        self.h5f = h5py.File(path, "r")
        self.iter_sampler = iter_sampler
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.h5f['iters'])

    def __getitem__(self, idx: int):
        X = self.h5f['iters'][idx]
        y = self.h5f['targets'][idx]

        iter_ = self.iter_sampler()
        x1 = X[:, :, iter_]
        x2 = X[:, :, iter_ - 1]
        x = np.stack((x1, x1 - x2), -1)

        if self.transforms:
            batch_x = np.expand_dims(x, axis=0)
            batch_y = np.expand_dims(y, axis=0)
            # print(batch_x.shape, batch_y.shape)
            x, y = self.transforms(batch_x, batch_y)
            x = x[0]
            y = y[0]
            # print(x.shape, y.shape)

        iters = torch.tensor(x, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        iters_permuted = iters.permute(2, 0, 1)
        targets_permuted = targets.permute(2, 0, 1)

        return {
            "iters": iters_permuted,
            "targets": targets_permuted
        }


if __name__ == "__main__":
    ds = TODataset("/Users/0x4ry4n/Desktop/dev/btp/h5ds/dataset.h5", uniform_sampler(), transforms=None)
    print(len(ds))
    print(ds[0]['iters'].shape)
    print(ds[0]['targets'].shape)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(ds[0]['iters'][:, :, 0])  # nth iter
    axarr[1].imshow(ds[0]['iters'][:, :, 1])  # n - (n-1)th iter
    axarr[2].imshow(ds[0]['targets'])  # target
    plt.show()