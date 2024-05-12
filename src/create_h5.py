import os
import h5py
import numpy as np
from tqdm import tqdm
from hyperparams import HyperParams

files = os.listdir("TOP4040")
iters_shape = (len(files), HyperParams.IMAGE_H, HyperParams.IMAGE_W, HyperParams.N_ITERS)
iters_chunk_shape = (1, HyperParams.IMAGE_H, HyperParams.IMAGE_W, 1)
target_shape = (len(files), HyperParams.IMAGE_H, HyperParams.IMAGE_W, 1)
target_chunk_shape = (1, HyperParams.IMAGE_H, HyperParams.IMAGE_W, 1)


with h5py.File("dataset.h5", 'w') as h5f:
    iters = h5f.create_dataset('iters', iters_shape, chunks=iters_chunk_shape)
    targets = h5f.create_dataset('targets', target_shape, chunks=target_chunk_shape)

    for i, file_name in tqdm(enumerate(files), total=len(files)):
        file_path = os.path.join("TOP4040", file_name)
        arr = np.load(file_path)['arr_0']
        arr = arr.transpose((1, 2, 0))
        iters[i] = arr

        th_ = arr.mean(axis=(0, 1), keepdims=True)
        targets[i] = (arr > th_).astype('float32')[:, :, [-1]]