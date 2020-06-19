# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Optional
from torch.utils.data.sampler import Sampler
import numpy as np


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = np.random.randint(2 ** 31)
        self._seed = int(seed)

    def __iter__(self):
        yield from self._infinite_indices()

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size)
            else:
                yield from torch.arange(self._size)


class EpisodicBatchSampler(Sampler):
    def __init__(self, n_classes, n_way, n_episodes, phase='train'):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes  # 100
        self.phase = phase

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            if self.phase == 'train':
                yield torch.randperm(self.n_classes)[:self.n_way].tolist()
            else:
                yield torch.arange(self.n_classes)[:self.n_way].tolist()
