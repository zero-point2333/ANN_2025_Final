# open_groundingdino/util/compat_sampler.py
import math
import numpy as np

class CompatDistributedSampler:
    """
    PyTorch-style distributed sampler for any Dataset implementing __len__.
    It shards indices by rank and supports set_epoch for deterministic shuffle.
    """
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        n = len(self.dataset)
        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(indices)

        if not self.drop_last:
            # pad to total_size
            if len(indices) < self.total_size:
                indices += indices[: (self.total_size - len(indices))]
        else:
            indices = indices[: self.total_size]

        # shard
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples