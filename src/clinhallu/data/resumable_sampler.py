from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):

  def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        epoch: int = 1,
        skip: int = 0,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self._skip = 0
        self.set_skip(skip)

    @property
    def total_size(self) -> int:
        return len(self.data_source)

    @property
    def skip(self) -> int:
        return self._skip

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_skip(self, skip: int) -> None:
        skip = int(skip)
        if skip < 0:
            raise ValueError("skip must be non-negative")
        self._skip = min(skip, self.total_size)

    def permutation(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed((self.seed * 1_000_003 + self.epoch) % (2**63 - 1))
        return torch.randperm(self.total_size, generator=generator)

    def __iter__(self) -> Iterator[int]:
        order = self.permutation()
        if self._skip:
            order = order[self._skip :]
        yield from order.tolist()

    def __len__(self) -> int:
        return max(self.total_size - self._skip, 0)
