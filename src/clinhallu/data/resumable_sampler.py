"""A shuffling sampler whose order depends only on ``(seed, epoch)``.

``DataLoader(shuffle=True, generator=g)`` draws its permutation from a generator
that advances every epoch. That is fine for an uninterrupted run, but it makes
resuming wrong in two ways:

* re-seeding ``g`` at process start replays epoch 1's order for whatever epoch
  the run actually resumes into, so examples get revisited and others skipped;
* there is no way to say "continue 3,200 examples into this epoch", which is
  exactly what mid-epoch resume needs.

:class:`ResumableRandomSampler` derives the permutation from a hash of
``(seed, epoch)``, so epoch *k* always yields the same order no matter when the
process started. ``set_skip()`` then drops the leading examples that a previous
process already trained on.
"""

from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Shuffle deterministically per epoch and support mid-epoch continuation.

    Args:
        data_source: dataset (only ``len()`` is used).
        seed: run seed; combined with the epoch to derive the permutation.
        epoch: epoch index this sampler currently represents.
        skip: number of examples at the front of the epoch to omit.
    """

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
        """Examples in a full epoch, ignoring any skip."""
        return len(self.data_source)

    @property
    def skip(self) -> int:
        return self._skip

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose permutation should be produced next."""
        self.epoch = int(epoch)

    def set_skip(self, skip: int) -> None:
        """Omit the first ``skip`` examples of the current epoch's order."""
        skip = int(skip)
        if skip < 0:
            raise ValueError("skip must be non-negative")
        self._skip = min(skip, self.total_size)

    def permutation(self) -> torch.Tensor:
        """The full, un-skipped order for the current epoch."""
        generator = torch.Generator()
        # Mixing with a large odd constant keeps neighbouring epochs from
        # producing correlated orders when seeds are small (13, 42, ...).
        generator.manual_seed((self.seed * 1_000_003 + self.epoch) % (2**63 - 1))
        return torch.randperm(self.total_size, generator=generator)

    def __iter__(self) -> Iterator[int]:
        order = self.permutation()
        if self._skip:
            order = order[self._skip :]
        yield from order.tolist()

    def __len__(self) -> int:
        return max(self.total_size - self._skip, 0)
