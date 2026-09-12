"""The sampler is what makes mid-epoch resume correct rather than approximate."""

from clinhallu.data.resumable_sampler import ResumableRandomSampler


class _Dataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def test_order_depends_only_on_seed_and_epoch():
    first = ResumableRandomSampler(_Dataset(100), seed=13, epoch=3)
    # A fresh process, a fresh object, arbitrary intervening RNG use:
    second = ResumableRandomSampler(_Dataset(100), seed=13, epoch=3)
    list(ResumableRandomSampler(_Dataset(100), seed=13, epoch=1))
    assert list(first) == list(second)


def test_different_epochs_give_different_orders():
    orders = [list(ResumableRandomSampler(_Dataset(200), seed=13, epoch=e)) for e in (1, 2, 3)]
    assert orders[0] != orders[1] != orders[2]
    # Every epoch is still a full permutation.
    assert all(sorted(order) == list(range(200)) for order in orders)


def test_skip_drops_exactly_the_consumed_prefix():
    full = list(ResumableRandomSampler(_Dataset(50), seed=7, epoch=2))
    resumed = ResumableRandomSampler(_Dataset(50), seed=7, epoch=2, skip=20)
    assert list(resumed) == full[20:]
    assert len(resumed) == 30


def test_interrupted_epoch_covers_every_example_exactly_once():
    """The union of the pre-crash and post-crash portions is the full epoch."""
    sampler = ResumableRandomSampler(_Dataset(64), seed=13, epoch=4)
    before = list(sampler)[:24]
    sampler.set_skip(24)
    after = list(sampler)
    assert sorted(before + after) == list(range(64))
    assert not set(before) & set(after)


def test_set_epoch_switches_the_order_and_resets_nothing_else():
    sampler = ResumableRandomSampler(_Dataset(30), seed=13, epoch=1, skip=5)
    sampler.set_epoch(2)
    assert sampler.skip == 5
    assert list(sampler) == list(ResumableRandomSampler(_Dataset(30), seed=13, epoch=2))[5:]


def test_skip_beyond_the_dataset_yields_nothing():
    sampler = ResumableRandomSampler(_Dataset(10), seed=1, skip=999)
    assert list(sampler) == []
    assert len(sampler) == 0
