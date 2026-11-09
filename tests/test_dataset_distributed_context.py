"""Tests for model-parallel-safe iterable dataset sharding."""

from itertools import islice

import pytest

from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset


def test_distributed_context_uses_data_parallel_topology():
    dataset = object.__new__(DistributedRepeatingDataset)
    dataset.rank = 3
    dataset.world_size = 8

    dataset.set_distributed_context(rank=1, world_size=4)

    assert dataset.rank == 1
    assert dataset.world_size == 4


def _index_dataset(global_rank: int, global_world_size: int):
    dataset = object.__new__(DistributedRepeatingDataset)
    dataset.rank = global_rank
    dataset.world_size = global_world_size
    dataset.total_len = 12
    dataset.shuffle = False
    dataset.reshuffle_each_epoch = False
    dataset.seed = 0
    dataset._epoch = 0
    dataset.replay_indices = None
    dataset._get_item_from_global_idx = lambda index: index
    return dataset


def test_tensor_parallel_ranks_share_the_data_parallel_shard():
    rank0_dataset = _index_dataset(global_rank=0, global_world_size=2)
    rank1_dataset = _index_dataset(global_rank=1, global_world_size=2)

    for dataset in (rank0_dataset, rank1_dataset):
        dataset.set_distributed_context(rank=0, world_size=1)

    rank0_indices = list(islice(iter(rank0_dataset), 12))
    rank1_indices = list(islice(iter(rank1_dataset), 12))
    assert rank0_indices == rank1_indices == list(range(12))


def test_data_parallel_ranks_receive_disjoint_shards():
    rank0_dataset = _index_dataset(global_rank=0, global_world_size=4)
    rank1_dataset = _index_dataset(global_rank=1, global_world_size=4)
    rank0_dataset.set_distributed_context(rank=0, world_size=2)
    rank1_dataset.set_distributed_context(rank=1, world_size=2)

    assert list(islice(iter(rank0_dataset), 6)) == [0, 2, 4, 6, 8, 10]
    assert list(islice(iter(rank1_dataset), 6)) == [1, 3, 5, 7, 9, 11]


def test_replay_indices_are_sharded_by_data_parallel_rank():
    rank0_dataset = _index_dataset(global_rank=0, global_world_size=4)
    rank1_dataset = _index_dataset(global_rank=1, global_world_size=4)
    rank0_dataset.set_distributed_context(rank=0, world_size=2)
    rank1_dataset.set_distributed_context(rank=1, world_size=2)
    replay = [8, 3, 11, 2]
    rank0_dataset.replay_indices = replay
    rank1_dataset.replay_indices = replay

    assert list(islice(iter(rank0_dataset), 4)) == [8, 11, 8, 11]
    assert list(islice(iter(rank1_dataset), 4)) == [3, 2, 3, 2]


@pytest.mark.parametrize("value", ["", "abc", "1,12"])
def test_replay_indices_parser_rejects_invalid_values(value):
    dataset = object.__new__(DistributedRepeatingDataset)
    dataset.total_len = 12
    if value == "":
        assert dataset._parse_replay_indices(value) is None
    else:
        with pytest.raises(ValueError):
            dataset._parse_replay_indices(value)


def test_replay_indices_parser_accepts_comma_and_whitespace():
    dataset = object.__new__(DistributedRepeatingDataset)
    dataset.total_len = 12
    assert dataset._parse_replay_indices("8, 3 11") == [8, 3, 11]


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_distributed_context_rejects_invalid_topology(rank, world_size):
    dataset = object.__new__(DistributedRepeatingDataset)

    with pytest.raises(ValueError):
        dataset.set_distributed_context(rank=rank, world_size=world_size)
