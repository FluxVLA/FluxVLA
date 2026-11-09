# Copyright 2026 Limx Dynamics
"""Epoch-local augmentation replay and mixed DataLoader worker ownership."""

import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
from fluxvla.engines.runners.mixed_training import (MixedTraining,
                                                    TaskDataStream,
                                                    _capture_rng, _restore_rng)
from fluxvla.transforms.transform_images import (ColorJitterImages,
                                                 RandomCropImages)


class AugmentedImages:
    """Real crop/jitter plus all three worker RNGs; no files or checkpoints."""

    stats = []

    def __init__(self):
        self.image = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32)
        self.crop = RandomCropImages(scale=0.95)
        self.jitter = ColorJitterImages()

    def __len__(self):
        return 12

    def __getitem__(self, index, statistics=None):
        result = self.jitter(self.crop(dict(images=[self.image.copy()])))
        return dict(
            index=int(index),
            pixels=hashlib.sha256(result['images'][0].tobytes()).hexdigest(),
            random=(random.random(), float(np.random.rand()), torch.rand(
                ()).item()))


def make_stream(workers, iterable=False):
    dataset = AugmentedImages()
    if iterable:
        dataset = DistributedRepeatingDataset(
            datasets=[dataset],
            statistic_keys=[],
            dataset_statistics={},
            reshuffle_each_epoch=True)
    return TaskDataStream(
        dataset=dataset,
        collator=list,
        batch_size=1,
        num_workers=workers,
        seed=42,
        rank=0,
        world_size=1)


def consume(stream, count):
    result = []
    for _ in range(count):
        result.append(stream.next())
        # Model-side random draws must survive iterator creation/replay too.
        result.append((random.random(), float(np.random.rand()), torch.rand(
            ()).item()))
    return result


@pytest.mark.parametrize('workers', [0, 2, 4])
@pytest.mark.parametrize('iterable', [False, True])
@pytest.mark.parametrize('epoch,offset', [(0, 3), (1, 3), (2, 11), (1, 12)])
def test_augmented_resume_matches_across_epochs(monkeypatch, workers, iterable,
                                                epoch, offset):
    monkeypatch.setenv('LOCAL_RANK', '0')
    uninterrupted = make_stream(workers, iterable)
    restored = None
    try:
        consume(uninterrupted, epoch * uninterrupted.length + offset)
        cursor, rng = uninterrupted.state_dict(), _capture_rng()
        assert cursor['epoch'] == epoch and cursor['offset'] == offset
        expected = consume(uninterrupted, 5)
        uninterrupted.close()
        restored = make_stream(workers, iterable)
        restored.load_state_dict(cursor)
        _restore_rng(rng)
        assert consume(restored, 5) == expected
    finally:
        uninterrupted.close()
        if restored is not None:
            restored.close()


def test_nonpersistent_workers_closed_on_rollover_restore_and_close(
        monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '0')
    stream = make_stream(2, iterable=True)
    try:
        assert not stream.loader.persistent_workers
        stream.next()
        first = list(stream.iterator._workers)
        for _ in range(stream.length):
            stream.next()
        assert all(not worker.is_alive() for worker in first)
        second = list(stream.iterator._workers)
        stream.load_state_dict(stream.state_dict())
        assert stream.iterator is None
        assert all(not worker.is_alive() for worker in second)
        stream.next()
        third = list(stream.iterator._workers)
        # Runner's generic loader cleanup cannot own nonpersistent workers.
        assert stream.loader._iterator is None
        mixed = SimpleNamespace(streams=dict(vla=stream))
        MixedTraining.close(mixed, runner=None)
        assert not mixed.streams and stream.iterator is None
        assert all(not worker.is_alive() for worker in third)
        stream.close()  # Idempotent, including runner finally/atexit cleanup.
    finally:
        stream.close()


@pytest.mark.parametrize('replay', [False, True])
def test_stream_closes_iterator_on_fetch_or_replay_failure(replay):
    stream = make_stream(0)
    calls = []

    class BrokenIterator:

        def __next__(self):
            raise StopIteration

        def _shutdown_workers(self):
            calls.append('closed')

    if replay:

        class BrokenLoader:
            dataset = stream.loader.dataset

            def __iter__(self):
                return BrokenIterator()

        stream.loader = BrokenLoader()
        stream.offset = 1
    else:
        stream.iterator = BrokenIterator()
    with pytest.raises(RuntimeError, match='advertised length'):
        stream.next()
    assert calls == ['closed'] and stream.iterator is None
