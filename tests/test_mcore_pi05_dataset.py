from types import SimpleNamespace

import pytest
import torch

from fluxvla.mcore.pi05.dataset import FixedPI0IterableDataset, build_pi0_datasets


def _write_fixed_batch(root, step: int, value: float) -> None:
    torch.save(
        {"actions": torch.tensor([[[value]]], dtype=torch.float32)},
        root / f"step_{step:06d}_rank_00.pt",
    )


def test_fixed_replay_resume_crosses_rollover_without_repeating_tail(tmp_path):
    for step in range(3):
        _write_fixed_batch(tmp_path, step, float(step))

    dataset = FixedPI0IterableDataset(
        str(tmp_path),
        dp_rank=0,
        dp_size=1,
        micro_batch_size=1,
        repeat=True,
        start_iteration=5,
    )

    values = [float(next(dataset)["actions"].item()) for _ in range(5)]
    assert values == [2.0, 0.0, 1.0, 2.0, 0.0]


def test_fixed_replay_resume_rollover_preserves_dp_rank_mapping(tmp_path):
    for step in range(3):
        for rank in range(4):
            torch.save(
                {"actions": torch.tensor([[[10.0 * step + rank]]])},
                tmp_path / f"step_{step:06d}_rank_{rank:02d}.pt",
            )

    rank0 = FixedPI0IterableDataset(
        str(tmp_path),
        dp_rank=0,
        dp_size=2,
        micro_batch_size=1,
        repeat=True,
        start_iteration=2,
    )
    rank1 = FixedPI0IterableDataset(
        str(tmp_path),
        dp_rank=1,
        dp_size=2,
        micro_batch_size=1,
        repeat=True,
        start_iteration=2,
    )

    values0 = [float(next(rank0)["actions"].item()) for _ in range(5)]
    values1 = [float(next(rank1)["actions"].item()) for _ in range(5)]
    assert values0 == [20.0, 22.0, 0.0, 2.0, 10.0]
    assert values1 == [21.0, 23.0, 1.0, 3.0, 11.0]


def test_fixed_replay_rejects_exhausted_non_repeating_resume(tmp_path):
    for step in range(2):
        _write_fixed_batch(tmp_path, step, float(step))

    with pytest.raises(ValueError, match="start iteration 2 is unavailable"):
        FixedPI0IterableDataset(
            str(tmp_path),
            dp_rank=0,
            dp_size=1,
            micro_batch_size=1,
            start_iteration=2,
        )


def test_dataset_builder_uses_checkpoint_iteration_for_replay_resume(tmp_path):
    for step in range(3):
        _write_fixed_batch(tmp_path, step, float(step))
    args = SimpleNamespace(
        pi0_fixed_batch_dir=str(tmp_path),
        pi0_fixed_batch_start_step=None,
        pi0_fixed_batch_repeat=True,
        iteration=4,
        micro_batch_size=1,
        global_batch_size=1,
    )

    train, valid, test = build_pi0_datasets(args)

    assert float(next(train)["actions"].item()) == 1.0
    assert float(next(valid)["actions"].item()) == 0.0
    assert test is None


def test_plain_dataset_resume_converts_optimizer_iteration_to_microbatch(monkeypatch):
    import fluxvla.mcore.pi05.dataset as dataset_module

    constructed = []

    class StubDataset:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(dataset_module, "PI0IterableDataset", StubDataset)
    args = SimpleNamespace(
        pi0_fixed_batch_dir=None,
        pi0_dataset_repo_id="test/repo",
        pi0_reference_dataloader=False,
        pipeline_model_parallel_size=1,
        micro_batch_size=2,
        global_batch_size=8,
        iteration=7,
    )

    dataset_module.build_pi0_datasets(args)

    assert constructed[0]["start_step"] == 28
    assert constructed[1]["start_step"] == 0
