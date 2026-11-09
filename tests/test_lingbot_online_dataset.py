from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return f"user:{messages[0]['content']}"

    def __call__(self, prompts, **kwargs):
        assert prompts == ["user:pick object"]
        assert kwargs["max_length"] == 6
        return {
            "input_ids": torch.tensor([[10, 11, 12, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
        }


class _FakeImageProcessor:
    def __call__(self, image):
        assert image.dtype == torch.uint8
        value = float(image.flatten()[0])
        return {
            "pixel_values": torch.full((4, 3), value),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()


class _FakeRawDataset:
    def __init__(self, count=1):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        state = torch.arange(14, dtype=torch.float32) / 13
        actions = state.unsqueeze(0).repeat(2, 1)
        sample = {
            "observation.state": state,
            "action": actions,
            "action_is_pad": torch.tensor([False, index % 2 == 1]),
            "task": "pick object",
        }
        for camera_index, camera in enumerate(
            (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )
        ):
            sample[camera] = torch.stack(
                [
                    torch.full((3, 2, 2), (camera_index + 1) / 10),
                    torch.full((3, 2, 2), (camera_index + 4) / 10),
                ]
            )
        return sample


@pytest.fixture()
def norm_stats(tmp_path):
    stats = {
        "norm_stats": {
            "observation.state.arm.position": {"q01": [0] * 12, "q99": [1] * 12},
            "observation.state.effector.position": {"q01": [0] * 2, "q99": [1] * 2},
            "action.arm.position": {"q01": [0] * 12, "q99": [1] * 12},
            "action.effector.position": {"q01": [0] * 2, "q99": [1] * 2},
        }
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def test_online_dataset_builds_model_ready_robotwin_sample(norm_stats):
    from fluxvla.datasets.lingbot_vla_v2 import LingBotOnlineDataset

    dataset = LingBotOnlineDataset(
        data_path="unused",
        norm_stats_path=str(norm_stats),
        tokenizer_path="unused",
        action_horizon=2,
        tokenizer_max_length=6,
        dataset=_FakeRawDataset(),
        processor=_FakeProcessor(),
        return_teacher_images=True,
        model_dtype=torch.float32,
    )
    sample = dataset[0]

    assert sample["state"].shape == (55,)
    assert sample["actions"].shape == (2, 55)
    assert sample["joint_mask"].shape == (2, 55)
    assert sample["state_joint_mask"].sum().item() == 14
    assert sample["action_joint_mask"].sum().item() == 14
    assert sample["state_joint_mask"][:12].all()
    assert sample["state_joint_mask"][12:28].sum().item() == 0
    assert sample["state_joint_mask"][28:30].all()
    assert sample["state_joint_mask"][30:].sum().item() == 0
    expected_arm = torch.arange(14, dtype=torch.float32)[
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    ] / 13
    expected_arm = expected_arm / (1 + 1.0e-6) * 2 - 1
    torch.testing.assert_close(sample["state"][:12], expected_arm)
    assert sample["images"].shape == (3, 4, 3)
    assert sample["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
    assert sample["images"][:, 0, 0].tolist() == [26.0, 51.0, 76.0]
    assert sample["pil_images"].shape == (3, 3, 2, 2)
    assert sample["future_pil_images"][:, 0, 0, 0].tolist() == [102, 128, 153]
    assert sample["lang_tokens"].tolist() == [10, 11, 12, 0, 0, 0]
    assert sample["lang_masks"].tolist() == [True, True, True, False, False, False]
    assert sample["sample_index"].item() == 0


def test_online_dataset_defaults_model_inputs_to_bfloat16(norm_stats):
    from fluxvla.datasets.lingbot_vla_v2 import LingBotOnlineDataset

    sample = LingBotOnlineDataset(
        data_path="unused",
        norm_stats_path=str(norm_stats),
        tokenizer_path="unused",
        action_horizon=2,
        tokenizer_max_length=6,
        dataset=_FakeRawDataset(),
        processor=_FakeProcessor(),
    )[0]

    assert sample["state"].dtype == torch.bfloat16
    assert sample["actions"].dtype == torch.bfloat16
    assert sample["images"].dtype == torch.bfloat16


def test_flow_targets_are_stable_per_sample_index(norm_stats):
    from fluxvla.datasets.lingbot_vla_v2 import LingBotOnlineDataset

    dataset = LingBotOnlineDataset(
        data_path="unused",
        norm_stats_path=str(norm_stats),
        tokenizer_path="unused",
        action_horizon=2,
        tokenizer_max_length=6,
        dataset=_FakeRawDataset(count=3),
        processor=_FakeProcessor(),
        model_dtype=torch.float32,
        flow_rng_seed=42,
    )
    first = dataset[2]
    second = dataset[2]
    assert first["noise"].shape == first["actions"].shape
    torch.testing.assert_close(first["noise"], second["noise"])
    torch.testing.assert_close(first["time"], second["time"])
    assert not torch.equal(first["noise"], dataset[1]["noise"])


def test_flow_rng_is_independent_of_batch_order():
    from fluxvla.datasets.lingbot_vla_v2.flow_rng import sample_flow_by_index

    indices = torch.tensor([11, 4])
    noise_a, time_a = sample_flow_by_index(indices, (2, 3, 5), seed=7)
    noise_b, time_b = sample_flow_by_index(indices.flip(0), (2, 3, 5), seed=7)
    torch.testing.assert_close(noise_a[0], noise_b[1])
    torch.testing.assert_close(time_a[0], time_b[1])


def test_online_iterable_uses_disjoint_deterministic_dp_samples():
    from fluxvla.mcore.lingbot_vla_v2.dataset import LingBotOnlineIterableDataset

    class Indices(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, index):
            return {"sample_index": torch.tensor(index)}

    actual = []
    for rank in range(2):
        iterator = LingBotOnlineIterableDataset(
            Indices(),
            dp_rank=rank,
            dp_size=2,
            micro_batch_size=1,
            sampler_seed=7,
            repeat=False,
        )
        actual.append([int(batch["sample_index"]) for batch in iterator])
    assert set(actual[0]).isdisjoint(actual[1])
    assert sorted(actual[0] + actual[1]) == list(range(8))


def test_online_iterable_resume_offset_rolls_over_epochs():
    from fluxvla.mcore.lingbot_vla_v2.dataset import LingBotOnlineIterableDataset

    class Indices(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"sample_index": torch.tensor(index)}

    resumed = LingBotOnlineIterableDataset(
        Indices(),
        dp_rank=0,
        dp_size=1,
        micro_batch_size=1,
        sampler_seed=7,
        start_iteration=5,
        repeat=False,
    )
    expected_sampler = torch.utils.data.DistributedSampler(
        list(range(4)), num_replicas=1, rank=0, shuffle=True, seed=7
    )
    expected_sampler.set_epoch(1)
    expected = list(expected_sampler)[1:]
    assert [int(batch["sample_index"]) for batch in resumed] == expected


def test_local_resume_batch_offset_accounts_for_gradient_accumulation():
    from fluxvla.mcore.lingbot_vla_v2.dataset import _local_batch_offset

    args = SimpleNamespace(
        consumed_train_samples=0,
        iteration=3,
        global_batch_size=64,
        micro_batch_size=4,
    )
    assert _local_batch_offset(args, dp_size=4) == 12
    args.consumed_train_samples = 128
    assert _local_batch_offset(args, dp_size=4) == 8


def test_lingbot_data_mode_arguments_default_to_online():
    import argparse

    from fluxvla.mcore.lingbot_vla_v2.provider import get_lingbot_args

    parser = argparse.ArgumentParser()
    get_lingbot_args(parser)
    args = parser.parse_args(
        [
            "--lingbot-native-config",
            "config",
            "--lingbot-checkpoint",
            "checkpoint",
            "--lingbot-tokenizer-path",
            "tokenizer",
        ]
    )
    assert args.lingbot_data_mode == "online"
    assert args.lingbot_manifest is None
    assert args.lingbot_online_teacher_mode == "disabled"


def test_online_flow_only_explicitly_disables_alignment_heads():
    from fluxvla.mcore.lingbot_vla_v2.provider import _model_config_overrides

    args = SimpleNamespace(
        lingbot_attention_implementation="eager",
        lingbot_vit_attn_implementation="flash_attention_2",
        lingbot_moe_implementation="auto",
        lingbot_data_mode="online",
        lingbot_online_teacher_mode="disabled",
    )
    overrides = _model_config_overrides(args, requested_sp=False, requested_ep=False)
    assert overrides["align_params"] == {}
