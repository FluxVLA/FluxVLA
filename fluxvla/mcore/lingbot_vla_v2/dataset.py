"""External MCore datasets for LingBot online training and replay diagnosis."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _stack(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([sample[key] for sample in samples])
        for key in samples[0]
    }


def _local_batch_offset(args, *, dp_size: int) -> int:
    consumed_samples = int(getattr(args, "consumed_train_samples", 0) or 0)
    if consumed_samples:
        divisor = dp_size * int(args.micro_batch_size)
        if consumed_samples % divisor:
            raise ValueError(
                "LingBot consumed train samples must align to DP x micro batch size: "
                f"samples={consumed_samples}, divisor={divisor}"
            )
        return consumed_samples // divisor
    local_batch_size = dp_size * int(args.micro_batch_size)
    if int(args.global_batch_size) % local_batch_size:
        raise ValueError(
            "LingBot global batch size must align to DP x micro batch size: "
            f"GB={args.global_batch_size}, divisor={local_batch_size}"
        )
    accumulation = int(args.global_batch_size) // local_batch_size
    return int(getattr(args, "iteration", 0) or 0) * accumulation


def _initial_epoch_and_offset(
    dataset,
    *,
    dp_size: int,
    micro_batch_size: int,
    start_batch: int,
):
    samples_per_rank = (len(dataset) + dp_size - 1) // dp_size
    batches_per_epoch = samples_per_rank // micro_batch_size
    if batches_per_epoch <= 0:
        raise ValueError(
            "LingBot dataset has no complete local microbatch: "
            f"samples={len(dataset)}, DP={dp_size}, MB={micro_batch_size}"
        )
    return divmod(start_batch, batches_per_epoch)


class LingBotOnlineIterableDataset:
    """Yield online-processed LeRobot batches in deterministic DP order."""

    def __init__(
        self,
        dataset,
        *,
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        sampler_seed: int = 0,
        start_iteration: int = 0,
        repeat: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.micro_batch_size = int(micro_batch_size)
        self.sampler_seed = int(sampler_seed)
        self.repeat = bool(repeat)
        self.num_workers = int(num_workers)
        if self.micro_batch_size <= 0:
            raise ValueError("LingBot MCore micro batch size must be positive")
        self.epoch, self.batch_offset = _initial_epoch_and_offset(
            dataset,
            dp_size=self.dp_size,
            micro_batch_size=self.micro_batch_size,
            start_batch=int(start_iteration),
        )
        if self.num_workers < 0:
            raise ValueError("LingBot online data worker count cannot be negative")
        self._iterator = self._iterate()

    def __iter__(self) -> LingBotOnlineIterableDataset:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        return next(self._iterator)

    def _iterate(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.dp_size,
                rank=self.dp_rank,
                shuffle=True,
                seed=self.sampler_seed,
                drop_last=False,
            )
            sampler.set_epoch(self.epoch)
            loader = DataLoader(
                self.dataset,
                batch_size=self.micro_batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            )
            for batch_index, batch in enumerate(loader):
                if batch_index >= self.batch_offset:
                    yield batch
            self.batch_offset = 0
            self.epoch += 1
            if not self.repeat:
                return


class LingBotReplayIterableDataset:
    """Yield model-ready batches in FluxVLA DistributedSampler order."""

    def __init__(
        self,
        manifest_path: str,
        *,
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        sampler_seed: int = 0,
        fixed_time: float = 0.5,
        replay_order: str = "official_step_rank",
        start_iteration: int = 0,
        repeat: bool = True,
    ) -> None:
        from fluxvla.datasets.lingbot_teacher_cache import LingBotTeacherCacheDataset

        self.dataset = LingBotTeacherCacheDataset(
            manifest_path=manifest_path,
            cache_in_memory=False,
            validate_targets=True,
            fixed_time=fixed_time,
            replay_order=replay_order,
        )
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.micro_batch_size = int(micro_batch_size)
        self.sampler_seed = int(sampler_seed)
        self.repeat = bool(repeat)
        if self.micro_batch_size <= 0:
            raise ValueError("LingBot MCore micro batch size must be positive")
        self.epoch, self.batch_offset = _initial_epoch_and_offset(
            self.dataset,
            dp_size=self.dp_size,
            micro_batch_size=self.micro_batch_size,
            start_batch=int(start_iteration),
        )
        self._iterator = self._iterate()

    def __iter__(self) -> LingBotReplayIterableDataset:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        return next(self._iterator)

    def _iterate(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.dp_size,
                rank=self.dp_rank,
                shuffle=not bool(
                    getattr(self.dataset, "deterministic_order", False)
                ),
                seed=self.sampler_seed,
                drop_last=False,
            )
            sampler.set_epoch(self.epoch)
            indices = list(iter(sampler))
            batches = [
                indices[start : start + self.micro_batch_size]
                for start in range(0, len(indices), self.micro_batch_size)
                if len(indices[start : start + self.micro_batch_size])
                == self.micro_batch_size
            ]
            for indices_for_batch in batches[self.batch_offset :]:
                yield _stack([self.dataset[index] for index in indices_for_batch])
            self.batch_offset = 0
            self.epoch += 1
            if not self.repeat:
                return


def build_lingbot_datasets(args):
    """Build train/validation iterables on MCore's data-parallel topology."""
    from megatron.core import parallel_state

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    start_batch = _local_batch_offset(args, dp_size=dp_size)
    mode = getattr(args, "lingbot_data_mode", "online")
    common = {
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "micro_batch_size": args.micro_batch_size,
        "sampler_seed": getattr(args, "lingbot_sampler_seed", 0),
    }
    if mode == "online":
        from fluxvla.datasets.lingbot_vla_v2 import LingBotOnlineDataset

        data_path = getattr(args, "lingbot_data_path", None)
        norm_stats = getattr(args, "lingbot_norm_stats", None)
        if not data_path:
            raise ValueError("--lingbot-data-path is required in online mode")
        if not norm_stats:
            raise ValueError("--lingbot-norm-stats is required in online mode")
        dataset = LingBotOnlineDataset(
            data_path=data_path,
            norm_stats_path=norm_stats,
            tokenizer_path=args.lingbot_tokenizer_path,
            action_horizon=getattr(args, "lingbot_action_horizon", 50),
            image_size=getattr(args, "lingbot_image_size", 256),
            tokenizer_max_length=getattr(args, "lingbot_tokenizer_max_length", 72),
            use_future_images=getattr(args, "lingbot_use_future_images", True),
            return_teacher_images=False,
            flow_rng_seed=getattr(args, "lingbot_flow_rng_seed", None),
        )
        online_common = {
            **common,
            "dataset": dataset,
            "num_workers": getattr(args, "lingbot_num_workers", 0),
        }
        train = LingBotOnlineIterableDataset(
            **online_common,
            start_iteration=start_batch,
            repeat=True,
        )
        valid = LingBotOnlineIterableDataset(**online_common, repeat=True)
    elif mode == "replay":
        manifest = getattr(args, "lingbot_manifest", None)
        if not manifest:
            raise ValueError("--lingbot-manifest is required in replay mode")
        replay_common = {
            **common,
            "manifest_path": manifest,
            "fixed_time": getattr(args, "lingbot_fixed_time", 0.5),
            "replay_order": getattr(args, "lingbot_replay_order", "official_step_rank"),
        }
        train = LingBotReplayIterableDataset(
            **replay_common,
            start_iteration=start_batch,
            repeat=True,
        )
        valid = LingBotReplayIterableDataset(**replay_common, repeat=True)
    else:
        raise ValueError(f"Unsupported LingBot data mode: {mode!r}")
    return train, valid, None


__all__ = [
    "LingBotOnlineIterableDataset",
    "LingBotReplayIterableDataset",
    "build_lingbot_datasets",
]
