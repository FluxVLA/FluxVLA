"""PI0/PI0.5 dataset provider for Pai-Megatron-Patch.

Wraps LeRobot datasets for Megatron's ``--dataloader-type external`` mode.

Each call to __next__ returns a dict with tensors ready for PI0Model.forward():
    images      : List[Tensor[B, C, H, W]]  one entry per camera
    img_masks   : List[Tensor[B]]            True = present
    tokens      : Tensor[B, L_text]          language token IDs
    text_masks  : Tensor[B, L_text]          True = real token
    states      : Tensor[B, state_dim]       robot state
    actions     : Tensor[B, T, action_dim]   action chunk

For shared_observation mode (multiple offsets):
    states      : Tensor[B, N, state_dim]
    actions     : Tensor[B, N, T, action_dim]
    offset_mask : Tensor[B, N]               True = valid offset

Usage in pretrain_pi0.py::

    from megatron_patch.data.pi0_dataset_provider import build_pi0_datasets

    def train_valid_test_datasets_provider(train_val_test_num_samples):
        return build_pi0_datasets(get_args())
"""

import os
import re
import sys
import gzip
import json
import numpy as np
from contextlib import nullcontext
from typing import Optional, List

import torch
from torch.utils.data import DataLoader as _DL


def _profile_range(name: str):
    """Emit online-input timing only in the rank-zero profiling trace."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if os.environ.get("PI05_PROFILE_MARKERS") == "1" and rank == 0:
        return torch.profiler.record_function(name)
    return nullcontext()


def _alignment_cpu_value(value):
    """Make a batch value safe to persist without losing alignment metadata."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _alignment_cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_alignment_cpu_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _rebuild_compact_pmp_batch(reference_batch, pmp_aux):
    """Materialize the PMP view from the single shared reference batch.

    Compact schema v3 stores preprocessor output once (the format consumed by
    pi05_training_0811).  PMP's image range/layout and action padding are
    deterministic, so retaining a second image copy is unnecessary.
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    model_images = pmp_aux.get("model_images")
    if model_images is not None:
        images = [image.to(dtype=torch.float32) for image in model_images]
        img_masks = [mask.to(dtype=torch.bool) for mask in pmp_aux["img_masks"]]
    else:
        image_keys = pmp_aux.get("image_keys") or [
        key for key in reference_batch if key.startswith("observation.images.")
        ]
        images = []
        img_masks = []
        # Camera order is semantic: the vision encoder receives a fixed sequence of
        # camera slots. The reference preprocessor's insertion order is captured
        # in schema-v3 artifacts, so do not reorder it during reconstruction.
        for key in image_keys:
            image = reference_batch[key].to(dtype=torch.float32)
            if image.ndim == 4 and image.shape[1] == 3:
                image = image.permute(0, 2, 3, 1)
            if image.shape[1:3] != (224, 224):
                image = resize_with_pad_torch(image, 224, 224)
            image = image * 2.0 - 1.0
            if image.ndim == 4:
                image = image.permute(0, 3, 1, 2)
            images.append(image)
            img_masks.append(torch.ones(image.shape[0], dtype=torch.bool))

    tokens = reference_batch.get("observation.language.tokens", reference_batch.get("tokens"))
    text_masks = reference_batch.get(
        "observation.language.attention_mask", reference_batch.get("text_masks")
    )
    actions = reference_batch.get("action", reference_batch.get("actions"))
    if actions.shape[-1] < 32:
        actions = torch.nn.functional.pad(actions, (0, 32 - actions.shape[-1]))
    batch = {
        "images": images,
        "img_masks": img_masks,
        "tokens": tokens,
        "text_masks": text_masks,
        "states": torch.zeros(actions.shape[0], 32, dtype=actions.dtype),
        "actions": actions,
        "noise": pmp_aux["noise"],
        "time": pmp_aux["time"],
    }
    if "subtask_spans" in pmp_aux:
        batch["subtask_spans"] = pmp_aux["subtask_spans"]
    return batch


class _RankBatchSampler:
    """Select the same per-rank batches as Accelerate's BatchSamplerShard."""

    def __init__(self, batch_sampler, rank: int, world_size: int):
        self.batch_sampler = batch_sampler
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for batch_idx, batch in enumerate(self.batch_sampler):
            if batch_idx % self.world_size == self.rank:
                yield batch

    def __len__(self):
        return (len(self.batch_sampler) + self.world_size - 1 - self.rank) // self.world_size


class FixedPI0IterableDataset:
    """Replay post-preprocessing LeRobot batches for strict numerical alignment."""

    _FILE_RE = re.compile(r"step_(\d+)_rank_(\d+)\.pt(?:\.gz)?$")

    def __init__(
        self,
        root: str,
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        is_valid: bool = False,
        repeat: bool = False,
        start_step: int = 0,
        start_iteration: int | None = None,
    ):
        from pathlib import Path

        paths = sorted(
            list(Path(root).glob("step_*_rank_*.pt"))
            + list(Path(root).glob("step_*_rank_*.pt.gz"))
        )
        self.files = {}
        for path in paths:
            match = self._FILE_RE.match(path.name)
            if match:
                self.files[(int(match.group(1)), int(match.group(2)))] = path
        if not self.files:
            raise FileNotFoundError(f"No fixed PI0 batches under {root}")

        self.steps = sorted({step for step, _ in self.files})
        source_ranks = sorted({rank for _, rank in self.files})
        if source_ranks != list(range(len(source_ranks))):
            raise ValueError(f"Fixed PI0 source ranks must be contiguous from zero: {source_ranks}")
        self.source_dp_size = len(source_ranks)
        if self.source_dp_size % dp_size:
            raise ValueError(
                f"Fixed source DP {self.source_dp_size} is not divisible by runtime DP {dp_size}"
            )
        expected = {(step, rank) for step in self.steps for rank in source_ranks}
        missing = sorted(expected - self.files.keys())
        if missing:
            raise ValueError(f"Fixed PI0 batches are incomplete; first missing entries: {missing[:8]}")

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.micro_batch_size = micro_batch_size
        self.is_valid = is_valid
        self.repeat = repeat
        if start_iteration is not None:
            if start_iteration < 0:
                raise ValueError("Fixed PI0 start iteration must be non-negative")
            if not repeat and start_iteration >= len(self.steps):
                raise ValueError(
                    f"Fixed PI0 start iteration {start_iteration} is unavailable; "
                    f"the replay contains {len(self.steps)} steps"
                )
            start_position = start_iteration % len(self.steps)
            self.start_step = self.steps[start_position]
        else:
            if start_step not in self.steps:
                raise ValueError(
                    f"Fixed PI0 start step {start_step} is unavailable; "
                    f"available range is [{self.steps[0]}, {self.steps[-1]}]"
                )
            start_position = self.steps.index(start_step)
            self.start_step = start_step

        first_batch = self._load(self.files[(self.steps[0], 0)])
        self.source_batch_size = first_batch["actions"].shape[0]
        if self.micro_batch_size <= self.source_batch_size:
            if self.source_batch_size % self.micro_batch_size:
                raise ValueError(
                    f"Fixed batch size {self.source_batch_size} is not divisible by "
                    f"MBS {self.micro_batch_size}"
                )
            self.chunks_per_source_batch = self.source_batch_size // self.micro_batch_size
            self.source_batches_per_microbatch = 1
            total_microbatches = self.source_dp_size * self.chunks_per_source_batch
        else:
            if self.micro_batch_size % self.source_batch_size:
                raise ValueError(
                    f"MBS {self.micro_batch_size} is not divisible by fixed batch size "
                    f"{self.source_batch_size}"
                )
            self.chunks_per_source_batch = 1
            self.source_batches_per_microbatch = (
                self.micro_batch_size // self.source_batch_size
            )
            if self.source_dp_size % self.source_batches_per_microbatch:
                raise ValueError(
                    f"Fixed source DP {self.source_dp_size} cannot form MBS "
                    f"{self.micro_batch_size} from batches of {self.source_batch_size}"
                )
            total_microbatches = self.source_dp_size // self.source_batches_per_microbatch
        if total_microbatches % self.dp_size:
            raise ValueError(
                f"Fixed source microbatches {total_microbatches} are not divisible by "
                f"runtime DP {self.dp_size}"
            )
        self.calls_per_step = total_microbatches // self.dp_size
        self.start_call_index = start_position * self.calls_per_step
        self.call_index = self.start_call_index

    @staticmethod
    def _load(path):
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as stream:
                value = torch.load(stream, map_location="cpu", weights_only=True)
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)

        # Alignment captures retain RNG and raw-input evidence around the
        # model-ready batch. Accept that envelope for compute-only replay while
        # preserving the original direct-batch format used by normal dumps.
        if isinstance(value, dict) and value.get("schema_version") == 3:
            value = _rebuild_compact_pmp_batch(value["reference_batch"], value["pmp_aux"])
        elif isinstance(value, dict) and "actions" not in value and isinstance(value.get("batch"), dict):
            value = value["batch"]
        return value

    def __iter__(self):
        return self

    def __next__(self):
        step_index = self.call_index // self.calls_per_step
        if step_index >= len(self.steps):
            if not (self.is_valid or self.repeat):
                raise RuntimeError(
                    f"Fixed PI0 batches exhausted after {len(self.steps)} steps; "
                    "dump more steps or reduce --train-iters."
                )
            # A resumed run first consumes the remainder of its current replay
            # cycle. Later cycles must restart at the beginning, not at the
            # resume offset, or the tail would repeat forever.
            self.call_index = self.start_call_index if self.is_valid else 0
            step_index = self.call_index // self.calls_per_step

        microbatch_index = self.call_index % self.calls_per_step
        step = self.steps[step_index]
        self.call_index += 1

        if self.source_batches_per_microbatch > 1:
            first_group = microbatch_index * self.source_batches_per_microbatch
            source_ranks = [
                self.dp_rank + (first_group + part) * self.dp_size
                for part in range(self.source_batches_per_microbatch)
            ]
            source_batches = [self._load(self.files[(step, rank)]) for rank in source_ranks]

            def merge(values):
                if isinstance(values[0], list):
                    return [merge([value[index] for value in values]) for index in range(len(values[0]))]
                return torch.cat(values, dim=0)

            return {
                key: merge([batch[key] for batch in source_batches])
                for key in source_batches[0]
            }

        logical_chunk = self.dp_rank + microbatch_index * self.dp_size
        source_rank = logical_chunk % self.source_dp_size
        chunk_index = logical_chunk // self.source_dp_size
        batch = self._load(self.files[(step, source_rank)])

        def take(value, start, end):
            if isinstance(value, list):
                # Camera tensors are batched along dim 0, while optional
                # per-sample annotations such as ``subtask_spans`` contain
                # ``None`` for samples without a caption target.
                return [None if item is None else item[start:end] for item in value]
            return value[start:end]

        start = chunk_index * self.micro_batch_size
        end = start + self.micro_batch_size
        return {key: take(value, start, end) for key, value in batch.items()}


class ReferencePI05IterableDataset:
    """Run the pi05_training input pipeline online with reference rank sharding."""

    def __init__(
        self,
        dataset_repo_id: str,
        dataset_root: str,
        policy_path: str,
        lerobot_src: str,
        micro_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        num_workers: int,
        prefetch_factor: int,
        image_transforms_enable: bool = True,
        tokenizer_path: str | None = None,
        subtask_captions_path: str | None = None,
        subtask_lang: str = "en",
        video_backend: str = "torchcodec",
        tolerance_s: float = 0.01,
        is_valid: bool = False,
        start_step: int = 0,
    ):
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.policy_path = policy_path
        self.lerobot_src = lerobot_src
        self.tokenizer_path = tokenizer_path
        self.subtask_captions_path = subtask_captions_path
        self.subtask_lang = subtask_lang
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.image_transforms_enable = image_transforms_enable
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.is_valid = is_valid
        if start_step < 0:
            raise ValueError("Reference PI0.5 start step must be non-negative")
        self.start_step = start_step
        self._loader = None
        self._loader_iter = None
        self._preprocessor = None
        self._image_keys = None
        self._noise_generator = None
        self._time_generator = None
        self._subtask_captions = None
        self._subtask_indices_by_index = None
        self._dump_step = start_step

    @property
    def _is_pi05se_policy(self) -> bool:
        """Identify a PI05se processor before loading its registered steps."""
        config_path = os.path.join(self.policy_path, "config.json")
        try:
            with open(config_path) as stream:
                return json.load(stream).get("type") == "pi05se"
        except (OSError, json.JSONDecodeError):
            return False

    def _activate_reference_lerobot(self):
        if not self.lerobot_src or not os.path.isdir(self.lerobot_src):
            raise FileNotFoundError(
                f"pi05_training source directory not found: {self.lerobot_src}"
            )
        if self.lerobot_src not in sys.path:
            sys.path.insert(0, self.lerobot_src)
        loaded = sys.modules.get("lerobot")
        loaded_path = os.path.realpath(getattr(loaded, "__file__", "")) if loaded else ""
        expected = os.path.realpath(self.lerobot_src)
        if loaded_path and not loaded_path.startswith(expected + os.sep):
            raise RuntimeError(
                "--pi0-reference-dataloader requires pi05_training/src to be imported "
                f"before another lerobot installation; loaded {loaded_path}, expected {expected}"
            )

    def _configure_pi05se_captions(self) -> list[str]:
        """Apply the caption catalog to the live PI05se tokenizer step.

        The processor is constructed by the training rank after distributed
        initialization.  Reapplying the cached catalog immediately before use
        prevents a reconstructed processor pipeline from silently reverting to
        its serialized ``None`` default.
        """
        if not (self._is_pi05se_policy and self.subtask_captions_path):
            return []
        if self._subtask_captions is None:
            import pandas as pd

            captions = pd.read_parquet(self.subtask_captions_path).sort_values("subtask_index")
            column = "subtask_en" if self.subtask_lang == "en" else "subtask"
            if captions["subtask_index"].tolist() != list(range(len(captions))):
                raise ValueError("subtask_index must be contiguous from zero")
            self._subtask_captions = captions[column].astype(str).tolist()

        configured = False
        for step in self._preprocessor.steps:
            if hasattr(step, "subtask_captions"):
                step.subtask_captions = self._subtask_captions
                configured = True
        if not configured:
            raise RuntimeError("PI05se preprocessor has no subtask_captions step")
        return self._subtask_captions

    def _ensure_loader(self):
        if self._loader is not None:
            return

        # Diagnostic-only compatibility path. The default remains strict v3;
        # callers must opt in while validating whether the v3 reader can consume
        # an existing v2.1 directory without conversion.
        if os.environ.get("PI05_ALLOW_LEGACY_V21") == "1":
            try:
                import lerobot.datasets.dataset_metadata as _metadata
                import lerobot.datasets.utils as _utils
                _metadata.check_version_compatibility = lambda *args, **kwargs: None
                _utils.check_version_compatibility = lambda *args, **kwargs: None
            except ImportError:
                pass
        self._activate_reference_lerobot()

        from accelerate.data_loader import BatchSamplerShard
        from lerobot.configs import FeatureType, NormalizationMode
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.sampler import EpisodeAwareSampler
        if self._is_pi05se_policy:
            from lerobot.policies.pi05_subtask_recovery.configuration_pi05se import PI05seConfig as PolicyConfig
            # Register PI05se's state tokenizer step before deserializing its processor pipeline.
            from lerobot.policies.pi05_subtask_recovery import processor_pi05se  # noqa: F401
        else:
            from lerobot.policies.pi05.configuration_pi05 import PI05Config as PolicyConfig
            # Import registers the base PI0.5-specific state tokenizer processor.
            from lerobot.policies.pi05 import processor_pi05  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.transforms import ImageTransforms, ImageTransformsConfig
        from lerobot.utils.constants import ACTION
        from lerobot.utils.feature_utils import dataset_to_policy_features

        metadata_probe = LeRobotDataset(
            self.dataset_repo_id,
            root=self.dataset_root,
            delta_timestamps=None,
            image_transforms=None,
            video_backend=self.video_backend,
            return_uint8=True,
            tolerance_s=self.tolerance_s,
        )
        fps = metadata_probe.meta.fps
        delta_timestamps = {ACTION: [index / fps for index in range(50)]}
        image_transforms = (
            ImageTransforms(ImageTransformsConfig(enable=True))
            if self.image_transforms_enable and not self.is_valid
            else None
        )
        dataset = LeRobotDataset(
            self.dataset_repo_id,
            root=self.dataset_root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=self.video_backend,
            return_uint8=True,
            tolerance_s=self.tolerance_s,
        )
        if self._is_pi05se_policy and self.subtask_captions_path:
            # The action delta-timestamp view above trims complementary scalar
            # columns from ``dataset.hf_dataset``.  Read the immutable label
            # column from metadata_probe, which addresses the identical global
            # frame rows without that projection.
            try:
                self._subtask_indices_by_index = torch.as_tensor(
                    metadata_probe.hf_dataset["subtask_index"], dtype=torch.long
                )
            except (AttributeError, KeyError, ValueError):
                # Some LeRobot releases project even the metadata probe to
                # runtime features.  Read only the two immutable source columns
                # instead of depending on that version-specific projection.
                from pathlib import Path
                import pyarrow.parquet as pq

                data_path = os.path.join(self.dataset_root, "data")
                data_files = sorted(
                    str(path)
                    for path in Path(data_path).rglob("*")
                    if path.is_file() and path.suffix == ".parquet"
                )
                if not data_files:
                    raise RuntimeError(f"No source parquet files found under {data_path}")
                # ParquetFile is deliberately used instead of pandas.read_parquet:
                # some pyarrow releases resolve a file path to its parent
                # directory dataset and merge a sibling ``.parquet.bak`` schema.
                label_frames = [
                    pq.ParquetFile(path).read(columns=["index", "subtask_index"]).to_pandas()
                    for path in data_files
                ]
                import pandas as pd
                label_table = pd.concat(label_frames, ignore_index=True)
                frame_ids = torch.as_tensor(label_table["index"].to_numpy(), dtype=torch.long)
                labels = torch.as_tensor(label_table["subtask_index"].to_numpy(), dtype=torch.long)
                if frame_ids.numel() == 0 or frame_ids.min() < 0:
                    raise RuntimeError("Invalid frame indices in PI05se source parquet")
                lookup = torch.full((int(frame_ids.max()) + 1,), -1, dtype=torch.long)
                lookup[frame_ids] = labels
                self._subtask_indices_by_index = lookup

        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=0,
            shuffle=not self.is_valid,
            seed=self.seed,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if self.start_step:
            from lerobot.datasets.sampler import compute_sampler_state

            sampler.load_state_dict(
                compute_sampler_state(
                    self.start_step,
                    len(sampler),
                    self.micro_batch_size,
                    self.dp_size,
                )
            )
        global_batches = torch.utils.data.BatchSampler(
            sampler, batch_size=self.micro_batch_size, drop_last=False
        )
        rank_batches = BatchSamplerShard(
            global_batches,
            num_processes=self.dp_size,
            process_index=self.dp_rank,
            split_batches=False,
            even_batches=True,
        )
        data_loader_generator = None
        data_loader_seed = os.environ.get("PI05_DATALOADER_SEED")
        if data_loader_seed is not None:
            data_loader_generator = torch.Generator()
            data_loader_generator.manual_seed(int(data_loader_seed))
        self._loader = _DL(
            dataset,
            batch_sampler=rank_batches,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            generator=data_loader_generator,
        )
        self._loader_iter = iter(self._loader)

        config = PolicyConfig(device="cuda")
        features = dataset_to_policy_features(dataset.meta.features)
        config.output_features = {
            key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
        }
        config.input_features = {
            key: feature for key, feature in features.items() if key not in config.output_features
        }
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
        self._image_keys = list(config.image_features)
        overrides = {
            "device_processor": {"device": f"cuda:{torch.cuda.current_device()}"},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        }
        if self.tokenizer_path:
            overrides["tokenizer_processor"] = {"tokenizer_name": self.tokenizer_path}
        self._preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
        )
        self._configure_pi05se_captions()

    def __iter__(self):
        return self

    def _next_raw_batch(self):
        try:
            return next(self._loader_iter)
        except StopIteration:
            # lerobot.utils.utils.cycle recreates the DataLoader iterator at an
            # epoch boundary. This advances EpisodeAwareSampler to the next
            # deterministic (seed, epoch) permutation while persistent workers
            # retain their image-transform RNG streams.
            self._loader_iter = iter(self._loader)
            try:
                return next(self._loader_iter)
            except StopIteration as error:
                raise RuntimeError("Reference PI0.5 DataLoader produced no batches") from error

    def __next__(self):
        self._ensure_loader()
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        with _profile_range("pi05_dataloader_wait"):
            raw_batch = self._next_raw_batch()
        if (
            self._is_pi05se_policy
            and self.subtask_captions_path
            and "subtask_index" not in raw_batch
        ):
            frame_indices = raw_batch.get("index")
            if not torch.is_tensor(frame_indices) or self._subtask_indices_by_index is None:
                raise RuntimeError(
                    "PI05se online batch lost subtask_index and has no tensor index "
                    "for dataset-backed reconstruction"
                )
            lookup_indices = frame_indices.detach().cpu().long()
            raw_batch["subtask_index"] = self._subtask_indices_by_index[lookup_indices].clone()
        input_subtask_indices = raw_batch.get("subtask_index")
        # Keep the diagnostic artifact at the same point as lerobot_train:
        # before uint8 images are converted to the preprocessor's float range.
        # The tensor references remain valid because the conversion below
        # replaces each mapping value rather than mutating it in place.
        raw_images_before_preprocess = {
            key: raw_batch[key] for key in self._image_keys if key in raw_batch
        }
        with _profile_range("pi05_cpu_preprocess"):
            for key in self._image_keys:
                if key not in raw_batch:
                    continue
                image = raw_batch[key]
                if image.dtype == torch.uint8:
                    raw_batch[key] = image.to(dtype=torch.float32) / 255.0
                elif os.environ.get("PI05_REFERENCE_UINT8_IMAGE_PATH") == "1":
                    # The reference trainer receives codec frames as uint8 and
                    # normalizes them here.  Some torchcodec versions expose
                    # the same pixels as float32 already divided by 255; use
                    # the reference's integer grid before normalization so a
                    # sub-ULP decode difference cannot be amplified by SigLIP.
                    raw_batch[key] = (
                        torch.round(image * 255.0)
                        .clamp_(0, 255)
                        .to(torch.uint8)
                        .to(torch.float32)
                        / 255.0
                    )
            self._configure_pi05se_captions()
            batch = self._preprocessor(raw_batch)
        preprocessor_images = {key: batch[key] for key in self._image_keys if key in batch}

        if self._is_pi05se_policy:
            from lerobot.policies.pi05_subtask_recovery.modeling_pi05se import resize_with_pad_torch, sample_beta
            from megatron_patch.model.pi0.subtask import derive_subtask_spans
        else:
            from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch, sample_beta
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        with _profile_range("pi05_tensor_preprocess"):
            images = []
            img_masks = []
            for key in self._image_keys:
                image = batch[key].to(dtype=torch.float32)
                channels_first = image.shape[1] == 3
                if channels_first:
                    image = image.permute(0, 2, 3, 1)
                if image.shape[1:3] != (224, 224):
                    image = resize_with_pad_torch(image, 224, 224)
                image = image * 2.0 - 1.0
                if channels_first:
                    image = image.permute(0, 3, 1, 2)
                images.append(image)
                img_masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=image.device))

        with _profile_range("pi05_flow_matching_inputs"):
            actions = batch[ACTION]
            if actions.shape[-1] < 32:
                actions = torch.nn.functional.pad(actions, (0, 32 - actions.shape[-1]))
            if self._noise_generator is None:
                flow_seed = int(os.environ.get("PI05_ALIGNMENT_FLOW_SEED", self.seed))
                self._noise_generator = torch.Generator(device=actions.device).manual_seed(flow_seed)
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=actions.shape,
                dtype=torch.float32,
                device=actions.device,
                generator=self._noise_generator,
            )
        if self._time_generator is None:
            self._time_generator = torch.Generator(device="cpu")
            alignment_seed = os.environ.get("PI05_ALIGNMENT_FLOW_SEED")
            if alignment_seed is None:
                from megatron_patch.model.pi0.reference_pi05_rng_state import (
                    reference_pi05_cpu_rng_state,
                )

                self._time_generator.set_state(reference_pi05_cpu_rng_state())
            else:
                self._time_generator.manual_seed(int(alignment_seed) + 1)
        global_cpu_state = torch.get_rng_state()
        try:
            torch.set_rng_state(self._time_generator.get_state())
            time = sample_beta(1.5, 1.0, actions.shape[0], actions.device) * 0.999 + 0.001
            self._time_generator.set_state(torch.get_rng_state())
        finally:
            torch.set_rng_state(global_cpu_state)
        result = {
            "images": images,
            "img_masks": img_masks,
            "tokens": batch[OBS_LANGUAGE_TOKENS],
            "text_masks": batch[OBS_LANGUAGE_ATTENTION_MASK],
            "states": torch.zeros(actions.shape[0], 32, device=actions.device),
            "actions": actions,
            "noise": noise,
            "time": time.to(dtype=torch.float32),
        }
        if self._is_pi05se_policy:
            raw_tasks = batch.get("raw_task")
            state_strs = batch.get("state_str")
            subtasks = batch.get("subtask_str")
            if raw_tasks is None or state_strs is None or subtasks is None:
                raise RuntimeError(
                    "PI05se preprocessor did not preserve raw_task/state_str/subtask_str; "
                    "cannot construct caption CE spans."
                )
            result["subtask_spans"] = derive_subtask_spans(
                self._preprocessor.steps[-2].input_tokenizer,
                raw_tasks,
                state_strs,
                subtasks,
                result["tokens"],
                result["text_masks"],
            )
            if self.subtask_captions_path and not any(span is not None for span in result["subtask_spans"]):
                raw_indices = raw_batch.get("subtask_index")
                if torch.is_tensor(raw_indices):
                    raw_indices = raw_indices.detach().cpu().tolist()
                if torch.is_tensor(input_subtask_indices):
                    input_subtask_indices = input_subtask_indices.detach().cpu().tolist()
                raise RuntimeError(
                    "PI05se caption CE was requested but no subtask span was produced; "
                    "refusing to silently train flow-only "
                    f"(captions={len(self._subtask_captions or [])}, "
                    f"input_subtask_index={input_subtask_indices}, "
                    f"raw_subtask_index={raw_indices}, processed_subtasks={subtasks})"
                )
        dump_dir = os.environ.get("PI0_ONLINE_BATCH_DUMP_DIR")
        dump_steps = int(os.environ.get("PI0_ONLINE_BATCH_DUMP_STEPS", "0"))
        if (
            not self.is_valid
            and dump_dir
            and (dump_steps <= 0 or self._dump_step < dump_steps)
        ):
            import gzip
            from pathlib import Path

            path = Path(dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            # Store one shared reference batch.  The PMP model-ready view is
            # reconstructed by FixedPI0IterableDataset, avoiding duplicate image
            # tensors while retaining exact stochastic inputs and CE/KI spans.
            raw_metadata = {
                key: value for key, value in raw_batch.items()
                if key in {"index", "frame_index", "episode_index", "task_index", "subtask_index",
                           "task", "raw_task", "state_str", "subtask_str"}
            }
            artifact = {
                "schema_version": 3,
                "step": self._dump_step,
                "dp_rank": self.dp_rank,
                "cpu_rng_before": cpu_rng_before,
                "cuda_rng_before": cuda_rng_before,
                "raw_metadata": _alignment_cpu_value(raw_metadata),
                "processed_metadata": _alignment_cpu_value({
                    key: batch.get(key) for key in ("raw_task", "state_str", "subtask_str", "task")
                }),
                "reference_batch": _alignment_cpu_value(batch),
                "pmp_aux": {
                    "image_keys": list(self._image_keys),
                    # Persist the exact tensors consumed by PI0Model.  Keeping
                    # this view avoids a second resize/range conversion during
                    # replay (older schema-v3 artifacts simply omit the fields).
                    "model_images": _alignment_cpu_value(result["images"]),
                    "img_masks": _alignment_cpu_value(result["img_masks"]),
                    "noise": _alignment_cpu_value(noise),
                    "time": _alignment_cpu_value(time),
                    "subtask_spans": _alignment_cpu_value(result.get("subtask_spans")),
                },
            }
            output = path / f"step_{self._dump_step:06d}_rank_{self.dp_rank:02d}.pt.gz"
            with gzip.open(output, "wb", compresslevel=1) as stream:
                torch.save(artifact, stream)
        self._dump_step += 1
        return result


class TopologyInvariantReferencePI05Dataset:
    """Map physical DP/GA layouts onto a fixed set of logical data streams.

    The logical topology is defined by ``GBS / MBS``. Runs with the same GBS
    and MBS therefore consume the same logical rank streams even when physical
    DP and gradient accumulation differ. Each stream owns its DataLoader
    workers and random generators, preserving augmentation, noise, and time.
    """

    def __init__(
        self,
        *,
        physical_dp_rank: int,
        physical_dp_size: int,
        logical_dp_size: int,
        stream_factory=ReferencePI05IterableDataset,
        **stream_kwargs,
    ):
        if logical_dp_size < physical_dp_size:
            raise ValueError(
                f"Logical DP {logical_dp_size} must be at least physical DP {physical_dp_size}"
            )
        if logical_dp_size % physical_dp_size:
            raise ValueError(
                f"Logical DP {logical_dp_size} must be divisible by physical DP "
                f"{physical_dp_size}"
            )
        self.physical_dp_rank = physical_dp_rank
        self.physical_dp_size = physical_dp_size
        self.logical_dp_size = logical_dp_size
        self.streams_per_rank = logical_dp_size // physical_dp_size
        self.logical_ranks = [
            physical_dp_rank + slot * physical_dp_size
            for slot in range(self.streams_per_rank)
        ]
        self.streams = [
            stream_factory(
                **stream_kwargs,
                dp_rank=logical_rank,
                dp_size=logical_dp_size,
            )
            for logical_rank in self.logical_ranks
        ]
        self.call_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        slot = self.call_index % self.streams_per_rank
        self.call_index += 1
        return next(self.streams[slot])


# ---------------------------------------------------------------------------
# PI0IterableDataset
# ---------------------------------------------------------------------------

class PI0IterableDataset:
    """Self-iterating PI0 dataset for Megatron external dataloader mode.

    Compatible with ``--dataloader-type external``: Megatron wraps this object
    directly in a RerunDataIterator and calls next() on it to fetch batches.

    Args:
        dataset_repo_id: LeRobot dataset repo ID (e.g. "lerobot/aloha_mobile_cabinet")
        dataset_root:    Local cache directory for the dataset
        micro_batch_size: Samples per micro-batch
        dp_rank:         Data parallel rank (for data sharding)
        dp_size:         Data parallel world size
        chunk_size:      Number of action tokens to predict
        max_delay_steps: Maximum temporal delay augmentation steps
        shared_observation: Use shared observation multi-offset training
        num_workers:     DataLoader workers
        seed:            Random seed
        image_keys:      List of image keys (e.g. ["observation.images.top"])
        tokenizer_path:  Path to PaliGemma tokenizer (for text tokenization)
        task_key:        Key for task description text in batch
        is_valid:        True = validation split
    """

    def __init__(
        self,
        dataset_repo_id: str,
        micro_batch_size: int,
        dp_rank: int,
        dp_size: int,
        chunk_size: int = 50,
        max_delay_steps: int = 4,
        shared_observation: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        episodes: Optional[List[int]] = None,
        dataset_root: Optional[str] = None,
        image_keys: Optional[List[str]] = None,
        tokenizer_path: Optional[str] = None,
        task_key: str = "task",
        is_valid: bool = False,
        normalize: bool = True,       # apply MEAN_STD norm to state/action (matches pi05_training)
        dataset_fps: Optional[float] = None,  # override FPS; None = read from info.json
        subtask_captions_path: Optional[str] = None,
        subtask_lang: str = "zh",
        start_step: int = 0,
    ):
        self.dataset_repo_id = dataset_repo_id
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.chunk_size = chunk_size
        self.max_delay_steps = max_delay_steps
        self.shared_observation = shared_observation
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_root = dataset_root
        self.image_keys = image_keys or ["observation.images.top"]
        self.tokenizer_path = tokenizer_path
        self.task_key = task_key
        self.is_valid = is_valid
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.episodes = episodes
        self.normalize = normalize
        self.dataset_fps = dataset_fps
        self.subtask_captions_path = subtask_captions_path
        self.subtask_lang = subtask_lang
        if start_step < 0:
            raise ValueError("PI0 start step must be non-negative")
        self.start_step = start_step
        self._subtask_captions = None

        self._loader = None
        self._loader_iter = None
        self._tokenizer = None
        self._stats = None   # loaded lazily in _ensure_loader

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        if self.tokenizer_path is None:
            return
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except Exception:
            pass

    def _ensure_loader(self):
        if self._loader is not None:
            return

        self._ensure_tokenizer()
        if self.subtask_captions_path:
            import pandas as pd

            table = pd.read_parquet(self.subtask_captions_path).sort_values("subtask_index")
            expected = list(range(len(table)))
            if table["subtask_index"].tolist() != expected:
                raise ValueError("subtask_index must be contiguous from zero")
            column = "subtask_en" if self.subtask_lang == "en" else "subtask"
            if column not in table.columns:
                raise ValueError(f"missing caption column {column!r} in {list(table.columns)}")
            self._subtask_captions = table[column].astype(str).tolist()

        from .temporal_dataset import (
            PI0SharedObservationDataset,
            PI0TemporalDataset,
            shared_observation_collate_fn,
        )
        from lerobot.datasets.sampler import EpisodeAwareSampler
        from lerobot.transforms import ImageTransforms, ImageTransformsConfig

        # Resolve dataset FPS: explicit override > info.json > fallback 30Hz
        fps = self.dataset_fps
        if fps is None and self.dataset_root is not None:
            import json, os
            info_path = os.path.join(self.dataset_root, "meta", "info.json")
            if os.path.isfile(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                fps = float(info.get("fps", 30))
        if fps is None:
            fps = 30.0  # safe default — warn rather than silently use wrong FPS

        # delta_timestamps: predict chunk_size future actions at the correct fps
        delta_timestamps = {
            "action": [i / fps for i in range(self.chunk_size)],
        }

        # Load normalization stats (MEAN_STD for state/action, matching pi05_training)
        if self.normalize and self.dataset_root is not None:
            import json, os
            stats_path = os.path.join(self.dataset_root, "meta", "stats.json")
            if os.path.isfile(stats_path):
                with open(stats_path) as f:
                    raw_stats = json.load(f)
                self._stats = {}
                for key in ("observation.state", "action"):
                    if key in raw_stats:
                        self._stats[key] = {
                            "mean": torch.tensor(raw_stats[key]["mean"], dtype=torch.float32),
                            "std":  torch.tensor(raw_stats[key]["std"],  dtype=torch.float32),
                        }

        if self.shared_observation and self.max_delay_steps > 0:
            dataset = PI0SharedObservationDataset(
                repo_id=self.dataset_repo_id,
                root=self.dataset_root,
                episodes=self.episodes,
                delta_timestamps=delta_timestamps,
                max_delay_steps=self.max_delay_steps,
            )
            collate_fn = shared_observation_collate_fn
        else:
            transform_config = ImageTransformsConfig(enable=not self.is_valid)
            dataset = PI0TemporalDataset(
                repo_id=self.dataset_repo_id,
                root=self.dataset_root,
                episodes=self.episodes,
                delta_timestamps=delta_timestamps,
                max_delay_steps=self.max_delay_steps if not self.is_valid else 0,
                image_transforms=ImageTransforms(transform_config) if transform_config.enable else None,
                video_backend="torchcodec",
            )
            collate_fn = None

        # lerobot 0.4.x: `episodes` param only controls download/check logic,
        # NOT the hf_dataset content. Manually restrict to requested episode frames.
        if self.episodes is not None:
            episode_set = set(self.episodes)
            valid_indices = []
            for ep_idx in sorted(episode_set):
                if ep_idx < len(dataset.meta.episodes):
                    ep_meta = dataset.meta.episodes[ep_idx]
                    from_idx = ep_meta['dataset_from_index']
                    to_idx = ep_meta['dataset_to_index']
                    valid_indices.extend(range(from_idx, to_idx))
            dataset = torch.utils.data.Subset(dataset, valid_indices)

        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            shuffle=not self.is_valid,
            seed=self.seed,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if self.start_step:
            from lerobot.datasets.sampler import compute_sampler_state

            sampler.load_state_dict(
                compute_sampler_state(
                    self.start_step,
                    len(sampler),
                    self.micro_batch_size,
                    self.dp_size,
                )
            )
        global_batches = torch.utils.data.BatchSampler(
            sampler, batch_size=self.micro_batch_size, drop_last=False
        )
        rank_batches = _RankBatchSampler(global_batches, self.dp_rank, self.dp_size)
        self._loader = _DL(
            dataset,
            batch_sampler=rank_batches,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        self._loader_iter = iter(self._loader)

    def __iter__(self):
        return self

    def __next__(self):
        self._ensure_loader()
        try:
            raw_batch = next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            raw_batch = next(self._loader_iter)
        return self._process_batch(raw_batch)

    @staticmethod
    def _resize_and_normalize_images(img: "torch.Tensor", target: int = 224) -> "torch.Tensor":
        """Resize [B, C, H, W] float32 [0,1] images to [B, C, target, target] in [-1,1].

        Uses aspect-ratio-preserving resize followed by center-pad, matching the
        resize_with_pad behaviour used in run_joint_loss_compare.py.
        """
        import torch.nn.functional as _F
        B, C, H, W = img.shape
        if H == target and W == target:
            # Already correct size — just renormalize to [-1,1]
            return img * 2.0 - 1.0
        scale = target / max(H, W)
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
        img = _F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)
        pad_H = target - new_H
        pad_W = target - new_W
        pad_top    = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left   = pad_W // 2
        pad_right  = pad_W - pad_left
        img = _F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return img * 2.0 - 1.0

    def _process_batch(self, raw_batch: dict) -> dict:
        """Convert a raw LeRobot batch into PI0Model input format."""
        from lerobot.utils.constants import ACTION, OBS_STATE  # type: ignore
        import torch.nn.functional as _F

        # ---- Images ----
        images = []
        img_masks = []
        for key in self.image_keys:
            if key in raw_batch:
                img = raw_batch[key]  # [B, C, H, W] or [B, N, C, H, W]
                if img.ndim == 5:
                    # Shared observation: take first offset's image (shared prefix)
                    img = img[:, 0]
                img = img.float()
                # Resize + normalize to match SigLIP/PaliGemma input format:
                #   lerobot returns [0,1]; model expects 224×224 in [-1,1]
                img = self._resize_and_normalize_images(img, target=224)
                images.append(img)
                img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))
            else:
                # Camera absent: placeholder zeros
                B = next(iter(raw_batch.values())).shape[0]
                images.append(torch.zeros(B, 3, 224, 224))
                img_masks.append(torch.zeros(B, dtype=torch.bool))

        # ---- State and actions ----
        states = raw_batch[OBS_STATE].float()   # [B, D_s] or [B, N, D_s]
        actions = raw_batch[ACTION].float()     # [B, T, D_a] or [B, N, T, D_a]

        # Apply MEAN_STD normalization (matches pi05_training MEAN_STD policy)
        if self._stats:
            if "observation.state" in self._stats:
                s = self._stats["observation.state"]
                mean = s["mean"].to(states.device)
                std  = s["std"].to(states.device)
                # Broadcast over leading batch/offset dims; only normalize actual DOF
                d = mean.shape[0]
                states[..., :d] = (states[..., :d] - mean) / std.clamp(min=1e-6)
            if "action" in self._stats:
                s = self._stats["action"]
                mean = s["mean"].to(actions.device)
                std  = s["std"].to(actions.device)
                d = mean.shape[0]
                actions[..., :d] = (actions[..., :d] - mean) / std.clamp(min=1e-6)

        # PI0.5 tokenizes normalized state into the language prompt, then does
        # not consume state as a separate model input.
        B = images[0].shape[0] if images else states.shape[0]
        task_texts = raw_batch.get(self.task_key, [""] * B)
        if isinstance(task_texts, torch.Tensor):
            task_texts = [str(t.item()) for t in task_texts]
        state_np = states.cpu().numpy()
        discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        from megatron_patch.model.pi0.subtask import build_low_level_task_prompt, derive_subtask_spans

        if self._subtask_captions is not None and states.ndim != 2:
            raise NotImplementedError("PI05se P0 does not support shared-observation input")
        subtask_indices = raw_batch.get("subtask_index")
        subtasks = []
        prompts = []
        state_strs = []
        for row, (task, state_values) in enumerate(zip(task_texts, discretized, strict=True)):
            state_str = " ".join(map(str, state_values))
            subtask = None
            if self._subtask_captions is not None and subtask_indices is not None:
                index = int(subtask_indices[row])
                if index >= len(self._subtask_captions):
                    raise ValueError(f"subtask_index {index} out of range")
                if index >= 0:
                    subtask = self._subtask_captions[index]
            subtasks.append(subtask)
            state_strs.append(state_str)
            prompts.append(build_low_level_task_prompt(str(task), state_str, subtask))

        if self._tokenizer is not None:
            enc = self._tokenizer(
                prompts,
                padding="max_length",
                padding_side="right",
                max_length=200,
                truncation=True,
                return_tensors="pt",
            )
            tokens = enc["input_ids"]
            text_masks = enc["attention_mask"].bool()
        else:
            tokens = torch.zeros(B, 1, dtype=torch.long)
            text_masks = torch.ones(B, 1, dtype=torch.bool)

        # Pad feature dims to model max (matching run_joint_loss_compare.py behavior)
        import torch.nn.functional as _F
        if states.shape[-1] < self.max_state_dim:
            states = _F.pad(states, (0, self.max_state_dim - states.shape[-1]))
        if actions.shape[-1] < self.max_action_dim:
            actions = _F.pad(actions, (0, self.max_action_dim - actions.shape[-1]))

        result = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "text_masks": text_masks,
            "states": torch.zeros_like(states),
            "actions": actions,
        }
        if self._subtask_captions is not None:
            if self._tokenizer is None:
                raise RuntimeError("PI05se captions require --pi0-tokenizer-path")
            result["subtask_spans"] = derive_subtask_spans(
                self._tokenizer, [str(task) for task in task_texts], state_strs,
                subtasks, tokens, text_masks,
            )

        if "offset_mask" in raw_batch:
            result["offset_mask"] = raw_batch["offset_mask"]
        if "action_is_pad" in raw_batch:
            result["action_is_pad"] = raw_batch["action_is_pad"]
        if self.shared_observation:
            result["shared_obs"] = True

        return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_pi0_datasets(args):
    """Build train and validation PI0IterableDataset objects.

    ``args`` must have the following attributes (added by get_patch_args):
        --pi0-dataset-repo-id  : LeRobot dataset repo ID
        --pi0-dataset-root     : local cache root (optional)
        --pi0-chunk-size       : action chunk length
        --pi0-max-delay-steps  : temporal delay augmentation
        --pi0-shared-observation : shared obs mode
        --pi0-image-keys       : camera key list (space-separated)
        --pi0-pretrained-path  : HF model path (for tokenizer)
        --micro-batch-size     : Megatron MBS
    """
    try:
        from megatron.core import parallel_state
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
    except Exception:
        dp_rank, dp_size = 0, 1
    if dp_size <= 0:
        dp_rank, dp_size = 0, 1

    fixed_batch_dir = getattr(args, "pi0_fixed_batch_dir", None)
    if fixed_batch_dir:
        fixed_start_step = getattr(args, "pi0_fixed_batch_start_step", None)
        fixed_start_iteration = None
        if fixed_start_step is None:
            fixed_start_iteration = getattr(args, "iteration", 0)
            fixed_start_step = 0
        train_ds = FixedPI0IterableDataset(
            fixed_batch_dir,
            dp_rank,
            dp_size,
            args.micro_batch_size,
            is_valid=False,
            repeat=getattr(args, "pi0_fixed_batch_repeat", False),
            start_step=fixed_start_step,
            start_iteration=fixed_start_iteration,
        )
        valid_ds = FixedPI0IterableDataset(
            fixed_batch_dir,
            dp_rank,
            dp_size,
            args.micro_batch_size,
            is_valid=True,
            start_step=fixed_start_step,
        )
        expected_gbs = train_ds.source_dp_size * train_ds.source_batch_size
        if args.global_batch_size != expected_gbs:
            raise ValueError(
                f"Strict fixed replay requires GBS={expected_gbs} "
                f"({train_ds.source_dp_size} source ranks x batch {train_ds.source_batch_size}), "
                f"got {args.global_batch_size}"
            )
        return train_ds, valid_ds, None

    repo_id = getattr(args, "pi0_dataset_repo_id", None)
    if repo_id is None:
        raise ValueError(
            "PI0 requires --pi0-dataset-repo-id pointing to a LeRobot dataset repo."
        )

    if os.environ.get("PI05_FLUXVLA_PARQUET") == "1":
        return _build_fluxvla_parquet_datasets(args, dp_rank, dp_size)

    if getattr(args, "pi0_reference_dataloader", False):
        reference_src = getattr(args, "pi0_reference_lerobot_src", None)
        reference_policy = getattr(args, "pi0_reference_policy_path", None)
        dataset_root = getattr(args, "pi0_dataset_root", None)
        if not reference_src or not reference_policy or not dataset_root:
            raise ValueError(
                "--pi0-reference-dataloader requires --pi0-reference-lerobot-src, "
                "--pi0-reference-policy-path, and --pi0-dataset-root"
            )
        physical_batch_size = args.micro_batch_size * dp_size
        if args.global_batch_size % physical_batch_size:
            raise ValueError(
                "Topology-invariant PI0.5 loading requires GBS divisible by MBS x DP: "
                f"{args.global_batch_size} % ({args.micro_batch_size} x {dp_size}) != 0."
            )
        logical_dp_size = args.global_batch_size // args.micro_batch_size
        if logical_dp_size % dp_size:
            raise ValueError(
                f"Logical DP GBS/MBS={logical_dp_size} must be divisible by physical DP {dp_size}"
            )
        reference_kwargs = dict(
            dataset_repo_id=repo_id,
            dataset_root=dataset_root,
            policy_path=reference_policy,
            lerobot_src=reference_src,
            tokenizer_path=getattr(args, "pi0_tokenizer_path", None),
            subtask_captions_path=getattr(args, "pi0_subtask_captions_path", None),
            subtask_lang=getattr(args, "pi0_subtask_lang", "en"),
            micro_batch_size=args.micro_batch_size,
            seed=getattr(args, "seed", 1234),
            num_workers=getattr(args, "num_workers", 0),
            prefetch_factor=getattr(args, "pi0_reference_prefetch_factor", 4),
            image_transforms_enable=bool(
                getattr(args, "pi0_reference_image_transforms_enable", 1)
            ),
        )
        topology_kwargs = dict(
            physical_dp_rank=dp_rank,
            physical_dp_size=dp_size,
            logical_dp_size=logical_dp_size,
        )
        train_ds = TopologyInvariantReferencePI05Dataset(
            **topology_kwargs,
            **reference_kwargs,
            is_valid=False,
            start_step=getattr(args, "iteration", 0),
        )
        valid_ds = TopologyInvariantReferencePI05Dataset(
            **topology_kwargs,
            **{**reference_kwargs, "seed": reference_kwargs["seed"] + 1},
            is_valid=True,
            start_step=0,
        )
        return train_ds, valid_ds, None

    pp_size = getattr(args, "pipeline_model_parallel_size", 1)
    actual_mbs = args.micro_batch_size if pp_size <= 1 else args.micro_batch_size - 1
    physical_micro_batch = actual_mbs * dp_size
    if args.global_batch_size % physical_micro_batch:
        raise ValueError(
            "PI0 loading requires GBS divisible by effective MBS x DP: "
            f"{args.global_batch_size} % ({actual_mbs} x {dp_size}) != 0."
        )
    microbatches_per_iteration = args.global_batch_size // physical_micro_batch

    image_keys_raw = getattr(args, "pi0_image_keys", None)
    if isinstance(image_keys_raw, str):
        image_keys = image_keys_raw.split()
    elif image_keys_raw is not None:
        image_keys = list(image_keys_raw)
    else:
        image_keys = ["observation.images.top"]

    common_kwargs = dict(
        dataset_repo_id=repo_id,
        micro_batch_size=actual_mbs,
        dp_rank=dp_rank,
        dp_size=dp_size,
        chunk_size=getattr(args, "pi0_chunk_size", 50),
        max_delay_steps=getattr(args, "pi0_max_delay_steps", 4),
        shared_observation=getattr(args, "pi0_shared_observation", False),
        num_workers=getattr(args, "num_workers", 0),
        seed=getattr(args, "seed", 1234),
        dataset_root=getattr(args, "pi0_dataset_root", None),
        image_keys=image_keys,
        tokenizer_path=(
            getattr(args, "pi0_tokenizer_path", None)
            or getattr(args, "pi0_pretrained_path", None)
        ),
        max_state_dim=getattr(args, "pi0_state_dim", 32),
        max_action_dim=getattr(args, "pi0_action_dim", 32),
        episodes=getattr(args, "pi0_dataset_episodes", None),
        normalize=getattr(args, "pi0_normalize", True),
        dataset_fps=getattr(args, "pi0_dataset_fps", None),
        subtask_captions_path=getattr(args, "pi0_subtask_captions_path", None),
        subtask_lang=getattr(args, "pi0_subtask_lang", "zh"),
        start_step=getattr(args, "iteration", 0) * microbatches_per_iteration,
    )

    train_ds = PI0IterableDataset(**common_kwargs, is_valid=False)
    valid_ds = PI0IterableDataset(
        **{**common_kwargs, "seed": common_kwargs["seed"] + 1, "start_step": 0},
        is_valid=True,
    )

    return train_ds, valid_ds, None


def _build_fluxvla_parquet_datasets(args, dp_rank: int, dp_size: int):
    """Build the same v2.1 ParquetDataset path used by FluxVLA FSDP.

    This adapter intentionally lives at the native boundary: LeRobot's v3
    dataset reader is not used, while FluxVLA's transforms/statistics remain
    the single preprocessing owner for both backends.
    """
    from torch.utils.data import DataLoader
    from fluxvla.engines import build_dataset_from_cfg
    from fluxvla.collators.dict_collator import DictCollator

    root = getattr(args, "pi0_dataset_root", None)
    if not root:
        raise ValueError("--pi0-dataset-root is required for PI05_FLUXVLA_PARQUET=1")
    dataset_cfg = dict(
        type="DistributedRepeatingDataset",
        statistic_keys=["observation.state", "timestamp", "action"],
        name_mappings={"observation.state": ["proprio"], "action": ["action"]},
        statistic_name="libero_10_no_noops",
        datasets=dict(
            type="ParquetDataset",
            data_root_path=root,
            transforms=[
                dict(type="ProcessParquetInputs", parquet_keys=[
                    "observation.state", "timestamp", "actions", "info", "stats", "action_masks"
                ], video_backend="torchcodec", video_keys=[
                    "observation.images.image", "observation.images.wrist_image"
                ], name_mappings={"observation.state": ["states"], "actions": ["actions"]}),
                dict(type="ParquetPrompter", use_conversation=False),
                dict(type="ProcessPrompts", tokenizer=dict(type="PaligemmaTokenizer")),
                dict(type="ResizeImages", height=224, width=224),
                dict(type="NormalizeImages", means=[[123.515625, 116.04492188, 103.59375]] * 2,
                     stds=[[58.27148438, 57.02636719, 57.27539062]] * 2),
                dict(type="NormalizeStatesAndActions", action_dim=32, state_dim=32,
                     state_key="proprio", action_key="action", norm_type="mean_std",
                     output_dtype="float32"),
            ],
            action_window_size=10, action_key="action", use_delta=False,
            statistic_name="libero_10_no_noops", window_start_idx=0,
        ),
        shuffle=True, seed=getattr(args, "seed", 1234),
    )
    dataset = build_dataset_from_cfg(dataset_cfg)
    keys = ["states", "images", "img_masks", "lang_tokens", "lang_masks", "actions", "action_masks", "timestamp"]
    collator = DictCollator(keys=keys, meta_keys=["task_description", "prompt", "info", "stats"])
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, num_workers=0,
                        collate_fn=collator, pin_memory=True)

    class _LoaderDataset:
        def __init__(self):
            self._iterator = None
            flow_seed = os.environ.get("PI05_ALIGNMENT_FLOW_SEED")
            self._noise_generator = (
                torch.Generator(device="cpu").manual_seed(int(flow_seed))
                if flow_seed is not None else None
            )
            self._time_generator = (
                torch.Generator(device="cpu").manual_seed(int(flow_seed) + 1)
                if flow_seed is not None else None
            )
        def __iter__(self):
            return self
        def __next__(self):
            if self._iterator is None:
                self._iterator = iter(loader)
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._iterator = iter(loader)
                batch = next(self._iterator)
            # Keep stochastic flow-matching inputs outside the model so both
            # backends consume the same tensors during numerical alignment.
            if self._noise_generator is not None and "noise" not in batch:
                actions = batch["actions"]
                batch["noise"] = torch.normal(
                    mean=0.0, std=1.0, size=actions.shape, dtype=torch.float32,
                    generator=self._noise_generator,
                )
                from fluxvla.engines.utils.model_utils import sample_beta
                global_state = torch.get_rng_state()
                try:
                    torch.set_rng_state(self._time_generator.get_state())
                    batch["time"] = sample_beta(
                        1.5, 1.0, actions.shape[0], device="cpu",
                        generator=self._time_generator,
                    ) * 0.999 + 0.001
                finally:
                    torch.set_rng_state(global_state)
            return batch
        def __len__(self):
            return max(1, len(dataset) // max(1, args.micro_batch_size))

    train = _LoaderDataset()
    valid = _LoaderDataset()
    return train, valid, None
