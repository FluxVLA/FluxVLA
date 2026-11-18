"""Map-style LingBot adapter composing the existing ParquetDataset reader.

LeRobot v2/v3 metadata readers share action-window assembly. Videos are decoded
online, including v3 episode offsets into shared files. No sampler lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.v2 import Resize

from fluxvla.datasets.utils.video_decode import build_lerobot_video_path, decode_video_frames
from fluxvla.engines import DATASETS
from fluxvla.transforms.lingbot_online import LingBotSampleTransform, ROBOTWIN_CAMERA_KEYS


@DATASETS.register_module()
class LingBotParquetDataset(Dataset):
    """Reuse Parquet window assembly and shared LingBot preprocessing.

    ``video_backend`` and ``resize_dtype`` are part of the input contract:
    different decoders/dtype boundaries are not assumed pixel equivalent.
    """

    seed_workers_by_dp_rank = True

    def __init__(
        self,
        data_path: str,
        norm_stats_path: str,
        tokenizer_path: str,
        *,
        video_backend: str = "torchcodec",
        resize_dtype: str = "float32",
        action_horizon: int = 50,
        image_size: int = 256,
        tokenizer_max_length: int = 72,
        use_future_images: bool = True,
        return_teacher_images: bool = False,
        model_dtype: torch.dtype = torch.bfloat16,
        processor=None,
        flow_rng_seed: int | None = None,
    ) -> None:
        from fluxvla.datasets.parquet_dataset import ParquetDataset
        from fluxvla.datasets.parquet_dataset_v3 import ParquetDatasetV3

        if not data_path:
            raise ValueError("Set dataset.data_path for LingBot Parquet input")
        root = Path(data_path).expanduser().resolve()
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        version = str(info.get("codebase_version", ""))
        if version.startswith("v2."):
            reader_type = ParquetDataset
        elif version.startswith("v3."):
            reader_type = ParquetDatasetV3
        else:
            raise ValueError(f"Unsupported LeRobot codebase_version: {version!r}")
        if action_horizon < 1 or image_size < 1 or float(info["fps"]) <= 0:
            raise ValueError("action_horizon, image_size and fps must be positive")
        if video_backend not in ("pyav", "torchcodec", "video_reader"):
            raise ValueError(f"Unsupported explicit video backend: {video_backend!r}")
        self.video_backend = video_backend
        if resize_dtype not in ("uint8", "float32"):
            raise ValueError("resize_dtype must be uint8 or float32")
        self.resize_dtype = resize_dtype
        self.resize = Resize((image_size, image_size))
        self.use_future_images = bool(use_future_images)
        self.transform = LingBotSampleTransform(
            norm_stats_path=norm_stats_path,
            tokenizer_path=tokenizer_path,
            action_horizon=action_horizon,
            tokenizer_max_length=tokenizer_max_length,
            return_teacher_images=return_teacher_images,
            model_dtype=model_dtype,
            processor=processor,
            flow_rng_seed=flow_rng_seed,
        )
        self.reader = reader_type(
            data_root_path=str(root), transforms=[], action_key="action",
            action_window_size=action_horizon, window_start_idx=0,
            frame_window_size=1, expose_index=True, task_filter_policy="preserve",
        )
        # LeRobot rounds delta timestamps to row offsets, clamps indices at
        # episode boundaries, then reads the stored timestamps. Do not replace
        # this with current_ts + offset: irregular timestamps can differ.
        self._episode_ends = {}
        previous = None
        timing = self.reader.dataset.select_columns(["episode_index", "timestamp"]).to_dict()
        for index, episode in enumerate(timing["episode_index"]):
            episode = int(episode)
            if episode != previous and episode in self._episode_ends:
                raise ValueError(f"Episode rows must be contiguous: episode={episode}")
            previous = episode
            self._episode_ends[episode] = index
        self._timestamps = torch.tensor(timing["timestamp"], dtype=torch.float32)
        self.action_horizon = int(action_horizon)
        self._camera_storage = {
            key: info.get("features", {}).get(key, {}).get("dtype", "video")
            for key in ROBOTWIN_CAMERA_KEYS
        }
        for key, storage in self._camera_storage.items():
            if storage not in ("video", "image"):
                raise ValueError(f"Unsupported camera storage: {key}={storage!r}")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        data = self.reader.__getitem__(index, {self.reader.statistic_name: {}})
        raw = dict(data)
        raw["action"] = data["actions"]
        raw["action_is_pad"] = ~torch.as_tensor(data["action_masks"], dtype=torch.bool)
        raw["task"] = data["task_description"]
        episode = int(data["episode_index"])
        episode_meta = self.reader.get_episode_metadata(0, episode)
        timestamps = [float(self._timestamps[index])]
        future = None
        if self.use_future_images:
            future_index = min(index + self.action_horizon - 1, self._episode_ends[episode])
            timestamps.append(float(self._timestamps[future_index]))
            if "image" in self._camera_storage.values():
                future = data if future_index == index else self.reader.dataset[future_index]
        unique = sorted(set(timestamps))
        for camera in ROBOTWIN_CAMERA_KEYS:
            if self._camera_storage[camera] == "image":
                # HF Image features decode embedded bytes lazily into PIL;
                # only the requested current/future rows are materialized.
                current = pil_to_tensor(data[camera].convert("RGB"))
                frames = [current]
                if future is not None:
                    frames.append(current if future is data else pil_to_tensor(future[camera].convert("RGB")))
            else:
                offset_key = f"videos/{camera}/from_timestamp"
                if data["info"]["codebase_version"].startswith("v3.") and offset_key not in episode_meta:
                    raise ValueError(f"Missing episode video offset: {offset_key}, episode={episode}")
                path = build_lerobot_video_path(data["data_root"], data["info"], episode_meta, episode, camera)
                offset = float(episode_meta.get(offset_key, 0.0))
                decoded = decode_video_frames(path, [ts + offset for ts in unique], 1e-4,
                                              backend=self.video_backend)
                frames = [decoded[unique.index(ts)] for ts in timestamps]
            frames = torch.stack(frames)
            if self.resize_dtype == "float32":
                frames = frames.float().div(255)
            raw[camera] = self.resize(frames)
        return self.transform(raw)
