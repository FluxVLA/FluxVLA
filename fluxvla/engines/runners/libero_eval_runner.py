# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import tqdm
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.eval_utils import (get_libero_dummy_action,
                                              get_libero_env,
                                              save_rollout_video)
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS
from .base_eval_runner import BaseEvalRunner

overwatch = initialize_overwatch(__name__)


def _get_libero_benchmark():
    try:
        from libero.libero import benchmark
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'LIBERO is required for simulation evaluation. Install it with '
            '`bash scripts/install_env.sh sim-only` or '
            '`bash scripts/install_env.sh full`.') from exc
    return benchmark


@RUNNERS.register_module()
class LiberoEvalRunner(BaseEvalRunner):
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.
    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        dataset (Dict): Configuration for the dataset to be used in evaluation.
        denormalize_action (Dict): Configuration for denormalizing actions.
        eval_chunk_size (int): Size of the chunks for evaluation.
            Default is 1.
        resize_size (int): Size to which images will be resized.
            Default is 224.
        num_trials_per_task (int): Number of trials per task in the evaluation.
            Default is 50.
        task_ids (list): Optional task id or task id list to evaluate.
            When ``None``, all tasks in the suite are evaluated.
        num_steps_wait (int): Number of steps to wait before
            starting evaluation.
            Default is 10.
        num_inference_steps (int): Number of denoising/inference steps passed
            to ``predict_action``. When ``None`` (default) the model's own
            default is used, preserving prior behavior.
        max_steps (int): Override for the per-suite maximum rollout length.
            When ``None`` (default) the built-in per-suite values are used.
        inference_seed (int): Seed forwarded to ``predict_action`` on every
            prediction (e.g. to match a source eval that re-seeds the action
            noise each call). When ``None`` (default) no seed is forwarded.
        allowed_missing_key_prefixes (tuple): Checkpoint keys with these
            prefixes may be missing when loading with ``strict=False``.
            Defaults to empty, which keeps strict missing-key validation.
        model_build_device (str): Optional device passed to the eval model
            config before construction. Defaults to ``None``, preserving the
            config's existing model construction behavior.
        model_build_dtype (str): Optional dtype passed to the eval model config
            before construction. Defaults to ``None``, preserving the config's
            existing model construction behavior.
        eval_shard_strategy (str): Episode assignment strategy. ``task``
            keeps all trials for a task on the same rank, reusing the LIBERO
            env. ``episode`` preserves the old round-robin episode
            sharding.
        preprocess_every_step (bool): Whether to build the next model batch
            after every simulator step. Chunked eval only preprocesses
            when a new action chunk is needed.
        save_rollout_videos (bool): Whether to save every rollout replay.
        save_failed_rollout_videos (bool): Whether to keep replay frames and
            save only failed rollouts when ``save_rollout_videos`` is false.
        save_multi_view_rollout_videos (bool): Whether replay videos include
            all configured image views. When false, only the first replay view
            is saved.
        rollout_dir (str): Optional directory for rollout videos. Relative
            paths are resolved under the active video root.
        run_id_suffix (str): Optional suffix appended to the eval run id.
            Useful when launching several single-task eval workers at once.
        result_output_dir (str): Optional manager output root. When set,
            per-worker eval artifacts are written under its ``eval_runs``
            subdirectory, and manager-compatible per-task result files are
            mirrored to
            ``<result_output_dir>/<suite>/gpu{gpu_id}_task{task_id}_results.json``.
        result_gpu_id (int): GPU id written to mirrored result filenames.
        mixed_precision_dtype (str): Data type for mixed precision training.
            Default is 'bf16'.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
            Default is True.
    """

    @staticmethod
    def _inject_checkpoint_tokenizer(dataset: Dict, ckpt_path: str) -> None:
        model_path = Path(ckpt_path).resolve().parent.parent
        tokenizer_path = model_path / 'tokenizer'
        if not tokenizer_path.is_dir():
            return

        for transform in dataset.get('transforms', []):
            tokenizer = transform.get('tokenizer')
            if isinstance(tokenizer, dict):
                tokenizer['model_path'] = tokenizer_path.as_posix()

    @staticmethod
    def _build_global_episodes(num_tasks: int,
                               num_trials_per_task: int,
                               task_ids=None) -> list:
        """Flat list of global episode indices (task-major ordering)."""
        if task_ids is None:
            task_ids = range(num_tasks)
        return [
            task_id * num_trials_per_task + trial_id for task_id in task_ids
            for trial_id in range(num_trials_per_task)
        ]

    @staticmethod
    def _get_local_episodes(global_episodes: list, rank: int,
                            world_size: int) -> list:
        """Episodes handled by ``rank`` under round-robin sharding."""
        return global_episodes[rank::world_size]

    @staticmethod
    def _get_local_task_ids(task_ids: list, rank: int,
                            world_size: int) -> list:
        """Task ids handled by ``rank`` under task-level sharding."""
        return list(task_ids)[rank::world_size]

    @staticmethod
    def _resolve_task_ids(num_tasks: int, task_ids=None) -> list:
        """Normalize optional task filters into an ordered task id list."""
        raw_task_ids = task_ids
        if raw_task_ids is None:
            resolved = list(range(num_tasks))
        elif isinstance(raw_task_ids, int):
            resolved = [raw_task_ids]
        elif isinstance(raw_task_ids, str):
            value = raw_task_ids.strip()
            if value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
            resolved = [
                int(item.strip()) for item in value.split(',')
                if item.strip() != ''
            ]
        else:
            resolved = [int(task) for task in raw_task_ids]

        if len(resolved) == 0:
            raise ValueError('At least one task id is required.')
        if len(set(resolved)) != len(resolved):
            raise ValueError(f'Duplicate task ids are not supported: '
                             f'{resolved}')
        invalid = [task for task in resolved if task < 0 or task >= num_tasks]
        if invalid:
            raise ValueError(f'Invalid task ids {invalid}; expected range [0, '
                             f'{num_tasks - 1}].')
        return resolved

    @classmethod
    def _build_local_episode_schedule(cls,
                                      num_tasks: int,
                                      num_trials_per_task: int,
                                      rank: int,
                                      world_size: int,
                                      shard_strategy: str,
                                      task_ids=None) -> list:
        """Build local episode ids in task-major order."""
        if task_ids is None:
            task_ids = list(range(num_tasks))
        else:
            task_ids = list(task_ids)
        strategy = str(shard_strategy).lower()
        if strategy == 'episode':
            return cls._get_local_episodes(
                cls._build_global_episodes(num_tasks, num_trials_per_task,
                                           task_ids), rank, world_size)
        if strategy != 'task':
            raise ValueError(
                f'Unsupported eval_shard_strategy: {shard_strategy}. '
                "Expected one of: ['task', 'episode'].")

        local_tasks = cls._get_local_task_ids(task_ids, rank, world_size)
        return [
            task_id * num_trials_per_task + trial_id for task_id in local_tasks
            for trial_id in range(num_trials_per_task)
        ]

    @staticmethod
    def _write_rank_progress(progress_dir: str, rank: int, episodes: int,
                             successes: int) -> None:
        os.makedirs(progress_dir, exist_ok=True)
        progress_path = os.path.join(progress_dir, f'rank{rank}.json')
        tmp_path = f'{progress_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(
                dict(rank=rank, episodes=episodes, successes=successes), f)
        os.replace(tmp_path, progress_path)

    @staticmethod
    def _read_rank_progress(progress_dir: str) -> tuple:
        total_episodes = 0
        total_successes = 0
        for progress_path in Path(progress_dir).glob('rank*.json'):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                total_episodes += int(progress.get('episodes', 0))
                total_successes += int(progress.get('successes', 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return total_episodes, total_successes

    @staticmethod
    def _get_max_steps(task_suite_name: str, override: int = None) -> int:
        """Per-suite rollout horizon."""
        suite_steps = {
            'libero_spatial': 220,
            'libero_object': 280,
            'libero_goal': 300,
            'libero_10': 520,
            'libero_90': 400,
        }
        if override is not None:
            return int(override)
        if task_suite_name not in suite_steps:
            raise ValueError(f'Unknown task suite: {task_suite_name}')
        return suite_steps[task_suite_name]

    @staticmethod
    def _repeat_initial_states(initial_states, num_trials: int):
        """Repeat LIBERO initial states when a suite exposes too few trials."""
        if len(initial_states) >= num_trials:
            return initial_states
        if len(initial_states) == 0:
            raise ValueError('LIBERO task has no initial states.')

        if hasattr(initial_states, 'extend'):
            while len(initial_states) < num_trials:
                need = num_trials - len(initial_states)
                initial_states.extend(initial_states[:need])
            return initial_states

        repeats = math.ceil(num_trials / len(initial_states))
        return list(initial_states) * repeats

    @staticmethod
    def _build_run_id(task_suite_name: str,
                      model_family: str,
                      run_timestamp: str,
                      suffix: str = None) -> str:
        """Shared, collision-free identifier for one evaluation run."""
        run_id = f'EVAL-{task_suite_name}-{model_family}-{run_timestamp}'
        if suffix:
            run_id = f'{run_id}-{suffix}'
        return run_id

    @staticmethod
    def _build_ckpt_tag(ckpt_path: str) -> str:
        """Stable per-checkpoint folder name for grouping eval runs."""
        return Path(ckpt_path).resolve().stem

    @staticmethod
    def _build_run_dir(ckpt_path: str,
                       run_id: str,
                       output_dir: str = None) -> str:
        """Per-checkpoint, per-run output directory."""
        root = (
            Path(output_dir).expanduser().resolve() if output_dir is not None
            else Path(ckpt_path).resolve().parent.parent)
        return os.path.join(root, 'eval_runs',
                            LiberoEvalRunner._build_ckpt_tag(ckpt_path),
                            run_id)

    @classmethod
    def _build_log_file_path(cls,
                             ckpt_path: str,
                             run_id: str,
                             rank: int,
                             output_dir: str = None) -> str:
        """Per-rank log path inside the per-run directory.

        Encoding the rank in the filename avoids the previous collision where
        ranks sharing a wall-clock second overwrote the same log file.
        """
        return os.path.join(
            cls._build_run_dir(ckpt_path, run_id, output_dir),
            f'rank{rank}.txt')

    def _should_collect_replay_images(self) -> bool:
        """Whether this run may need replay frames for rollout videos."""
        return self.save_rollout_videos or self.save_failed_rollout_videos

    def _should_save_rollout_video(self, success: bool) -> bool:
        """Whether to write the rollout video for one completed episode."""
        return self.save_rollout_videos or (self.save_failed_rollout_videos
                                            and not bool(success))

    def _get_replay_img_keys(self) -> list:
        """Find eval image keys used for lightweight replay frames."""
        img_keys = getattr(self.dataset, 'img_keys', None)
        if not img_keys:
            for transform in getattr(self.dataset, 'transforms', []):
                img_keys = getattr(transform, 'img_keys', None)
                if img_keys:
                    break
        if not img_keys:
            img_keys = ['agentview_image']

        img_keys = list(dict.fromkeys(img_keys))
        if self.save_multi_view_rollout_videos:
            return img_keys
        return img_keys[:1]

    @staticmethod
    def _get_replay_view_name(img_key: str) -> str:
        """Map LIBERO observation keys to concise replay view labels."""
        view_names = {
            'agentview_image': 'image',
            'robot0_eye_in_hand_image': 'wrist_image',
        }
        return view_names.get(img_key, img_key)

    def _get_replay_image(self, obs: Dict, replay_img=None):
        """Extract replay frame, falling back to dataset-produced frame."""
        img_keys = self._get_replay_img_keys()
        try:
            if len(img_keys) == 1:
                img = obs[img_keys[0]]
                return img[::-1, ::-1].copy()

            replay_images = {}
            for img_key in img_keys:
                img = obs[img_key]
                replay_images[self._get_replay_view_name(img_key)] = \
                    img[::-1, ::-1].copy()
            return replay_images
        except KeyError:
            if replay_img is not None:
                return replay_img
            raise

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 norm_stats_key: str = None,
                 dataset_stats_path: str = None,
                 eval_chunk_size: int = 1,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 task_ids=None,
                 num_steps_wait: int = 10,
                 num_inference_steps: int = None,
                 max_steps: int = None,
                 inference_seed: int = None,
                 allowed_missing_key_prefixes: tuple = (),
                 model_build_device: str = None,
                 model_build_dtype: str = None,
                 eval_shard_strategy: str = 'episode',
                 preprocess_every_step: bool = True,
                 save_rollout_videos: bool = True,
                 save_failed_rollout_videos: bool = False,
                 save_multi_view_rollout_videos: bool = False,
                 rollout_dir: str = None,
                 run_id_suffix: str = None,
                 result_output_dir: str = None,
                 result_gpu_id: int = None,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)
        self.set_common_eval_attrs(cfg, seed, ckpt_path, model_family,
                                   mixed_precision_dtype,
                                   enable_mixed_precision_training)
        if (model_build_device is not None
                and str(model_build_device).startswith('cuda')
                and torch.cuda.is_available()):
            torch.cuda.set_device(self.device_id)
        model_cfg = self.prepare_eval_model_cfg(
            cfg,
            model_build_device=model_build_device,
            model_build_dtype=model_build_dtype)
        self.vla = build_vla_from_cfg(model_cfg).eval()
        # Load checkpoint weights if ckpt_path is provided
        if ckpt_path is not None:
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'

            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # A sibling .safetensors is preferred when available because
                # the .pt file also contains the optimizer/scheduler state
                # which is unnecessary for inference and quickly exhausts
                # CPU RAM when loaded on every rank (SIGKILL / exit -9).
                sf_candidate = (
                    ckpt_path[:-len('.pt')] +
                    '.safetensors' if ckpt_path.endswith('.pt') else None)
                if sf_candidate is not None and os.path.exists(sf_candidate):
                    state_dict = load_file(sf_candidate, device='cpu')
                else:
                    # mmap=True avoids copying the whole checkpoint into RAM
                    # on every rank.
                    try:
                        checkpoint = torch.load(
                            ckpt_path, map_location='cpu', mmap=True)
                    except TypeError:
                        checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        state_dict = checkpoint['model']
                        # Drop optimizer/scheduler state ASAP to reclaim RAM.
                        checkpoint.pop('optimizer_state_dict', None)
                        checkpoint.pop('scheduler_state_dict', None)
                        checkpoint.pop('optimizer_state_index_to_name', None)
                    else:
                        state_dict = checkpoint
                    del checkpoint
                    gc.collect()
            self.load_eval_state_dict(state_dict, allowed_missing_key_prefixes)
            del state_dict
            gc.collect()
        data_stat_path = (
            dataset_stats_path if dataset_stats_path is not None else
            self.default_stats_path(self.ckpt_path))
        assert os.path.exists(data_stat_path), \
            f'Dataset statistics file not found at {data_stat_path}!'
        # Load dataset and denormalization action
        denormalize_action['norm_stats'] = data_stat_path
        self.norm_stats_key = norm_stats_key or f'{task_suite_name}_no_noops'
        dataset['task_suite_name'] = task_suite_name
        dataset['norm_stats_key'] = self.norm_stats_key
        dataset['norm_stats'] = data_stat_path
        self._inject_checkpoint_tokenizer(dataset, ckpt_path)
        self.dataset = build_dataset_from_cfg(dataset)
        self.denormalize_action = build_transform_from_cfg(denormalize_action)
        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.task_ids = task_ids
        self.num_steps_wait = num_steps_wait
        self.num_inference_steps = num_inference_steps
        self.max_steps = max_steps
        self.inference_seed = inference_seed
        self.model_build_device = model_build_device
        self.model_build_dtype = self._resolve_model_build_dtype(
            model_build_dtype)
        self.eval_shard_strategy = eval_shard_strategy
        self.preprocess_every_step = preprocess_every_step
        self.save_rollout_videos = save_rollout_videos
        self.save_failed_rollout_videos = save_failed_rollout_videos
        self.save_multi_view_rollout_videos = save_multi_view_rollout_videos
        self.rollout_dir = rollout_dir
        self.run_id_suffix = run_id_suffix
        self.result_output_dir = result_output_dir
        self.result_gpu_id = (
            self.device_id if result_gpu_id is None else int(result_gpu_id))

        if os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                norm_stats = json.load(f)
            self.update_model_norm_stats(norm_stats)
        else:
            overwatch.warning(
                'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
            )

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        if self.enable_mixed_precision_training:
            self.vla.to(
                device=self.device_id, dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda(self.device_id)

    def cleanup(self) -> None:
        """Release per-suite evaluation resources before the next suite."""
        self.vla = None
        self.dataset = None
        self.denormalize_action = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except RuntimeError:
                    pass

    def run(self):
        """Run the evaluation process."""
        benchmark = _get_libero_benchmark()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        num_tasks = num_tasks_in_suite
        task_ids = self._resolve_task_ids(num_tasks, self.task_ids)
        overwatch.info(f'Task suite: {self.task_suite_name}')
        overwatch.info(f'Running evaluation on {num_tasks_in_suite} tasks '
                       f'with {self.num_trials_per_task} trials each.')
        if len(task_ids) != num_tasks:
            overwatch.info(f'Using task ids: {task_ids}')
        overwatch.info(f'Using model family: {self.model_family}')
        overwatch.info(f'Using resize size: {self.resize_size}')
        overwatch.info(f'Using evaluation chunk size: {self.eval_chunk_size}')
        overwatch.info(
            f'Using eval shard strategy: {self.eval_shard_strategy}')
        if self.model_build_device is not None:
            overwatch.info(
                f'Using model build device: {self.model_build_device}')
        if self.model_build_dtype is not None:
            overwatch.info(
                f'Using model build dtype: {self.model_build_dtype}')
        overwatch.info(
            f'Using mixed precision dtype: {self.mixed_precision_dtype}')
        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = self._build_local_episode_schedule(
            num_tasks,
            self.num_trials_per_task,
            rank,
            world_size,
            self.eval_shard_strategy,
            task_ids=task_ids)
        # Use a single run timestamp shared across ranks so every rank writes
        # into the same per-run directory. Broadcasting from rank 0 also avoids
        # the previous collision where ranks landing in the same wall-clock
        # second produced (and overwrote) the same ``EVAL-...-<second>.txt``.
        run_timestamp = (
            time.strftime('%Y_%m_%d-%H_%M_%S') if rank == 0 else None)
        run_timestamp_holder = [run_timestamp]
        dist.broadcast_object_list(run_timestamp_holder, src=0)
        run_timestamp = run_timestamp_holder[0]
        run_id = self._build_run_id(
            self.task_suite_name,
            self.model_family,
            run_timestamp,
            suffix=self.run_id_suffix)
        # Isolate each evaluation run in its own directory. Manager-launched
        # workers keep their artifacts under the manager output root.
        self.run_dir = self._build_run_dir(
            self.ckpt_path, run_id, output_dir=self.result_output_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        progress_dir = os.path.join(self.run_dir, 'rank_progress')
        total_eval_episodes = len(
            self._build_global_episodes(num_tasks, self.num_trials_per_task,
                                        task_ids))
        local_log_filepath = self._build_log_file_path(self.ckpt_path, run_id,
                                                       rank,
                                                       self.result_output_dir)
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = torch.zeros(
            1, device=torch.cuda.current_device()), torch.zeros(
                1, device=torch.cuda.current_device())
        unnorm_key = self.task_suite_name
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            candidate_unnorm_key = f'{unnorm_key}_no_noops'
            if (unnorm_key not in self.vla.norm_stats
                    and candidate_unnorm_key in self.vla.norm_stats):
                unnorm_key = candidate_unnorm_key
            assert unnorm_key in self.vla.norm_stats, (
                f'Action un-norm key {unnorm_key} '
                'not found in VLA norm_stats!')
        if rank == 0:
            pbar = tqdm.tqdm(
                total=total_eval_episodes,
                desc='Evaluation global',
                dynamic_ncols=True)
        else:
            pbar = None
        # Per-task accumulators aggregated across ranks at the end.
        cuda_dev = torch.cuda.current_device()
        task_successes = torch.zeros(num_tasks, device=cuda_dev)
        task_episodes = torch.zeros(num_tasks, device=cuda_dev)
        task_durations = torch.zeros(num_tasks, device=cuda_dev)
        # Per-(task, trial) success grid so rank 0 can reconstruct the exact
        # ``success_episodes`` / ``failure_episodes`` lists. Episodes are
        # sharded across ranks, so each rank fills only its own trials and the
        # grid is summed across ranks at the end. ``-1`` marks trials not run
        # by this rank.
        trial_success_grid = torch.full((num_tasks, self.num_trials_per_task),
                                        -1.0,
                                        device=cuda_dev)
        # Wall-clock start time of the first trial each rank runs per task.
        task_start_times = [float('inf')] * num_tasks
        max_steps = self._get_max_steps(self.task_suite_name, self.max_steps)
        rank_episode_count = 0
        rank_success_count = 0
        current_task_id = None
        env = None
        initial_states = None
        task_description = None
        try:
            for local_id in local_episodes:
                # Get task ID from local episode index
                task_id = local_id // self.num_trials_per_task
                # Get trial ID within the task
                trial_id = local_id % self.num_trials_per_task

                # Log the current task and trial
                overwatch.info(f'Evaluating Task {task_id}, Trial {trial_id}')
                log_file.write(
                    f'Evaluating Task {task_id}, Trial {trial_id}\n')

                if task_id != current_task_id:
                    if env is not None:
                        env.close()
                    task = task_suite.get_task(task_id)
                    initial_states = self._repeat_initial_states(
                        task_suite.get_task_init_states(task_id),
                        self.num_trials_per_task)
                    env, task_description = get_libero_env(
                        task, resolution=256)
                    current_task_id = task_id
                    overwatch.info(f'\nTask: {task_description}')
                    log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[trial_id])
                is_new_episode = True

                # Setup
                t = 0
                replay_images = []
                next_batch = None

                overwatch.info(f'Starting episode {trial_id+1}...')

                log_file.write(f'Starting episode {trial_id+1}...\n')
                episode_start = time.time()
                task_start_times[task_id] = min(task_start_times[task_id],
                                                episode_start)
                done = False
                while t < max_steps + self.num_steps_wait:
                    # IMPORTANT: Do nothing for the first
                    # few timesteps
                    # because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < self.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue
                    if next_batch is None:
                        obs['task_description'] = task_description
                        obs['is_new_episode'] = is_new_episode
                        batch, replay_img = self.dataset(obs)
                        if (self._should_collect_replay_images()
                                and len(replay_images) == 0):
                            replay_images.append(
                                self._get_replay_image(obs, replay_img))
                    else:
                        batch = next_batch
                        next_batch = None
                    is_new_episode = False
                    batch['unnorm_key'] = unnorm_key
                    predict_kwargs = dict(batch)
                    if self.num_inference_steps is not None:
                        predict_kwargs['num_inference_steps'] = \
                            self.num_inference_steps
                    if self.inference_seed is not None:
                        predict_kwargs['seed'] = self.inference_seed
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            actions = self.vla.predict_action(**predict_kwargs)
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :self.eval_chunk_size, :].float().cpu().numpy()
                    else:
                        assert len(actions.shape) == 2, \
                            f'Unexpected action shape: {actions.shape}'
                        actions = actions[0, None, :].float().cpu().numpy()
                    for action in actions:
                        inputs = dict(
                            action=action,
                            task_suite_name=self.task_suite_name,
                            norm_stats_key=self.norm_stats_key,
                        )
                        action_denormed = self.denormalize_action(inputs)
                        obs, reward, done, info = env.step(
                            action_denormed.tolist())
                        if done:
                            total_successes += 1
                            rank_success_count += 1
                            if self._should_collect_replay_images():
                                replay_images.append(
                                    self._get_replay_image(obs))
                            next_batch = None
                            break
                        if self.preprocess_every_step:
                            obs['task_description'] = task_description
                            obs['is_new_episode'] = False
                            batch, replay_img = self.dataset(obs)
                            if self._should_collect_replay_images():
                                replay_images.append(
                                    self._get_replay_image(obs, replay_img))
                            next_batch = batch
                        else:
                            if self._should_collect_replay_images():
                                replay_images.append(
                                    self._get_replay_image(obs))
                            next_batch = None
                        t += 1
                    if done:
                        break
                total_episodes += 1
                rank_episode_count += 1
                episode_duration = time.time() - episode_start
                task_successes[task_id] += float(bool(done))
                task_episodes[task_id] += 1.0
                task_durations[task_id] += episode_duration
                trial_success_grid[task_id, trial_id] = float(bool(done))
                if self._should_save_rollout_video(done):
                    video_root = (
                        self.result_output_dir if self.result_output_dir
                        is not None else self.run_dir)
                    rollout_dir = self.rollout_dir
                    if (rollout_dir is None
                            and self.result_output_dir is not None):
                        rollout_dir = os.path.join(self.result_output_dir,
                                                   self.task_suite_name,
                                                   'videos')
                    if rollout_dir is not None:
                        rollout_dir = os.path.expanduser(rollout_dir)
                        if not os.path.isabs(rollout_dir):
                            rollout_dir = os.path.join(video_root, rollout_dir)
                    save_rollout_video(
                        replay_images,
                        f'task{task_id}_trial{trial_id}',
                        success=done,
                        task_description=task_description,
                        work_dir=video_root,
                        log_file=log_file,
                        rollout_dir=rollout_dir,
                        save_multi_view=self.save_multi_view_rollout_videos)
                rank_success_rate = (
                    rank_success_count / rank_episode_count * 100)
                self._write_rank_progress(progress_dir, rank,
                                          rank_episode_count,
                                          rank_success_count)
                display_episode_count = rank_episode_count
                display_success_count = rank_success_count
                display_success_rate = rank_success_rate
                if rank == 0 and pbar is not None:
                    display_episode_count, display_success_count = \
                        self._read_rank_progress(progress_dir)
                    display_success_rate = (
                        display_success_count / max(display_episode_count, 1) *
                        100)
                    pbar_delta = max(display_episode_count - pbar.n, 0)
                    overwatch.info('[eval-progress] '
                                   f'episodes={display_episode_count}/'
                                   f'{total_eval_episodes} '
                                   f'successes={display_success_count} '
                                   f'success_rate={display_success_rate:.2f}%')
                    pbar.set_postfix(
                        successes=display_success_count,
                        episodes=display_episode_count,
                        success_rate=f'{display_success_rate:.1f}%')
                    if pbar_delta > 0:
                        pbar.update(pbar_delta)

                # except Exception as e:
                #     print(f'Error during action prediction: {e}')
                #     log_file.write(f'Caught exception: {e}\n')
                #     action = get_libero_dummy_action()
                log_file.write(f'Success: {done}\n')
                log_file.write('# local episodes completed so far: '
                               f'{rank_episode_count}\n')
                success_log = (f'# local successes: {rank_success_count} '
                               f'({rank_success_rate:.1f}%)\n')
                log_file.write(success_log)
                log_file.flush()
        finally:
            if env is not None:
                env.close()
            if pbar is not None:
                pbar.close()

        global_episodes = total_episodes.clone()
        global_successes = total_successes.clone()
        task_start_times = torch.tensor(task_start_times, device=cuda_dev)
        dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_successes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_episodes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_durations, op=dist.ReduceOp.SUM)
        # Each (task, trial) is run by exactly one rank under the configured
        # sharding, so MAX recovers that rank's 0/1 outcome (unrun cells stay
        # ``-1``); MIN recovers the earliest per-task start time.
        dist.all_reduce(trial_success_grid, op=dist.ReduceOp.MAX)
        dist.all_reduce(task_start_times, op=dist.ReduceOp.MIN)
        global_episode_count = int(global_episodes[0].item())
        global_success_count = int(global_successes[0].item())
        global_success_rate = (
            global_success_count / max(global_episode_count, 1) * 100)
        if rank == 0:
            overwatch.info(f'# episodes completed: {global_episode_count}')
            overwatch.info(f'# successes: {global_success_count} '
                           f'({global_success_rate:.1f}%)')
            summary_path = os.path.join(self.run_dir, 'summary.txt')
            with open(summary_path, 'w') as sf:
                sf.write(f'task_suite: {self.task_suite_name}\n')
                sf.write(f'model_family: {self.model_family}\n')
                sf.write(f'task_ids: {task_ids}\n')
                sf.write(f'num_trials_per_task: {self.num_trials_per_task}\n')
                sf.write(f'eval_chunk_size: {self.eval_chunk_size}\n')
                sf.write(f'num_steps_wait: {self.num_steps_wait}\n')
                sf.write(f'num_inference_steps: {self.num_inference_steps}\n')
                sf.write(f'max_steps: {self.max_steps}\n')
                sf.write(f'eval_shard_strategy: {self.eval_shard_strategy}\n')
                sf.write(
                    f'preprocess_every_step: {self.preprocess_every_step}\n')
                sf.write(f'save_rollout_videos: '
                         f'{self.save_rollout_videos}\n')
                sf.write(f'save_failed_rollout_videos: '
                         f'{self.save_failed_rollout_videos}\n')
                sf.write(f'save_multi_view_rollout_videos: '
                         f'{self.save_multi_view_rollout_videos}\n')
                sf.write(f'rollout_dir: {self.rollout_dir}\n')
                sf.write(f'seed: {self.seed}\n')
                sf.write(f'# successes: {global_success_count} / '
                         f'{global_episode_count} '
                         f'({global_success_rate:.1f}%)\n')  # noqa: E231
            overwatch.info(f'[*] Wrote eval summary to {summary_path}')
            self._write_libero_summary_artifacts(task_suite, num_tasks,
                                                 task_successes.cpu(),
                                                 task_episodes.cpu(),
                                                 task_durations.cpu(),
                                                 trial_success_grid.cpu(),
                                                 task_start_times.cpu())
        log_file.close()
        dist.barrier()
        return

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Human-readable duration for summary files."""
        seconds = int(round(seconds))
        if seconds < 60:
            return f'{seconds:02d}s'
        if seconds < 3600:
            return f'{seconds // 60:02d}m{seconds % 60:02d}s'
        hours, rem = divmod(seconds, 3600)
        return f'{hours:02d}h{rem // 60:02d}m{rem % 60:02d}s'

    def _write_libero_summary_artifacts(self, task_suite, num_tasks,
                                        task_successes, task_episodes,
                                        task_durations, trial_success_grid,
                                        task_start_times):
        """Write LIBERO per-task and per-suite summary artifacts.

        For a single task suite, the ``Overall`` column equals that suite.
        Per-task ``duration`` is the summed wall time of the task's episodes
        across ranks; ``Average Time (s)`` is ``total_time / total_tasks`` and
        ``Max Time (s)`` is the longest per-task duration. Inputs are CPU
        tensors of shape ``[num_tasks]`` already reduced across ranks.
        """
        suite = self.task_suite_name
        task_results = {}
        ordered_task_ids = []
        total_successes = 0
        total_trials = 0
        total_time = 0.0
        max_time = 0.0
        # Per-task result JSONs live under a per-suite subdirectory.
        suite_dir = os.path.join(self.run_dir, suite)
        os.makedirs(suite_dir, exist_ok=True)
        status_dir = os.path.join(self.run_dir, 'task_status')
        os.makedirs(status_dir, exist_ok=True)
        manager_suite_dir = None
        if self.result_output_dir is not None:
            manager_suite_dir = os.path.join(self.result_output_dir, suite)
            os.makedirs(manager_suite_dir, exist_ok=True)
        for task_id in range(num_tasks):
            eps = int(task_episodes[task_id].item())
            if eps == 0:
                continue
            succ = int(task_successes[task_id].item())
            dur = float(task_durations[task_id].item())
            try:
                description = task_suite.get_task(task_id).language
            except Exception:  # noqa: BLE001
                description = ''
            # Reconstruct per-trial success / failure episode indices from the
            # reduced grid (``1`` success, ``0`` failure, ``-1`` not run).
            success_episodes = []
            failure_episodes = []
            for trial_id in range(self.num_trials_per_task):
                outcome = int(trial_success_grid[task_id, trial_id].item())
                if outcome == 1:
                    success_episodes.append(trial_id)
                elif outcome == 0:
                    failure_episodes.append(trial_id)
            start_ts = float(task_start_times[task_id].item())
            start_str = (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts))
                if start_ts != float('inf') else '')
            task_results[f'{suite}_{task_id}'] = {
                'success_rate': succ / eps * 100,
                'duration': dur,
                'total_episodes': eps,
                'successes': succ,
                'task_description': description,
            }
            # Per-task results.json.
            per_task = {
                'task_suite': suite,
                'task_id': task_id,
                'task_description': description,
                'successes': succ,
                'total_episodes': eps,
                'success_episodes': success_episodes,
                'failure_episodes': failure_episodes,
                'start_time': start_str,
                'duration': dur,
                'gpu_id': self.result_gpu_id,
            }
            with open(
                    os.path.join(suite_dir, f'task{task_id}_results.json'),
                    'w') as jf:
                json.dump(per_task, jf, indent=4)
            if manager_suite_dir is not None:
                manager_result_path = os.path.join(
                    manager_suite_dir,
                    f'gpu{self.result_gpu_id}_task{task_id}_results.json')
                with open(manager_result_path, 'w') as jf:
                    json.dump(per_task, jf, indent=4)
            # task_status/<suite>_task{id}.status -- ``STATE|succ|total|ts``.
            with open(
                    os.path.join(status_dir, f'{suite}_task{task_id}.status'),
                    'w') as stf:
                stf.write(f'SUCCESS|{succ}|{eps}|{int(start_ts)}'
                          if start_ts != float('inf') else f'SUCCESS|{succ}|'
                          f'{eps}|0')
            overwatch.info(f'Task {task_id} completed: {succ}/{eps} successes')
            overwatch.info(f'Time taken: {dur:.2f} seconds')
            ordered_task_ids.append(task_id)
            total_successes += succ
            total_trials += eps
            total_time += dur
            max_time = max(max_time, dur)

        if total_trials == 0:
            return
        completed_tasks = len(ordered_task_ids)
        suite_success_rate = total_successes / total_trials * 100
        avg_time = total_time / completed_tasks if completed_tasks else 0.0
        with open(os.path.join(self.run_dir, 'failed_tasks.txt'), 'w'):
            pass

        # Human-readable per-suite statistics block appended to summary.txt.
        summary_path = os.path.join(self.run_dir, 'summary.txt')
        with open(summary_path, 'a') as sf:
            sf.write('\n=== Evaluation Results Summary ===\n')
            sf.write(f'\n{suite}:\n')
            sf.write(f'- Tasks completed: {completed_tasks}\n')
            sf.write(f'- Total attempts: {total_trials}\n')
            sf.write(f'- Successful attempts: {total_successes}\n')
            sf.write(f'- Success rate: {suite_success_rate:.2f}%\n')
            sf.write(f'- Total time: {self._format_duration(total_time)}\n')
            sf.write('- Average time per task: '
                     f'{self._format_duration(avg_time)}\n')
            sf.write('- Longest task time: '
                     f'{self._format_duration(max_time)}\n')

        # summary.csv -- single suite, so ``Overall`` mirrors the suite column.
        title = os.path.basename(self.ckpt_path)
        summary_csv = os.path.join(self.run_dir, 'summary.csv')
        with open(summary_csv, 'w') as f:
            f.write(f'{title}\n')
            f.write(f',{suite},Overall\n')
            f.write('Success Rate (%),'
                    f'{suite_success_rate:.2f},{suite_success_rate:.2f}\n')
            f.write(f'Average Time (s),{avg_time:.2f},{avg_time:.2f}\n')
            f.write(f'Max Time (s),{max_time:.2f},{max_time:.2f}\n')

        # task_success_rates.csv -- one row per task, ordered by task id.
        task_csv = os.path.join(self.run_dir, 'task_success_rates.csv')
        with open(task_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Description', 'Success Rate (%)'])
            for task_id in ordered_task_ids:
                res = task_results[f'{suite}_{task_id}']
                writer.writerow([
                    f'{suite}_{task_id}', res['task_description'],
                    f"{res['success_rate']:.2f}"
                ])

        # summary.json -- detailed per-suite / per-task breakdown.
        cfg_filename = getattr(self.cfg, 'filename', None)
        config_name = (
            os.path.splitext(os.path.basename(cfg_filename))[0]
            if cfg_filename else '')
        summary_json = os.path.join(self.run_dir, 'summary.json')
        with open(summary_json, 'w') as f:
            json.dump(
                {
                    'run_id': os.path.basename(self.run_dir),
                    'ckpt': self.ckpt_path,
                    'config': config_name,
                    'suite_stats': {
                        suite: {
                            'total_tasks': completed_tasks,
                            'total_trials': total_trials,
                            'total_successes': total_successes,
                            'total_time': total_time,
                            'max_time': max_time,
                        }
                    },
                    'task_results': task_results,
                    'overall': {
                        'average_success_rate': suite_success_rate,
                        'total_time': total_time,
                        'average_task_time': avg_time,
                    },
                },
                f,
                indent=4)
        overwatch.info(f'[*] Wrote LIBERO summary artifacts to {self.run_dir}')
