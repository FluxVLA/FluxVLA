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
"""RoboCasa simulation evaluation runner."""

import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS
from .base_eval_runner import BaseEvalRunner

overwatch = initialize_overwatch(__name__)

# Split RoboCasa GR1 29D actions into the dict format required by env.step.
# FluxVLA converted data order: left_arm + left_hand + right_arm + right_hand
# + waist.
ROBOCASA_FLUXVLA_ACTION_KEYS = {
    'action.left_arm': (0, 7),  # left arm, 7D
    'action.left_hand': (7, 13),  # left hand, 6D
    'action.right_arm': (13, 20),  # right arm, 7D
    'action.right_hand': (20, 26),  # right hand, 6D
    'action.waist': (26, 29),  # waist, 3D
}

# Official GR00T N1.5 fourier_gr1_arms_waist order:
# left_arm + right_arm + left_hand + right_hand + waist.
ROBOCASA_N15_ACTION_KEYS = {
    'action.left_arm': (0, 7),  # left arm, 7D
    'action.right_arm': (7, 14),  # right arm, 7D
    'action.left_hand': (14, 20),  # left hand, 6D
    'action.right_hand': (20, 26),  # right hand, 6D
    'action.waist': (26, 29),  # waist, 3D
}


@RUNNERS.register_module()
class RobocasaEvalRunner(BaseEvalRunner):
    """Runner for evaluating VLA models on Robocasa simulation tasks.

    Args:
        cfg: Full config object containing the model section.
        seed: Random seed.
        ckpt_path: Path to the model checkpoint.
        model_family: Model family name, such as ``pi0`` or ``groot``.
        task_list: RoboCasa Gymnasium environment names.
        dataset: Evaluation dataset config.
        denormalize_action: Action denormalization transform config.
        eval_chunk_size: Number of predicted actions executed per step.
        max_episode_steps: Maximum number of environment steps per episode.
        num_trials_per_task: Number of trials for each task.
        task_ids: Optional task id filter. Used by manager workers to run
            one or a few RoboCasa tasks.
        eval_shard_strategy: Episode assignment strategy. ``task`` keeps all
            trials for a task on the same rank; ``episode`` preserves the old
            round-robin episode sharding.
        mixed_precision_dtype: Mixed precision dtype name.
        enable_mixed_precision_training: Whether autocast is enabled.
        unnorm_key: Top-level key in the dataset statistics.
        output_dir: Optional output root. Eval artifacts are written under
            ``<output_dir>/eval_runs/<ckpt>/<run_id>`` to match LIBERO.
        save_video: Whether to save rollout videos.
        rollout_video_key: Observation image key used for rollout videos.
        action_order: Action split order. Defaults to ``n15`` for GR00T and
            ``fluxvla`` otherwise.
        norm_stats_path: Optional explicit dataset statistics path.
        grouped_norm_stats: Whether to load one statistics file per group.
        norm_stats_group_names: Per-task group names for grouped statistics.
        deterministic_env: Whether to seed RoboCasa env construction/reset.
        deterministic_action_sampling: Whether to seed stochastic action
            sampling before every policy call.
        run_id_suffix: Optional suffix appended to the eval run id.
        result_output_dir: Optional manager output root. When set, per-task
            result files are mirrored to
            ``<result_output_dir>/robocasa/gpu{gpu_id}_task{task_id}_results.json``.
        result_gpu_id: GPU id written to mirrored result filenames.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_list: List[str],
                 dataset: Dict,
                 denormalize_action: Dict,
                 eval_chunk_size: int = 10,
                 max_episode_steps: int = 720,
                 num_trials_per_task: int = 50,
                 task_ids=None,
                 eval_shard_strategy: str = 'episode',
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 unnorm_key: str = 'robocasa_gr1_test',
                 output_dir: Optional[str] = None,
                 save_video: bool = True,
                 rollout_video_key: Optional[str] = (
                     'video.ego_view_pad_res256_freq20'),
                 norm_stats_path: Optional[str] = None,
                 action_order: Optional[str] = None,
                 grouped_norm_stats: bool = False,
                 norm_stats_group_names: Optional[List[str]] = None,
                 deterministic_env: bool = True,
                 deterministic_action_sampling: bool = True,
                 run_id_suffix: Optional[str] = None,
                 result_output_dir: Optional[str] = None,
                 result_gpu_id: Optional[int] = None,
                 **kwargs):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg)

        self.device_id = overwatch.local_rank()

        # Build model.
        self.vla = self.build_eval_vla(cfg)

        # Load checkpoint weights.
        if ckpt_path is not None:
            assert Path(ckpt_path).exists(), \
                f'Checkpoint not found: {ckpt_path}'

            # Support checkpoint directories, single safetensors, and .pt
            # files.
            if os.path.isdir(ckpt_path):
                # Directory path: merge all sharded safetensors files.
                overwatch.info(
                    f'Loading checkpoint from directory: {ckpt_path}')
                state_dict = dict()
                safetensors_files = sorted([
                    f for f in os.listdir(ckpt_path)
                    if f.endswith('.safetensors') and f.startswith('model-')
                ])
                if not safetensors_files:
                    raise FileNotFoundError(
                        f'No model-*.safetensors files found in {ckpt_path}')
                for file in safetensors_files:
                    file_path = os.path.join(ckpt_path, file)
                    overwatch.info(f'  Loading {file}...')
                    state_dict.update(load_file(file_path, device='cpu'))
                overwatch.info(
                    f'Loaded {len(safetensors_files)} safetensors files')
            elif ckpt_path.endswith('.safetensors'):
                # Single safetensors file.
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # PyTorch checkpoint.
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)

            # Apply name_mapping only when the checkpoint uses source prefixes.
            # This is needed for official checkpoints such as GR00T-N1.5-3B.
            # name_mapping is usually defined in cfg.model, not
            # cfg.inference_model.
            #
            # Detection rules:
            # - official GR00T-N1.5-3B uses source prefixes and needs mapping
            # - FluxVLA-trained checkpoints already use model-native prefixes
            # This keeps PI0.5, LIBERO, and fine-tuned GR00T paths compatible.
            model_cfg = (
                cfg.model if hasattr(cfg, 'model') else cfg.inference_model)
            if 'name_mapping' in model_cfg and model_cfg['name_mapping']:
                # model_cfg.name_mapping keys are model-native prefixes, while
                # values are external checkpoint source prefixes. Fine-tuned
                # checkpoints are already native and must not be remapped.
                def _has_prefix(key: str, prefix: str) -> bool:
                    return key == prefix or key.startswith(f'{prefix}.')

                model_prefixes = list(model_cfg['name_mapping'].keys())
                ckpt_prefixes = list(model_cfg['name_mapping'].values())
                has_native_keys = any(
                    any(_has_prefix(k, p) for k in state_dict.keys())
                    for p in model_prefixes)
                needs_mapping = any(
                    any(_has_prefix(k, p) for k in state_dict.keys())
                    for p in ckpt_prefixes) and not has_native_keys

                if needs_mapping:
                    overwatch.info(
                        'Detected checkpoint with external key format, '
                        'applying name_mapping...')
                    mapped_state_dict = {}
                    for model_key, ckpt_key in \
                            model_cfg['name_mapping'].items():
                        for k, v in state_dict.items():
                            if _has_prefix(k, ckpt_key):
                                new_key = model_key + k[len(ckpt_key):]
                                mapped_state_dict[new_key] = v
                    state_dict = mapped_state_dict
                    overwatch.info(
                        f'Applied name_mapping: {len(state_dict)} keys after '
                        f'mapping')
                else:
                    overwatch.info('Checkpoint already in native format, '
                                   'skipping name_mapping')

            # Handle shared tensors (e.g., embed_tokens and lm_head in GR00T)
            from fluxvla.engines.utils.checkpoint_utils import \
                handle_shared_tensors
            state_dict = handle_shared_tensors(state_dict,
                                               self.vla.state_dict(),
                                               overwatch)

            self.vla.load_state_dict(state_dict, strict=True)

        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.grouped_norm_stats = grouped_norm_stats
        self.norm_stats_group_names = norm_stats_group_names or []
        work_dir = Path(self.ckpt_path).resolve().parent.parent

        # Statistics can come from an explicit path, a default file, or groups.
        if norm_stats_path is not None and self.grouped_norm_stats:
            raise ValueError('norm_stats_path cannot be used together with '
                             'grouped_norm_stats')

        if self.grouped_norm_stats:
            assert len(self.norm_stats_group_names) == len(task_list), (
                'norm_stats_group_names must have the same length as '
                'task_list')
            self._stats_full_by_group: Dict[str, dict] = {}
            self._denorm_by_group: Dict[str, Any] = {}
            for g in sorted(set(self.norm_stats_group_names)):
                gpath = work_dir / f'dataset_statistics_{g}.json'
                assert gpath.is_file(), (
                    f'[grouped_norm_stats] missing {gpath}. Grouped training '
                    f'should create it via save_grouped_dataset_statistics')
                with open(gpath, 'r', encoding='utf-8') as f:
                    self._stats_full_by_group[g] = json.load(f)
                assert unnorm_key in self._stats_full_by_group[g], (
                    f'{gpath} does not contain unnorm_key={unnorm_key!r}')
            for g in self._stats_full_by_group:
                da_cfg = copy.deepcopy(denormalize_action)
                da_cfg['norm_stats'] = self._stats_full_by_group[g]
                self._denorm_by_group[g] = build_transform_from_cfg(da_cfg)
            first_g = self.norm_stats_group_names[0]
            data_stat_path = str(work_dir /
                                 f'dataset_statistics_{first_g}.json')
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = self._denorm_by_group[first_g]
        else:
            data_stat_path = (
                str(Path(norm_stats_path).expanduser().resolve())
                if norm_stats_path is not None else os.path.join(
                    work_dir, 'dataset_statistics.json'))
            assert os.path.exists(data_stat_path), \
                f'dataset_statistics.json not found at {data_stat_path}'
            denormalize_action['norm_stats'] = data_stat_path
            dataset['norm_stats'] = data_stat_path
            dataset['unnorm_key'] = unnorm_key
            self.dataset = build_dataset_from_cfg(dataset)
            self.denormalize_action = build_transform_from_cfg(
                denormalize_action)

        if model_family == 'groot' and hasattr(self.dataset, 'transforms'):
            transform_names = [
                t.__class__.__name__ for t in self.dataset.transforms
            ]
            if 'PreparePromptWithState' in transform_names:
                raise RuntimeError(
                    'GR00T RoboCasa eval must not use PreparePromptWithState. '
                    'That is the PI0.5 text-state prompt and will make this '
                    f'evaluation invalid. Current transforms: '
                    f'{transform_names}')

        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        if action_order is None:
            action_order = 'n15' if model_family == 'groot' else 'fluxvla'
        if action_order not in ('fluxvla', 'n15'):
            raise ValueError(f'Unsupported action_order={action_order}. '
                             "Expected 'fluxvla' or 'n15'.")
        self.action_keys = (
            ROBOCASA_N15_ACTION_KEYS
            if action_order == 'n15' else ROBOCASA_FLUXVLA_ACTION_KEYS)
        self.action_order = action_order
        self.task_list = task_list
        self.max_episode_steps = max_episode_steps
        self.num_trials_per_task = num_trials_per_task
        self.task_ids = task_ids
        self.eval_shard_strategy = eval_shard_strategy
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.unnorm_key = unnorm_key
        self.distributed_state = overwatch.distributed_state

        if 'save_rollout_videos' in kwargs:
            save_video = self._coerce_bool(kwargs['save_rollout_videos'])
        self.save_video = save_video
        self.rollout_video_key = rollout_video_key
        self.deterministic_env = deterministic_env
        self.deterministic_action_sampling = deterministic_action_sampling
        self.run_id_suffix = run_id_suffix
        self.result_output_dir = result_output_dir
        self.result_gpu_id = (
            self.device_id if result_gpu_id is None else int(result_gpu_id))

        # Attach norm_stats to the model for heads that consume them.
        if self.grouped_norm_stats:
            first_g = self.norm_stats_group_names[0]
            self.vla.norm_stats = self._stats_full_by_group[first_g]
        elif os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                self.vla.norm_stats = json.load(f)

        self._active_denorm = self.denormalize_action

    @staticmethod
    def _normalize_seed(seed: int) -> int:
        return int(seed) % np.iinfo(np.uint32).max

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return bool(value)

    def _get_rollout_video_keys(self) -> List[str]:
        """Return candidate obs image keys for stable rollout videos."""
        keys = []
        if self.rollout_video_key:
            keys.append(self.rollout_video_key)

        for transform in getattr(self.dataset, 'transforms', []):
            img_key = getattr(transform, 'img_key', None)
            if img_key:
                keys.append(img_key)

        keys.extend([
            'video.ego_view_pad_res256_freq20',
            'video.ego_view_bg_crop_pad_res256_freq20',
        ])
        return list(dict.fromkeys(keys))

    def _get_rollout_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """Extract a single consistent obs frame for rollout video saving."""
        for img_key in self._get_rollout_video_keys():
            frame = obs.get(img_key, None)
            if isinstance(frame, np.ndarray):
                return frame.copy()
        return None

    def _episode_seed(self, local_id: int) -> int:
        return self._normalize_seed(self.seed + local_id)

    def _action_seed(self, local_id: int, step: int) -> int:
        return self._normalize_seed(self.seed + 1_000_003 * (local_id + 1) +
                                    step)

    @staticmethod
    def _seed_python_numpy_torch(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _seed_torch(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run_setup(self):
        """Initialize CUDA placement and model state."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(self.device_id)
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        self.vla.cuda(self.device_id)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(round(seconds))
        if seconds < 60:
            return f'{seconds:02d}s'
        if seconds < 3600:
            return f'{seconds // 60:02d}m{seconds % 60:02d}s'
        hours, rem = divmod(seconds, 3600)
        return f'{hours:02d}h{rem // 60:02d}m{rem % 60:02d}s'

    @staticmethod
    def _robocasa_group_name(env_name: str) -> str:
        task_name = str(env_name).split('/')[-1]
        if 'ToCabinet' in task_name:
            return 'Cabinet'
        if 'ToDrawer' in task_name:
            return 'Drawer'
        if 'ToMicrowave' in task_name:
            return 'Microwave'
        return 'Generalization'

    @staticmethod
    def _build_global_episodes(num_tasks: int,
                               num_trials_per_task: int,
                               task_ids=None) -> list:
        """Flat list of global episode indices in task-major order."""
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
    def _build_run_id(model_family: str,
                      timestamp: str,
                      suffix: Optional[str] = None) -> str:
        run_id = f'EVAL-robocasa-{model_family}-{timestamp}'
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
                       output_dir: Optional[str] = None) -> str:
        """Per-checkpoint, per-run output directory.

        This mirrors ``LiberoEvalRunner`` so RoboCasa and LIBERO eval outputs
        share the same filesystem layout.
        """
        if output_dir is not None:
            root = Path(output_dir).expanduser().resolve()
        else:
            root = Path(ckpt_path).resolve().parent.parent
        return os.path.join(root, 'eval_runs',
                            RobocasaEvalRunner._build_ckpt_tag(ckpt_path),
                            run_id)

    def _write_robocasa_summary_artifacts(self, run_dir: Path, run_id: str,
                                          task_successes, task_episodes,
                                          task_durations) -> str:
        group_order = ['Cabinet', 'Drawer', 'Microwave', 'Generalization']
        summary_dir = run_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
        task_result_dir = summary_dir / 'robocasa'
        task_result_dir.mkdir(parents=True, exist_ok=True)
        manager_result_dir = None
        if self.result_output_dir is not None:
            manager_result_dir = (
                Path(self.result_output_dir).expanduser().resolve() /
                'robocasa')
            manager_result_dir.mkdir(parents=True, exist_ok=True)
        group_stats = {
            group: {
                'total_tasks': 0,
                'total_trials': 0,
                'total_successes': 0,
                'total_time': 0.0,
                'max_time': 0.0,
            }
            for group in group_order
        }
        task_stats = {}
        total_trials = 0
        total_successes = 0
        total_time = 0.0
        overall_max_time = 0.0
        for task_id, env_name in enumerate(self.task_list):
            eps = int(task_episodes[task_id].item())
            if eps == 0:
                continue
            succ = int(task_successes[task_id].item())
            dur = float(task_durations[task_id].item())
            rate = succ / max(eps, 1) * 100
            group = self._robocasa_group_name(env_name)
            stats = group_stats[group]
            stats['total_tasks'] += 1
            stats['total_trials'] += eps
            stats['total_successes'] += succ
            stats['total_time'] += dur
            stats['max_time'] = max(stats['max_time'], dur)
            task_stats[str(env_name)] = {
                'task_id': task_id,
                'group': group,
                'total_episodes': eps,
                'successes': succ,
                'success_rate': rate,
                'duration': dur,
            }
            per_task = {
                'task_id': task_id,
                'env_name': str(env_name),
                'group': group,
                'successes': succ,
                'total_episodes': eps,
                'success_rate': rate,
                'duration': dur,
                'gpu_id': self.result_gpu_id,
            }
            with open(
                    task_result_dir / f'task{task_id}_results.json',
                    'w',
                    encoding='utf-8') as f:
                json.dump(per_task, f, indent=4)
            if manager_result_dir is not None:
                manager_result_path = (
                    manager_result_dir /
                    f'gpu{self.result_gpu_id}_task{task_id}_results.json')
                with open(manager_result_path, 'w', encoding='utf-8') as f:
                    json.dump(per_task, f, indent=4)
            total_trials += eps
            total_successes += succ
            total_time += dur
            overall_max_time = max(overall_max_time, dur)

        overall_rate = total_successes / max(total_trials, 1) * 100
        summary_csv = summary_dir / 'summary.csv'
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            group_rates = []
            for group in group_order:
                stats = group_stats[group]
                if stats['total_trials'] > 0:
                    rate = stats['total_successes'] / \
                        max(stats['total_trials'], 1) * 100
                    group_rates.append(f'{rate:.2f}')
                else:
                    group_rates.append('')
            writer.writerow([''] + group_order + ['all'])
            writer.writerow([
                'Success Rate (%)',
                *group_rates,
                f'{overall_rate:.2f}',
            ])
            writer.writerow([
                'Episodes',
                *[group_stats[group]['total_trials'] for group in group_order],
                total_trials,
            ])
            writer.writerow([
                'Successes',
                *[
                    group_stats[group]['total_successes']
                    for group in group_order
                ],
                total_successes,
            ])

        task_csv = summary_dir / 'task_success_rates.csv'
        with open(task_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Task', 'Group', 'Successes', 'Episodes', 'Success Rate (%)'])
            for env_name, stats in task_stats.items():
                writer.writerow([
                    env_name, stats['group'], stats['successes'],
                    stats['total_episodes'], f"{stats['success_rate']:.2f}"
                ])

        summary_txt = summary_dir / 'summary.txt'
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write('=== RoboCasa Evaluation Results Summary ===\n')
            for group in group_order:
                stats = group_stats[group]
                rate = (
                    stats['total_successes'] / max(stats['total_trials'], 1) *
                    100 if stats['total_trials'] > 0 else 0.0)
                f.write(f'\n{group}:\n')
                f.write(f"- Tasks completed: {stats['total_tasks']}\n")
                f.write(f"- Total attempts: {stats['total_trials']}\n")
                f.write(f'- Successful attempts: '
                        f"{stats['total_successes']}\n")
                f.write(f'- Success rate: {rate:.2f}%\n')
                f.write(f'- Total time: '
                        f"{self._format_duration(stats['total_time'])}\n")
            f.write('\nOverall statistics:\n')
            f.write(f'- Success rate: {overall_rate:.2f}%\n')
            f.write(f'- Total attempts: {total_trials}\n')
            f.write(f'- Successful attempts: {total_successes}\n')
            f.write(f'- Total time: {self._format_duration(total_time)}\n')

        cfg_filename = getattr(self.cfg, 'filename', None)
        summary_json = summary_dir / 'summary.json'
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'run_id':
                    run_id,
                    'ckpt':
                    self.ckpt_path,
                    'config':
                    cfg_filename or '',
                    'model_family':
                    self.model_family,
                    'action_order':
                    self.action_order,
                    'task_ids':
                    self._resolve_task_ids(len(self.task_list), self.task_ids),
                    'group_stats':
                    group_stats,
                    'task_stats':
                    task_stats,
                    'overall': {
                        'success_rate': overall_rate,
                        'total_episodes': total_trials,
                        'successes': total_successes,
                        'total_time': total_time,
                        'max_time': overall_max_time,
                    },
                },
                f,
                indent=4)
        return str(summary_json)

    def run(self):
        """Run the RoboCasa evaluation loop."""
        import gymnasium as gym

        from fluxvla.engines.utils.eval_utils import check_robosuite_runtime

        check_robosuite_runtime('RoboCasa simulation evaluation')

        # Trigger RoboCasa Gymnasium environment registration.
        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        num_tasks = len(self.task_list)
        task_ids = self._resolve_task_ids(num_tasks, self.task_ids)
        global_episode_ids = self._build_global_episodes(
            num_tasks, self.num_trials_per_task, task_ids)

        overwatch.info(f'Robocasa Eval: {len(task_ids)}/{num_tasks} tasks, '
                       f'{self.num_trials_per_task} trials each')
        overwatch.info(f'Model family: {self.model_family}, '
                       f'chunk_size: {self.eval_chunk_size}')
        overwatch.info(f'Task ids: {task_ids}, '
                       f'shard_strategy: {self.eval_shard_strategy}')
        overwatch.info(f'Deterministic env: {self.deterministic_env}, '
                       f'deterministic action sampling: '
                       f'{self.deterministic_action_sampling}, '
                       f'base seed: {self.seed}')

        rank = overwatch.rank()
        world_size = overwatch.world_size()
        rank_schedules = [
            self._build_local_episode_schedule(
                num_tasks,
                self.num_trials_per_task,
                schedule_rank,
                world_size,
                self.eval_shard_strategy,
                task_ids=task_ids) for schedule_rank in range(world_size)
        ]
        local_episodes = rank_schedules[rank]
        num_local_episodes = max(len(schedule) for schedule in rank_schedules)

        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = self._build_run_id(
            self.model_family, data_time, suffix=self.run_id_suffix)
        output_root = self.output_dir
        if output_root is None:
            output_root = self.result_output_dir
        self.run_dir = self._build_run_dir(
            self.ckpt_path, run_id, output_dir=output_root)
        os.makedirs(self.run_dir, exist_ok=True)
        log_filepath = os.path.join(self.run_dir, f'rank{rank}.txt')
        log_file = open(log_filepath, 'w', encoding='utf-8', buffering=1)
        log_file.write(f'Rank {rank}/{world_size}, seed={self.seed}\n')
        log_file.write(f'PYTHONHASHSEED={os.environ.get("PYTHONHASHSEED")}\n')
        log_file.write(f'Deterministic env: {self.deterministic_env}, '
                       f'deterministic action sampling: '
                       f'{self.deterministic_action_sampling}\n')
        log_file.write(f'Task ids: {task_ids}\n')
        log_file.write(f'Eval shard strategy: {self.eval_shard_strategy}\n')

        total_episodes = torch.zeros(1, device=torch.cuda.current_device())
        total_successes = torch.zeros(1, device=torch.cuda.current_device())
        task_successes = torch.zeros(
            num_tasks, device=torch.cuda.current_device())
        task_episodes = torch.zeros(
            num_tasks, device=torch.cuda.current_device())
        task_durations = torch.zeros(
            num_tasks, device=torch.cuda.current_device())

        pbar = None
        if rank == 0:
            pbar = tqdm.tqdm(
                total=len(global_episode_ids),
                desc='Robocasa Eval',
                dynamic_ncols=True)

        for idx in range(num_local_episodes):
            if idx >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[idx]
                task_id = local_id // self.num_trials_per_task
                trial_id = local_id % self.num_trials_per_task
                env_name = self.task_list[task_id]

                overwatch.info(
                    f'Task {task_id} ({env_name}), Trial {trial_id}')
                log_file.write(
                    f'Task {task_id} ({env_name}), Trial {trial_id}\n')

                # Create RoboCasa environment.
                episode_seed = self._episode_seed(local_id)
                if self.deterministic_env:
                    self._seed_python_numpy_torch(episode_seed)
                    env = gym.make(env_name, seed=episode_seed)
                    obs, info = env.reset(seed=episode_seed)
                else:
                    env = gym.make(env_name)
                    obs, info = env.reset()
                log_file.write(f'Episode seed: {episode_seed}\n')

                # In grouped mode, switch stats, denormalizer, and
                # vla.norm_stats per task.
                if self.grouped_norm_stats:
                    gname = self.norm_stats_group_names[task_id]
                    self.dataset.set_active_stats_blob(
                        self._stats_full_by_group[gname][self.unnorm_key])
                    self._active_denorm = self._denorm_by_group[gname]
                    self.vla.norm_stats = self._stats_full_by_group[gname]
                else:
                    if hasattr(self.dataset, 'set_active_stats_blob'):
                        self.dataset.set_active_stats_blob(None)
                    self._active_denorm = self.denormalize_action

                # Keep the task description identical to the training text from
                # tasks.jsonl. Do not split prefixes such as
                # "unlocked_waist: ..."; otherwise the prompt distribution
                # shifts.
                task_desc = obs.get('annotation.human.coarse_action', '')
                overwatch.info(f'Task desc: {task_desc}')
                log_file.write(f'Task desc: {task_desc}\n')

                # Evaluation loop.
                success = False
                replay_images = []
                if self.save_video:
                    frame = self._get_rollout_frame(obs)
                    if frame is not None:
                        replay_images.append(frame)
                t = 0
                episode_start = time.time()

                while t < self.max_episode_steps:
                    # Build input dict for the dataset transform pipeline.
                    obs['task_description'] = task_desc
                    batch, _ = self.dataset(obs)
                    debug_info = getattr(self.dataset, 'last_debug', {})
                    if t == 0:
                        state_arr = (
                            batch['states'].detach().float().cpu().numpy()
                            if 'states' in batch else None)
                        prompt_preview = (
                            debug_info.get('text') or debug_info.get('prompt')
                            or '')
                        overwatch.info(
                            f'Prompt preview: {prompt_preview[:240]}')
                        log_file.write(
                            f'Prompt preview: {prompt_preview[:240]}\n')
                        if state_arr is not None:
                            state_min = format(state_arr.min(), '.6g')
                            state_max = format(state_arr.max(), '.6g')
                            overwatch.info(f'State range: min={state_min}, '
                                           f'max={state_max}')
                            log_file.write(f'State range: min={state_min}, '
                                           f'max={state_max}\n')
                    batch['unnorm_key'] = self.unnorm_key

                    # Model inference.
                    if self.deterministic_action_sampling:
                        self._seed_torch(self._action_seed(local_id, t))
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            actions = self.vla.predict_action(**batch)

                    # actions shape: (1, chunk_size, max_action_dim)
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :self.eval_chunk_size, :].cpu().numpy()
                    else:
                        actions = actions[0, None, :].cpu().numpy()

                    if t == 0:
                        action_min = format(actions.min(), '.6g')
                        action_max = format(actions.max(), '.6g')
                        action_mean = format(actions.mean(), '.6g')
                        overwatch.info(f'Normalized action chunk: '
                                       f'min={action_min}, '
                                       f'max={action_max}, '
                                       f'mean={action_mean}')
                        log_file.write(f'Normalized action chunk: '
                                       f'min={action_min}, '
                                       f'max={action_max}, '
                                       f'mean={action_mean}\n')

                    # Execute one action chunk.
                    for action in actions:
                        # Denormalize from [-1, 1] to raw joint positions.
                        denorm_input = dict(
                            action=action,
                            task_suite_name=self.unnorm_key,
                        )
                        action_denormed = self._active_denorm(denorm_input)

                        if t == 0:
                            denorm_min = format(action_denormed.min(), '.6g')
                            denorm_max = format(action_denormed.max(), '.6g')
                            overwatch.info(f'Denorm action: '
                                           f'min={denorm_min}, '
                                           f'max={denorm_max}')
                            log_file.write(f'Denorm action: '
                                           f'min={denorm_min}, '
                                           f'max={denorm_max}\n')

                        # Split 29D action into RoboCasa's dict action format.
                        action_dict = {}
                        for key, (start, end) in self.action_keys.items():
                            action_dict[key] = action_denormed[start:end]

                        # Step the environment.
                        obs, reward, terminated, truncated, info = \
                            env.step(action_dict)

                        # Collect rendered frames for rollout videos.
                        if self.save_video:
                            frame = self._get_rollout_frame(obs)
                            if frame is not None:
                                replay_images.append(frame)

                        t += 1
                        if info.get('success', False):
                            success = True
                            break
                        if terminated or truncated:
                            break

                    if success or terminated or truncated:
                        break

                # Record result.
                if success:
                    total_successes += 1
                    task_successes[task_id] += 1
                total_episodes += 1
                task_episodes[task_id] += 1
                task_durations[task_id] += time.time() - episode_start

                result_str = 'SUCCESS' if success else 'FAIL'
                overwatch.info(f'  Result: {result_str} (steps={t})')
                log_file.write(f'  Result: {result_str} (steps={t})\n')

                # Save rollout video.
                if self.save_video and replay_images:
                    video_dir = os.path.join(self.run_dir, 'rollouts')
                    os.makedirs(video_dir, exist_ok=True)
                    task_short = env_name.split('/')[-1][:30]
                    video_name = (
                        f'task{task_id}_trial{trial_id}_{result_str}_'
                        f'{task_short}.mp4')
                    video_path = os.path.join(video_dir, video_name)
                    try:
                        writer = imageio.get_writer(
                            video_path,
                            fps=20,
                            codec='h264',
                            output_params=['-pix_fmt', 'yuv420p'])
                        for frame in replay_images:
                            if isinstance(frame, np.ndarray):
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).clip(0, 255).astype(
                                        np.uint8)
                                writer.append_data(frame)
                        writer.close()
                        overwatch.info(f'  Video saved: {video_path} '
                                       f'({len(replay_images)} frames)')
                    except Exception as e:
                        overwatch.warning(f'  Video save failed: {e}')

                env.close()
                step_tensor = torch.ones(1, device=torch.cuda.current_device())

            # Distributed synchronization.
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes_done = total_episodes.clone()
            global_successes_done = total_successes.clone()
            dist.all_reduce(global_episodes_done, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes_done, op=dist.ReduceOp.SUM)

            if rank == 0:
                n_ep = int(global_episodes_done[0])
                n_succ = int(global_successes_done[0])
                rate = n_succ / n_ep * 100 if n_ep > 0 else 0
                rate_text = format(rate, '.1f')
                overwatch.info(f'Progress: {n_ep} episodes, '
                               f'{n_succ} successes ({rate_text}%)')
                log_file.write(f'Progress: {n_ep} episodes, '
                               f'{n_succ} successes ({rate_text}%)\n')
                log_file.flush()

        dist.barrier()
        global_episodes = total_episodes.clone()
        global_successes = total_successes.clone()
        dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_successes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_episodes, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_durations, op=dist.ReduceOp.SUM)
        if rank == 0:
            n_ep = int(global_episodes[0].item())
            n_succ = int(global_successes[0].item())
            rate = n_succ / max(n_ep, 1) * 100
            overwatch.info(f'Robocasa final: {n_succ}/{n_ep} '
                           f'({rate:.2f}%)')
            log_file.write(f'Robocasa final: {n_succ}/{n_ep} '
                           f'({rate:.2f}%)\n')
            summary_json = self._write_robocasa_summary_artifacts(
                Path(self.run_dir),
                run_id,
                task_successes.cpu(),
                task_episodes.cpu(),
                task_durations.cpu(),
            )
            overwatch.info(f'[*] Wrote Robocasa summary to {summary_json}')
        log_file.close()
        dist.barrier()
        return
