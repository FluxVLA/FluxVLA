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
"""GR00T N1.7 native FluxVLA registration shell.

Layer 1 makes the model buildable through FluxVLA's VLAS registry. Layer 2
adds lightweight checkpoint metadata loading without instantiating the large
N1.7 model or processor.
"""

import copy
import importlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _or_policy

from fluxvla.engines import (VLAS, build_head_from_cfg,
                             build_vlm_backbone_from_cfg, initialize_overwatch)
from fluxvla.transforms.modality_state_action import \
    resolve_groot_n17_embodiment_key
from .llava_vla import LlavaVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class GrootN17VLA(LlavaVLA):
    """Native FluxVLA shell for GR00T N1.7.

    The class intentionally avoids importing official Isaac-GR00T at module
    import time. This keeps FluxVLA's existing LIBERO/RoboCasa paths importable
    while we port N1.7 layer by layer. It inherits FluxVLA's LlavaVLA base
    interface but overrides runtime assembly, forward, and prediction with the
    native N1.7 backbone/action-head contract.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        embodiment_tag: str = 'LIBERO_PANDA',
        action_horizon: int = 8,
        action_dim: Optional[int] = None,
        use_flash_attention: Optional[bool] = None,
        qwen3_runtime: str = 'hf_53',
        vlm_backbone: Optional[Dict[str, Any]] = None,
        vla_head: Optional[Dict[str, Any]] = None,
        freeze_vision_backbone: bool = True,
        freeze_llm_backbone: bool = True,
        freeze_vlm_backbone: bool = False,
        freeze_projector: bool = False,
        freeze_vla_head: bool = False,
        load_metadata: bool = True,
        norm_stats: Optional[Dict[str, Any]] = None,
        use_relative_action: Optional[bool] = None,
        apply_sincos_state_encoding: Optional[bool] = None,
        name_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        native_vlm_backbone_cfg = copy.deepcopy(vlm_backbone)
        native_vla_head_cfg = copy.deepcopy(vla_head)
        if name_mapping is None:
            name_mapping = {
                'vlm_backbone.vlm.': 'backbone.model.',
                'vla_head.': 'action_head.',
            }

        super().__init__(
            vla_head=None,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_projector=freeze_projector,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_vla_head=freeze_vla_head,
            norm_stats=norm_stats,
            name_mapping=name_mapping,
        )
        self.model_path = model_path
        self.processor_path = processor_path
        self.inline_processor_kwargs = copy.deepcopy(processor_kwargs)
        self.embodiment_tag = embodiment_tag
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.use_flash_attention = use_flash_attention
        self.qwen3_runtime = str(qwen3_runtime).lower()
        if self.qwen3_runtime not in ('hf_53', 'compat_457'):
            raise ValueError('GrootN17VLA qwen3_runtime must be "hf_53" or '
                             f'"compat_457", got {qwen3_runtime!r}.')
        if 'processor_runtime' in kwargs:
            raise ValueError(
                'GrootN17VLA no longer supports processor_runtime; use the '
                'split transform and collator pipeline.')
        self.qwen3_runtime_summary: Optional[Dict[str, Any]] = None
        self.checkpoint_use_flash_attention = None
        self.load_metadata = load_metadata
        self._native_vlm_backbone_cfg = native_vlm_backbone_cfg
        self._native_vla_head_cfg = native_vla_head_cfg
        if (self._native_vlm_backbone_cfg is None
                or self._native_vla_head_cfg is None):
            raise ValueError(
                'GrootN17VLA requires config-visible vlm_backbone and '
                'vla_head modules.')
        self.use_relative_action = use_relative_action
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.extra_cfg = dict(kwargs)
        self.num_inference_timesteps = None
        self.max_state_dim = None
        self.max_action_dim = None
        self.use_percentiles = False
        self.use_mean_std = False
        self.clip_outliers = True

        self.checkpoint_dir: Optional[Path] = None
        self.model_config: Dict[str, Any] = {}
        self.processor_config: Dict[str, Any] = {}
        self.statistics: Dict[str, Any] = {}
        self.embodiment_id_map: Dict[str, int] = {}
        self.safetensors_index: Dict[str, Any] = {}
        self.safetensors_shards = []
        self.available_modalities = []
        self.available_statistics = []
        self.active_embodiment_key = self.resolve_embodiment_key(
            embodiment_tag)

        self.all_module_keys = ['vlm_backbone', 'vla_head']

        # Placeholder so generic module utilities have a device anchor before
        # Layer 2 instantiates the real N1.7 modules.
        self._device_anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        if self.model_path is not None and self.load_metadata:
            self._load_checkpoint_metadata(Path(self.model_path))

        overwatch.info('Initialized GrootN17VLA shell: '
                       f'embodiment_tag={self.embodiment_tag}, '
                       f'model_path={self.model_path}')

    @staticmethod
    def _load_json(path: Path, required: bool = True) -> Dict[str, Any]:
        if not path.is_file():
            if required:
                raise FileNotFoundError(f'Missing GR00T N1.7 metadata: {path}')
            return {}
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _resolve_processor_config_path(self, checkpoint_dir: Path) -> Path:
        if self.processor_path is not None:
            processor_path = Path(self.processor_path).expanduser().resolve()
            if processor_path.is_dir():
                return processor_path / 'processor_config.json'
            return processor_path

        root_config = checkpoint_dir / 'processor_config.json'
        if root_config.is_file():
            return root_config
        return checkpoint_dir / 'processor' / 'processor_config.json'

    def _resolve_processor_metadata_dir(self, checkpoint_dir: Path) -> Path:
        if self.processor_path is None:
            return checkpoint_dir
        processor_path = Path(self.processor_path).expanduser().resolve()
        if processor_path.is_dir():
            return processor_path
        if processor_path.name == 'processor_config.json':
            return processor_path.parent
        return checkpoint_dir

    @classmethod
    def resolve_embodiment_key(cls,
                               embodiment_tag: Optional[str] = None,
                               env_name: Optional[str] = None) -> str:
        """Resolve names to an N1.7 modality/statistics key."""
        return resolve_groot_n17_embodiment_key(
            embodiment_tag or 'LIBERO_PANDA', env_name)

    def _load_checkpoint_metadata(self, model_path: Path) -> None:
        checkpoint_dir = model_path.expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise ValueError('GrootN17VLA Layer 2 expects model_path to be a '
                             f'checkpoint directory, got: {checkpoint_dir}')

        self.checkpoint_dir = checkpoint_dir
        processor_metadata_dir = self._resolve_processor_metadata_dir(
            checkpoint_dir)
        self.model_config = self._load_json(checkpoint_dir / 'config.json')
        if self.inline_processor_kwargs is not None:
            processor_kwargs = copy.deepcopy(self.inline_processor_kwargs)
            self.processor_config = {'processor_kwargs': processor_kwargs}
            self.statistics = copy.deepcopy(
                processor_kwargs.get('statistics', {}))
            self.embodiment_id_map = copy.deepcopy(
                processor_kwargs.get('embodiment_id_mapping', {}))
        else:
            self.processor_config = self._load_json(
                self._resolve_processor_config_path(checkpoint_dir))
            processor_statistics_path = (
                processor_metadata_dir / 'statistics.json')
            checkpoint_statistics_path = checkpoint_dir / 'statistics.json'
            if (self.processor_path is not None
                    and processor_statistics_path.is_file()):
                statistics_path = processor_statistics_path
            else:
                statistics_path = checkpoint_statistics_path
            if not statistics_path.is_file():
                statistics_path = processor_statistics_path
            self.statistics = self._load_json(statistics_path)
            processor_embodiment_path = (
                processor_metadata_dir / 'embodiment_id.json')
            checkpoint_embodiment_path = checkpoint_dir / 'embodiment_id.json'
            if (self.processor_path is not None
                    and processor_embodiment_path.is_file()):
                embodiment_path = processor_embodiment_path
            else:
                embodiment_path = checkpoint_embodiment_path
            if not embodiment_path.is_file():
                embodiment_path = processor_embodiment_path
            self.embodiment_id_map = self._load_json(
                embodiment_path, required=False)
        self.safetensors_index = self._load_json(
            checkpoint_dir / 'model.safetensors.index.json', required=False)

        weight_map = self.safetensors_index.get('weight_map', {})
        self.safetensors_shards = sorted(set(weight_map.values()))
        processor_kwargs = self.processor_config.get('processor_kwargs', {})
        modality_configs = processor_kwargs.get('modality_configs', {})
        self.available_modalities = sorted(modality_configs.keys())
        self.available_statistics = sorted(self.statistics.keys())
        self.active_embodiment_key = self.resolve_embodiment_key(
            self.embodiment_tag)

        self.action_horizon = int(
            self.model_config.get('action_horizon', self.action_horizon))
        self.num_inference_timesteps = self.model_config.get(
            'num_inference_timesteps')
        self.action_dim = self.model_config.get('max_action_dim',
                                                self.action_dim)
        self.max_state_dim = self.model_config.get('max_state_dim')
        self.max_action_dim = self.model_config.get('max_action_dim')
        self.checkpoint_use_flash_attention = self.model_config.get(
            'use_flash_attention')
        if self.use_flash_attention is None:
            self.use_flash_attention = bool(
                self.checkpoint_use_flash_attention)
        processor_kwargs = self.processor_config.get('processor_kwargs', {})
        if self.use_relative_action is None:
            self.use_relative_action = bool(
                processor_kwargs.get('use_relative_action', False))
        if self.apply_sincos_state_encoding is None:
            self.apply_sincos_state_encoding = bool(
                processor_kwargs.get(
                    'apply_sincos_state_encoding',
                    self.model_config.get('apply_sincos_state_encoding',
                                          False)))
        self.use_percentiles = bool(
            processor_kwargs.get(
                'use_percentiles',
                self.model_config.get('use_percentiles', False)))
        self.use_mean_std = bool(
            processor_kwargs.get('use_mean_std',
                                 self.model_config.get('use_mean_std', False)))
        self.clip_outliers = bool(processor_kwargs.get('clip_outliers', True))

    def _apply_qwen3_runtime(
        self,
        patch_gr00t_backbone: bool = True,
    ) -> Dict[str, Any]:
        """Apply the selected Qwen3-VL runtime compatibility layer."""
        compat = importlib.import_module(
            'fluxvla.models.compat.qwen3vl_457_compat')
        apply_runtime = getattr(compat, 'apply_qwen3vl_runtime')
        self.qwen3_runtime_summary = apply_runtime(
            self.qwen3_runtime,
            patch_gr00t_backbone=patch_gr00t_backbone,
        )
        return self.qwen3_runtime_summary

    @staticmethod
    def _namespace_from_dict(data: Dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**dict(data))

    def _native_n17_config(self) -> SimpleNamespace:
        cfg = dict(self.model_config)
        diffusion_cfg = dict(cfg.get('diffusion_model_cfg') or {})
        input_embedding_dim = cfg.get('input_embedding_dim')
        if input_embedding_dim is None:
            input_embedding_dim = (
                int(diffusion_cfg.get('num_attention_heads', 32)) *
                int(diffusion_cfg.get('attention_head_dim', 48)))
        defaults = {
            'hidden_size': cfg.get('hidden_size', 1024),
            'input_embedding_dim': input_embedding_dim,
            'backbone_embedding_dim': cfg.get('backbone_embedding_dim', 2048),
            'max_action_dim': cfg.get('max_action_dim', self.max_action_dim),
            'max_state_dim': cfg.get('max_state_dim', self.max_state_dim),
            'action_horizon': cfg.get('action_horizon', self.action_horizon),
            'state_history_length': cfg.get('state_history_length', 1),
            'num_inference_timesteps': cfg.get('num_inference_timesteps', 4),
            'max_num_embodiments': cfg.get('max_num_embodiments', 32),
            'use_alternate_vl_dit': cfg.get('use_alternate_vl_dit', True),
            'attend_text_every_n_blocks': cfg.get('attend_text_every_n_blocks',
                                                  2),
            'use_vlln': cfg.get('use_vlln', True),
            'vl_self_attention_cfg': cfg.get('vl_self_attention_cfg'),
            'add_pos_embed': cfg.get('add_pos_embed', True),
            'max_seq_len': cfg.get('max_seq_len', 1024),
            'state_dropout_prob': cfg.get('state_dropout_prob', 0.0),
            'noise_beta_alpha': cfg.get('noise_beta_alpha', 1.5),
            'noise_beta_beta': cfg.get('noise_beta_beta', 1.0),
            'noise_s': cfg.get('noise_s', 0.999),
            'num_timestep_buckets': cfg.get('num_timestep_buckets', 1000),
            'diffusion_model_cfg': diffusion_cfg,
            'select_layer': cfg.get('select_layer', 16),
            'reproject_vision': cfg.get('reproject_vision', True),
            'use_flash_attention': bool(self.use_flash_attention),
            'load_bf16': cfg.get('load_bf16', False),
        }
        for key in (
                'state_dropout_prob',
                'reproject_vision',
                'load_bf16',
        ):
            if key in self.extra_cfg and self.extra_cfg[key] is not None:
                defaults[key] = self.extra_cfg[key]
        cfg.update(defaults)
        return self._namespace_from_dict(cfg)

    def _load_prefixed_state_dict(
        self,
        checkpoint_prefix: str,
        module_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        if self.checkpoint_dir is None:
            raise ValueError('Native runtime requires model_path metadata.')
        weight_map = self.safetensors_index.get('weight_map', {})
        keys = [key for key in weight_map if key.startswith(checkpoint_prefix)]
        if not keys:
            raise KeyError('No checkpoint weights found for prefix '
                           f'{checkpoint_prefix!r}')
        shards = sorted({weight_map[key] for key in keys})
        safetensors_torch = importlib.import_module('safetensors.torch')
        load_file = getattr(safetensors_torch, 'load_file')
        state_dict = {}
        wanted = set(keys)
        for shard in shards:
            tensors = load_file(str(self.checkpoint_dir / shard), device='cpu')
            for key, value in tensors.items():
                if key in wanted:
                    state_dict[key] = value
        if len(state_dict) != len(keys):
            missing = sorted(wanted - set(state_dict))
            raise KeyError(
                f'Missing {len(missing)} tensors for {checkpoint_prefix!r}: '
                f'{missing[:5]}')
        if checkpoint_prefix == 'backbone.':
            lm_head_key = 'backbone.model.lm_head.weight'
            embed_key = (
                'backbone.model.model.language_model.embed_tokens.weight')
            if lm_head_key not in state_dict and embed_key in state_dict:
                state_dict[lm_head_key] = state_dict[embed_key]

        mapping = [(target, source)
                   for target, source in (self.name_mapping or {}).items()
                   if target.startswith(module_prefix)
                   and source.startswith(checkpoint_prefix)]
        if not mapping:
            raise KeyError('No name_mapping entry connects checkpoint prefix '
                           f'{checkpoint_prefix!r} to module prefix '
                           f'{module_prefix!r}.')
        remapped = {}
        for source_name, value in state_dict.items():
            matches = [(target, source) for target, source in mapping
                       if source_name.startswith(source)]
            if len(matches) != 1:
                raise KeyError(
                    f'Expected one name_mapping match for {source_name!r}, '
                    f'got {matches!r}.')
            target, source = matches[0]
            target_name = target + source_name[len(source):]
            remapped[target_name[len(module_prefix):]] = value
        return remapped

    def _ensure_native_runtime(self) -> Dict[str, Any]:
        """Load the FluxVLA-native backbone and action head."""
        backbone, action_head = self.vlm_backbone, self.vla_head
        if backbone is not None and action_head is not None:
            return {
                'status': 'already_loaded',
                'checkpoint_dir': str(self.checkpoint_dir),
                'all_module_keys': list(self.all_module_keys or []),
            }
        if self.checkpoint_dir is None:
            raise ValueError('Native runtime requires model_path metadata.')
        self._apply_qwen3_runtime(patch_gr00t_backbone=False)

        config = self._native_n17_config()
        backbone = build_vlm_backbone_from_cfg(
            copy.deepcopy(self._native_vlm_backbone_cfg),
            default_args={
                'select_layer': config.select_layer,
                'reproject_vision': config.reproject_vision,
                'use_flash_attention': config.use_flash_attention,
                'load_bf16': False,
                'qwen3_runtime': self.qwen3_runtime,
            })
        backbone_state_dict = self._load_prefixed_state_dict(
            'backbone.', 'vlm_backbone.')
        backbone_load = backbone.load_state_dict(
            backbone_state_dict,
            strict=True,
            assign=True,
        )
        backbone.finalize_checkpoint_load()
        backbone.eval()

        action_head = build_head_from_cfg(
            copy.deepcopy(self._native_vla_head_cfg),
            default_args={'config': config})
        action_head_load = action_head.load_state_dict(
            self._load_prefixed_state_dict('action_head.', 'vla_head.'),
            strict=True,
        )
        action_head.eval()

        self.vlm_backbone = backbone
        self.vla_head = action_head
        return {
            'status': 'ok',
            'checkpoint_dir': str(self.checkpoint_dir),
            'all_module_keys': list(self.all_module_keys),
            'qwen3_runtime': self.qwen3_runtime,
            'qwen3_runtime_summary': self.qwen3_runtime_summary,
            'native_backbone': {
                'class': type(backbone).__name__,
                'attr': 'vlm_backbone',
                'missing_keys': list(backbone_load.missing_keys),
                'unexpected_keys': list(backbone_load.unexpected_keys),
            },
            'native_action_head': {
                'class': type(action_head).__name__,
                'attr': 'vla_head',
                'missing_keys': list(action_head_load.missing_keys),
                'unexpected_keys': list(action_head_load.unexpected_keys),
            },
        }

    @staticmethod
    def _move_batch_to_device_dtype(batch: Dict[str,
                                                Any], device: torch.device,
                                    dtype: torch.dtype) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    moved[key] = value.to(device=device, dtype=dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        return moved

    def _prepare_native_eval_runtime(
        self,
        dtype: str,
    ) -> tuple[torch.device, torch.dtype]:
        self._ensure_native_runtime()
        backbone, action_head = self.vlm_backbone, self.vla_head
        if backbone is None or action_head is None:
            raise RuntimeError('Failed to load native N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        target_dtype = getattr(torch, dtype)
        backbone.to(device=target_device, dtype=target_dtype)
        action_head.to(device=target_device, dtype=target_dtype)
        backbone.eval()
        action_head.eval()
        return target_device, target_dtype

    @staticmethod
    def _extract_predict_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = batch.get('inputs', batch)
        if hasattr(inputs, 'data') and isinstance(inputs.data, dict):
            return inputs.data
        return inputs

    @staticmethod
    def _has_fluxvla_action_inputs(inputs: Dict[str, Any]) -> bool:
        required = {
            'lang_tokens',
            'lang_masks',
            'images',
            'image_grid_thw',
            'states',
            'embodiment_ids',
        }
        return required.issubset(inputs.keys())

    @staticmethod
    def _normalize_predict_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one-sample eval tensors to the batched train contract."""
        normalized = dict(inputs)
        for key in (
                'lang_tokens',
                'lang_masks',
                'images',
                'image_grid_thw',
                'states',
                'embodiment_ids',
        ):
            if key not in normalized:
                continue
            value = normalized[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if key in ('lang_tokens', 'lang_masks') and value.ndim == 1:
                value = value.unsqueeze(0)
            elif key == 'states':
                if value.ndim == 1:
                    value = value.unsqueeze(0).unsqueeze(0)
                elif value.ndim == 2:
                    value = value.unsqueeze(0)
            elif key == 'embodiment_ids':
                if value.ndim == 0:
                    value = value.reshape(1)
                elif value.ndim > 1:
                    value = value.reshape(-1)
            normalized[key] = value
        return normalized

    def _predict_n17_action(
        self,
        inputs: Dict[str, Any],
        dtype: str = 'bfloat16',
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Run N1.7 inference through the shared LlavaVLA contract."""
        device, torch_dtype = self._prepare_native_eval_runtime(dtype)
        model_inputs = {
            key: value
            for key, value in inputs.items() if key in {
                'lang_tokens',
                'lang_masks',
                'images',
                'image_grid_thw',
                'states',
                'embodiment_ids',
            }
        }
        model_inputs = self._normalize_predict_inputs(model_inputs)
        model_inputs = self._move_batch_to_device_dtype(
            model_inputs, device, torch_dtype)
        with torch.inference_mode():
            return super().predict_action(**model_inputs, seed=seed)

    @staticmethod
    def _module_has_trainable_parameters(module: Optional[nn.Module]) -> bool:
        return module is not None and any(param.requires_grad
                                          for param in module.parameters())

    def from_pretrained(self):
        """Runner-facing loader hook.

        Native module construction adapts the official sharded source weights;
        runner checkpoints are handled separately by ``load_state_dict``.
        """
        self._ensure_native_runtime()
        # DDP invokes this hook after its initial freeze pass, when the lazy
        # native modules do not exist yet. Reapply the common VLA policy now
        # that the backbone and head have been materialized.
        self.freeze_backbones()
        return self

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for loaded native N1.7 modules."""
        policies = []
        backbone, action_head = self.vlm_backbone, self.vla_head
        if (self._module_has_trainable_parameters(backbone)
                and hasattr(backbone, 'get_fsdp_wrapping_policy')):
            policies.append(backbone.get_fsdp_wrapping_policy())
        if (self._module_has_trainable_parameters(action_head)
                and hasattr(action_head, 'get_fsdp_wrapping_policy')):
            policies.append(action_head.get_fsdp_wrapping_policy())
        if policies:
            return partial(_or_policy, policies=policies)

        def _no_wrap_policy(module, recurse, nonwrapped_numel):
            del module, recurse, nonwrapped_numel
            return False

        return _no_wrap_policy

    @staticmethod
    def _has_state_prefix(state_dict, prefix: str) -> bool:
        return any(key.startswith(prefix) for key in state_dict.keys())

    @classmethod
    def _remap_state_prefix_if_needed(cls, state_dict, source: str,
                                      target: str):
        if (not cls._has_state_prefix(state_dict, source)
                or cls._has_state_prefix(state_dict, target)):
            return state_dict
        return {
            (target + key[len(source):] if key.startswith(source) else key):
            value
            for key, value in state_dict.items()
        }

    def _remap_native_state_dict_keys(self, state_dict):
        """Support old N1.7 checkpoint prefixes."""
        state_dict = self._remap_state_prefix_if_needed(
            state_dict, 'n17_backbone.', 'vlm_backbone.')
        state_dict = self._remap_state_prefix_if_needed(
            state_dict, 'n17_action_head.', 'vla_head.')
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        self._ensure_native_runtime()
        state_dict = self._remap_native_state_dict_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _extract_forward_inputs(args, kwargs) -> Dict[str, Any]:
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise TypeError('GrootN17VLA.forward accepts either one dict '
                                'argument or keyword tensor inputs.')
            if kwargs:
                raise TypeError(
                    'GrootN17VLA.forward does not accept both args '
                    'and kwargs.')
            batch = args[0]
        else:
            batch = kwargs
        return batch.get('inputs', batch)

    def _resolve_runtime_device(self) -> torch.device:
        target_device = self._device_anchor.device
        if target_device.type == 'cpu' and torch.cuda.is_available():
            target_device = torch.device('cuda')
        return target_device

    @staticmethod
    def _infer_batch_dtype(
        inputs: Dict[str, Any],
        default: torch.dtype = torch.bfloat16,
    ) -> torch.dtype:
        for value in inputs.values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                return value.dtype
        return default

    def _prepare_native_forward_modules(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[torch.nn.Module, torch.nn.Module, torch.device, torch.dtype]:
        self._ensure_native_runtime()
        backbone, action_head = self.vlm_backbone, self.vla_head
        if backbone is None or action_head is None:
            raise RuntimeError('Failed to load native N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        dtype = self._infer_batch_dtype(inputs)
        current_device = next(iter(action_head.parameters())).device
        if current_device.type == 'cpu':
            backbone.to(device=target_device, dtype=dtype)
            action_head.to(device=target_device, dtype=dtype)
        return backbone, action_head, target_device, dtype

    def forward(self, *args, **kwargs):
        inputs = self._extract_forward_inputs(args, kwargs)
        _, _, device, dtype = self._prepare_native_forward_modules(inputs)
        inputs = self._move_batch_to_device_dtype(inputs, device, dtype)
        return super().forward(**inputs)

    def predict_action(self, **batch):
        inputs = self._extract_predict_inputs(batch)
        if self._has_fluxvla_action_inputs(inputs):
            return self._predict_n17_action(
                inputs,
                dtype=batch.get('dtype', 'bfloat16'),
                seed=batch.get('seed'),
            )
        raise NotImplementedError(
            'GrootN17VLA.predict_action expects FluxVLA inputs containing '
            'lang_tokens, lang_masks, images, image_grid_thw, states, and '
            'embodiment_ids.')
