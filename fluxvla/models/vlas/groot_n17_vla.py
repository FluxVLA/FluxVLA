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
from functools import partial
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _or_policy

from fluxvla.engines import (VLAS, build_head_from_cfg,
                             build_vlm_backbone_from_cfg,
                             initialize_overwatch)
from .llava_vla import LlavaVLA


overwatch = initialize_overwatch(__name__)


N17_EMBODIMENT_ALIASES = {
    'ROBOCASA_GR1_TABLETOP': 'robocasa_gr1_tabletop',
    'robocasa_gr1_tabletop': 'robocasa_gr1_tabletop',
    'gr1_unified': 'robocasa_gr1_tabletop',
    'LIBERO_PANDA': 'libero_sim',
    'libero_sim': 'libero_sim',
}

N17_ENV_PREFIX_TO_EMBODIMENT = {
    'gr1_unified': 'robocasa_gr1_tabletop',
    'libero_sim': 'libero_sim',
}

N17_DEFAULT_EMBODIMENT_IDS = {
    'robocasa_gr1_tabletop': 10,
    'libero_sim': 2,
}

N17_BUILTIN_MODALITY_CONFIGS = {
    'robocasa_gr1_tabletop': {
        'video': {
            'delta_indices': [0],
            'modality_keys': ['ego_view_bg_crop_pad_res256_freq20'],
        },
        'state': {
            'delta_indices': [0],
            'modality_keys': [
                'left_arm',
                'right_arm',
                'left_hand',
                'right_hand',
                'waist',
            ],
            'sin_cos_embedding_keys': [
                'left_arm',
                'right_arm',
                'left_hand',
                'right_hand',
                'waist',
            ],
        },
        'action': {
            'delta_indices': list(range(8)),
            'modality_keys': [
                'left_arm',
                'right_arm',
                'left_hand',
                'right_hand',
                'waist',
            ],
            'action_representations': [
                'RELATIVE',
                'RELATIVE',
                'RELATIVE',
                'RELATIVE',
                'ABSOLUTE',
            ],
        },
        'language': {
            'delta_indices': [0],
            'modality_keys': ['task'],
        },
    },
    'libero_sim': {
        'video': {
            'delta_indices': [0],
            'modality_keys': ['image', 'wrist_image'],
        },
        'state': {
            'delta_indices': [0],
            'modality_keys': [
                'x',
                'y',
                'z',
                'roll',
                'pitch',
                'yaw',
                'gripper',
            ],
        },
        'action': {
            'delta_indices': list(range(16)),
            'modality_keys': [
                'x',
                'y',
                'z',
                'roll',
                'pitch',
                'yaw',
                'gripper',
            ],
        },
        'language': {
            'delta_indices': [0],
            'modality_keys': ['annotation.human.action.task_description'],
        },
    },
}


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
        embodiment_tag: str = 'ROBOCASA_GR1_TABLETOP',
        model_name: str = 'nvidia/Cosmos-Reason2-2B',
        backbone_model_path: Optional[str] = None,
        official_gr00t_path: Optional[str] = None,
        action_horizon: int = 8,
        action_dim: Optional[int] = None,
        use_flash_attention: Optional[bool] = None,
        load_mode: str = 'official',
        qwen3_runtime: str = 'hf_53',
        assembly_runtime: str = 'official',
        vlm_backbone: Optional[Dict[str, Any]] = None,
        vla_head: Optional[Dict[str, Any]] = None,
        load_metadata: bool = True,
        norm_stats: Optional[Dict[str, Any]] = None,
        use_relative_action: Optional[bool] = None,
        apply_sincos_state_encoding: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(vla_head=None, norm_stats=norm_stats)
        self.model_path = model_path
        self.processor_path = processor_path
        self.embodiment_tag = embodiment_tag
        self.model_name = model_name
        self.backbone_model_path = backbone_model_path
        self.official_gr00t_path = (
            official_gr00t_path or os.environ.get('FLUXVLA_GROOT_N17_PATH'))
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.use_flash_attention = use_flash_attention
        self.load_mode = str(load_mode).lower()
        if self.load_mode not in ('official', 'native_safe'):
            raise ValueError('GrootN17VLA load_mode must be "official" or '
                             f'"native_safe", got {load_mode!r}.')
        self.qwen3_runtime = str(qwen3_runtime).lower()
        if self.qwen3_runtime not in ('hf_53', 'compat_457'):
            raise ValueError('GrootN17VLA qwen3_runtime must be "hf_53" or '
                             f'"compat_457", got {qwen3_runtime!r}.')
        if 'processor_runtime' in kwargs:
            raise ValueError(
                'GrootN17VLA no longer supports processor_runtime; use the '
                'split transform and collator pipeline.')
        self.assembly_runtime = str(assembly_runtime).lower()
        if self.assembly_runtime not in ('official', 'native'):
            raise ValueError(
                'GrootN17VLA assembly_runtime must be "official" or '
                f'"native", got {assembly_runtime!r}.')
        self.qwen3_runtime_summary: Optional[Dict[str, Any]] = None
        self.checkpoint_use_flash_attention = None
        self.load_metadata = load_metadata
        self._native_vlm_backbone_cfg = copy.deepcopy(vlm_backbone)
        self._native_vla_head_cfg = copy.deepcopy(vla_head)
        self.norm_stats = norm_stats
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

        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True
        self.llm_backbone = None
        self.vlm_backbone = None
        self.all_module_keys = ['n17_model']

        # Placeholder so generic module utilities have a device anchor before
        # Layer 2 instantiates the real N1.7 modules.
        self._device_anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        self.n17_model = None
        self.n17_backbone = None
        self.n17_action_head = None
        self.action_codec = None

        if self.model_path is not None and self.load_metadata:
            self._load_checkpoint_metadata(Path(self.model_path))

        overwatch.info(
            'Initialized GrootN17VLA shell: '
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
        """Resolve FluxVLA/official names to an N1.7 modality/statistics key."""
        if env_name:
            env_prefix = env_name.split('/', 1)[0]
            if env_prefix in N17_ENV_PREFIX_TO_EMBODIMENT:
                return N17_ENV_PREFIX_TO_EMBODIMENT[env_prefix]

        tag = embodiment_tag or 'ROBOCASA_GR1_TABLETOP'
        if tag in N17_EMBODIMENT_ALIASES:
            return N17_EMBODIMENT_ALIASES[tag]
        lower_tag = tag.lower()
        if lower_tag in N17_EMBODIMENT_ALIASES:
            return N17_EMBODIMENT_ALIASES[lower_tag]
        return lower_tag

    def _load_checkpoint_metadata(self, model_path: Path) -> None:
        checkpoint_dir = model_path.expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise ValueError('GrootN17VLA Layer 2 expects model_path to be a '
                             f'checkpoint directory, got: {checkpoint_dir}')

        self.checkpoint_dir = checkpoint_dir
        processor_metadata_dir = self._resolve_processor_metadata_dir(
            checkpoint_dir)
        self.model_config = self._load_json(checkpoint_dir / 'config.json')
        self.processor_config = self._load_json(
            self._resolve_processor_config_path(checkpoint_dir))
        processor_statistics_path = processor_metadata_dir / 'statistics.json'
        checkpoint_statistics_path = checkpoint_dir / 'statistics.json'
        if self.processor_path is not None and processor_statistics_path.is_file():
            statistics_path = processor_statistics_path
        else:
            statistics_path = checkpoint_statistics_path
        if not statistics_path.is_file():
            statistics_path = processor_statistics_path
        self.statistics = self._load_json(statistics_path)
        processor_embodiment_path = processor_metadata_dir / 'embodiment_id.json'
        checkpoint_embodiment_path = checkpoint_dir / 'embodiment_id.json'
        if self.processor_path is not None and processor_embodiment_path.is_file():
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

        if self.backbone_model_path is None:
            self.model_name = self.model_config.get('model_name',
                                                    self.model_name)
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
            self.use_flash_attention = bool(self.checkpoint_use_flash_attention)
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
                                 self.model_config.get('use_mean_std',
                                                       False)))
        self.clip_outliers = bool(processor_kwargs.get('clip_outliers', True))

    @property
    def effective_backbone_model_name(self) -> str:
        if self.backbone_model_path:
            return self.backbone_model_path
        if self._native_vlm_backbone_cfg:
            model_name = self._native_vlm_backbone_cfg.get('model_name')
            if model_name:
                return model_name
        return self.backbone_model_path or self.model_name

    def _ensure_official_gr00t_importable(self) -> None:
        if self.official_gr00t_path:
            path = str(Path(self.official_gr00t_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)

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
    def _compact_exception(exc: BaseException) -> Dict[str, str]:
        return {
            'type': type(exc).__name__,
            'message': str(exc).splitlines()[0] if str(exc) else '',
        }

    def official_load_probe(self,
                            load_model: bool = False,
                            local_files_only: bool = True,
                            trust_remote_code: bool = True,
                            load_mode: Optional[str] = None) -> Dict[str, Any]:
        """Probe official N1.7 AutoConfig/AutoModel loading.

        This method is intentionally opt-in and keeps official Isaac-GR00T
        imports out of FluxVLA module import time.
        """
        if self.checkpoint_dir is None:
            raise ValueError('official_load_probe requires model_path metadata.')
        resolved_load_mode = str(load_mode or self.load_mode).lower()
        if resolved_load_mode not in ('official', 'native_safe'):
            raise ValueError('load_mode must be "official" or "native_safe", '
                             f'got {resolved_load_mode!r}.')
        self._ensure_official_gr00t_importable()
        qwen3_runtime_summary = self._apply_qwen3_runtime()
        result = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'effective_backbone_model_name': self.effective_backbone_model_name,
            'load_mode': resolved_load_mode,
            'qwen3_runtime': self.qwen3_runtime,
            'qwen3_runtime_summary': qwen3_runtime_summary,
            'load_model_requested': load_model,
        }
        try:
            importlib.import_module('gr00t.model')
            result['official_registration'] = 'ok'
        except Exception as exc:  # pragma: no cover - smoke helper
            result['official_registration'] = self._compact_exception(exc)
            return result

        transformers = importlib.import_module('transformers')
        AutoConfig = getattr(transformers, 'AutoConfig')

        try:
            config = AutoConfig.from_pretrained(
                self.checkpoint_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            if self.backbone_model_path is not None:
                config.model_name = self.effective_backbone_model_name
            if self.use_flash_attention is not None:
                config.use_flash_attention = bool(self.use_flash_attention)
            for key in (
                    'tune_projector',
                    'tune_diffusion_model',
                    'tune_vlln',
                    'tune_llm',
                    'tune_visual',
                    'tune_top_llm_layers',
                    'reproject_vision',
                    'backbone_trainable_params_fp32',
                    'state_dropout_prob',
                    'use_relative_action',
            ):
                if key in self.extra_cfg and self.extra_cfg[key] is not None:
                    setattr(config, key, self.extra_cfg[key])
            if hasattr(config, 'load_bf16'):
                config.load_bf16 = bool(self.extra_cfg.get('load_bf16', False))
            result['auto_config'] = {
                'status': 'ok',
                'class': type(config).__name__,
                'model_type': getattr(config, 'model_type', None),
                'architectures': getattr(config, 'architectures', None),
                'model_name': getattr(config, 'model_name', None),
                'use_flash_attention': getattr(config, 'use_flash_attention',
                                               None),
                'load_bf16': getattr(config, 'load_bf16', None),
            }
        except Exception as exc:  # pragma: no cover - smoke helper
            result['auto_config'] = self._compact_exception(exc)
            return result

        if load_model:
            AutoModel = getattr(transformers, 'AutoModel')
            auto_model_ok = False

            try:
                model_kwargs = {
                    'config': config,
                    'local_files_only': local_files_only,
                    'trust_remote_code': trust_remote_code,
                    'output_loading_info': True,
                }
                if resolved_load_mode == 'native_safe':
                    model_kwargs.update({
                        'low_cpu_mem_usage': False,
                        'device_map': None,
                        'transformers_loading_kwargs': {
                            'local_files_only': local_files_only,
                            'trust_remote_code': trust_remote_code,
                            'low_cpu_mem_usage': False,
                            'device_map': None,
                        },
                    })
                model, loading_info = AutoModel.from_pretrained(
                    self.checkpoint_dir,
                    **model_kwargs,
                )
                model.eval()
                self.n17_model = model
                result['auto_model'] = {
                    'status': 'ok',
                    'class': type(model).__name__,
                    'missing_keys': len(loading_info.get('missing_keys', [])),
                    'unexpected_keys':
                    len(loading_info.get('unexpected_keys', [])),
                    'mismatched_keys':
                    len(loading_info.get('mismatched_keys', [])),
                }
                auto_model_ok = True
            except Exception as exc:  # pragma: no cover - smoke helper
                result['auto_model'] = self._compact_exception(exc)

            if not auto_model_ok and resolved_load_mode == 'native_safe':
                try:
                    Gr00tN1d7 = getattr(
                        importlib.import_module(
                            'gr00t.model.gr00t_n1d7.gr00t_n1d7'),
                        'Gr00tN1d7')
                    load_sharded_checkpoint = getattr(
                        importlib.import_module('transformers.trainer_utils'),
                        'load_sharded_checkpoint')
                    model = Gr00tN1d7(
                        config,
                        transformers_loading_kwargs={
                            'local_files_only': local_files_only,
                            'trust_remote_code': trust_remote_code,
                            'low_cpu_mem_usage': False,
                            'device_map': None,
                        },
                    )
                    load_result = load_sharded_checkpoint(
                        model,
                        str(self.checkpoint_dir),
                        strict=False,
                        prefer_safe=True,
                    )
                    model.eval()
                    self.n17_model = model
                    missing_keys = getattr(load_result, 'missing_keys', [])
                    unexpected_keys = getattr(load_result, 'unexpected_keys',
                                              [])
                    result['manual_model'] = {
                        'status': 'ok',
                        'class': type(model).__name__,
                        'missing_keys': len(missing_keys),
                        'unexpected_keys': len(unexpected_keys),
                        'missing_key_sample': list(missing_keys[:5]),
                        'unexpected_key_sample': list(unexpected_keys[:5]),
                    }
                except Exception as exc:  # pragma: no cover - smoke helper
                    result['manual_model'] = self._compact_exception(exc)

        return result

    def get_embodiment_id(self, embodiment_key: Optional[str] = None) -> int:
        key = embodiment_key or self.active_embodiment_key
        if key in self.embodiment_id_map:
            return int(self.embodiment_id_map[key])
        if key in N17_DEFAULT_EMBODIMENT_IDS:
            return N17_DEFAULT_EMBODIMENT_IDS[key]
        raise KeyError(f'No GR00T N1.7 embodiment id for {key!r}')

    def get_modality_config(self, embodiment_key: Optional[str] = None) -> Dict:
        key = embodiment_key or self.active_embodiment_key
        processor_kwargs = self.processor_config.get('processor_kwargs', {})
        modality_configs = processor_kwargs.get('modality_configs', {})
        if key in modality_configs:
            return modality_configs[key]
        if key in N17_BUILTIN_MODALITY_CONFIGS:
            return N17_BUILTIN_MODALITY_CONFIGS[key]
        raise KeyError(f'No GR00T N1.7 modality config for {key!r}')

    def get_statistics(self,
                       embodiment_key: Optional[str] = None) -> Optional[Dict]:
        key = embodiment_key or self.active_embodiment_key
        return self.statistics.get(key)

    def select_embodiment_metadata(
        self,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if embodiment_tag is None and env_name is None:
            key = self.active_embodiment_key
        else:
            key = self.resolve_embodiment_key(embodiment_tag, env_name)
        processor_kwargs = self.processor_config.get('processor_kwargs', {})
        modality_configs = processor_kwargs.get('modality_configs', {})
        has_checkpoint_modality = key in modality_configs
        modality_config = self.get_modality_config(key)
        statistics = self.get_statistics(key)
        return {
            'embodiment_key': key,
            'embodiment_id': self.get_embodiment_id(key),
            'modality_config': modality_config,
            'modality_source':
            'checkpoint' if has_checkpoint_modality else 'builtin',
            'has_modality_config': bool(modality_config),
            'has_statistics': statistics is not None,
            'statistics': statistics,
        }

    def _ensure_official_runtime(
        self,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        """Lazily load the official model runtime."""
        if self.n17_model is not None:
            return {
                'status': 'already_loaded',
                'checkpoint_dir': str(self.checkpoint_dir),
            }
        return self.official_load_probe(
            load_model=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    def _ensure_native_action_codec(self):
        """Build the metadata-only codec used by the split eval path."""
        if self.action_codec is not None:
            return self.action_codec
        if not self.processor_config or not self.statistics:
            raise ValueError('Native action codec requires processor metadata.')
        codec_module = importlib.import_module(
            'fluxvla.transforms.modality_state_action')
        codec_cls = getattr(codec_module, 'ModalityStateActionCodec')
        processor_kwargs = self.processor_config.get('processor_kwargs', {})
        codec = codec_cls(
            modality_configs=processor_kwargs.get('modality_configs', {}),
            statistics=self.statistics,
            use_percentiles=self.use_percentiles,
            clip_outliers=self.clip_outliers,
            apply_sincos_state_encoding=bool(
                self.apply_sincos_state_encoding),
            use_relative_action=bool(self.use_relative_action),
        )
        codec.eval()
        self.action_codec = codec
        return codec

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
            'model_name': self.effective_backbone_model_name,
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
            'attend_text_every_n_blocks':
            cfg.get('attend_text_every_n_blocks', 2),
            'use_vlln': cfg.get('use_vlln', True),
            'vl_self_attention_cfg': cfg.get('vl_self_attention_cfg'),
            'add_pos_embed': cfg.get('add_pos_embed', True),
            'max_seq_len': cfg.get('max_seq_len', 1024),
            'state_dropout_prob': cfg.get('state_dropout_prob', 0.0),
            'noise_beta_alpha': cfg.get('noise_beta_alpha', 1.5),
            'noise_beta_beta': cfg.get('noise_beta_beta', 1.0),
            'noise_s': cfg.get('noise_s', 0.999),
            'num_timestep_buckets': cfg.get('num_timestep_buckets', 1000),
            'tune_projector': cfg.get('tune_projector', True),
            'tune_diffusion_model': cfg.get('tune_diffusion_model', True),
            'tune_vlln': cfg.get('tune_vlln', True),
            'diffusion_model_cfg': diffusion_cfg,
            'select_layer': cfg.get('select_layer', 16),
            'tune_llm': cfg.get('tune_llm', False),
            'tune_visual': cfg.get('tune_visual', False),
            'reproject_vision': cfg.get('reproject_vision', True),
            'use_flash_attention': bool(self.use_flash_attention),
            'load_bf16': cfg.get('load_bf16', False),
            'tune_top_llm_layers': cfg.get('tune_top_llm_layers', 0),
            'backbone_trainable_params_fp32':
            cfg.get('backbone_trainable_params_fp32', False),
        }
        for key in (
                'tune_projector',
                'tune_diffusion_model',
                'tune_vlln',
                'tune_llm',
                'tune_visual',
                'tune_top_llm_layers',
                'reproject_vision',
                'load_bf16',
                'backbone_trainable_params_fp32',
        ):
            if key in self.extra_cfg and self.extra_cfg[key] is not None:
                defaults[key] = self.extra_cfg[key]
        cfg.update(defaults)
        return self._namespace_from_dict(cfg)

    def _load_prefixed_state_dict(self, prefix: str) -> Dict[str, torch.Tensor]:
        if self.checkpoint_dir is None:
            raise ValueError('Native runtime requires model_path metadata.')
        weight_map = self.safetensors_index.get('weight_map', {})
        keys = [key for key in weight_map if key.startswith(prefix)]
        if not keys:
            raise KeyError(f'No checkpoint weights found for prefix {prefix!r}')
        shards = sorted({weight_map[key] for key in keys})
        safetensors_torch = importlib.import_module('safetensors.torch')
        load_file = getattr(safetensors_torch, 'load_file')
        state_dict = {}
        prefix_len = len(prefix)
        wanted = set(keys)
        for shard in shards:
            tensors = load_file(str(self.checkpoint_dir / shard), device='cpu')
            for key, value in tensors.items():
                if key in wanted:
                    state_dict[key[prefix_len:]] = value
        if len(state_dict) != len(keys):
            missing = sorted(wanted - {prefix + key for key in state_dict})
            raise KeyError(f'Missing {len(missing)} tensors for {prefix!r}: '
                           f'{missing[:5]}')
        if prefix == 'backbone.':
            lm_head_key = 'model.lm_head.weight'
            embed_key = 'model.model.language_model.embed_tokens.weight'
            if lm_head_key not in state_dict and embed_key in state_dict:
                state_dict[lm_head_key] = state_dict[embed_key]
        return state_dict

    def _native_runtime_modules(self):
        backbone = (
            self.vlm_backbone
            if self.vlm_backbone is not None else self.n17_backbone)
        action_head = (
            self.vla_head
            if self.vla_head is not None else self.n17_action_head)
        return backbone, action_head

    def _ensure_native_runtime(
        self,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        """Load the FluxVLA-native backbone and action head."""
        backbone, action_head = self._native_runtime_modules()
        if backbone is not None and action_head is not None:
            return {
                'status': 'already_loaded',
                'checkpoint_dir': str(self.checkpoint_dir),
                'assembly_runtime': 'native',
                'all_module_keys': list(self.all_module_keys or []),
            }
        if self.checkpoint_dir is None:
            raise ValueError('Native runtime requires model_path metadata.')
        self._apply_qwen3_runtime(patch_gr00t_backbone=False)

        config = self._native_n17_config()
        backbone_attr = 'n17_backbone'
        if self._native_vlm_backbone_cfg is not None:
            backbone_attr = 'vlm_backbone'
            backbone = build_vlm_backbone_from_cfg(
                copy.deepcopy(self._native_vlm_backbone_cfg),
                default_args={
                    'model_name':
                    self.effective_backbone_model_name,
                    'tune_llm':
                    config.tune_llm,
                    'tune_visual':
                    config.tune_visual,
                    'select_layer':
                    config.select_layer,
                    'reproject_vision':
                    config.reproject_vision,
                    'use_flash_attention':
                    config.use_flash_attention,
                    'load_bf16':
                    False,
                    'tune_top_llm_layers':
                    config.tune_top_llm_layers,
                    'trainable_params_fp32':
                    config.backbone_trainable_params_fp32,
                    'transformers_loading_kwargs': {
                        'local_files_only': local_files_only,
                        'trust_remote_code': trust_remote_code,
                    },
                    'qwen3_runtime':
                    self.qwen3_runtime,
                })
        else:
            backbones = importlib.import_module(
                'fluxvla.models.backbones.vlms.groot_n17_qwen3_backbone')
            GrootN17Qwen3Backbone = getattr(backbones, 'GrootN17Qwen3Backbone')
            backbone = GrootN17Qwen3Backbone(
                model_name=self.effective_backbone_model_name,
                tune_llm=config.tune_llm,
                tune_visual=config.tune_visual,
                select_layer=config.select_layer,
                reproject_vision=config.reproject_vision,
                use_flash_attention=config.use_flash_attention,
                load_bf16=False,
                tune_top_llm_layers=config.tune_top_llm_layers,
                trainable_params_fp32=config.backbone_trainable_params_fp32,
                transformers_loading_kwargs={
                    'local_files_only': local_files_only,
                    'trust_remote_code': trust_remote_code,
                },
                qwen3_runtime=self.qwen3_runtime,
            )
        backbone_load = backbone.load_state_dict(
            self._load_prefixed_state_dict('backbone.'),
            strict=True,
        )
        backbone.eval()

        action_head_attr = 'n17_action_head'
        if self._native_vla_head_cfg is not None:
            action_head_attr = 'vla_head'
            action_head = build_head_from_cfg(
                copy.deepcopy(self._native_vla_head_cfg),
                default_args={'config': config})
        else:
            heads = importlib.import_module(
                'fluxvla.models.heads.groot_n17_action_head')
            GrootN17ActionHead = getattr(heads, 'GrootN17ActionHead')
            action_head = GrootN17ActionHead(config)
        action_head_load = action_head.load_state_dict(
            self._load_prefixed_state_dict('action_head.'),
            strict=True,
        )
        action_head.eval()

        if backbone_attr == 'vlm_backbone':
            self.vlm_backbone = backbone
            self.n17_backbone = None
        else:
            self.n17_backbone = backbone
            self.vlm_backbone = None
        if action_head_attr == 'vla_head':
            self.vla_head = action_head
            self.n17_action_head = None
        else:
            self.n17_action_head = action_head
            self.vla_head = None
        self.all_module_keys = [backbone_attr, action_head_attr]
        return {
            'status': 'ok',
            'checkpoint_dir': str(self.checkpoint_dir),
            'assembly_runtime': 'native',
            'all_module_keys': list(self.all_module_keys),
            'qwen3_runtime': self.qwen3_runtime,
            'qwen3_runtime_summary': self.qwen3_runtime_summary,
            'native_backbone': {
                'class': type(backbone).__name__,
                'attr': backbone_attr,
                'missing_keys': list(backbone_load.missing_keys),
                'unexpected_keys': list(backbone_load.unexpected_keys),
            },
            'native_action_head': {
                'class': type(action_head).__name__,
                'attr': action_head_attr,
                'missing_keys': list(action_head_load.missing_keys),
                'unexpected_keys': list(action_head_load.unexpected_keys),
            },
        }

    @staticmethod
    def _move_batch_to_device_dtype(batch: Dict[str, Any], device: torch.device,
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

    def _native_prepare_inputs(self, inputs: Dict[str, Any],
                               device: torch.device, dtype: torch.dtype):
        moved = self._move_batch_to_device_dtype(inputs, device, dtype)
        backbone_inputs = {
            key: moved[key]
            for key in ('input_ids', 'attention_mask', 'pixel_values',
                        'image_grid_thw')
        }
        return backbone_inputs, moved

    def _run_native_backbone_head(
        self,
        inputs: Dict[str, Any],
        mode: str = 'loss',
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone, action_head = self._native_runtime_modules()
        if backbone is None or action_head is None:
            raise RuntimeError('Native N1.7 runtime is not loaded.')
        device = next(iter(action_head.parameters())).device
        dtype = next(iter(action_head.parameters())).dtype
        backbone_inputs, action_inputs = self._native_prepare_inputs(
            inputs, device, dtype)
        backbone_outputs = backbone(backbone_inputs)
        if mode == 'loss':
            return action_head(
                input_features=backbone_outputs.backbone_features,
                states=action_inputs['state'],
                attention_mask=backbone_outputs.backbone_attention_mask,
                embodiment_ids=action_inputs['embodiment_id'],
                actions=action_inputs['action'],
                action_masks=action_inputs['action_mask'],
                image_mask=backbone_outputs.image_mask,
                sample_weight=action_inputs.get('sample_weight'),
            )
        if mode == 'action':
            prev_actions = (
                action_inputs['action'] if 'action' in action_inputs else None)
            return action_head.get_action(
                input_features=backbone_outputs.backbone_features,
                states=action_inputs['state'],
                attention_mask=backbone_outputs.backbone_attention_mask,
                embodiment_ids=action_inputs['embodiment_id'],
                image_mask=backbone_outputs.image_mask,
                prev_actions=prev_actions,
                options=options,
            )
        raise ValueError(f'Unsupported native backbone/head mode: {mode!r}')

    def native_get_action(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._run_native_backbone_head(inputs, mode='action')

    def native_forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._run_native_backbone_head(inputs, mode='loss')

    def _prepare_native_eval_runtime(
        self,
        dtype: str,
    ) -> tuple[torch.device, torch.dtype]:
        self._ensure_native_runtime()
        backbone, action_head = self._native_runtime_modules()
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
    def _action_for_decode(normalized_action: np.ndarray) -> np.ndarray:
        if normalized_action.ndim == 3:
            if normalized_action.shape[0] != 1:
                raise ValueError('N1.7 eval decode expects batch=1, got '
                                 f'action shape {normalized_action.shape}')
            return normalized_action[0]
        return normalized_action

    @staticmethod
    def _flatten_decoded_action(decoded: Dict[str, np.ndarray],
                                action_keys: list[str]) -> torch.Tensor:
        flat_action = np.concatenate(
            [np.asarray(decoded[key], dtype=np.float32) for key in action_keys],
            axis=-1,
        ).astype(np.float32)
        return torch.from_numpy(flat_action[None, ...])

    def _decode_n17_action_to_env_tensor(
        self,
        normalized_action: np.ndarray,
        embodiment_key: Optional[str] = None,
        raw_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Decode split-path actions through the metadata-only codec."""
        if embodiment_key is None:
            embodiment_key = self.active_embodiment_key
        codec = self._ensure_native_action_codec()
        action_for_decode = self._action_for_decode(normalized_action)
        decoded = codec.decode_action(
            action_for_decode, embodiment_key, state=raw_state)
        action_keys = codec.modality_configs[
            embodiment_key]['action']['modality_keys']
        return self._flatten_decoded_action(decoded, action_keys).to(
            device=self._device_anchor.device, dtype=torch.float32)

    @staticmethod
    def _extract_predict_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = batch.get('inputs', batch)
        if hasattr(inputs, 'data') and isinstance(inputs.data, dict):
            return inputs.data
        return inputs

    @staticmethod
    def _has_split_action_inputs(inputs: Dict[str, Any]) -> bool:
        required = {
            'input_ids',
            'attention_mask',
            'pixel_values',
            'image_grid_thw',
            'state',
            'embodiment_id',
        }
        return required.issubset(inputs.keys())

    @staticmethod
    def _normalize_split_predict_inputs(
            inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one-sample eval tensors to the batched train contract."""
        normalized = dict(inputs)
        for key in (
                'input_ids',
                'attention_mask',
                'pixel_values',
                'image_grid_thw',
                'state',
                'embodiment_id',
        ):
            if key not in normalized:
                continue
            value = normalized[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if key in ('input_ids', 'attention_mask') and value.ndim == 1:
                value = value.unsqueeze(0)
            elif key == 'state':
                if value.ndim == 1:
                    value = value.unsqueeze(0).unsqueeze(0)
                elif value.ndim == 2:
                    value = value.unsqueeze(0)
            elif key == 'embodiment_id':
                if value.ndim == 0:
                    value = value.reshape(1)
                elif value.ndim > 1:
                    value = value.reshape(-1)
            normalized[key] = value
        return normalized

    def _predict_n17_action_from_split_inputs(
        self,
        inputs: Dict[str, Any],
        dtype: str = 'bfloat16',
        raw_state: Optional[Dict[str, Any]] = None,
        embodiment_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Run N1.7 inference from split tensor inputs and decode actions."""
        if self.assembly_runtime != 'native':
            raise NotImplementedError(
                'Split N1.7 predict_action is only supported for native '
                f'assembly_runtime, got {self.assembly_runtime!r}.')
        self._prepare_native_eval_runtime(dtype)
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {
                'input_ids',
                'attention_mask',
                'pixel_values',
                'image_grid_thw',
                'state',
                'embodiment_id',
            }
        }
        model_inputs = self._normalize_split_predict_inputs(model_inputs)
        with torch.inference_mode():
            model_pred = self._run_native_backbone_head(
                model_inputs, mode='action')
        normalized_action = model_pred['action_pred'].float().cpu().numpy()
        return self._decode_n17_action_to_env_tensor(
            normalized_action,
            embodiment_key=embodiment_key,
            raw_state=raw_state)

    def freeze_backbones(self) -> None:
        """Freeze native N1.7 backbone modules when they are loaded."""
        backbone, _ = self._native_runtime_modules()
        if backbone is not None:
            backbone.requires_grad_(False)
            backbone.eval()
        elif self.n17_model is not None:
            self.n17_model.backbone.requires_grad_(False)
            self.n17_model.backbone.eval()
        else:
            overwatch.info('GrootN17VLA runtime is not loaded yet.')

    @staticmethod
    def _module_has_trainable_parameters(module: Optional[nn.Module]) -> bool:
        return module is not None and any(
            param.requires_grad for param in module.parameters())

    def from_pretrained(self):
        """Runner-facing loader hook.

        Checkpoint loading is handled by the native/official runtime loaders
        because N1.7 checkpoints are sharded safetensors with nested prefixes.
        """
        if self.assembly_runtime == 'native':
            self._ensure_native_runtime()
            self.freeze_backbones()
        else:
            self._ensure_official_runtime()
        return self

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for loaded native N1.7 modules."""
        policies = []
        backbone, action_head = self._native_runtime_modules()
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
        """Support old N1.7 checkpoint prefixes after config-visible modules."""
        backbone, action_head = self._native_runtime_modules()
        if backbone is not None:
            if self.vlm_backbone is backbone:
                state_dict = self._remap_state_prefix_if_needed(
                    state_dict, 'n17_backbone.', 'vlm_backbone.')
            elif self.n17_backbone is backbone:
                state_dict = self._remap_state_prefix_if_needed(
                    state_dict, 'vlm_backbone.', 'n17_backbone.')
        if action_head is not None:
            if self.vla_head is action_head:
                state_dict = self._remap_state_prefix_if_needed(
                    state_dict, 'n17_action_head.', 'vla_head.')
            elif self.n17_action_head is action_head:
                state_dict = self._remap_state_prefix_if_needed(
                    state_dict, 'vla_head.', 'n17_action_head.')
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.assembly_runtime == 'native':
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
                raise TypeError('GrootN17VLA.forward does not accept both args '
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
        backbone, action_head = self._native_runtime_modules()
        if backbone is None or action_head is None:
            raise RuntimeError('Failed to load native N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        dtype = self._infer_batch_dtype(inputs)
        current_device = next(iter(action_head.parameters())).device
        if current_device.type == 'cpu':
            backbone.to(device=target_device, dtype=dtype)
            action_head.to(device=target_device, dtype=dtype)
        return backbone, action_head, target_device, dtype

    def _prepare_official_forward_model(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
        self._ensure_official_runtime()
        if self.n17_model is None:
            raise RuntimeError('Failed to load official N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        dtype = self._infer_batch_dtype(inputs)
        self.n17_model.to(device=target_device, dtype=dtype)
        self.n17_model.eval()
        return self.n17_model, target_device, dtype

    def forward(self, *args, **kwargs):
        inputs = self._extract_forward_inputs(args, kwargs)

        if self.assembly_runtime == 'native':
            self._prepare_native_forward_modules(inputs)
            return self.native_forward(inputs)

        n17_model, target_device, dtype = self._prepare_official_forward_model(
            inputs)
        moved = self._move_batch_to_device_dtype(inputs, target_device, dtype)
        return n17_model(moved)

    def predict_action(self, **batch):
        inputs = self._extract_predict_inputs(batch)
        if self._has_split_action_inputs(inputs):
            return self._predict_n17_action_from_split_inputs(
                inputs,
                dtype=batch.get('dtype', 'bfloat16'),
                raw_state=batch.get('n17_raw_state', batch.get('raw_state')),
                embodiment_key=batch.get('n17_embodiment_key',
                                         batch.get('embodiment_key')),
            )
        raise NotImplementedError(
            'GrootN17VLA.predict_action expects split tensor inputs containing '
            'input_ids, attention_mask, pixel_values, '
            'image_grid_thw, state, and embodiment_id.')
