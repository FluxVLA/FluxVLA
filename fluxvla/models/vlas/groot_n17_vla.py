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
    native N1.7 processor/backbone/action-head contract.
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
        processor_runtime: str = 'official',
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
        self.processor_runtime = str(processor_runtime).lower()
        if self.processor_runtime not in ('official', 'native'):
            raise ValueError(
                'GrootN17VLA processor_runtime must be "official" or '
                f'"native", got {processor_runtime!r}.')
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
        self.processor = None

        if self.model_path is not None and self.load_metadata:
            self._load_checkpoint_metadata(Path(self.model_path))

        overwatch.info(
            'Initialized GrootN17VLA shell: '
            f'embodiment_tag={self.embodiment_tag}, '
            f'processor_runtime={self.processor_runtime}, '
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

    def metadata_summary(self) -> Dict[str, Any]:
        """Return lightweight checkpoint metadata for smoke tests."""
        active = self.select_embodiment_metadata()
        return {
            'checkpoint_dir':
            str(self.checkpoint_dir) if self.checkpoint_dir else None,
            'model_type':
            self.model_config.get('model_type'),
            'architecture':
            self.model_config.get('architectures', []),
            'model_name':
            self.model_name,
            'effective_backbone_model_name':
            self.effective_backbone_model_name,
            'action_horizon':
            self.action_horizon,
            'num_inference_timesteps':
            self.num_inference_timesteps,
            'action_dim':
            self.action_dim,
            'max_state_dim':
            self.max_state_dim,
            'max_action_dim':
            self.max_action_dim,
            'use_relative_action':
            self.use_relative_action,
            'apply_sincos_state_encoding':
            self.apply_sincos_state_encoding,
            'use_percentiles':
            self.use_percentiles,
            'use_mean_std':
            self.use_mean_std,
            'clip_outliers':
            self.clip_outliers,
            'use_flash_attention':
            self.use_flash_attention,
            'checkpoint_use_flash_attention':
            self.checkpoint_use_flash_attention,
            'load_mode':
            self.load_mode,
            'qwen3_runtime':
            self.qwen3_runtime,
            'processor_runtime':
            self.processor_runtime,
            'assembly_runtime':
            self.assembly_runtime,
            'qwen3_runtime_summary':
            self.qwen3_runtime_summary,
            'num_safetensors_shards':
            len(self.safetensors_shards),
            'num_weight_tensors':
            len(self.safetensors_index.get('weight_map', {})),
            'has_processor_config':
            bool(self.processor_config),
            'num_modalities':
            len(self.available_modalities),
            'num_statistics':
            len(self.available_statistics),
            'active_embodiment_key':
            self.active_embodiment_key,
            'embodiment_id':
            active['embodiment_id'],
            'has_active_modality_config':
            active['has_modality_config'],
            'active_modality_source':
            active['modality_source'],
            'has_active_statistics':
            active['has_statistics'],
        }

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
                            load_processor: bool = True,
                            load_model: bool = False,
                            local_files_only: bool = True,
                            trust_remote_code: bool = True,
                            load_mode: Optional[str] = None) -> Dict[str, Any]:
        """Probe official N1.7 AutoConfig/AutoProcessor/AutoModel loading.

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
            'load_processor_requested': load_processor,
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

        if load_processor:
            AutoProcessor = getattr(transformers, 'AutoProcessor')
            processor_kwargs = {}
            if self.backbone_model_path is not None:
                processor_kwargs['model_name'] = self.effective_backbone_model_name

            try:
                processor = AutoProcessor.from_pretrained(
                    self.checkpoint_dir,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    **processor_kwargs,
                )
                processor.eval()
                self.processor = processor
                result['auto_processor'] = {
                    'status': 'ok',
                    'class': type(processor).__name__,
                    'model_name': getattr(processor, 'model_name', None),
                    'max_state_dim': getattr(processor, 'max_state_dim', None),
                    'max_action_dim': getattr(processor, 'max_action_dim',
                                              None),
                }
            except Exception as exc:  # pragma: no cover - smoke helper
                result['auto_processor'] = self._compact_exception(exc)

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

    def modality_summary(self,
                         embodiment_tag: Optional[str] = None,
                         env_name: Optional[str] = None) -> Dict[str, Any]:
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        summary = {
            'embodiment_key': selected['embodiment_key'],
            'embodiment_id': selected['embodiment_id'],
            'modality_source': selected['modality_source'],
            'has_statistics': selected['has_statistics'],
        }
        for modality in ('video', 'state', 'action', 'language'):
            cfg = modality_config.get(modality, {})
            summary[modality] = {
                'delta_indices': cfg.get('delta_indices'),
                'modality_keys': cfg.get('modality_keys'),
            }
        return summary

    @staticmethod
    def _stat_dim(stat: Optional[Dict[str, Any]]) -> Optional[int]:
        if not stat:
            return None
        for field in ('q01', 'q99', 'min', 'max', 'mean', 'std'):
            value = stat.get(field)
            if isinstance(value, list):
                arr = np.asarray(value)
                if arr.ndim == 1:
                    return int(arr.shape[0])
                if arr.ndim >= 2:
                    return int(arr.shape[-1])
        return None

    @staticmethod
    def _fallback_key_dim(key: str) -> int:
        # Used only for metadata-only smoke when a checkpoint lacks statistics.
        # Current built-in LIBERO fields are all scalar.
        del key
        return 1

    @staticmethod
    def _modality_keys(modality_config: Dict[str, Any],
                       modality: str) -> list:
        return list(modality_config.get(modality, {}).get('modality_keys')
                    or [])

    @staticmethod
    def _modality_steps(modality_config: Dict[str, Any],
                        modality: str) -> int:
        return len(modality_config.get(modality, {}).get('delta_indices') or [])

    def _relative_action_keys(self, modality_config: Dict[str, Any]) -> set:
        action_cfg = modality_config.get('action', {})
        action_keys = self._modality_keys(modality_config, 'action')
        reps = action_cfg.get('action_representations')
        if reps is None:
            reps = [
                (cfg or {}).get('rep')
                for cfg in (action_cfg.get('action_configs') or [])
            ]
        return {
            key
            for key, rep in zip(action_keys, reps)
            if str(rep).upper() == 'RELATIVE'
        }

    def _action_config_for_key(self, modality_config: Dict[str, Any],
                               key: str) -> Dict[str, Any]:
        action_cfg = modality_config.get('action', {})
        action_keys = self._modality_keys(modality_config, 'action')
        action_configs = action_cfg.get('action_configs') or []
        if key in action_keys:
            idx = action_keys.index(key)
        else:
            raise KeyError(f'Action key {key!r} is not in modality config')
        if idx < len(action_configs):
            return dict(action_configs[idx] or {})
        reps = action_cfg.get('action_representations') or []
        rep = reps[idx] if idx < len(reps) else 'ABSOLUTE'
        return {
            'rep': rep,
            'type': 'NON_EEF',
            'format': 'DEFAULT',
            'state_key': key,
        }

    def _stat_for_key(self, statistics: Optional[Dict[str, Any]],
                      modality: str, key: str,
                      relative_action_keys: set) -> Optional[Dict[str, Any]]:
        if not statistics:
            return None
        if (modality == 'action' and self.use_relative_action
                and key in relative_action_keys
                and key in statistics.get('relative_action', {})):
            stat = dict(statistics['relative_action'][key])
            # Official N1.7 replaces relative-action norm params with the
            # raw relative_action dict, and normalize_values_minmax consumes
            # its min/max fields even when processor use_percentiles=True.
            # Mirror that behavior in the lightweight helper path.
            if 'min' in stat and 'max' in stat:
                stat['q01'] = stat['min']
                stat['q99'] = stat['max']
            return stat
        return statistics.get(modality, {}).get(key)

    def _key_layout(self, modality_config: Dict[str, Any],
                    statistics: Optional[Dict[str, Any]],
                    modality: str) -> Dict[str, Any]:
        cursor = 0
        entries = []
        relative_action_keys = self._relative_action_keys(modality_config)
        sincos_keys = set()
        if modality == 'state' and self.apply_sincos_state_encoding:
            sincos_keys = set(
                modality_config.get('state',
                                    {}).get('sin_cos_embedding_keys') or [])

        for key in self._modality_keys(modality_config, modality):
            stat = self._stat_for_key(statistics, modality, key,
                                      relative_action_keys)
            raw_dim = self._stat_dim(stat)
            has_stats = raw_dim is not None
            if raw_dim is None:
                raw_dim = self._fallback_key_dim(key)
            processed_dim = raw_dim
            if modality == 'state' and key in sincos_keys:
                processed_dim = raw_dim * 2
            entries.append({
                'key': key,
                'start': cursor,
                'end': cursor + processed_dim,
                'raw_dim': raw_dim,
                'processed_dim': processed_dim,
                'has_stats': has_stats,
                'uses_relative_action_stats':
                modality == 'action' and key in relative_action_keys
                and bool(statistics)
                and key in statistics.get('relative_action', {}),
                'uses_sincos': modality == 'state' and key in sincos_keys,
            })
            cursor += processed_dim
        return {
            'total_dim': cursor,
            'entries': entries,
        }

    @staticmethod
    def _padding_mask(total_dim: int, max_dim: Optional[int]) -> list:
        padded_dim = int(max_dim or total_dim)
        if total_dim > padded_dim:
            raise ValueError(
                f'N1.7 layout dim {total_dim} exceeds max_dim={padded_dim}')
        return [True] * total_dim + [False] * (padded_dim - total_dim)

    def _normalization_bounds(self, stat: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.use_mean_std:
            mean = np.asarray(stat['mean'], dtype=np.float32)
            std = np.asarray(stat['std'], dtype=np.float32)
            std = np.maximum(std, 1e-8)
            return {'mean': mean, 'std': std}
        if self.use_percentiles and 'q01' in stat and 'q99' in stat:
            low = np.asarray(stat['q01'], dtype=np.float32)
            high = np.asarray(stat['q99'], dtype=np.float32)
        else:
            low = np.asarray(stat['min'], dtype=np.float32)
            high = np.asarray(stat['max'], dtype=np.float32)
        high = np.maximum(high, low + 1e-8)
        return {'min': low, 'max': high}

    def _normalize_values(self, values: np.ndarray,
                          stat: Optional[Dict[str, Any]]) -> np.ndarray:
        if stat is None:
            return values.astype(np.float32)
        params = self._normalization_bounds(stat)
        if self.use_mean_std:
            normalized = (values - params['mean']) / params['std']
        else:
            normalized = (
                (values - params['min']) / (params['max'] - params['min']))
            normalized = 2.0 * normalized - 1.0
        if self.clip_outliers:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized.astype(np.float32)

    def _unnormalize_values(self, values: np.ndarray,
                            stat: Optional[Dict[str, Any]]) -> np.ndarray:
        if stat is None:
            return values.astype(np.float32)
        params = self._normalization_bounds(stat)
        if self.use_mean_std:
            unnormalized = values * params['std'] + params['mean']
        else:
            clipped = np.clip(values, -1.0, 1.0)
            unnormalized = (
                (clipped + 1.0) / 2.0 *
                (params['max'] - params['min']) + params['min'])
        return unnormalized.astype(np.float32)

    @staticmethod
    def _synthetic_values_from_stat(stat: Optional[Dict[str, Any]],
                                    steps: int, dim: int) -> np.ndarray:
        if stat is not None and isinstance(stat.get('mean'), list):
            base = np.asarray(stat['mean'], dtype=np.float32)
        else:
            base = np.zeros(dim, dtype=np.float32)
        if base.ndim == 1:
            if base.shape[-1] != dim:
                raise ValueError(f'Synthetic stat dim mismatch: '
                                 f'got {base.shape}, expected dim={dim}')
            return np.repeat(base[None, :], steps, axis=0)
        if base.ndim == 2:
            if base.shape[-1] != dim:
                raise ValueError(f'Synthetic stat dim mismatch: '
                                 f'got {base.shape}, expected dim={dim}')
            if base.shape[0] == steps:
                return base
            if base.shape[0] == 1:
                return np.repeat(base, steps, axis=0)
            raise ValueError(f'Synthetic stat dim mismatch: got {base.shape}, '
                             f'expected steps={steps}, dim={dim}')
        raise ValueError(f'Unsupported synthetic stat shape: {base.shape}')

    @staticmethod
    def _apply_sincos(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.sin(values), np.cos(values)], axis=-1)

    @staticmethod
    def _coerce_sequence_values(values: Any, steps: int, dim: int,
                                name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[-1] != dim:
                raise ValueError(f'{name} dim mismatch: got {arr.shape}, '
                                 f'expected dim={dim}')
            return np.repeat(arr[None, :], steps, axis=0)
        if arr.ndim == 2:
            if arr.shape[-1] != dim:
                raise ValueError(f'{name} dim mismatch: got {arr.shape}, '
                                 f'expected dim={dim}')
            if arr.shape[0] == steps:
                return arr
            if arr.shape[0] == 1:
                return np.repeat(arr, steps, axis=0)
            raise ValueError(f'{name} step mismatch: got {arr.shape}, '
                             f'expected steps={steps}, dim={dim}')
        raise ValueError(f'{name} must be 1D or 2D, got shape {arr.shape}')

    @staticmethod
    def _last_state_reference(state_values: Dict[str, Any], state_key: str,
                              dim: int) -> np.ndarray:
        if state_key not in state_values:
            raise KeyError(f'Missing reference state key {state_key!r}')
        state = np.asarray(state_values[state_key], dtype=np.float32)
        if state.ndim == 1:
            reference = state
        elif state.ndim == 2:
            reference = state[-1]
        else:
            raise ValueError(f'Reference state {state_key!r} must be 1D or 2D, '
                             f'got shape {state.shape}')
        if reference.shape[-1] != dim:
            raise ValueError(f'Reference state {state_key!r} dim mismatch: '
                             f'got {reference.shape[-1]}, expected {dim}')
        return reference

    def _convert_relative_action(self, action: np.ndarray,
                                 reference_state: np.ndarray,
                                 action_config: Dict[str, Any],
                                 to_absolute: bool) -> np.ndarray:
        action_type = str(action_config.get('type', 'NON_EEF')).upper()
        action_format = str(action_config.get('format', 'DEFAULT')).upper()
        if action_type != 'NON_EEF' or action_format != 'DEFAULT':
            raise NotImplementedError(
                'Layer 6 only supports NON_EEF/DEFAULT relative action '
                f'conversion, got type={action_type}, format={action_format}')
        if to_absolute:
            return action + reference_state
        return action - reference_state

    def _maybe_convert_action_representation(
        self,
        action: np.ndarray,
        raw_state: Optional[Dict[str, Any]],
        key: str,
        action_config: Dict[str, Any],
        to_absolute: bool,
    ) -> np.ndarray:
        if (not self.use_relative_action
                or str(action_config.get('rep', '')).upper() != 'RELATIVE'):
            return action
        if raw_state is None:
            raise ValueError('Raw state is required for relative action '
                             f'conversion of key {key!r}')
        state_key = action_config.get('state_key') or key
        reference = self._last_state_reference(raw_state, state_key,
                                               action.shape[-1])
        return self._convert_relative_action(action, reference, action_config,
                                             to_absolute)

    def _process_modality_values(self, modality_config: Dict[str, Any],
                                 statistics: Optional[Dict[str, Any]],
                                 modality: str, steps: int,
                                 layout_entries: list,
                                 values_by_key: Optional[Dict[str,
                                                              Any]] = None,
                                 raw_state: Optional[Dict[str, Any]] = None,
                                 normalize: bool = True) -> np.ndarray:
        chunks = []
        relative_action_keys = self._relative_action_keys(modality_config)
        sincos_keys = set()
        if modality == 'state' and self.apply_sincos_state_encoding:
            sincos_keys = set(
                modality_config.get('state',
                                    {}).get('sin_cos_embedding_keys') or [])
        for entry in layout_entries:
            key = entry['key']
            stat = self._stat_for_key(statistics, modality, key,
                                      relative_action_keys)
            if values_by_key is None:
                values = self._synthetic_values_from_stat(
                    stat, steps, entry['raw_dim'])
            else:
                if key not in values_by_key:
                    raise KeyError(f'Missing {modality} key {key!r}')
                values = self._coerce_sequence_values(
                    values_by_key[key], steps, entry['raw_dim'],
                    f'{modality}.{key}')
            if modality == 'action' and normalize:
                action_config = self._action_config_for_key(
                    modality_config, key)
                values = self._maybe_convert_action_representation(
                    values, raw_state, key, action_config, to_absolute=False)
            if modality == 'state' and key in sincos_keys:
                values = self._apply_sincos(values)
            elif normalize:
                values = self._normalize_values(values, stat)
            if values.shape[-1] != entry['processed_dim']:
                raise ValueError(f'Processed dim mismatch for {key}: '
                                 f'{values.shape[-1]} vs '
                                 f'{entry["processed_dim"]}')
            chunks.append(values.astype(np.float32))
        if not chunks:
            return np.zeros((steps, 0), dtype=np.float32)
        return np.concatenate(chunks, axis=-1)

    def _split_flat_action(self, flat_action: np.ndarray,
                           layout_entries: list) -> Dict[str, np.ndarray]:
        action = np.asarray(flat_action, dtype=np.float32)
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError('Layer 6 split supports unbatched or batch=1 '
                                 f'action, got shape {action.shape}')
            action = action[0]
        if action.ndim != 2:
            raise ValueError(f'Action must have shape (T, D), got '
                             f'{action.shape}')
        chunks = {}
        for entry in layout_entries:
            chunks[entry['key']] = action[:, entry['start']:entry['end']]
        return chunks

    def _extract_flat_or_nested_value(self, observation: Dict[str, Any],
                                      modality: str, key: str) -> Any:
        if modality in observation and isinstance(observation[modality], dict):
            if key in observation[modality]:
                return observation[modality][key]
        flat_key = f'{modality}.{key}'
        if flat_key in observation:
            return observation[flat_key]
        if modality == 'language' and key in observation:
            return observation[key]
        if modality == 'language':
            for alias in ('annotation.human.coarse_action',
                          'annotation.human.action.task_description',
                          'language', 'task'):
                if alias in observation:
                    return observation[alias]
        raise KeyError(f'Missing observation key {flat_key!r}')

    @staticmethod
    def _coerce_batched_video(values: Any, steps: int,
                              key: str) -> np.ndarray:
        video = np.asarray(values)
        if video.dtype != np.uint8:
            video = video.astype(np.uint8)
        if video.ndim == 3:
            video = video[None, None, ...]
        elif video.ndim == 4:
            if video.shape[0] == steps:
                video = video[None, ...]
            elif video.shape[0] == 1:
                video = video[:, None, ...]
            else:
                raise ValueError(f'Video {key!r} shape {video.shape} cannot '
                                 f'map to steps={steps}')
        elif video.ndim != 5:
            raise ValueError(f'Video {key!r} must be HWC, THWC, or BTHWC, '
                             f'got shape {video.shape}')
        if video.shape[1] != steps:
            raise ValueError(f'Video {key!r} temporal dim mismatch: '
                             f'{video.shape[1]} vs {steps}')
        if video.shape[-1] != 3:
            raise ValueError(f'Video {key!r} must have 3 channels, got '
                             f'{video.shape[-1]}')
        return video

    @staticmethod
    def _coerce_batched_state(values: Any, steps: int, dim: int,
                              key: str) -> np.ndarray:
        state = np.asarray(values, dtype=np.float32)
        if state.ndim == 1:
            if state.shape[-1] != dim:
                raise ValueError(f'State {key!r} dim mismatch: '
                                 f'{state.shape[-1]} vs {dim}')
            state = np.repeat(state[None, :], steps, axis=0)[None, ...]
        elif state.ndim == 2:
            if state.shape[-1] != dim:
                raise ValueError(f'State {key!r} dim mismatch: '
                                 f'{state.shape[-1]} vs {dim}')
            if state.shape[0] == steps:
                state = state[None, ...]
            elif state.shape[0] == 1:
                state = np.repeat(state, steps, axis=0)[None, ...]
            else:
                raise ValueError(f'State {key!r} step mismatch: '
                                 f'{state.shape[0]} vs {steps}')
        elif state.ndim == 3:
            if state.shape[1] != steps or state.shape[-1] != dim:
                raise ValueError(f'State {key!r} shape mismatch: '
                                 f'{state.shape}, expected B,{steps},{dim}')
        else:
            raise ValueError(f'State {key!r} must be D, TD, or BTD, '
                             f'got shape {state.shape}')
        return state.astype(np.float32)

    @staticmethod
    def _coerce_batched_language(values: Any, steps: int) -> list:
        if isinstance(values, str):
            return [[values] * steps]
        if isinstance(values, list):
            if values and all(isinstance(item, str) for item in values):
                if len(values) == steps:
                    return [values]
                if len(values) == 1:
                    return [values * steps]
            if values and all(isinstance(item, list) for item in values):
                for item in values:
                    if len(item) != steps:
                        raise ValueError('Language temporal dim mismatch: '
                                         f'{len(item)} vs {steps}')
                return values
        raise ValueError('Language must be a string, list[str], or '
                         'list[list[str]]')

    def build_batched_observation(
        self,
        observation: Dict[str, Any],
        task: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build official-style batched N1.7 observation without tokenization."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        video_steps = self._modality_steps(modality_config, 'video')
        state_steps = layout['state_steps']
        language_steps = self._modality_steps(modality_config, 'language')
        video = {}
        for key in self._modality_keys(modality_config, 'video'):
            value = self._extract_flat_or_nested_value(observation, 'video',
                                                       key)
            video[key] = self._coerce_batched_video(value, video_steps, key)
        state = {}
        for entry in layout['state_layout']:
            value = self._extract_flat_or_nested_value(observation, 'state',
                                                       entry['key'])
            state[entry['key']] = self._coerce_batched_state(
                value, state_steps, entry['raw_dim'], entry['key'])
        language = {}
        for key in self._modality_keys(modality_config, 'language'):
            if task is not None:
                value = task
            else:
                value = self._extract_flat_or_nested_value(
                    observation, 'language', key)
            language[key] = self._coerce_batched_language(
                value, language_steps)
        batch_sizes = {next(iter(video.values())).shape[0]}
        batch_sizes.update(v.shape[0] for v in state.values())
        batch_sizes.update(len(v) for v in language.values())
        if len(batch_sizes) != 1:
            raise ValueError(f'Inconsistent observation batch sizes: '
                             f'{sorted(batch_sizes)}')
        return {
            'video': video,
            'state': state,
            'language': language,
            'embodiment_key': selected['embodiment_key'],
            'embodiment_id': selected['embodiment_id'],
        }

    def encode_eval_observation(
        self,
        observation: Dict[str, Any],
        task: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode a batch=1 eval observation through the native state path."""
        batched = self.build_batched_observation(observation, task,
                                                 embodiment_tag, env_name)
        batch_size = next(iter(batched['state'].values())).shape[0]
        if batch_size != 1:
            raise ValueError('Layer 7 eval observation smoke supports batch=1, '
                             f'got batch={batch_size}')
        raw_state = {
            key: value[0]
            for key, value in batched['state'].items()
        }
        encoded = self.encode_state_action_dict(raw_state,
                                                embodiment_tag=embodiment_tag,
                                                env_name=env_name)
        action_steps = self.state_action_layout_summary(
            embodiment_tag, env_name)['action_steps']
        encoded.update({
            'video': batched['video'],
            'language': batched['language'],
            'raw_state': raw_state,
            'embodiment_key': batched['embodiment_key'],
            'embodiment_id': np.asarray([batched['embodiment_id']],
                                        dtype=np.int32),
            'action_horizon_mask': np.ones((1, action_steps),
                                           dtype=np.float32),
        })
        return encoded

    def decode_action_for_env(
        self,
        action: np.ndarray,
        raw_state: Dict[str, Any],
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
        prefix_action_keys: bool = True,
    ) -> Dict[str, np.ndarray]:
        decoded = self.decode_action_array(action, raw_state, embodiment_tag,
                                           env_name, to_absolute=True)
        if prefix_action_keys:
            return {f'action.{key}': value for key, value in decoded.items()}
        return decoded

    @staticmethod
    def _pad_values(values: np.ndarray, max_dim: int) -> np.ndarray:
        if values.shape[-1] > max_dim:
            raise ValueError(f'Cannot pad dim {values.shape[-1]} to {max_dim}')
        padded = np.zeros((values.shape[0], max_dim), dtype=np.float32)
        padded[:, :values.shape[-1]] = values
        return padded

    def state_action_layout_summary(
        self,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return state/action layout, padding masks, and expected shapes."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        statistics = selected['statistics']
        state_layout = self._key_layout(modality_config, statistics, 'state')
        action_layout = self._key_layout(modality_config, statistics, 'action')
        state_steps = len(
            modality_config.get('state', {}).get('delta_indices') or [])
        action_steps = len(
            modality_config.get('action', {}).get('delta_indices') or [])
        max_state_dim = int(self.max_state_dim or state_layout['total_dim'])
        max_action_dim = int(self.max_action_dim or action_layout['total_dim'])
        return {
            'embodiment_key': selected['embodiment_key'],
            'embodiment_id': selected['embodiment_id'],
            'has_statistics': selected['has_statistics'],
            'state_total_dim': state_layout['total_dim'],
            'action_total_dim': action_layout['total_dim'],
            'max_state_dim': max_state_dim,
            'max_action_dim': max_action_dim,
            'state_steps': state_steps,
            'action_steps': action_steps,
            'state_mask_true': sum(
                self._padding_mask(state_layout['total_dim'], max_state_dim)),
            'action_mask_true': sum(
                self._padding_mask(action_layout['total_dim'],
                                   max_action_dim)),
            'state_shape': (1, state_steps, max_state_dim),
            'action_shape': (1, action_steps, max_action_dim),
            'state_layout': state_layout['entries'],
            'action_layout': action_layout['entries'],
        }

    def state_action_processor_smoke(
        self,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize synthetic state/action values, concatenate, pad, and mask.

        This is a metadata/statistics smoke test. It does not instantiate the
        official N1.7 processor or the model weights.
        """
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        if selected['statistics'] is None:
            raise ValueError('Layer 5 processor smoke requires statistics for '
                             f'{selected["embodiment_key"]!r}')
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        statistics = selected['statistics']
        state_values = self._process_modality_values(
            modality_config,
            statistics,
            'state',
            layout['state_steps'],
            layout['state_layout'],
        )
        action_values = self._process_modality_values(
            modality_config,
            statistics,
            'action',
            layout['action_steps'],
            layout['action_layout'],
        )
        padded_state = self._pad_values(state_values, layout['max_state_dim'])
        padded_action = self._pad_values(action_values,
                                         layout['max_action_dim'])
        state_mask = np.asarray(
            self._padding_mask(layout['state_total_dim'],
                               layout['max_state_dim']),
            dtype=bool)
        action_mask = np.asarray(
            self._padding_mask(layout['action_total_dim'],
                               layout['max_action_dim']),
            dtype=bool)
        return {
            'embodiment_key': layout['embodiment_key'],
            'embodiment_id': layout['embodiment_id'],
            'state': padded_state[None, ...],
            'action': padded_action[None, ...],
            'state_mask': state_mask[None, None, :],
            'action_mask': action_mask[None, None, :],
            'state_shape': tuple(padded_state[None, ...].shape),
            'action_shape': tuple(padded_action[None, ...].shape),
            'state_mask_true': int(state_mask.sum()),
            'action_mask_true': int(action_mask.sum()),
            'state_value_range': (
                float(padded_state[:, :layout['state_total_dim']].min()),
                float(padded_state[:, :layout['state_total_dim']].max()),
            ),
            'action_value_range': (
                float(padded_action[:, :layout['action_total_dim']].min()),
                float(padded_action[:, :layout['action_total_dim']].max()),
            ),
        }

    def encode_state_action_dict(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode raw state/action dictionaries into N1.7 padded arrays."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        if selected['statistics'] is None:
            raise ValueError('Encoding requires checkpoint statistics for '
                             f'{selected["embodiment_key"]!r}')
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        statistics = selected['statistics']
        state_values = self._process_modality_values(
            modality_config,
            statistics,
            'state',
            layout['state_steps'],
            layout['state_layout'],
            values_by_key=state,
            normalize=True,
        )
        padded_state = self._pad_values(state_values, layout['max_state_dim'])
        state_mask = np.asarray(
            self._padding_mask(layout['state_total_dim'],
                               layout['max_state_dim']),
            dtype=bool)
        result = {
            'embodiment_key': layout['embodiment_key'],
            'embodiment_id': layout['embodiment_id'],
            'state': padded_state[None, ...],
            'state_mask': state_mask[None, None, :],
            'state_shape': tuple(padded_state[None, ...].shape),
            'state_mask_true': int(state_mask.sum()),
        }
        if action is not None:
            action_values = self._process_modality_values(
                modality_config,
                statistics,
                'action',
                layout['action_steps'],
                layout['action_layout'],
                values_by_key=action,
                raw_state=state,
                normalize=True,
            )
            padded_action = self._pad_values(action_values,
                                             layout['max_action_dim'])
            action_mask = np.asarray(
                self._padding_mask(layout['action_total_dim'],
                                   layout['max_action_dim']),
                dtype=bool)
            result.update({
                'action': padded_action[None, ...],
                'action_mask': action_mask[None, None, :],
                'action_shape': tuple(padded_action[None, ...].shape),
                'action_mask_true': int(action_mask.sum()),
            })
        return result

    def decode_action_array(
        self,
        action: np.ndarray,
        raw_state: Optional[Dict[str, Any]] = None,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
        to_absolute: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Split and denormalize a padded N1.7 action array by action key."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        if selected['statistics'] is None:
            raise ValueError('Action decode requires checkpoint statistics for '
                             f'{selected["embodiment_key"]!r}')
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        statistics = selected['statistics']
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 3:
            action_arr = action_arr[:, :layout['action_steps'], :]
        else:
            action_arr = action_arr[:layout['action_steps'], :]
        action_chunks = self._split_flat_action(action_arr,
                                                layout['action_layout'])
        relative_action_keys = self._relative_action_keys(modality_config)
        decoded = {}
        for entry in layout['action_layout']:
            key = entry['key']
            stat = self._stat_for_key(statistics, 'action', key,
                                      relative_action_keys)
            values = self._unnormalize_values(action_chunks[key], stat)
            action_config = self._action_config_for_key(modality_config, key)
            if to_absolute:
                values = self._maybe_convert_action_representation(
                    values, raw_state, key, action_config, to_absolute=True)
            decoded[key] = values
        return decoded

    def dict_processor_roundtrip_smoke(
        self,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Round-trip synthetic raw dicts through encode and action decode."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        statistics = selected['statistics']
        if statistics is None:
            raise ValueError('Round-trip smoke requires checkpoint statistics '
                             f'for {selected["embodiment_key"]!r}')

        def _raw_dict(modality: str, entries: list, steps: int) -> Dict[str,
                                                                        Any]:
            values = {}
            relative_keys = self._relative_action_keys(
                selected['modality_config'])
            for entry in entries:
                stat = self._stat_for_key(statistics, modality, entry['key'],
                                          relative_keys)
                values[entry['key']] = self._synthetic_values_from_stat(
                    stat, steps, entry['raw_dim'])
            return values

        raw_state = _raw_dict('state', layout['state_layout'],
                              layout['state_steps'])
        raw_action = _raw_dict('action', layout['action_layout'],
                               layout['action_steps'])
        for entry in layout['action_layout']:
            key = entry['key']
            action_config = self._action_config_for_key(
                selected['modality_config'], key)
            if (self.use_relative_action
                    and str(action_config.get('rep', '')).upper()
                    == 'RELATIVE'):
                state_key = action_config.get('state_key') or key
                reference = self._last_state_reference(
                    raw_state, state_key, entry['raw_dim'])
                raw_action[key] = raw_action[key] + reference
        encoded = self.encode_state_action_dict(raw_state, raw_action,
                                                embodiment_tag, env_name)
        decoded_action = self.decode_action_array(
            encoded['action'],
            raw_state=raw_state,
            embodiment_tag=embodiment_tag,
            env_name=env_name,
            to_absolute=True,
        )
        max_abs_error = 0.0
        for key, target in raw_action.items():
            err = np.max(np.abs(decoded_action[key] - target))
            max_abs_error = max(max_abs_error, float(err))
        return {
            'embodiment_key': layout['embodiment_key'],
            'state_shape': encoded['state_shape'],
            'action_shape': encoded.get('action_shape'),
            'state_mask_true': encoded['state_mask_true'],
            'action_mask_true': encoded.get('action_mask_true'),
            'decoded_action_keys': list(decoded_action.keys()),
            'max_abs_roundtrip_error': max_abs_error,
        }

    def eval_observation_smoke(
        self,
        task: str = 'pick up the object',
        image_size: int = 256,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Smoke-test flat eval observation formatting and action decode keys."""
        selected = self.select_embodiment_metadata(embodiment_tag, env_name)
        layout = self.state_action_layout_summary(embodiment_tag, env_name)
        modality_config = selected['modality_config']
        statistics = selected['statistics']
        if statistics is None:
            raise ValueError('Eval observation smoke requires checkpoint '
                             f'statistics for {selected["embodiment_key"]!r}')
        observation = {}
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for key in self._modality_keys(modality_config, 'video'):
            observation[f'video.{key}'] = image
        for entry in layout['state_layout']:
            stat = self._stat_for_key(statistics, 'state', entry['key'],
                                      self._relative_action_keys(
                                          modality_config))
            state_value = self._synthetic_values_from_stat(
                stat, layout['state_steps'], entry['raw_dim'])
            observation[f'state.{entry["key"]}'] = state_value
        language_keys = self._modality_keys(modality_config, 'language')
        if language_keys:
            observation[language_keys[0]] = task
        encoded = self.encode_eval_observation(observation, task,
                                               embodiment_tag, env_name)
        zero_action = np.zeros(
            (layout['action_steps'], layout['max_action_dim']),
            dtype=np.float32)
        decoded = self.decode_action_for_env(
            zero_action,
            raw_state=encoded['raw_state'],
            embodiment_tag=embodiment_tag,
            env_name=env_name,
            prefix_action_keys=True,
        )
        video_shapes = {
            key: tuple(value.shape)
            for key, value in encoded['video'].items()
        }
        language_shapes = {
            key: (len(value), len(value[0]))
            for key, value in encoded['language'].items()
        }
        return {
            'embodiment_key': encoded['embodiment_key'],
            'embodiment_id_shape': tuple(encoded['embodiment_id'].shape),
            'state_shape': encoded['state_shape'],
            'state_mask_true': encoded['state_mask_true'],
            'video_shapes': video_shapes,
            'language_shapes': language_shapes,
            'language_sample': next(iter(encoded['language'].values()))[0][0],
            'action_horizon_mask_shape':
            tuple(encoded['action_horizon_mask'].shape),
            'decoded_env_action_keys': list(decoded.keys()),
        }

    def official_forward_probe(
        self,
        task: str = 'pick up the object',
        image_size: int = 256,
        device: Optional[str] = None,
        dtype: str = 'bfloat16',
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        """Run a minimal official processor + model get_action smoke."""
        load_result = self.official_load_probe(
            load_processor=self.processor_runtime == 'official',
            load_model=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.processor_runtime == 'native':
            try:
                self._ensure_native_processor(
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                )
                load_result['native_processor'] = {
                    'status': 'ok',
                    'class': type(self.processor).__name__,
                    'model_name': getattr(self.processor, 'model_name', None),
                    'max_state_dim': getattr(self.processor, 'max_state_dim',
                                             None),
                    'max_action_dim': getattr(self.processor, 'max_action_dim',
                                             None),
                }
            except Exception as exc:  # pragma: no cover - smoke helper
                load_result['native_processor'] = self._compact_exception(exc)
        if self.processor is None:
            return {
                'load_result': load_result,
                'forward': {
                    'status': 'skipped',
                    'reason': 'processor was not loaded',
                },
            }
        if self.n17_model is None:
            return {
                'load_result': load_result,
                'forward': {
                    'status': 'skipped',
                    'reason': 'model was not loaded',
                },
            }
        try:
            import torch as _torch

            target_device = device
            if target_device is None:
                target_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
            target_dtype = getattr(_torch, dtype)
            self.n17_model.to(device=target_device, dtype=target_dtype)
            self.n17_model.eval()

            gr00t_policy = importlib.import_module('gr00t.policy.gr00t_policy')
            data_types = importlib.import_module('gr00t.data.types')
            embodiment_tags = importlib.import_module(
                'gr00t.data.embodiment_tags')
            rec_to_dtype = getattr(gr00t_policy, '_rec_to_dtype')
            MessageType = getattr(data_types, 'MessageType')
            VLAStepData = getattr(data_types, 'VLAStepData')
            EmbodimentTag = getattr(embodiment_tags, 'EmbodimentTag')
            embodiment = EmbodimentTag.resolve(self.embodiment_tag)

            observation_summary = self.eval_observation_smoke(
                task=task, image_size=image_size)
            del observation_summary
            selected = self.select_embodiment_metadata()
            layout = self.state_action_layout_summary()
            modality_config = selected['modality_config']
            statistics = selected['statistics']
            observation = {}
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            for key in self._modality_keys(modality_config, 'video'):
                observation[f'video.{key}'] = image
            for entry in layout['state_layout']:
                stat = self._stat_for_key(statistics, 'state', entry['key'],
                                          self._relative_action_keys(
                                              modality_config))
                observation[f'state.{entry["key"]}'] = (
                    self._synthetic_values_from_stat(
                        stat, layout['state_steps'], entry['raw_dim']))
            language_keys = self._modality_keys(modality_config, 'language')
            if language_keys:
                observation[language_keys[0]] = task

            batched = self.build_batched_observation(observation, task=task)
            raw_state = {key: value[0] for key, value in batched['state'].items()}
            vla_step = VLAStepData(
                images={key: value[0] for key, value in batched['video'].items()},
                states=raw_state,
                actions={},
                text=next(iter(batched['language'].values()))[0][0],
                embodiment=embodiment,
            )
            messages = [{
                'type': MessageType.EPISODE_STEP.value,
                'content': vla_step,
            }]
            processed = self.processor(messages)
            collated = self.processor.collator([processed])
            collated = rec_to_dtype(collated, dtype=target_dtype)
            with _torch.inference_mode():
                model_pred = self.n17_model.get_action(**collated)
            normalized_action = model_pred['action_pred'].float().cpu().numpy()
            decoded = self.decode_action_for_env(
                normalized_action,
                raw_state=raw_state,
                prefix_action_keys=True,
            )
            return {
                'load_result': load_result,
                'forward': {
                    'status': 'ok',
                    'device': target_device,
                    'dtype': str(target_dtype),
                    'normalized_action_shape': tuple(normalized_action.shape),
                    'decoded_action_keys': list(decoded.keys()),
                    'decoded_action_shapes': {
                        key: tuple(value.shape)
                        for key, value in decoded.items()
                    },
                },
            }
        except Exception as exc:  # pragma: no cover - smoke helper
            return {
                'load_result': load_result,
                'forward': self._compact_exception(exc),
            }

    def _ensure_official_runtime(
        self,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        """Lazily load the selected processor and official model runtime."""
        if self.processor is not None and self.n17_model is not None:
            return {
                'status': 'already_loaded',
                'checkpoint_dir': str(self.checkpoint_dir),
                'processor_runtime': self.processor_runtime,
            }
        load_result = self.official_load_probe(
            load_processor=self.processor_runtime == 'official',
            load_model=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.processor_runtime == 'native':
            self._ensure_native_processor(
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            load_result['native_processor'] = {
                'status': 'ok',
                'class': type(self.processor).__name__,
                'model_name': getattr(self.processor, 'model_name', None),
                'max_state_dim': getattr(self.processor, 'max_state_dim', None),
                'max_action_dim': getattr(self.processor, 'max_action_dim',
                                          None),
            }
        return load_result

    def _ensure_native_processor(
        self,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        """Load FluxVLA-native N1.7 processor from checkpoint metadata."""
        if self.checkpoint_dir is None:
            raise ValueError('Native processor requires model_path metadata.')
        processors = importlib.import_module('fluxvla.processors')
        GrootN17Processor = getattr(processors, 'GrootN17Processor')
        processor_kwargs = {}
        if self.backbone_model_path is not None:
            processor_kwargs['model_name'] = self.effective_backbone_model_name
        processor_source = self._resolve_processor_metadata_dir(
            self.checkpoint_dir)
        processor = GrootN17Processor.from_pretrained(
            processor_source,
            transformers_loading_kwargs={
                'local_files_only': local_files_only,
                'trust_remote_code': trust_remote_code,
            },
            **processor_kwargs,
        )
        processor.eval()
        self.processor = processor

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
        """Load FluxVLA-native processor, backbone, and action head."""
        backbone, action_head = self._native_runtime_modules()
        if (self.processor is not None and backbone is not None
                and action_head is not None):
            return {
                'status': 'already_loaded',
                'checkpoint_dir': str(self.checkpoint_dir),
                'processor_runtime': 'native',
                'assembly_runtime': 'native',
                'all_module_keys': list(self.all_module_keys or []),
            }
        if self.checkpoint_dir is None:
            raise ValueError('Native runtime requires model_path metadata.')
        self._ensure_native_processor(
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
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
            'processor_runtime': 'native',
            'assembly_runtime': 'native',
            'all_module_keys': list(self.all_module_keys),
            'qwen3_runtime': self.qwen3_runtime,
            'qwen3_runtime_summary': self.qwen3_runtime_summary,
            'native_processor': {
                'class': type(self.processor).__name__,
                'model_name': getattr(self.processor, 'model_name', None),
                'max_state_dim': getattr(self.processor, 'max_state_dim', None),
                'max_action_dim': getattr(self.processor, 'max_action_dim',
                                          None),
            },
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
        from transformers.feature_extraction_utils import BatchFeature

        moved = self._move_batch_to_device_dtype(inputs, device, dtype)
        backbone_inputs = BatchFeature(data={
            key: moved[key]
            for key in ('input_ids', 'attention_mask', 'pixel_values',
                        'image_grid_thw')
        })
        action_inputs = BatchFeature(data=moved)
        return backbone_inputs, action_inputs

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
            return action_head.forward_tensors(
                input_features=backbone_outputs.backbone_features,
                states=action_inputs.state,
                attention_mask=backbone_outputs.backbone_attention_mask,
                embodiment_ids=action_inputs.embodiment_id,
                actions=action_inputs.action,
                action_masks=action_inputs.action_mask,
                image_mask=backbone_outputs.get('image_mask'),
                sample_weight=action_inputs.get('sample_weight'),
            )
        if mode == 'action':
            prev_actions = (
                action_inputs['action'] if 'action' in action_inputs else None)
            return action_head.get_action_tensors(
                input_features=backbone_outputs.backbone_features,
                states=action_inputs.state,
                attention_mask=backbone_outputs.backbone_attention_mask,
                embodiment_ids=action_inputs.embodiment_id,
                image_mask=backbone_outputs.get('image_mask'),
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
        if self.processor is None or backbone is None or action_head is None:
            raise RuntimeError('Failed to load native N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        target_dtype = getattr(torch, dtype)
        backbone.to(device=target_device, dtype=target_dtype)
        action_head.to(device=target_device, dtype=target_dtype)
        backbone.eval()
        action_head.eval()
        return target_device, target_dtype

    def _prepare_official_eval_runtime(
        self,
        dtype: str,
    ) -> tuple[torch.device, torch.dtype]:
        self._ensure_official_runtime()
        if self.processor is None or self.n17_model is None:
            raise RuntimeError('Failed to load official N1.7 runtime.')
        target_device = self._resolve_runtime_device()
        target_dtype = getattr(torch, dtype)
        self.n17_model.to(device=target_device, dtype=target_dtype)
        self.n17_model.eval()
        return target_device, target_dtype

    @staticmethod
    def _recursive_to_dtype(value: Any, dtype: torch.dtype) -> Any:
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                return value.to(dtype=dtype)
            return value
        if isinstance(value, dict):
            return {
                key: GrootN17VLA._recursive_to_dtype(item, dtype)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(
                GrootN17VLA._recursive_to_dtype(item, dtype) for item in value)
        return value

    def _predict_n17_raw_action_from_observation(
        self,
        observation: Dict[str, Any],
        task: Optional[str] = None,
        dtype: str = 'bfloat16',
    ) -> torch.Tensor:
        """Run N1.7 inference and return raw flat env actions."""
        if self.assembly_runtime == 'native':
            target_device, target_dtype = self._prepare_native_eval_runtime(
                dtype)
        else:
            target_device, target_dtype = self._prepare_official_eval_runtime(
                dtype)

        task_text = task or observation.get('task') or observation.get(
            'task_description', 'pick up the object')
        batched = self.build_batched_observation(observation, task=task_text)
        raw_state = {
            key: value[0]
            for key, value in batched['state'].items()
        }
        if self.assembly_runtime == 'native':
            vla_step = SimpleNamespace(
                images={
                    key: value[0]
                    for key, value in batched['video'].items()
                },
                states=raw_state,
                actions={},
                text=next(iter(batched['language'].values()))[0][0],
                embodiment=self.active_embodiment_key,
            )
            messages = [{'type': 'episode_step', 'content': vla_step}]
        else:
            data_types = importlib.import_module('gr00t.data.types')
            embodiment_tags = importlib.import_module(
                'gr00t.data.embodiment_tags')
            MessageType = getattr(data_types, 'MessageType')
            VLAStepData = getattr(data_types, 'VLAStepData')
            EmbodimentTag = getattr(embodiment_tags, 'EmbodimentTag')
            embodiment = EmbodimentTag.resolve(self.embodiment_tag)
            vla_step = VLAStepData(
                images={
                    key: value[0]
                    for key, value in batched['video'].items()
                },
                states=raw_state,
                actions={},
                text=next(iter(batched['language'].values()))[0][0],
                embodiment=embodiment,
            )
            messages = [{
                'type': MessageType.EPISODE_STEP.value,
                'content': vla_step,
            }]
        processed = self.processor(messages)
        collated = self.processor.collator([processed])
        collated = self._recursive_to_dtype(collated, target_dtype)
        with torch.inference_mode():
            if self.assembly_runtime == 'native':
                model_pred = self._run_native_backbone_head(
                    collated['inputs'], mode='action')
            else:
                model_pred = self.n17_model.get_action(**collated)
        normalized_action = model_pred['action_pred'].float().cpu().numpy()
        return self._decode_n17_action_to_env_tensor(
            normalized_action,
            embodiment_key=batched['embodiment_key'],
            raw_state=raw_state)

    def _decode_n17_action_to_env_tensor(
        self,
        normalized_action: np.ndarray,
        embodiment_key: Optional[str] = None,
        raw_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Decode normalized N1.7 actions and flatten them for the env."""
        if self.processor is None:
            self._ensure_native_processor()
        if embodiment_key is None:
            embodiment_key = self.active_embodiment_key
        if hasattr(self.processor, 'decode_action'):
            if normalized_action.ndim == 3:
                if normalized_action.shape[0] != 1:
                    raise ValueError('N1.7 eval decode expects batch=1, got '
                                     f'action shape {normalized_action.shape}')
                action_for_decode = normalized_action[0]
            else:
                action_for_decode = normalized_action
            decoded = self.processor.decode_action(
                action_for_decode, embodiment_key, state=raw_state)
            action_keys = self.processor.modality_configs[
                embodiment_key]['action']['modality_keys']
        else:
            decoded = self.decode_action_array(normalized_action,
                                               raw_state=raw_state,
                                               embodiment_tag=embodiment_key)
            action_keys = [
                entry['key'] for entry in self.state_action_layout_summary(
                    embodiment_key)['action_layout']
            ]
        flat_action = np.concatenate(
            [np.asarray(decoded[key], dtype=np.float32) for key in action_keys],
            axis=-1,
        ).astype(np.float32)
        return torch.from_numpy(flat_action[None, ...]).to(
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
        if 'n17_observation' in batch:
            return self._predict_n17_raw_action_from_observation(
                batch['n17_observation'],
                task=batch.get('n17_task'),
                dtype=batch.get('dtype', 'bfloat16'),
            )
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
            'GrootN17VLA.predict_action expects either an `n17_observation` '
            'batch produced by an N1.7 eval dataset or split tensor inputs '
            'containing input_ids, attention_mask, pixel_values, '
            'image_grid_thw, state, and embodiment_id.')
