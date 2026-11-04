# Copyright 2026 Limx Dynamics
"""A standalone VLM sharing FluxVLA's training and checkpoint contracts."""

from fluxvla.engines import VLAS
from .base_vla import BaseVLA
from .causal_vlm_training import CausalVLMTrainingMixin


@VLAS.register_module()
class VLMWrapper(CausalVLMTrainingMixin, BaseVLA):

    def __init__(self,
                 vlm_backbone,
                 freeze_vlm_backbone=False,
                 pretrained_name_or_path=None,
                 name_mapping=None,
                 strict_mapping=True,
                 ignore_index=-100,
                 **runner_options):
        lora_fields = {
            'use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout',
            'lora_target_modules', 'modules_to_save', 'init_lora_weights'
        }
        unknown = set(runner_options) - lora_fields
        if unknown:
            raise TypeError(f'Unknown VLMWrapper options: {sorted(unknown)}')
        super().__init__(
            vlm_backbone=vlm_backbone,
            freeze_vlm_backbone=freeze_vlm_backbone,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
            ignore_index=ignore_index)
        self.all_module_keys = ['vlm_backbone']

    def validate_training_task(self, task):
        if task != 'vlm':
            raise ValueError('VLMWrapper requires runner.train_task="vlm".')
        super().validate_training_task(task)

    def forward(self, *args, task='vlm', **batch):
        return super().forward(*args, task=task, **batch)

    @property
    def config(self):
        return self.vlm_backbone.vlm.config
