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
"""Optional language training for policies retaining a causal LM."""

from fluxvla.engines.losses.causal_lm import causal_lm_loss


class CausalVLMTrainingMixin:
    """Opt-in task dispatch and language supervision on the existing backbone.

    This mixin has no constructor, parameters or child modules. Policies
    supply ``forward_vla``; language-only wrappers reject the VLA task.
    BaseVLA and policies that do not opt in keep their own forward contract.
    """

    def forward(self, *args, task='vla', **batch):
        """Dispatch inside DDP/FSDP, preserving the default VLA call."""
        self.validate_training_task(task)
        if task == 'vlm':
            return self.forward_vlm(*args, **batch)
        return self.forward_vla(*args, **batch)

    def get_vlm_fsdp_wrapping_policy(self):
        """Keep conditional vision/projectors in the root FSDP unit."""
        self.validate_training_task('vlm')
        if self.vlm_backbone is not None:
            return self.vlm_backbone.get_fsdp_wrapping_policy()
        return self.llm_backbone.get_fsdp_wrapping_policy()

    def validate_training_task(self, task):
        if task == 'vla':
            return
        if task != 'vlm':
            raise ValueError(f'Unknown training task: {task}')
        if self.vlm_backbone is not None:
            supported = getattr(self.vlm_backbone, 'supports_vlm_training',
                                False)
        else:
            lm = getattr(self.llm_backbone, 'llm', None)
            supported = (
                lm is not None
                and callable(getattr(lm, 'get_output_embeddings', None))
                and lm.get_output_embeddings() is not None)
        if not supported:
            raise ValueError(f'{type(self).__name__} does not retain a '
                             'supported causal VLM training path.')

    def prepare_training_task(self, task):
        self.validate_training_task(task)
        if task == 'vlm':
            # Called again after LoRA injection. Never re-enable parameters:
            # adapters and the configured backbone freezing remain intact.
            if self.vla_head is not None:
                self.vla_head.requires_grad_(False)
            if self.vlm_backbone is not None:
                native = self.vlm_backbone.vlm
                active = {id(p) for p in native.parameters()}
                for parameter in self.vlm_backbone.parameters():
                    if id(parameter) not in active:
                        parameter.requires_grad_(False)
                prepare = getattr(self.vlm_backbone, 'prepare_vlm_training',
                                  None)
                if callable(prepare):
                    prepare()

    def forward_vlm(self,
                    *,
                    input_ids=None,
                    labels=None,
                    attention_mask=None,
                    pixel_values=None,
                    lang_tokens=None,
                    lang_masks=None,
                    images=None,
                    multimodal_indices=None,
                    **inputs):
        self.validate_training_task('vlm')
        for name, first, alias in (('input_ids', input_ids, lang_tokens),
                                   ('attention_mask', attention_mask,
                                    lang_masks), ('pixel_values', pixel_values,
                                                  images)):
            if first is not None and alias is not None:
                raise ValueError(
                    f'Pass {name} or its FluxVLA alias, not both.')
        input_ids = input_ids if input_ids is not None else lang_tokens
        attention_mask = (
            attention_mask if attention_mask is not None else lang_masks)
        pixel_values = pixel_values if pixel_values is not None else images
        if input_ids is None or labels is None:
            raise ValueError('VLM training requires input_ids and labels.')
        if labels.shape != input_ids.shape:
            raise ValueError('VLM labels must match input_ids.')
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(),
                                        self.ignore_index)
        if self.vlm_backbone is not None:
            if multimodal_indices is not None:
                raise ValueError('Native VLMs use processor visual tokens, '
                                 'not multimodal_indices.')
            if pixel_values is not None:
                inputs['pixel_values'] = pixel_values
            return self.vlm_backbone(
                task='vlm',
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                ignore_index=self.ignore_index,
                **inputs)
        if inputs:
            raise ValueError('Unsupported separate-backbone VLM fields: '
                             f'{sorted(inputs)}')
        if pixel_values is None or (multimodal_indices is not None
                                    and len(multimodal_indices) == 0):
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True)
            fused_labels = labels
        else:
            # The established fusion path inserts ignore_index at every
            # visual position and preserves the matching fused label order.
            output, _, fused_labels = self.forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                multimodal_indices=multimodal_indices,
                return_fused_labels=True,
                compute_lm_loss=False)
        return causal_lm_loss(output.logits, fused_labels, self.ignore_index)
