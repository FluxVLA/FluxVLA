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

import math

import torch
import torch.nn.functional as F

from fluxvla.engines import VLAS
from fluxvla.engines.losses.causal_lm import causal_lm_loss
from fluxvla.engines.utils.model_utils import (create_sinusoidal_pos_embedding,
                                               make_att_2d_masks)
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi0_flowmatching import PI0FlowMatching

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatching(PI0FlowMatching):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2504.16054

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self, **kwargs):
        assert kwargs.get('state_proj') is None, (
            'PI05FlowMatching does not use state_proj.')
        super().__init__(**kwargs)

    @property
    def config(self):
        return self.llm_backbone.config

    def validate_training_task(self, task):
        if task == 'vla':
            return
        if task not in ('vlm', 'mixed'):
            raise ValueError(f'Unknown training task: {task}')
        if (self.vlm_backbone is not None or self.llm_backbone is None
                or not hasattr(self.llm_backbone, 'embed_tokens')):
            raise ValueError('Pi05 VLM training requires the separate Gemma '
                             'backbone with its pretrained token embeddings.')
        if getattr(self.llm_backbone.norm, 'cond_dim', None) is not None:
            raise ValueError(
                'The Pi05 language backbone must be unconditional '
                'for VLM training.')

    def prepare_training_task(self, task):
        self.validate_training_task(task)
        self._training_task = task
        if task == 'mixed':
            # The expert consumes projected actions and never embeds tokens.
            self.llm_expert.embed_tokens.requires_grad_(False)
        if task == 'vlm':
            active = {
                id(parameter)
                for module in (self.vision_backbone, self.projector,
                               self.llm_backbone) if module is not None
                for parameter in module.parameters()
            }
            # Preserve configured freezes, including after LoRA injection.
            for parameter in self.parameters():
                if id(parameter) not in active:
                    parameter.requires_grad_(False)

    def inactive_training_parameters(self, task):
        """Parameters skipped by a task, including FSDP's zero grad slices."""
        if task == 'vlm':
            active = {
                id(p)
                for module in (self.vision_backbone, self.projector,
                               self.llm_backbone) if module is not None
                for p in module.parameters()
            }
            return [p for p in self.parameters() if id(p) not in active]
        return list(self.llm_expert.embed_tokens.parameters())

    def get_vlm_fsdp_wrapping_policy(self):
        self.validate_training_task('vlm')
        language_layers = set(self.llm_backbone.layers)

        def wrap_policy(module, recurse, nonwrapped_numel):
            # Keep the tied embedding/output weight, conditional vision and
            # unused action expert in the root unit. Only active language
            # decoder layers have independent forward/all-gather boundaries.
            return recurse or module in language_layers

        return wrap_policy

    def get_mixed_fsdp_wrapping_policy(self):
        """Shard MLP calls shared by joint attention and causal decoding.

        Keep vision in the root: ranks can have text-only/image VLM batches
        in the same step. The expert MLPs run on every rank's VLA steps and
        are skipped on every rank's VLM steps. Gemma attention weights and
        the tied vocabulary head stay in the root for direct weight access.
        """
        self.validate_training_task('mixed')
        mlps = {
            layer.mlp
            for backbone in (self.llm_backbone, self.llm_expert)
            for layer in backbone.layers
        }

        def wrap_policy(module, recurse, nonwrapped_numel):
            return recurse or module in mlps

        return wrap_policy

    def get_fsdp_execution_block_wrapping_policy(self):
        if getattr(self, '_training_task', 'vla') == 'vlm':
            return self.get_vlm_fsdp_wrapping_policy()
        return super().get_fsdp_execution_block_wrapping_policy()

    def forward(self, *args, task='vla', **batch):
        if task == 'mixed':
            raise ValueError('Mixed training dispatches one vla or vlm '
                             'task per forward call.')
        self.validate_training_task(task)
        if task == 'vlm':
            return self.forward_vlm(*args, **batch)
        return self.forward_vla(*args, **batch)

    def forward_vla(self, *args, **batch):
        return super().forward(*args, **batch)

    def forward_vlm(self,
                    *,
                    labels,
                    input_ids=None,
                    attention_mask=None,
                    images=None,
                    img_masks=None,
                    lang_tokens=None,
                    lang_masks=None):
        """Train the shared SigLIP/Gemma with a causal text objective.

        Images form a bidirectional prefix; every text token can attend only
        to that prefix and preceding text. In particular, vision tokens never
        see answer tokens. Labels cover text only and are shifted once by
        causal_lm_loss. PaliGemma ties its output head to the token embedding,
        so no new parameters or checkpoint keys are required.
        """
        self.validate_training_task('vlm')
        if input_ids is not None and lang_tokens is not None:
            raise ValueError('Pass input_ids or lang_tokens, not both.')
        if attention_mask is not None and lang_masks is not None:
            raise ValueError('Pass attention_mask or lang_masks, not both.')
        tokens = input_ids if input_ids is not None else lang_tokens
        mask = attention_mask if attention_mask is not None else lang_masks
        if tokens is None or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError(
                'Pi05 VLM input_ids must be int64 with shape (B,L).')
        if labels.shape != tokens.shape:
            raise ValueError('VLM labels must match input_ids.')
        if mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        if mask.shape != tokens.shape:
            raise ValueError('VLM attention_mask must match input_ids.')
        mask = mask.bool()
        labels = labels.masked_fill(~mask, self.ignore_index)
        embeddings, padding, ar_mask = self._embed_vlm_prefix(
            tokens, mask, images, img_masks)
        attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(padding, ar_mask))
        positions = (padding.long().cumsum(-1) - 1).clamp_min(0)
        output = self.llm_backbone(
            inputs_embeds=self._cast_gemma_input(embeddings),
            attention_mask=attention,
            position_ids=positions,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True)
        hidden = output.last_hidden_state[:, -tokens.shape[1]:]
        logits = F.linear(hidden, self.llm_backbone.embed_tokens.weight)
        return causal_lm_loss(logits, labels, self.ignore_index)

    def _embed_vlm_prefix(self, tokens, mask, images, img_masks):
        """Share image/text embeddings and attention between SFT and decode."""
        if images is None:
            if img_masks is not None:
                raise ValueError('img_masks requires images.')
            embeddings = self.llm_backbone.embed_tokens(tokens)
            embeddings = embeddings * math.sqrt(embeddings.shape[-1])
            padding = mask
            ar_mask = torch.ones_like(mask)
        else:
            if (images.ndim != 4 or images.shape[0] != tokens.shape[0]
                    or images.shape[1] == 0 or images.shape[1] % 3):
                raise ValueError(
                    'Pi05 VLM images must have shape (B,3*N,H,W).')
            if self.vision_backbone is None or self.projector is None:
                raise ValueError('Image SFT requires vision and projector.')
            if img_masks is None:
                img_masks = torch.ones(
                    tokens.shape[0],
                    images.shape[1] // 3,
                    dtype=torch.bool,
                    device=images.device)
            if img_masks.shape != (tokens.shape[0], images.shape[1] // 3):
                raise ValueError('img_masks must have shape (B,N).')
            embeddings, padding, ar_mask = self.embed_prefix(
                images, tokens, img_masks.bool(), mask)
            ar_mask = ar_mask.clone()
            ar_mask[:, -tokens.shape[1]:] = True
        return embeddings, padding, ar_mask

    @torch.inference_mode()
    def generate_text(self,
                      input_ids,
                      attention_mask=None,
                      images=None,
                      img_masks=None,
                      max_new_tokens=32,
                      eos_token_id=None,
                      pad_token_id=None,
                      do_sample=False):
        """Greedy text decoding with a KV cache; return prompt + new token IDs.

        Both left and right padded prompts are supported. Finished rows emit
        padding after EOS. Image encoding runs once, using the same causal
        text / bidirectional image-prefix attention as ``forward_vlm``.
        """
        self.validate_training_task('vlm')
        if self.training:
            raise ValueError('Call model.eval() before generate_text.')
        if do_sample:
            raise ValueError('Pi05 generate_text currently supports greedy '
                             'decoding only (do_sample=False).')
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer.')
        if (input_ids.ndim != 2 or input_ids.dtype != torch.long
                or input_ids.shape[1] == 0):
            raise ValueError('input_ids must be nonempty int64 (B,L).')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask = attention_mask.bool()
        if mask.shape != input_ids.shape or not mask.any(-1).all():
            raise ValueError('Each prompt needs at least one unmasked token.')
        eos = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id)
        pad = (
            self.config.pad_token_id if pad_token_id is None else pad_token_id)
        pad = 0 if pad is None else pad
        embeddings, padding, ar_mask = self._embed_vlm_prefix(
            input_ids, mask, images, img_masks)
        if (embeddings.shape[1] + max_new_tokens - 1 >
                self.config.max_position_embeddings):
            raise ValueError('Prompt plus generation exceeds context length.')
        attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(padding, ar_mask))
        positions = (padding.long().cumsum(-1) - 1).clamp_min(0)
        output = self.llm_backbone(
            inputs_embeds=self._cast_gemma_input(embeddings),
            attention_mask=attention,
            position_ids=positions,
            use_cache=True,
            return_dict=True)
        indices = torch.arange(mask.shape[1], device=mask.device)
        last = indices.expand_as(mask).masked_fill(~mask, -1).max(-1).values
        last += embeddings.shape[1] - input_ids.shape[1]
        hidden = output.last_hidden_state[
            torch.arange(mask.shape[0], device=mask.device), last]
        cache = output.past_key_values
        finished = torch.zeros(
            mask.shape[0], dtype=torch.bool, device=mask.device)
        generated = []
        for step in range(max_new_tokens):
            logits = F.linear(hidden, self.llm_backbone.embed_tokens.weight)
            token = logits.argmax(-1).masked_fill(finished, pad)
            generated.append(token)
            if eos is not None:
                finished |= token == eos
            if finished.all() or step + 1 == max_new_tokens:
                break
            positions = padding.long().sum(-1, keepdim=True)
            padding = torch.cat([padding, (~finished)[:, None]], dim=1)
            embeddings = self.llm_backbone.embed_tokens(token[:, None])
            embeddings = embeddings * math.sqrt(embeddings.shape[-1])
            output = self.llm_backbone(
                inputs_embeds=self._cast_gemma_input(embeddings),
                attention_mask=self._prepare_attention_masks_4d(
                    padding[:, None, :]),
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True)
            hidden = output.last_hidden_state[:, -1]
            cache = output.past_key_values
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,)
                for scalar time, or (bsize, n_action_steps) for per-position
                time (used in training-time RTC).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor]: A tuple containing the embedded suffix
                tokens, padding masks, attention masks, and adarms_cond.
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = states.shape[0]
        device = states.device

        # Set attention masks so that image and language
        # inputs do not attend to action tokens

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        if self.openpi_fp32_flow:
            with self._disable_autocast(noisy_actions):
                time_emb = create_sinusoidal_pos_embedding(
                    timestep.float(),
                    self.proj_width,
                    min_period=4e-3,
                    max_period=4.0,
                    device=device,
                    dtype=torch.float32)
                action_emb = self.action_in_proj(noisy_actions.float())
                time_emb = F.silu(self.time_mlp_in(time_emb))
                time_emb = F.silu(self.time_mlp_out(time_emb))
        else:
            dtype = states.dtype
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.proj_width,
                min_period=4e-3,
                max_period=4.0,
                device=device)
            time_emb = time_emb.type(dtype=dtype)

            # Fuse timestep + action information using an MLP
            action_emb = self.action_in_proj(noisy_actions)
            time_emb = F.silu(self.time_mlp_in(time_emb))
            time_emb = F.silu(self.time_mlp_out(time_emb))
        # Add to input tokens
        embs.append(action_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image and language inputs do not attend
        # to action tokens.  Derive the mask from the supplied action window
        # because evaluation and truncated episodes can use a horizon
        # different from the training configuration.
        if action_time_dim == 0:
            raise ValueError('noisy_actions must contain at least one step')
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, time_emb
