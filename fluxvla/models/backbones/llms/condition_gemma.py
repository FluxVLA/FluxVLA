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

from typing import Callable, Optional, Sequence, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast,
                                           TokenClassifierOutput)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging

try:
    from transformers.utils import TransformersKwargs
except ImportError:
    from transformers.utils import LossKwargs as TransformersKwargs

from fluxvla_export.components.condition_gemma import (  # noqa: F401
    GemmaAttention, GemmaDecoderLayer, GemmaMLP, GemmaPreTrainedModel,
    GemmaRMSNorm, GemmaRotaryEmbedding, _gated_residual, apply_rotary_pos_emb,
    eager_attention_forward, repeat_kv, rotate_half)
from fluxvla_export.components.condition_gemma import (ConditionGemmaModel as
                                                       _ConditionGemmaModel)

from fluxvla.engines import LLM_BACKBONES
from fluxvla.engines.utils.fsdp_wrapping import \
    build_transformer_layer_wrap_policy

logger = logging.get_logger(__name__)


@LLM_BACKBONES.register_module()
class ConditionGemmaModel(_ConditionGemmaModel):
    """Training registration and wrapping for the shared Gemma computation."""

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GemmaAttention

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (
            self.llm.model.embed_tokens,
            self.llm.model.layers[-1],
            self.llm.lm_head,
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.
        """
        return build_transformer_layer_wrap_policy(
            {self.transformer_layer_cls, nn.Parameter})


# Alias for compatibility
GemmaModel = ConditionGemmaModel


class KwargsForCausalLM(FlashAttentionKwargs, TransformersKwargs):
    ...


@auto_docstring
class GemmaForCausalLM(GemmaPreTrainedModel, GenerationMixin):
    """GemmaForCausalLM is a model for causal language modeling.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Returns:
        GemmaForCausalLM: The model.
    """
    _tied_weights_keys = ['lm_head.weight']
    _tp_plan = {'lm_head': 'colwise_rep'}
    _pp_plan = {'lm_head': (['hidden_states'], ['logits'])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`,
            *optional*):
            Labels for computing the masked language modeling loss.
            Indices should either be in `[0, ..., config.vocab_size]` or -100
            (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with
            labels in `[0, ..., config.vocab_size]`.

        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)` or
            `(batch_size, sequence_length, cond_dim)`, *optional*):
            Condition for ADARMS. A batch-level condition is broadcast across
            the sequence; a per-position condition is applied token-wise.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(
        ...     generate_ids, skip_special_tokens=True,
        ...     clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)

        # decoder outputs consists of
        # (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            adarms_cond=adarms_cond,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float
        # if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) else logits_to_keep)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(custom_intro="""
    The Gemma Model transformer with a sequence classification head on top
    (linear layer).

    [`GemmaForSequenceClassification`] uses the last token in order to do
    the classification, as other causal models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the
    position of the last token. If a `pad_token_id` is defined in the
    configuration, it finds the last token that is not a padding token in
    each row. If no `pad_token_id` is defined, it simply takes the last
    value in each row of the batch. Since it cannot guess the padding tokens
    when `inputs_embeds` are passed instead of `input_ids`, it does the same
    (take the last value in each row of the batch).
    """)
class GemmaForSequenceClassification(GemmaPreTrainedModel):
    """GemmaForSequenceClassification is a model for sequence classification.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Returns:
        GemmaForSequenceClassification: The model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adarms_cond: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed
            (Mean-Square loss), If `config.num_labels > 1` a classification
            loss is computed (Cross-Entropy).

        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)` or
            `(batch_size, sequence_length, cond_dim)`, *optional*):
            Condition for ADARMS. A batch-level condition is broadcast across
            the sequence; a per-position condition is applied token-wise.
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            adarms_cond=adarms_cond,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                'Cannot handle batch sizes > 1 if no padding token is '
                'defined.')
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost
            # token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                logits.device, torch.int32)
            token_indices = torch.arange(
                input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f'{self.__class__.__name__} will not detect padding tokens '
                'in `inputs_embeds`. Results may be unexpected if using '
                'padding tokens in conjunction with `inputs_embeds.`')

        pooled_logits = logits[torch.arange(batch_size, device=logits.device),
                               last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GemmaForTokenClassification(GemmaPreTrainedModel):
    """GemmaForTokenClassification is a model for token classification.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Returns:
        GemmaForTokenClassification: The model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        if getattr(config, 'classifier_dropout', None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, 'hidden_dropout', None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adarms_cond: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed
            (Mean-Square loss), If `config.num_labels > 1` a classification
            loss is computed (Cross-Entropy).

        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)` or
            `(batch_size, sequence_length, cond_dim)`, *optional*):
            Condition for ADARMS. A batch-level condition is broadcast across
            the sequence; a per-position condition is applied token-wise.
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            adarms_cond=adarms_cond,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    'GemmaModel',
    'GemmaForCausalLM',
    'GemmaForSequenceClassification',
    'GemmaForTokenClassification',
    'GemmaPreTrainedModel',
]
