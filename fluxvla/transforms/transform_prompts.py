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

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ProcessCosmos25Prompt:
    """Tokenize one raw prompt with the Cosmos 2.5 chat template.

    The configured tokenizer is loaded from the same diffusers component as
    :class:`Cosmos25Backbone`. The transform emits the standard FluxVLA
    ``lang_tokens`` and ``lang_masks`` sample fields.
    """

    SYSTEM_PROMPT = (
        'You are a helpful assistant who will provide prompts to an image '
        'generator.')

    def __init__(self,
                 tokenizer: Dict,
                 input_key: str = 'prompt',
                 remove_input_key: bool = False) -> None:
        from fluxvla.engines import build_tokenizer_from_cfg
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.input_key = input_key
        self.remove_input_key = bool(remove_input_key)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        tokenizer = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
        conversations = [
            {
                'role': 'system',
                'content': [{
                    'type': 'text',
                    'text': self.SYSTEM_PROMPT,
                }],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': str(prompt),
                }],
            },
        ]
        token_ids = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            add_vision_id=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding='max_length',
        )
        if isinstance(token_ids, str):
            token_ids = tokenizer(
                token_ids,
                add_special_tokens=False,
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
            )
        if isinstance(token_ids, dict):
            token_ids = token_ids['input_ids']
        elif hasattr(token_ids, 'input_ids'):
            token_ids = token_ids.input_ids
        return torch.as_tensor(token_ids, dtype=torch.long)

    def __call__(self, inputs: Dict) -> Dict:
        if self.input_key not in inputs:
            raise KeyError(
                f"ProcessCosmos25Prompt requires '{self.input_key}'.")
        token_ids = self._tokenize_prompt(inputs[self.input_key])
        if token_ids.ndim == 2 and token_ids.shape[0] == 1:
            token_ids = token_ids.squeeze(0)
        if token_ids.ndim != 1:
            raise ValueError('ProcessCosmos25Prompt expected one token '
                             f'sequence, got shape {tuple(token_ids.shape)}.')

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        else:
            token_mask = token_ids.ne(int(pad_token_id))
        inputs['lang_tokens'] = token_ids.cpu().numpy().astype(
            np.int64, copy=False)
        inputs['lang_masks'] = token_mask.cpu().numpy().astype(
            np.bool_, copy=False)
        if self.remove_input_key:
            inputs.pop(self.input_key, None)
        return inputs


@TRANSFORMS.register_module()
class QwenVLImageTokenExpandAndTokenize:
    """Tokenize Qwen-VL text after expanding image placeholders.

    Qwen3-VL expands every ``<|image_pad|>`` according to the corresponding
    ``image_grid_thw`` row before tokenization. This transform exposes that
    processor step as a config-visible text transform while leaving image
    preprocessing to a preceding image transform.
    """

    def __init__(
        self,
        tokenizer: Dict,
        text_key: str = 'text',
        image_grid_thw_key: str = 'image_grid_thw',
        input_ids_key: str = 'input_ids',
        attention_mask_key: str = 'attention_mask',
        mm_token_type_ids_key: Optional[str] = None,
        expanded_text_key: Optional[str] = None,
        image_token: str = '<|image_pad|>',
        video_token: str = '<|video_pad|>',
        placeholder_token: str = '<|placeholder|>',
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        image_mm_token_type_id: int = 1,
        video_mm_token_type_id: int = 2,
        merge_size: int = 2,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_tensors: str = 'np',
        squeeze_batch: bool = True,
        strict_num_images: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        padding_side: str = 'left',
        use_eos_as_pad: bool = False,
        output_keys: Optional[List[str]] = None,
    ) -> None:
        from fluxvla.engines import build_tokenizer_from_cfg

        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.text_key = text_key
        self.image_grid_thw_key = image_grid_thw_key
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.mm_token_type_ids_key = mm_token_type_ids_key
        self.expanded_text_key = expanded_text_key
        self.image_token = image_token
        self.video_token = video_token
        self.placeholder_token = placeholder_token
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_mm_token_type_id = image_mm_token_type_id
        self.video_mm_token_type_id = video_mm_token_type_id
        self.merge_size = merge_size
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.return_tensors = return_tensors
        self.squeeze_batch = squeeze_batch
        self.strict_num_images = strict_num_images
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.output_keys = tuple(
            output_keys) if output_keys is not None else None

        tokenizer_obj = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
        if hasattr(tokenizer_obj, 'padding_side'):
            tokenizer_obj.padding_side = padding_side
        if use_eos_as_pad and getattr(tokenizer_obj, 'pad_token_id',
                                      None) is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        if self.image_token_id is None:
            self.image_token_id = tokenizer_obj.convert_tokens_to_ids(
                self.image_token)
        if self.video_token_id is None:
            self.video_token_id = tokenizer_obj.convert_tokens_to_ids(
                self.video_token)

    def _expand_text(self, text: str, image_grid_thw: Any) -> str:
        grids = np.asarray(image_grid_thw)
        if grids.ndim == 1:
            grids = grids.reshape(1, -1)
        if grids.ndim != 2:
            raise ValueError(
                f'{self.image_grid_thw_key} must be 1D or 2D, got '
                f'shape {grids.shape}')

        merge_length = self.merge_size**2
        expanded = str(text)
        image_index = 0
        while self.image_token in expanded:
            if image_index >= len(grids):
                raise ValueError(
                    f'Text contains more {self.image_token!r} tokens than '
                    f'{self.image_grid_thw_key} rows.')
            num_image_tokens = int(np.prod(grids[image_index]) // merge_length)
            expanded = expanded.replace(
                self.image_token,
                self.placeholder_token * num_image_tokens,
                1,
            )
            image_index += 1
        expanded = expanded.replace(self.placeholder_token, self.image_token)

        if self.strict_num_images and image_index != len(grids):
            raise ValueError(
                f'Text contains {image_index} {self.image_token!r} tokens, '
                f'but {self.image_grid_thw_key} has {len(grids)} rows.')
        return expanded

    def __call__(self, inputs: Dict) -> Dict:
        if self.text_key not in inputs:
            raise KeyError(f'Missing text key: {self.text_key!r}')
        if self.image_grid_thw_key not in inputs:
            raise KeyError(
                f'Missing image grid key: {self.image_grid_thw_key!r}')

        expanded_text = self._expand_text(inputs[self.text_key],
                                          inputs[self.image_grid_thw_key])
        encoded = self.tokenizer(
            [expanded_text],
            padding=self.padding,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
            **self.tokenizer_kwargs,
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        if self.squeeze_batch:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]

        if self.padding == 'max_length':
            max_length = self.tokenizer_kwargs.get('max_length')
            if max_length is None:
                raise ValueError(
                    "padding='max_length' requires tokenizer_kwargs.max_length"
                )
            if input_ids.shape[0] != max_length:
                raise ValueError(
                    'Tokenized Qwen-VL input has length '
                    f'{input_ids.shape[0]}, '
                    f'expected {max_length}. Increase max_length rather than '
                    'truncating image tokens.')

        inputs[self.input_ids_key] = np.asarray(input_ids, dtype=np.int64)
        inputs[self.attention_mask_key] = np.asarray(
            attention_mask, dtype=np.int64)
        if self.mm_token_type_ids_key is not None:
            mm_token_type_ids = np.zeros_like(
                inputs[self.input_ids_key], dtype=np.int64)
            mm_token_type_ids[inputs[self.input_ids_key] ==
                              self.image_token_id] = (
                                  self.image_mm_token_type_id)
            if self.video_token_id is not None:
                mm_token_type_ids[inputs[self.input_ids_key] ==
                                  self.video_token_id] = (
                                      self.video_mm_token_type_id)
            inputs[self.mm_token_type_ids_key] = mm_token_type_ids
        if self.expanded_text_key is not None:
            inputs[self.expanded_text_key] = expanded_text
        if self.output_keys is not None:
            inputs = {
                key: inputs[key]
                for key in self.output_keys if key in inputs
            }
        return inputs


@TRANSFORMS.register_module()
class ProcessPrompts():
    """Process and tokenize prompts for language models.
    This class handles the tokenization of prompts using a specified
    tokenizer, ensuring that the tokenized output adheres to a maximum
    length. It can also optionally prepare labels for language modeling
    tasks by creating a copy of the tokenized input.

    Args:
        tokenizer (Dict): Configuration for building the tokenizer.
        max_len (int, optional): Maximum length for tokenized prompts.
            Defaults to 180.
        with_labels (bool, optional): Whether to include labels in
            the output. Defaults to False.
    """

    def __init__(self,
                 tokenizer: Dict,
                 model_path: str = None,
                 max_len: int = 180,
                 with_labels: bool = False,
                 with_state: bool = False,
                 ignore_index: int = -100,
                 negative_prompt=None):
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.with_labels = with_labels
        self.with_state = with_state
        self.ignore_index = ignore_index
        self.negative_prompt = negative_prompt

    def _tokenize_single_prompt(self,
                                prompt: str,
                                state: np.ndarray | None = None):
        if state is not None:
            tokens = self.tokenizer(
                prompt, state=state, add_special_tokens=True)['input_ids']
        else:
            tokens = self.tokenizer(
                prompt, add_special_tokens=True)['input_ids']
        token_mask = [True] * len(tokens)
        tokens_len = len(tokens)
        if self.max_len is not None:
            if tokens_len < self.max_len:
                padding = [False] * (self.max_len - tokens_len)
                tokens = tokens + padding
                token_mask = token_mask + padding
            else:
                tokens = tokens[:self.max_len]
                token_mask = token_mask[:self.max_len]
        return tokens, token_mask

    def __call__(self, inputs):
        """Tokenize and process the prompt in the input data.
        The method tokenizes the 'prompt' field in the input dictionary,
        applies padding or truncation to meet the maximum length requirement,
        and optionally creates a 'labels' field for language modeling tasks.

        Args:
            inputs (Dict): Input data containing a 'prompt' key.
        """
        assert 'prompt' in inputs, "Data must contain 'prompt' key."
        if self.with_state:
            assert 'state' in inputs, "Data must contain 'state' key."
            state = inputs['state']
        else:
            state = None
        tokens, token_mask = self._tokenize_single_prompt(
            inputs['prompt'], state)
        lang_tokens = [tokens]
        lang_masks = [token_mask]
        if self.negative_prompt is not None:
            negative_tokens, negative_token_mask = (
                self._tokenize_single_prompt(self.negative_prompt, state))
            lang_tokens.append(negative_tokens)
            lang_masks.append(negative_token_mask)
        labels = list(tokens)
        inputs['lang_tokens'] = np.array(tokens)
        inputs['lang_masks'] = np.array(token_mask)
        if self.with_labels:
            assert 'actions' in inputs, "Data must contain 'actions' key."
            actions = inputs['actions']
            inputs['labels'] = np.array(labels)
            inputs['labels'][:-(len(actions[0]) + 1)] = self.ignore_index
        return inputs


@TRANSFORMS.register_module()
class ProcessPromptsWithImage:
    """Process and tokenize prompts with image context for language models.
    This class handles the tokenization of prompts that include image context
    using a specified tokenizer. It constructs a structured text format that
    incorporates system prompts, user prompts, and image placeholders. The
    tokenized output adheres to a maximum length, and it can optionally prepare
    labels for language modeling tasks.
    The text format is inspired by the GR00T model's input structure.

    Args:
        tokenizer (Dict): Configuration for building the tokenizer.
        max_len (int, optional): Maximum length for tokenized prompts.
            Defaults to 180.
        with_labels (bool, optional): Whether to include labels in
            the output. Defaults to False.
        # ===== GR00T text template options =====
        add_system (bool, optional): Whether to add a system prompt section.
            Defaults to True.
        system_prompt (str, optional): The system prompt text.
            Defaults to "You are a helpful assistant.".
        add_assistant_stub (bool, optional): Whether to append an assistant
            stub at the end. Defaults to True.
        task_pos (str, optional): Position of the task description relative
            to images ('after_images' or 'before_images').
            Defaults to 'after_images'.
        front_eos_repeat (int, optional): Number of <|endoftext|>
            tokens to prepend.
            Defaults to 0.
        eos_token_str (str, optional): String representation of the
            end-of-text token.
            Defaults to "<|endoftext|>".
        im_start (str, optional): String marking the start of an
            image section.
            Defaults to "<|im_start|>".
        im_end (str, optional): String marking the end of an
            image section.
            Defaults to "<|im_end|>".
        image_tag_template (str, optional): Template for image tags.
            Defaults to "<image {i}>".
        img_start (str, optional): String marking the start of
            image context.
            Defaults to "<img>".
        img_end (str, optional): String marking the end of
            image context.
            Defaults to "</img>".
        img_context_token (str, optional): Token representing
            image context.
            Defaults to "<IMG_CONTEXT>".
        # ===== image token expansion options =====
        img_tokens_source (str, optional): Source for determining
            the number of image tokens ('from_inputs' or 'fixed').
            Defaults to 'fixed'.
        fixed_img_tokens (int, optional): Fixed number of tokens per image
            if img_tokens_source is 'fixed'. Defaults to 256.
        num_images (int, optional): Number of images to include.
            Defaults to 2.
        # ===== tokenization/padding options =====
        pad_to_max_len (bool, optional): Whether to pad
            sequences to max_len. Defaults to True.
        padding_side (str, optional): Side to apply padding
            ('left' or 'right'). Defaults to 'left'.
        use_eos_as_pad (bool, optional): Whether to use the
            EOS token as the padding token if no pad token
            is defined. Defaults to True.
        return_text (bool, optional): Whether to include
            the constructed text in the output for debugging.
            Defaults to False.
    """

    def __init__(
            self,
            tokenizer: Dict,
            max_len: int = 180,
            with_labels: bool = False,
            # ===== GR00T text template options =====
            add_system: bool = True,
            system_prompt: str = 'You are a helpful assistant.',
            add_assistant_stub: bool = True,
            task_pos: str = 'after_images',
            front_eos_repeat: int = 0,
            # tag definitions
            eos_token_str: str = '<|endoftext|>',
            im_start: str = '<|im_start|>',
            im_end: str = '<|im_end|>',
            image_tag_template: str = '<image {i}>',
            img_start: str = '<img>',
            img_end: str = '</img>',
            img_context_token: str = '<IMG_CONTEXT>',
            # ===== image token expansion options =====
            # 'from_inputs': read inputs['num_image_tokens']
            # 'fixed': always use fixed_img_tokens
            img_tokens_source: str = 'fixed',
            fixed_img_tokens: Optional[int] = 256,
            num_images: Optional[int] = 3,
            num_image_tokens_key: str = 'num_image_tokens',
            image_grid_thw_key: str = 'image_grid_thw',
            image_merge_size: int = 2,
            # ===== tokenization/padding options =====
            pad_to_max_len: bool = True,
            padding_side: str = 'left',  # 'left' or 'right'
            use_eos_as_pad: bool = True,
            truncate: bool = True,
            lowercase_task_description: bool = False,
            strip_task_punctuation: bool = False,
            attention_mask_dtype: str = 'int32',
            return_text: bool = False,
            output_keys: Optional[List[str]] = None,
            model_path=None):  # noqa: E129
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.with_labels = with_labels

        self.add_system = add_system
        self.system_prompt = system_prompt
        self.add_assistant_stub = add_assistant_stub
        self.task_pos = task_pos

        self.front_eos_repeat = front_eos_repeat
        self.eos_token_str = eos_token_str
        self.im_start = im_start
        self.im_end = im_end
        self.image_tag_template = image_tag_template
        self.img_start = img_start
        self.img_end = img_end
        self.img_context_token = img_context_token

        self.img_tokens_source = img_tokens_source
        self.num_images = num_images
        self.fixed_img_tokens = fixed_img_tokens
        self.num_image_tokens_key = num_image_tokens_key
        self.image_grid_thw_key = image_grid_thw_key
        self.image_merge_size = image_merge_size

        self.pad_to_max_len = pad_to_max_len
        self.padding_side = padding_side
        self.use_eos_as_pad = use_eos_as_pad
        self.truncate = truncate
        self.lowercase_task_description = lowercase_task_description
        self.strip_task_punctuation = strip_task_punctuation
        self.attention_mask_dtype = np.dtype(attention_mask_dtype)
        self.return_text = return_text
        self.output_keys = tuple(
            output_keys) if output_keys is not None else None

        # If tokenizer has no pad_token, use eos as pad (common trick)
        if self.tokenizer.pad_token_id is None and self.use_eos_as_pad:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # === Build GR00T-style text string ===
    def _build_text(self, task_desc: str,
                    num_tokens_per_image: List[int]) -> str:
        parts = []
        # (0) prepend eos repeats
        if self.front_eos_repeat > 0:
            parts.append(self.eos_token_str * self.front_eos_repeat)

        # (1) system section
        if self.add_system:
            parts.append(
                f'{self.im_start}system\n{self.system_prompt}{self.im_end}\n')

        # (2) user section
        parts.append(f'{self.im_start}user\n')

        if self.task_pos == 'before_images' and task_desc:
            parts.append(task_desc + '\n')

        # insert images: <image i><img> + repeated <IMG_CONTEXT> + </img>
        for i, n_tok in enumerate(num_tokens_per_image, start=1):
            parts.append(self.image_tag_template.format(i=i))
            parts.append(self.img_start)
            parts.append(self.img_context_token * int(n_tok))
            parts.append(self.img_end)

        if self.task_pos == 'after_images' and task_desc:
            parts.append(task_desc)

        parts.append(f'{self.im_end}\n')

        # (3) assistant stub
        if self.add_assistant_stub:
            parts.append(f'{self.im_start}assistant\n')

        return ''.join(parts)

    def _resolve_image_token_counts(self, inputs: Dict) -> List[int]:
        if self.img_tokens_source == 'fixed':
            if self.fixed_img_tokens is None or self.num_images is None:
                raise ValueError(
                    'fixed image tokens require fixed_img_tokens and '
                    'num_images.')
            return [int(self.fixed_img_tokens)] * int(self.num_images)
        if self.img_tokens_source == 'from_inputs':
            if self.num_image_tokens_key not in inputs:
                raise KeyError(
                    f'Missing image token key: {self.num_image_tokens_key!r}')
            counts = np.asarray(inputs[self.num_image_tokens_key]).reshape(-1)
            return [int(value) for value in counts]
        if self.img_tokens_source == 'from_image_grid_thw':
            if self.image_grid_thw_key not in inputs:
                raise KeyError(
                    f'Missing image grid key: {self.image_grid_thw_key!r}')
            grids = np.asarray(inputs[self.image_grid_thw_key])
            if grids.ndim == 1:
                grids = grids[None, :]
            if grids.ndim != 2 or grids.shape[-1] != 3:
                raise ValueError(
                    f'{self.image_grid_thw_key} must have shape [N, 3], '
                    f'got {grids.shape}.')
            merge_length = self.image_merge_size**2
            return [int(np.prod(grid) // merge_length) for grid in grids]
        raise ValueError(
            f'Unsupported img_tokens_source: {self.img_tokens_source!r}')

    def __call__(self, inputs: Dict):
        """Tokenize and process the prompt with image
        context in the input data.The method constructs a
        structured text format incorporating system
        prompts, user prompts, and image placeholders.
        It then tokenizes this text, applies padding or
        truncation to meet the maximum length
        requirement, and optionally creates a 'labels'
        field for language modeling tasks.
        """
        assert 'task_description' in inputs, "inputs must contain 'task_description'"  # noqa: E501

        # (1) resolve per-image token counts
        per_img = self._resolve_image_token_counts(inputs)

        # (2) build GR00T-style text
        task_description = str(inputs['task_description'])
        if self.lowercase_task_description:
            task_description = task_description.lower()
        if self.strip_task_punctuation:
            task_description = re.sub(r'[^\w\s]', '', task_description)
        text = self._build_text(task_description, per_img)

        # (3) tokenize
        encoded = self.tokenizer(text, add_special_tokens=True)
        tokens = encoded.input_ids
        mask = [1] * len(tokens)
        labels = list(tokens)

        # (4) pad/truncate
        if self.max_len is not None and self.pad_to_max_len:
            L = len(tokens)
            pad_id = self.tokenizer.pad_token_id if \
                self.tokenizer.pad_token_id is not None \
                else self.tokenizer.eos_token_id
            if L < self.max_len:
                pad_len = self.max_len - L
                if self.padding_side == 'left':
                    tokens = [pad_id] * pad_len + tokens
                    mask = [0] * pad_len + mask
                    if self.with_labels:
                        labels = [-100] * pad_len + labels
                else:
                    tokens = tokens + [pad_id] * pad_len
                    mask = mask + [0] * pad_len
                    if self.with_labels:
                        labels = labels + [-100] * pad_len
            else:
                if not self.truncate and L > self.max_len:
                    raise ValueError(
                        f'Tokenized prompt has length {L}, exceeding '
                        f'max_len={self.max_len}.')
                if self.padding_side == 'left':
                    tokens = tokens[-self.max_len:]
                    mask = mask[-self.max_len:]
                    if self.with_labels:
                        labels = labels[-self.max_len:]
                else:
                    tokens = tokens[:self.max_len]
                    mask = mask[:self.max_len]
                    if self.with_labels:
                        labels = labels[:self.max_len]

        inputs['lang_tokens'] = np.asarray(tokens, dtype=np.int64)
        inputs['lang_masks'] = np.asarray(
            mask, dtype=self.attention_mask_dtype)
        if self.with_labels:
            inputs['labels'] = np.asarray(labels, dtype=np.int64)
        if self.return_text:
            inputs['text'] = text
        if self.output_keys is not None:
            inputs = {
                key: inputs[key]
                for key in self.output_keys if key in inputs
            }
        return inputs


@TRANSFORMS.register_module()
class LiberoPromptFromInputs:
    """Build and tokenize Libero evaluation prompt.

    Constructs: "In: What action should the robot
    take to {task} ?\nOut:" + suffix
    Pads/truncates to max_len and outputs
    'lang_tokens' and 'lang_masks'.

    Args:
        tokenizer (Dict): Tokenizer config for build_tokenizer_from_cfg.
        max_len (int): Maximum token length.
        pad_token_id (int): Pad id to use if padding needed.
        prompt_suffix (str): Suffix appended after 'Out:'.
    """

    def __init__(self,
                 tokenizer: Dict,
                 model_path: str = None,
                 max_len: int = 180,
                 pad_token_id: int = 0,
                 prompt_suffix: str = '',
                 use_conversation: bool = True,
                 negative_prompt: str = None,
                 add_new_line: bool = False,
                 prompt_template: str = None) -> None:
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.prompt_suffix = prompt_suffix
        self.use_conversation = use_conversation
        self.negative_prompt = negative_prompt
        self.add_new_line = add_new_line
        self.prompt_template = prompt_template

    def _tokenize_single_prompt(self, prompt: str):
        token_ids = self.tokenizer(prompt)['input_ids']
        mask = [True] * len(token_ids)
        if self.max_len is not None:
            if len(token_ids) < self.max_len:
                pad_len = self.max_len - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_len
                mask = mask + [False] * pad_len
            else:
                token_ids = token_ids[:self.max_len]
                mask = mask[:self.max_len]
        return token_ids, mask

    def __call__(self, inputs: Dict) -> Dict:
        assert 'task_description' in inputs, "inputs must contain 'task_description'"  # noqa: E501
        task_description = inputs['task_description']
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(task=task_description)
        elif self.use_conversation:
            prompt = ('In: What action should the robot take to ' +
                      str(task_description).lower() + '?\nOut:' +
                      self.prompt_suffix)
        else:
            prompt = task_description
        if self.add_new_line:
            prompt += '\n'
        tokens, token_mask = self._tokenize_single_prompt(prompt)
        if self.negative_prompt is not None:
            negative_tokens, negative_token_mask = (
                self._tokenize_single_prompt(self.negative_prompt))
            tokens = [tokens, negative_tokens]
            token_mask = [token_mask, negative_token_mask]

        inputs['lang_tokens'] = np.asarray(tokens, dtype=np.int64)
        inputs['lang_masks'] = np.asarray(token_mask, dtype=np.bool_)
        return inputs


@TRANSFORMS.register_module()
class TokenizeText:
    """Tokenize task text for CLIP-based SARM training and inference."""

    def __init__(self,
                 tokenizer: Dict,
                 max_length: int = 77,
                 text_key: str = 'task_description',
                 output_ids_key: str = 'text_input_ids',
                 output_attention_mask_key: str = 'text_attention_mask'):
        """Initialize text tokenization.

        Args:
            tokenizer (Dict): Tokenizer config passed to the FluxVLA registry.
            max_length (int): Maximum token sequence length.
            text_key (str): Input text key in the sample dictionary.
            output_ids_key (str): Output key for token ids.
            output_attention_mask_key (str): Output key for attention masks.
        """
        from fluxvla.engines import build_tokenizer_from_cfg
        from fluxvla.engines.utils.hf_hub import resolve_hf_local_path
        tokenizer = dict(tokenizer)
        model_path = tokenizer.get('model_path')
        if isinstance(model_path, str):
            tokenizer['model_path'] = resolve_hf_local_path(model_path)
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_length = max_length
        self.text_key = text_key
        self.output_ids_key = output_ids_key
        self.output_attention_mask_key = output_attention_mask_key

    def __call__(self, data: Dict) -> Dict:
        """Tokenize the configured text field.

        Args:
            data (Dict): Sample dictionary containing ``text_key``.

        Returns:
            Dict: Sample dictionary with token ids and attention mask.
        """
        encoded = self.tokenizer(
            data[self.text_key],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np',
        )
        data[self.output_ids_key] = encoded['input_ids'][0].astype(np.int64)
        data[self.
             output_attention_mask_key] = encoded['attention_mask'][0].astype(
                 np.int64)
        return data
