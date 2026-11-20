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
"""Local, subtask-supervised VLM backend for the ordinary planner interface."""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.root import PLANNERS
from .backends import _planner_camera_observation, _validate_image_keys


def resolve_planner_checkpoint(path):
    """Resolve a published FluxVLA checkpoint; never load base HF weights."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        published = path / 'latest-checkpoint.safetensors'
        if published.is_symlink() and published.is_file():
            return published.resolve()
        checkpoints = []
        for candidate in path.glob('step-*.safetensors'):
            match = re.fullmatch(r'step-(\d+)-.+\.safetensors', candidate.name)
            if match and candidate.is_file():
                checkpoints.append((int(match[1]), candidate.name, candidate))
        if not checkpoints:
            raise FileNotFoundError(f'No planner checkpoints in {path}')
        return max(checkpoints)[-1]
    if not path.is_file():
        raise FileNotFoundError(f'Planner checkpoint not found: {path}')
    if path.suffix != '.safetensors':
        raise ValueError('Local VLM planner requires a full FluxVLA '
                         '.safetensors checkpoint, not a LoRA adapter')
    return path


class _CandidateTrie:
    """Decode only literal candidate texts, not IDs or fuzzy replacements."""

    def __init__(self, tokenizer, candidates, eos_token_id, max_new_tokens):
        self.root = {}
        self.choices = {}
        self.texts = {}
        self.longest = 0
        seen_ids = set()
        for candidate in candidates:
            choice = candidate.get('choice')
            instruction = candidate.get('instruction')
            if (not isinstance(choice, str) or not choice
                    or choice in {'blocked', 'request_finish'}
                    or choice in seen_ids or not isinstance(instruction, str)
                    or not instruction.strip()):
                raise ValueError('Expected distinct candidate IDs and literal '
                                 'training instructions')
            seen_ids.add(choice)
            if instruction in self.texts:
                raise ValueError('Duplicate candidate instructions')
            ids = tokenizer.encode(instruction, add_special_tokens=False)
            if not ids or any(token in tokenizer.all_special_ids
                              for token in ids):
                raise ValueError('Candidate text contains special tokens')
            # No punctuation/case/semantic normalization: preserve VLA labels.
            decoded = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            if decoded != instruction:
                raise ValueError('Candidate tokenizer does not preserve the '
                                 'literal training instruction')
            sequence = tuple(ids) + (eos_token_id, )
            if sequence in self.choices:
                raise ValueError('Candidates have ambiguous token sequences')
            self.choices[sequence] = choice
            self.texts[instruction] = choice
            self.longest = max(self.longest, len(sequence))
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})
        if not self.choices:
            raise ValueError('Local planner requires current-task candidates')
        if self.longest > max_new_tokens:
            raise ValueError(
                f'Training candidates need up to {self.longest} '
                f'tokens including EOS; raise max_new_tokens '
                f'above {max_new_tokens}. No labels were dropped.')

    def allowed_tokens(self, prompt_length):

        def allowed(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError(
                    'Local planner supports one snapshot at a time')
            node = self.root
            for token in input_ids[prompt_length:].tolist():
                if token not in node:
                    raise ValueError(
                        'Generation left the candidate token trie')
                node = node[token]
            if not node:
                raise ValueError('Generation continued after candidate EOS')
            return list(node)

        return allowed


@PLANNERS.register_module()
class LocalVLMSubtaskPlanner:
    """Current cameras + goal -> trained subtask text -> candidate ID.

    Uses the saved native architecture/processor and strict full checkpoint
    loading. No API, simulator predicates, candidate list in the prompt, or
    training frame/episode index enters the VLM. The vocabulary constraint
    acts only during decoding. These SFT labels do not train task completion,
    so this backend deliberately supports completion_mode='disabled' only.
    """

    def __init__(self,
                 checkpoint_path,
                 assets_path=None,
                 image_keys=('cam_high', 'cam_left_wrist', 'cam_right_wrist'),
                 processor_type='qwen3_5',
                 image_max_pixels=262144,
                 max_input_tokens=4096,
                 max_new_tokens=256,
                 device=None,
                 torch_dtype='bf16',
                 attn_implementation='sdpa',
                 constrained_decoding=True,
                 completion_mode='disabled'):
        if completion_mode != 'disabled':
            raise ValueError('Subtask-only VLM checkpoints cannot judge '
                             'completion; use completion_mode="disabled"')
        if type(max_new_tokens) is not int or max_new_tokens < 2:
            raise ValueError('max_new_tokens must be an integer of at least 2')
        if type(constrained_decoding) is not bool:
            raise ValueError('constrained_decoding must be a boolean')
        self.completion_mode = completion_mode
        self.image_keys = _validate_image_keys(image_keys)
        if not self.image_keys:
            raise ValueError('Local VLM planner requires camera image_keys')
        self.checkpoint_path = resolve_planner_checkpoint(checkpoint_path)
        self.assets_path = (
            Path(assets_path).expanduser().resolve() if assets_path else
            self.checkpoint_path.parent.parent / 'vlm_assets')
        if not (self.assets_path / 'config.json').is_file():
            raise FileNotFoundError(
                'Missing saved VLM architecture/processor '
                f'in {self.assets_path}; specify assets_path')
        self.device = torch.device(
            device or (f'cuda:{torch.cuda.current_device()}'
                       if torch.cuda.is_available() else 'cpu'))
        self.dtype = str_to_dtype(torch_dtype)
        if self.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError('Planner torch_dtype must be fp32, bf16 or fp16')
        self.max_new_tokens = max_new_tokens
        self.constrained_decoding = constrained_decoding
        self._candidate_key = None
        self._trie = None
        self.last_prediction = None
        from fluxvla.transforms.vlm_transforms import ProcessVLMSFTInputs
        self.processor = ProcessVLMSFTInputs(
            processor_path=str(self.assets_path),
            processor_type=processor_type,
            image_max_pixels=image_max_pixels,
            truncation_max_length=max_input_tokens,
            enable_thinking=False,
            processor_kwargs=dict(local_files_only=True))
        # Load the saved processor now; a worker must not claim ready until
        # both model weights and its tokenizer are usable.
        self.processor._get_processor()
        self.model = self._load_model(attn_implementation, torch_dtype)
        print(
            f'[FluxVLA][local-vlm] ready checkpoint={self.checkpoint_path} '
            f'device={self.device} dtype={self.dtype}',
            flush=True)

    def _load_model(self, attn_implementation, torch_dtype):
        from fluxvla.engines.utils.builder import build_vla_from_cfg

        # Architecture only: do NOT reload the original pretrained HF weights.
        with (self.assets_path / 'config.json').open() as source:
            architecture = json.load(source)
        model = build_vla_from_cfg(
            dict(
                type='VLMWrapper',
                vlm_backbone=dict(
                    type='CausalVLMBackbone',
                    model_config=architecture,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation)))
        state = load_file(str(self.checkpoint_path), device='cpu')
        model.load_state_dict(state, strict=True)
        del state
        return model.requires_grad_(False).eval().to(self.device)

    def reset(self):
        # Vocabulary/tokenizer caches are stateless; no images or KV cache
        # survive reset, including the existing per-request serving mode.
        self.last_prediction = None

    @torch.inference_mode()
    def choose(self, *, goal, observation, facts, candidates, history,
               feedback):
        from fluxvla.transforms.subtask_vlm_transforms import \
            build_subtask_vlm_messages

        self.last_prediction = None
        if facts:
            raise ValueError('Local subtask planner must not receive facts')
        cameras = _planner_camera_observation(self.image_keys, observation)
        messages = build_subtask_vlm_messages(
            goal, [Image.fromarray(cameras[key]) for key in self.image_keys],
            self.image_keys)
        # History/feedback are intentionally unused: the training prompt has
        # only the current views and whole-task goal, never an answer or IDs.
        tokenizer = self.processor._get_processor().tokenizer
        eos = self.processor.eos_token_id
        if type(eos) is not int:
            raise ValueError('Local planner requires one EOS token ID')
        key = tuple((c['choice'], c['instruction']) for c in candidates)
        if key != self._candidate_key:
            self._trie = _CandidateTrie(tokenizer, candidates, eos,
                                        self.max_new_tokens)
            self._candidate_key = key
        inputs = self.processor.encode_messages(messages)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_length = inputs['input_ids'].shape[-1]
        options = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos,
            pad_token_id=eos,
            use_cache=True)
        if self.constrained_decoding:
            options['prefix_allowed_tokens_fn'] = self._trie.allowed_tokens(
                prompt_length)
        started = time.monotonic()
        with torch.autocast(
                self.device.type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32):
            sequences = self.model.generate_text(**inputs, **options)
        ids = sequences[0, prompt_length:].tolist()
        # A truncated prefix is never a valid completed instruction, even if
        # it is textually equal to a shorter candidate. Require its EOS.
        if not ids or ids[-1] != eos:
            raise RuntimeError('Local VLM generation ended without EOS; '
                               'increase max_new_tokens')
        text = tokenizer.decode(
            ids[:-1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        choice = self._trie.texts.get(text)
        if choice is None or (self.constrained_decoding and
                              self._trie.choices.get(tuple(ids)) != choice):
            raise ValueError('Local VLM output is not a current-task training '
                             f'candidate: {text!r}; no fuzzy fallback applied')
        self.last_prediction = dict(
            text=text,
            choice=choice,
            generated_tokens=len(ids),
            inference_time_s=time.monotonic() - started,
            constrained_decoding=self.constrained_decoding)
        return dict(choice=choice)
