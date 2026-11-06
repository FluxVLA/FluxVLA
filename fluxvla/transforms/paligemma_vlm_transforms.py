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
"""PaliGemma conversation and image-prefix transforms for VLM SFT."""

import json
import os
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from PIL import Image

from fluxvla.engines import TRANSFORMS
from .transform_images import ResizeImagesWithPad


# Keep the old registry key so saved training configs remain loadable.
@TRANSFORMS.register_module(
    name=['ProcessPaliGemmaVLMSFTInputs', 'ProcessPi05VLMSFTInputs'])
class ProcessPaliGemmaVLMSFTInputs:
    """Encode canonical conversations using a PaliGemma SentencePiece model.

    Format: BOS, system/user text + newline, assistant text + EOS. Only
    assistant tokens and their EOS are supervised. Images in the first user
    turn form a visual prefix; later images and videos are rejected rather
    than silently moved ahead of earlier answers. Multi-turn text is causal.
    Images are center-padded to a square and normalized to [-1, 1]. This
    contract is independent of robot states/actions; it is not the native
    Hugging Face PaliGemma processor's bidirectional image/question prefix.
    """

    def __init__(self, tokenizer_path, image_size=224, max_length=1024):
        if type(image_size) is not int or image_size < 1:
            raise ValueError('image_size must be a positive integer.')
        if type(max_length) is not int or max_length < 2:
            raise ValueError('max_length must be an integer >= 2.')
        path = Path(tokenizer_path).expanduser()
        self.tokenizer_path = str(path /
                                  'tokenizer.model' if path.is_dir() else path)
        self.image_size = image_size
        self.max_length = max_length
        self.resize = ResizeImagesWithPad(
            height=image_size, width=image_size, backend='pil')
        self._tokenizer = None
        self._pid = None

    def _get_tokenizer(self):
        if self._tokenizer is None or self._pid != os.getpid():
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=self.tokenizer_path)
            self._pid = os.getpid()
            if min(self._tokenizer.bos_id(), self._tokenizer.eos_id()) < 0:
                raise ValueError('VLM SFT requires BOS/EOS tokenizer IDs.')
        return self._tokenizer

    def __getstate__(self):
        return dict(self.__dict__, _tokenizer=None, _pid=None)

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'tokenizer.model').write_bytes(
            self._get_tokenizer().serialized_model_proto())
        (path / 'paligemma_vlm_preprocessing.json').write_text(
            json.dumps(
                dict(
                    tokenizer='PaliGemma SentencePiece',
                    image_size=self.image_size,
                    image_resize='pil_bilinear_center_pad',
                    image_range=[-1, 1],
                    max_length=self.max_length,
                    text_format='BOS + user/newline + assistant/EOS',
                    attention='bidirectional_images_causal_text'),
                indent=2) + '\n',
            encoding='utf-8')

    def __call__(self, data):
        messages = data.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Configure BuildVLMMessages before VLM SFT.')
        tokenizer = self._get_tokenizer()
        tokens, labels = [tokenizer.bos_id()], [-100]
        image_paths = []
        user_turns = 0
        previous = None
        for message in messages:
            role = message['role']
            if (role not in ('system', 'user', 'assistant')
                    or (role == 'system' and user_turns)
                    or (role == 'assistant' and previous != 'user')):
                raise ValueError('Invalid VLM SFT conversation roles.')
            user_turns += role == 'user'
            text = []
            for part in message['content']:
                if part['type'] == 'text':
                    text.append(part['text'])
                elif part['type'] == 'image':
                    if role != 'user' or user_turns != 1:
                        raise ValueError('VLM SFT images must occur in the '
                                         'first user turn.')
                    image_paths.append(part['image'])
                else:
                    raise ValueError('VLM SFT supports text and images only.')
            text = ''.join(text)
            if role == 'assistant' and not text.strip():
                raise ValueError('Assistant replies must contain text.')
            encoded = tokenizer.encode(text, out_type=int)
            encoded += ([tokenizer.eos_id()] if role == 'assistant' else
                        tokenizer.encode('\n', out_type=int))
            tokens.extend(encoded)
            labels.extend(encoded if role == 'assistant' else [-100] *
                          len(encoded))
            previous = role
        tokens, labels = tokens[:self.max_length], labels[:self.max_length]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError('No assistant targets remain after truncation.')
        return dict(
            data,
            input_ids=torch.tensor(tokens, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=torch.ones(len(tokens), dtype=torch.bool),
            images=self._load_images(image_paths))

    def encode_prompt(self, question, image_paths=()):
        """Encode inference input without accepting or returning answers."""
        tokenizer = self._get_tokenizer()
        tokens = ([tokenizer.bos_id()] + tokenizer.encode(question) +
                  tokenizer.encode('\n'))
        if len(tokens) > self.max_length:
            raise ValueError('VLM generation prompt exceeds max_length.')
        inputs = dict(
            input_ids=torch.tensor(tokens, dtype=torch.long),
            attention_mask=torch.ones(len(tokens), dtype=torch.bool))
        if image_paths:
            inputs['images'] = self._load_images(image_paths)
        return inputs

    def decode_answer(self, token_ids):
        """Decode new tokens only, dropping EOS and following padding."""
        tokenizer = self._get_tokenizer()
        ids = list(token_ids)
        if tokenizer.eos_id() in ids:
            ids = ids[:ids.index(tokenizer.eos_id())]
        ids = [
            i for i in ids if i not in (tokenizer.pad_id(), tokenizer.bos_id())
        ]
        return tokenizer.decode(ids).strip()

    def _load_images(self, image_paths):
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(
                    np.array(image.convert('RGB')).transpose(2, 0, 1))
        if images:
            resized = self.resize(dict(images=images))['images']
            pixels = torch.from_numpy(resized.copy()).float() / 127.5 - 1
        else:
            pixels = torch.empty(0, self.image_size, self.image_size)
        return pixels
