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
"""VLM instruction-tuning processors: build messages and run model-specific
processor (e.g. Qwen3VL apply_chat_template + preprocess_qwen_visual_*).
"""

from typing import Any, Dict, List, Optional

import torch

# Optional Qwen3-VL processor for SFT data
try:
    from transformers import Qwen3VLProcessor
    _HAS_QWEN3_VL = True
except ImportError:
    Qwen3VLProcessor = None
    _HAS_QWEN3_VL = False

# Optional Qwen3.5 processor (AutoProcessor for Qwen3.5-0.8B etc.)
try:
    from transformers import AutoProcessor
    _HAS_QWEN3_5_VL = True
except ImportError:
    AutoProcessor = None
    _HAS_QWEN3_5_VL = False


def build_messages_llava(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build chat messages from LLaVA-format item (conversations+image/video).

    Item should have:
        - "conversations": list of {"from": "human"|"gpt", "value": str}
        - optional "image" or "images": str or list of paths
        - optional "video" or "videos": str or list of paths

    User text may contain <image> and <video> placeholders; they are consumed
    in order from image_pool and video_pool.

    Returns:
        messages: list of {"role": "user"|"assistant", "content": [...]}
        where content is list of {"type": "text"|"image"|"video", ...}.
    """
    import re

    images = item.get('image') or item.get('images') or []
    if isinstance(images, str):
        images = [images]

    videos = item.get('video') or item.get('videos') or []
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [{'type': 'image', 'image': img} for img in images]
    video_pool = [{'type': 'video', 'video': vid} for vid in videos]

    messages = []
    for turn in item['conversations']:
        role = 'user' if turn.get('from') == 'human' else 'assistant'
        text = turn.get('value', '')

        if role == 'user':
            content = []
            parts = re.split(r'(<image>|<video>)', text)
            for seg in parts:
                if seg == '<image>':
                    if not image_pool:
                        raise ValueError(
                            'Number of <image> placeholders exceeds '
                            'the number of provided images')
                    content.append(image_pool.pop(0))
                elif seg == '<video>':
                    if not video_pool:
                        raise ValueError(
                            'Number of <video> placeholders exceeds '
                            'the number of provided videos')
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({'type': 'text', 'text': seg.strip()})
            messages.append({'role': role, 'content': content})
        else:
            messages.append({
                'role': role,
                'content': [{
                    'type': 'text',
                    'text': text
                }]
            })

    if image_pool:
        raise ValueError(f'{len(image_pool)} image(s) remain unused '
                         '(not consumed by placeholders)')
    if video_pool:
        raise ValueError(f'{len(video_pool)} video(s) remain unused '
                         '(not consumed by placeholders)')

    return messages


def get_qwen3_vl_processor(
    model_path: str,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    trust_remote_code: bool = True,
) -> Any:
    """Load Qwen3VLProcessor from path. Optionally set image/video size."""
    if not _HAS_QWEN3_VL:
        raise ImportError(
            'Qwen3VLProcessor not available; install transformers with '
            'Qwen3-VL support.')
    processor = Qwen3VLProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code)
    if image_max_pixels is not None:
        processor.image_processor.size = {
            'longest_edge': image_max_pixels,
            'shortest_edge': 256 * 32 * 32,
        }
    if video_max_pixels is not None:
        processor.video_processor.size = {
            'longest_edge': video_max_pixels,
            'shortest_edge': 256 * 32 * 32 * 2,
        }
    return processor


def get_qwen3_5_processor(
    model_path: str,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    trust_remote_code: bool = True,
) -> Any:
    """Load processor for Qwen3.5 (e.g. Qwen3.5-0.8B) via AutoProcessor."""
    if not _HAS_QWEN3_5_VL:
        raise ImportError(
            'AutoProcessor not available; install transformers with '
            'Qwen3.5 support.')
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code)
    if image_max_pixels is not None and hasattr(processor, 'image_processor'):
        if hasattr(processor.image_processor, 'size'):
            processor.image_processor.size = {
                'longest_edge': image_max_pixels,
                'shortest_edge': 256 * 32 * 32,
            }
    if video_max_pixels is not None and hasattr(processor, 'video_processor'):
        if hasattr(processor.video_processor, 'size'):
            processor.video_processor.size = {
                'longest_edge': video_max_pixels,
                'shortest_edge': 256 * 32 * 32 * 2,
            }
    return processor


def process_item_qwen3_5(
    item: Dict[str, Any],
    processor: Any,
    data_root: Optional[str] = None,
    truncation_max_length: int = 0,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Process one LLaVA-format item with Qwen3.5 processor for SFT.
    Same flow as process_item_qwen3_vl: build_messages_llava +
    apply_chat_template + preprocess_*_training_label when available.
    """
    if data_root:
        item = _resolve_paths_in_item(item, data_root)
    if image_max_pixels is not None and hasattr(processor, 'image_processor'):
        if hasattr(processor.image_processor, 'size'):
            processor.image_processor.size = {
                'longest_edge': image_max_pixels,
                'shortest_edge': 256 * 32 * 32,
            }
    if video_max_pixels is not None and hasattr(processor, 'video_processor'):
        if hasattr(processor.video_processor, 'size'):
            processor.video_processor.size = {
                'longest_edge': video_max_pixels,
                'shortest_edge': 256 * 32 * 32 * 2,
            }
    messages = build_messages_llava(item)
    if truncation_max_length > 0:
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
            truncation=True,
            max_length=truncation_max_length,
        )
    else:
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
            truncation=False,
        )
    data_dict = None
    for method_name in ('preprocess_qwen_visual_training_label',
                        'preprocess_visual_training_label'):
        if hasattr(processor, method_name):
            data_dict = getattr(processor, method_name)(processed)
            break
    if data_dict is None:
        data_dict = dict(processed)
        if 'input_ids' in data_dict:
            data_dict['labels'] = _mask_labels_for_sft(
                data_dict['input_ids'],
                processor.tokenizer,
            )
    return data_dict


def process_item_qwen3_vl(
    item: Dict[str, Any],
    processor: Any,
    data_root: Optional[str] = None,
    truncation_max_length: int = 0,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Process one LLaVA-format item with Qwen3VLProcessor for SFT.

    Matches LLaVA training dataloader behavior:
    - Builds messages with build_messages_llava(item).
    - Optionally sets processor image/video size before apply_chat_template
      (same keys as LLaVA: longest_edge, shortest_edge).
    - apply_chat_template(..., tokenize=True, add_generation_prompt=False,
      return_dict=True, return_tensors='pt', truncation=..., max_length=...).
    - Then processor.preprocess_qwen_visual_training_label(processed) when
      available; otherwise fallback _mask_labels_for_sft.

    Returns:
        Dict with at least: input_ids, attention_mask, labels; optionally
        pixel_values, image_grid_thw (and pixel_values_videos, video_grid_thw
        if present in processor output).
    """
    # Resolve paths if data_root is set
    if data_root:
        item = _resolve_paths_in_item(item, data_root)

    # Match LLaVA: set image/video size before each apply_chat_template
    if image_max_pixels is not None:
        processor.image_processor.size = {
            'longest_edge': image_max_pixels,
            'shortest_edge': 256 * 32 * 32,
        }
    if video_max_pixels is not None:
        processor.video_processor.size = {
            'longest_edge': video_max_pixels,
            'shortest_edge': 256 * 32 * 32 * 2,
        }

    messages = build_messages_llava(item)

    if truncation_max_length > 0:
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
            truncation=True,
            max_length=truncation_max_length,
        )
    else:
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
            truncation=False,
        )

    # Prefer processor's training-label method (masks non-assistant with -100)
    data_dict = None
    for method_name in ('preprocess_qwen_visual_training_label',
                        'preprocess_visual_training_label'):
        if hasattr(processor, method_name):
            data_dict = getattr(processor, method_name)(processed)
            break
    if data_dict is None:
        data_dict = dict(processed)
        if 'input_ids' in data_dict:
            data_dict['labels'] = _mask_labels_for_sft(
                data_dict['input_ids'],
                processor.tokenizer,
            )
    return data_dict


def _get_assistant_start_token_ids(tokenizer: Any) -> List[int]:
    """Token ids that precede the assistant reply.
    Uses apply_chat_template with [user, assistant+placeholder] then finds
    the placeholder so we get the same prefix as in the full sequence.
    """
    try:
        # Build tokenized seq with "...assistant\n" then one placeholder char
        placeholder = '\u200b'  # zero-width space, usually single token
        messages = [
            {
                'role': 'user',
                'content': 'x'
            },
            {
                'role': 'assistant',
                'content': placeholder
            },
        ]
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        if isinstance(out, list) and len(out) > 0:
            ph_ids = tokenizer.encode(placeholder, add_special_tokens=False)
            if not ph_ids:
                raise ValueError('placeholder not tokenized')
            # First occurrence of placeholder in out; prefix is before it
            for i in range(len(out) - len(ph_ids) + 1):
                if out[i:i + len(ph_ids)] == ph_ids:
                    return list(out[:i])
            # Placeholder not found (e.g. merged); use tokens before last
            return list(
                out[:-len(ph_ids)]) if len(out) > len(ph_ids) else list(out)
    except Exception:
        pass
    try:
        return tokenizer.encode(
            '<|im_start|>assistant\n', add_special_tokens=False)
    except Exception:
        pass
    try:
        return tokenizer.encode('assistant\n', add_special_tokens=False)
    except Exception:
        return []


def _mask_labels_for_sft(
    input_ids: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Set labels to -100 except for the assistant reply (SFT).
    Uses the same chat template as the tokenizer to find where the assistant
    reply starts, so masking matches the official template exactly.
    """
    labels = input_ids.clone()
    if labels.dim() == 2:
        labels = labels[0]
    labels = labels.view(-1)
    input_flat = labels.tolist()
    assistant_start = _get_assistant_start_token_ids(tokenizer)
    start_pos = 0
    if len(assistant_start) > 0:
        for i in range(len(input_flat) - len(assistant_start) + 1):
            if input_flat[i:i + len(assistant_start)] == assistant_start:
                start_pos = i + len(assistant_start)
        labels[:start_pos] = -100
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    return labels.view_as(input_ids)


def _resolve_paths_in_item(
    item: Dict[str, Any],
    data_root: str,
) -> Dict[str, Any]:
    """Resolve image/video paths in item relative to data_root."""
    import os
    out = dict(item)

    for key in ('image', 'images', 'video', 'videos'):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, str):
            if not os.path.isabs(val):
                val = os.path.join(data_root, val)
            out[key] = val
        elif isinstance(val, list):
            out[key] = [
                os.path.join(data_root, p) if not os.path.isabs(p) else p
                for p in val
            ]
    return out


def get_processor_for_vlm(
    processor_type: str,
    model_path: str,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Get a processor instance for the given VLM type.

    Supported processor_type: 'qwen3_vl', 'qwen3_5'.
    """
    if processor_type == 'qwen3_vl':
        return get_qwen3_vl_processor(
            model_path,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
            **kwargs,
        )
    if processor_type == 'qwen3_5':
        return get_qwen3_5_processor(
            model_path,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
            **kwargs,
        )
    raise ValueError(f'Unknown VLM processor_type="{processor_type}". '
                     'Supported: qwen3_vl, qwen3_5.')


def process_item_for_vlm(
    processor_type: str,
    processor: Any,
    item: Dict[str, Any],
    data_root: Optional[str] = None,
    truncation_max_length: int = 0,
    image_max_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Dispatch to the right per-item processing for the VLM type."""
    if processor_type == 'qwen3_vl':
        return process_item_qwen3_vl(
            item,
            processor,
            data_root=data_root,
            truncation_max_length=truncation_max_length,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
        )
    if processor_type == 'qwen3_5':
        return process_item_qwen3_5(
            item,
            processor,
            data_root=data_root,
            truncation_max_length=truncation_max_length,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
        )
    raise ValueError(f'Unknown VLM processor_type="{processor_type}".')
