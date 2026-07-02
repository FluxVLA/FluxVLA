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
"""Run standalone FluxVLA-native Cosmos3 inference.

This script is a developer/demo harness for official Cosmos3 checkpoints. It
can reproduce selected Cosmos3 README examples or run a JSON/JSONL list of
custom samples across text/image/video/action/reasoning modes.

LIBERO policy evaluation still goes through ``scripts/eval.sh`` and
``LiberoEvalRunner``; this file is for checkpoint smoke tests and
offline demos.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints/Cosmos3-Nano'
DEFAULT_RUN_DIR = PROJECT_ROOT / 'work_dirs/cosmos3_readme_repro'
README_EXAMPLES = ('i2v', 't2v', 'action_fd', 'action_id', 'reasoning')

HELP_EPILOG = """Examples:
  # Reproduce selected official Cosmos3 README samples.
  python scripts/cosmos3_fluxvla_infer.py --readme-repro --examples i2v,t2v

  # Run a custom JSON/JSONL sample batch.
  python scripts/cosmos3_fluxvla_infer.py -i samples.jsonl \\
      -o work_dirs/cosmos3_infer --checkpoint checkpoints/Cosmos3-Nano

  # Smoke-test checkpoint/model loading only.
  python scripts/cosmos3_fluxvla_infer.py --readme-repro --examples i2v --load-only  # noqa: E501
"""

IMAGE_DEFAULTS = dict(
    resolution='720',
    aspect_ratio='1,1',
    fps=24,
    num_frames=1,
    num_steps=50,
    guidance=4.0,
    shift=3.0,
    negative_prompt='',
)
VIDEO_DEFAULTS = dict(
    resolution='720',
    aspect_ratio='16,9',
    fps=24,
    num_frames=189,
    num_steps=35,
    guidance=6.0,
    shift=10.0,
    negative_prompt='',
)
ACTION_DEFAULTS = dict(
    resolution='480',
    aspect_ratio='16,9',
    fps=24,
    num_steps=30,
    guidance=1.0,
    shift=10.0,
    negative_prompt='',
    image_size=480,
    action_chunk_size=16,
)

MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    'text2image': dict(IMAGE_DEFAULTS),
    'text2video': dict(VIDEO_DEFAULTS),
    'image2video': dict(VIDEO_DEFAULTS, condition_frame_indexes_vision=[0]),
    'video2video': dict(VIDEO_DEFAULTS, condition_frame_indexes_vision=[0, 1]),
    'forward_dynamics': dict(ACTION_DEFAULTS),
    'inverse_dynamics': dict(ACTION_DEFAULTS),
    'policy': dict(ACTION_DEFAULTS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run standalone FluxVLA-native Cosmos3 demos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG)
    add = parser.add_argument
    add('-i', '--input', help='JSON/JSONL sample file.')
    add('-o', '--output-dir', help='Inference output dir.')
    add('--checkpoint',
        default=str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)),
        help='Cosmos3 checkpoint root.')
    add('--model-size',
        choices=('auto', 'nano', 'super'),
        default='auto',
        help='Model config size.')
    add('--vae-path', default=None, help='Wan2.2 VAE checkpoint.')
    add('--device', default='cuda:0', help='Torch device.')
    add('--dtype',
        choices=('bf16', 'fp16', 'fp32'),
        default='bf16',
        help='Model compute dtype.')
    add('--num-steps', type=int, default=None, help='Diffusion steps.')
    add('--guidance', type=float, default=None, help='CFG scale.')
    add('--shift', type=float, default=None, help='Scheduler shift.')
    add('--seed', type=int, default=None, help='Sample seed.')
    add('--resolution', default=None, help='Visual resolution, e.g. 480/720.')
    add('--aspect-ratio', default=None, help='Aspect ratio, e.g. 16,9.')
    add('--fps', type=int, default=None, help='Video/action fps.')
    add('--num-frames', type=int, default=None, help='Raw video frame count.')
    add('--negative-prompt', default=None, help='Visual negative prompt.')
    add('--embodiment-id',
        type=int,
        default=None,
        help='Action embodiment id.')
    add('--raw-action-dim',
        type=int,
        default=None,
        help='Unpadded action dimension.')
    add('--action-chunk-size',
        type=int,
        default=None,
        help='Action horizon/chunk size.')
    add('--max-new-tokens',
        type=int,
        default=None,
        help='Reasoning generation cap.')
    add('--max-samples', type=int, default=None, help='Run first N samples.')
    add('--use-karras-sigmas',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override scheduler sigma policy.')
    add('--load-only',
        action='store_true',
        help='Load checkpoints and exit before sampling.')
    add('--readme-repro',
        action='store_true',
        help='Run Cosmos3-Nano README examples.')
    add('--run-dir',
        default=str(DEFAULT_RUN_DIR.relative_to(PROJECT_ROOT)),
        help='README reproduction working directory.')
    add('--examples',
        default='all',
        help='README subset: i2v,t2v,action_fd,action_id,reasoning.')
    add('--reasoning-max-new-tokens',
        type=int,
        default=128,
        help='README reasoning generation cap.')
    add('--prepare-only',
        action='store_true',
        help='Write README input JSON files and exit.')
    return parser.parse_args()


def resolve_under_root(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def dtype_from_name(name: str) -> torch.dtype:
    return {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }[name]


def mode_defaults(mode: str) -> dict[str, Any]:
    if mode not in MODE_DEFAULTS:
        raise ValueError(f'Unsupported Cosmos3 mode: {mode!r}.')
    return dict(MODE_DEFAULTS[mode])


def sample_value(sample: dict[str, Any],
                 defaults: dict[str, Any],
                 key: str,
                 override: Any = None) -> Any:
    if override is not None:
        return override
    if key in sample and sample[key] is not None:
        return sample[key]
    return defaults.get(key)


def _read_prompt_path(path: str | Path, base_dir: Path) -> str:
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = base_dir / prompt_path
    if prompt_path.suffix.lower() == '.json':
        return json.dumps(json.loads(prompt_path.read_text()))
    return prompt_path.read_text().strip()


def _expand_prompt_paths(sample: dict[str, Any], base_dir: Path) -> None:
    if sample.get('prompt_path') is not None and sample.get('prompt') is None:
        sample['prompt'] = _read_prompt_path(sample['prompt_path'], base_dir)
    for key in ('negative_prompt_path', 'negative_prompt_file'):
        if sample.get(
                key) is not None and sample.get('negative_prompt') is None:
            sample['negative_prompt'] = _read_prompt_path(
                sample[key], base_dir)


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == '.jsonl':
        samples = [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text())
        samples = data if isinstance(data, list) else [data]
    for index, sample in enumerate(samples):
        if 'model_mode' not in sample:
            raise ValueError(f'Sample {index} has no model_mode.')
        _expand_prompt_paths(sample, path.parent)
    return samples


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def _selected_examples(text: str) -> list[str]:
    if text == 'all':
        return list(README_EXAMPLES)
    examples = [item.strip() for item in text.split(',') if item.strip()]
    unknown = sorted(set(examples) - set(README_EXAMPLES))
    if unknown:
        raise ValueError(
            f'Unknown README examples: {unknown}. Valid: {README_EXAMPLES}')
    return examples


def _readme_visual_sample(assets: Path,
                          name: str,
                          mode: str,
                          prompt_file: str,
                          seed: int,
                          vision_file: str | None = None) -> dict[str, Any]:
    sample = dict(
        name=name,
        model_mode=mode,
        prompt_path=str(assets / prompt_file),
        negative_prompt_path=str(assets / 'negative_prompt.json'),
        width=1280,
        height=720,
        num_frames=189,
        fps=24,
        num_steps=35,
        guidance=6.0,
        shift=10.0,
        seed=seed,
    )
    if vision_file is not None:
        sample['vision_path'] = str(assets / vision_file)
    return sample


def _readme_samples(checkpoint: Path, reasoning_max_new_tokens: int):
    assets = checkpoint / 'assets'
    action_spec_path = (
        assets / 'example_action_fd_agibotworld_action_chunks.json')
    action_spec = json.loads(action_spec_path.read_text())
    reasoning_prompt = json.loads(
        (assets / 'example_reasoning_prompt.json').read_text())
    action_id_common = dict(
        model_mode='inverse_dynamics',
        prompt='You are an autonomous vehicle planning system.',
        embodiment_id=1,
        raw_action_dim=9,
        action_chunk_size=60,
        width=832,
        height=480,
        fps=10,
        num_steps=30,
        shift=10.0,
        seed=0,
    )
    return {
        'i2v':
        _readme_visual_sample(
            assets,
            'readme_i2v_seed1111',
            'image2video',
            'example_i2v_prompt.json',
            1111,
            'example_i2v_input.jpg',
        ),
        't2v':
        _readme_visual_sample(
            assets,
            'readme_t2v_seed123',
            'text2video',
            'example_t2v_prompt.json',
            123,
        ),
        'action_fd': {
            'name':
            'readme_action_fd_agibotworld_4chunk_seed0',
            'model_mode':
            'forward_dynamics',
            'prompt':
            action_spec.get('prompt', 'Pickup items in the supermarket'),
            'vision_path':
            str(assets / 'example_action_fd_agibotworld_first_frame.png'),
            'action_chunks_path':
            str(action_spec_path),
            'embodiment_id':
            15,
            'raw_action_dim':
            29,
            'action_chunk_size':
            int(action_spec.get('action_chunk_size', 16)),
            'width':
            640,
            'height':
            720,
            'fps':
            int(action_spec.get('fps', 10)),
            'num_steps':
            30,
            'guidance':
            1.0,
            'shift':
            10.0,
            'seed':
            0,
        },
        'action_id': [
            dict(
                action_id_common,
                name=f'readme_action_id_av_{idx}_seed0',
                vision_path=str(assets /
                                f'example_action_id_av_{idx}_input.mp4'),
            ) for idx in (0, 1)
        ],
        'reasoning': {
            'name': 'readme_reasoning_seed0',
            'model_mode': 'reasoning',
            'prompt': reasoning_prompt['prompt'],
            'max_new_tokens': reasoning_max_new_tokens,
            'vision_path': str(assets / 'example_reasoning_input.png'),
            'seed': 0,
        },
    }


def _prepare_readme_inputs(args: argparse.Namespace):
    checkpoint = resolve_under_root(args.checkpoint)
    run_dir = resolve_under_root(args.run_dir)
    examples = _selected_examples(args.examples)
    samples_by_example = _readme_samples(checkpoint,
                                         args.reasoning_max_new_tokens)
    input_dir = run_dir / 'inputs'
    selected_samples: list[dict[str, Any]] = []
    for example in examples:
        payload = samples_by_example[example]
        _write_json(input_dir / f'readme_{example}.json', payload)
        items = payload if isinstance(payload, list) else [payload]
        for sample in items:
            _expand_prompt_paths(sample, input_dir)
        selected_samples.extend(items)
    return checkpoint, run_dir, selected_samples


def parse_prompt_json(prompt: str) -> dict[str, Any]:
    try:
        value = json.loads(prompt)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def parse_duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r'(?:(\d+):)?(\d+(?:\.\d+)?)s?', text)
    if match is None:
        return None
    return float(match.group(1) or 0) * 60.0 + float(match.group(2))


def _vision_args(width: int,
                 height: int,
                 fps: int,
                 num_frames: int,
                 *,
                 resolution=None,
                 aspect_ratio=None) -> dict[str, Any]:
    return dict(
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        width=int(width),
        height=int(height),
        fps=int(fps),
        num_frames=int(num_frames),
    )


def resolve_vision_args(sample: dict[str, Any],
                        *,
                        resolution: str | None = None,
                        aspect_ratio: str | None = None,
                        fps: int | None = None,
                        num_frames: int | None = None) -> dict[str, Any]:
    from fluxvla.models.third_party_models.cosmos3.data.vfm.utils import (
        IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO)

    mode = sample['model_mode']
    defaults = mode_defaults(mode)
    prompt_meta = parse_prompt_json(str(sample.get('prompt', '')))
    resolved_fps = int(sample_value(sample, defaults, 'fps', fps))
    resolved_num_frames = sample_value(sample, defaults, 'num_frames',
                                       num_frames)
    if resolved_num_frames is None:
        duration = parse_duration_seconds(prompt_meta.get('duration'))
        resolved_num_frames = (
            int(round(duration * resolved_fps)) if duration is not None else 1)
    resolved_num_frames = int(resolved_num_frames)

    if sample.get('width') is not None and sample.get('height') is not None:
        return _vision_args(
            sample['width'],
            sample['height'],
            resolved_fps,
            resolved_num_frames,
            resolution=sample.get('resolution'),
            aspect_ratio=sample.get('aspect_ratio'),
        )

    resolved_aspect = str(
        aspect_ratio or sample.get('aspect_ratio')
        or prompt_meta.get('aspect_ratio') or defaults.get('aspect_ratio')
        or '16,9')
    resolved_resolution = (
        resolution or sample.get('resolution') or sample.get('image_size'))
    if resolved_resolution is None:
        prompt_res = prompt_meta.get('resolution')
        if isinstance(prompt_res, dict) and {'W', 'H'} <= set(prompt_res):
            return _vision_args(
                prompt_res['W'],
                prompt_res['H'],
                resolved_fps,
                resolved_num_frames,
                resolution=None,
                aspect_ratio=resolved_aspect,
            )
        resolved_resolution = defaults.get('resolution', '720')
    resolved_resolution = str(resolved_resolution)

    table = (
        IMAGE_RES_SIZE_INFO
        if resolved_num_frames == 1 else VIDEO_RES_SIZE_INFO)
    if resolved_resolution not in table:
        raise ValueError(f'Unsupported resolution {resolved_resolution!r}.')
    if resolved_aspect not in table[resolved_resolution]:
        raise ValueError(f'Unsupported aspect ratio {resolved_aspect!r} for '
                         f'resolution {resolved_resolution}.')
    width, height = table[resolved_resolution][resolved_aspect]
    return _vision_args(
        width,
        height,
        resolved_fps,
        resolved_num_frames,
        resolution=resolved_resolution,
        aspect_ratio=resolved_aspect,
    )


def latent_num_frames(num_frames: int, temporal_compression: int) -> int:
    if num_frames == 1:
        return 1
    if (num_frames - 1) % temporal_compression != 0:
        raise ValueError('Wan2.2 VAE expects output frames to be '
                         f'1+{temporal_compression}n, got {num_frames}.')
    return 1 + (num_frames - 1) // temporal_compression


def download_if_needed(path_or_url: str | Path,
                       cache_dir: Path,
                       *,
                       name: str | None = None) -> Path:
    text = str(path_or_url)
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {'http', 'https'}:
        return Path(text).expanduser()

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
    out = cache_dir / f'{name or "asset"}_{digest}{suffix}'
    if not out.exists():
        urllib.request.urlretrieve(text, out)
    return out


def resize_center_crop(frames_tchw: torch.Tensor, height: int,
                       width: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    orig_h, orig_w = int(frames_tchw.shape[-2]), int(frames_tchw.shape[-1])
    scale = max(width / orig_w, height / orig_h)
    resize_h = max(height, int(round(orig_h * scale)))
    resize_w = max(width, int(round(orig_w * scale)))
    frames_tchw = TF.resize(frames_tchw, [resize_h, resize_w])
    return TF.center_crop(frames_tchw, [height, width])


def _repeat_last_to_length(tensor: torch.Tensor, length: int,
                           dim: int) -> torch.Tensor:
    if tensor.shape[dim] == 0 or tensor.shape[dim] >= length:
        return tensor.narrow(dim, 0, length)
    repeat = [1] * tensor.dim()
    repeat[dim] = length - tensor.shape[dim]
    pad = tensor.select(dim, tensor.shape[dim] - 1).unsqueeze(dim)
    return torch.cat([tensor, pad.repeat(*repeat)], dim=dim)


def load_media_frames(path_or_url: str | Path,
                      *,
                      cache_dir: Path,
                      height: int,
                      width: int,
                      max_frames: int,
                      keep: str = 'first') -> torch.Tensor:
    path = download_if_needed(path_or_url, cache_dir, name='vision')
    if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        with path.open('rb') as handle:
            image = Image.open(handle).convert('RGB')
        frames_tchw = torch.from_numpy(np.array(image)).permute(
            2, 0, 1).unsqueeze(0).float()
    else:
        import torchvision.io

        frames, _, _ = torchvision.io.read_video(str(path), pts_unit='sec')
        if keep == 'last':
            frames = frames[-max_frames:]
        else:
            frames = frames[:max_frames]
        frames_tchw = frames.permute(0, 3, 1, 2).float()
    frames_tchw = resize_center_crop(frames_tchw, height, width)
    frames_tchw = _repeat_last_to_length(frames_tchw, max_frames, dim=0)
    return frames_tchw.permute(1, 0, 2, 3) / 127.5 - 1.0


def load_actions(path_or_url: str | Path, *, cache_dir: Path,
                 action_chunk_size: int) -> torch.Tensor:
    path = download_if_needed(path_or_url, cache_dir, name='action')
    tensor = torch.tensor(json.loads(path.read_text()), dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f'Expected action JSON [T,D], got {tensor.shape}.')
    return _repeat_last_to_length(tensor, action_chunk_size, dim=0)


def chat_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{'role': 'user', 'content': prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return torch.tensor([ids], dtype=torch.long)


def _decoded_frames_uint8(decoded: torch.Tensor):
    sample = ((decoded[0].detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
    return (sample.permute(1, 2, 3, 0).numpy() * 255.0).round().clip(
        0, 255).astype('uint8')


def _write_mp4(frames,
               output_path: Path,
               fps: float,
               quality: int = 10) -> Path:
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        quality=quality,
        macro_block_size=1,
        ffmpeg_params=['-s', f'{frames.shape[2]}x{frames.shape[1]}'],
        output_params=['-f', 'mp4'],
    )
    return output_path


def save_media(decoded: torch.Tensor,
               output_base: Path,
               *,
               fps: float,
               image_quality: int = 95,
               video_quality: int = 10) -> Path:
    if decoded.dim() != 5:
        raise ValueError(f'Expected decoded [B,C,T,H,W], got {decoded.shape}.')
    frames = _decoded_frames_uint8(decoded)
    output_base = output_base.with_suffix('')
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if frames.shape[0] == 1:
        output_path = output_base.with_suffix('.jpg')
        Image.fromarray(
            frames[0], mode='RGB').save(
                output_path, format='JPEG', quality=image_quality)
        return output_path

    return _write_mp4(
        frames,
        output_base.with_suffix('.mp4'),
        fps=fps,
        quality=video_quality,
    )


def save_action_json(actions: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(actions.detach().float().cpu().tolist(), indent=2) + '\n')
    return output_path


def default_config_path(model_size: str) -> Path:
    relative = {
        'nano': 'configs/cosmos3/cosmos3nano_libero_10_full_finetune.py',
        'super': 'configs/cosmos3/cosmos3super_libero_10_full_finetune.py',
    }[model_size]
    return PROJECT_ROOT / relative


def resolve_model_size(checkpoint: str, requested: str) -> str:
    if requested != 'auto':
        return requested
    return 'super' if 'super' in Path(checkpoint).name.lower() else 'nano'


def checkpoint_root_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.parent if path.name == 'transformer' else path


def default_vae_path(checkpoint: str | Path) -> str:
    checkpoint_root = checkpoint_root_path(checkpoint)
    return str(checkpoint_root.parent / 'Wan2.2-TI2V-5B' / 'Wan2.2_VAE.pth')


def to_plain(value):
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain(item) for item in value)
    return value


def _sequence_plan(**kwargs):
    from fluxvla.models.third_party_models.cosmos3.data.vfm import \
        sequence_packing

    return sequence_packing.SequencePlan(**kwargs)


def build_and_load_fluxvla_model(*,
                                 checkpoint: str,
                                 model_size: str,
                                 vae_path: str | None,
                                 dtype: torch.dtype,
                                 device: torch.device,
                                 action_horizon: int | None = None,
                                 include_visual: bool = False):
    from mmengine.config import Config

    from fluxvla.models.vlas.cosmos3_flowmatching import Cosmos3FlowMatching

    checkpoint_root = checkpoint_root_path(checkpoint)
    transformer_path = (
        checkpoint_root / 'transformer' if
        (checkpoint_root / 'transformer').is_dir() else checkpoint_root)
    vision_encoder_path = checkpoint_root / 'vision_encoder'
    cfg = Config.fromfile(str(default_config_path(model_size)))
    model_kwargs = to_plain(cfg.model)
    model_kwargs.pop('type', None)
    model_kwargs['pretrained_name_or_path'] = str(transformer_path)
    vlm_backbone_cfg = model_kwargs.get('vlm_backbone')
    if isinstance(vlm_backbone_cfg, dict):
        vlm_backbone_cfg['include_visual'] = bool(include_visual)
        if vision_encoder_path.is_dir():
            vlm_backbone_cfg['vision_encoder_path'] = str(vision_encoder_path)
    elif include_visual:
        raise ValueError('Reasoning image inputs require a configurable '
                         'vlm_backbone with include_visual=True.')
    tokenizer_cfg = model_kwargs.setdefault(
        'vision_vae',
        dict(type='Cosmos3Wan22VAE'),
    )
    tokenizer_cfg['pretrained_name_or_path'] = vae_path or default_vae_path(
        checkpoint_root)
    model_kwargs.update(
        torch_dtype=dtype,
        freeze_vlm_backbone=True,
        freeze_non_moe_vlm_backbone=False,
        enable_vision_loss=False,
    )
    if action_horizon is not None:
        model_kwargs['action_horizon'] = int(action_horizon)
    model = Cosmos3FlowMatching(**model_kwargs)
    model.to(device)
    model.from_pretrained()
    if include_visual:
        from transformers import GenerationConfig

        model.vlm_backbone.generation_config = (
            GenerationConfig.from_pretrained(str(checkpoint_root)))
    model.eval()
    return model


def _negative_prompt(sample: dict[str, Any], args: argparse.Namespace) -> str:
    value = sample_value(
        sample,
        mode_defaults(sample['model_mode']),
        'negative_prompt',
        args.negative_prompt,
    )
    return '' if value is None else str(value)


def _sampling_value(sample: dict[str, Any], args: argparse.Namespace,
                    key: str) -> Any:
    return sample_value(
        sample,
        mode_defaults(sample['model_mode']),
        key,
        getattr(args, key.replace('-', '_'), None),
    )


def _apply_sampling_config(model, sample: dict[str, Any],
                           args: argparse.Namespace) -> None:
    cfg = model.rectified_flow_inference_config
    base_cfg = getattr(model, '_script_base_rectified_flow_inference_config',
                       None)
    if base_cfg is None:
        base_cfg = dict(cfg)
        model._script_base_rectified_flow_inference_config = base_cfg
    cfg.update(base_cfg)

    cfg['num_steps'] = int(_sampling_value(sample, args, 'num_steps'))
    cfg['shift'] = float(_sampling_value(sample, args, 'shift'))

    use_karras_sigmas = args.use_karras_sigmas
    if use_karras_sigmas is None and sample.get(
            'use_karras_sigmas') is not None:
        use_karras_sigmas = sample['use_karras_sigmas']
    if use_karras_sigmas is not None:
        cfg['use_karras_sigmas'] = bool(use_karras_sigmas)


def _sample_seed(sample: dict[str, Any],
                 args: argparse.Namespace,
                 default: int | None = 7) -> int | None:
    value = args.seed if args.seed is not None else sample.get('seed', default)
    return None if value is None else int(value)


def _latent_size(vision_args: dict[str, Any]) -> tuple[int, int]:
    return (
        max(1,
            int(vision_args['height']) // 16),
        max(1,
            int(vision_args['width']) // 16),
    )


def _text_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    return chat_input_ids(tokenizer, prompt).to(device=device)


def _decode_and_save(model, latents: torch.Tensor, output_base: Path,
                     fps: int) -> Path:
    return save_media(
        model.decode_vision_latents(latents), output_base, fps=float(fps))


def _save_frame_png(frame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode='RGB').save(output_path)
    return output_path


def _reasoner_max_new_tokens(sample: dict[str, Any],
                             args: argparse.Namespace) -> int:
    if args.max_new_tokens is not None:
        return int(args.max_new_tokens)
    if sample.get('max_new_tokens') is not None:
        return int(sample['max_new_tokens'])
    if sample.get('max_tokens') is not None:
        return int(sample['max_tokens'])
    return 128


def _prepare_samples(args: argparse.Namespace):
    if args.readme_repro:
        checkpoint, run_dir, samples = _prepare_readme_inputs(args)
        if args.prepare_only:
            print(f'Prepared inputs in {run_dir / "inputs"}')
            return checkpoint, run_dir / 'outputs', []
        output_dir = run_dir / 'outputs'
    else:
        if args.input is None or args.output_dir is None:
            raise SystemExit('--input and --output-dir are required unless '
                             '--readme-repro is set.')
        checkpoint = resolve_under_root(args.checkpoint)
        output_dir = resolve_under_root(args.output_dir)
        samples = load_samples(args.input)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    for index, sample in enumerate(samples):
        sample.setdefault('name', f'{sample["model_mode"]}_{index}')
    return checkpoint, output_dir, samples


@dataclass
class ActionContext:
    embodiment_id: int
    raw_action_dim: int
    action_chunk_size: int
    raw_frames: int
    vision_args: dict[str, Any]
    text_ids: torch.Tensor
    video: torch.Tensor
    conditioning_fps: float


class Cosmos3InferApp:
    RUNNER_METHODS = {
        'text2image': '_run_vision',
        'text2video': '_run_vision',
        'image2video': '_run_vision',
        'video2video': '_run_vision',
        'forward_dynamics': '_run_action',
        'inverse_dynamics': '_run_action',
        'policy': '_run_action',
        'reasoning': '_run_reasoning',
    }
    ACTION_MODES = {'forward_dynamics', 'inverse_dynamics', 'policy'}

    def __init__(self, *, args: argparse.Namespace, checkpoint: Path,
                 output_dir: Path, samples: list[dict[str, Any]]):
        self.args = args
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.samples = samples
        self.checkpoint_root = checkpoint_root_path(checkpoint)
        self.model_size = resolve_model_size(
            str(self.checkpoint_root), args.model_size)
        self.model = None
        self.tokenizer = None
        self.processor = None

    # App lifecycle.
    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_components()
        if self.args.load_only:
            print('loaded')
            return

        manifest = self._manifest()
        for sample in self.samples:
            report = self._run_sample(sample)
            report['name'] = sample['name']
            manifest['samples'].append(report)
            print(json.dumps(report, indent=2))

        manifest_path = self.output_dir / 'fluxvla_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'wrote {manifest_path}')

    def _load_components(self) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_root), trust_remote_code=True)
        if self._needs_reasoner:
            self.processor = AutoProcessor.from_pretrained(
                str(self.checkpoint_root), trust_remote_code=True)
        self.model = build_and_load_fluxvla_model(
            checkpoint=str(self.checkpoint_root),
            model_size=self.model_size,
            vae_path=self.args.vae_path,
            dtype=dtype_from_name(self.args.dtype),
            device=torch.device(self.args.device),
            action_horizon=self._max_action_horizon(),
            include_visual=self._needs_reasoner)

    @property
    def _needs_reasoner(self) -> bool:
        return any(sample['model_mode'] == 'reasoning'
                   for sample in self.samples)

    def _manifest(self) -> dict[str, Any]:
        return dict(
            backend='fluxvla',
            input='readme-repro'
            if self.args.readme_repro else str(self.args.input),
            checkpoint=str(self.checkpoint_root),
            model_size=self.model_size,
            output_dir=str(self.output_dir),
            samples=[],
        )

    def _max_action_horizon(self) -> int | None:
        horizons = [
            self._action_chunk_size(sample) for sample in self.samples
            if sample['model_mode'] in self.ACTION_MODES
        ]
        return max(horizons) if horizons else None

    # Per-sample dispatch.
    @torch.no_grad()
    def _run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample_dir = self.output_dir / sample['name']
        sample_dir.mkdir(parents=True, exist_ok=True)
        method_name = self.RUNNER_METHODS.get(sample['model_mode'])
        if method_name is None:
            raise ValueError(f'Unsupported mode {sample["model_mode"]!r}.')
        return getattr(self, method_name)(sample, sample_dir)

    # Vision and reasoning modes.
    def _run_reasoning(self, sample: dict[str, Any],
                       sample_dir: Path) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{
            'type': 'text',
            'text': str(sample.get('prompt', '')),
        }]
        if sample.get('vision_path') is not None:
            image_path = download_if_needed(
                sample['vision_path'],
                cache_dir=sample_dir / 'inputs',
                name='reasoning_image',
            )
            content.insert(0, {
                'type': 'image',
                'image': Image.open(image_path).convert('RGB')
            })
        inputs = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': content
            }],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        seed = _sample_seed(sample, self.args, default=None)
        if seed is not None:
            torch.manual_seed(seed)
            if self.model.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
        generation_kwargs = {
            key: sample[key]
            for key in ('do_sample', 'temperature', 'top_k', 'top_p',
                        'repetition_penalty') if sample.get(key) is not None
        }
        output_ids = self.model.generate_reasoner_text(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            mm_token_type_ids=inputs.get('mm_token_type_ids'),
            max_new_tokens=_reasoner_max_new_tokens(sample, self.args),
            return_only_new_tokens=True,
            **generation_kwargs,
        )
        text = self.processor.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        output_path = sample_dir / f'{sample["name"]}.txt'
        output_path.write_text(text + '\n')
        return dict(
            mode=sample['model_mode'],
            output=str(output_path),
            text=text,
            tokens=int(output_ids.shape[-1]),
        )

    def _run_vision(self, sample: dict[str, Any],
                    sample_dir: Path) -> dict[str, Any]:
        mode = sample['model_mode']
        vision_args = resolve_vision_args(
            sample,
            resolution=self.args.resolution,
            aspect_ratio=self.args.aspect_ratio,
            fps=self.args.fps,
            num_frames=self.args.num_frames,
        )
        text_ids = _text_ids(self.tokenizer, str(sample.get('prompt', '')),
                             self.model.device)
        negative_ids = _text_ids(self.tokenizer,
                                 _negative_prompt(sample, self.args),
                                 self.model.device)
        latent_frames = latent_num_frames(
            int(vision_args['num_frames']),
            self.model.vision_vae.temporal_compression_factor)
        condition_images, condition_indexes = self._vision_conditions(
            sample, vision_args)

        _apply_sampling_config(self.model, sample, self.args)
        latent_height, latent_width = _latent_size(vision_args)
        latents = self.model.generate_vision_latents(
            images=condition_images,
            text_token_ids=text_ids,
            negative_text_token_ids=negative_ids,
            sequence_plan=_sequence_plan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(condition_indexes or []),
            ),
            num_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            seed=_sample_seed(sample, self.args),
            guidance=float(_sampling_value(sample, self.args, 'guidance')),
            conditioning_fps=float(vision_args['fps']),
        )
        output_path = _decode_and_save(self.model, latents,
                                       sample_dir / sample['name'],
                                       int(vision_args['fps']))
        return dict(
            mode=mode,
            output=str(output_path),
            vision_args=vision_args,
            latent_shape=list(latents.shape),
        )

    def _vision_conditions(self, sample: dict[str, Any],
                           vision_args: dict[str, Any]):
        mode = sample['model_mode']
        if mode not in {'image2video', 'video2video'}:
            return None, None
        condition_indexes = sample.get(
            'condition_frame_indexes_vision',
            mode_defaults(mode).get('condition_frame_indexes_vision', [0]),
        )
        condition_raw_frames = (1 if mode == 'image2video' else int(
            vision_args['num_frames']))
        condition_images = load_media_frames(
            sample['vision_path'],
            cache_dir=self.output_dir / sample['name'] / 'inputs',
            height=int(vision_args['height']),
            width=int(vision_args['width']),
            max_frames=condition_raw_frames,
            keep=str(sample.get('condition_video_keep', 'first')),
        ).unsqueeze(0)
        return condition_images, condition_indexes

    # Action modes.
    def _run_action(self, sample: dict[str, Any],
                    sample_dir: Path) -> dict[str, Any]:
        ctx = self._prepare_action_context(sample, sample_dir)
        _apply_sampling_config(self.model, sample, self.args)
        if sample['model_mode'] == 'forward_dynamics':
            return self._run_forward_dynamics(sample, sample_dir, ctx)
        self.model.action_horizon = ctx.action_chunk_size
        if sample['model_mode'] == 'policy':
            return self._run_policy_action(sample, sample_dir, ctx)
        if sample['model_mode'] == 'inverse_dynamics':
            return self._run_inverse_dynamics(sample, sample_dir, ctx)
        raise ValueError(f'Unsupported action mode {sample["model_mode"]!r}.')

    def _action_chunk_size(self, sample: dict[str, Any]) -> int:
        defaults = mode_defaults(sample['model_mode'])
        return int(
            self.args.action_chunk_size or sample.get('action_chunk_size')
            or defaults.get('action_chunk_size'))

    def _action_meta(self, sample: dict[str, Any]) -> tuple[int, int, int]:
        embodiment_id = self.args.embodiment_id
        if embodiment_id is None and sample.get('embodiment_id') is not None:
            embodiment_id = int(sample['embodiment_id'])
        if embodiment_id is None:
            raise ValueError('Action modes require explicit `embodiment_id`.')
        if int(embodiment_id) < 0:
            raise ValueError(
                f'embodiment_id must be non-negative, got {embodiment_id}.')
        raw_action_dim = self.args.raw_action_dim or sample.get(
            'raw_action_dim')
        if raw_action_dim is None:
            raise ValueError('Action modes require explicit `raw_action_dim`.')
        return int(embodiment_id), int(
            raw_action_dim), self._action_chunk_size(sample)

    def _prepare_action_context(self, sample: dict[str, Any],
                                sample_dir: Path) -> ActionContext:
        embodiment_id, raw_action_dim, action_chunk_size = self._action_meta(
            sample)
        raw_frames = action_chunk_size + 1
        vision_args = resolve_vision_args(
            sample,
            resolution=self.args.resolution,
            aspect_ratio=self.args.aspect_ratio,
            fps=self.args.fps,
            num_frames=raw_frames,
        )
        video = load_media_frames(
            sample['vision_path'],
            cache_dir=sample_dir / 'inputs',
            height=int(vision_args['height']),
            width=int(vision_args['width']),
            max_frames=raw_frames,
            keep='first',
        ).unsqueeze(0)
        return ActionContext(
            embodiment_id=embodiment_id,
            raw_action_dim=raw_action_dim,
            action_chunk_size=action_chunk_size,
            raw_frames=raw_frames,
            vision_args=vision_args,
            text_ids=_text_ids(self.tokenizer, str(sample.get('prompt', '')),
                               self.model.device),
            video=video,
            conditioning_fps=float(vision_args['fps']),
        )

    def _load_action_chunks(self, path_or_url: str | Path, *, cache_dir: Path,
                            action_chunk_size: int) -> torch.Tensor:
        path = download_if_needed(path_or_url, cache_dir, name='action_chunks')
        data = json.loads(path.read_text())
        chunks = data['action_chunks'] if isinstance(data, dict) else data
        tensor = torch.tensor(chunks, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(
                f'Expected action chunks [N,T,D], got {tensor.shape}.')
        return _repeat_last_to_length(tensor, action_chunk_size, dim=1)

    def _forward_dynamics_latents(self, *, video: torch.Tensor,
                                  negative_text_ids: torch.Tensor,
                                  actions: torch.Tensor, sample: dict[str,
                                                                      Any],
                                  ctx: ActionContext) -> torch.Tensor:
        latent_frames = latent_num_frames(
            ctx.action_chunk_size + 1,
            self.model.vision_vae.temporal_compression_factor)
        latent_height, latent_width = _latent_size(ctx.vision_args)
        return self.model.generate_vision_latents(
            images=video,
            text_token_ids=ctx.text_ids,
            negative_text_token_ids=negative_text_ids,
            actions=actions.unsqueeze(0),
            embodiment_id=ctx.embodiment_id,
            raw_action_dim=ctx.raw_action_dim,
            sequence_plan=_sequence_plan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
                condition_frame_indexes_action=list(
                    range(ctx.action_chunk_size)),
            ),
            num_frames=latent_frames,
            action_horizon=ctx.action_chunk_size,
            latent_height=latent_height,
            latent_width=latent_width,
            seed=_sample_seed(sample, self.args),
            guidance=float(_sampling_value(sample, self.args, 'guidance')),
            conditioning_fps=ctx.conditioning_fps,
            action_fps=ctx.conditioning_fps,
        )

    def _run_forward_dynamics(self, sample: dict[str, Any], sample_dir: Path,
                              ctx: ActionContext) -> dict[str, Any]:
        if sample.get('action_chunks_path') is not None:
            return self._run_forward_dynamics_chunks(sample, sample_dir, ctx)
        negative_ids = _text_ids(self.tokenizer,
                                 _negative_prompt(sample, self.args),
                                 self.model.device)
        actions = load_actions(
            sample['action_path'],
            cache_dir=sample_dir / 'inputs',
            action_chunk_size=ctx.action_chunk_size,
        )
        latents = self._forward_dynamics_latents(
            video=ctx.video,
            negative_text_ids=negative_ids,
            actions=actions,
            sample=sample,
            ctx=ctx,
        )
        output_path = _decode_and_save(self.model, latents,
                                       sample_dir / sample['name'],
                                       int(ctx.vision_args['fps']))
        return self._action_report(
            sample,
            ctx,
            output=str(output_path),
            latent_shape=list(latents.shape))

    def _run_forward_dynamics_chunks(self, sample: dict[str,
                                                        Any], sample_dir: Path,
                                     ctx: ActionContext) -> dict[str, Any]:
        negative_ids = _text_ids(self.tokenizer,
                                 _negative_prompt(sample, self.args),
                                 self.model.device)
        action_chunks = self._load_action_chunks(
            sample['action_chunks_path'],
            cache_dir=sample_dir / 'inputs',
            action_chunk_size=ctx.action_chunk_size,
        )
        current_frame_path = Path(str(sample['vision_path']))
        chunk_outputs = []
        stitched_frames = []
        for chunk_idx, actions in enumerate(action_chunks):
            video = load_media_frames(
                current_frame_path,
                cache_dir=sample_dir / 'inputs',
                height=int(ctx.vision_args['height']),
                width=int(ctx.vision_args['width']),
                max_frames=ctx.raw_frames,
                keep='first',
            ).unsqueeze(0)
            latents = self._forward_dynamics_latents(
                video=video,
                negative_text_ids=negative_ids,
                actions=actions,
                sample=sample,
                ctx=ctx,
            )
            frames = _decoded_frames_uint8(
                self.model.decode_vision_latents(latents))
            chunk_path = _write_mp4(
                frames, sample_dir / sample['name'] /
                f'{sample["name"]}_chunk_{chunk_idx:02d}.mp4',
                float(ctx.vision_args['fps']))
            chunk_outputs.append(str(chunk_path))
            stitched_frames.extend(frames[1:])
            if chunk_idx + 1 < len(action_chunks):
                current_frame_path = _save_frame_png(
                    frames[-1], sample_dir / 'inputs' /
                    f'{sample["name"]}_ar_frame_{chunk_idx + 1:02d}.png')

        output_path = _write_mp4(
            np.asarray(stitched_frames),
            sample_dir / sample['name'] / f'{sample["name"]}.mp4',
            float(ctx.vision_args['fps']),
        )
        return self._action_report(
            sample,
            ctx,
            output=str(output_path),
            chunk_outputs=chunk_outputs,
            action_chunks=int(action_chunks.shape[0]),
            stitched_frames=len(stitched_frames),
        )

    def _run_policy_action(self, sample: dict[str, Any], sample_dir: Path,
                           ctx: ActionContext) -> dict[str, Any]:
        latent_frames = self._action_latent_frames(ctx)
        result = self.model.generate_joint(
            images=ctx.video,
            text_token_ids=ctx.text_ids,
            embodiment_id=ctx.embodiment_id,
            raw_action_dim=ctx.raw_action_dim,
            sequence_plan=_sequence_plan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
            ),
            num_frames=latent_frames,
            action_horizon=ctx.action_chunk_size,
            seed=_sample_seed(sample, self.args),
            conditioning_fps=ctx.conditioning_fps,
        )
        action_path = save_action_json(result['actions'][0],
                                       sample_dir / f'{sample["name"]}.json')
        video_path = _decode_and_save(self.model, result['vision_latents'],
                                      sample_dir / f'{sample["name"]}_rollout',
                                      int(ctx.vision_args['fps']))
        return self._action_report(
            sample,
            ctx,
            output=str(action_path),
            video_output=str(video_path),
            action_shape=list(result['actions'].shape),
            latent_shape=list(result['vision_latents'].shape),
        )

    def _run_inverse_dynamics(self, sample: dict[str, Any], sample_dir: Path,
                              ctx: ActionContext) -> dict[str, Any]:
        latent_frames = self._action_latent_frames(ctx)
        result = self.model.generate_inverse_dynamics(
            images=ctx.video,
            text_token_ids=ctx.text_ids,
            embodiment_id=ctx.embodiment_id,
            raw_action_dim=ctx.raw_action_dim,
            sequence_plan=_sequence_plan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(range(latent_frames)),
                has_action=True,
            ),
            num_frames=latent_frames,
            action_horizon=ctx.action_chunk_size,
            seed=_sample_seed(sample, self.args),
            conditioning_fps=ctx.conditioning_fps,
            action_fps=ctx.conditioning_fps,
        )
        output_path = save_action_json(result['actions'][0],
                                       sample_dir / f'{sample["name"]}.json')
        return self._action_report(
            sample,
            ctx,
            output=str(output_path),
            action_shape=list(result['actions'].shape),
        )

    def _action_latent_frames(self, ctx: ActionContext) -> int:
        return latent_num_frames(
            ctx.raw_frames, self.model.vision_vae.temporal_compression_factor)

    def _action_report(self, sample: dict[str, Any], ctx: ActionContext,
                       **extra) -> dict[str, Any]:
        return dict(
            mode=sample['model_mode'],
            embodiment_id=ctx.embodiment_id,
            raw_action_dim=ctx.raw_action_dim,
            action_chunk_size=ctx.action_chunk_size,
            vision_args=ctx.vision_args,
            **extra,
        )


def main() -> None:
    args = parse_args()
    checkpoint, output_dir, samples = _prepare_samples(args)
    if args.prepare_only and args.readme_repro:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    Cosmos3InferApp(
        args=args,
        checkpoint=checkpoint,
        output_dir=output_dir,
        samples=samples).run()


if __name__ == '__main__':
    main()
