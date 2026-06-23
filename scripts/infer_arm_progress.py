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
"""Infer ARM advantage labels and visualize episode progress.

Official implementation of https://arxiv.org/abs/2604.03037

Runs strided episode inference, builds cumulative progress from interval
deltas and the done frame, and renders a side-by-side camera + progress
chart video.
"""

from __future__ import annotations
import argparse
import importlib
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
from mmengine import Config, DictAction
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('DATASETS_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
for logger_name in ('datasets', 'datasets.config', 'transformers',
                    'transformers.utils.import_utils'):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arm_awbc.progress_reconstruction import (  # noqa: E402
    build_cumulative_progress, run_strided_episode_inference)

logger = logging.getLogger(__name__)


def _ensure_namespace_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None and hasattr(module, '__path__'):
        return module
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _install_lightweight_fluxvla_imports() -> Dict[str, Any]:
    """Expose only the registry utilities needed by ARM inference."""
    package_paths = {
        'fluxvla':
        REPO_ROOT / 'fluxvla',
        'fluxvla.collators':
        REPO_ROOT / 'fluxvla' / 'collators',
        'fluxvla.datasets':
        REPO_ROOT / 'fluxvla' / 'datasets',
        'fluxvla.datasets.utils':
        REPO_ROOT / 'fluxvla' / 'datasets' / 'utils',
        'fluxvla.engines':
        REPO_ROOT / 'fluxvla' / 'engines',
        'fluxvla.engines.utils':
        REPO_ROOT / 'fluxvla' / 'engines' / 'utils',
        'fluxvla.models':
        REPO_ROOT / 'fluxvla' / 'models',
        'fluxvla.models.backbones':
        REPO_ROOT / 'fluxvla' / 'models' / 'backbones',
        'fluxvla.models.backbones.llms':
        REPO_ROOT / 'fluxvla' / 'models' / 'backbones' / 'llms',
        'fluxvla.models.vlas':
        REPO_ROOT / 'fluxvla' / 'models' / 'vlas',
        'fluxvla.tokenizers':
        REPO_ROOT / 'fluxvla' / 'tokenizers',
        'fluxvla.transforms':
        REPO_ROOT / 'fluxvla' / 'transforms',
    }
    for name, path in package_paths.items():
        _ensure_namespace_package(name, path)

    root = importlib.import_module('fluxvla.engines.utils.root')
    builder = importlib.import_module('fluxvla.engines.utils.builder')
    overwatch = importlib.import_module('fluxvla.engines.utils.overwatch')

    registry_names = [
        'TOKENIZERS',
        'TRANSFORMS',
        'DATASETS',
        'LLM_BACKBONES',
        'VISION_BACKBONES',
        'PROJECTORS',
        'HEADS',
        'VLAS',
        'RUNNERS',
        'COLLATORS',
        'METRICS',
        'PROCESSORS',
        'VLM_BACKBONES',
        'OPERATORS',
    ]
    builder_names = [
        'build_collator_from_cfg',
        'build_dataset_from_cfg',
        'build_from_cfg',
        'build_head_from_cfg',
        'build_llm_backbone_from_cfg',
        'build_projector_from_cfg',
        'build_tokenizer_from_cfg',
        'build_transform_from_cfg',
        'build_vision_backbone_from_cfg',
        'build_vla_from_cfg',
        'build_vlm_backbone_from_cfg',
    ]
    engines_pkg = sys.modules['fluxvla.engines']
    utils_pkg = sys.modules['fluxvla.engines.utils']
    for attr in registry_names:
        value = getattr(root, attr)
        setattr(engines_pkg, attr, value)
        setattr(utils_pkg, attr, value)
    for attr in builder_names:
        value = getattr(builder, attr)
        setattr(engines_pkg, attr, value)
        setattr(utils_pkg, attr, value)
    setattr(engines_pkg, 'initialize_overwatch',
            overwatch.initialize_overwatch)
    setattr(utils_pkg, 'initialize_overwatch', overwatch.initialize_overwatch)
    return {name: getattr(builder, name) for name in builder_names}


def _register_arm_runtime_modules() -> Dict[str, Any]:
    """Register only modules required by the ARM inference config."""
    builders = _install_lightweight_fluxvla_imports()
    importlib.import_module('fluxvla.transforms.transform_inputs')
    importlib.import_module('fluxvla.transforms.transform_images')
    importlib.import_module('fluxvla.transforms.normalize')
    importlib.import_module('fluxvla.transforms.transform_prompts')
    for module_name in [
            'fluxvla.collators.dict_collator',
            'fluxvla.datasets.arm_dataset',
            'fluxvla.models.backbones.llms.arm',
            'fluxvla.models.vlas.arm_reward_model',
            'fluxvla.tokenizers.pretrained_tokenizer',
    ]:
        importlib.import_module(module_name)
    return builders


def parse_args():
    """Parse command-line arguments for ARM inference."""
    parser = argparse.ArgumentParser(
        description='Infer ARM advantage and visualize episode progress.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings in xxx=yyy format.')
    # JSONL batch export (optional)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None)
    # Episode visualization
    parser.add_argument(
        '--episode-idx',
        type=int,
        default=None,
        help='Episode index for ARM visualization. '
        'When set, runs strided episode inference and builds a video.')
    parser.add_argument('--dataset-idx', type=int, default=0)
    parser.add_argument(
        '--inference-stride',
        type=int,
        default=150,
        help='Inference every N frames (150 = 5s at 30fps).')
    parser.add_argument(
        '--image-key',
        type=str,
        default=None,
        help='Camera key for raw frame decode (default: first video_keys).')
    parser.add_argument(
        '--fps', type=int, default=30, help='Source video FPS.')
    parser.add_argument(
        '--vis-fps',
        type=int,
        default=5,
        help='Output visualization video FPS (frames sampled 1/sec).')
    parser.add_argument(
        '--chart-width',
        type=int,
        default=900,
        help='Width of the progress chart panel in pixels.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./arm_viz',
        help='Directory for visualization frames and video.')
    args = parser.parse_args()
    if args.output_path is None and args.episode_idx is None:
        parser.error('Set --episode-idx for visualization or '
                     '--output-path for JSONL export.')
    if args.inference_stride <= 0:
        parser.error('--inference-stride must be positive')
    if args.fps <= 0 or args.vis_fps <= 0:
        parser.error('--fps and --vis-fps must be positive')
    return args


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _progress_delta(p0: float, p1: float, eps: float = 1e-3) -> int:
    delta = p1 - p0
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0


def normalize_image_to_hwc(image) -> np.ndarray:
    """Convert image-like input to (H, W, 3) uint8."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        return np.array(image.convert('RGB'))
    else:
        image = np.asarray(image)

    image = np.squeeze(image)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3:
        if image.shape[0] in (1, 3) and image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def draw_progress_chart(
    pred_progress: np.ndarray,
    gt_progress: np.ndarray | None,
    current_sec: int,
    total_secs: int,
    height: int,
    width: int = 900,
    interval_predictions: List[int] | None = None,
    interval_targets: List[int] | None = None,
    cls_prediction: int | None = None,
    cls_target: int | None = None,
    window_sec_indices: List[int] | None = None,
    inference_keyframe_secs: List[int] | None = None,
) -> np.ndarray:
    """Draw progress panel (camera height x chart width, RGB)."""
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        font_bold = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    panel = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    pad_l, pad_r, pad_t, pad_b = 40, 15, 30, 90
    chart_x0 = pad_l
    chart_x1 = width - pad_r
    chart_w = chart_x1 - chart_x0
    available_h = max(1, height - pad_t - pad_b)
    chart_h = int(available_h * 0.8)
    chart_y0 = pad_t
    chart_y1 = chart_y0 + chart_h

    draw.text((pad_l, 6), 'Task Progress', fill=(40, 40, 40), font=font_bold)

    axis_color = (80, 80, 80)
    tick_color = (140, 140, 140)
    text_color = (60, 60, 60)
    draw.line([(chart_x0, chart_y0), (chart_x0, chart_y1)],
              fill=axis_color,
              width=1)
    draw.line([(chart_x0, chart_y1), (chart_x1, chart_y1)],
              fill=axis_color,
              width=1)

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y1 - int(tick * chart_h)
        draw.line([(chart_x0 - 4, y), (chart_x0, y)], fill=tick_color, width=1)
        draw.text((2, y - 7),
                  f'{int(tick * 100)}%',
                  fill=text_color,
                  font=font)

    tick_interval = max(1, total_secs // 5)
    for t in range(0, total_secs + 1, tick_interval):
        x = chart_x0 + int(t / max(total_secs - 1, 1) * chart_w)
        draw.line([(x, chart_y1), (x, chart_y1 + 4)], fill=tick_color, width=1)
        draw.text((x - 8, chart_y1 + 6), f'{t}s', fill=text_color, font=font)

    if total_secs > 1:
        pts = []
        for i, p in enumerate(pred_progress):
            x = chart_x0 + int(i / (total_secs - 1) * chart_w)
            y = chart_y1 - int(float(p) * chart_h)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(40, 120, 240), width=2)
        for x, y in pts:
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(40, 120, 240))

        if inference_keyframe_secs is not None:
            for keyframe_sec in inference_keyframe_secs:
                if 0 <= keyframe_sec < total_secs:
                    kx = chart_x0 + int(keyframe_sec /
                                        (total_secs - 1) * chart_w)
                    ky = chart_y1 - int(
                        float(pred_progress[keyframe_sec]) * chart_h)
                    draw.ellipse([(kx - 5, ky - 5), (kx + 5, ky + 5)],
                                 fill=(255, 165, 0),
                                 outline=(200, 100, 0),
                                 width=1)

    if (gt_progress is not None and len(gt_progress) == total_secs
            and total_secs > 1):
        pts_gt = []
        for i, p in enumerate(gt_progress):
            x = chart_x0 + int(i / (total_secs - 1) * chart_w)
            y = chart_y1 - int(float(p) * chart_h)
            pts_gt.append((x, y))
        for i in range(len(pts_gt) - 1):
            draw.line([pts_gt[i], pts_gt[i + 1]], fill=(10, 170, 60), width=2)
        for x, y in pts_gt:
            draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(10, 170, 60))

    if total_secs > 1:
        cx = chart_x0 + int(current_sec / (total_secs - 1) * chart_w)
    else:
        cx = chart_x0
    draw.line([(cx, chart_y0), (cx, chart_y1)], fill=(220, 40, 40), width=2)

    if window_sec_indices is not None and len(window_sec_indices) == 5:
        labels = ['-4', '-3', '-2', '-1', 'T']
        colors = [(200, 100, 200), (180, 120, 200), (160, 140, 200),
                  (140, 160, 200), (120, 180, 200)]
        for win_sec, label, color in zip(window_sec_indices, labels, colors):
            if 0 <= win_sec < total_secs:
                if total_secs > 1:
                    wx = chart_x0 + int(win_sec / (total_secs - 1) * chart_w)
                else:
                    wx = chart_x0
                if abs(wx - cx) > 2:
                    draw.line([(wx, chart_y0), (wx, chart_y1)],
                              fill=color,
                              width=1)
                    try:
                        font_small = ImageFont.truetype(
                            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                            10)
                    except OSError:
                        font_small = font
                    draw.text((wx - 8, chart_y0 + 2),
                              label,
                              fill=color,
                              font=font_small)

    bottom_text_y0 = chart_y1 + 25
    cur_pred = (
        float(pred_progress[current_sec])
        if current_sec < len(pred_progress) else 0.0)
    pred_pct = '%.2f%%' % (100.0 * cur_pred)
    if gt_progress is not None and current_sec < len(gt_progress):
        cur_gt = float(gt_progress[current_sec])
        gt_pct = '%.2f%%' % (100.0 * cur_gt)
        text = ('t=%ds  P_pred=%s  P_gt=%s' % (current_sec, pred_pct, gt_pct))
    else:
        text = 't=%ds  P_pred=%s' % (current_sec, pred_pct)
    draw.text((pad_l, bottom_text_y0),
              text,
              fill=(120, 80, 20),
              font=font_bold)

    line_y = bottom_text_y0 + 24
    if current_sec > 0 and current_sec < len(pred_progress):
        prev_progress = float(pred_progress[current_sec - 1])
        curr_progress = float(pred_progress[current_sec])
        delta_pred_curr = _progress_delta(prev_progress, curr_progress)
        if interval_predictions is not None and len(interval_predictions) == 4:
            iv_pred_str = ' '.join('%+d' % p for p in interval_predictions)
            draw.text((pad_l, line_y),
                      'Δ_pred: %s (curr=%+d)' % (iv_pred_str, delta_pred_curr),
                      fill=(40, 120, 240),
                      font=font)
        else:
            draw.text((pad_l, line_y),
                      'Δ_pred: curr=%+d' % delta_pred_curr,
                      fill=(40, 120, 240),
                      font=font)
        line_y += 18
    elif interval_predictions is not None:
        iv_pred = ' '.join('%+d' % p for p in interval_predictions)
        draw.text((pad_l, line_y),
                  'Δ_pred: %s' % iv_pred,
                  fill=(40, 120, 240),
                  font=font)
        line_y += 18

    if interval_targets is not None and len(interval_targets) == 4:
        iv_gt_str = ' '.join('%+d' % p for p in interval_targets)
        if (gt_progress is not None and current_sec > 0
                and current_sec < len(gt_progress)):
            delta_gt_curr = _progress_delta(
                float(gt_progress[current_sec - 1]),
                float(gt_progress[current_sec]))
            draw.text((pad_l, line_y),
                      'Δ_gt:   %s (curr=%+d)' % (iv_gt_str, delta_gt_curr),
                      fill=(10, 170, 60),
                      font=font)
        else:
            draw.text((pad_l, line_y),
                      f'Δ_gt:   {iv_gt_str}',
                      fill=(10, 170, 60),
                      font=font)
        line_y += 18

    if cls_prediction is not None:
        s_color_pred = (0, 150, 0) if cls_prediction == 1 else (180, 0, 0)
        s_text_pred = 'Done' if cls_prediction == 1 else 'In progress'
        draw.text((pad_l, line_y),
                  f'Done_pred: {s_text_pred}',
                  fill=s_color_pred,
                  font=font)
        line_y += 18
    if cls_target is not None:
        s_color_gt = (0, 150, 0) if cls_target == 1 else (180, 0, 0)
        s_text_gt = 'Done' if cls_target == 1 else 'In progress'
        draw.text((pad_l, line_y),
                  f'Done_gt:   {s_text_gt}',
                  fill=s_color_gt,
                  font=font)

    return np.array(panel)


def _episode_bounds(dataset: Dataset, dataset_idx: int,
                    episode_idx: int) -> Tuple[int, int]:
    episode_ranges = getattr(dataset, 'episode_ranges', None)
    if not episode_ranges:
        raise ValueError('Dataset has no episode_ranges.')
    key = (dataset_idx, episode_idx)
    if key not in episode_ranges:
        available = sorted({ep for (_, ep) in episode_ranges})
        raise ValueError(
            f'Episode {episode_idx} not found for dataset_idx={dataset_idx}. '
            f'Available episode indices: {available[:20]}...')
    return episode_ranges[key]


def _read_progress(dataset: Dataset, frame_idx: int) -> float:
    row = dataset.dataset[frame_idx]
    progress = row.get('progress', 0.0)
    if isinstance(progress, torch.Tensor):
        return float(progress.item())
    return float(progress)


def _decode_raw_frame(dataset: Dataset, frame_idx: int,
                      image_key: str) -> np.ndarray:
    from fluxvla.datasets.utils.video_decode import (
        build_lerobot_video_path, decode_video_frames_torchvision)
    row = dataset.dataset[frame_idx]
    dataset_idx = dataset._get_dataset_index(frame_idx)
    episode_index = int(row['episode_index'])
    video_path = build_lerobot_video_path(
        dataset.data_root_path[dataset_idx],
        dataset.info[dataset_idx],
        dataset.episodes_meta[dataset_idx][episode_index],
        episode_index,
        image_key,
    )
    timestamp = float(row['timestamp'])
    frames = decode_video_frames_torchvision(video_path, [timestamp])
    return normalize_image_to_hwc(frames.numpy())


def run_episode_inference(
    model: Any,
    dataset: Dataset,
    collator: Any,
    episode_idx: int,
    dataset_idx: int,
    inference_stride: int,
    device: str,
    interval_eps: float = 1e-3,
) -> Tuple[List[dict], np.ndarray]:
    """Strided inference on one episode; returns results and dense progress."""
    ep_start, ep_end = _episode_bounds(dataset, dataset_idx, episode_idx)
    episode_length = ep_end - ep_start
    n_history = int(getattr(model, 'n_history_steps', 4))
    frame_gap = int(getattr(dataset, 'frame_gap', 30))

    logger.info('Episode %d (dataset_idx=%d): frames %d..%d (length=%d)',
                episode_idx, dataset_idx, ep_start, ep_end, episode_length)

    inference_results = run_strided_episode_inference(
        model=model,
        dataset=dataset,
        collator=collator,
        ep_start=ep_start,
        ep_end=ep_end,
        inference_stride=inference_stride,
        device=device,
        progress_desc=f'Episode {episode_idx}',
    )

    for result in inference_results:
        frame_idx = result['frame_index']
        window_frame_indices = []
        for step in range(n_history, -1, -1):
            target_idx = frame_idx - step * frame_gap
            clamped_idx = max(ep_start, min(ep_end - 1, target_idx))
            window_frame_indices.append(clamped_idx)

        interval_targets_list = []
        for i in range(n_history):
            p0 = _read_progress(dataset, window_frame_indices[i])
            p1 = _read_progress(dataset, window_frame_indices[i + 1])
            interval_targets_list.append(_progress_delta(p0, p1, interval_eps))

        result.update({
            'episode_index': episode_idx,
            'interval_targets': interval_targets_list,
            'window_frame_indices': window_frame_indices,
        })

    frame_progress = build_cumulative_progress(inference_results,
                                               episode_length)
    return inference_results, frame_progress


def _encode_video_from_dir(imgs_dir: Path, video_path: Path, fps: int) -> None:
    import imageio

    frame_paths = sorted(imgs_dir.glob('frame-*.png'))
    if not frame_paths:
        raise FileNotFoundError(f'No frames found in {imgs_dir}')
    frames = [imageio.imread(path) for path in frame_paths]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(video_path), frames, fps=fps)
    logger.info('Video saved to %s', video_path)


def create_visualization_video(
    dataset: Dataset,
    episode_idx: int,
    dataset_idx: int,
    frame_progress: np.ndarray,
    inference_results: List[dict],
    output_dir: Path,
    image_key: str,
    fps: int,
    vis_fps: int,
    chart_width: int,
) -> Path:
    """Build 1 frame/sec panels (camera | chart) and encode mp4."""
    ep_start, ep_end = _episode_bounds(dataset, dataset_idx, episode_idx)
    episode_length = ep_end - ep_start

    sec_frame_indices = list(range(0, episode_length, fps))
    total_secs = len(sec_frame_indices)
    sec_pred_progress = np.array(
        [float(frame_progress[i]) for i in sec_frame_indices],
        dtype=np.float32)

    infer_local = np.array(
        [r['frame_index'] - ep_start for r in inference_results],
        dtype=np.int32)
    infer_cls_preds = [r.get('cls_prediction') for r in inference_results]

    inference_keyframe_secs = []
    for result in inference_results:
        keyframe_local = result['frame_index'] - ep_start
        keyframe_sec = keyframe_local // fps
        if 0 <= keyframe_sec < total_secs:
            inference_keyframe_secs.append(keyframe_sec)
    inference_keyframe_secs = sorted(set(inference_keyframe_secs))

    sec_gt_progress_list: List[float] = []
    for local_idx in sec_frame_indices:
        frame_idx = ep_start + local_idx
        sec_gt_progress_list.append(_read_progress(dataset, frame_idx))
    sec_gt_progress = np.array(sec_gt_progress_list, dtype=np.float32)

    vis_dir = output_dir / ('episode_%04d' % episode_idx)
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Creating %d visualization frames (1/sec) for episode %d...',
                total_secs, episode_idx)

    for sec_idx, local_idx in enumerate(
            tqdm(sec_frame_indices, desc='Creating frames')):
        frame_idx = ep_start + local_idx
        image = _decode_raw_frame(dataset, frame_idx, image_key)
        height, _ = image.shape[:2]

        window_sec_indices = []
        for offset in range(-4, 1):
            win_sec = max(0, min(sec_idx + offset, total_secs - 1))
            window_sec_indices.append(win_sec)

        interval_preds = []
        for i in range(len(window_sec_indices) - 1):
            sec0 = window_sec_indices[i]
            sec1 = window_sec_indices[i + 1]
            if sec0 < len(sec_pred_progress) and sec1 < len(sec_pred_progress):
                interval_preds.append(
                    _progress_delta(
                        float(sec_pred_progress[sec0]),
                        float(sec_pred_progress[sec1])))
            else:
                interval_preds.append(0)

        interval_tgts = []
        for i in range(len(window_sec_indices) - 1):
            sec0 = window_sec_indices[i]
            sec1 = window_sec_indices[i + 1]
            if sec0 < len(sec_gt_progress) and sec1 < len(sec_gt_progress):
                interval_tgts.append(
                    _progress_delta(
                        float(sec_gt_progress[sec0]),
                        float(sec_gt_progress[sec1])))
            else:
                interval_tgts.append(0)

        nearest_pos = int(np.argmin(np.abs(infer_local - local_idx)))
        cls_pred = infer_cls_preds[nearest_pos]
        cls_target = None
        if sec_idx < len(sec_gt_progress):
            cls_target = 1 if float(sec_gt_progress[sec_idx]) >= (1.0 -
                                                                  1e-3) else 0

        chart_panel = draw_progress_chart(
            pred_progress=sec_pred_progress,
            gt_progress=sec_gt_progress,
            current_sec=sec_idx,
            total_secs=total_secs,
            height=height,
            width=chart_width,
            interval_predictions=interval_preds,
            interval_targets=interval_tgts,
            cls_prediction=cls_pred,
            cls_target=cls_target,
            window_sec_indices=window_sec_indices,
            inference_keyframe_secs=inference_keyframe_secs,
        )

        frame_img = Image.fromarray(image)
        draw = ImageDraw.Draw(frame_img)
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10),
                  f't={sec_idx}s  frame={frame_idx}',
                  fill=(255, 255, 80),
                  font=font)

        combined = np.concatenate([np.array(frame_img), chart_panel], axis=1)
        frame_name = 'frame-%06d.png' % sec_idx
        Image.fromarray(combined).save(vis_dir / frame_name)

    video_name = 'episode_%04d_visualization.mp4' % episode_idx
    video_path = output_dir / video_name
    logger.info(
        'Encoding video (%d frames @ %dfps) to %s',
        total_secs,
        vis_fps,
        video_path,
    )
    _encode_video_from_dir(vis_dir, video_path, vis_fps)
    return video_path


def _write_jsonl(output_path: str, records: List[Dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def _run_jsonl_export(
    model: Any,
    dataset: Dataset,
    collator: Any,
    args: argparse.Namespace,
) -> None:
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator)
    results: List[Dict] = []
    frame_index = int(getattr(model, 'n_obs_steps', 0))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            success_prob, interval_pred = model.predict_advantage(
                images=batch['images'],
                text_input_ids=batch['text_input_ids'],
                text_attention_mask=batch['text_attention_mask'],
                states=batch['states'],
                lengths=batch['lengths'],
            )
            success_prob = success_prob.detach().cpu()
            interval_pred = interval_pred.detach().cpu()
            batch_size = len(batch['episode_index'])
            for idx in range(batch_size):
                center_interval_idx = min(frame_index - 1,
                                          max(interval_pred.shape[1] - 1, 0))
                record = {
                    'episode_index': _as_int(batch['episode_index'][idx]),
                    'current_index': _as_int(batch['current_index'][idx]),
                    'task_description': batch['task_description'][idx],
                    'pred_success_prob': float(success_prob[idx]),
                    'pred_success': bool(success_prob[idx] >= 0.5),
                    'pred_interval': int(interval_pred[idx,
                                                       center_interval_idx]),
                    'pred_interval_sequence': interval_pred[idx].tolist(),
                }
                if 'progress' in batch:
                    record['gt_progress'] = _as_float(batch['progress'][idx])
                results.append(record)

    assert args.output_path is not None
    _write_jsonl(args.output_path, results)
    logger.info('Wrote %d records to %s', len(results), args.output_path)


def main():
    """Run ARM JSONL export and/or episode visualization."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    builders = _register_arm_runtime_modules()
    build_collator_from_cfg = builders['build_collator_from_cfg']
    build_dataset_from_cfg = builders['build_dataset_from_cfg']
    build_vla_from_cfg = builders['build_vla_from_cfg']

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset_cfg = cfg.get('inference_dataset', cfg.train_dataloader.dataset)
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['training'] = False
    dataset = cast(Dataset, build_dataset_from_cfg(dataset_cfg))

    collator = build_collator_from_cfg(cfg.runner.collator.copy())
    model = cast(Any, build_vla_from_cfg(cfg.model))
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(args.device)).eval()

    if args.output_path is not None:
        _run_jsonl_export(model, dataset, collator, args)

    if args.episode_idx is not None:
        image_key = args.image_key
        if image_key is None:
            video_keys = getattr(dataset, 'video_keys', None)
            if not video_keys:
                raise ValueError(
                    'Specify --image-key or set video_keys in dataset config.')
            image_key = video_keys[0]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inference_results, frame_progress = run_episode_inference(
            model=model,
            dataset=dataset,
            collator=collator,
            episode_idx=args.episode_idx,
            dataset_idx=args.dataset_idx,
            inference_stride=args.inference_stride,
            device=args.device,
            interval_eps=float(getattr(dataset, 'interval_eps', 1e-3)),
        )

        video_path = create_visualization_video(
            dataset=dataset,
            episode_idx=args.episode_idx,
            dataset_idx=args.dataset_idx,
            frame_progress=frame_progress,
            inference_results=inference_results,
            output_dir=output_dir,
            image_key=image_key,
            fps=args.fps,
            vis_fps=args.vis_fps,
            chart_width=args.chart_width,
        )

        results_path = output_dir / ('episode_%04d_results.json' %
                                     args.episode_idx)
        with open(results_path, 'w', encoding='utf-8') as handle:
            json.dump(
                {
                    'episode_idx': args.episode_idx,
                    'dataset_idx': args.dataset_idx,
                    'inference_results': inference_results,
                    'frame_progress': frame_progress.tolist(),
                },
                handle,
                indent=2,
            )

        logger.info('%s', '=' * 60)
        logger.info('Visualization complete!')
        logger.info('Video: %s', video_path)
        logger.info('Results: %s', results_path)
        logger.info('%s', '=' * 60)


if __name__ == '__main__':
    main()
