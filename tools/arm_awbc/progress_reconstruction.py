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
"""Rebuild dense ARM progress curves from interval + success head outputs.

Official implementation of https://arxiv.org/abs/2604.03037
"""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


def extract_last_interval_delta(interval_predictions: List[int]) -> int:
    """Return the last interval label in the causal window (t-1 -> t)."""
    if not interval_predictions:
        return 0
    return int(interval_predictions[-1])


def build_cumulative_progress(
    inference_results: List[dict],
    episode_length: int,
) -> np.ndarray:
    """Build dense per-frame progress in ``[0, 1]`` from strided keyframes.

    The curve accumulates interval-head deltas ``{-1, 0, +1}`` across inference
    keyframes, normalizes to ``[0, 1]`` using the first success-head "done"
    prediction as the completion anchor, then linearly interpolates to every
    frame in the episode.
    """
    if not inference_results:
        return np.zeros(episode_length, dtype=np.float32)

    ep_start = inference_results[0]['frame_index']

    keyframe_local: List[int] = []
    keyframe_score: List[float] = []
    keyframe_cls_pred: List[int] = []

    cumulative = 0.0
    for result in inference_results:
        delta = extract_last_interval_delta(
            result.get('interval_predictions', []))
        cumulative += delta
        local_idx = result['frame_index'] - ep_start
        keyframe_local.append(local_idx)
        keyframe_score.append(cumulative)
        keyframe_cls_pred.append(result.get('cls_prediction', 0))

    keyframe_local_arr = np.array(keyframe_local, dtype=np.float32)
    keyframe_score_arr = np.array(keyframe_score, dtype=np.float32)
    keyframe_cls_arr = np.array(keyframe_cls_pred, dtype=np.int32)

    first_done_idx = None
    first_done_local = None
    for i, cls_pred in enumerate(keyframe_cls_arr):
        if cls_pred == 1:
            first_done_idx = i
            first_done_local = int(keyframe_local_arr[i])
            break

    if first_done_idx is not None:
        scores_before_done = keyframe_score_arr[:first_done_idx + 1]
        if len(scores_before_done) > 1:
            score_min = scores_before_done.min()
            score_max = scores_before_done.max()
            if score_max > score_min:
                keyframe_norm_before = ((scores_before_done - score_min) /
                                        (score_max - score_min))
            else:
                keyframe_norm_before = np.linspace(
                    0.0, 1.0, len(scores_before_done), dtype=np.float32)
        else:
            keyframe_norm_before = np.array([1.0], dtype=np.float32)
        keyframe_norm_before[-1] = 1.0
        keyframe_norm_after = np.ones(
            len(keyframe_score_arr) - first_done_idx - 1, dtype=np.float32)
        keyframe_norm = np.concatenate(
            [keyframe_norm_before, keyframe_norm_after])
    else:
        score_min = keyframe_score_arr.min()
        score_max = keyframe_score_arr.max()
        if score_max > score_min:
            keyframe_norm = ((keyframe_score_arr - score_min) /
                             (score_max - score_min))
        else:
            keyframe_norm = np.linspace(
                0.0, 1.0, len(keyframe_score_arr), dtype=np.float32)

    all_frames = np.arange(episode_length, dtype=np.float32)
    frame_progress = np.interp(all_frames, keyframe_local_arr,
                               keyframe_norm).astype(np.float32)
    if first_done_local is not None:
        frame_progress[first_done_local:] = 1.0
    return frame_progress


def run_strided_episode_inference(
    model: Any,
    dataset: Any,
    collator: Any,
    ep_start: int,
    ep_end: int,
    inference_stride: int,
    device: str,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> List[Dict[str, Any]]:
    """Run strided ARM inference on one episode and return keyframe records."""
    model.eval()
    inference_frame_indices = list(range(ep_start, ep_end, inference_stride))
    if ep_end > ep_start and (ep_end - 1) not in inference_frame_indices:
        inference_frame_indices.append(ep_end - 1)

    inference_results: List[Dict[str, Any]] = []
    frame_iter = inference_frame_indices
    if show_progress:
        frame_iter = tqdm(
            inference_frame_indices,
            desc=progress_desc or f'frames {ep_start}..{ep_end - 1}',
            leave=False,
        )

    with torch.inference_mode():
        for frame_idx in frame_iter:
            sample = dataset[frame_idx]
            batch = collator([sample])
            success_prob, interval_pred, _ = model.predict_advantage(
                images=batch['images'],
                text_input_ids=batch['text_input_ids'],
                text_attention_mask=batch['text_attention_mask'],
                states=batch['states'],
                lengths=batch['lengths'],
                return_interval_probs=True,
            )

            interval_predictions = interval_pred[0].cpu().numpy().tolist()
            cls_prob = float(success_prob[0].item())
            inference_results.append({
                'frame_index': frame_idx,
                'interval_predictions': interval_predictions,
                'cls_prob': cls_prob,
                'cls_prediction': int(cls_prob >= 0.5),
            })
    return inference_results
