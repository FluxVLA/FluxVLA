"""Megatron forward step and loss closure for native PI0.5 training."""

from __future__ import annotations

import torch
from megatron.core import parallel_state


def _move(value, device):
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value.to(device) if isinstance(value, torch.Tensor) else value


def forward_step(data_iterator, model):
    device = torch.cuda.current_device()
    if parallel_state.is_pipeline_first_stage():
        batch = next(data_iterator)
        images = _move(batch.get("images"), device)
        img_masks = _move(batch.get("img_masks"), device)
        # FluxVLA DictCollator stacks camera lists as [B,N,C,H,W], while the
        # native PI0 prefix contract consumes one [B,C,H,W] tensor per camera.
        if isinstance(images, torch.Tensor) and images.ndim == 5:
            images = [images[:, idx] for idx in range(images.shape[1])]
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            # A single-sample collate can leave the camera axis as the leading
            # dimension: [N,C,H,W].  Convert each camera to [B=1,C,H,W].
            if (isinstance(img_masks, torch.Tensor) and img_masks.ndim == 2
                    and images.shape[0] == img_masks.shape[0]
                    and images.shape[1] == 3 * img_masks.shape[1]):
                images = [images[:, 3 * idx : 3 * (idx + 1)] for idx in range(img_masks.shape[1])]
                img_masks = [img_masks[:, idx] for idx in range(img_masks.shape[1])]
            elif isinstance(img_masks, torch.Tensor) and img_masks.ndim == 1 \
                    and images.shape[0] == img_masks.shape[0]:
                images = [images[idx].unsqueeze(0) for idx in range(images.shape[0])]
                img_masks = [img_masks[idx].unsqueeze(0) for idx in range(img_masks.shape[0])]
            else:
                images = [images]
        elif isinstance(images, torch.Tensor) and images.ndim == 3:
            images = [images.unsqueeze(0)]
        if isinstance(img_masks, torch.Tensor) and img_masks.ndim == 2:
            img_masks = [img_masks[:, idx] for idx in range(img_masks.shape[1])]
        elif isinstance(img_masks, torch.Tensor) and img_masks.ndim == 0:
            img_masks = [img_masks.reshape(1)]
        elif isinstance(images, list):
            images = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
            if isinstance(img_masks, torch.Tensor) and img_masks.ndim == 1:
                img_masks = [img_masks[idx].reshape(1) for idx in range(img_masks.shape[0])]
        output = model(
            images=images,
            img_masks=img_masks,
            tokens=_move(batch.get("tokens", batch.get("lang_tokens")), device),
            text_masks=_move(batch.get("text_masks", batch.get("lang_masks")), device),
            states=_move(batch.get("states"), device),
            actions=_move(batch.get("actions"), device),
            noise=_move(batch.get("noise"), device),
            time=_move(batch.get("time"), device),
            shared_obs=bool(batch.get("shared_observation", False)),
            offset_mask=_move(batch.get("offset_mask"), device),
            action_is_pad=_move(batch.get("action_is_pad"), device),
            subtask_spans=batch.get("subtask_spans"),
        )
    else:
        output = model()

    def loss_func(loss):
        if not parallel_state.is_pipeline_last_stage():
            return loss, {}
        reduced = loss.detach().float().clone()
        group = parallel_state.get_data_parallel_group()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced, group=group)
            reduced /= parallel_state.get_data_parallel_world_size()
        return loss, {"lm_loss": reduced}

    return output, loss_func
