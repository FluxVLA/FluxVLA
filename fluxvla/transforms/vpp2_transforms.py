"""RoboDojo EE16 and VPP2's T-shaped RGB condition pipeline."""

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from fluxvla.engines.utils.root import TRANSFORMS

CAMERA_KEYS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def resize_frames(frames, size):
    return TF.resize(
        frames, list(size), InterpolationMode.BILINEAR, antialias=True)


def compose_mosaic(head, left, right):
    """Compose RGB uint8 [T,H,W,3] cameras; evaluation does not crop wrists."""
    views = []
    for value in (head, left, right):
        value = torch.as_tensor(np.asarray(value).copy())
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[
                -1] != 3 or value.dtype != torch.uint8:
            raise ValueError('Cameras must be RGB uint8 [T,H,W,3]')
        views.append(value.permute(0, 3, 1, 2).float().div(255.))
    if len({v.shape[0] for v in views}) != 1:
        raise ValueError('Camera histories must have equal lengths')
    head = resize_frames(views[0], (480, 640))
    wrists = torch.cat([resize_frames(v, (240, 240)) for v in views[1:]],
                       dim=2)
    return torch.cat((head, wrists), dim=3)


def prepare_mosaic_video(frames, size=(240, 416), quantize=False):
    """Resize [T,3,H,W] in [0,1] to normalized [3,T,H,W]."""
    video = resize_frames(frames, size).mul(2.).sub(1.)
    if quantize:
        # The reference history cache rounds after the normalized resize.
        video = video.add(1.).mul(127.5).round().clamp(0, 255).to(torch.uint8)
        video = video.float().div(127.5).sub(1.)
    return video.permute(1, 0, 2, 3).contiguous()


@TRANSFORMS.register_module()
class VPP2ComposeRoboDojoMosaic:

    def __init__(self,
                 keys=CAMERA_KEYS,
                 out_key='condition_video',
                 video_size=(240, 416),
                 quantize=False):
        self.keys, self.out_key = tuple(keys), out_key
        self.video_size, self.quantize = tuple(video_size), quantize

    def __call__(self, data):
        data[self.out_key] = prepare_mosaic_video(
            compose_mosaic(*(data[k] for k in self.keys)), self.video_size,
            self.quantize)
        return data


@TRANSFORMS.register_module()
class VPP2HistoryToConditionVideo(VPP2ComposeRoboDojoMosaic):

    def __init__(self,
                 history_keys=tuple(k + '_history' for k in CAMERA_KEYS),
                 out_key='condition_video',
                 video_size=(240, 416),
                 num_frames=9):
        super().__init__(history_keys, out_key, video_size, quantize=True)
        self.num_frames = num_frames

    def __call__(self, data):
        if any(
                np.asarray(data[k]).ndim != 4
                or len(data[k]) != self.num_frames for k in self.keys):
            raise ValueError(
                f'Expected {self.num_frames} frames in each camera history')
        return super().__call__(data)


@TRANSFORMS.register_module()
class VPP2PackEEState:

    def __init__(self,
                 keys=('left_ee_pose', 'left_gripper_qpos', 'right_ee_pose',
                       'right_gripper_qpos'),
                 out_key='states'):
        self.keys, self.out_key = tuple(keys), out_key

    def __call__(self, data):
        values = [
            np.asarray(data[k], dtype=np.float32).reshape(-1)
            for k in self.keys
        ]
        if [v.size for v in values] != [7, 1, 7, 1]:
            raise ValueError(
                'EE16 requires [L xyz+qwxyz, L grip, R xyz+qwxyz, R grip]')
        state = np.concatenate(values)
        if not np.isfinite(state).all():
            raise ValueError('EE16 state contains nonfinite values')
        data[self.out_key] = state
        return data


__all__ = [
    'VPP2ComposeRoboDojoMosaic', 'VPP2HistoryToConditionVideo',
    'VPP2PackEEState'
]
