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

import math
import os
import sys
import time
from ctypes.util import find_library
from importlib import import_module
from pathlib import Path

import imageio
import numpy as np
import torch
from mmengine.utils import digit_version
from PIL import Image

from .image_ops import (crop_and_resize_numpy, jpeg_roundtrip_numpy,
                        resize_hwc_lanczos3_numpy)

OPENVLA_V01_SYSTEM_PROMPT = (
    'A chat between a curious user and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the user's questions."  # noqa: E501
)

ROBOSUITE_MINIMUM_VERSION = '1.5.0'
ROBOSUITE_MAXIMUM_VERSION = '1.5.2'
NVIDIA_EGL_VENDOR_JSON = """{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
"""

# TODO: Change to data pipeline.


def resize_image(img, resize_size):
    """
    Resizes an image to the specified size using NumPy/OpenCV Lanczos3.

    Args:
        img (np.ndarray): The input image to resize, expected
            to be in HWC format.
        resize_size (int or tuple): The target size for resizing.
            If an int is provided, the image will be resized
            to (resize_size, resize_size). If a tuple is provided,
            it should be in the format (height, width).

    Returns:
        np.ndarray: The resized image, clipped to the range [0, 255]
            and converted to uint8.
    """
    assert isinstance(resize_size, tuple)
    img = jpeg_roundtrip_numpy(img)
    return resize_hwc_lanczos3_numpy(img, resize_size[0], resize_size[1])


def _has_nvidia_egl_library() -> bool:
    if find_library('EGL_nvidia'):
        return True

    common_paths = (
        '/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0',
        '/usr/lib64/libEGL_nvidia.so.0',
        '/usr/lib/aarch64-linux-gnu/libEGL_nvidia.so.0',
    )
    return any(os.path.exists(path) for path in common_paths)


def _nvidia_egl_vendor_file_candidates() -> tuple[Path, ...]:
    env_path = os.environ.get('FLUXVLA_EGL_VENDOR_FILE')
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        Path(sys.prefix) / 'etc' / 'fluxvla' / 'egl_vendor.d' /
        '10_nvidia.json',
    ])
    return tuple(candidates)


def _nvidia_egl_vendor_file_write_candidates() -> tuple[Path, ...]:
    env_path = os.environ.get('FLUXVLA_EGL_VENDOR_FILE')
    if env_path:
        return (Path(env_path).expanduser(), )
    return (Path(sys.prefix) / 'etc' / 'fluxvla' / 'egl_vendor.d' /
            '10_nvidia.json', )


def _is_nvidia_egl_vendor_file(path: Path) -> bool:
    try:
        return path.is_file() and 'libEGL_nvidia.so.0' in path.read_text()
    except OSError:
        return False


def _ensure_nvidia_egl_vendor_file() -> str | None:
    for path in _nvidia_egl_vendor_file_candidates():
        if _is_nvidia_egl_vendor_file(path):
            return path.as_posix()

    for path in _nvidia_egl_vendor_file_write_candidates():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(NVIDIA_EGL_VENDOR_JSON)
            return path.as_posix()
        except OSError:
            continue

    return None


def configure_mujoco_egl_defaults() -> None:
    """Default MuJoCo offscreen rendering to NVIDIA EGL when available.

    A system with only Mesa's GLVND vendor JSON can still create an EGL
    context, but LIBERO rendered pixels differ from the NVIDIA EGL path used by
    the reference evaluation machines. Configure this before importing
    robosuite / MuJoCo so simulation observations stay in-distribution.
    """
    mujoco_gl = os.environ.get('MUJOCO_GL')
    if mujoco_gl in {'osmesa', 'glfw', 'glx'}:
        return

    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    if os.environ.get('MUJOCO_GL') != 'egl':
        return
    if os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES'):
        return
    if not _has_nvidia_egl_library():
        return

    vendor_file = _ensure_nvidia_egl_vendor_file()
    if vendor_file is not None:
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = vendor_file


def check_robosuite_runtime(context='simulation'):
    """Validate robosuite only on code paths that actually need simulation."""
    configure_mujoco_egl_defaults()
    try:
        robosuite = import_module('robosuite')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'robosuite is required for {context}. Install it with '
            '`bash scripts/install_env.sh sim-only` or '
            '`bash scripts/install_env.sh full`.') from exc

    robosuite_version = digit_version(robosuite.__version__)
    if not (robosuite_version >= digit_version(ROBOSUITE_MINIMUM_VERSION)
            and robosuite_version < digit_version(ROBOSUITE_MAXIMUM_VERSION)):
        raise RuntimeError(
            f'Robosuite=={robosuite.__version__} is used but incompatible '
            f'for {context}. Please install robosuite>='
            f'{ROBOSUITE_MINIMUM_VERSION}, <{ROBOSUITE_MAXIMUM_VERSION}.')

    if not hasattr(robosuite, 'load_controller_config'):
        raise RuntimeError(
            'The installed robosuite is missing load_controller_config. '
            'Please install the patched robosuite from '
            'git+https://github.com/yinchimaoliang/robosuite.git@4099c09.')

    return robosuite


def get_libero_env(task, resolution=256, controller='OSC_POSE'):
    """Initializes a Libero environment for a given task.

    Args:
        task: The task object containing the problem folder and BDDL file.
        resolution (int): The resolution for the camera images.

    Returns:
        env: The initialized Libero environment.
        task_description (str): The language description of the task.
    """
    check_robosuite_runtime('LIBERO simulation evaluation')
    try:
        libero_module = import_module('libero.libero')
        libero_envs = import_module('libero.libero.envs')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'LIBERO is required for simulation evaluation. Install it with '
            '`bash scripts/install_env.sh sim-only` or '
            '`bash scripts/install_env.sh full`.') from exc

    task_description = task.language
    task_bddl_file = os.path.join(
        libero_module.get_libero_path('bddl_files'), task.problem_folder,
        task.bddl_file)
    env_args = {
        'bddl_file_name': task_bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
        'controller': controller,
    }
    env = libero_envs.OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state  # noqa: E501
    return env, task_description


def get_libero_image(obs, resize_size, img_key='agentview_image'):
    """Extracts image from observations and preprocesses it."""
    assert (resize_size is None or isinstance(resize_size, int)
            or isinstance(resize_size, tuple))
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[img_key]
    img = img[::-1, ::
              -1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    if resize_size is None:
        return img.copy()
    img = resize_image(img, resize_size)
    return img


def get_libero_dummy_action():
    """Returns a dummy action for the Libero environment.

    Returns:
        list: A dummy action consisting of zeros, which is suitable
            for the Libero environment.
    """
    return [0, 0, 0, 0, 0, 0, -1]


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55  # noqa: E501

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def crop_and_resize(image, crop_scale, batch_size):
    """Center-crop with TensorFlow semantics used by the LIBERO checkpoints."""
    try:
        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            pass

        image = tf.convert_to_tensor(np.asarray(image))
        if image.shape.ndims not in (3, 4):
            raise ValueError(
                f'Expected HWC or NHWC image, got shape {image.shape}')
        expanded_dims = False
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True
        if int(image.shape[0]) != batch_size:
            raise ValueError(
                f'Expected batch size {batch_size}, got {int(image.shape[0])}')

        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)

        new_heights = tf.reshape(
            tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size, ))
        new_widths = tf.reshape(
            tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size, ))
        height_offsets = (1 - new_heights) / 2
        width_offsets = (1 - new_widths) / 2
        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )

        image = tf.image.crop_and_resize(image, bounding_boxes,
                                         tf.range(batch_size), (224, 224))
        if expanded_dims:
            image = image[0]
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        return image.numpy()
    except ModuleNotFoundError:
        image = np.asarray(image)
        if image.ndim == 4:
            assert image.shape[0] == batch_size
        elif image.ndim == 3:
            assert batch_size == 1
        else:
            raise ValueError(
                f'Expected HWC or NHWC image, got shape {image.shape}')
        return crop_and_resize_numpy(image, crop_scale, output_size=(224, 224))


def get_vla_action(vla,
                   processor,
                   base_vla_name,
                   obs,
                   task_label,
                   unnorm_key,
                   device,
                   center_crop=False):
    """Predicts an action using the VLA model based on the provided
        observations and task label.

    Args:
        vla: The VLA model instance used for action prediction.
        processor: The processor used to prepare inputs for the VLA model.
        base_vla_name (str): The base name of the VLA model, used to determine
            the prompt format.
        obs (dict): Observations containing the full image and other
            relevant data.
        task_label (str): The label describing the task to be performed.
        unnorm_key (str): Key for unnormalizing actions.
        device: The device on which the model is
            running (e.g., 'cuda' or 'cpu').
        center_crop (bool): Whether to apply center cropping to the image.

    Returns:
        action: The predicted action from the VLA model.
    """
    image = Image.fromarray(obs['full_image'])
    image = image.convert('RGB')

    # (If trained with image augmentations) Center crop image and then
    # resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height
    # and width (post-crop), multiply
    # the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        image = Image.fromarray(crop_and_resize(np.array(image), 0.9, 1))
        image = image.convert('RGB')

    # Build VLA prompt
    if 'openvla-v01' in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f'{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:'  # noqa: E501,E231
        )
    else:  # OpenVLA
        prompt = f'In: What action should the robot take to {task_label.lower()}?\nOut:'  # noqa: E501,E231

    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def save_rollout_video(rollout_images,
                       idx,
                       success,
                       task_description,
                       work_dir,
                       log_file=None):
    """Saves a video of the rollout images to a file.

    Args:
        rollout_images (list): List of images representing the rollout.
        idx (int): Episode index for naming the video file.
        success (bool): Whether the task was successful.
        task_description (str): Description of the task,
            used in the filename.
        work_dir (str): Directory where the video will be saved.
        log_file (file object, optional): File to log the save path.
            Defaults to None.

    Returns:
        str: The path to the saved video file.
    """
    date = time.strftime('%Y_%m_%d')
    date_time = time.strftime('%Y_%m_%d-%H_%M_%S')
    rollout_dir = os.path.join(work_dir, 'rollouts', date)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(
        ' ', '_').replace('\n', '_').replace('.', '_')[:50]
    mp4_path = f'{rollout_dir}/{date_time}--episode={idx}--success={success}--task={processed_task_description}.mp4'  # noqa: E501
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path
