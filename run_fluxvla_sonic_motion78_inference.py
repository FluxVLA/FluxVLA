"""
Run local FluxVLA Sonic motion78 inference for gear_sonic latent WBC control.

This is the local-checkpoint equivalent of starVLA's WebSocket inference
runner.  It keeps the same sensor input and ZMQ latent-action output flow, but
loads a FluxVLA checkpoint directly from:

    /home/limx/Erwin/FluxVLA/work_dirs/gr00t_sonic_motion78_full

Action layout:
    [0:64]   motion_token
    [64:71]  left_hand_joints
    [71:78]  right_hand_joints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

ERWIN_ROOT = Path("/home/limx/Erwin")
for _repo_root in (
    ERWIN_ROOT / "FluxVLA",
    ERWIN_ROOT / "GR00T-WholeBodyControl",
):
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import cv2 as cv
import numpy as np
import torch
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


DEFAULT_FLUXVLA_ROOT = Path("/home/limx/Erwin/FluxVLA")
DEFAULT_CKPT_DIR = DEFAULT_FLUXVLA_ROOT / "work_dirs/gr00t_sonic_motion78_full_1"
DEFAULT_CONFIG = DEFAULT_FLUXVLA_ROOT / "configs/gr00t/gr00t_eagle_3b_sonic_motion78.py"


@dataclass
class InferenceConfig:
    ckpt_dir: str = str(DEFAULT_CKPT_DIR)
    """FluxVLA work_dir containing checkpoints/, tokenizer/, dataset_statistics.json."""

    ckpt_path: str = ""
    """Optional explicit .safetensors/.pth checkpoint path. If empty, auto-select from ckpt_dir/checkpoints."""

    config: str = str(DEFAULT_CONFIG)
    """FluxVLA mmengine config used for this checkpoint."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 0
    """Deprecated compatibility flag. Actual action chunk size is read from the FluxVLA config."""

    rate: float = 4
    """Rate at which we run the VLA forward pass (Hz)."""

    camera_host: str = "localhost"
    camera_port: int = 5555
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    prompt: str = "pick up the cylinder and throw it into the trash bin"
    image_size: tuple[int, int] = (224, 224)
    mixed_precision_dtype: str = "bf16"
    strict_load: bool = True
    verbose_timing: bool = False
    log_action_stats: bool = False
    latency_compensation: bool = False
    max_motion_token_abs: float = 1.25


def print_green(x: str):
    print(f"\033[92m{x}\033[0m")


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _sleep_remaining(t_start: float, loop_period: float):
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    checkpoint_dir = ckpt_dir / "checkpoints"
    candidates = sorted(
        list(checkpoint_dir.glob("*.safetensors"))
        + list(checkpoint_dir.glob("*.pth"))
        + list(checkpoint_dir.glob("*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_train_dataset_cfg(mmcfg: Any) -> dict[str, Any]:
    train_ds = mmcfg.train_dataloader["dataset"]
    while isinstance(train_ds, dict):
        if "transforms" in train_ds:
            return train_ds
        if "dataset" in train_ds:
            train_ds = train_ds["dataset"]
            continue
        if "datasets" in train_ds:
            train_ds = train_ds["datasets"]
            continue
        break
    return train_ds


def _extract_norm_type(mmcfg: Any) -> str:
    train_ds = _get_train_dataset_cfg(mmcfg)
    transforms = train_ds.get("transforms", [])
    for transform in transforms:
        if transform.get("type") == "NormalizeStatesAndActions":
            return transform.get("norm_type", "mean_std")
    return "min_max"


def _extract_embodiment_id(mmcfg: Any) -> int | None:
    train_ds = _get_train_dataset_cfg(mmcfg)
    transforms = train_ds.get("transforms", [])
    for transform in transforms:
        if transform.get("type") == "ProcessParquetInputs":
            return transform.get("embodiment_id", None)
    return 0


class FluxVLASonicMotion78Policy:
    """Local FluxVLA adapter matching the starVLA client-side policy API."""

    ACTION_DIM = 78
    MOTION_DIM = 64
    HAND_DIM = 7

    def __init__(
        self,
        ckpt_dir: str,
        ckpt_path: str,
        config_path: str,
        image_size: tuple[int, int],
        mixed_precision_dtype: str,
        strict_load: bool,
    ):
        from mmengine.config import Config
        from fluxvla.engines import (
            build_dataset_from_cfg,
            build_transform_from_cfg,
            build_vla_from_cfg,
        )

        self.ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        self.ckpt_path = (
            Path(ckpt_path).expanduser().resolve()
            if ckpt_path
            else _find_latest_checkpoint(self.ckpt_dir)
        )
        self.config_path = Path(config_path).expanduser().resolve()
        self.image_size = tuple(image_size)
        self.mixed_precision_dtype = mixed_precision_dtype
        self.device_type = "cuda"

        if not torch.cuda.is_available():
            raise RuntimeError("FluxVLA PrivateInferenceDataset uses .cuda(); CUDA is required.")

        self.mmcfg = Config.fromfile(str(self.config_path))
        self.norm_type = _extract_norm_type(self.mmcfg)
        self.embodiment_id = _extract_embodiment_id(self.mmcfg)

        head_cfg = self.mmcfg.inference_model["vla_head"]
        self.action_chunk_size = int(head_cfg.get("traj_length", 1))
        self.state_dim = int(head_cfg.get("state_dim", 64))
        self.padded_action_dim = int(head_cfg.get("action_dim", 128))
        self.ori_action_dim = int(head_cfg.get("ori_action_dim", self.ACTION_DIM))
        if self.ori_action_dim != self.ACTION_DIM:
            raise ValueError(
                f"Expected ori_action_dim={self.ACTION_DIM} for sonic_motion78, "
                f"got {self.ori_action_dim}"
            )

        stats_path = self.ckpt_dir / "dataset_statistics.json"
        raw_stats = _load_json(stats_path)
        statistic_name = self.mmcfg.train_dataloader["dataset"].get(
            "statistic_name", "sonic_motion78"
        )
        if statistic_name not in raw_stats:
            fallback = next(iter(raw_stats))
            print(
                f"[Warning] statistic_name {statistic_name!r} not in {stats_path}; "
                f"falling back to {fallback!r}"
            )
            statistic_name = fallback
        self.statistic_name = statistic_name
        self.private_stats = {"private": raw_stats[statistic_name]}

        self.denormalize_action = build_transform_from_cfg(
            dict(
                type="DenormalizePrivateAction",
                norm_stats=self.private_stats,
                action_dim=self.ori_action_dim,
                norm_type=self.norm_type,
                denorm_action=True,
            )
        )
        self.dataset = build_dataset_from_cfg(self._build_eval_dataset_cfg())

        self.vla = build_vla_from_cfg(self.mmcfg.inference_model)
        state_dict = self._load_state_dict(self.ckpt_path)
        self.vla.load_state_dict(state_dict, strict=strict_load)
        self.vla.eval().cuda()

        print_green(f"Loaded FluxVLA checkpoint: {self.ckpt_path}")
        print_green(
            "FluxVLA dims: "
            f"state_dim={self.state_dim}, action_dim={self.padded_action_dim}, "
            f"ori_action_dim={self.ori_action_dim}, traj_length={self.action_chunk_size}, "
            f"norm_type={self.norm_type}, statistic_name={self.statistic_name}"
        )

    def _load_state_dict(self, ckpt_path: Path) -> dict[str, torch.Tensor]:
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(ckpt_path), device="cpu")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint

    def _build_eval_dataset_cfg(self) -> dict[str, Any]:
        tokenizer_cfg = dict(
            type="PretrainedTokenizer",
            model_path="fluxvla/models/third_party_models/eagle2_hg_model",
        )
        return dict(
            type="PrivateInferenceDataset",
            norm_stats=self.private_stats,
            model_path=str(self.ckpt_dir),
            img_keys=["ego_view"],
            embodiment_id=self.embodiment_id,
            max_len=600,
            transforms=[
                dict(
                    type="NormalizeStatesAndActions",
                    state_key="proprio",
                    action_key="action",
                    state_dim=self.state_dim,
                    action_dim=self.padded_action_dim,
                    norm_type=self.norm_type,
                ),
                dict(
                    type="ProcessPromptsWithImage",
                    max_len=600,
                    num_images=1,
                    tokenizer=tokenizer_cfg,
                ),
                dict(type="ResizeImages", height=self.image_size[1], width=self.image_size[0]),
                dict(
                    type="NormalizeImages",
                    means=[[123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062]],
                ),
            ],
        )

    def _autocast_context(self):
        if self.mixed_precision_dtype.lower() in ("bf16", "bfloat16"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self.mixed_precision_dtype.lower() in ("fp16", "float16", "half"):
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cuda", enabled=False)

    def predict_action(self, image: np.ndarray, state: np.ndarray, language_prompt: str) -> dict:
        resized = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)
        batch = self.dataset(
            {
                "ego_view": resized,
                "qpos": np.asarray(state, dtype=np.float32),
                "task_description": language_prompt,
            }
        )
        with torch.inference_mode(), self._autocast_context():
            raw_action = self.vla.predict_action(**batch)

        raw_np = raw_action.detach().float().cpu().numpy()
        actions = np.asarray(self.denormalize_action(dict(action=raw_np)), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] != self.ACTION_DIM:
            raise RuntimeError(f"Expected denormalized action dim 78, got {actions.shape}")

        return {
            "motion_token": actions[:, : self.MOTION_DIM],
            "left_hand_joints": actions[:, self.MOTION_DIM : self.MOTION_DIM + self.HAND_DIM],
            "right_hand_joints": actions[:, self.MOTION_DIM + self.HAND_DIM : self.ACTION_DIM],
        }


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    image = camera_msg["images"]["ego_view"]

    left_hand_q = np.asarray(state_msg["left_hand_q"], dtype=np.float32).copy()
    right_hand_q = np.asarray(state_msg["right_hand_q"], dtype=np.float32).copy()
    body_q = np.asarray(state_msg["body_q"], dtype=np.float32)

    # Copy index finger data to middle finger (hardware coupling).
    left_hand_q[5] = left_hand_q[3]
    left_hand_q[6] = left_hand_q[4]

    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat).astype(np.float32)

    whole_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_hand_q,
        right_hand_actuated_joint_values=right_hand_q,
    ).astype(np.float32)

    state = np.concatenate([whole_q, projected_gravity], axis=0)
    assert state.shape[0] == 46, f"Expected state dim 46, got {state.shape[0]}"

    return {
        "image": image,
        "state": state,
        "language_prompt": language_prompt,
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }


def _format_range(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    return (
        f"{name}: shape={value.shape}, "
        f"min={value.min():.4f}, max={value.max():.4f}, "
        f"first={np.array2string(value[0], precision=3, suppress_small=True)}, "
        f"last={np.array2string(value[-1], precision=3, suppress_small=True)}"
    )


def _format_action_samples(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return f"{name}[0]={np.array2string(value, precision=3, suppress_small=True)}"
    indices = sorted(set([0, value.shape[0] // 2, value.shape[0] - 1]))
    parts = [
        f"{name}[{idx}]={np.array2string(value[idx], precision=3, suppress_small=True)}"
        for idx in indices
    ]
    return "; ".join(parts)


def run_policy_inference_and_process(
    policy: FluxVLASonicMotion78Policy,
    observation: dict,
    log_action_stats: bool = False,
    max_motion_token_abs: float = 1.25,
):
    try:
        processed_action = policy.predict_action(
            image=observation["image"],
            state=observation["state"],
            language_prompt=observation["language_prompt"],
        )
        motion_abs_max = float(np.abs(processed_action["motion_token"]).max())
        if motion_abs_max > max_motion_token_abs:
            print(
                f"[Warning] motion_token max ({motion_abs_max:.4f}) > "
                f"{max_motion_token_abs:.4f}. Exceeds action bound, skipping."
            )
            return None
        if log_action_stats:
            print_green(_format_range("motion_token", processed_action["motion_token"]))
            print_green(_format_range("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_range("right_hand", processed_action["right_hand_joints"]))
            print_green(_format_action_samples("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_action_samples("right_hand", processed_action["right_hand_joints"]))
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    continue
                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)
                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


def main(config: InferenceConfig):
    pause_loop = True
    robot_model = get_g1_robot_model(waist_location="lower_and_upper_body")
    policy = FluxVLASonicMotion78Policy(
        ckpt_dir=config.ckpt_dir,
        ckpt_path=config.ckpt_path,
        config_path=config.config,
        image_size=config.image_size,
        mixed_precision_dtype=config.mixed_precision_dtype,
        strict_load=config.strict_load,
    )
    print_green(f"Action chunk size from FluxVLA config: {policy.action_chunk_size}")

    state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    camera_subscriber = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)
    telemetry = Telemetry(window_size=100)

    loop_period = 1.0 / config.action_publish_rate
    cpp_loop_running = False
    cpp_mode = "OFF"
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate
    zmq_frame_counter = 0
    language_prompt_ref: list[str] = [config.prompt]
    prompt_prefix = "prompt:"

    def publish_initial_pose():
        left_hand = _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        right_hand = _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)

    def send_cpp_control_command(start: bool, planner: bool = False):
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            cpp_loop_running = start
            cpp_mode = "PLANNER" if (start and planner) else ("POSE" if start else "OFF")
            return True
        except Exception as e:
            print(f"Warning: Failed to send control command: {e}")
            return False

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time, zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(prompt_prefix):
            new_prompt = key[len(prompt_prefix):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            return

        if key == "i":
            zmq_frame_counter = 0
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            last_inference_time = 0.0
            if cpp_loop_running and cpp_mode == "PLANNER":
                send_cpp_control_command(start=True, planner=False)
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
        elif key == "k":
            if cpp_loop_running:
                send_cpp_control_command(start=False, planner=(cpp_mode == "PLANNER"))
            else:
                send_cpp_control_command(start=True, planner=True)
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print_green(f"left initial hand closed: {initial_pose_left_hand_closed}")
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print_green(f"right initial hand closed: {initial_pose_right_hand_closed}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=policy,
                observation=obs,
                log_action_stats=config.log_action_stats,
                max_motion_token_abs=config.max_motion_token_abs,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = (
                    calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, policy.action_chunk_size
                    )
                    if config.latency_compensation
                    else 0
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                )
            except queue.Empty:
                pass

            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=inference_busy_event.is_set(),
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                time.sleep(0.2)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    _sleep_remaining(t_start, loop_period)
                    continue

                motion_token = np.asarray(cached_action_chunk["motion_token"], dtype=np.float32)
                left_hand_joints = np.asarray(cached_action_chunk["left_hand_joints"], dtype=np.float32)
                right_hand_joints = np.asarray(cached_action_chunk["right_hand_joints"], dtype=np.float32)

                horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                current_idx = min(action_chunk_index, horizon - 1)

                if motion_token.ndim == 2:
                    motion_token = motion_token[current_idx]
                if left_hand_joints.ndim == 2:
                    left_hand_joints = left_hand_joints[current_idx]
                if right_hand_joints.ndim == 2:
                    right_hand_joints = right_hand_joints[current_idx]

                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1

                zmq_message = pack_latent_action_message(
                    motion_token,
                    frame_index,
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                )
                zmq_socket.send(zmq_message)
                action_chunk_index = min(action_chunk_index + 1, policy.action_chunk_size - 1)

            if config.verbose_timing and (time.monotonic() - t_start) > 0:
                telemetry.log_timing_info(context="FluxVLA Sonic motion78 Inference Loop", threshold=0.0)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("FluxVLA sonic_motion78 inference loop terminated by user")
    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
