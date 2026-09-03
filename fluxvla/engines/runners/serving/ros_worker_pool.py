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
"""Episode-affine process workers for multi-GPU ROS inference.

Each worker owns one complete FluxVLA policy on one device.  Requests from an
episode stay on that worker until FluxThemis acknowledges ``episode_end``.
This is deliberately replica-based serving, not tensor/model parallelism:
stateful dataset and model histories must never be interleaved across active
episodes.
"""
from __future__ import annotations
import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait as wait_connections
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class _EpisodeLease:
    worker_index: int
    seed: int


@dataclass(frozen=True)
class _StartingPolicyWorker:
    worker_index: int
    device: str
    process: Any
    connection: Connection
    deadline: float


class EpisodeAffinityPolicyPool:
    """Dispatch complete episodes across independent policy backends."""

    supports_episode_affinity = True

    def __init__(self,
                 backends: Sequence[Any],
                 lease_timeout_s: float = 900.0) -> None:
        if isinstance(backends,
                      (str, bytes)) or not isinstance(backends, Sequence):
            raise TypeError('backends must be a sequence')
        if not backends:
            raise ValueError('backends cannot be empty')
        if isinstance(lease_timeout_s,
                      bool) or not isinstance(lease_timeout_s, (int, float)):
            raise TypeError('lease_timeout_s must be a number')
        if lease_timeout_s <= 0:
            raise ValueError('lease_timeout_s must be positive')
        for backend in backends:
            if not callable(getattr(backend, 'predict', None)):
                raise TypeError('every backend must define predict()')

        self._backends = tuple(backends)
        self.lease_timeout_s = float(lease_timeout_s)
        self.worker_devices = tuple(
            str(getattr(backend, 'device', index))
            for index, backend in enumerate(self._backends))
        self._condition = threading.Condition(threading.RLock())
        self._leases: dict[str, _EpisodeLease] = {}
        self._worker_episodes: list[str
                                    | None] = [None for _ in self._backends]
        self._next_worker = 0
        self._closed = False
        self._fatal_error: RuntimeError | None = None

    @property
    def worker_count(self) -> int:
        return len(self._backends)

    @property
    def active_episode_ids(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._leases)

    def predict(self,
                observation: Any,
                unnorm_key: str,
                seed: int,
                *,
                episode_id: str,
                reset: bool = False) -> tuple[np.ndarray, float]:
        """Lease a backend and run one prediction without changing affinity."""
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_id must be a non-empty string')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('seed must be an integer')
        if not isinstance(reset, bool):
            raise TypeError('reset must be a bool')

        lease, first_request = self._acquire_lease(episode_id, seed)
        worker_observation = dict(observation)
        worker_observation['is_new_episode'] = bool(first_request or reset)
        backend = self._backends[lease.worker_index]
        try:
            return backend.predict(
                worker_observation,
                unnorm_key=unnorm_key,
                seed=seed,
            )
        except Exception as exc:
            worker_device = self.worker_devices[lease.worker_index]
            failure = RuntimeError(
                'FluxVLA inference worker '
                f'{lease.worker_index} ({worker_device}) '
                f'failed during episode {episode_id!r}: {exc}')
            with self._condition:
                self._fatal_error = failure
                self._condition.notify_all()
            raise failure from exc

    def release_episode(self, episode_id: str) -> bool:
        """Release one episode after its durable ``episode_end`` report."""
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_id must be a non-empty string')
        with self._condition:
            lease = self._leases.pop(episode_id, None)
            if lease is None:
                return False
            if self._worker_episodes[lease.worker_index] == episode_id:
                self._worker_episodes[lease.worker_index] = None
            self._condition.notify_all()
            return True

    def release_all(self) -> None:
        """Clear every affinity lease after a terminal run event."""
        with self._condition:
            self._leases.clear()
            self._worker_episodes = [None for _ in self._backends]
            self._condition.notify_all()

    def close(self) -> None:
        """Stop all backends.  Safe to call repeatedly."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        errors = []
        for backend in self._backends:
            close = getattr(backend, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover - cleanup guard
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                'Failed to close one or more FluxVLA inference workers: ' +
                '; '.join(str(error) for error in errors))

    def _acquire_lease(self, episode_id: str,
                       seed: int) -> tuple[_EpisodeLease, bool]:
        deadline = time.monotonic() + self.lease_timeout_s
        with self._condition:
            while True:
                self._raise_if_unavailable()
                existing = self._leases.get(episode_id)
                if existing is not None:
                    if existing.seed != seed:
                        raise ValueError(
                            f'Episode {episode_id!r} changed seed from '
                            f'{existing.seed} to {seed}')
                    return existing, False

                worker_index = self._find_free_worker()
                if worker_index is not None:
                    lease = _EpisodeLease(worker_index=worker_index, seed=seed)
                    self._leases[episode_id] = lease
                    self._worker_episodes[worker_index] = episode_id
                    self._next_worker = (worker_index + 1) % self.worker_count
                    return lease, True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        'Timed out waiting for a free inference worker; '
                        f'{self.worker_count} workers are serving episodes '
                        f'{sorted(self._leases)}')
                self._condition.wait(remaining)

    def _find_free_worker(self) -> int | None:
        for offset in range(self.worker_count):
            index = (self._next_worker + offset) % self.worker_count
            if self._worker_episodes[index] is None:
                return index
        return None

    def _raise_if_unavailable(self) -> None:
        if self._closed:
            raise RuntimeError('FluxVLA inference worker pool is closed')
        if self._fatal_error is not None:
            raise self._fatal_error


class _ProcessPolicyBackend:
    """Synchronous IPC proxy for one spawned policy process."""

    def __init__(self, process: Any, connection: Connection, device: str,
                 request_timeout_s: float) -> None:
        self.process = process
        self.connection = connection
        self.device = device
        self.request_timeout_s = request_timeout_s
        self._lock = threading.RLock()
        self._request_sequence = 0
        self._closed = False

    def predict(self, observation: Any, unnorm_key: str,
                seed: int) -> tuple[np.ndarray, float]:
        with self._lock:
            self._ensure_alive()
            self._request_sequence += 1
            request_id = self._request_sequence
            try:
                self.connection.send({
                    'op': 'predict',
                    'request_id': request_id,
                    'observation': observation,
                    'unnorm_key': unnorm_key,
                    'seed': seed,
                })
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(
                    f'Inference worker on {self.device} disconnected') from exc
            if not self.connection.poll(self.request_timeout_s):
                raise TimeoutError(
                    f'Inference worker on {self.device} exceeded '
                    f'{self.request_timeout_s:g}s request timeout')
            try:
                response = self.connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError(
                    f'Inference worker on {self.device} disconnected') from exc
            if response.get('request_id') != request_id:
                raise RuntimeError(
                    f'Inference worker on {self.device} returned an '
                    'out-of-order response')
            if response.get('op') == 'error':
                raise RuntimeError(
                    f"{response.get('error', 'unknown worker error')}\n"
                    f"{response.get('traceback', '')}".rstrip())
            if response.get('op') != 'result':
                raise RuntimeError(
                    f'Inference worker on {self.device} returned an invalid '
                    f"message {response.get('op')!r}")
            actions = np.asarray(response['actions'], dtype=np.float32)
            return actions, float(response['inference_time_s'])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.process.is_alive():
                try:
                    self.connection.send({'op': 'shutdown'})
                    if self.connection.poll(2.0):
                        self.connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self.connection.close()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)

    def _ensure_alive(self) -> None:
        if self._closed:
            raise RuntimeError(f'Inference worker on {self.device} is closed')
        if not self.process.is_alive():
            raise RuntimeError(
                f'Inference worker on {self.device} exited with code '
                f'{self.process.exitcode}')


def spawn_ros_policy_pool(
        cfg: Any,
        ckpt_path: str,
        devices: Sequence[str],
        service_name: str | None = None,
        startup_timeout_s: float = 900.0,
        startup_parallelism: int = 1,
        request_timeout_s: float = 120.0,
        lease_timeout_s: float = 900.0) -> EpisodeAffinityPolicyPool:
    """Spawn one fully initialized FluxVLA policy process per device."""
    normalized_devices = tuple(str(device) for device in devices)
    if len(normalized_devices) < 2:
        raise ValueError('A process worker pool requires at least two devices')
    for name, value in {
            'startup_timeout_s': startup_timeout_s,
            'request_timeout_s': request_timeout_s,
            'lease_timeout_s': lease_timeout_s,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'{name} must be a number')
        if value <= 0:
            raise ValueError(f'{name} must be positive')
    if isinstance(startup_parallelism,
                  bool) or not isinstance(startup_parallelism, int):
        raise TypeError('startup_parallelism must be an integer')
    if startup_parallelism < 1:
        raise ValueError('startup_parallelism must be positive')

    context = mp.get_context('spawn')
    worker_count = len(normalized_devices)
    startup_parallelism = min(startup_parallelism, worker_count)
    backends: list[_ProcessPolicyBackend | None] = [None] * worker_count
    starting: dict[Connection, _StartingPolicyWorker] = {}
    next_worker_index = 0

    def start_available_workers() -> None:
        nonlocal next_worker_index
        while (next_worker_index < worker_count
               and len(starting) < startup_parallelism):
            worker_index = next_worker_index
            device = normalized_devices[worker_index]
            print(
                '[FluxVLA] Loading inference replica '
                f'{worker_index + 1}/{worker_count} on {device}.',
                flush=True,
            )
            parent_connection, child_connection = context.Pipe(duplex=True)
            process = context.Process(
                target=_ros_policy_worker_main,
                args=(
                    child_connection,
                    cfg,
                    str(ckpt_path),
                    device,
                    service_name,
                    worker_index,
                ),
                name=f'fluxvla-ros-worker-{worker_index}',
            )
            try:
                process.start()
            except BaseException:
                parent_connection.close()
                child_connection.close()
                raise
            child_connection.close()
            starting[parent_connection] = _StartingPolicyWorker(
                worker_index=worker_index,
                device=device,
                process=process,
                connection=parent_connection,
                deadline=time.monotonic() + float(startup_timeout_s),
            )
            next_worker_index += 1

    try:
        start_available_workers()
        while starting:
            now = time.monotonic()
            next_deadline = min(worker.deadline
                                for worker in starting.values())
            timeout_s = max(0.0, next_deadline - now)
            ready_connections = wait_connections(
                tuple(starting), timeout=timeout_s)
            if not ready_connections:
                timed_out = min(
                    starting.values(), key=lambda worker: worker.deadline)
                raise TimeoutError(
                    f'Worker {timed_out.worker_index} on '
                    f'{timed_out.device} did not become ready within '
                    f'{startup_timeout_s:g}s')

            for parent_connection in ready_connections:
                worker = starting[parent_connection]
                worker_index = worker.worker_index
                device = worker.device
                process = worker.process
                try:
                    ready = parent_connection.recv()
                except (EOFError, OSError) as exc:
                    process.join(timeout=1.0)
                    raise RuntimeError(
                        f'FluxVLA worker {worker_index} on {device} exited '
                        'during startup') from exc
                if ready.get('op') != 'ready':
                    process.join(timeout=1.0)
                    raise RuntimeError(
                        f'FluxVLA worker {worker_index} on {device} failed '
                        f"to start: {ready.get('error', 'unknown error')}\n"
                        f"{ready.get('traceback', '')}".rstrip())
                if (ready.get('worker_index') != worker_index
                        or ready.get('device') != device):
                    raise RuntimeError(
                        f'FluxVLA worker {worker_index} on {device} returned '
                        'mismatched startup identity')

                backends[worker_index] = _ProcessPolicyBackend(
                    process=process,
                    connection=parent_connection,
                    device=device,
                    request_timeout_s=float(request_timeout_s),
                )
                del starting[parent_connection]
                print(
                    '[FluxVLA] Inference replica '
                    f'{worker_index + 1}/{worker_count} ready on {device}.',
                    flush=True,
                )

            start_available_workers()

        ready_backends = [
            backend for backend in backends if backend is not None
        ]
        if len(ready_backends) != worker_count:
            raise RuntimeError('Not all FluxVLA inference workers started')
        return EpisodeAffinityPolicyPool(
            backends=ready_backends,
            lease_timeout_s=float(lease_timeout_s),
        )
    except BaseException:
        for worker in starting.values():
            try:
                worker.connection.close()
            except OSError:
                pass
            try:
                if worker.process.is_alive():
                    worker.process.terminate()
                if worker.process.pid is not None:
                    worker.process.join(timeout=5.0)
            except (AssertionError, OSError):
                pass
        for backend in backends:
            if backend is None:
                continue
            try:
                backend.close()
            except Exception:
                pass
        raise


def _ros_policy_worker_main(connection: Connection, cfg: Any, ckpt_path: str,
                            device: str, service_name: str | None,
                            worker_index: int) -> None:
    """Child entry point.  Import model-building code only after spawn."""
    policy = None
    try:
        import torch

        worker_device = torch.device(device)
        if worker_device.type == 'cuda':
            # Select the replica device before any model/backbone constructor
            # can allocate CUDA state.
            torch.cuda.set_device(worker_device)
        from .ros_server import build_ros_policy_from_config

        policy = build_ros_policy_from_config(
            cfg,
            ckpt_path=ckpt_path,
            device=device,
            service_name=service_name,
        )
        connection.send({
            'op': 'ready',
            'worker_index': worker_index,
            'device': device,
        })
        while True:
            message = connection.recv()
            operation = message.get('op')
            if operation == 'shutdown':
                connection.send({'op': 'shutdown_ack'})
                break
            if operation != 'predict':
                connection.send({
                    'op':
                    'error',
                    'request_id':
                    message.get('request_id'),
                    'error':
                    f'Unsupported worker operation {operation!r}',
                })
                continue
            request_id = message.get('request_id')
            try:
                actions, inference_time_s = policy.predict(
                    message['observation'],
                    unnorm_key=message['unnorm_key'],
                    seed=message['seed'],
                )
                connection.send({
                    'op': 'result',
                    'request_id': request_id,
                    'actions': actions,
                    'inference_time_s': inference_time_s,
                })
            except BaseException as exc:
                connection.send({
                    'op': 'error',
                    'request_id': request_id,
                    'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(),
                })
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as exc:
        try:
            connection.send({
                'op': 'startup_error' if policy is None else 'fatal_error',
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        close = getattr(policy, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        connection.close()
