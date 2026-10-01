"""Opt-in physics-time demo recording, without splitting action chunks."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class _CountedScene:

    def __init__(self, scene, recorder):
        self._scene = scene
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self, *args, **kwargs):
        result = self._scene.step(*args, **kwargs)
        self._recorder.physics_steps += 1
        return result


class PhysicsVideoRecorder:
    """Stream original camera RGB at 25 Hz of simulated time, not wall time."""

    def __init__(self, task, folder, *, seed, fps=25, writer_factory=None):
        self.task = task
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.seed = int(seed)
        self.fps = int(fps)
        self.dt = float(task.scene.get_timestep())
        self.stride = round(1 / (self.fps * self.dt))
        if self.stride < 1 or not np.isclose(self.stride * self.dt * self.fps,
                                             1):
            raise ValueError('Video FPS must divide the physics step rate')
        if writer_factory is None:
            import imageio.v2 as imageio
            writer_factory = imageio.get_writer
        self.writers = {}
        try:
            for view in ('head', 'three_cameras'):
                self.writers[view] = writer_factory(
                    str(self.folder / f'{view}.mp4'),
                    fps=self.fps,
                    codec='libx264',
                    quality=8,
                    macro_block_size=2,
                    ffmpeg_params=['-threads', '1', '-movflags', '+faststart'])
        except BaseException:
            for writer in self.writers.values():
                writer.close()
            raise
        self.physics_steps = 0
        self.sample_steps = []
        self.closed = False

    def capture(self, *, force=False):
        if self.closed or (self.sample_steps
                           and self.sample_steps[-1] == self.physics_steps):
            return
        if not force and self.physics_steps % self.stride:
            return
        # Do not call get_obs() or overwrite the policy's now_obs.
        self.task.cameras.update_picture()
        rgb = self.task.cameras.get_rgb()
        views = [
            np.asarray(rgb[name]['rgb'], dtype=np.uint8)
            for name in ('head_camera', 'left_camera', 'right_camera')
        ]
        if len({view.shape for view in views}) != 1:
            raise ValueError('Demo expects equal-sized head and wrist cameras')
        self.writers['head'].append_data(views[0])
        self.writers['three_cameras'].append_data(
            np.concatenate(views, axis=1))
        self.sample_steps.append(self.physics_steps)

    @contextmanager
    def instrument(self):
        scene = self.task.scene
        render = self.task._update_render
        had_instance_render = '_update_render' in self.task.__dict__
        instance_render = self.task.__dict__.get('_update_render')

        def record_render(*args, **kwargs):
            result = render(*args, **kwargs)
            self.capture()
            return result

        self.task.scene = _CountedScene(scene, self)
        self.task._update_render = record_render
        try:
            yield
        finally:
            self.task.scene = scene
            if had_instance_render:
                self.task._update_render = instance_render
            else:
                del self.task._update_render

    def finish(self, *, success=None, complete=False):
        if self.closed:
            return
        try:
            # Include terminal poses that fall between regular samples.
            if getattr(self.task, 'cameras', None) is not None:
                self.task._update_render()
                self.capture(force=True)
        finally:
            for writer in self.writers.values():
                writer.close()
            self.closed = True
        manifest = {
            'seed':
            self.seed,
            'success':
            success,
            'complete':
            complete,
            'fps':
            self.fps,
            'physics_timestep':
            self.dt,
            'physics_steps':
            self.physics_steps,
            'frames':
            len(self.sample_steps),
            'sample_physics_steps':
            self.sample_steps,
            'simulation_seconds':
            self.physics_steps * self.dt,
            'video_seconds':
            len(self.sample_steps) / self.fps,
            'three_camera_order': ['head', 'left_wrist', 'right_wrist'],
            'note': ('Demo only; full unchanged chunks. '
                     'Final pose may add one off-grid frame.'),
        }
        with (self.folder / 'recording.json').open('x') as stream:
            json.dump(manifest, stream, indent=2)


def install_demo_runtime(folder):
    """Install after strict seeds, only in demo environment workers."""
    from robotwin.envs import vector_env as runtime

    if getattr(runtime.SubEnv, '_flux_demo_folder', None):
        if runtime.SubEnv._flux_demo_folder != str(folder):
            raise RuntimeError('Conflicting demo output directories')
        return

    class DemoSubEnv(runtime.SubEnv):
        _flux_demo_folder = str(folder)

        def reset(self, env_seed=None):
            self._finish_recording()
            super().reset(env_seed)
            self._demo_recorder = None
            self._demo_finished = False

        def _finish_recording(self):
            recorder = getattr(self, '_demo_recorder', None)
            if recorder is not None and not recorder.closed:
                recorder.finish(complete=False)

        def step(self, actions):
            if self.get_instruction() is None:
                self.reset()
            if getattr(self, '_demo_finished', False):
                return super().step(actions)
            if getattr(self, '_demo_recorder', None) is None:
                self._demo_recorder = PhysicsVideoRecorder(
                    self.task,
                    Path(folder) / f'seed_{self.env_seed}',
                    seed=self.env_seed)
            recorder = self._demo_recorder
            try:
                with recorder.instrument():
                    result = super().step(actions)
            except BaseException:
                recorder.finish(complete=False)
                raise
            success = bool(np.asarray(result['terminated']).any())
            if success or self.task.take_action_cnt >= self.task.step_lim:
                recorder.finish(success=success, complete=True)
                self._demo_finished = True
            return result

        def close(self, clear_cache=True):
            self._finish_recording()
            return super().close(clear_cache=clear_cache)

    runtime.SubEnv = DemoSubEnv
