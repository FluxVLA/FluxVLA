import json

import numpy as np
import pytest

from fluxvla.rl.benchmarks.robotwin.video import PhysicsVideoRecorder


class Scene:

    def __init__(self):
        self.steps = 0

    def get_timestep(self):
        return 1 / 250

    def step(self):
        self.steps += 1


class Cameras:

    def update_picture(self):
        pass

    def get_rgb(self):
        return {
            name: {
                'rgb': np.zeros((8, 8, 3), dtype=np.uint8)
            }
            for name in ('head_camera', 'left_camera', 'right_camera')
        }


class Task:

    def __init__(self):
        self.scene = Scene()
        self.cameras = Cameras()
        self.now_obs = object()
        self.render_calls = 0

    def _update_render(self):
        self.render_calls += 1


class Writer:

    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


def test_recording_preserves_physics_observation_and_render(tmp_path):
    task = Task()
    original_scene, original_obs = task.scene, task.now_obs
    writers = []

    def factory(*args, **kwargs):
        writer = Writer()
        writers.append(writer)
        return writer

    recorder = PhysicsVideoRecorder(
        task, tmp_path / 'demo', seed=7, writer_factory=factory)
    for _ in range(2):
        with recorder.instrument():
            task._update_render()
            for _ in range(53):
                task.scene.step()
                task._update_render()
        assert task.scene is original_scene
        assert '_update_render' not in task.__dict__
    assert task.scene.steps == 106
    assert task.now_obs is original_obs
    assert task.render_calls == 108
    recorder.finish(success=True, complete=True)
    recorder.finish()
    assert recorder.sample_steps == [*range(0, 101, 10), 106]
    assert all(
        writer.closed and len(writer.frames) == 12 for writer in writers)
    assert writers[1].frames[0].shape == (8, 24, 3)
    metadata = json.loads((tmp_path / 'demo/recording.json').read_text())
    assert metadata['success'] and metadata['complete']
    assert metadata['physics_steps'] == 106


def test_recording_restores_hooks_after_failure(tmp_path):
    task = Task()
    scene = task.scene
    recorder = PhysicsVideoRecorder(
        task,
        tmp_path / 'demo',
        seed=1,
        writer_factory=lambda *a, **k: Writer())
    with pytest.raises(RuntimeError), recorder.instrument():
        task.scene.step()
        raise RuntimeError('simulation failure')
    assert task.scene is scene
    assert '_update_render' not in task.__dict__
    recorder.finish(complete=False)
    assert not json.loads(
        (tmp_path / 'demo/recording.json').read_text())['complete']
