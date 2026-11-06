"""Check checkpointed campaign progression without starting Ray or GPUs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    directory = (
        Path(__file__).resolve().parents[2] / 'tools/rl/robotwin_conservative')
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('conservative_launch',
                                                  directory / 'launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = module.BASE = tmp_path
    module.WEIGHTS = tmp_path / 'sft'
    module.RESULTS = tmp_path / 'results'
    return module


def make_checkpoint(path):
    actor = path / 'actor'
    (actor / 'model_state_dict').mkdir(parents=True)
    (actor / 'model_state_dict/full_weights.pt').write_bytes(b'weights')
    shards = actor / 'dcp_checkpoint'
    shards.mkdir()
    (shards / '.metadata').write_bytes(b'metadata')
    for rank in range(8):
        (shards / f'__{rank}_0.distcp').write_bytes(b'shard')


def test_incomplete_checkpoint_is_rejected(launcher):
    checkpoint = launcher.checkpoint_path(1)
    make_checkpoint(checkpoint)
    launcher.validate_checkpoint(checkpoint)
    (checkpoint / 'actor/dcp_checkpoint/__7_0.distcp').unlink()
    with pytest.raises(RuntimeError, match='Incomplete'):
        launcher.validate_checkpoint(checkpoint)


def test_campaign_restores_last_successful_segment(launcher, monkeypatch):
    commands = []

    class StopAfterResume(Exception):
        pass

    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=123)

    def supervise(child, log, root):
        if len(commands) == 2:
            raise StopAfterResume
        make_checkpoint(launcher.checkpoint_path(1))
        return 0, None

    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    monkeypatch.setattr(launcher, 'supervise', supervise)
    monkeypatch.setattr(launcher.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(launcher.tempfile, 'mkdtemp',
                        lambda **kwargs: '/tmp/unused-test-ray')
    with pytest.raises(StopAfterResume):
        launcher.campaign()
    assert 'runner.max_steps=1' in commands[0]
    assert not any(x.startswith('runner.resume_dir=') for x in commands[0])
    assert 'runner.max_steps=5' in commands[1]
    assert (f'runner.resume_dir={launcher.checkpoint_path(1)}' in commands[1])
    ledger = json.loads((launcher.ROOT / 'checkpoints.json').read_text())
    assert [item['step'] for item in ledger] == [1]


def test_failed_segment_does_not_advance_checkpoint(launcher, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'Popen',
                        lambda *args, **kwargs: SimpleNamespace(pid=123))
    monkeypatch.setattr(launcher, 'supervise', lambda *args:
                        (0, 'host memory pressure'))
    monkeypatch.setattr(launcher.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(launcher.tempfile, 'mkdtemp',
                        lambda **kwargs: '/tmp/unused-test-ray')
    with pytest.raises(SystemExit):
        launcher.campaign()
    status = json.loads((launcher.ROOT / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert not (launcher.ROOT / 'checkpoints.json').exists()
