"""PR #161 portability, shared-input and worker contract regressions."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/lingbot_vla_v2'


def _load_test_helpers(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# pytest uses importlib mode and this directory is not a Python package.
_helpers = _load_test_helpers(
    'lingbot_online_test_helpers',
    Path(__file__).with_name('test_lingbot_online_dataset.py'))
_FakeProcessor, _FakeRawDataset, norm_stats = (_helpers._FakeProcessor,
                                               _helpers._FakeRawDataset,
                                               _helpers.norm_stats)
_TransformedDataset = _helpers._TransformedDataset


def test_portable_config_paths_from_other_cwd(monkeypatch, tmp_path):
    inputs = tmp_path / 'read only inputs with spaces'
    inputs.mkdir()
    for name in ('CHECKPOINT', 'TOKENIZER_PATH', 'NATIVE_CONFIG', 'MANIFEST',
                 'DATA_PATH'):
        monkeypatch.setenv(f'LINGBOT_{name}', str(inputs / name))
    monkeypatch.setenv('WORK_DIR', str(tmp_path / 'separate output'))
    monkeypatch.setenv('LINGBOT_NUM_WORKERS', '2')
    monkeypatch.chdir(tmp_path)
    replay = Config.fromfile(
        str(CONFIGS / 'lingbot_vla_v2_gb64_4gpu_parity.py'))
    online = Config.fromfile(
        str(CONFIGS / 'lingbot_vla_v2_online_flow_only_100.py'))
    assert replay.train_dataloader.dataset.manifest_path == str(inputs /
                                                                'MANIFEST')
    assert replay.runner.metric.run_dir == str(tmp_path / 'separate output')
    assert replay.train_dataloader.per_device_num_workers == 0
    assert online.train_dataloader.per_device_num_workers == 2
    assert online.train_dataloader.dataset.data_path == str(inputs /
                                                            'DATA_PATH')
    assert online.model.native_config_path == str(inputs / 'NATIVE_CONFIG')
    assert Path(online.train_dataloader.dataset.norm_stats_path).is_file()
    for path in CONFIGS.glob('*.py'):
        assert '/ceph/' not in path.read_text()


def test_relocated_norm_stats_are_byte_identical():
    stats = CONFIGS / 'assets/robotwin_norm_stats.json'
    assert hashlib.sha256(stats.read_bytes()).hexdigest() == (
        '5fc926443cebbd6484194982a768c4bd07772aa176dfbc64c6c6b59784855d76')
    assert not (
        ROOT /
        'fluxvla/datasets/lingbot_vla_v2/assets/robotwin_norm_stats.json'
    ).exists()


def test_missing_external_inputs_fail_explicitly(tmp_path):
    from fluxvla.datasets.lingbot_teacher_cache import \
        LingBotTeacherCacheDataset
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    with pytest.raises(ValueError, match='LINGBOT_MANIFEST'):
        LingBotTeacherCacheDataset('')
    with pytest.raises(FileNotFoundError, match='manifest not found'):
        LingBotTeacherCacheDataset(str(tmp_path / 'missing.json'))
    with pytest.raises(ValueError, match='LINGBOT_NATIVE_CONFIG'):
        LingBotVLAV2(native_config_path='')
    with pytest.raises(ValueError, match='LINGBOT_CHECKPOINT'):
        LingBotVLAV2(pretrained_name_or_path='')


def test_shared_transform_is_registered_and_exact(norm_stats):
    from fluxvla.engines import TRANSFORMS
    from fluxvla.transforms.lingbot_online import LingBotSampleTransform

    kwargs = dict(
        norm_stats_path=str(norm_stats),
        tokenizer_path='unused',
        action_horizon=2,
        tokenizer_max_length=6,
        processor=_FakeProcessor(),
        model_dtype=torch.float32,
        return_teacher_images=True,
        flow_rng_seed=42)
    online = _TransformedDataset(_FakeRawDataset(3), **kwargs)
    transform = LingBotSampleTransform(**kwargs)
    for index in range(3):
        actual = transform({**_FakeRawDataset(3)[index], 'index': index})
        expected = online[index]
        assert actual.keys() == expected.keys()
        for key in actual:
            assert torch.equal(actual[key], expected[key]), key
    assert TRANSFORMS.get('LingBotSampleTransform') is LingBotSampleTransform
    assert TRANSFORMS.get('LingBotQwen3VLImageTransform') is not None


class _RandomDataset(Dataset):

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            'sample_index':
            torch.tensor(index),
            'rng':
            torch.tensor(
                [random.random(),
                 np.random.rand(),
                 torch.rand(()).item()])
        }


def test_worker_seed_uses_logical_dp_not_local_rank(monkeypatch):
    from fluxvla.engines.utils.torch_utils import worker_init_function

    python_state, numpy_state, torch_state = random.getstate(
    ), np.random.get_state(), torch.get_rng_state()
    try:
        streams = []
        for local_rank, dp_rank in ((0, 1), (3, 1), (3, 2)):
            monkeypatch.setenv('LOCAL_RANK', str(local_rank))
            torch.manual_seed(7)
            worker_init_function(0, data_parallel_rank=dp_rank)
            streams.append((random.random(), np.random.rand(), torch.rand(
                ()).item()))
        assert streams[0] == streams[1]
        assert streams[1] != streams[2]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@pytest.mark.parametrize('workers', [0, 2])
def test_online_worker_sample_contract_survives_rollover_and_resume(
        norm_stats, workers):
    from fluxvla.mcore.lingbot_vla_v2.dataset import \
        LingBotOnlineIterableDataset

    dataset = _TransformedDataset(
        _FakeRawDataset(8),
        norm_stats_path=str(norm_stats),
        tokenizer_path='unused',
        processor=_FakeProcessor(),
        action_horizon=2,
        tokenizer_max_length=6,
        flow_rng_seed=42,
    )
    full = LingBotOnlineIterableDataset(
        dataset,
        dp_rank=0,
        dp_size=2,
        micro_batch_size=2,
        sampler_seed=42,
        num_workers=workers)
    batches = [next(full) for _ in range(4)]
    resumed = LingBotOnlineIterableDataset(
        dataset,
        dp_rank=0,
        dp_size=2,
        micro_batch_size=2,
        sampler_seed=42,
        num_workers=workers,
        start_iteration=3)
    actual = next(resumed)
    for key in actual:
        assert torch.equal(actual[key], batches[3][key]), key
    # Native DataLoader-style batching and the MCore iterable consume the
    # same sampler stream and per-index flow targets in the first epoch.
    sampler = DistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=42)
    native = DataLoader(
        dataset, batch_size=2, sampler=sampler, num_workers=workers)
    for expected, batch in zip(native, batches[:2]):
        for key in expected:
            assert torch.equal(expected[key], batch[key]), key


def test_mcore_matches_native_worker_random_streams():
    from fluxvla.engines.utils.torch_utils import worker_init_function
    from fluxvla.mcore.lingbot_vla_v2.dataset import \
        LingBotOnlineIterableDataset

    dataset = _RandomDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=42)
    native = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        drop_last=True,
        generator=torch.Generator().manual_seed(43),
        worker_init_fn=partial(worker_init_function, data_parallel_rank=1))
    mcore = LingBotOnlineIterableDataset(
        dataset,
        dp_rank=1,
        dp_size=2,
        micro_batch_size=2,
        sampler_seed=42,
        num_workers=2,
        repeat=False)
    for expected, actual in zip(native, mcore, strict=True):
        for key in expected:
            assert torch.equal(expected[key], actual[key]), key


def test_unknown_parquet_version_is_explicitly_rejected(tmp_path):
    from fluxvla.datasets.lingbot_vla_v2 import LingBotParquetDataset

    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(
        json.dumps({'codebase_version': 'v4.0'}))
    with pytest.raises(ValueError, match='Unsupported LeRobot'):
        LingBotParquetDataset(
            str(tmp_path), 'unused', 'unused', video_backend='pyav')


def test_only_one_production_lingbot_dataset_entrypoint():
    from fluxvla.datasets.lingbot_vla_v2 import LingBotParquetDataset
    from fluxvla.engines import DATASETS

    assert DATASETS.get('LingBotParquetDataset') is LingBotParquetDataset
    assert DATASETS.get('LingBotOnlineDataset') is None
    assert not (ROOT /
                'fluxvla/datasets/lingbot_vla_v2/online_dataset.py').exists()
    online = Config.fromfile(
        str(CONFIGS / 'lingbot_vla_v2_online_flow_only_100.py'))
    assert online.train_dataloader.dataset.type == 'LingBotParquetDataset'


def test_mcore_online_builder_uses_same_dataset_and_decode_contract(
        monkeypatch):
    import fluxvla.datasets.lingbot_vla_v2 as datasets
    import fluxvla.mcore.lingbot_vla_v2.dataset as mcore_data

    # No initialized process group or CUDA required to check provider wiring.
    parallel_state = SimpleNamespace(
        get_data_parallel_rank=lambda: 1,
        get_data_parallel_world_size=lambda: 2)
    monkeypatch.setitem(sys.modules, 'megatron.core',
                        SimpleNamespace(parallel_state=parallel_state))
    captured = {}

    def build_dataset(**kwargs):
        captured.update(kwargs)
        return _RandomDataset()

    monkeypatch.setattr(datasets, 'LingBotParquetDataset', build_dataset)
    args = SimpleNamespace(
        micro_batch_size=2,
        global_batch_size=4,
        lingbot_data_path='online-data',
        lingbot_norm_stats='stats',
        lingbot_tokenizer_path='tokenizer',
        lingbot_video_backend='pyav',
        lingbot_resize_dtype='uint8')
    train, valid, test = mcore_data.build_lingbot_datasets(args)
    assert captured['data_path'] == 'online-data'
    assert captured['video_backend'] == 'pyav'
    assert captured['resize_dtype'] == 'uint8'
    assert train.dataset is valid.dataset
    assert train.dp_rank == 1 and train.dp_size == 2
    assert test is None


@pytest.mark.parametrize('horizon', [1, 2, 5])
@pytest.mark.parametrize('version', ['v2.1', 'v3.0'])
@pytest.mark.parametrize('task', ['pick object', 'static', 'empty'])
def test_parquet_adapter_matches_online_contract_at_episode_boundaries(
    tmp_path,
    norm_stats,
    monkeypatch,
    horizon,
    version,
    task,
):
    from datasets import Dataset as HFDataset
    from fluxvla.datasets.lingbot_vla_v2 import (LingBotParquetDataset,
                                                 parquet_dataset)
    from fluxvla.transforms.lingbot_online import ROBOTWIN_CAMERA_KEYS

    root = tmp_path / 'data with spaces'
    (root / 'meta').mkdir(parents=True)
    (root / 'data').mkdir()
    info = dict(
        codebase_version=version,
        fps=50,
        chunks_size=1000,
        video_path='videos/{video_key}/episode_{episode_index:06d}.mp4')
    offsets = {}
    if version.startswith('v2'):
        (root / 'meta/tasks.jsonl'
         ).write_text(json.dumps(dict(task_index=0, task=task)) + '\n')
        (root / 'meta/episodes_stats.jsonl').write_text('{}\n')
        (root / 'meta/episodes.jsonl').write_text('{}\n')
    else:
        info['video_path'] = ('videos/{video_key}/chunk-{chunk_index:03d}/'
                              'file-{file_index:03d}.mp4')
        HFDataset.from_list([dict(task_index=0, __index_level_0__=task)
                             ]).to_parquet(str(root / 'meta/tasks.parquet'))
        (root / 'meta/stats.json').write_text('{}')
        (root / 'meta/episodes/chunk-000').mkdir(parents=True)
        records = []
        for episode in (1, 0):  # Resolve by ID, not metadata row order.
            record = dict(
                episode_index=episode,
                dataset_from_index=episode * 3,
                dataset_to_index=episode * 3 + 3)
            for camera_index, camera in enumerate(ROBOTWIN_CAMERA_KEYS):
                offsets[episode, camera] = float(episode * 2 + camera_index)
                record.update({
                    f'videos/{camera}/chunk_index':
                    1,
                    f'videos/{camera}/file_index':
                    2,
                    f'videos/{camera}/from_timestamp':
                    offsets[episode, camera]
                })
            records.append(record)
        HFDataset.from_list(records).to_parquet(
            str(root / 'meta/episodes/chunk-000/file-000.parquet'))
    (root / 'meta/info.json').write_text(json.dumps(info))
    # Stored timestamp, not current_ts + fixed dt, defines the future frame.
    rows = [
        dict(
            episode_index=index // 3,
            task_index=0,
            timestamp=[0.0, 0.018, 0.041][index % 3],
            **{
                'observation.state': [index / 10] * 14,
                'action': [index / 10] * 14
            }) for index in range(6)
    ]
    HFDataset.from_list(rows).to_parquet(str(root / 'data/episode.parquet'))

    decode_calls = []

    def decode(path, timestamps, tolerance_s, *, backend):
        decode_calls.append((path, sorted(set(timestamps))))
        assert backend == 'pyav'
        match = re.search(r'episode_(\d+)', path)
        episode = int(match.group(1)) if match else 0
        if version.startswith('v3'):
            assert 'chunk-001/file-002.mp4' in path
        camera = next(i for i, key in enumerate(ROBOTWIN_CAMERA_KEYS)
                      if key in path)
        return torch.stack([
            torch.full((3, 2, 2),
                       episode * 30 + camera * 6 + round(ts * 50) % 200,
                       dtype=torch.uint8) for ts in timestamps
        ])

    monkeypatch.setattr(parquet_dataset, 'decode_video_frames', decode)

    class ReferenceRaw(Dataset):

        def __len__(self):
            return 6

        def __getitem__(self, index):
            end = index // 3 * 3 + 2
            indices = [min(index + offset, end) for offset in range(horizon)]
            raw = {
                **rows[index], 'index':
                index,
                'task':
                task,
                'action':
                torch.tensor([rows[i]['action'] for i in indices]),
                'action_is_pad':
                torch.tensor(
                    [index + offset > end for offset in range(horizon)])
            }
            for camera in ROBOTWIN_CAMERA_KEYS:
                path = str(root / info['video_path'].format(
                    video_key=camera,
                    episode_index=index // 3,
                    chunk_index=1,
                    file_index=2))
                offset = offsets.get((index // 3, camera), 0.0)
                raw[camera] = decode(
                    path, [
                        rows[index]['timestamp'] + offset,
                        rows[indices[-1]]['timestamp'] + offset
                    ],
                    0.1,
                    backend='pyav').float().div(255)
            return raw

    processor = _FakeProcessor()
    processor.tokenizer = _helpers._FakeTokenizer()
    processor.tokenizer.expected_prompt = f'user:{task}'
    settings = dict(
        norm_stats_path=str(norm_stats),
        tokenizer_path='unused',
        action_horizon=horizon,
        tokenizer_max_length=6,
        processor=processor,
        return_teacher_images=True,
        model_dtype=torch.float32,
        flow_rng_seed=42)
    online = _TransformedDataset(ReferenceRaw(), **settings)
    parquet = LingBotParquetDataset(
        str(root), video_backend='pyav', image_size=2, **settings)
    for index in range(6):
        decode_calls.clear()
        expected, actual = online[index], parquet[index]
        assert actual.keys() == expected.keys()
        for key in actual:
            assert torch.equal(actual[key], expected[key]), (index, key)
        assert int(actual['sample_index']) == index
        for expected_call, actual_call in zip(
                decode_calls[:3], decode_calls[3:], strict=True):
            assert actual_call[0] == expected_call[0]
            np.testing.assert_allclose(
                actual_call[1], expected_call[1], rtol=0, atol=1e-7)


def test_parquet_preserve_policy_never_resamples_task_text():
    helpers = _load_test_helpers(
        'parquet_test_helpers',
        ROOT / 'test/test_datasets/test_parquet_temporal_window.py')

    dataset = helpers._make_dataset(task_filter_policy='preserve')
    dataset.tasks = [[dict(task='static')]]
    sample = dataset.__getitem__(4, {'test': {}})
    assert int(sample['index']) == 4
    assert sample['task_description'] == 'static'
    np.testing.assert_array_equal(sample['actions'], [[4], [4], [4]])
    np.testing.assert_array_equal(sample['action_masks'], [1, 0, 0])


@pytest.mark.parametrize('storage', ['image', 'video'])
@pytest.mark.parametrize('resize_dtype', ['uint8', 'float32'])
@pytest.mark.parametrize('use_future', [False, True])
def test_v3_real_media_decode_and_tail_padding(tmp_path, norm_stats, storage,
                                               resize_dtype, use_future):
    from PIL import Image

    from datasets import Dataset as HFDataset
    from datasets import Image as HFImage
    from fluxvla.datasets.lingbot_vla_v2 import LingBotParquetDataset
    from fluxvla.transforms.lingbot_online import ROBOTWIN_CAMERA_KEYS

    root = tmp_path / 'real media'
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data').mkdir()
    info = dict(
        codebase_version='v3.0',
        fps=50,
        video_path=('videos/{video_key}/chunk-{chunk_index:03d}/'
                    'file-{file_index:03d}.mp4'),
        features={
            key: dict(dtype=storage, shape=[3, 4, 4])
            for key in ROBOTWIN_CAMERA_KEYS
        })
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/stats.json').write_text('{}')
    HFDataset.from_list([dict(task_index=0, __index_level_0__='pick object')
                         ]).to_parquet(str(root / 'meta/tasks.parquet'))
    episodes = [
        dict(
            episode_index=ep,
            dataset_from_index=ep * 3,
            dataset_to_index=ep * 3 + 3) for ep in (0, 1)
    ]
    rows = [
        dict(
            episode_index=i // 3,
            task_index=0,
            timestamp=(i % 3) / 50,
            **{
                'observation.state': [i / 10] * 14,
                'action': [i / 10] * 14
            }) for i in range(6)
    ]
    pixels = {}
    for ci, camera in enumerate(ROBOTWIN_CAMERA_KEYS):
        for i in range(6):
            # Nonconstant spatial patterns exercise resize's dtype boundary.
            pixels[camera, i] = (np.arange(48).reshape(4, 4, 3) + i * 20 +
                                 ci * 30).astype(np.uint8)
        if storage == 'image':
            for i, row in enumerate(rows):
                row[camera] = Image.fromarray(pixels[camera, i])
        else:
            av = pytest.importorskip('av')
            pytest.importorskip('torchcodec')
            path = root / info['video_path'].format(
                video_key=camera, chunk_index=0, file_index=0)
            path.parent.mkdir(parents=True)
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('libx264rgb', rate=50)
                stream.width = stream.height = 4
                stream.pix_fmt = 'rgb24'
                stream.options = {'crf': '0', 'preset': 'ultrafast'}
                for i in range(6):
                    for packet in stream.encode(
                            av.VideoFrame.from_ndarray(
                                pixels[camera, i], format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            for ep, record in enumerate(episodes):
                record.update({
                    f'videos/{camera}/chunk_index': 0,
                    f'videos/{camera}/file_index': 0,
                    f'videos/{camera}/from_timestamp': ep * 3 / 50
                })
    dataset = HFDataset.from_list(rows)
    if storage == 'image':
        for camera in ROBOTWIN_CAMERA_KEYS:
            dataset = dataset.cast_column(camera, HFImage())
    dataset.to_parquet(str(root / 'data/file-000.parquet'))
    HFDataset.from_list(episodes).to_parquet(
        str(root / 'meta/episodes/chunk-000/file-000.parquet'))
    # A leftover legacy file must not shadow the canonical v3 metadata.
    (root / 'meta/episodes.jsonl').write_text('{}\n')
    actual = LingBotParquetDataset(
        str(root),
        str(norm_stats),
        'unused',
        action_horizon=5,
        image_size=2,
        tokenizer_max_length=6,
        processor=_FakeProcessor(),
        resize_dtype=resize_dtype,
        use_future_images=use_future,
        return_teacher_images=True)
    from torchvision.transforms.v2 import Resize

    for i in range(6):
        end = i // 3 * 3 + 2
        raw = {
            **rows[i], 'index':
            i,
            'task':
            'pick object',
            'action':
            torch.tensor([rows[min(i + j, end)]['action'] for j in range(5)]),
            'action_is_pad':
            torch.tensor([i + j > end for j in range(5)])
        }
        for camera in ROBOTWIN_CAMERA_KEYS:
            indices = [i, end] if use_future else [i]
            frames = torch.stack([
                torch.from_numpy(pixels[camera, j]).permute(2, 0, 1)
                for j in indices
            ])
            if resize_dtype == 'float32':
                frames = frames.float() / 255
            raw[camera] = Resize((2, 2))(frames)
        expected = actual.transform(raw)
        sample = actual[i]
        for key in expected:
            assert torch.equal(sample[key], expected[key]), (storage, i, key)
    with pytest.raises(IndexError):
        actual[len(actual)]
    if storage == 'video':
        metadata = actual.reader.get_episode_metadata(0, 0)
        del metadata[f'videos/{ROBOTWIN_CAMERA_KEYS[0]}/from_timestamp']
        with pytest.raises(ValueError, match='Missing episode video offset'):
            actual[0]
    else:
        assert not set(ROBOTWIN_CAMERA_KEYS) & set(
            actual.reader._window_rows().column_names)


@pytest.mark.parametrize('mode', ['online', 'replay'])
def test_mcore_launcher_validates_and_forwards_external_paths(tmp_path, mode):
    # Capture argv instead of starting torchrun; no GPU/model is needed.
    inputs = tmp_path / 'external inputs with spaces'
    inputs.mkdir()
    for name in ('checkpoint', 'tokenizer', 'dataset', 'megatron'):
        (inputs / name).mkdir()
    (inputs / 'model.yaml').write_text('model: {}\ntrain: {}\n')
    (inputs / 'manifest.json').write_text('{}\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    captured = tmp_path / 'argv.json'
    python = bindir / 'python'
    python.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                      "open(os.environ['CAPTURE_ARGV'], 'w').write("
                      'json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    env = {
        **os.environ, 'PATH': f"{bindir}{os.pathsep}{os.environ['PATH']}",
        'CAPTURE_ARGV': str(captured),
        'MEGATRON_ROOT': str(inputs),
        'LINGBOT_CHECKPOINT': str(inputs / 'checkpoint'),
        'LINGBOT_TOKENIZER_PATH': str(inputs / 'tokenizer'),
        'LINGBOT_NATIVE_CONFIG': str(inputs / 'model.yaml'),
        'LINGBOT_DATA_PATH': str(inputs / 'dataset'),
        'LINGBOT_MANIFEST': str(inputs / 'manifest.json'),
        'LINGBOT_DATA_MODE': mode,
        'LINGBOT_NUM_WORKERS': '2'
    }
    script = ROOT / 'scripts/train_mcore_lingbot_vla_v2.sh'
    subprocess.run(['bash', str(script)], env=env, cwd=tmp_path, check=True)
    args = json.loads(captured.read_text())
    assert args[args.index('--lingbot-data-mode') + 1] == mode
    assert args[args.index('--lingbot-checkpoint') + 1] == str(inputs /
                                                               'checkpoint')
    assert args[args.index('--lingbot-native-config') + 1] == str(inputs /
                                                                  'model.yaml')
    assert args[args.index('--lingbot-sampler-seed') + 1] == '42'
    if mode == 'online':
        assert args[args.index('--lingbot-num-workers') + 1] == '2'
        assert Path(args[args.index('--lingbot-norm-stats') + 1]).is_file()
        assert '--lingbot-manifest' not in args
    else:
        assert args[args.index('--lingbot-manifest') + 1] == str(
            inputs / 'manifest.json')
        assert '--lingbot-data-path' not in args
    captured.unlink()
    env['LINGBOT_NATIVE_CONFIG'] = str(inputs / 'missing.yaml')
    result = subprocess.run(['bash', str(script)],
                            env=env,
                            cwd=tmp_path,
                            capture_output=True,
                            text=True)
    assert result.returncode != 0
    assert 'Model YAML not found' in result.stderr
    assert not captured.exists()


def test_loader_generator_does_not_consume_global_torch_rng():
    from fluxvla.mcore.lingbot_vla_v2.dataset import \
        LingBotOnlineIterableDataset

    class Indices(Dataset):

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {'sample_index': torch.tensor(index)}

    state = torch.get_rng_state().clone()
    iterator = LingBotOnlineIterableDataset(
        Indices(), dp_rank=0, dp_size=1, micro_batch_size=1)
    next(iterator)
    assert torch.equal(state, torch.get_rng_state())


def test_native_loader_generator_explicit_seed_and_legacy_fallback(
        monkeypatch):
    from fluxvla.engines.runners.base_train_runner import BaseTrainRunner

    runner = SimpleNamespace(_data_parallel_rank=lambda: 2)
    state = torch.get_rng_state().clone()
    generator = BaseTrainRunner._build_dataloader_generator(runner, seed=7)
    assert torch.equal(generator.get_state(),
                       torch.Generator().manual_seed(9).get_state())
    assert torch.equal(state, torch.get_rng_state())
    monkeypatch.delenv('EXPERIMENT_GLOBAL_SEED', raising=False)
    assert BaseTrainRunner._build_dataloader_generator(runner) is None
    monkeypatch.setenv('EXPERIMENT_GLOBAL_SEED', '11')
    generator = BaseTrainRunner._build_dataloader_generator(runner)
    assert torch.equal(generator.get_state(),
                       torch.Generator().manual_seed(13).get_state())
