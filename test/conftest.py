"""Offline, deterministic defaults for unit and checkpoint regression tests."""

import random
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

# Standalone hardware benchmarks are kept at their original test paths.
# They are not pytest unit tests and must not execute during collection.
collect_ignore = [
    'test_models/test_gr00t_orin.py',
    'test_models/test_pi05_orin.py',
]


def pytest_addoption(parser):
    parser.addoption(
        '--run-checkpoint-tests',
        action='store_true',
        help='Include full-size pretrained/checkpoint numerical regressions.')
    parser.addoption(
        '--test-order-seed',
        type=int,
        default=None,
        help='Shuffle selected tests using an isolated seeded RNG.')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-checkpoint-tests'):
        selected, checkpoint_tests = [], []
        for item in items:
            target = (
                checkpoint_tests
                if item.get_closest_marker('checkpoint') else selected)
            target.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=checkpoint_tests)
    order_seed = config.getoption('--test-order-seed')
    if order_seed is not None:
        random.Random(order_seed).shuffle(items)


@pytest.fixture(autouse=True)
def checkpoint_runtime(request, monkeypatch):
    if request.node.get_closest_marker('checkpoint') is None:
        yield
        return
    if not torch.cuda.is_available():
        pytest.skip('Checkpoint regression requires CUDA.')
    # Historical fixtures/configs use repository-relative paths.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    yield
    # Release large per-test modules before constructing the next policy.
    if request.instance is not None:
        for name in ('vla', 'llm_backbone', 'vlm_backbone', 'siglip_vit'):
            if hasattr(request.instance, name):
                delattr(request.instance, name)
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def pytest_configure(config):
    # Apply before collecting modules that import HF clients / model configs.
    patch = pytest.MonkeyPatch()
    for key, value in {
            'HF_HUB_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
            'WANDB_MODE': 'disabled',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    }.items():
        patch.setenv(key, value)
    config.add_cleanup(patch.undo)


@pytest.fixture(scope='session', autouse=True)
def deterministic_torch():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


@pytest.fixture(autouse=True)
def isolated_random_state():
    """A test must neither inherit nor leak another test's random stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    # Do not initialize CUDA just to run CPU unit tests.
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized() else None)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):

    def blocked(*args, **kwargs):
        raise AssertionError('Unit tests must mock network access.')

    monkeypatch.setattr(socket, 'create_connection', blocked)
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
