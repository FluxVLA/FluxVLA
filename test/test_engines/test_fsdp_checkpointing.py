import gc
import zipfile
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    checkpoint_wrapper
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType

# Importing FluxVLA's optional RADIO backbone queries device capability at
# module import time. Stub only that query; all tests below remain CPU-only.
with mock.patch.object(
        torch.cuda, 'get_device_capability', return_value=(8, 0)):
    from fluxvla.engines.runners import base_train_runner, fsdp_train_runner
    from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner


class _TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)
        self.output = nn.Linear(4, 2)
        self.register_buffer('scale', torch.tensor(0.5))

    def forward(self, inputs):
        return (self.output(self.block(inputs)) * self.scale).sum()


@pytest.fixture()
def single_process_group(tmp_path):
    if dist.is_initialized():
        assert dist.get_world_size() == 1
        yield
        return

    rendezvous = tmp_path / 'gloo-rendezvous'
    dist.init_process_group(
        'gloo', init_method=f'file://{rendezvous}', rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _nested_cpu_fsdp_model():
    model = _TinyModel()
    model.block = FSDP(
        checkpoint_wrapper(model.block),
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=torch.device('cpu'))
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=torch.device('cpu'))


def _assert_tensor_tree_on_cpu(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
    elif isinstance(value, dict):
        for item in value.values():
            _assert_tensor_tree_on_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_tensor_tree_on_cpu(item)


def test_move_checkpoint_tensors_to_cpu_does_not_mutate_live_state():
    pair_type = namedtuple('Pair', ('first', 'second'))
    cpu_tensor = torch.ones(1)
    with FakeTensorMode():
        cuda_tensor = torch.ones(2, device='cuda')
        state = OrderedDict([
            ('tensor', cuda_tensor),
            ('nested', {
                'list': [cuda_tensor],
                'tuple': (cuda_tensor, ),
                'namedtuple': pair_type(cuda_tensor, 3),
                'size': torch.Size((2, 3)),
            }),
        ])
        state._metadata = OrderedDict([('', {'version': 1})])
        state['cpu_tensor'] = cpu_tensor
        original_nested = state['nested']

        converted = FSDPTrainRunner._move_checkpoint_tensors_to_cpu(state)

    assert converted is not state
    assert converted['nested'] is not original_nested
    assert state['tensor'] is cuda_tensor
    assert state['tensor'].device.type == 'cuda'
    assert state['nested']['list'][0] is cuda_tensor
    assert state['nested']['tuple'][0] is cuda_tensor
    assert converted['cpu_tensor'] is cpu_tensor
    assert converted._metadata == OrderedDict([('', {'version': 1})])
    assert isinstance(converted['nested']['namedtuple'], pair_type)
    assert isinstance(converted['nested']['size'], torch.Size)
    _assert_tensor_tree_on_cpu(converted)


def test_checkpoint_cpu_copy_breaks_live_optimizer_container_aliases():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()

    parameter = next(model.parameters())
    exported = optimizer.state_dict()
    parameter_id = exported['param_groups'][0]['params'][0]
    assert exported['state'][parameter_id] is optimizer.state[parameter]

    converted = FSDPTrainRunner._move_checkpoint_tensors_to_cpu(exported)

    assert converted is not exported
    assert converted['state'] is not exported['state']
    assert converted['state'][parameter_id] is not optimizer.state[parameter]
    assert exported['state'][parameter_id] is optimizer.state[parameter]


def test_nested_fsdp_checkpoint_has_stable_keys_and_cpu_locations(
        single_process_group, tmp_path):
    vla = _nested_cpu_fsdp_model()
    optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-3)
    vla(torch.ones(2, 4)).backward()
    optimizer.step()

    runner = object.__new__(FSDPTrainRunner)
    runner.vla = vla
    runner.optimizer = optimizer
    runner.lr_scheduler = None
    runner.tokenizer = None
    runner.change_key_name = False
    runner.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
    runner.fsdp_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True)
    runner.max_keep_ckpts = 2

    with mock.patch.object(
            fsdp_train_runner.overwatch,
            'is_rank_zero', return_value=True), mock.patch.object(
                torch.cuda,
                'synchronize'), mock.patch.object(torch.cuda, 'empty_cache'):
        runner.save_checkpoint(
            tmp_path, global_step=1, epoch=0, train_loss=1.0)

    checkpoint_path = next((tmp_path / 'checkpoints').glob('*.pt'))
    safetensors_path = checkpoint_path.with_suffix('.safetensors')
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True)

    expected_keys = list(_TinyModel().state_dict())
    assert list(checkpoint['model']) == expected_keys
    _TinyModel().load_state_dict(checkpoint['model'], strict=True)
    _assert_tensor_tree_on_cpu(checkpoint)

    with safe_open(safetensors_path, framework='pt', device='cpu') as handle:
        assert list(handle.keys()) == sorted(expected_keys)

    with zipfile.ZipFile(checkpoint_path) as archive:
        pickle_name = next(name for name in archive.namelist()
                           if name.endswith('/data.pkl'))
        pickle_payload = archive.read(pickle_name)
    assert b'cuda' not in pickle_payload
    assert b'cpu' in pickle_payload

    del checkpoint
    del runner
    del optimizer
    del vla
    gc.collect()


def test_resume_deserializes_checkpoint_on_cpu():
    runner = object.__new__(FSDPTrainRunner)
    runner.resume_from = 'checkpoint.pt'
    runner.metric = SimpleNamespace(global_step=0)
    runner.current_epoch = 0
    runner.optimizer = None
    runner.lr_scheduler = None

    with mock.patch.object(
            base_train_runner.torch, 'load',
            return_value={}) as load, mock.patch.object(
                base_train_runner.overwatch,
                'is_rank_zero',
                return_value=False), mock.patch.object(base_train_runner.dist,
                                                       'barrier'):
        runner.resume()

    load.assert_called_once_with(
        'checkpoint.pt', map_location='cpu', weights_only=True)
