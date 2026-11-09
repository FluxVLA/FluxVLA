# Copyright 2026 Limx Dynamics
"""Two-GPU FP32 mixed-update parity, without pretrained weights or data.

Run with torchrun --standalone --nproc-per-node=2. This is deliberately not
pytest-collected: CPU CI must not silently skip a real distributed regression.
An independent autograd/all-reduce oracle preserves globally-unused None grads.
--isolated-imports only isolates package initializers, NOT CUDA/DDP/FSDP math.
"""

import argparse
import copy
import json
import os
import runpy
import sys
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 3e-6, 1e-4


def batch(task, step, micro, rank):
    generator = torch.Generator().manual_seed(900 + step * 30 + micro * 3 +
                                              rank)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).cuda()

    if task == 'vlm':
        inputs = dict(
            input_ids=torch.tensor([[2, 5, 6, 1]], device='cuda'),
            labels=torch.tensor([[-100, -100 if rank else 5, 6, 1]],
                                device='cuda'))
        # Unequal target counts and image/text-only ranks in the same step.
        if rank == 0:
            inputs['images'] = randn(1, 3, 8, 8)
        return inputs
    return dict(
        images=randn(1, 3, 8, 8),
        img_masks=torch.ones(1, 1, device='cuda').bool(),
        lang_tokens=torch.tensor([[2, 5, 1]], device='cuda'),
        lang_masks=torch.ones(1, 3, device='cuda').bool(),
        states=randn(1, 32),
        actions=randn(1, 4, 32),
        noise=randn(1, 4, 32),
        time=torch.tensor([0.25 + rank * 0.2], device='cuda'),
        return_loss_statistics=True)


def reduce_reference_gradients(reference, denominator):
    dist.all_reduce(denominator)
    params = [p for p in reference.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in params],
                        device='cuda',
                        dtype=torch.int64)
    dist.all_reduce(used)
    for index, param in enumerate(params):
        if not used[index]:
            param.grad = None
            continue
        grad = param.grad if param.grad is not None else torch.zeros_like(
            param)
        dist.all_reduce(grad)
        param.grad = grad * (1.0 / denominator).to(grad.dtype)


def full_state(model):
    return model.module.state_dict() if isinstance(
        model, DDP) else model.state_dict()


def run_case(kind, sync_each_microbatch, image_pattern, make_policy,
             mixed_class, config):
    rank = int(os.environ['LOCAL_RANK'])
    torch.manual_seed(13)
    policy = make_policy().cuda().train()
    policy.prepare_training_task('mixed')
    reference = copy.deepcopy(policy)
    if kind == 'ddp':
        wrapped = DDP(
            policy,
            device_ids=[rank],
            find_unused_parameters=True,
            static_graph=False,
            gradient_as_bucket_view=True)
    else:
        wrapped = FSDP(
            policy,
            device_id=rank,
            use_orig_params=True,
            auto_wrap_policy=policy.get_mixed_fsdp_wrapping_policy())
        if kind == 'fsdp_checkpointed':
            layer_cls = reference.llm_backbone.transformer_layer_cls
            apply_activation_checkpointing(
                wrapped,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda module: isinstance(module, layer_cls))
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=1e-4, weight_decay=0.01)
    ref_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=1e-4, weight_decay=0.01)
    mixed = mixed_class(config, sync_each_microbatch=sync_each_microbatch)
    runner = SimpleNamespace(
        vla=wrapped,
        optimizer=optimizer,
        metric=SimpleNamespace(commit=lambda **kwargs: None),
        _unwrap_vla=lambda: wrapped.module)
    results = []
    for step, task in enumerate(('vla', 'vlm', 'vla', 'vlm')):
        previous = {
            name: value.detach().clone()
            for name, value in full_state(wrapped).items()
        }
        mixed.active_task = task
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        denominator = torch.zeros((), device='cuda', dtype=torch.float64)
        for micro in range(2):
            inputs = batch(task, step, micro, rank)
            if task == 'vlm' and (image_pattern == 'text_only' or
                                  (image_pattern == 'first_micro_only'
                                   and micro == 1)):
                inputs.pop('images', None)
            context = (
                nullcontext()
                if sync_each_microbatch or micro == 1 else wrapped.no_sync())
            with context:
                output = wrapped(task=task, **inputs)
                mixed.backward(runner, output, should_step=micro == 1)
            expected = reference(task=task, **inputs)
            expected['loss_sum'].backward()
            denominator += expected['num_tokens' if task ==
                                    'vlm' else 'num_action_targets']
        reduce_reference_gradients(reference, denominator)
        optimizer.step()
        ref_optimizer.step()
        observed = full_state(wrapped)
        expected = reference.state_dict()
        assert observed.keys() == expected.keys()
        largest = max(
            float((value - expected[name]).abs().max())
            for name, value in observed.items() if value.numel())
        differences = [
            name for name, value in observed.items()
            if not torch.allclose(value, expected[name], atol=ATOL, rtol=RTOL)
        ]
        # Exact non-update of EVERY globally-unused parameter, not just o_proj.
        inactive_updates = [
            name for name, param in reference.named_parameters()
            if param.grad is None
            and not torch.equal(previous[name], observed[name])
        ]
        tail = wrapped.module.llm_backbone.layers[-1].self_attn.o_proj.weight
        tail_step = optimizer.state.get(tail, {}).get('step')
        local_step = None if tail_step is None else float(tail_step)
        state_steps = [None] * dist.get_world_size()
        parameters = {
            name.replace('_fsdp_wrapped_module.',
                         '').replace('_checkpoint_wrapped_module.', ''):
            parameter
            for name, parameter in wrapped.module.named_parameters()
        }
        step_errors = []
        for name, ref_parameter in reference.named_parameters():
            parameter = parameters[name]
            if not parameter.numel():
                continue  # This FSDP rank does not own a slice.
            actual_step = optimizer.state.get(parameter, {}).get('step', 0)
            ref_step = ref_optimizer.state.get(ref_parameter,
                                               {}).get('step', 0)
            if float(actual_step) != float(ref_step):
                step_errors.append(name)
        dist.all_gather_object(state_steps, (local_step, step_errors))
        optimizer_step_mismatches = [
            name for _, errors in state_steps for name in errors
        ]
        state_steps = [value for value, _ in state_steps]
        expected_step = (step + 1) // 2
        assert all(value in (None, expected_step) for value in state_steps)
        assert expected_step == 0 or expected_step in state_steps
        results.append(
            dict(
                step=step,
                task=task,
                max_parameter_difference=largest,
                differing_parameters=differences,
                inactive_updates=inactive_updates,
                optimizer_step_mismatches=optimizer_step_mismatches,
                language_tail_adam_steps=state_steps))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isolated-imports', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    torch.set_num_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.isolated_imports:
        keys = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')
        environment = {
            key: os.environ.pop(key)
            for key in keys if key in os.environ
        }
        try:
            with patch.object(torch.cuda, 'is_available', return_value=False):
                runpy.run_path(str(
                    ROOT / 'test/_cpu_model_runtime.py'))['bootstrap']()
        finally:
            os.environ.update(environment)
    from fluxvla.engines.runners.mixed_training import MixedTraining
    units = runpy.run_path(str(ROOT / 'test/test_models/test_vlm_training.py'))
    mixed_units = runpy.run_path(
        str(ROOT / 'test/test_runners/test_mixed_training.py'))
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Production package imports may initialize distributed Overwatch first.
    if not dist.is_initialized():
        dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    assert dist.get_world_size(
    ) == 2, 'Run this regression with exactly two GPUs.'
    results = {}
    try:
        for kind in ('ddp', 'fsdp', 'fsdp_checkpointed'):
            for sync in (True, False):
                for pattern in ('asymmetric', 'first_micro_only', 'text_only'):
                    key = f'{kind}_sync={sync}_images={pattern}'
                    results[key] = run_case(kind, sync, pattern,
                                            units['pi05'].__wrapped__,
                                            MixedTraining,
                                            mixed_units['configuration']())
                    if rank == 0:
                        print(key, json.dumps(results[key]), flush=True)
                    torch.cuda.empty_cache()
        if rank == 0 and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
        assert all(
            not step['differing_parameters'] and not step['inactive_updates']
            and not step['optimizer_step_mismatches']
            for case in results.values() for step in case), results
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
