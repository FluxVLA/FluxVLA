# Copyright 2026 Limx Dynamics
"""Real, local migration evidence; run each repository in a fresh interpreter.

Capture BEFORE changing the original worktree, then compare in the SAME Python
installation. No model source, weights, or purported golden outputs are bundled
with this test. All expected tensors are produced by the requested
original repo.

Example::

    python test/export_components_probe.py capture --repo /tmp/original \
        --expected-commit 0439d4a --output /tmp/component-baseline
    python test/export_components_probe.py compare --repo /path/to/FluxVLA \
        --baseline /tmp/component-baseline --output /tmp/component-comparison

Only the existing test/_cpu_model_runtime.py isolates legacy package imports.
Shared component imports are ordinary imports, not namespace substitutes.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

SCHEMA_VERSION = 1
SEED = 20261104


def tiny_gemma_config(protected=False, conditional=False):
    return dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        adarms_cond_dim=16 if conditional else None,
        hidden_activation='gelu_pytorch_tanh',
        adarms_fp32=protected,
        attention_math_fp32=protected,
        fp32_layers=[0] if protected else [])


def tiny_vision_config():
    return dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
        attention_dropout=0.0)


def tiny_policy_config(policy_type, protected=False):
    """Match the repository's real tiny PI fixture, with two Gemma blocks."""

    def linear(in_dim, out_dim):
        return dict(type='LinearProjector', in_dim=in_dim, out_dim=out_dim)

    backbone = tiny_gemma_config(protected=protected)
    expert = tiny_gemma_config(
        protected=protected, conditional=policy_type == 'PI05FlowMatching')
    cfg = dict(
        type=policy_type,
        llm_backbone=dict(type='ConditionGemmaModel', **backbone),
        llm_expert=dict(type='ConditionGemmaModel', **expert),
        vision_backbone=dict(
            type='SigLIPViTBackbone',
            vision_backbone_id='siglip_224',
            vision_config=tiny_vision_config(),
            openpi_stem_fp32=protected,
            preserve_fp32_residuals=protected),
        projector=linear(16, 16),
        proj_width=16,
        action_in_proj=linear(4, 16),
        action_out_proj=linear(16, 4),
        max_action_dim=4,
        ori_action_dim=3,
        n_action_steps=3,
        num_steps=2,
        enable_mixed_precision_training=False,
        freeze_vision_backbone=False,
        freeze_llm_backbone=False)
    if policy_type == 'PI0FlowMatching':
        cfg.update(
            state_proj=linear(4, 16),
            action_time_mlp_in=linear(32, 16),
            action_time_mlp_out=linear(16, 16))
    else:
        cfg.update(time_mlp_in=linear(16, 16), time_mlp_out=linear(16, 16))
    return cfg


def case_specs():
    cases = []
    for amp in (False, True):
        precision = 'bf16' if amp else 'fp32'
        for protected in (False, True):
            suffix = f'{precision}-' + ('protected'
                                        if protected else 'default')
            for policy in ('PI0FlowMatching', 'PI05FlowMatching'):
                cases.append(
                    dict(
                        name=f'{policy}-{suffix}',
                        kind='policy',
                        amp=amp,
                        config=tiny_policy_config(policy, protected)))
            for conditional in (False, True):
                cases.append(
                    dict(
                        name=('gemma-'
                              f'{"conditional" if conditional else "plain"}'
                              f'-{suffix}'),
                        kind='gemma',
                        amp=amp,
                        config=tiny_gemma_config(protected, conditional)))
            cases.append(
                dict(
                    name=f'siglip-{suffix}',
                    kind='siglip',
                    amp=amp,
                    config=dict(
                        vision_config=tiny_vision_config(),
                        openpi_stem_fp32=protected,
                        preserve_fp32_residuals=protected)))
        cases.append(
            dict(
                name=f'projector-{precision}',
                kind='projector',
                amp=amp,
                config=dict(in_dim=16, out_dim=7)))
        cases.append(
            dict(
                name=f'helpers-{precision}',
                kind='helpers',
                amp=amp,
                config=tiny_gemma_config()))
    return cases


def fixture_inputs(kind):
    import torch

    if kind == 'policy':
        return dict(
            images=torch.linspace(-1, 1, 2 * 6 * 8 * 8).reshape(2, 6, 8, 8),
            img_masks=torch.tensor([[True, True], [True, False]]),
            lang_tokens=torch.tensor([[1, 5, 2, 0], [1, 7, 3, 2]]),
            lang_masks=torch.tensor([[True, True, True, False], [True] * 4]),
            states=torch.linspace(-0.5, 0.5, 8).reshape(2, 4),
            noise=torch.linspace(-1, 1, 24).reshape(2, 3, 4),
            actions=torch.linspace(-0.5, 0.5, 24).reshape(2, 3, 4),
            action_masks=torch.tensor([[True, True, False],
                                       [True, False, False]]),
            time=torch.tensor([0.25, 0.75]))
    if kind == 'gemma':
        return dict(
            hidden=torch.linspace(-0.7, 0.9, 2 * 7 * 16).reshape(2, 7, 16),
            condition=torch.linspace(-0.9, 0.6, 2 * 7 * 16).reshape(2, 7, 16),
            pad_mask=torch.tensor([[True, True, False, True, True, True, True],
                                   [True] * 7]),
            block_mask=torch.tensor(
                [[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=torch.bool))
    if kind == 'siglip':
        return dict(
            images=torch.linspace(-1, 1, 2 * 6 * 8 * 8).reshape(2, 6, 8, 8))
    if kind == 'projector':
        return dict(
            hidden=torch.linspace(-0.8, 0.6, 2 * 3 * 16).reshape(2, 3, 16))
    if kind == 'helpers':
        return dict(
            query=torch.linspace(-2, 3, 2 * 2 * 4 * 8).reshape(2, 2, 4, 8),
            key=torch.linspace(-1, 2, 2 * 1 * 4 * 8).reshape(2, 1, 4, 8),
            value=torch.linspace(-3, 1, 2 * 1 * 4 * 8).reshape(2, 1, 4, 8),
            pad_mask=torch.tensor([[True, True, False, True], [True] * 4]),
            block_mask=torch.tensor([[0, 0, 1, 0], [0, 1, 0, 1]],
                                    dtype=torch.bool),
            time=torch.tensor([[0.0, 0.25], [0.5, 1.0]]))
    raise ValueError(f'Unknown probe kind: {kind}')


def build_case_model(spec, shared=False):
    if spec['kind'] == 'policy':
        from fluxvla.engines import build_vla_from_cfg
        return build_vla_from_cfg(spec['config']).float().eval()
    if spec['kind'] in ('gemma', 'helpers'):
        name = ('fluxvla_export.components.condition_gemma'
                if shared else 'fluxvla.models.backbones.llms.condition_gemma')
        cls = importlib.import_module(name).ConditionGemmaModel
        return cls(**spec['config']).float().eval()
    if spec['kind'] == 'siglip':
        cfg = dict(spec['config'])
        if shared:
            from fluxvla_export.components.siglip import SigLIPViTBackbone
            from transformers import SiglipVisionConfig, SiglipVisionModel
            vision = SiglipVisionModel(
                SiglipVisionConfig(**cfg.pop('vision_config')))
            return SigLIPViTBackbone(vision, **cfg).float().eval()
        from fluxvla.models.backbones.visions.siglip_vit import \
            SigLIPViTBackbone
        return SigLIPViTBackbone(
            vision_backbone_id='siglip_224', **cfg).float().eval()
    if spec['kind'] == 'projector':
        name = ('fluxvla_export.components.projector'
                if shared else 'fluxvla.models.projectors.linear_projector')
        return importlib.import_module(name).LinearProjector(
            **spec['config']).float().eval()
    raise ValueError(f'Unknown probe kind: {spec["kind"]}')


def _clone_tensors(values):
    return {name: value.detach().clone() for name, value in values.items()}


def _record(results, name, value):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name}: expected a real Tensor, got {type(value)}')
    if name in results:
        raise ValueError(f'Duplicate recorded tensor: {name}')
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise AssertionError(f'{name}: non-finite probe output')
    results[name] = value.detach().cpu().contiguous().clone()


def _record_cache(results, cache, prefix):
    if cache is None or not cache.layers:
        raise AssertionError(f'{prefix}: original model returned no cache')
    for index, layer in enumerate(cache.layers):
        _record(results, f'{prefix}.{index}.key', layer.keys)
        _record(results, f'{prefix}.{index}.value', layer.values)


def _execute_policy(model, inputs, results):
    import torch

    from fluxvla.engines.utils.model_utils import make_att_2d_masks

    result = model(**_clone_tensors(inputs))
    _record(results, 'loss', result['loss'])
    _record(results, 'training_velocity', result['predictions'])
    result['loss'].backward()
    inference = {
        name: value
        for name, value in inputs.items()
        if name not in ('actions', 'action_masks', 'time')
    }
    with torch.no_grad():
        prefix, pad, blocks = model.embed_prefix(**_clone_tensors(inference))
        prefix = model._cast_gemma_input(prefix)
        mask = model._prepare_attention_masks_4d(
            make_att_2d_masks(pad, blocks))
        positions = torch.cumsum(pad, dim=1) - 1
        model.llm_backbone.config._attn_implementation = (
            model.attention_implementation)
        encoded = model.llm_backbone(
            inputs_embeds=prefix,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=True)
        _record(results, 'prefix.embeddings', prefix)
        _record(results, 'prefix.padding', pad)
        _record(results, 'prefix.blocks', blocks)
        _record(results, 'prefix.hidden', encoded.last_hidden_state)
        _record_cache(results, encoded.past_key_values, 'prefix.cache')
        velocity = model.denoise_step(inputs['states'].clone(), pad.clone(),
                                      encoded.past_key_values,
                                      inputs['noise'].clone(),
                                      inputs['time'].clone())
        _record(results, 'single_step_velocity', velocity)
        # predict_action mutates its noise argument.
        # Never pass the stored input.
        action = model.predict_action(**_clone_tensors(inference)).clone()
        repeated = model.predict_action(**_clone_tensors(inference)).clone()
        torch.testing.assert_close(action, repeated, rtol=0, atol=0)
        _record(results, 'final_action', action)


def _execute_gemma(model, inputs, results, helpers):
    import torch

    hidden = inputs['hidden'].clone().requires_grad_()
    condition = inputs['condition'].clone().requires_grad_()
    conditional = model.config.adarms_cond_dim is not None
    pad = inputs['pad_mask']
    mask = helpers.make_att_2d_masks(pad, inputs['block_mask'])[:, None]
    mask = torch.zeros_like(
        mask, dtype=torch.float32).masked_fill(~mask,
                                               torch.finfo(torch.bfloat16).min)
    positions = torch.cumsum(pad, dim=1) - 1
    output = model(
        inputs_embeds=hidden,
        adarms_cond=condition if conditional else None,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False)
    loss = output.last_hidden_state.square().mean()
    _record(results, 'hidden', output.last_hidden_state)
    _record(results, 'loss', loss)
    loss.backward()
    _record(results, 'input_gradient', hidden.grad)
    if conditional:
        _record(results, 'condition_gradient', condition.grad)
    with torch.no_grad():
        prefix = model(
            inputs_embeds=hidden[:, :3],
            adarms_cond=condition[:, :3] if conditional else None,
            attention_mask=mask[:, :, :3, :3],
            position_ids=positions[:, :3],
            use_cache=True)
        _record(results, 'prefix.hidden', prefix.last_hidden_state)
        _record_cache(results, prefix.past_key_values, 'prefix.cache')
        suffix = model(
            inputs_embeds=hidden[:, 3:],
            adarms_cond=condition[:, 3:] if conditional else None,
            attention_mask=mask[:, :, 3:],
            position_ids=positions[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=False)
        _record(results, 'cached_suffix.hidden', suffix.last_hidden_state)


def _execute_helpers(model, inputs, results, helpers):
    import torch

    q = inputs['query'].clone().requires_grad_()
    k = inputs['key'].clone().requires_grad_()
    v = inputs['value'].clone().requires_grad_()
    module = model.layers[
        0].self_attn  # A real attention nn.Module, not a stub.
    block_mask = helpers.make_att_2d_masks(inputs['pad_mask'],
                                           inputs['block_mask'])
    mask = torch.zeros_like(
        block_mask[:, None],
        dtype=torch.float32).masked_fill(~block_mask[:, None],
                                         torch.finfo(torch.bfloat16).min)
    values = dict(
        mask=block_mask,
        repeated_kv=helpers.repeat_kv(k, 2),
        rotated_half=helpers.rotate_half(q))
    positions = torch.arange(4)[None].expand(2, -1)
    values['rope'] = helpers.apply_rope(q.transpose(1, 2), positions)
    cos, sin = model.rotary_emb(q.transpose(1, 2).flatten(2), positions)
    values['rotary_query'], values[
        'rotary_key'] = helpers.apply_rotary_pos_emb(q, k, cos, sin)
    values['sinusoidal'] = helpers.create_sinusoidal_pos_embedding(
        inputs['time'], 16, min_period=4e-3, max_period=4.0)
    values['residual'] = helpers.gated_residual(q, q * 0.25, q.sigmoid())
    for name in ('eager_attention_forward', 'sdpa_attention_forward',
                 'sdpa_math_fp32_attention_forward'):
        output, weights = getattr(helpers, name)(module, q, k, v, mask,
                                                 8**-0.5)
        values[name] = output
        if weights is not None:
            values[name + '.weights'] = weights
    loss = sum(value.float().square().mean() for value in values.values()
               if value.requires_grad)
    for name, value in values.items():
        _record(results, name, value)
    _record(results, 'loss', loss)
    loss.backward()
    for name, value in (('query', q), ('key', k), ('value', v)):
        _record(results, name + '.gradient', value.grad)


def execute_case(model, spec, inputs, shared=False):
    import torch

    results = {}
    original_inputs = _clone_tensors(inputs)
    rng = torch.random.get_rng_state().clone()
    helpers = importlib.import_module(
        'fluxvla_export.components.tensor_utils'
        if shared else 'fluxvla.engines.utils.model_utils')
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=spec['amp']):
        if spec['kind'] == 'policy':
            _execute_policy(model, inputs, results)
        elif spec['kind'] == 'gemma':
            _execute_gemma(model, inputs, results, helpers)
        elif spec['kind'] == 'helpers':
            _execute_helpers(model, inputs, results, helpers)
        else:
            name = 'images' if spec['kind'] == 'siglip' else 'hidden'
            value = inputs[name].clone().requires_grad_()
            output = model(value)
            loss = output.square().mean()
            _record(results, 'output', output)
            _record(results, 'loss', loss)
            loss.backward()
            _record(results, 'input_gradient', value.grad)
    torch.testing.assert_close(
        torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for name, value in inputs.items():
        torch.testing.assert_close(
            value, original_inputs[name], rtol=0, atol=0)
    parameters = list(model.named_parameters())
    gradients, absent_gradients = [], []
    for name, parameter in parameters:
        if parameter.grad is None:
            absent_gradients.append(name)
        else:
            gradients.append(name)
            _record(results, 'gradient.' + name, parameter.grad)
    return results, dict(
        parameter_keys=[name for name, _ in parameters],
        requires_grad_keys=[
            name for name, value in parameters if value.requires_grad
        ],
        gradient_keys=gradients,
        no_gradient_keys=absent_gradients,
        rng_preserved=True,
        fixture_inputs_preserved=True)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args],
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _read_json(path):
    return json.loads(Path(path).read_text())


def _bootstrap(repo):
    for key, value in dict(
            CUDA_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',
            WANDB_MODE='disabled').items():
        os.environ[key] = value
    # Keep unrelated installed packages, but never resolve a model via the
    # script's own checkout or an ambient PYTHONPATH from another checkout.
    sys.path[:] = [
        entry for entry in sys.path
        if entry and not ((Path(entry).resolve() / 'fluxvla').is_dir() or
                          (Path(entry).resolve() / 'fluxvla_export').is_dir())
    ]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'packages/fluxvla-export/src'))
    os.chdir(repo)
    runtime = runpy.run_path(str(repo / 'test/_cpu_model_runtime.py'))
    cleanup = runtime['bootstrap']()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return cleanup


def _provenance(repo):
    import torch
    import transformers

    sources = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == 'fluxvla' or name.startswith('fluxvla.') or name
                == 'fluxvla_export' or name.startswith('fluxvla_export.')):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue  # Existing CPU bootstrap's explicit namespace packages.
        path = Path(filename).resolve()
        if not path.is_relative_to(repo):
            raise AssertionError(
                f'Wrong-repository import: {name} resolved to {path}, '
                f'not {repo}')
        sources[str(path.relative_to(repo))] = _hash(path)
    sources['test/_cpu_model_runtime.py'] = _hash(repo /
                                                  'test/_cpu_model_runtime.py')
    return dict(
        repo=str(repo),
        commit=_git(repo, 'rev-parse', 'HEAD'),
        working_tree_status=_git(repo, 'status', '--porcelain'),
        source_sha256=sources,
        probe_sha256=_hash(__file__),
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            resolved_executable=str(Path(sys.executable).resolve()),
            prefix=str(Path(sys.prefix).resolve()),
            torch=torch.__version__,
            transformers=transformers.__version__,
            dependency_versions={
                name: importlib.metadata.version(name)
                for name in ('safetensors', 'torchvision', 'timm', 'numpy',
                             'mmengine')
            },
            platform=platform.platform(),
            machine=platform.machine(),
            threads=torch.get_num_threads(),
            deterministic=True))


def _tensor_layout(values):
    return {
        name: dict(shape=list(value.shape), dtype=str(value.dtype))
        for name, value in values.items()
    }


def _save_case(path, spec, model, state, inputs, results, details):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=False)
    save_file(_clone_tensors(state), str(path / 'state.safetensors'))
    save_file(_clone_tensors(inputs), str(path / 'inputs.safetensors'))
    save_file(results, str(path / 'results.safetensors'))
    metadata = dict(
        spec=spec,
        state_keys=list(model.state_dict()),
        state_layout=_tensor_layout(state),
        result_layout=_tensor_layout(results),
        **details)
    metadata['file_sha256'] = {
        name: _hash(path / name)
        for name in ('state.safetensors', 'inputs.safetensors',
                     'results.safetensors')
    }
    _write_json(path / 'case.json', metadata)
    return metadata


def _compare_tensors(actual, expected, label, rtol, atol):
    import torch

    if set(actual) != set(expected):
        raise AssertionError(
            f'{label}: tensor keys differ: '
            f'missing={sorted(set(expected) - set(actual))}, '
            f'unexpected={sorted(set(actual) - set(expected))}')
    max_error = 0.0
    groups = {}
    for name, value in actual.items():
        reference = expected[name]
        torch.testing.assert_close(
            value,
            reference,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: f'{label}/{name}: {msg}')
        if value.numel() and value.is_floating_point():
            error = (value.double() - reference.double()).abs().max().item()
            max_error = max(max_error, error)
            if name.startswith('gradient.'):
                group = 'parameter_gradients'
            elif name.startswith('prefix.cache.'):
                group = 'prefix_cache'
            elif name.startswith('prefix.'):
                group = name.replace('.', '_')
            elif name in ('loss', 'training_velocity', 'single_step_velocity',
                          'final_action'):
                group = name
            else:
                group = 'other_outputs'
            groups[group] = max(groups.get(group, 0.0), error)
    return dict(
        max_absolute_error=max_error, max_absolute_error_by_group=groups)


def run(args):
    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f'Refusing to overwrite probe artifacts: {output}')
    if output.is_relative_to(repo) or output.is_relative_to(
            Path(__file__).resolve().parents[1]):
        raise ValueError(
            'Probe tensor artifacts must live outside both source checkouts.')
    if args.command == 'capture':
        commit = _git(repo, 'rev-parse', 'HEAD')
        if not commit.startswith(args.expected_commit):
            raise AssertionError(
                f'Original commit mismatch: {commit} != {args.expected_commit}'
            )
        _git(repo, 'diff', '--exit-code', 'HEAD', '--', 'fluxvla',
             'test/_cpu_model_runtime.py')
        baseline_manifest = None
    else:
        args.baseline = args.baseline.resolve(strict=True)
        baseline_manifest = _read_json(args.baseline / 'manifest.json')
        if baseline_manifest['failures']:
            raise AssertionError('Original baseline contains failures; '
                                 'inspect its manifest before comparison.')
        if baseline_manifest['schema_version'] != SCHEMA_VERSION:
            raise AssertionError(
                'Probe schema differs from the stored baseline.')
        if repo == Path(baseline_manifest['provenance']['repo']):
            raise ValueError(
                'Compare must run the migrated repo, not the original repo.')
        if baseline_manifest['cases'] != case_specs():
            raise AssertionError(
                'Case specifications changed since original capture.')
        expected_completed = [
            dict(case=spec['name'], target='training', status='passed')
            for spec in case_specs()
        ]
        if (baseline_manifest['command'] != 'capture'
                or baseline_manifest['completed'] != expected_completed):
            raise AssertionError(
                'Baseline is not a complete original capture.')
    output.mkdir(parents=True)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        command=args.command,
        seed=SEED,
        cases=case_specs(),
        completed=[],
        failures=[])
    if baseline_manifest is not None:
        manifest['baseline'] = dict(
            path=str(args.baseline),
            manifest_sha256=_hash(args.baseline / 'manifest.json'),
            old_commit=baseline_manifest['provenance']['commit'])
        manifest['tolerance'] = dict(rtol=args.rtol, atol=args.atol)
    cleanup = _bootstrap(repo)
    import torch
    from safetensors.torch import load_file

    try:
        provenance = _provenance(repo)
        if baseline_manifest is not None:
            if provenance['environment'] != baseline_manifest['provenance'][
                    'environment']:
                raise AssertionError('Original and migrated runs must use '
                                     'the exact same environment.')
        for spec in case_specs():
            targets = ('training', 'shared') if (
                args.command == 'compare'
                and spec['kind'] != 'policy') else ('training', )
            for target in targets:
                label = spec['name'] + '/' + target
                try:
                    torch.manual_seed(SEED)
                    model = build_case_model(spec, shared=target == 'shared')
                    if baseline_manifest is None:
                        state = _clone_tensors(model.state_dict())
                        inputs = fixture_inputs(spec['kind'])
                        reference_metadata = None
                    else:
                        baseline_path = args.baseline / spec[
                            'name'] / 'training'
                        reference_metadata = _read_json(baseline_path /
                                                        'case.json')
                        for filename, sha256 in reference_metadata[
                                'file_sha256'].items():
                            if _hash(baseline_path / filename) != sha256:
                                raise AssertionError(
                                    f'Baseline artifact hash mismatch: '
                                    f'{baseline_path / filename}')
                        state = load_file(
                            str(baseline_path / 'state.safetensors'))
                        inputs = load_file(
                            str(baseline_path / 'inputs.safetensors'))
                        # Check constructor state/layout before loading;
                        # strict=True below separately proves complete
                        # checkpoint compatibility.
                        if list(model.state_dict()
                                ) != reference_metadata['state_keys']:
                            raise AssertionError(
                                f'{label}: complete state key order changed')
                        _compare_tensors(model.state_dict(), state,
                                         label + '/initialization', 0, 0)
                        model.load_state_dict(state, strict=True)
                    results, details = execute_case(
                        model, spec, inputs, shared=target == 'shared')
                    metadata = _save_case(output / spec['name'] / target, spec,
                                          model, state, inputs, results,
                                          details)
                    comparison = dict(
                        case=spec['name'], target=target, status='passed')
                    if reference_metadata is not None:
                        for name in ('parameter_keys', 'requires_grad_keys',
                                     'gradient_keys', 'no_gradient_keys',
                                     'state_layout', 'result_layout'):
                            if metadata[name] != reference_metadata[name]:
                                raise AssertionError(
                                    f'{label}: {name} differs from original')
                        expected = load_file(
                            str(baseline_path / 'results.safetensors'))
                        comparison.update(
                            _compare_tensors(results, expected, label,
                                             args.rtol, args.atol))
                    manifest['completed'].append(comparison)
                    print(json.dumps(comparison), flush=True)
                except Exception as exc:
                    # Preserve ALL actual original failures, then exit nonzero.
                    # A failed case is never converted into a golden or a skip.
                    failure = dict(
                        case=spec['name'],
                        target=target,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc())
                    manifest['failures'].append(failure)
                    print(json.dumps(failure), file=sys.stderr, flush=True)
                manifest['provenance'] = _provenance(repo)
                _write_json(output / 'manifest.json', manifest)
    finally:
        manifest['provenance'] = _provenance(repo)
        _write_json(output / 'manifest.json', manifest)
        cleanup()
    if manifest['failures']:
        raise RuntimeError(f'{len(manifest["failures"])} probe cases failed; '
                           f'inspect {output / "manifest.json"}')
    print(
        json.dumps(
            dict(
                status='passed',
                manifest=str(output / 'manifest.json'),
                completed=len(manifest['completed']))),
        flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser(
        'capture',
        help='Capture real original state, inputs, outputs and gradients.')
    compare = commands.add_parser(
        'compare',
        help=('Compare migrated training and shared components '
              'to original artifacts.'))
    for command in (capture, compare):
        command.add_argument('--repo', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    capture.add_argument('--expected-commit', required=True)
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument('--rtol', type=float, default=0.0)
    compare.add_argument('--atol', type=float, default=0.0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
