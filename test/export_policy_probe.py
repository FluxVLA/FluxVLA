# Copyright 2026 Limx Dynamics
"""Capture original PI0.5 policy evidence, then compare in fresh interpreters.

The stable export_components_probe remains the PI0/training/component probe.
This extension additionally compares a separately constructed deployment model
and its real strict safetensors reference loader against ORIGINAL computations.
Artifacts must stay outside either checkout; the original checkout is retained.

    python test/export_policy_probe.py capture --repo /tmp/original \
        --expected-commit FULL_SHA --output /tmp/policy-original
    python test/export_policy_probe.py compare --repo /path/to/FluxVLA \
        --baseline /tmp/policy-original --target training \
        --output /tmp/training
    python test/export_policy_probe.py compare --repo /path/to/FluxVLA \
        --baseline /tmp/policy-original --target standalone \
        --output /tmp/independent
"""

import argparse
import itertools
import json
import os
import sys
import traceback
from pathlib import Path

if __package__:
    from . import export_components_probe as previous
else:
    import export_components_probe as previous

SCHEMA_VERSION = 1
INPUT_NAMES = ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states',
               'noise')
GEMMA_FIELDS = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'head_dim',
                'vocab_size', 'max_position_embeddings', 'hidden_activation',
                'rms_norm_eps', 'rope_theta', 'attention_bias',
                'attention_dropout', 'use_adarms', 'adarms_cond_dim',
                'use_cache', 'pad_token_id', 'bos_token_id', 'eos_token_id')
VISION_FIELDS = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                 'num_attention_heads', 'num_channels', 'image_size',
                 'patch_size', 'hidden_activation', 'layer_norm_eps',
                 'attention_dropout', 'vision_use_head')


def case_specs():
    cases = []
    for amp, flow, residual, protected in itertools.product((False, True),
                                                            repeat=4):
        cfg = previous.tiny_policy_config('PI05FlowMatching', protected)
        cfg.update(
            enable_mixed_precision_training=amp,
            openpi_fp32_flow=flow,
            preserve_fp32_residuals=residual,
            attention_implementation='sdpa',
            rtc_training_config=None)
        # Schema 1 excludes an unused vision pooling head. Construct that exact
        # architecture in the ORIGINAL model; never drop checkpoint keys.
        cfg['vision_backbone']['vision_config']['vision_use_head'] = False
        cases.append(
            dict(
                name=f'pi05-{"bf16" if amp else "fp32"}'
                f'-flow{int(flow)}-residual{int(residual)}'
                f'-protected{int(protected)}',
                kind='policy',
                amp=amp,
                config=cfg))
    return cases


def gemma_document(config):
    # Transformers 5.3 migrates the supplied rope_theta into rope_parameters.
    # Read that actual value; do not substitute a numeric constructor default.
    if config.rope_parameters['rope_type'] != 'default':
        raise AssertionError('Schema 1 cannot represent non-default RoPE')
    return dict(
        type='condition_gemma',
        **{
            name: (config.rope_parameters['rope_theta']
                   if name == 'rope_theta' else getattr(config, name))
            for name in GEMMA_FIELDS
        })


def deployment_document(model, inputs, state_path, inputs_path):
    """Map resolved ORIGINAL training fields, never infer model defaults.

    Architecture is read from original instantiated HF configs and the original
    linear modules; precision is read from the original policy attributes.
    The generated JSON contains every schema field, including disabled flags.
    Synthetic preprocessing descriptions identify this fixture generator, not a
    real tokenizer or normalization pipeline. Their hashes bind that source.
    """
    vision = model.vision_backbone
    backbone, expert = model.llm_backbone.config, model.llm_expert.config
    linear_names = ('projector', 'action_in_proj', 'action_out_proj',
                    'time_mlp_in', 'time_mlp_out')
    architecture = dict(
        type='pi05',
        vision=dict(
            type='siglip',
            **{
                k:
                getattr(vision.vision.config,
                        'hidden_act' if k == 'hidden_activation' else k)
                for k in VISION_FIELDS
            }),
        backbone=gemma_document(backbone),
        expert=gemma_document(expert),
        proj_width=model.proj_width,
        state_dim=inputs['states'].shape[-1],
        action_dim=model.max_action_dim,
        action_horizon=model.n_action_steps,
        state_projection=model.state_proj is not None)
    for name in linear_names:
        layer = getattr(model, name).projector
        architecture[name] = dict(
            type='linear',
            in_dim=layer.in_features,
            out_dim=layer.out_features,
            bias=layer.bias is not None)
    parameter_dtypes = {str(value.dtype) for value in model.parameters()}
    if parameter_dtypes != {'torch.float32'}:
        raise AssertionError(
            f'Original master parameter dtypes: {parameter_dtypes}')
    if (backbone.adarms_fp32 != expert.adarms_fp32
            or backbone.attention_math_fp32 != expert.attention_math_fp32):
        raise AssertionError('Schema 1 requires common Gemma precision flags')
    generator_hash = previous._hash(previous.__file__)
    return dict(
        schema_version=1,
        recipe='pi05-v1',
        checkpoint=dict(
            format='fluxvla-native',
            kind='dense',
            sha256=previous._hash(state_path)),
        inputs=dict(
            format='safetensors',
            sha256=previous._hash(inputs_path),
            tensors={
                name:
                dict(
                    dtype=str(inputs[name].dtype)[6:],
                    shape=list(inputs[name].shape))
                for name in INPUT_NAMES
            }),
        model=architecture,
        inference=dict(
            mode='plain', solver='euler', num_steps=model.num_steps,
            rtc=False),
        precision=dict(
            parameter_dtype='float32',
            autocast_dtype=('bfloat16' if model.enable_mixed_precision_training
                            else None),
            attention_backend=model.attention_implementation,
            flow_fp32=model.openpi_fp32_flow,
            residual_fp32=model.preserve_fp32_residuals,
            vision_stem_fp32=vision.openpi_stem_fp32,
            vision_residual_fp32=vision.preserve_fp32_residuals,
            adarms_fp32=backbone.adarms_fp32,
            attention_math_fp32=backbone.attention_math_fp32,
            backbone_fp32_layers=list(backbone.fp32_layers),
            expert_fp32_layers=list(expert.fp32_layers)),
        preprocessing=dict(
            id='synthetic-original-policy-probe-v1',
            tokenizer_id='fixture-generator-not-a-tokenizer',
            tokenizer_sha256=generator_hash,
            prompt_id='literal-token-fixture-no-prompt-rendering',
            normalization_id='linspace-fixtures-no-normalization',
            normalization_sha256=generator_hash,
            state_semantics='synthetic-continuous-state-unused-by-pi05',
            action_semantics='synthetic-padded-four-dimensional-actions',
            image_transform_id='linspace-packed-two-view-rgb-no-preprocessing',
            view_order=['synthetic-view-0', 'synthetic-view-1'],
            state_in_prompt=False))


def cached_evidence(model, spec, inputs):
    """Exercise original public methods; this contains no replacement math."""
    import torch
    from fluxvla_export.components.tensor_utils import make_att_2d_masks

    results = {}

    def record(name, value):
        previous._record(results, name, value)

    state = previous._clone_tensors(model.state_dict())
    saved_inputs = previous._clone_tensors(inputs)
    rng = torch.random.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    values = {name: inputs[name].clone() for name in INPUT_NAMES}
    values['images'].requires_grad_()
    values['noise'].requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=spec['amp']):
        prefix, pad, blocks = model.embed_prefix(**values)
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
            use_cache=True)
        for name, value in (('embeddings', prefix), ('padding', pad),
                            ('blocks', blocks), ('hidden',
                                                 encoded.last_hidden_state)):
            record('prefix.' + name, value)
        cache = encoded.past_key_values
        previous._record_cache(results, cache, 'prefix.cache')
        loss = 0
        times = dict(
            scalar=inputs['time'],
            per_token=inputs['time'][:, None] *
            torch.tensor([[0.0, 0.5, 1.0]]))
        for mode, time in times.items():
            record(f'{mode}.time', time)
            suffix = model.embed_suffix(values['states'], values['noise'],
                                        time)
            for name, value in zip(
                ('embeddings', 'padding', 'blocks', 'condition'), suffix):
                record(f'{mode}.suffix.{name}', value)
            velocity = model.denoise_step(values['states'], pad, cache,
                                          values['noise'], time)
            record(f'{mode}.velocity', velocity)
            loss = loss + velocity.float().square().mean()
        record('cached_loss', loss)
        loss.backward()
        after_cache = {}
        previous._record_cache(after_cache, cache, 'prefix.cache')
        previous._compare_tensors(after_cache,
                                  {k: results[k]
                                   for k in after_cache},
                                  'immutable prefix cache', 0, 0)
        with torch.no_grad():
            inference = {name: inputs[name] for name in INPUT_NAMES}
            action = model.predict_action(**previous._clone_tensors(inference))
            repeated = model.predict_action(
                **previous._clone_tensors(inference))
            torch.testing.assert_close(action, repeated, rtol=0, atol=0)
            record('final_action', action)
    gradient_keys, no_gradient_keys = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            no_gradient_keys.append(name)
        else:
            gradient_keys.append(name)
            record('gradient.' + name, parameter.grad)
    for name in ('images', 'noise'):
        record('input_gradient.' + name, values[name].grad)
    previous._compare_tensors(model.state_dict(), state, 'immutable state', 0,
                              0)
    previous._compare_tensors(inputs, saved_inputs, 'immutable fixtures', 0, 0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), rng, rtol=0, atol=0)
    return results, dict(
        gradient_keys=gradient_keys,
        no_gradient_keys=no_gradient_keys,
        state_preserved=True,
        cache_preserved=True,
        rng_preserved=True,
        fixture_inputs_preserved=True)


def provenance(repo):
    result = previous._provenance(repo)
    result['policy_probe_sha256'] = previous._hash(__file__)
    return result


def check_original(manifest):
    source = manifest['provenance']
    repo = Path(source['repo']).resolve(strict=True)
    if previous._git(repo, 'rev-parse', 'HEAD') != source['commit']:
        raise AssertionError('Retained original commit changed')
    if previous._git(repo, 'status', '--porcelain'):
        raise AssertionError('Retained original worktree is no longer clean')
    for name, digest in source['source_sha256'].items():
        if previous._hash(repo / name) != digest:
            raise AssertionError(f'Retained original source changed: {name}')


def bootstrap_standalone(repo):
    # No training CPU namespace bootstrap in this interpreter.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
                'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    sys.path[:] = [
        entry for entry in sys.path
        if entry and not ((Path(entry).resolve() / 'fluxvla').is_dir() or
                          (Path(entry).resolve() / 'fluxvla_export').is_dir())
    ]
    sys.path.insert(0, str(repo / 'packages/fluxvla-export/src'))
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return lambda: None


def run(args):
    repo, output = args.repo.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite probe output: {output}')
    if any(
            output.is_relative_to(root)
            for root in (repo, Path(__file__).resolve().parents[1])):
        raise ValueError('Tensor artifacts must be outside source checkouts')
    capture = args.command == 'capture'
    target = 'training' if capture else args.target
    baseline = None
    if capture:
        if previous._git(repo, 'rev-parse', 'HEAD') != args.expected_commit:
            raise AssertionError('Expected exact full original commit SHA')
        if previous._git(repo, 'status', '--porcelain'):
            raise AssertionError('Original checkout must be entirely clean')
    else:
        args.baseline = args.baseline.resolve(strict=True)
        baseline = previous._read_json(args.baseline / 'manifest.json')
        if (baseline['schema_version'] != SCHEMA_VERSION
                or baseline['command'] != 'capture' or baseline['failures']
                or baseline['cases'] != case_specs()
                or len(baseline['completed']) != len(case_specs())):
            raise AssertionError('Incomplete or incompatible original capture')
        check_original(baseline)
        if repo == Path(baseline['provenance']['repo']):
            raise ValueError(
                'Comparison must not import the original checkout')
        if (baseline['provenance']['policy_probe_sha256']
                != previous._hash(__file__)
                or baseline['provenance']['probe_sha256'] != previous._hash(
                    previous.__file__)):
            raise AssertionError(
                'Policy probe or reused helper changed after original capture')
    output.mkdir(parents=True)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        command=args.command,
        target=target,
        seed=previous.SEED,
        cases=case_specs(),
        completed=[],
        failures=[])
    if baseline:
        manifest['baseline'] = dict(
            path=str(args.baseline),
            manifest_sha256=previous._hash(args.baseline / 'manifest.json'))
    cleanup = bootstrap_standalone(
        repo) if target == 'standalone' else previous._bootstrap(repo)
    import torch
    from safetensors.torch import load_file, save_file
    from fluxvla_export.config import load_config
    from fluxvla_export.checkpoint import load_checkpoint
    try:
        if baseline and provenance(
                repo)['environment'] != baseline['provenance']['environment']:
            raise AssertionError('Use the exact original capture environment')
        for spec in case_specs():
            label = spec['name'] + '/' + target
            try:
                torch.manual_seed(previous.SEED)
                if capture:
                    model = previous.build_case_model(spec)
                    state = previous._clone_tensors(model.state_dict())
                    inputs = previous.fixture_inputs('policy')
                    expected = metadata = None
                else:
                    source = args.baseline / spec['name']
                    metadata = previous._read_json(source / 'case.json')
                    for name, digest in metadata['file_sha256'].items():
                        if previous._hash(source / name) != digest:
                            raise AssertionError(
                                f'Original artifact changed: {source / name}')
                    inputs = load_file(str(source / 'inputs.safetensors'))
                    expected = load_file(str(source / 'results.safetensors'))
                    if target == 'standalone':
                        from fluxvla_export.recipes.pi05.model import (
                            build_pi05_model)
                        model = build_pi05_model(
                            load_config(
                                source / 'deployment.json',
                                recipe='pi05-v1')).eval()
                    else:
                        model = previous.build_case_model(spec)
                    state = load_checkpoint(
                        source / 'state.safetensors',
                        expected_sha256=metadata['file_sha256']
                        ['state.safetensors'],
                        expected_state=model.state_dict())
                    if list(model.state_dict()) != metadata['state_keys']:
                        raise AssertionError(
                            f'{label}: full state order changed')
                    previous._compare_tensors(model.state_dict(), state,
                                              label + '/initialization', 0, 0)
                    model.load_state_dict(state, strict=True)
                results, details = {}, {}
                if target == 'training':
                    results, details = previous.execute_case(
                        model, spec, inputs)
                cached, cached_details = cached_evidence(model, spec, inputs)
                results.update({
                    'cached.' + name: value
                    for name, value in cached.items()
                })
                details['cached'] = cached_details
                if target == 'standalone':
                    from fluxvla_export.recipes.pi05.reference import (
                        prepare_reference)
                    rng = torch.random.get_rng_state().clone()
                    reference = prepare_reference(
                        config=source / 'deployment.json',
                        checkpoint=source / 'state.safetensors',
                        inputs=source / 'reference-inputs.safetensors',
                        device='cpu')
                    original_state = previous._clone_tensors(
                        reference.model.state_dict())
                    original_inputs = previous._clone_tensors(reference.inputs)
                    previous._compare_tensors(original_state, state,
                                              label + '/strict-loaded-state',
                                              0, 0)
                    previous._compare_tensors(
                        original_inputs,
                        {name: inputs[name]
                         for name in INPUT_NAMES},
                        label + '/strict-loaded-fixtures', 0, 0)
                    for _ in range(2):
                        actions = reference.run_reference()
                        previous._compare_tensors(
                            actions,
                            {'actions': expected['cached.final_action']},
                            label + '/strict-reference', 0, 0)
                    previous._compare_tensors(reference.model.state_dict(),
                                              original_state,
                                              label + '/reference-state', 0, 0)
                    previous._compare_tensors(reference.inputs,
                                              original_inputs,
                                              label + '/reference-fixtures', 0,
                                              0)
                    torch.testing.assert_close(
                        torch.random.get_rng_state(), rng, rtol=0, atol=0)
                    before_forward = previous._clone_tensors(inputs)
                    with torch.no_grad(), torch.autocast(
                            'cpu', dtype=torch.bfloat16, enabled=spec['amp']):
                        forward = model(*(inputs[name]
                                          for name in INPUT_NAMES))
                    torch.testing.assert_close(
                        forward,
                        expected['cached.final_action'],
                        rtol=0,
                        atol=0)
                    previous._compare_tensors(
                        inputs, before_forward,
                        label + '/six-input-forward-fixtures', 0, 0)
                    previous._compare_tensors(
                        model.state_dict(), state,
                        label + '/six-input-forward-state', 0, 0)
                    torch.testing.assert_close(
                        torch.random.get_rng_state(), rng, rtol=0, atol=0)
                    if forward.data_ptr() == inputs['noise'].data_ptr():
                        raise AssertionError(
                            'Six-input forward aliases fixture noise')
                    forbidden = sorted(
                        name for name in sys.modules
                        if name == 'fluxvla' or name.startswith('fluxvla.')
                        or name in ('mmengine', 'timm', 'accelerate'))
                    if forbidden:
                        raise AssertionError(
                            f'Training dependencies imported: {forbidden}')
                    details[
                        'strict_reference_repeated_without_mutation'] = True
                case_path = output / spec['name']
                saved = previous._save_case(case_path, spec, model, state,
                                            inputs, results, details)
                if capture:
                    fixture = {name: inputs[name] for name in INPUT_NAMES}
                    save_file(fixture,
                              str(case_path / 'reference-inputs.safetensors'))
                    document = deployment_document(
                        model, inputs, case_path / 'state.safetensors',
                        case_path / 'reference-inputs.safetensors')
                    previous._write_json(case_path / 'deployment.json',
                                         document)
                    load_config(
                        case_path / 'deployment.json', recipe='pi05-v1')
                    for name in ('reference-inputs.safetensors',
                                 'deployment.json'):
                        saved['file_sha256'][name] = previous._hash(case_path /
                                                                    name)
                    saved['deployment_mapping'] = dict(
                        architecture=(
                            'original vision.vision.config / '
                            'llm_backbone.config / llm_expert.config / '
                            'named LinearProjector.projector'),
                        dimensions=
                        ('original proj_width/max_action_dim/n_action_steps; '
                         'states fixture last dimension'),
                        precision=('original policy/vision/Gemma attributes; '
                                   'FP32 parameter dtypes checked'),
                        preprocessing=(
                            'synthetic fixture generator source hash, '
                            'not production preprocessing'))
                    previous._write_json(case_path / 'case.json', saved)
                comparison = dict(
                    case=spec['name'], target=target, status='passed')
                if expected is not None:
                    selected = expected if target == 'training' else {
                        k: v
                        for k, v in expected.items() if k.startswith('cached.')
                    }
                    comparison.update(
                        previous._compare_tensors(results, selected, label, 0,
                                                  0))
                    if details['cached'] != metadata['cached']:
                        raise AssertionError(
                            f'{label}: cached gradient membership changed')
                    if target == 'training':
                        for name in ('parameter_keys', 'requires_grad_keys',
                                     'gradient_keys', 'no_gradient_keys'):
                            if details[name] != metadata[name]:
                                raise AssertionError(
                                    f'{label}: {name} changed')
                manifest['completed'].append(comparison)
                print(json.dumps(comparison), flush=True)
            except Exception as error:
                manifest['failures'].append(
                    dict(
                        case=spec['name'],
                        target=target,
                        error_type=type(error).__name__,
                        message=str(error),
                        traceback=traceback.format_exc()))
                print(
                    json.dumps(manifest['failures'][-1]),
                    file=sys.stderr,
                    flush=True)
            manifest['provenance'] = provenance(repo)
            previous._write_json(output / 'manifest.json', manifest)
    finally:
        manifest['provenance'] = provenance(repo)
        previous._write_json(output / 'manifest.json', manifest)
        cleanup()
    check_original(manifest if capture else baseline)
    if manifest['failures']:
        raise RuntimeError(f'{len(manifest["failures"])} cases failed: '
                           f'{output / "manifest.json"}')
    print(
        json.dumps(
            dict(
                status='passed',
                completed=len(manifest['completed']),
                manifest=str(output / 'manifest.json'))),
        flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture')
    compare = commands.add_parser('compare')
    for command in (capture, compare):
        command.add_argument('--repo', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    capture.add_argument('--expected-commit', required=True)
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument(
        '--target', choices=('training', 'standalone'), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
