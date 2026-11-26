"""Export inference artifacts without modifying resumable RL checkpoints."""

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch


def _read_policy(checkpoint):
    source = Path(checkpoint).expanduser().resolve()
    if source.is_dir():
        source = source / 'actor/model_state_dict/full_weights.pt'
    if not source.is_file():
        raise FileNotFoundError(f'Full actor checkpoint not found: {source}')
    try:
        state = torch.load(
            source, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError:
        # Older torch.save files do not support memory mapping.
        state = torch.load(source, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        state = state.get('model', state.get('state_dict', state))
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a non-empty full actor state dict')
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in state.items()):
        raise ValueError('Full actor state dict must map names to tensors')
    if any('_flat_param' in key.split('.') for key in state):
        raise ValueError('Export requires consolidated full_weights.pt')
    policy = {
        key: value
        for key, value in state.items() if not key.startswith('value_head.')
    }
    if not policy:
        raise ValueError('Checkpoint contains no policy weights')
    return source, policy, len(state) - len(policy)


def export_policy(checkpoint, output):
    """Export a native state dict, preserving all policy names and dtypes."""
    destination = Path(output).expanduser().resolve()
    if destination.suffix not in ('.pt', '.pth'):
        raise ValueError('Export path must end in .pt or .pth')
    if destination.exists():
        raise FileExistsError(f'Export already exists: {destination}')
    _, policy, removed = _read_policy(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(policy, stream)
    return len(policy), removed


def _deep_update(target, updates):
    for key, value in dict(updates or {}).items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_native_bundle(checkpoint, output_dir, rl_cfg):
    """Atomically publish a portable SmolVLA LIBERO inference directory.

    The complete RL checkpoint stays untouched. Configuration comes from the
    actual RL run, including architecture overrides and sampling settings.
    """
    from mmengine import Config
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    from fluxvla.models.vlas.smolvla_rl_builder import resolve_sft_artifacts

    cfg = OmegaConf.to_container(
        rl_cfg,
        resolve=True) if OmegaConf.is_config(rl_cfg) else copy.deepcopy(rl_cfg)
    actor = cfg['actor']['model']
    if actor['model_type'] != 'fluxvla_smolvla':
        raise ValueError(
            'Native bundles currently support SmolVLA LIBERO only')
    options = actor['fluxvla']
    adapter = options['observation_adapter']
    if adapter.get('type') != 'LiberoRLAdapter':
        raise ValueError('Expected LiberoRLAdapter for native bundle export')
    if adapter.get('flux_cfg') is not None:
        raise ValueError(
            'Custom adapter flux_cfg cannot be exported as a native bundle')
    compute_dtype = options.get('compute_dtype', 'bf16')
    if compute_dtype not in ('bf16', 'fp32'):
        raise ValueError('Expected compute_dtype=bf16 or fp32')
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Export already exists: {destination}')
    stats_path, tokenizer = resolve_sft_artifacts(
        actor['model_path'],
        norm_stats_path=options.get('norm_stats_path'),
        tokenizer_path=options.get('tokenizer_path'))
    tokenizer = Path(adapter.get('tokenizer_path')
                     or tokenizer).expanduser().resolve()
    if not (tokenizer / 'tokenizer_config.json').is_file():
        raise FileNotFoundError(f'Tokenizer artifacts missing: {tokenizer}')
    stats = adapter.get('norm_stats')
    if stats is None:
        stats = json.loads(stats_path.read_text())
    elif isinstance(stats, str):
        stats = json.loads(Path(stats).expanduser().read_text())
    flux = Config.fromfile(
        str(Path(options['config_path']).expanduser().resolve()))
    model = copy.deepcopy(flux.model.to_dict())
    model['num_steps'] = int(actor['num_steps'])
    _deep_update(model, options.get('model_overrides'))
    if options.get('action_horizon') is not None:
        model['chunk_size'] = int(options['action_horizon'])
    if model.get('type') != 'SmolVLAFlowMatching':
        raise ValueError('Expected native SmolVLAFlowMatching model')
    model.update(
        pretrained_name_or_path=None, name_mapping=None, strict_mapping=True)
    # Match the RL builder's .float() before loading the checkpoint.
    model['vlm_backbone']['torch_dtype'] = 'float32'
    model['llm_expert']['torch_dtype'] = 'float32'
    evaluation = copy.deepcopy(flux.eval.to_dict())
    evaluation.update(
        model_build_dtype='float32',
        enable_mixed_precision_training=compute_dtype == 'bf16',
        mixed_precision_dtype='bf16',
        eval_chunk_size=int(actor['num_action_chunks']),
        norm_stats_key=flux.train_dataloader.dataset.statistic_name,
        norm_stats_path='{{ fileDirname }}/dataset_statistics.json',
        feishu_report=False)
    if int(actor['action_dim']) != 7:
        raise ValueError('LIBERO native evaluation requires action_dim=7')
    if evaluation['norm_stats_key'] not in stats:
        raise ValueError('Missing SFT normalization statistics for LIBERO')
    if evaluation['dataset'].get('type') != 'LiberoParquetEvalDataset':
        raise ValueError('Expected LiberoParquetEvalDataset')
    evaluation['dataset']['state_dtype'] = ('bfloat16' if compute_dtype
                                            == 'bf16' else 'float32')
    # These pipelines are the ones the RL adapter builds from the Flux config.
    for transform in evaluation['dataset']['transforms']:
        if transform['type'] == 'LiberoPromptFromInputs':
            transform['tokenizer'][
                'model_path'] = '{{ fileDirname }}/tokenizer'
    source, policy, removed = _read_policy(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix='.native_export-', dir=destination.parent))
    try:
        # Clone shared storage so safetensors retains every native key.
        save_file({k: v.contiguous().clone()
                   for k, v in policy.items()},
                  str(staging / 'model.safetensors'))
        (staging / 'dataset_statistics.json'
         ).write_text(json.dumps(stats, indent=2) + '\n')
        shutil.copytree(tokenizer, staging / 'tokenizer', symlinks=False)
        Config(dict(model=model,
                    eval=evaluation)).dump(str(staging / 'eval_config.py'))
        OmegaConf.save(OmegaConf.create(cfg), staging / 'rl_config.yaml')
        manifest = dict(
            format_version=1,
            model_type=actor['model_type'],
            source_checkpoint=str(source),
            initial_sft=str(actor['model_path']),
            policy_tensor_count=len(policy),
            removed_value_head_tensors=removed,
            compute_dtype=compute_dtype,
            num_steps=model['num_steps'],
            action_horizon=model['chunk_size'],
            action_chunk=evaluation['eval_chunk_size'],
            model_sha256=_sha256(staging / 'model.safetensors'),
            stats_sha256=_sha256(staging / 'dataset_statistics.json'),
            resumable=False)
        (staging /
         'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (staging / 'README.md').write_text(
            '# Native FluxVLA evaluation bundle\n\n'
            'Run from an installed FluxVLA repository '
            'with its environment activated.\n'
            'Set BUNDLE to this directory after copying '
            'it to the destination.\n\n'
            '```bash\n'
            'export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1\n'
            'python scripts/eval.py --config "$BUNDLE/eval_config.py" '
            '--ckpt-path "$BUNDLE/model.safetensors" --cfg-options '
            'eval.feishu_report=False\n'
            '```\n\n'
            'For eight GPUs replace `python` with '
            '`torchrun --standalone --nproc_per_node=8`.\n'
            'This is an inference export. '
            'Resume PPO from the original full checkpoint.\n')
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
