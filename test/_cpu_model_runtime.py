# Copyright 2026 Limx Dynamics
"""Explicit import isolation for real CPU model unit tests, not an app runtime.

FluxVLA's package initializers eagerly import unrelated CUDA-only models and
RADIO globally replaces timm attention. For CPU unit tests ONLY, namespace
packages locate the original source modules without running those initializers.
Registries, builders, model classes and all tensor operations remain unchanged.
This does not test normal ``import fluxvla`` or CUDA integration; the full GPU
suite still uses the production import path in a separate interpreter.
"""

import importlib
import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path


def _export_public(module, package):
    names = getattr(module, '__all__', None)
    if names is None:
        names = [name for name in vars(module) if not name.startswith('_')]
    for name in names:
        setattr(package, name, getattr(module, name))


def bootstrap():
    """Load actual CPU components; return cleanup for this pytest session."""
    import torch
    import transformers

    if torch.cuda.is_available():
        raise RuntimeError('CPU unit mode requires CUDA_VISIBLE_DEVICES="".')
    if transformers.__version__ != '5.3.0':
        raise RuntimeError('CPU model tests require Transformers 5.3.0.')
    if any(name == 'fluxvla' or name.startswith('fluxvla.')
           for name in sys.modules):
        raise RuntimeError('CPU isolation must run before importing FluxVLA.')

    root = Path(__file__).resolve().parents[1]
    packages = (
        'fluxvla',
        'fluxvla.engines',
        'fluxvla.models',
        'fluxvla.models.backbones',
        'fluxvla.models.backbones.llms',
        'fluxvla.models.backbones.visions',
        'fluxvla.models.backbones.vlms',
        'fluxvla.models.heads',
        'fluxvla.models.projectors',
        'fluxvla.models.vlas',
        'fluxvla.tokenizers',
    )
    for name in packages:
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [
            str(root.joinpath(*name.split('.')))
        ]
        module = module_from_spec(spec)
        sys.modules[name] = module
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)

    # The actual engine utils initializer is CPU safe; reuse its real exports.
    utils = importlib.import_module('fluxvla.engines.utils')
    _export_public(utils, sys.modules['fluxvla.engines'])

    modules = (
        'models.backbones.llms.llama2',
        'models.backbones.llms.gemma',
        'models.backbones.llms.qwen2',
        'models.backbones.llms.condition_gemma',
        'models.backbones.visions.siglip_vit',
        'models.backbones.visions.dinosiglip_vit',
        'models.backbones.vlms.paligemma',
        'models.projectors.linear_projector',
        'models.projectors.mlp_projector',
        'models.projectors.fused_projector',
        'models.projectors.domain_aware_linear',
        'models.heads.openvla_head',
        'models.heads.llava_action_head',
        'models.heads.flow_matching_head',
        'models.heads.dit4dit_action_head',
        'tokenizers.action_tokenizer',
        'models.vlas.open_vla',
        'models.vlas.llava_vla',
        'models.vlas.pi0_flowmatching',
        'models.vlas.pi05_flowmatching',
    )
    for name in modules:
        module = importlib.import_module('fluxvla.' + name)
        _export_public(module, sys.modules[module.__package__])

    def cleanup():
        for name in tuple(sys.modules):
            if name == 'fluxvla' or name.startswith('fluxvla.'):
                del sys.modules[name]

    return cleanup
