# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real tiny PI0.5 fixtures and hash-bound safetensors for package tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from fluxvla_export.config import load_config
from fluxvla_export.recipes.pi05.model import build_pi05_model
from test_config import config_document


def write_config(path: Path, document: dict):
    path.write_text(json.dumps(document), encoding='utf-8')
    return load_config(path, recipe='pi05-v1')


def fixture_tensors(config) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device='cpu').manual_seed(9821)
    tensors = {}
    for spec in config.input_specs:
        dtype = getattr(torch, spec.dtype)
        if dtype == torch.bool:
            value = torch.ones(spec.shape, dtype=dtype)
            # Two batch elements, incomplete views and left/right padding.
            if spec.name == 'img_masks':
                value[0, -1] = False
            else:
                value[0, -1] = False
                value[1, 0] = False
        elif dtype == torch.int64:
            value = torch.randint(
                0, config.model.vocab_size, spec.shape, generator=generator)
        else:
            value = torch.randn(spec.shape, generator=generator).to(dtype)
        tensors[spec.name] = value
    return tensors


class Artifacts:
    """A real constructed model, checkpoint, fixture and matching JSON."""

    def __init__(self, directory: Path, document: dict | None = None):
        self.document = config_document() if document is None else document
        self.config_path = directory / 'deployment.json'
        self.checkpoint_path = directory / 'weights.safetensors'
        self.inputs_path = directory / 'inputs.safetensors'
        self.config = write_config(self.config_path, self.document)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7364)
            self.model = build_pi05_model(self.config)
        self.tensors = fixture_tensors(self.config)
        self.save_checkpoint(self.model.state_dict())
        self.save_inputs(self.tensors)

    def _bind(self, key: str, path: Path) -> None:
        self.document[key]['sha256'] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        self.config = write_config(self.config_path, self.document)

    def save_checkpoint(self, tensors) -> None:
        save_file(dict(tensors), self.checkpoint_path)
        self._bind('checkpoint', self.checkpoint_path)

    def save_inputs(self, tensors) -> None:
        save_file(dict(tensors), self.inputs_path)
        self._bind('inputs', self.inputs_path)

    def prepare(self, *, device='cpu'):
        from fluxvla_export.recipes.pi05.reference import prepare_reference

        return prepare_reference(
            config=self.config_path,
            checkpoint=self.checkpoint_path,
            inputs=self.inputs_path,
            device=device)
