"""Tensor conversion shared by independent benchmark adapters."""

import numpy as np
import torch


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
