"""Minimal fixed-configuration equivalent of LingBot's pretuned decorator."""

from __future__ import annotations

import triton


def algo_key_scaled(names, scales, rest_key=None):
    """Retain the upstream decorator contract for the grouped kernels."""

    def key_maker(**kwargs):
        values = []
        for name, scale in zip(names, scales, strict=True):
            value = f"{name.lower()}{kwargs[name] // scale}"
            if scale != 1:
                value += f"x{scale}"
            values.append(value)
        if rest_key is not None:
            values.extend(str(kwargs[key]) for key in rest_key)
        return "_".join(values)

    return key_maker


class _FixedConfigKernel(triton.KernelInterface):
    def __init__(self, fn, fallback):
        self.fn = fn
        self.fallback = fallback

    def run(self, *args, **kwargs):
        return self.fn.run(*args, **kwargs, **self.fallback)


def pretuned(*, algo_key=None, fallback=None):
    """Use the same fallback tile as official LingBot when no tuning DB ships."""
    del algo_key
    if fallback is None:
        raise ValueError("pretuned requires a fallback configuration")

    def decorator(fn):
        return _FixedConfigKernel(fn, fallback)

    return decorator
