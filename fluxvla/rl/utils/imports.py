"""Lazy symbol loading without importing models or simulators eagerly."""

from importlib import import_module


def resolve_symbol(path):
    module, name = path.rsplit('.', 1)
    return getattr(import_module(module), name)
