def index_first_axis(*args, **kwargs):
    raise RuntimeError("flash-attn shim called; use eager attention for diagnostics")


def pad_input(*args, **kwargs):
    raise RuntimeError("flash-attn shim called; use eager attention for diagnostics")


def unpad_input(*args, **kwargs):
    raise RuntimeError("flash-attn shim called; use eager attention for diagnostics")
