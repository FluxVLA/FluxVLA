"""Test-only compatibility for the official LingBot launcher."""

try:
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModel
except Exception:
    pass

try:
    from lerobot.datasets import utils as _utils
    from lerobot.datasets import io_utils as _io_utils

    if not hasattr(_utils, "hf_transform_to_torch"):
        _utils.hf_transform_to_torch = _io_utils.hf_transform_to_torch
except Exception:
    pass
