def apply_rotary_emb(x, cos, sin, *args, **kwargs):
    """Eager compatibility implementation for the diagnostic import path."""
    return (x * cos) + (_rotate_half(x) * sin)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return __import__("torch").cat((-x2, x1), dim=-1)
