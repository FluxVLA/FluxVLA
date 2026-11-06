"""Driver and Ray worker registration through RLINF_EXT_MODULE."""


def build_pi05(cfg, torch_dtype=None):
    from fluxvla.models.vlas.pi05_rl_builder import build_pi05_policy

    return build_pi05_policy(cfg, torch_dtype)


def build_smolvla(cfg, torch_dtype=None):
    from fluxvla.models.vlas.smolvla_rl_builder import build_smolvla_policy

    return build_smolvla_policy(cfg, torch_dtype)


def register():
    """Idempotently register the external policy in the current process."""
    from rlinf.models import register_model

    register_model('fluxvla_pi05', build_pi05, category='embodied', force=True)
    register_model(
        'fluxvla_smolvla', build_smolvla, category='embodied', force=True)


register_all = register
