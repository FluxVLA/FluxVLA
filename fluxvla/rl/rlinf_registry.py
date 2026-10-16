"""Driver and Ray worker registration through RLINF_EXT_MODULE."""

from fluxvla.rl.models.registry import (POLICY_SPECS, get_policy_spec,
                                        resolve_symbol)


def build_registered_policy(cfg, torch_dtype=None):
    """Dispatch by declaration, including future non-flow policy builders."""
    spec = get_policy_spec(cfg.model_type)
    return resolve_symbol(spec.builder)(cfg, torch_dtype)


def build_pi05(cfg, torch_dtype=None):
    from fluxvla.rl.models.builder import build_pi05_policy

    return build_pi05_policy(cfg, torch_dtype)


def build_smolvla(cfg, torch_dtype=None):
    from fluxvla.rl.models.builder import build_smolvla_policy

    return build_smolvla_policy(cfg, torch_dtype)


def register():
    """Idempotently register the external policy in the current process."""
    from rlinf.models import register_model

    from fluxvla.rl.utils.config import register_model_configs

    register_model_configs()
    for model_type in POLICY_SPECS:
        register_model(
            model_type,
            build_registered_policy,
            category='embodied',
            force=True)


register_all = register
