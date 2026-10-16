"""Validated LIBERO recipe boundaries, independent of model execution."""


def validate_config(cfg, phase):
    if cfg.env[phase].task_suite_name != 'libero_10':
        raise ValueError('The LIBERO adapter currently supports LIBERO-10')
