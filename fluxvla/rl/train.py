"""Launch Flux flow policies with the RLinf PPO/FSDP backend."""

import importlib.util
import os
from pathlib import Path

from fluxvla.rl.benchmarks.registry import get_benchmark_spec
from fluxvla.rl.models.registry import get_policy_spec
from fluxvla.rl.utils.imports import resolve_symbol


def prepare_environment():
    """Expose absolute config locations to Hydra and the extension to Ray."""
    from fluxvla.rl.utils.config import register_model_configs

    flux_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec('rlinf')
    if spec is None or spec.origin is None:
        raise ImportError('Install RLinf core into the RL environment; '
                          'see docs/rlinf.md')
    rlinf_root = Path(spec.origin).resolve().parents[1]
    config_dir = Path(
        os.environ.get('RLINF_CONFIG_DIR',
                       rlinf_root / 'examples/embodiment/config'))
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f'RLinf configs missing: {config_dir}. Set RLINF_CONFIG_DIR')
    os.environ['FLUXVLA_ROOT'] = str(flux_root)
    os.environ['RLINF_CONFIG_DIR'] = str(config_dir.resolve())
    # Support a source checkout without requiring an editable install.
    python_path = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if str(flux_root) not in python_path:
        os.environ['PYTHONPATH'] = os.pathsep.join(
            [str(flux_root)] + [path for path in python_path if path])
    extension = 'fluxvla.rl.rlinf_registry'
    existing = os.environ.get('RLINF_EXT_MODULE')
    if existing and existing != extension:
        raise ValueError(
            f'Conflicting RLINF_EXT_MODULE={existing}. Expected {extension}')
    os.environ['RLINF_EXT_MODULE'] = extension
    os.environ['FLUX_RL_EXTERNAL_DISTRIBUTED'] = '1'
    register_model_configs()
    return flux_root


def validate_frontend_cfg(cfg, *, evaluation=False):
    """Check v1 boundaries before validate_cfg starts the Ray cluster."""
    if (cfg.algorithm.loss_type != 'actor_critic'
            or cfg.algorithm.adv_type != 'gae'):
        raise ValueError('FluxVLA RL v1 supports PPO actor_critic + gae only')
    if cfg.algorithm.entropy_bonus != 0:
        raise ValueError('FluxVLA PPO v1 requires entropy_bonus=0; '
                         'entropy remains available via forward')
    if cfg.actor.training_backend != 'fsdp' or cfg.actor.model.joint_logprob:
        raise ValueError(
            'FluxVLA PPO v1 requires FSDP and joint_logprob=False')
    if cfg.runner.get('use_training_pipeline', False) or cfg.runner.get(
            'enable_decoupled_mode', False):
        raise ValueError('FluxVLA RL v1 uses the synchronous embodied runner')
    if cfg.runner.get('only_eval', False) and not evaluation:
        raise ValueError('Use periodic validation or the policy eval API; '
                         'this CLI trains PPO')
    get_policy_spec(cfg.actor.model.model_type).validate_frontend(cfg)
    if (cfg.actor.model.precision != 'fp32'
            or cfg.rollout.model.precision != 'fp32'):
        raise ValueError('Keep model master weights fp32; '
                         'fluxvla.compute_dtype selects network precision')
    if cfg.actor.fsdp_config.mixed_precision.param_dtype != 'fp32':
        raise ValueError('Keep FSDP param_dtype=fp32 to match rollout; '
                         'the policy autocasts BF16 compute')
    if cfg.actor.model.fluxvla.compute_dtype == 'bf16' and (
            cfg.actor.fsdp_config.strategy != 'fsdp2'
            or cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs):
        raise ValueError('BF16 requires FSDP2 with cast_forward_inputs=False '
                         'to preserve FP32 chains')
    if cfg.actor.get('enable_sft_co_train', False) or cfg.actor.model.get(
            'is_lora', False):
        raise ValueError('Mixed SFT and LoRA are not supported in v1')
    if cfg.actor.fsdp_config.gradient_checkpointing:
        raise ValueError('Gradient checkpointing is not supported in v1')
    if cfg.rollout.get('enable_torch_compile', False) or cfg.rollout.get(
            'enable_cuda_graph', False):
        raise ValueError('Compile and CUDA graph are not supported in v1')
    for phase in ('train', 'eval'):
        get_benchmark_spec(cfg.env[phase].env_type).validate(cfg, phase)
    if not Path(cfg.actor.model.model_path).expanduser().exists():
        raise FileNotFoundError(
            'Set actor.model.model_path to an existing initial SFT checkpoint')


def run(cfg):
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.embodied_fsdp_actor_worker import \
        EmbodiedFSDPActor

    from fluxvla.rl.rlinf_registry import register
    from fluxvla.rl.workers.rollout import FluxRolloutWorker

    register()
    validate_frontend_cfg(cfg)
    if 'wandb' in cfg.runner.logger.logger_backends:
        from fluxvla.rl.utils.logging import require_wandb
        require_wandb()
    cfg = validate_cfg(cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    groups = {}
    benchmark = get_benchmark_spec(cfg.env.train.env_type)
    env_worker = resolve_symbol(benchmark.env_worker_class)
    actor_worker = EmbodiedFSDPActor
    if (benchmark.audit_actor
            or cfg.get('experiment', {}).get('audit_actor', False)):
        from fluxvla.rl.workers.actor import AuditedEmbodiedActor
        actor_worker = AuditedEmbodiedActor
    for name, worker in (('actor', actor_worker),
                         ('rollout', FluxRolloutWorker), ('env', env_worker)):
        groups[name] = worker.create_group(cfg).launch(
            cluster,
            name=cfg[name].group_name,
            placement_strategy=placement.get_strategy(name))
    runner = resolve_symbol(benchmark.runner_class)(cfg=cfg, **groups)
    runner.init_workers()
    runner.run()


def main():
    import hydra
    import torch.multiprocessing as mp

    from fluxvla.rl.rlinf_registry import register

    root = prepare_environment()
    register()
    mp.set_start_method('spawn', force=True)

    @hydra.main(
        version_base='1.3',
        config_path=str(root / 'configs/rl'),
        config_name='benchmarks/libero/pi05/ppo')
    def launch(cfg):
        run(cfg)

    launch()


if __name__ == '__main__':
    main()
