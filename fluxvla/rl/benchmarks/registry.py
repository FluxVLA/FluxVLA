"""Benchmark declarations; no eager simulator or policy imports."""

from dataclasses import dataclass

from fluxvla.rl.utils.imports import resolve_symbol


@dataclass(frozen=True)
class BenchmarkSpec:
    adapter_factory: str
    config_validator: str
    environment_class: str = None
    env_worker_class: str = 'fluxvla.rl.workers.env.FluxEnvWorker'
    runner_class: str = 'rlinf.runners.embodied_runner.EmbodiedRunner'
    audit_actor: bool = False
    pair_eval_rng: bool = False

    def validate(self, cfg, phase):
        resolve_symbol(self.config_validator)(cfg, phase)

    def get_environment_class(self, default):
        return (resolve_symbol(self.environment_class)
                if self.environment_class else default)


BENCHMARKS = {
    'libero':
    BenchmarkSpec(
        adapter_factory='fluxvla.rl.benchmarks.libero.adapter.build_adapter',
        config_validator=(
            'fluxvla.rl.benchmarks.libero.protocol.validate_config')),
    'robotwin':
    BenchmarkSpec(
        adapter_factory='fluxvla.rl.benchmarks.robotwin.adapter.build_adapter',
        config_validator=(
            'fluxvla.rl.benchmarks.robotwin.protocol.validate_config'),
        environment_class=(
            'fluxvla.rl.benchmarks.robotwin.env.AuditedRoboTwinEnv'),
        env_worker_class=(
            'fluxvla.rl.benchmarks.robotwin.worker.RoboTwinEnvWorker'),
        runner_class='fluxvla.rl.benchmarks.robotwin.runner.RoboTwinRunner',
        audit_actor=True,
        pair_eval_rng=True),
}


def get_benchmark_spec(name):
    try:
        return BENCHMARKS[name]
    except KeyError as error:
        raise ValueError(f'Unknown benchmark: {name}') from error


def build_observation_adapter(flux_cfg, options, norm_stats=None):
    spec = get_benchmark_spec(options.get('observation_adapter', 'libero'))
    return resolve_symbol(spec.adapter_factory)(flux_cfg, options, norm_stats)
