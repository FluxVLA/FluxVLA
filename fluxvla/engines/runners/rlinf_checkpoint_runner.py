"""Keep RLinf resume checkpoints and publish separate native policy bundles."""

from pathlib import Path

from rlinf.runners.embodied_runner import EmbodiedRunner

from fluxvla.engines.utils.rl_export import export_native_bundle


class FluxEmbodiedRunner(EmbodiedRunner):

    def _save_checkpoint(self):
        # Wait for all actor shards and consolidated weights before exporting.
        super()._save_checkpoint()
        if (not self.cfg.runner.get('export_native_policy', True)
                or self.cfg.actor.model.model_type != 'fluxvla_smolvla'):
            return
        step = (
            Path(self.cfg.runner.logger.log_path) /
            self.cfg.runner.logger.experiment_name / 'checkpoints' /
            f'global_step_{self.global_step}')
        try:
            from omegaconf import OmegaConf

            # Keep the exact run config even if inference packaging fails.
            OmegaConf.save(
                OmegaConf.create(
                    OmegaConf.to_container(self.cfg, resolve=True)),
                step / 'rl_config.yaml')
            manifest = export_native_bundle(step, step / 'native_policy',
                                            self.cfg)
            self.logger.info(
                f"Native policy exported to {step / 'native_policy'} "
                f"({manifest['policy_tensor_count']} tensors).")
        except Exception:
            # Inference packaging is secondary; a valid PPO resume checkpoint
            # must remain available if copying artifacts runs out of space.
            self.logger.exception(
                f'Native policy export failed at {step}; full PPO checkpoint '
                'is preserved. Retry scripts/rl_export.py --native-bundle '
                f'--rl-config {step / "rl_config.yaml"}.')
