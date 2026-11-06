# Vendored from VPP2, revision 3365872910244f14df83003062f7c563f14c6d9e.
# See ../../THIRD_PARTY_NOTICES.md for licenses and source notices.
import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with configurable training sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        eps: float = 1e-10,
        train_sampling_strategy: str = "shifted_uniform",
        beta_alpha: float = 7.0,
        beta_beta: float = 1.0,
    ):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        if beta_alpha <= 0 or beta_beta <= 0:
            raise ValueError(
                f"`beta_alpha` and `beta_beta` must be positive, got {beta_alpha}, {beta_beta}"
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self.train_sampling_strategy = str(train_sampling_strategy)
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        if self.train_sampling_strategy not in {"shifted_uniform", "beta", "pure_noise"}:
            raise ValueError(
                f"Unsupported train_sampling_strategy: {self.train_sampling_strategy}. "
                "Expected one of: shifted_uniform, beta, pure_noise."
            )
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        if self.train_sampling_strategy == "shifted_uniform":
            u = torch.rand((batch_size,), device=device, dtype=torch.float32)
            sigma = self._phi(u, self.shift)
        elif self.train_sampling_strategy == "beta":
            alpha = torch.full((batch_size,), self.beta_alpha, device=device, dtype=torch.float32)
            beta = torch.full((batch_size,), self.beta_beta, device=device, dtype=torch.float32)
            sigma = torch._standard_gamma(alpha)
            sigma = sigma / (sigma + torch._standard_gamma(beta)).clamp(min=self.eps)
            sigma = sigma.clamp(min=0.0, max=1.0)
        elif self.train_sampling_strategy == "pure_noise":
            # sigma=1 makes add_noise return the noise exactly, with no clean sample mixed in.
            sigma = torch.ones((batch_size,), device=device, dtype=torch.float32)
        else:
            raise RuntimeError(f"Unhandled train_sampling_strategy: {self.train_sampling_strategy}")
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.train_sampling_strategy == "pure_noise":
            return torch.ones_like(timestep, dtype=torch.float32)
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
