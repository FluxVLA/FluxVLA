# Copyright 2026 Limx Dynamics
# Licensed under the Apache License, Version 2.0.
"""FP32 transition distributions matching RLinf OpenPI flow_sde semantics.

These are denoising-transition log probabilities, not the marginal likelihood
of the final environment action. The policy samples one stochastic step and
passes elementwise log probabilities to RLinf for its configured aggregation.
"""

import math

import torch


def timesteps(num_steps: int, device) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError('flow_sde requires num_steps >= 2')
    return torch.cat(
        (torch.linspace(1., 1. / num_steps, num_steps,
                        device=device), torch.zeros(1, device=device)))


def transition(x,
               velocity,
               indices,
               *,
               num_steps,
               noise_level,
               stochastic=True):
    """Return FP32 mean and standard deviation for a batched denoise step."""
    x, velocity = x.float(), velocity.float()
    times = timesteps(num_steps, x.device)
    t = times[indices][:, None, None]
    dt = (times[indices] - times[indices + 1])[:, None, None]
    x0, x1 = x - t * velocity, x + (1 - t) * velocity
    if stochastic:
        denom = torch.where(times == 1, times[1], times)
        sigmas = noise_level * torch.sqrt(times / (1 - denom))[:-1]
        sigma = sigmas[indices][:, None, None]
        weight1 = t - dt - sigma.square() * dt / (2 * t)
        std = dt.sqrt() * sigma
    else:
        weight1 = t - dt
        std = torch.zeros_like(t)
    mean = x0 * (1 - (t - dt)) + x1 * weight1
    return mean, std.expand_as(x)


def gaussian_logprob(sample, mean, std):
    sample, mean, std = sample.float(), mean.float(), std.float()
    safe_std = torch.where(std == 0, torch.ones_like(std), std)
    logprob = (-safe_std.log() - 0.5 * math.log(2 * math.pi) - 0.5 *
               ((sample - mean) / safe_std).square())
    return torch.where(std == 0, torch.zeros_like(logprob), logprob)


def gaussian_entropy(std):
    std = std.float()
    safe_std = torch.where(std == 0, torch.ones_like(std), std)
    entropy = 0.5 * math.log(2 * math.pi * math.e) + safe_std.log()
    return torch.where(std == 0, torch.zeros_like(entropy), entropy)
