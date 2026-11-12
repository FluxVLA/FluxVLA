import torch
from transformers import GemmaConfig

from fluxvla.models.backbones.llms.condition_gemma import GemmaRotaryEmbedding


def test_gemma_rope_keeps_frequency_buffers_fp32():
    config = GemmaConfig(
        head_dim=256,
        hidden_size=2048,
        num_attention_heads=8,
        rope_theta=10000.0,
    )
    rotary = GemmaRotaryEmbedding(config).to(dtype=torch.bfloat16)

    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32


def test_default_rope_rebuilds_frequency_after_external_buffer_cast():
    config = GemmaConfig(
        head_dim=256,
        hidden_size=2048,
        num_attention_heads=8,
        rope_theta=10000.0,
    )
    rotary = GemmaRotaryEmbedding(config)
    position_ids = torch.tensor([[0, 1, 256]], dtype=torch.long)
    hidden = torch.zeros(1, 3, config.head_dim, dtype=torch.bfloat16)
    expected_inv_freq, _ = rotary.compute_default_rope_parameters(config)

    # Reproduce FSDP MixedPrecision(buffer_dtype=bf16), which does not call
    # the child module's _apply override.
    rotary.inv_freq = rotary.inv_freq.to(torch.bfloat16)
    rotary.original_inv_freq = rotary.original_inv_freq.to(torch.bfloat16)
    cos, sin = rotary(hidden, position_ids)

    freqs = position_ids.float().unsqueeze(-1) * expected_inv_freq
    expected = torch.cat((freqs, freqs), dim=-1)
    assert torch.equal(cos, expected.cos().to(torch.bfloat16))
    assert torch.equal(sin, expected.sin().to(torch.bfloat16))
    assert torch.equal(rotary.runtime_inv_freq(), expected_inv_freq)
