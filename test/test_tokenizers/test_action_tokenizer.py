# Copyright 2026 Limx Dynamics
"""Test quantization itself rather than downloading an LLM tokenizer."""

from unittest.mock import Mock

import numpy as np
import pytest

from fluxvla.engines import build_tokenizer_from_cfg
from fluxvla.tokenizers import action_tokenizer as module


@pytest.fixture
def tokenizer(monkeypatch):
    text_tokenizer = Mock(vocab_size=100)
    text_tokenizer.decode.side_effect = lambda ids: ' '.join(map(str, ids))
    text_tokenizer.batch_decode.side_effect = (
        lambda batch: [' '.join(map(str, ids)) for ids in batch])
    loader = Mock(return_value=text_tokenizer)
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', loader)
    tokenizer = build_tokenizer_from_cfg(
        dict(
            type='ActionTokenizer', model_path='local-test-tokenizer', bins=5))
    loader.assert_called_once_with(
        'local-test-tokenizer', trust_remote_code=True)
    return tokenizer


@pytest.mark.parametrize('batched', [False, True])
def test_quantization_clips_and_decodes_bin_centers(tokenizer, batched):
    actions = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    if batched:
        actions = np.stack([actions, actions])
    text = tokenizer(actions)
    expected_text = '99 98 97 96 95'
    assert text == ([expected_text, expected_text]
                    if batched else expected_text)

    ids = np.array([99, 98, 97, 96, 95])
    expected = np.array([-0.75, -0.25, 0.25, 0.75, 0.75])
    if batched:
        ids, expected = np.stack([ids, ids]), np.stack([expected, expected])
    np.testing.assert_array_equal(
        tokenizer.decode_token_ids_to_actions(ids), expected)
    assert tokenizer.vocab_size == 5
