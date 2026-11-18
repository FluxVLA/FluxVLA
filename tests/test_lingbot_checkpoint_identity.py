import json

import pytest

from tools.lingbot_checkpoint_identity import assert_same_checkpoint, checkpoint_identity


def _checkpoint(path, config="same"):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model": config}), encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (path / "part.safetensors").write_bytes(b"weights-" + config.encode())


def test_checkpoint_identity_and_compare(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left)
    _checkpoint(right)
    assert checkpoint_identity(left)["weight_total_bytes"] == checkpoint_identity(right)["weight_total_bytes"]
    assert_same_checkpoint(checkpoint_identity(left), checkpoint_identity(right))


def test_checkpoint_identity_rejects_different_weights(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left)
    _checkpoint(right, config="other")
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        assert_same_checkpoint(checkpoint_identity(left), checkpoint_identity(right))
