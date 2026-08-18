# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import lru_cache

import torch
import torch.nn.functional as F

from fluxvla.models.third_party_models.cosmos3.data.vfm import sequence_packing

FactoredSequencePack = sequence_packing.FactoredSequencePack
JointSequencePack = sequence_packing.JointSequencePack

_AUTO_BACKENDS = {None, 'auto'}
_FLASH2_BACKENDS = {'flash2', 'flash_attention_2'}
_FLASH3_BACKENDS = {'flash3', 'flash_attention_3'}
_SDPA_BACKENDS = {'sdpa', 'torch_sdpa'}
_SUPPORTED_BACKENDS = (
    _AUTO_BACKENDS | _FLASH2_BACKENDS | _FLASH3_BACKENDS | _SDPA_BACKENDS)
_FLASH_DTYPES = {torch.float16, torch.bfloat16}


@lru_cache(maxsize=1)
def _load_flash2_varlen():
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception as exc:
        return None, str(exc)
    return flash_attn_varlen_func, None


@lru_cache(maxsize=1)
def _load_flash3_varlen():
    try:
        import flash_attn_interface
    except Exception as exc:
        return None, str(exc)
    return flash_attn_interface.flash_attn_varlen_func, None


class SplitInfo:
    """FluxVLA-native packed two-way attention metadata.

    Cosmos3 Nano/Super checkpoints use dense two-way MoT attention: causal
    understanding tokens attend causally within each sample, and full
    generation tokens attend to all tokens from the same sample. Upstream
    sparse experimental paths are intentionally not exposed in FluxVLA.
    """

    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
    ):
        if sum(sample_lens) != sum(split_lens):
            raise ValueError(
                f'Sum of sample lens {sum(sample_lens)} is not equal to '
                f'sum of split lens {sum(split_lens)}')

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == 'causal':
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == 'full':
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)
        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens
        self.actual_len = actual_len


def _check_backend(backend: str | None) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            'FluxVLA Cosmos3 packed attention supports "auto", "flash2", '
            '"flash_attention_2", "flash3", "flash_attention_3", and "sdpa". '
            f'Got {backend!r}.')


def _flash_inputs_compatible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[bool, str | None]:
    if query.device.type != 'cuda':
        return False, 'flash-attn requires CUDA tensors'
    if key.device != query.device or value.device != query.device:
        return False, 'flash-attn requires Q/K/V to be on the same device'
    if query.dtype not in _FLASH_DTYPES:
        return False, (
            f'flash-attn requires fp16/bf16 tensors, got {query.dtype}')
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False, 'flash-attn requires Q/K/V to have the same dtype'
    if query.shape[-1] > 256 or key.shape[-1] > 256 or value.shape[-1] > 256:
        return False, 'flash-attn only supports head_dim <= 256'
    if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
        return False, 'flash-attn does not handle empty Q/K/V tensors'
    return True, None


def _flash2_compatible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[bool, str | None]:
    can_run, reason = _flash_inputs_compatible(query, key, value)
    if not can_run:
        return False, reason
    _, import_error = _load_flash2_varlen()
    if import_error is not None:
        return False, f'flash-attn 2 is not importable: {import_error}'
    return True, None


def _flash3_compatible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[bool, str | None]:
    can_run, reason = _flash_inputs_compatible(query, key, value)
    if not can_run:
        return False, reason
    capability = torch.cuda.get_device_capability(query.device)
    if capability != (9, 0):
        return False, ('flash-attn 3 is enabled for sm90 only, '
                       f'got sm{capability[0]}{capability[1]}')
    _, import_error = _load_flash3_varlen()
    if import_error is not None:
        return False, f'flash-attn 3 is not importable: {import_error}'
    return True, None


def _flash2_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    flash_attn_varlen_func, import_error = _load_flash2_varlen()
    if flash_attn_varlen_func is None:
        raise RuntimeError(f'flash-attn 2 is not available: {import_error}')
    return flash_attn_varlen_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=q_offsets.to(device=query.device,
                                  dtype=torch.int32).contiguous(),
        cu_seqlens_k=kv_offsets.to(device=query.device,
                                   dtype=torch.int32).contiguous(),
        max_seqlen_q=int(max_q_len),
        max_seqlen_k=int(max_kv_len),
        dropout_p=0.0,
        causal=is_causal,
    )


def _flash3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    flash_attn_varlen_func, import_error = _load_flash3_varlen()
    if flash_attn_varlen_func is None:
        raise RuntimeError(f'flash-attn 3 is not available: {import_error}')
    output = flash_attn_varlen_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=q_offsets.to(device=query.device,
                                  dtype=torch.int32).contiguous(),
        cu_seqlens_k=kv_offsets.to(device=query.device,
                                   dtype=torch.int32).contiguous(),
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=int(max_q_len),
        max_seqlen_k=int(max_kv_len),
        softmax_scale=None,
        causal=is_causal,
        deterministic=False,
    )
    return output[0] if isinstance(output, tuple) else output


def _sdpa_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    del max_q_len, max_kv_len
    outputs = []
    original_dtype = query.dtype
    q_offsets = q_offsets.to(device=query.device)
    kv_offsets = kv_offsets.to(device=query.device)
    with torch.autocast(device_type=query.device.type, enabled=False):
        for index in range(q_offsets.numel() - 1):
            q_start = int(q_offsets[index].item())
            q_end = int(q_offsets[index + 1].item())
            kv_start = int(kv_offsets[index].item())
            kv_end = int(kv_offsets[index + 1].item())
            q = query[q_start:q_end].float().transpose(0, 1).unsqueeze(0)
            k = key[kv_start:kv_end].float().transpose(0, 1).unsqueeze(0)
            v = value[kv_start:kv_end].float().transpose(0, 1).unsqueeze(0)
            if k.shape[1] != q.shape[1]:
                repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal)
            outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0).to(dtype=original_dtype)


def _run_flash_backend(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor | None, str | None]:
    if backend == 'flash3':
        can_flash, reason = _flash3_compatible(query, key, value)
        if can_flash:
            return _flash3_varlen_attention(
                query,
                key,
                value,
                q_offsets,
                kv_offsets,
                max_q_len,
                max_kv_len,
                is_causal=is_causal,
            ), None
        return None, reason

    can_flash, reason = _flash2_compatible(query, key, value)
    if can_flash:
        return _flash2_varlen_attention(
            query,
            key,
            value,
            q_offsets,
            kv_offsets,
            max_q_len,
            max_kv_len,
            is_causal=is_causal,
        ), None
    return None, reason


def _packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    is_causal: bool = False,
    backend: str | None = None,
) -> torch.Tensor:
    """Run packed varlen attention without materializing a block mask.

    Mirrors Cosmos3's production assumption: packed MoT attention requires a
    compatible flash-attn varlen backend.
    """
    if backend in _AUTO_BACKENDS:
        reasons = {}
        for candidate in ('flash3', 'flash2'):
            output, reason = _run_flash_backend(
                candidate,
                query,
                key,
                value,
                q_offsets,
                kv_offsets,
                max_q_len,
                max_kv_len,
                is_causal=is_causal,
            )
            if output is not None:
                return output
            reasons[candidate] = reason
        raise RuntimeError(
            'Cosmos3 packed attention requires flash-attn varlen support, '
            'but no compatible backend was found: '
            f'flash3={reasons["flash3"]}; flash2={reasons["flash2"]}')

    if backend in _SDPA_BACKENDS:
        return _sdpa_varlen_attention(
            query,
            key,
            value,
            q_offsets,
            kv_offsets,
            max_q_len,
            max_kv_len,
            is_causal=is_causal,
        )

    selected_backend = 'flash3' if backend in _FLASH3_BACKENDS else 'flash2'
    output, reason = _run_flash_backend(
        selected_backend,
        query,
        key,
        value,
        q_offsets,
        kv_offsets,
        max_q_len,
        max_kv_len,
        is_causal=is_causal,
    )
    if output is not None:
        return output
    raise RuntimeError(
        f'Cosmos3 packed attention requires {selected_backend} varlen '
        f'support, but it is not compatible: {reason}')


def two_way_attention(
    packed_query_states: FactoredSequencePack | JointSequencePack,
    packed_key_states: FactoredSequencePack | JointSequencePack,
    packed_value_states: FactoredSequencePack | JointSequencePack,
    backend: str | None = None,
    packed_key_states_normalized: FactoredSequencePack
    | JointSequencePack
    | None = None,
) -> FactoredSequencePack | JointSequencePack:
    """Run dense two-way MoT attention on packed sequences."""
    _check_backend(backend)

    causal_q, causal_q_offsets = sequence_packing.get_causal_seq(
        packed_query_states)
    causal_k, causal_k_offsets = sequence_packing.get_causal_seq(
        packed_key_states)
    causal_v, _ = sequence_packing.get_causal_seq(packed_value_states)
    full_q, full_q_offsets = sequence_packing.get_full_only_seq(
        packed_query_states)
    sample_offsets = packed_query_states['sample_offsets']

    causal_out = _packed_varlen_attention(
        causal_q,
        causal_k,
        causal_v,
        causal_q_offsets,
        causal_k_offsets,
        packed_query_states['max_causal_len'],
        packed_key_states['max_causal_len'],
        is_causal=True,
        backend=backend,
    ).flatten(-2, -1)
    packed_key_normalized = (
        packed_key_states_normalized
        if packed_key_states_normalized is not None else packed_key_states)
    full_out = _packed_varlen_attention(
        full_q,
        sequence_packing.get_all_seq(packed_key_normalized),
        sequence_packing.get_all_seq(packed_value_states),
        full_q_offsets,
        sample_offsets,
        packed_query_states['max_full_len'],
        packed_key_states['max_sample_len'],
        backend=backend,
    ).flatten(-2, -1)

    return sequence_packing.from_mode_splits(causal_out, full_out,
                                             packed_query_states)


def build_packed_sequence(
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    pad_for_cuda_graphs: bool = False,
) -> tuple[FactoredSequencePack | JointSequencePack, SplitInfo]:
    """Build the two-way packed sequence and attention metadata."""

    device = packed_sequence.device
    attention_meta = SplitInfo(
        split_lens=split_lens,
        attn_modes=attn_modes,
        sample_lens=sample_lens,
        actual_len=int(packed_sequence.shape[0]),
    )
    input_pack = sequence_packing.factored_from_joint_sequence(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    input_pack.pop('split_lens', None)
    input_pack.pop('attn_modes', None)
    return input_pack, attention_meta
