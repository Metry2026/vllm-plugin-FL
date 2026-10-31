# SPDX-License-Identifier: Apache-2.0
"""Opt-in functional replacement of the observed forward_static compile path."""
import os
import torch

_installed = False
_op = None


@torch.compiler.assume_constant_result
def model_supported():
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    hf = config.model_config.hf_config
    return (
        config.parallel_config.tensor_parallel_size == 1
        and getattr(hf, 'architectures', None) == ['LlamaForCausalLM']
        and hf.hidden_size == 2048 and hf.num_attention_heads == 16
        and hf.num_key_value_heads == 2
        and config.model_config.dtype == torch.bfloat16
    )


def supported(positions, query, key, head_size, rotary_dim, table, is_neox_style):
    """Metadata-only eligibility; upstream retains responsibility for positions."""
    return (
        head_size == 128 and rotary_dim == 128 and is_neox_style is True
        and key is not None and query.is_cuda
        and positions.device == query.device == key.device == table.device
        and query.dtype == key.dtype == table.dtype == torch.bfloat16
        and positions.dtype == torch.int64
        and query.ndim == key.ndim == 2 and positions.ndim == 1
        and 0 < query.shape[0] <= 2048
        and query.shape[1] == 2048 and key.shape[1] == 256
        and key.shape[0] == positions.shape[0] == query.shape[0]
        and query.stride() == key.stride() == (2560, 1)
        and positions.stride() == (1,)
        and table.shape == (131072, 128) and table.stride() == (128, 1)
    )


def register_op():
    global _op
    if _op is not None:
        return _op
    # Import the backend by its source path without assuming an unhit public
    # FlagGems rotary symbol is the active implementation.
    import importlib
    kernel = importlib.import_module('flag_gems.runtime.backend._iluvatar.fused.paired_rope')

    @torch.library.custom_op('vllm_fl::paired_rope', mutates_args=())
    def op(query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor,
           table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel.paired_rope(query, key, positions, table)

    @op.register_fake
    def fake(query, key, positions, table):
        return (torch.empty(query.shape, dtype=query.dtype, device=query.device),
                torch.empty(key.shape, dtype=key.dtype, device=key.device))

    _op = op
    return op


def apply_patch():
    global _installed
    if _installed or os.environ.get('VLLM_FL_PAIRED_ROPE', '0') != '1':
        return False
    from vllm.platforms import current_platform
    if getattr(current_platform, 'vendor_name', '') != 'iluvatar':
        return False
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    op = register_op()
    original = RotaryEmbedding.forward_static

    def forward_static(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style):
        if torch.compiler.is_compiling() and model_supported() and supported(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style):
            return op(query, key, positions, cos_sin_cache)
        return original(positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style)

    forward_static.task013_original = original
    RotaryEmbedding.forward_static = staticmethod(forward_static)
    _installed = True
    return True
