# SPDX-License-Identifier: Apache-2.0
"""Opt-in O204 entry for the bound compiled LlamaMLP gate/up domain."""

import functools
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_LLAMA_SHA256 = "1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384"
_OP = None


def supported(layer, x, llama_type, linear_type, method_type, act_types, native_act, torch):
    """Tensor metadata and exact module contracts only; no tensor value reads."""
    if type(layer) is not llama_type:
        return False
    gate = layer.gate_up_proj
    if type(gate) is not linear_type or type(layer.act_fn) not in act_types:
        return False
    # In the installed vLLM this native activation is a static method.
    # CustomOp may retain it directly or wrap it with torch.compile.
    method = getattr(layer.act_fn, "_forward_method", None)
    if getattr(method, "_torchdynamo_orig_callable", method) is not native_act:
        return False
    if type(gate.quant_method) is not method_type:
        return False
    if getattr(gate, "tp_size", None) != 1 or gate.bias is not None:
        return False
    if getattr(gate, "quant_config", None) is not None:
        return False
    if tuple(getattr(gate, "output_sizes", ())) != (6144, 6144):
        return False
    w = gate.weight
    if (not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16
            or x.shape[1] != 2048 or not 0 < x.shape[0] <= 2048
            or not x.is_contiguous() or x.requires_grad):
        return False
    if (w.device != x.device or w.dtype != x.dtype
            or tuple(w.shape) != (12288, 2048)
            or not w.is_contiguous() or w.requires_grad):
        return False
    return True


def install():
    global _OP
    if os.environ.get("VLLM_FL_O204_GATE_UP_SWIGLU", "0") != "1":
        return False
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        return False
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return False
    import torch
    from vllm.model_executor.models import llama
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear, UnquantizedLinearMethod,
    )
    from vllm_fl.ops.activation import SiluAndMulFL

    cls = llama.LlamaMLP
    if hasattr(cls.forward, "o204_original"):
        return True
    original = cls.forward
    if (hashlib.sha256(Path(llama.__file__).read_bytes()).hexdigest() != _LLAMA_SHA256
            or Path(original.__code__.co_filename).resolve() != Path(llama.__file__).resolve()):
        logger.warning("O204 unsupported LlamaMLP source/forward; using original")
        return False

    if _OP is None:
        @torch.library.custom_op("vllm_fl::o204_gate_up_swiglu", mutates_args=())
        def compiled_gate_up(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            from flag_gems.runtime.backend._iluvatar.fused.paired_gateup_swiglu import gate_up_swiglu
            return gate_up_swiglu(x, weight)

        @compiled_gate_up.register_fake
        def fake_gate_up(x, weight):
            return x.new_empty((x.shape[0], 6144))

        _OP = compiled_gate_up

    @functools.wraps(original)
    def forward(self, x):
        # The proved generated activation keeps SiLU in FP32. Eager and other
        # contracts retain their own complete original projection/activation.
        if (torch.compiler.is_compiling()
                and supported(self, x, cls, MergedColumnParallelLinear,
                              UnquantizedLinearMethod, (SiluAndMul, SiluAndMulFL),
                              SiluAndMul.forward_native, torch)):
            x = _OP(x, self.gate_up_proj.weight)
            x, _ = self.down_proj(x)
            return x
        return original(self, x)

    forward.o204_original = original
    cls.forward = forward
    logger.info("O204 compiled gate/up + SwiGLU installed; guarded original chain retained")
    return True
