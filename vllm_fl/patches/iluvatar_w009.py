# SPDX-License-Identifier: Apache-2.0
"""Default-off W009 selection prototypes; exactly one F/L/D arm at a time.

These hooks expose component/forward experiments, not a qualified service path.
No installed sources or global compiler functions are modified.
"""
import functools
import hashlib
import os
from pathlib import Path


SOURCE_SHA256 = {
    "linear": "0bd60103d85d7183791077c46e0770d0a2adc8e8811b162b9084c3cbe92ba662",
    "linear_utils": "061adffc6af55ef3b001557c0fad738e474c5e4134fa3dc35cb967bdbc90a121",
    "llama": "1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384",
}
M_BUCKETS = frozenset((1, 32, 2048))
FFN_WEIGHTS = frozenset(((12288, 2048), (2048, 6144)))
_INSTALLED_ARM = None


def _mark_installed(arm):
    global _INSTALLED_ARM
    _INSTALLED_ARM = arm
    return True


def _source(module, key):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE_SHA256[key]:
        raise RuntimeError("W009 unsupported source identity: " + key)


def linear_supported(layer, x, bias, torch):
    w = layer.weight
    return (x.ndim == 2 and x.shape[0] in M_BUCKETS and x.is_cuda
            and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and w.device == x.device and tuple(w.shape) in FFN_WEIGHTS
            and w.shape[1] == x.shape[1] and getattr(layer, "tp_size", None) == 1
            and (bias is None or (bias.ndim == 1 and bias.shape[0] == w.shape[0]
                                 and bias.dtype == x.dtype and bias.device == x.device)))


def make_linear_dispatch(original, target, torch, method_type, *, audit=False):
    """Remove repeated platform selection; execute the exact same GEMM function."""
    counts = dict(candidate=0, fallback=0)

    @functools.wraps(original)
    def apply(self, layer, x, bias=None):
        hit = type(self) is method_type and linear_supported(layer, x, bias, torch)
        if audit:
            counts["candidate" if hit else "fallback"] += 1
        if not hit:
            return original(self, layer, x, bias)
        return target(layer, x, layer.weight, bias)

    apply.w009_original = original
    apply.w009_counts = counts
    return apply


def ffn_supported(owner, x, torch, method_type):
    gate, down = owner.gate_up_proj, owner.down_proj
    if (type(gate).__name__ != "MergedColumnParallelLinear"
            or type(down).__name__ != "RowParallelLinear"
            or type(owner.act_fn).__name__ != "SiluAndMulFL"
            or type(gate.quant_method) is not method_type
            or type(down.quant_method) is not method_type
            or not getattr(gate, "return_bias", False)
            or not getattr(down, "return_bias", False)
            or getattr(gate, "gather_output", True)
            or gate.bias is not None or down.bias is not None):
        return False
    return (linear_supported(gate, x, None, torch)
            and tuple(gate.weight.shape) == (12288, 2048)
            and tuple(down.weight.shape) == (2048, 6144)
            and down.weight.device == x.device and down.weight.dtype == x.dtype
            and getattr(down, "tp_size", None) == 1
            and x.stride(-1) == 1 and x.stride(0) >= x.shape[1]
            and gate.weight.stride(-1) == 1
            and gate.weight.stride(0) >= gate.weight.shape[1])


def make_ffn_dispatch(original, fused, torch, method_type, *, config=0,
                      round_silu=False, policy_valid=lambda: True, audit=False):
    if config not in (0, 1):
        raise ValueError("P1 F permits exactly two configurations")
    counts = dict(candidate=0, fallback=0)

    @functools.wraps(original)
    def forward(self, x):
        hit = policy_valid() and ffn_supported(self, x, torch, method_type)
        if audit:
            counts["candidate" if hit else "fallback"] += 1
        if not hit:
            return original(self, x)
        hidden = fused(x, self.gate_up_proj.weight, config=config, round_silu=round_silu)
        output, _ = self.down_proj(hidden)
        return output

    forward.w009_original = original
    forward.w009_counts = counts
    return forward


def selected_arm(environ):
    flags = {name: environ.get(key, "0") == "1" for name, key in (
        ("F", "VLLM_FL_W009_FFN"), ("L", "VLLM_FL_W009_LINEAR"),
        ("D", "VLLM_FL_O303_DECODE"))}
    active = [name for name, enabled in flags.items() if enabled]
    if len(active) > 1:
        raise ValueError("W009 selection arms must not be combined")
    if not active:
        return None
    if any(value not in ("", "0") for key, value in environ.items()
           if key.startswith(("VLLM_FL_W007", "VLLM_FL_O4"))):
        raise ValueError("W009 requires W007/O4 disabled")
    if (environ.get("VLLM_FL_W006_PAGED_GQA", "0") != "1"
            or environ.get("VLLM_FL_W006_BLOCK_Q", "8") != "8"
            or environ.get("VLLM_FL_W006_BLOCK_N", "32") != "32"
            or environ.get("VLLM_BATCH_INVARIANT", "0") != "0"):
        raise ValueError("W009 requires C0 W006 Q8/KV32, batch invariance off")
    return active[0]


def install():
    arm = selected_arm(os.environ)
    if _INSTALLED_ARM is not None and arm != _INSTALLED_ARM:
        raise RuntimeError("W009 arm changes or disabling require a fresh process")
    if arm is None:
        return False
    import torch
    from vllm.platforms import current_platform
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        raise RuntimeError("W009 prototypes require Iluvatar")
    audit = os.environ.get("VLLM_FL_W009_AUDIT", "0") == "1"
    if arm == "D":
        from vllm.v1.attention.backends import triton_attn
        from vllm.v1.attention.ops import triton_unified_attention as ops
        from .iluvatar_paged_gqa import supported as base_guard
        from .iluvatar_decode_gqa import load_reader, make_dispatch
        if hasattr(triton_attn.unified_attention, "w009_original"):
            return _mark_installed(arm)
        if not hasattr(triton_attn.unified_attention, "w006_original"):
            raise RuntimeError("W009 D requires installed C0 dispatch")
        directory = os.environ.get("VLLM_FL_W009_READER_DIR")
        if not directory:
            raise RuntimeError("Provide a new experiment-owned W009 reader directory")
        module, identity = load_reader(ops.__file__, directory)
        dispatch = make_dispatch(triton_attn.unified_attention, module.unified_attention,
            torch, base_guard, segments=int(os.environ.get("VLLM_FL_W009_SEGMENTS", "4")), audit=audit)
        dispatch.w009_identity = identity
        triton_attn.unified_attention = dispatch
        return _mark_installed(arm)
    from vllm.model_executor.layers import linear, utils
    _source(linear, "linear"); _source(utils, "linear_utils")
    if arm == "L":
        cls = linear.UnquantizedLinearMethod
        if hasattr(cls.apply, "w009_original"):
            return _mark_installed(arm)
        if utils.dispatch_unquantized_gemm() is not utils.default_unquantized_gemm:
            raise RuntimeError("W009 L requires the audited default GEMM route")
        cls.apply = make_linear_dispatch(cls.apply, utils.default_unquantized_gemm,
                                         torch, cls, audit=audit)
        return _mark_installed(arm)
    from vllm.model_executor.models import llama
    from vllm_fl.dispatch import get_default_manager, get_policy_epoch, is_dump_enabled
    from flag_gems.runtime.backend._iluvatar.fused.gate_up_swiglu import gate_up_swiglu
    _source(llama, "llama")
    if hasattr(llama.LlamaMLP.forward, "w009_original"):
        return _mark_installed(arm)
    mgr = get_default_manager()
    impl = mgr.resolve("silu_and_mul")
    module_name = getattr(impl, "__module__", "")
    if ".backends.flaggems." in module_name:
        round_silu = False
    elif ".backends.vendor.iluvatar." in module_name:
        round_silu = True
    else:
        raise RuntimeError("W009 F activation rounding contract is not audited")
    epoch = (id(mgr), mgr.policy_epoch, get_policy_epoch())
    def policy_valid():
        current = get_default_manager()
        return (not is_dump_enabled()
                and (id(current), current.policy_epoch, get_policy_epoch()) == epoch)
    llama.LlamaMLP.forward = make_ffn_dispatch(llama.LlamaMLP.forward,
        gate_up_swiglu, torch, linear.UnquantizedLinearMethod,
        config=int(os.environ.get("VLLM_FL_W009_FFN_CONFIG", "0")),
        round_silu=round_silu, policy_valid=policy_valid, audit=audit)
    return _mark_installed(arm)
