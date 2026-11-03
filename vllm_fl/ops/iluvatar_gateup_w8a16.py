"""Task026 opt-in single-layer projection prototype, retained BF16 fallback.

Call prepare after native weight loading and before capture. No automatic model
installation is provided by this prototype; full-model integration is pending
the complete projection cost decision. No existing runner or helper is changed.
"""
import os
import time
import torch
import triton
import triton.language as tl

PREFIX = 'model.layers.0.mlp.gate_up_proj'
# Current actual parent selections, frozen before candidate measurements.
CONFIGS = {
    8: (8, 256, 64, 16, 1),
    37: (64, 256, 64, 16, 1),
    248: (128, 128, 32, 8, 2),
    2048: (256, 256, 32, 16, 2),
}


def quantize_cpu(weight):
    """Fixed FP32 maxabs/127 and RNE; zero-row scale exactly one."""
    assert weight.device.type == 'cpu' and weight.dtype == torch.bfloat16
    w = weight.to(torch.float32)
    if not bool(torch.isfinite(w).all()):
        raise ValueError('non-finite weight')
    scale = w.abs().amax(dim=1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


class GateUpW8A16Method:
    def __init__(self, layer, original, q, scale):
        self.original = original
        self.weight = layer.weight
        self.weight_version = layer.weight._version
        self.q = q
        self.scale = scale

    def eligible(self, layer, x, bias):
        return (
            layer.prefix == PREFIX and bias is None
            and layer.weight is self.weight
            and layer.weight._version == self.weight_version
            and tuple(layer.weight.shape) == (12288, 2048)
            and layer.weight.dtype == torch.bfloat16
            and x.ndim == 2 and x.shape[0] in CONFIGS and x.shape[1] == 2048
            and x.dtype == torch.bfloat16 and x.is_cuda
            and x.device == self.q.device and x.is_contiguous()
        )

    def apply(self, layer, x, bias=None):
        if not self.eligible(layer, x, bias):
            return self.original.apply(layer, x, bias)
        from flag_gems.fused.gateup_w8a16 import gateup_w8a16_kernel
        m = x.shape[0]
        bm, bn, bk, warps, stages = CONFIGS[m]
        out = torch.empty((m, 12288), dtype=x.dtype, device=x.device)
        with torch.cuda.device(x.device):
            gateup_w8a16_kernel[(triton.cdiv(m, bm) * triton.cdiv(12288, bn), 1)](
                x, self.q, out, self.scale, m, 12288, 2048,
                x.stride(0), x.stride(1), self.q.stride(1), self.q.stride(0),
                out.stride(0), out.stride(1), acc_dtype=tl.float32,
                input_precision=None, fp8_fast_accum=True, BLOCK_M=bm,
                BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_K=1, EVEN_K=True,
                AB_DTYPE=tl.bfloat16, UPGRADE=False, UPGRADE_A_OFFS=False,
                UPGRADE_B_OFFS=False, UPGRADE_C_OFFS=False,
                num_warps=warps, num_stages=stages,
            )
        return out


def prepare(layer):
    """No action unless explicitly enabled; synchronous loading phase only."""
    if os.environ.get('VLLM_FL_TASK026_GATEUP_W8A16', '0') != '1':
        return None
    if (layer.prefix != PREFIX or layer.bias is not None
            or tuple(layer.weight.shape) != (12288, 2048)
            or layer.weight.dtype != torch.bfloat16
            or not layer.weight.is_cuda or not layer.weight.is_contiguous()
            or type(layer.quant_method).__name__ != 'UnquantizedLinearMethod'):
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('weight preparation must precede capture')
    started = time.perf_counter()
    q_cpu, scale_cpu = quantize_cpu(layer.weight.detach().cpu())
    q = q_cpu.to(layer.weight.device)
    scale = scale_cpu.to(layer.weight.device)
    torch.cuda.current_stream().synchronize()
    original = layer.quant_method
    method = GateUpW8A16Method(layer, original, q, scale)
    layer.quant_method = method
    return dict(seconds=time.perf_counter() - started,
                original_weight_retained=True,
                original_weight_bytes=layer.weight.numel()*layer.weight.element_size(),
                added_persistent_bytes=q.numel()+scale.numel()*scale.element_size(),
                CPU_FP32_temporaries=True, GPU_FP32_weight_temporary=False,
                preparation_stream=int(torch.cuda.current_stream().cuda_stream))
