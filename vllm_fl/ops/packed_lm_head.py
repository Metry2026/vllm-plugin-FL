"""Task031 first variant: frozen actual-head prototype, never auto-registered.

Only a pre-bound R8 domain uses the parent's identical JIT/config with different
weight strides. Production post-load/reload integration is not claimed here.
"""
import types
import torch
from flag_gems.runtime import torch_device_fn


def version(weight):
    try:
        return weight._version
    except RuntimeError:
        return None


class FrozenPackedLMHead:
    def __init__(self, head, parent_jit, config):
        weight = head.weight
        assert tuple(weight.shape) == (130560, 2048)
        assert weight.dtype == torch.bfloat16 and weight.device.type == 'cuda'
        assert weight.stride() == (2048, 1) and weight.storage_offset() == 0
        assert config['SPLIT_K'] == 1 and config['GROUP_M'] == 8
        self.head = head
        self.weight = weight  # Strong object binding, not an address cache.
        self.weight_version = version(weight)
        self.weight_pointer = weight.data_ptr()
        self.parent_jit = parent_jit
        self.config = dict(config)
        self.original_apply = head.quant_method.apply
        self.original_own_apply = 'apply' in head.quant_method.__dict__
        self.packed = weight.detach().T.contiguous()
        self.actual_compiled = None
        self.installed = False
        self.frozen_diagnostic_scope = True

    def eligible(self, layer, x, bias):
        return (
            self.frozen_diagnostic_scope and layer is self.head
            and layer.weight is self.weight
            and version(layer.weight) == self.weight_version
            and layer.weight.data_ptr() == self.weight_pointer
            and tuple(layer.weight.shape) == (130560, 2048)
            and layer.weight.stride() == (2048, 1)
            and layer.weight.storage_offset() == 0
            and layer.weight.dtype == torch.bfloat16
            and x.ndim == 2 and tuple(x.shape) == (8, 2048)
            and x.stride() == (2048, 1) and x.storage_offset() == 0
            and x.dtype == torch.bfloat16 and x.device == self.weight.device
            and bias is None and self.packed.shape == (2048, 130560)
            and self.packed.stride() == (130560, 1)
            and self.packed.dtype == torch.bfloat16
            and self.packed.device == self.weight.device
        )

    def apply(self, layer, x, bias=None):
        if not self.eligible(layer, x, bias):
            return self.original_apply(layer, x, bias)
        output = torch.empty((8, 130560), dtype=x.dtype, device=x.device)
        c = self.config
        grid = ((8 + c['BLOCK_M'] - 1) // c['BLOCK_M']
                * ((130560 + c['BLOCK_N'] - 1) // c['BLOCK_N']), 1)
        input_flat = x.view(8, 2048)
        with torch_device_fn.device(x.device):
            compiled = self.parent_jit[grid](
                input_flat, self.packed, self.packed, output, 8, 130560, 2048,
                input_flat.stride(0), input_flat.stride(1), 1, 130560,
                output.stride(0), output.stride(1), 0,
                BIAS=False, BLOCK_M=c['BLOCK_M'], BLOCK_N=c['BLOCK_N'],
                BLOCK_K=c['BLOCK_K'], GROUP_M=c['GROUP_M'], SPLIT_K=1,
                EVEN_K=True, num_warps=c['num_warps'],
                num_stages=c['num_stages'], num_ctas=c['num_ctas'],
            )
        if self.actual_compiled is None:
            self.actual_compiled = compiled
        return output.view(8, 130560)

    def install(self):
        if not self.installed:
            owner = self
            def apply(_method, layer, x, bias=None):
                return owner.apply(layer, x, bias)
            self.head.quant_method.apply = types.MethodType(apply, self.head.quant_method)
            self.installed = True

    def restore(self):
        if self.installed:
            if self.original_own_apply:
                self.head.quant_method.apply = self.original_apply
            else:
                del self.head.quant_method.apply
            self.installed = False
