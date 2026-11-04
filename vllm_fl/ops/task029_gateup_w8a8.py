"""Explicitly called task029 prototype. No registration or default enablement."""
import torch

def quantize_weight_cpu(weight):
 if weight.ndim!=2 or weight.shape!=(12288,2048) or weight.dtype!=torch.bfloat16 or weight.device.type!='cpu':
  raise ValueError('fixed layer0 gate/up W[12288,2048] CPU BF16 required')
 w=weight.float()
 if not bool(torch.isfinite(w).all()):raise ValueError('nonfinite weight rejected before quantization')
 scale=w.abs().amax(1)/127
 scale=torch.where(scale==0,torch.ones_like(scale),scale)
 q=torch.clamp(torch.round(w/scale[:,None]),-127,127).to(torch.int8)
 return q.t().contiguous(),scale.contiguous()

def apply(x,b,scale):
 from flag_gems.fused.gateup_w8a8_fixed import quantize_rows,gateup_w8a8_kernel
 if x.ndim!=2 or x.shape[1]!=2048 or not x.is_contiguous() or x.dtype!=torch.bfloat16:
  raise ValueError('prototype requires contiguous BF16 [M,2048]')
 m=x.shape[0]
 if m==0:return torch.empty((0,12288),dtype=x.dtype,device=x.device)
 q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((m,),device=x.device,dtype=torch.float32)
 y=torch.empty((m,12288),device=x.device,dtype=torch.bfloat16)
 quantize_rows[(m,)](x,q,s,m,2048,num_warps=4,num_stages=1)
 gateup_w8a8_kernel[((m+31)//32,384)](q,b,s,scale,y,y,m,12288,False,num_warps=4,num_stages=1)
 return y
