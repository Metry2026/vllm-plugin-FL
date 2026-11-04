"""Task030 explicit-only mixed cache owner/spec, no production registration."""
from dataclasses import dataclass
import copy
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVQuantMode

@dataclass(frozen=True,kw_only=True)
class MixedVFullAttentionSpec(FullAttentionSpec):
 def __post_init__(self):
  super().__post_init__()
  if (self.block_size,self.num_kv_heads,self.head_size,self.head_size_v)!=(16,2,128,128) or self.dtype!=torch.uint8 or self.kv_quant_mode!=KVQuantMode.NONE:
   raise ValueError('task030 fixed mixed raw-byte page spec')
 @property
 def real_page_size_bytes(self):return 12416
 @classmethod
 def merge(cls,specs):
  if not specs or any(type(s) is not cls or s!=specs[0] for s in specs):raise ValueError('mixed spec identity/fields differ')
  return copy.deepcopy(specs[0])
 def is_uniform_with_collection(self,specs):return all(type(s) is type(self) and s==self for s in specs.values())

class MixedPageOwner:
 def __init__(self,raw):
  if raw.dtype!=torch.uint8 or raw.ndim!=2 or raw.shape[1]!=12416 or raw.stride(1)!=1 or raw.stride(0)<12416 or raw.stride(0)%4 or raw.storage_offset()%4:
   raise ValueError('mixed raw page dimensions/alignment/overlap')
  self.raw=raw;S=raw.stride(0);B=raw.shape[0]
  b=raw.view(torch.bfloat16);q=raw.view(torch.int8);s=raw.view(torch.float32)
  self.k=torch.as_strided(b,(B,16,2,128),(S//2,256,128,1),b.storage_offset())
  self.v=torch.as_strided(q,(B,16,2,128),(S,256,128,1),q.storage_offset()+8192)
  self.scale=torch.as_strided(s,(B,16,2),(S//4,2,1),s.storage_offset()+3072)
  self.signature=(raw.data_ptr(),raw.storage_offset(),tuple(raw.shape),tuple(raw.stride()))
 def views(self):
  r=self.raw
  if (r.data_ptr(),r.storage_offset(),tuple(r.shape),tuple(r.stride()))!=self.signature:raise ValueError('mixed cache owner identity changed')
  return self.k,self.v,self.scale

def write(owner,key,value,slots):
 from flag_gems.fused.mixed_v_writer import mixed_v_writer
 k,v,s=owner.views()
 if key.dtype!=torch.bfloat16 or value.dtype!=torch.bfloat16 or key.shape[1:]!=(2,128) or key.shape!=value.shape or len(slots)>key.shape[0]:raise ValueError('fixed K/V input domain')
 mixed_v_writer[(slots.shape[0],2)](key,value,k,v,s,slots,*key.stride(),*value.stride(),*k.stride(),*v.stride(),*s.stride(),num_warps=4,num_stages=1)
