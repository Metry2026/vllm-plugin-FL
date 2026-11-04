"""Opt-in task030 complete persistent mixed-V cache integration."""
from dataclasses import fields
from pathlib import Path
import hashlib,json,os
import torch

_INSTALLED=False
_OWNERS={}
_RUNNER_KEYS={}
_VALIDATED_CONFIGS=set()
_AUDIT_ROOT=os.environ.get('TASK030_LIFECYCLE_AUDIT_DIR')
_NEW_BLOCK_AUDITED=False

def _audit(name,data):
 if _AUDIT_ROOT:
  p=Path(_AUDIT_ROOT);p.mkdir(parents=True,exist_ok=True)
  (p/(name+'-'+str(os.getpid())+'.json')).write_text(json.dumps(data,indent=2,default=str)+'\n')

def _config(c):
 if id(c) in _VALIDATED_CONFIGS:return
 from vllm.config import CUDAGraphMode,CompilationMode
 m=c.model_config;h=m.hf_config;p=c.parallel_config;cc=c.cache_config
 assert h.architectures==['LlamaForCausalLM'] and (h.hidden_size,h.num_hidden_layers,h.num_attention_heads,h.num_key_value_heads)==(2048,42,16,2)
 assert m.dtype==torch.bfloat16 and m.quantization is None and not m.use_mla
 assert hashlib.sha256((Path(m.model)/'config.json').read_bytes()).hexdigest()=='59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180'
 assert (p.tensor_parallel_size,p.pipeline_parallel_size,p.data_parallel_size,p.decode_context_parallel_size)==(1,1,1,1) and not p.use_ubatching
 assert cc.block_size==16 and cc.cache_dtype=='auto' and not cc.kv_sharing_fast_prefill
 assert c.kv_transfer_config is None and c.speculative_config is None and c.lora_config is None
 assert c.compilation_config.cudagraph_mode==CUDAGraphMode.FULL_DECODE_ONLY and c.compilation_config.mode==CompilationMode.VLLM_COMPILE
 _VALIDATED_CONFIGS.add(id(c))

def _owner(cache):
 owner=_OWNERS.get(id(cache))
 if owner is None or owner.raw is not cache:raise RuntimeError('task030 cache was not allocated/bound by mixed spec; no old-format fallback')
 return owner

def install():
 global _INSTALLED
 if os.environ.get('VLLM_FL_TASK030_MIXED_V_CACHE','0')!='1':return False
 if _INSTALLED:return True
 from vllm.platforms import current_platform
 if getattr(current_platform,'vendor_name',None)!='iluvatar':raise RuntimeError('task030 requires iluvatar')
 from vllm.model_executor.layers.attention import Attention
 from vllm.v1.attention.backends import triton_attn as backend
 from vllm.v1.kv_cache_interface import FullAttentionSpec
 from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
 from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
 from vllm_fl.worker.model_runner import ModelRunnerFL
 from vllm_fl.ops.mixed_v_cache import MixedVFullAttentionSpec,MixedPageOwner,write
 from vllm_fl.ops.mixed_v_unified_attention import unified_attention as native
 from vllm_fl.patches.iluvatar_paged_gqa import supported
 from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_mixed_v import paged_gqa_attention as scalar
 KVCacheSpecRegistry.register(MixedVFullAttentionSpec,FullAttentionManager,uniform_type_base_spec=MixedVFullAttentionSpec)
 import inspect
 assert hashlib.sha256(Path(inspect.getsourcefile(ModelRunnerFL)).read_bytes()).hexdigest()=='fe1679644f096ad3559277468fe2607ac44ddaee000c75f71ae78ac1aa151117'
 assert hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest()=='1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605'
 old_spec=Attention.get_kv_cache_spec
 old_runner_spec=ModelRunnerFL.get_kv_cache_spec
 old_init=ModelRunnerFL.initialize_kv_cache_tensors
 old_reshape=ModelRunnerFL._reshape_kv_cache_tensors
 old_new=FullAttentionManager.allocate_new_blocks
 old_writer=backend.TritonAttentionImpl.do_kv_cache_update
 old_reader=backend.TritonAttentionImpl.forward
 def spec(layer,config):
  _config(config);s=old_spec(layer,config)
  if s is None:return None
  assert type(s) is FullAttentionSpec and layer.layer_name.startswith('model.layers.') and layer.layer_name.endswith('.self_attn.attn')
  impl=layer.impl
  assert (impl.num_heads,impl.num_kv_heads,impl.head_size)==(16,2,128)
  assert impl.sliding_window==(-1,-1) and impl.alibi_slopes is None and impl.sinks is None and impl.logits_soft_cap==0 and not impl.use_alibi_sqrt and impl.chunk_lookback==-1 and not impl.use_td
  assert not impl.fused_rope_kvcache_supported() and not impl._is_per_token_head_quant
  values={f.name:getattr(s,f.name) for f in fields(s)};values['dtype']=torch.uint8
  return MixedVFullAttentionSpec(**values)
 def runner_spec(r):
  _config(r.vllm_config);v=old_runner_spec(r)
  assert len(v)==42 and all(type(s) is MixedVFullAttentionSpec and s.page_size_bytes==12416 for s in v.values())
  return v
 def init(r,*a,**kw):
  for k in _RUNNER_KEYS.pop(id(r),[]):_OWNERS.pop(k,None)
  return old_init(r,*a,**kw)
 def reshape(r,raws,kernel_sizes):
  _config(r.vllm_config);result={};keys=[];descriptions=[]
  packing={}
  for t in r.kv_cache_config.kv_cache_tensors:
   if t.block_stride>0:
    for name in t.shared_by:packing[name]=(t.offset,t.block_stride)
  for group in r._kv_cache_spec_attn_group_iterator():
   s=group.kv_cache_spec
   assert type(s) is MixedVFullAttentionSpec and kernel_sizes[group.kv_cache_group_id]==16
   for name in group.layer_names:
    source=raws[name];assert source.dtype in (torch.int8,torch.uint8)
    # The precise parent allocator owns signed byte storage; reinterpret only.
    source=source.view(torch.uint8)
    if name in packing:
     offset,stride=packing[name];assert offset%4==0 and stride%4==0
     raw=source.view(-1,stride)[:,offset:offset+12416]
    else:raw=source.view(-1,12416)
    owner=MixedPageOwner(raw);_OWNERS[id(raw)]=owner;keys.append(id(raw));result[name]=raw
    descriptions.append(dict(layer=name,shape=list(raw.shape),stride=list(raw.stride()),offset=raw.storage_offset(),storage_ptr=raw.untyped_storage().data_ptr(),K=list(owner.k.stride()),V=list(owner.v.stride()),scale=list(owner.scale.stride())))
  assert len(result)==42;_RUNNER_KEYS[id(r)]=keys
  _audit('allocation',dict(layers=descriptions,raw_payload=sum(t.numel() for t in result.values()),num_blocks=r.kv_cache_config.num_blocks,kv_transfer_config=str(r.vllm_config.kv_transfer_config)))
  return result
 def new_blocks(manager,*a,**kw):
  global _NEW_BLOCK_AUDITED
  new=old_new(manager,*a,**kw)
  if type(manager.kv_cache_spec) is MixedVFullAttentionSpec:
   # The parent exact-type tuple deliberately did not append for this new type.
   manager.new_block_ids.extend(b.block_id for b in new)
   if new and not _NEW_BLOCK_AUDITED:
    _audit('new-blocks',dict(spec_type=type(manager.kv_cache_spec).__name__,ids=[b.block_id for b in new],registered_ids=list(manager.new_block_ids),page_bytes=manager.kv_cache_spec.page_size_bytes));_NEW_BLOCK_AUDITED=True
  return new
 def writer(impl,layer,key,value,cache,slots):
  if cache.numel()==0:return old_writer(impl,layer,key,value,cache,slots)
  write(_owner(cache),key,value,slots)
 def reader(impl,layer,query,key,value,cache,md,output,output_scale=None,output_block_scale=None):
  if md is None:return old_reader(impl,layer,query,key,value,cache,md,output,output_scale,output_block_scale)
  assert output_scale is None and output_block_scale is None and not md.use_cascade
  k,v,s=_owner(cache).views();T=md.num_actual_tokens
  a=dict(q=query[:T],k=k,v=v,out=output[:T],cu_seqlens_q=md.query_start_loc,max_seqlen_q=md.max_query_len,seqused_k=md.seq_lens,max_seqlen_k=md.max_seq_len,softmax_scale=impl.scale,causal=md.causal,window_size=impl.sliding_window,block_table=md.block_table,softcap=impl.logits_soft_cap,q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=md.seq_threshold_3D,num_par_softmax_segments=md.num_par_softmax_segments,softmax_segm_output=md.softmax_segm_output,softmax_segm_max=md.softmax_segm_max,softmax_segm_expsum=md.softmax_segm_expsum,alibi_slopes=impl.alibi_slopes,output_scale=None,qq_bias=None,sinks=impl.sinks,mm_prefix_range=md.mm_prefix_range_tensor,use_alibi_sqrt=impl.use_alibi_sqrt,chunk_lookback=impl.chunk_lookback,use_td=impl.use_td)
  # Owner validates complete mixed V bounds/strides. Reuse the parent's Q/K,
  # output and metadata policy with the proven same-shaped K as dtype witness.
  if supported(dict(a,v=k),torch):
   scalar(a['q'],k,v,s,a['out'],a['cu_seqlens_q'],a['seqused_k'],a['block_table'],a['softmax_scale'],block_q=8,block_n=32)
  else:native(**a,v_scale_cache=s,mixed_v_int8=True)
  return output
 Attention.get_kv_cache_spec=spec
 ModelRunnerFL.get_kv_cache_spec=runner_spec
 ModelRunnerFL.initialize_kv_cache_tensors=init
 ModelRunnerFL._reshape_kv_cache_tensors=reshape
 FullAttentionManager.allocate_new_blocks=new_blocks
 backend.TritonAttentionImpl.do_kv_cache_update=writer
 backend.TritonAttentionImpl.forward=reader
 _INSTALLED=True
 _audit('installed',dict(module=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runner_base=inspect.getsourcefile(ModelRunnerFL),manager_base=inspect.getsourcefile(FullAttentionManager),parent_rope_fusion='requires false before mixed spec allocation',default_enabled=False))
 return True
