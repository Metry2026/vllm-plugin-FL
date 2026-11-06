"""Default-off private forward-bound long-decode split; all else uses parent."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
from pathlib import Path
import sys

import torch
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode

_ACTIVE = ContextVar('task070_current_forward', default=None)
_INSTALLED = False
_BASELINE = _DISPATCH = _FUSED = None
_BACKEND = _FG = _PAGED = _RUNNER_CLASS = _BATCH_CLASS = None
_BUILD_FUNCTION = _EXECUTE_FUNCTION = _MODEL_FUNCTION = None
_SOURCE_PINS = {'vllm.v1.attention.backends.triton_attn': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605', 'vllm.v1.attention.ops.triton_unified_attention': 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0', 'vllm.v1.worker.gpu_input_batch': 'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65', 'vllm_fl.worker.model_runner': '1d01ac497ad946431cc0a063d5e86116573b3c260fc912e811a9372add5ea131', 'vllm_fl.patches.iluvatar_paged_gqa': '3962f8a666bd8fac8100221e8e349a816096a2d8f5b39f7ed1141c03f815d598', 'flag_gems.runtime.backend._iluvatar.fused.paged_gqa_single_stage': 'd56590191021a2dd96b88d772966d17008c8168a3589fa6ce0d2fab9e4200fa8', 'flag_gems.runtime.backend._iluvatar.fused.iluvatar_mixed_decode': 'fe974d1abcb97516320e4a96f31a9244ebe857278cb6d435f761bbceb35c33b2'}


def _observable():
    from torch.nn.modules import module
    return (sys.getprofile() is not None or sys.gettrace() is not None
            or _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
            or torch.autograd._profiler_enabled()
            or module._global_forward_hooks or module._global_forward_pre_hooks
            or module._global_backward_hooks or module._global_backward_pre_hooks)


def _call_model(original, args, kwargs):
    """One unchanged model call; a boundary for controlled paired diagnostics."""
    return original(*args, **kwargs)


def _proof(runner, kwargs, result, frame):
    """Read canonical existing CPU inputs once; never inspect optimistic lengths."""
    n = kwargs.get('num_reqs', 0)
    if (frame is None or frame['runner'] is not runner or frame['in_model'] or n < 2
            or kwargs.get('max_query_len', 0) <= 8
            or not kwargs.get('num_scheduled_tokens')
            or kwargs.get('use_spec_decode') or kwargs.get('for_cudagraph_capture')
            or kwargs.get('ubatch_slices') is not None
            or kwargs.get('num_reqs_padded') not in (None, n)
            or type(runner) is not _RUNNER_CLASS
            or runner._build_attention_metadata is not runner._task070_bound_build
            or runner.execute_model is not runner._task070_bound_execute
            or runner._model_forward is not runner._task070_bound_model
            or runner.speculative_config is not None
            or runner.use_async_spec_decode or _observable()
            or torch.cuda.is_current_stream_capturing()):
        return None
    batch = runner.input_batch
    if type(batch) is not _BATCH_CLASS or batch.num_reqs != n:
        return None
    ids = tuple(batch.req_ids)
    if (len(ids) != n or len(set(ids)) != n
            or any(batch.req_id_to_index.get(rid) != i for i, rid in enumerate(ids))):
        return None
    scheduled = kwargs['num_scheduled_tokens']
    if any(rid not in scheduled or type(scheduled[rid]) is not int
           or scheduled[rid] <= 0 for rid in ids):
        return None
    lengths = tuple(scheduled[rid] for rid in ids)
    # A conservative whole-call domain avoids a new target-mask/H2D path:
    # every single-token row must be a qualified long ordinary decode.
    if 1 not in lengths or max(lengths) != kwargs['max_query_len']:
        return None
    computed = batch.num_computed_tokens_cpu[:n].tolist()
    prompts = batch.num_prompt_tokens_cpu_tensor[:n].tolist()
    if any(c < 0 or p < 0 for c, p in zip(computed, prompts)):
        return None
    if any(q == 1 and (c < 4096 or c < p)
           for q, c, p in zip(lengths, computed, prompts)):
        return None
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    if (cu[-1] != kwargs['num_tokens']
            or runner.query_start_loc.cpu[:n + 1].tolist() != cu):
        return None
    metadata = result[0]
    if not isinstance(metadata, dict) or not metadata:
        return None
    values = tuple(metadata.values())
    first = values[0]
    if any(type(m) is not _BACKEND.TritonAttentionMetadata for m in values):
        return None
    if (first.query_start_loc.shape != (n + 1,)
            or first.seq_lens.shape != (n,)
            or first.block_table.shape[0] != n
            or first.num_actual_tokens != cu[-1]
            or first.max_query_len != max(lengths)
            or first.query_start_loc.data_ptr() != runner.query_start_loc.gpu.data_ptr()
            or first.seq_lens.data_ptr() != runner.seq_lens.data_ptr()
            or any(m.query_start_loc is not first.query_start_loc
                   or m.seq_lens is not first.seq_lens
                   or m.block_table is not first.block_table for m in values)):
        return None
    # Hooks can alter metadata between layers. Check the actual model tree once
    # per forward; its cost is part of this optimization's complete endpoint.
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
           or m._backward_pre_hooks for m in runner.model.modules()):
        return None
    return dict(runner=runner, batch=batch, model=runner.model,
                epoch=runner._task070_forward_epoch, total=cu[-1],
                max_query=max(lengths), cu=first.query_start_loc,
                lens=first.seq_lens, bt=first.block_table,
                stream=torch.cuda.current_stream().cuda_stream)


def _eligible(a, frame):
    proof = None if frame is None else frame.get('proof')
    if (proof is None or not frame['in_model'] or _observable()
            or torch.cuda.is_current_stream_capturing()):
        return False
    runner = proof['runner']
    if (runner._task070_forward_epoch != proof['epoch']
            or runner.input_batch is not proof['batch']
            or runner.model is not proof['model']
            or type(runner)._build_attention_metadata is not _BUILD_FUNCTION
            or type(runner).execute_model is not _EXECUTE_FUNCTION
            or type(runner)._model_forward is not _MODEL_FUNCTION
            or runner._build_attention_metadata is not runner._task070_bound_build
            or runner.execute_model is not runner._task070_bound_execute
            or runner._model_forward is not runner._task070_bound_model
            or _BACKEND.unified_attention is not _DISPATCH
            or _FG.mixed_decode_attention is not _FUSED
            or torch.cuda.current_stream().cuda_stream != proof['stream']
            or a.get('cu_seqlens_q') is not proof['cu']
            or a.get('seqused_k') is not proof['lens']
            or a.get('block_table') is not proof['bt']
            or a.get('max_seqlen_q') != proof['max_query']
            or not _PAGED.supported(a, torch)):
        return False
    q, k, v, out = (a[name] for name in ('q', 'k', 'v', 'out'))
    tensors = (q, k, v, out, proof['cu'], proof['lens'], proof['bt'])
    if (q.shape[0] != proof['total'] or k.shape[1] != 16
            or torch.is_grad_enabled()
            or any(type(t) is not torch.Tensor or t.requires_grad for t in tensors)
            or q.device.index != torch.cuda.current_device()
            or any(torch._C._overlaps(out, t)
                   for t in (q, k, v, proof['cu'], proof['lens'], proof['bt']))):
        return False
    return True


def install(runner):
    global _INSTALLED, _BASELINE, _DISPATCH, _FUSED
    global _BACKEND, _FG, _PAGED, _RUNNER_CLASS, _BATCH_CLASS
    global _BUILD_FUNCTION, _EXECUTE_FUNCTION, _MODEL_FUNCTION
    import os
    if (os.getenv('VLLM_FL_TASK070_MIXED_DECODE', '0') != '1'
            or not getattr(runner, '_task069_private_sampling_installed', False)
            or not getattr(runner, '_task066_unit_temperature_installed', False)):
        return False
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends import triton_attn as backend
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from vllm_fl.patches import iluvatar_paged_gqa as paged
    from flag_gems.runtime.backend._iluvatar.fused import iluvatar_mixed_decode as fg
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
            or type(runner) is not ModelRunnerFL
            or runner.speculative_config is not None):
        return False
    if not _SOURCE_PINS:
        return False
    try:
        if any(hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() != digest
               for name, digest in _SOURCE_PINS.items()):
            return False
    except (OSError, ImportError, AttributeError):
        return False
    original_build = runner._build_attention_metadata
    original_execute = runner.execute_model
    original_model = runner._model_forward
    if (getattr(original_build, '__func__', None)
            is not ModelRunnerFL._build_attention_metadata
            or getattr(original_execute, '__func__', None)
            is not ModelRunnerFL.execute_model
            or getattr(original_model, '__func__', None)
            is not ModelRunnerFL._model_forward):
        return False
    if _INSTALLED:
        if backend.unified_attention is not _DISPATCH:
            return False
    else:
        if not hasattr(backend.unified_attention, 'w006_original'):
            return False
        _BACKEND, _FG, _PAGED = backend, fg, paged
        _RUNNER_CLASS, _BATCH_CLASS = ModelRunnerFL, InputBatch
        _BUILD_FUNCTION = ModelRunnerFL._build_attention_metadata
        _EXECUTE_FUNCTION = ModelRunnerFL.execute_model
        _MODEL_FUNCTION = ModelRunnerFL._model_forward
        _BASELINE = backend.unified_attention
        _FUSED = fg.mixed_decode_attention

        @wraps(_BASELINE)
        def dispatch(*args, **kwargs):
            frame = _ACTIVE.get()
            try:
                eligible = not args and _eligible(kwargs, frame)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                eligible = False
            if not eligible:
                return _BASELINE(*args, **kwargs)
            _FUSED(*(kwargs[name] for name in (
                'q', 'k', 'v', 'out', 'cu_seqlens_q', 'seqused_k',
                'block_table', 'softmax_scale')))
            return None
        _DISPATCH = dispatch
        backend.unified_attention = dispatch
        _INSTALLED = True
    if (getattr(original_build, '__func__', None) is not _BUILD_FUNCTION
            or getattr(original_execute, '__func__', None) is not _EXECUTE_FUNCTION
            or getattr(original_model, '__func__', None) is not _MODEL_FUNCTION):
        return False
    runner._task070_forward_epoch = 0

    @wraps(original_build)
    def build(*args, **kwargs):
        frame = _ACTIVE.get()
        if frame is None:
            return original_build(*args, **kwargs)
        if frame is not None and frame['runner'] is runner:
            frame['proof'] = None
        result = original_build(*args, **kwargs)
        runner._task070_forward_epoch += 1
        if not args:
            try:
                proof = _proof(runner, kwargs, result, frame)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                proof = None
            if frame is not None and frame['runner'] is runner:
                frame['proof'] = proof
        return result

    @wraps(original_model)
    def model(*args, **kwargs):
        frame = _ACTIVE.get()
        proof = (frame.get('proof') if frame is not None
                 and frame['runner'] is runner and not frame['in_model'] else None)
        if proof is None:
            return _call_model(original_model, args, kwargs)
        token = _ACTIVE.set(dict(runner=runner, proof=proof, in_model=True))
        try:
            return _call_model(original_model, args, kwargs)
        finally:
            _ACTIVE.reset(token)
            if frame is not None and frame['runner'] is runner:
                frame['proof'] = None

    @wraps(original_execute)
    def execute(*args, **kwargs):
        runner._task070_forward_epoch += 1
        scheduled = kwargs.get('scheduler_output', args[0] if args else None)
        try:
            counts = tuple(scheduled.num_scheduled_tokens.values())
            potentially_mixed = 1 in counts and max(counts, default=0) > 8
        except (AttributeError, TypeError, ValueError):
            potentially_mixed = False
        if not potentially_mixed and _ACTIVE.get() is None:
            return original_execute(*args, **kwargs)
        token = _ACTIVE.set(dict(runner=runner, proof=None, in_model=False))
        try:
            return original_execute(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    runner._build_attention_metadata = runner._task070_bound_build = build
    runner.execute_model = runner._task070_bound_execute = execute
    runner._model_forward = runner._task070_bound_model = model
    return True
