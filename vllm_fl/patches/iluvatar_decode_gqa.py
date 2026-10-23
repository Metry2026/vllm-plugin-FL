# SPDX-License-Identifier: Apache-2.0
"""W009 bounded BF16 Decode strategy using the audited upstream 3D kernels.

No private/global workspace cache: all buffers remain owned by caller metadata.
Service admission needs independent real Graph/stream ownership validation.
"""
import functools
import hashlib
import importlib.util
from pathlib import Path
import sys


READER_SHA256 = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"


def build_source(original):
    if hashlib.sha256(original).hexdigest() != READER_SHA256:
        raise RuntimeError("W009 D requires the audited upstream reader")
    source = original.decode("utf-8")
    old = "    return 16 if element_size >= 2 else 32"
    if source.count(old) != 1:
        raise RuntimeError("W009 Decode tile seam changed")
    # The dispatch admits BF16/no-window/no-sink Decode only. Other calls use C0.
    return source.replace(old, "    return 32  # W009 BF16 Decode tile; guarded caller", 1).encode()


def load_reader(original_path, directory):
    source = build_source(Path(original_path).read_bytes())
    digest = hashlib.sha256(source).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("decode_" + digest + ".py")
    if path.exists():
        if path.read_bytes() != source:
            raise RuntimeError("W009 generated module already has different bytes")
    else:
        with path.open("xb") as f:
            f.write(source)
    name = "w009_decode_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(base_sha256=READER_SHA256, reader_sha256=digest,
                        recipe="BF16 Decode KV tile 32; original 3D/merge; caller-owned buffers")


def supported(a, torch, segments, base_guard):
    if a.get("max_seqlen_q") != 1 or segments not in (4, 16):
        return False
    if not base_guard(dict(a, max_seqlen_q=2), torch):
        return False
    if any(a.get(n) is not None for n in ("k_descale", "v_descale")):
        return False
    q = a["q"]
    b = a["seqused_k"].shape[0]
    if b > a.get("seq_threshold_3D", -1) or b == 0:
        return False
    shapes = ((b, 16, segments, 128), (b, 16, segments), (b, 16, segments))
    buffers = []
    for name, shape in zip(("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"), shapes):
        buf = a.get(name)
        if (buf is None or buf.device != q.device or buf.dtype != torch.float32
                or not buf.is_contiguous() or buf.numel() < _product(shape)):
            return False
        buffers.append(buf)
    intervals = [_interval(t) for t in (*buffers, a['q'], a['k'], a['v'], a['out'])]
    for index in range(3):
        lo, hi = intervals[index]
        if any(lo < other_hi and other_lo < hi
               for other_lo, other_hi in intervals[index + 1:]):
            return False
    return True


def _interval(t):
    # Include padding/interleaving in K/V views, rather than assuming numel
    # covers their whole storage span. This may conservatively reject holes.
    span = (1 + sum((n - 1) * stride for n, stride in zip(t.shape, t.stride()))) if t.numel() else 0
    return t.data_ptr(), t.data_ptr() + span * t.element_size()


def _product(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def make_dispatch(c0, reader, torch, base_guard, *, segments=4, audit=False):
    if segments not in (4, 16):
        raise ValueError("P1 D only J4/J16, KV tile32")
    counts = dict(candidate=0, fallback=0)

    @functools.wraps(c0)
    def dispatch(*args, **kwargs):
        hit = not args and supported(kwargs, torch, segments, base_guard)
        if audit:
            counts["candidate" if hit else "fallback"] += 1
        if not hit:
            return c0(*args, **kwargs)
        reader(**dict(kwargs, num_par_softmax_segments=segments))
        return None

    dispatch.w009_original = c0
    dispatch.w009_counts = counts
    dispatch.w009_strategy = dict(segments=segments, kv_tile=32, grouped_heads=8,
                                 workspace="caller-owned; no allocation/cache in dispatch")
    return dispatch
