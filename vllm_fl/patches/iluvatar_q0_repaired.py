# SPDX-License-Identifier: Apache-2.0
"""Component-only Q0 local workaround, unregistered and disabled by default.

Produce an independently identified module from exactly one audited upstream
reader. The compiler pipeline and installed upstream files are unchanged.
Model/service admission is deliberately separate from this source factory.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys


BASE_READER_SHA256 = "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
RECIPE = "int8-no-sink-2d-L0-visible-zero-row-empty-v1"


def build_source(original: bytes) -> bytes:
    """Three guarded edits; every other upstream byte is preserved."""
    if hashlib.sha256(original).hexdigest() != BASE_READER_SHA256:
        raise RuntimeError("Q0-repaired requires the audited upstream source")
    source = original.decode("utf-8")
    edits = (
        (
            "    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)",
            "    if (not IS_3D) and (not USE_SINKS) and KV_QUANT_MODE == 2:\n"
            "        L = tl.full([BLOCK_M], 0.0, dtype=tl.float32)\n"
            "    else:\n"
            "        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)",
        ),
        (
            "        acc = acc / L[:, None]",
            "        if (not IS_3D) and (not USE_SINKS) and KV_QUANT_MODE == 2:\n"
            "            visible_kv = tl.maximum(0, tl.minimum(seq_len, context_len + query_pos + 1))\n"
            "            zero_row = visible_kv == 0\n"
            "            safe_denominator = tl.where(zero_row, 1.0, L)\n"
            "            acc = tl.where(zero_row[:, None], 0.0, acc / safe_denominator[:, None])\n"
            "        else:\n"
            "            acc = acc / L[:, None]",
        ),
        (
            "    # Resolve causal: bool or per-seq tensor.",
            "    if kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD and sinks is None and q.shape[0] == 0:\n"
            "        return None\n"
            "    # Resolve causal: bool or per-seq tensor.",
        ),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError("Q0-repaired source seam changed")
        source = source.replace(old, new, 1)
    return source.encode("utf-8")


def load_reader(original_path, generated_path):
    """Load a new local module; callers own its path, gates and dispatch."""
    original_path, generated_path = Path(original_path), Path(generated_path)
    source = build_source(original_path.read_bytes())
    with generated_path.open("xb") as handle:
        handle.write(source)
    digest = hashlib.sha256(source).hexdigest()
    name = "iluvatar_q0_repaired_" + digest
    spec = importlib.util.spec_from_file_location(name, generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    identity = dict(recipe=RECIPE, base_reader_sha256=BASE_READER_SHA256,
                    reader_sha256=digest, factory_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    compiler_pipeline="unchanged; no pass ablation", registered_in_engine=False)
    return module, identity
