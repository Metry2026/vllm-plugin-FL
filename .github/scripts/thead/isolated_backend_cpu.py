# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the shared reference operators selected by THead policy.

The fixture isolates accelerator-dependent package initialization and device
configuration discovery. It executes the real registry, policy parser, dispatch
manager, backend registrations and reference math. Numeric code is not replaced.
The explicit registry tests dispatch selection and failure fallback, but does not
claim coverage of plugin startup, hardware discovery or device kernels.
"""

import functools
import importlib
import importlib.machinery
import math
import os
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def backend():
    original = {
        k: v
        for k, v in sys.modules.items()
        if k == "vllm_fl" or k.startswith("vllm_fl.")
    }
    assert not original, "Run this explicit isolated suite in its own pytest process"
    try:
        # Namespace shells specify only package paths; all tested modules/classes
        # execute from their actual source files, without AST or method extraction.
        for name in [
            "vllm_fl",
            "vllm_fl.dispatch",
            "vllm_fl.dispatch.backends",
            "vllm_fl.dispatch.backends.vendor",
            "vllm_fl.dispatch.backends.vendor.thead",
            "vllm_fl.dispatch.backends.reference",
            "vllm_fl.dispatch.backends.reference.impl",
        ]:
            package = ModuleType(name)
            package.__path__ = [str(ROOT / name.replace(".", "/"))]
            package.__package__ = name
            package.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            sys.modules[name] = package
        # The explicit YAML policy bypasses hardware configuration discovery.
        # Raising here prevents the stub from silently choosing a different policy.
        utils = ModuleType("vllm_fl.utils")

        def reject_device_discovery():
            raise AssertionError(
                "CPU dispatch tests must use the explicit THead policy"
            )

        utils.get_op_config = reject_device_discovery
        sys.modules[utils.__name__] = utils
        registry_module = importlib.import_module("vllm_fl.dispatch.registry")
        policy_module = importlib.import_module("vllm_fl.dispatch.policy")
        manager_module = importlib.import_module("vllm_fl.dispatch.manager")
        reference = importlib.import_module(
            "vllm_fl.dispatch.backends.reference.register_ops"
        )
        thead = importlib.import_module(
            "vllm_fl.dispatch.backends.vendor.thead.register_ops"
        )
        op_types = importlib.import_module("vllm_fl.dispatch.types")
        policy = policy_module.policy_from_config(
            str(ROOT / "vllm_fl/dispatch/config/thead.yaml")
        )

        def make_manager():
            registry = registry_module.OpRegistry()
            reference.register_builtins(registry)
            thead.register_builtins(registry)
            manager = manager_module.OpManager(registry)
            # Registration above is intentional: hardware/plugin discovery is
            # outside this isolated test, while selection and call stay real.
            manager._state.initialized = True
            manager._state.init_pid = os.getpid()
            return manager

        manager = make_manager()
        backend = SimpleNamespace(
            make_manager=make_manager, policy=policy, op_types=op_types
        )
        for op_name in ("rms_norm", "rotary_embedding", "silu_and_mul"):
            setattr(backend, op_name, functools.partial(manager.call, op_name))
        with policy_module.policy_context(policy):
            yield backend
    finally:
        for name in list(sys.modules):
            if name == "vllm_fl" or name.startswith("vllm_fl."):
                sys.modules.pop(name, None)
        sys.modules.update(original)


def _tolerance(dtype):
    return {
        torch.float32: (3e-6, 3e-6),
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _rms_oracle(x, residual, weight, epsilon, variance_size, weighted):
    # Scalar arithmetic provides an independent mathematical oracle. Precision
    # boundaries are part of the operator's mixed-dtype public contract.
    values = x.reshape(-1, x.shape[-1]).double().tolist()
    additions = (
        None
        if residual is None
        else residual.reshape(-1, x.shape[-1]).double().tolist()
    )
    result, sums = [], []
    for row_number, row in enumerate(values):
        row = [_float32(value) for value in row]
        if additions is not None:
            row = [
                _float32(value + _float32(additions[row_number][i]))
                for i, value in enumerate(row)
            ]
        count = len(row) if variance_size is None else variance_size
        inverse = 1.0 / math.sqrt(
            sum(value * value for value in row[:count]) / count + epsilon
        )
        normalized = [value * inverse for value in row]
        if weighted:
            rounded = torch.tensor(normalized, dtype=weight.dtype).double().tolist()
            normalized = [value * float(weight[i]) for i, value in enumerate(rounded)]
            normalized = torch.tensor(normalized, dtype=weight.dtype).double().tolist()
        result.append(normalized)
        sums.append(row)
    output = torch.tensor(result, dtype=x.dtype).reshape(x.shape)
    updated = (
        None if residual is None else torch.tensor(sums, dtype=x.dtype).reshape(x.shape)
    )
    return output, updated


@pytest.mark.parametrize(
    "x_dtype,weight_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("residual_mode", ["none", "same", "fp32"])
@pytest.mark.parametrize(
    "pass_weight,pass_weight_add",
    [(True, True), (False, True), (True, False), (False, False)],
)
@pytest.mark.parametrize("variance_size", [None, 3])
def test_reference_dispatch_rms_mixed_dtype_residual_flags_variance(
    backend,
    x_dtype,
    weight_dtype,
    residual_mode,
    pass_weight,
    pass_weight_add,
    variance_size,
):
    x = (
        torch.linspace(-3.25, 4.75, 96, dtype=torch.float32)
        .reshape(2, 3, 16)[..., ::2]
        .to(x_dtype)
    )
    residual = (
        None
        if residual_mode == "none"
        else torch.linspace(0.17, -0.63, x.numel())
        .reshape(x.shape)
        .to(torch.float32 if residual_mode == "fp32" else x_dtype)
    )
    weight = torch.tensor(
        [0.3, -0.7, 1.2, 0.6, -1.4, 0.9, 1.7, -0.2], dtype=weight_dtype
    )
    obj = SimpleNamespace(
        weight=weight,
        variance_epsilon=0.07,
        variance_size_override=variance_size,
        pass_weight=pass_weight,
        pass_weight_add=pass_weight_add,
    )
    saved_x = x.clone()
    saved_residual = None if residual is None else residual.clone()
    expected, expected_residual = _rms_oracle(
        x,
        residual,
        weight,
        obj.variance_epsilon,
        variance_size,
        pass_weight if residual is None else pass_weight_add,
    )
    actual = backend.rms_norm(obj, x, residual)
    if residual is None:
        assert isinstance(actual, torch.Tensor)
        output = actual
    else:
        assert isinstance(actual, tuple) and len(actual) == 2
        output, updated = actual
        assert updated.dtype == x_dtype and updated.device.type == "cpu"
        torch.testing.assert_close(updated, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
    assert (
        output.dtype == x_dtype
        and output.shape == x.shape
        and output.device.type == "cpu"
    )
    effective_dtype = (
        weight_dtype
        if (pass_weight if residual is None else pass_weight_add)
        and x_dtype == torch.float32
        else x_dtype
    )
    # A weighted fp32 output retains the weight dtype's quantization boundary.
    rtol, atol = _tolerance(effective_dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)


def _rotary_oracle(tensor, cos, sin, positions, interleaved):
    result = tensor.clone()
    half = tensor.shape[-1] // 2
    # Independent per-pair 2x2 rotations, rather than a vectorized copied formula.
    import itertools

    for index in itertools.product(*(range(size) for size in tensor.shape[:-1])):
        position = int(
            positions[index[0]]
            if tensor.ndim == 3
            else positions[index[2]]
            if positions.ndim == 1
            else positions[index[0], index[2]]
        )
        for pair in range(half):
            left, right = (
                (2 * pair, 2 * pair + 1) if interleaved else (pair, pair + half)
            )
            cosine = float(cos[position, pair].to(tensor.dtype))
            sine = float(sin[position, pair].to(tensor.dtype))
            a = float(tensor[index + (left,)])
            b = float(tensor[index + (right,)])
            result[index + (left,)] = a * cosine - b * sine
            result[index + (right,)] = b * cosine + a * sine
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["3d", "4d_shared", "4d_batched"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["neox", "gptj"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("partial_tail", [False, True])
def test_reference_dispatch_rotary_layout_style_inplace_caller_view_tail(
    backend, dtype, layout, interleaved, inplace, partial_tail
):
    rotary_size = 8
    width = 12 if partial_tail else rotary_size
    qshape = (4, 3, width) if layout == "3d" else (2, 3, 4, width)
    kshape = (4, 2, width) if layout == "3d" else (2, 2, 4, width)
    query = (
        torch.linspace(-1.75, 2.25, math.prod(qshape) * 2)
        .reshape(qshape[:-1] + (width * 2,))[..., ::2]
        .to(dtype)
    )
    key = (
        torch.linspace(1.25, -2.5, math.prod(kshape) * 2)
        .reshape(kshape[:-1] + (width * 2,))[..., ::2]
        .to(dtype)
    )
    before_q, before_k = query.clone(), key.clone()
    angles = [
        [
            (position + 1) * (frequency + 1) * 0.17
            for frequency in range(rotary_size // 2)
        ]
        for position in range(7)
    ]
    cos = torch.tensor(
        [[math.cos(angle) for angle in row] for row in angles], dtype=torch.float64
    )
    sin = torch.tensor(
        [[math.sin(angle) for angle in row] for row in angles], dtype=torch.float64
    )
    positions = torch.tensor(
        [[0, 4, 1, 3], [6, 2, 5, 0]] if layout == "4d_batched" else [0, 4, 1, 3],
        dtype=torch.int64,
    )
    # The FL caller passes the rotary prefix view and uses the returned tensors.
    # Reference dispatch keeps its existing out-of-place contract for either
    # inplace argument, including when the input is a view with an unpassed tail.
    qview, kview = query[..., :rotary_size], key[..., :rotary_size]
    expected_q = _rotary_oracle(qview, cos, sin, positions, interleaved)
    expected_k = _rotary_oracle(kview, cos, sin, positions, interleaved)
    actual_q, actual_k = backend.rotary_embedding(
        None,
        qview,
        kview,
        cos,
        sin,
        positions,
        rotary_interleaved=interleaved,
        inplace=inplace,
    )
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)
    assert (
        actual_q.dtype == actual_k.dtype == dtype
        and actual_q.device.type == actual_k.device.type == "cpu"
    )
    assert (
        actual_q.data_ptr() != qview.data_ptr()
        and actual_k.data_ptr() != kview.data_ptr()
    )
    torch.testing.assert_close(query, before_q, rtol=0, atol=0)
    torch.testing.assert_close(key, before_k, rtol=0, atol=0)
    if partial_tail:
        torch.testing.assert_close(
            query[..., rotary_size:], before_q[..., rotary_size:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            key[..., rotary_size:], before_k[..., rotary_size:], rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16)])
def test_reference_dispatch_silu_numeric_oracle(backend, dtype, shape):
    x = (
        torch.linspace(-4.25, 3.75, math.prod(shape) * 2)
        .reshape(shape[:-1] + (shape[-1] * 2,))[..., ::2]
        .to(dtype)
    )
    original = x.clone()
    half = shape[-1] // 2
    expected_rows = []
    for row in x.reshape(-1, shape[-1]).double().tolist():
        activated = [value / (1.0 + math.exp(-value)) for value in row[:half]]
        rounded = torch.tensor(activated, dtype=dtype).double().tolist()
        expected_rows.append(
            [value * row[half + index] for index, value in enumerate(rounded)]
        )
    expected = torch.tensor(expected_rows, dtype=dtype).reshape(shape[:-1] + (half,))
    actual = backend.silu_and_mul(None, x)
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert (
        actual.dtype == dtype
        and actual.shape == expected.shape
        and actual.device.type == "cpu"
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_reference_dispatch_rms_defaults_apply_weight(backend, dtype, with_residual):
    x = torch.tensor(
        [[0.31, -1.27, 2.43, -0.57], [1.11, 0.43, -0.89, 1.91]], dtype=dtype
    )
    residual = (
        torch.tensor(
            [[0.17, -0.31, 0.11, 0.23], [-0.09, 0.37, -0.13, 0.29]], dtype=torch.float32
        )
        if with_residual
        else None
    )
    weight = torch.tensor([0.3, -0.7, 1.2, 0.6], dtype=torch.float32)
    # Both flag attributes and variance_size_override are genuinely absent.
    obj = SimpleNamespace(weight=weight, variance_epsilon=0.07)
    expected, expected_residual = _rms_oracle(
        x, residual, weight, obj.variance_epsilon, None, True
    )
    actual = backend.rms_norm(obj, x, residual)
    output = actual[0] if with_residual else actual
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    if with_residual:
        torch.testing.assert_close(actual[1], expected_residual, rtol=0, atol=0)


def test_thead_registers_only_vendor_attention(backend):
    manager = backend.make_manager()
    vendor_ops = {
        op_name
        for op_name, impls in manager.registry.snapshot().impls_by_op.items()
        if any(impl.impl_id == "vendor.thead" for impl in impls)
    }
    assert vendor_ops == {"attention_backend"}
    assert backend.policy.get_per_op_order("attention_backend")[0] == "vendor:thead"
    for op_name in ("rms_norm", "rotary_embedding", "silu_and_mul"):
        assert backend.policy.get_per_op_order(op_name) == ["flagos", "reference"]
        assert manager.get_selected_impl_id(op_name) == "reference.torch"


@pytest.mark.parametrize("op_name", ["rms_norm", "rotary_embedding", "silu_and_mul"])
@pytest.mark.parametrize("flagos_state", ["available", "unavailable", "raises"])
def test_thead_dispatch_prefers_flagos_and_falls_back_to_reference(
    backend, op_name, flagos_state
):
    manager = backend.make_manager()
    calls = []
    sentinel = object()

    def flagos_impl(*args, **kwargs):
        calls.append(op_name)
        if flagos_state == "raises":
            raise NotImplementedError("unsupported test input")
        return sentinel

    flagos_impl._is_available = lambda: flagos_state != "unavailable"
    manager.registry.register_impl(
        backend.op_types.OpImpl(
            op_name=op_name,
            impl_id="default.flagos",
            kind=backend.op_types.BackendImplKind.DEFAULT,
            priority=backend.op_types.BackendPriority.DEFAULT,
            fn=flagos_impl,
        )
    )
    if op_name == "rms_norm":
        x = torch.ones(2, 4)
        args = (SimpleNamespace(weight=torch.ones(4), variance_epsilon=0.0), x)
        expected = x
    elif op_name == "rotary_embedding":
        query, key = torch.ones(2, 3, 4), torch.full((2, 1, 4), 2.0)
        args = (None, query, key, torch.ones(2, 2), torch.zeros(2, 2), torch.arange(2))
        expected = (query, key)
    else:
        args = (None, torch.zeros(2, 8))
        expected = torch.zeros(2, 4)
    actual = manager.call(op_name, *args)
    if flagos_state == "available":
        assert actual is sentinel and calls == [op_name]
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert calls == ([op_name] if flagos_state == "raises" else [])
        if flagos_state == "raises":
            assert manager.get_failed_impls(op_name) == {op_name: {"default.flagos"}}
            # A failed kernel is not retried on every subsequent call.
            torch.testing.assert_close(manager.call(op_name, *args), expected)
            assert calls == [op_name]
