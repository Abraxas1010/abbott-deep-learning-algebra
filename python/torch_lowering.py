from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

try:
    from .json_schema import BroadcastedOperationSchema, ReindexingSchema
except ImportError:
    from json_schema import BroadcastedOperationSchema, ReindexingSchema


@dataclass(slots=True)
class LoweringStep:
    name: str
    details: dict[str, Any]


@dataclass(slots=True)
class TorchLoweringPlan:
    op_name: str
    explicit_broadcast_dims: list[list[int]]
    weave_orders: dict[str, list[list[int]]]
    notes: list[str]
    steps: list[LoweringStep] = field(default_factory=list)
    backend_preference: str = "reference"


def _axis_size_map(schema: BroadcastedOperationSchema) -> dict[int, int]:
    return {axis.uid: axis.size for axis in schema.uid_config.axes}


def _expected_shape(axes: list[Any]) -> tuple[int, ...]:
    return tuple(axis.size for axis in axes)


def _validate_tensor_shape(name: str, tensor: np.ndarray, expected: tuple[int, ...]) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, got {tensor.shape}")


def _default_steps(schema: BroadcastedOperationSchema) -> list[LoweringStep]:
    if schema.name == "convolution":
        return [
            LoweringStep(
                name="extract_windows",
                details={
                    "reindexing": schema.reindexings[0].cod_axes if schema.reindexings else [],
                    "mode": schema.uid_config.op_params.get(11, "error"),
                },
            ),
            LoweringStep(
                name="contract_kernel",
                details={"einsum": "bclk,ck->bcl"},
            ),
        ]
    if schema.name == "self_attention":
        return [
            LoweringStep(name="split_heads", details={"heads": schema.uid_config.op_params.get(20, 1)}),
            LoweringStep(name="score_matmul", details={"einsum": "htd,hsd->hts"}),
            LoweringStep(name="softmax", details={"axis": -1}),
            LoweringStep(name="value_matmul", details={"einsum": "hts,hsd->htd"}),
            LoweringStep(name="merge_heads", details={}),
        ]
    return [
        LoweringStep(
            name="metadata_only",
            details={"reason": "unsupported op_name for specialized lowering"},
        )
    ]


def lower_to_torch_plan(schema: BroadcastedOperationSchema) -> TorchLoweringPlan:
    """
    Build an explicit lowering plan.

    The plan stays faithful to the paper's requirement that broadcasting metadata
    remain explicit before any backend call occurs.
    """

    notes = [
        "Use explicit affine reindexing before contraction.",
        "Do not treat implicit backend broadcasting as the semantic source of truth.",
        "Current execution is a NumPy reference path with optional torch tensor conversion.",
    ]
    return TorchLoweringPlan(
        op_name=schema.name,
        explicit_broadcast_dims=[r.cod_axes for r in schema.reindexings],
        weave_orders={
            "inputs": schema.input_weaves,
            "outputs": schema.output_weaves,
        },
        notes=notes,
        steps=_default_steps(schema),
    )


def _apply_affine_reindexing(
    tensor: np.ndarray,
    schema: BroadcastedOperationSchema,
    input_index: int,
    reindexing: ReindexingSchema,
    *,
    boundary: str = "error",
) -> np.ndarray:
    input_schema = schema.inputs[input_index]
    _validate_tensor_shape(
        f"input[{input_index}]",
        tensor,
        _expected_shape(input_schema.axes),
    )
    axis_sizes = _axis_size_map(schema)
    cod_shape = tuple(axis_sizes[uid] for uid in reindexing.cod_axes)
    if len(reindexing.linear) != len(reindexing.dom_axes):
        raise ValueError("linear rows must match domain axes")
    if len(reindexing.offset) != len(reindexing.dom_axes):
        raise ValueError("offset rows must match domain axes")
    if any(len(row) != len(reindexing.cod_axes) for row in reindexing.linear):
        raise ValueError("each affine row must match codomain axis count")

    result = np.empty(cod_shape, dtype=tensor.dtype)
    for cod_index in np.ndindex(*cod_shape):
        source_coords_by_uid: dict[int, int] = {}
        for uid, row, offset in zip(reindexing.dom_axes, reindexing.linear, reindexing.offset):
            coord = sum(coeff * cod_index[col] for col, coeff in enumerate(row)) + offset
            size = axis_sizes[uid]
            if boundary == "wrap":
                coord %= size
            elif boundary == "clamp":
                coord = min(max(coord, 0), size - 1)
            elif not 0 <= coord < size:
                raise ValueError(f"reindexing moved coordinate {coord} out of bounds for axis {uid}")
            source_coords_by_uid[uid] = coord

        ordered_index = tuple(source_coords_by_uid[axis.uid] for axis in input_schema.axes)
        result[cod_index] = tensor[ordered_index]
    return result


def _stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def _normalize_boundary_mode(mode: str) -> str:
    if mode == "circular":
        return "wrap"
    if mode == "replicate":
        return "clamp"
    return mode


def _execute_convolution_numpy(
    schema: BroadcastedOperationSchema,
    tensors: list[Any],
) -> dict[str, Any]:
    if len(tensors) != 2:
        raise ValueError("convolution expects [signal, kernel]")
    signal = np.asarray(tensors[0], dtype=np.float64)
    kernel = np.asarray(tensors[1], dtype=np.float64)
    _validate_tensor_shape("signal", signal, _expected_shape(schema.inputs[0].axes))
    _validate_tensor_shape("kernel", kernel, _expected_shape(schema.inputs[1].axes))
    if not schema.reindexings:
        raise ValueError("convolution requires an explicit reindexing")

    padding_mode = _normalize_boundary_mode(str(schema.uid_config.op_params.get(11, "error")))
    patches = _apply_affine_reindexing(
        signal,
        schema,
        0,
        schema.reindexings[0],
        boundary=padding_mode,
    )
    result = np.einsum("bclk,ck->bcl", patches, kernel)
    return {
        "backend": "numpy",
        "op_name": schema.name,
        "result": result,
        "intermediates": {"patches": patches},
        "notes": ["executed via explicit affine reindexing and kernel contraction"],
    }


def _execute_attention_numpy(
    schema: BroadcastedOperationSchema,
    tensors: list[Any],
) -> dict[str, Any]:
    if len(tensors) != 3:
        raise ValueError("self_attention expects [query, key, value]")
    query, key, value = (np.asarray(t, dtype=np.float64) for t in tensors)
    expected = _expected_shape(schema.inputs[0].axes)
    _validate_tensor_shape("query", query, expected)
    _validate_tensor_shape("key", key, expected)
    _validate_tensor_shape("value", value, expected)

    heads = int(schema.uid_config.op_params.get(20, 1))
    token_count, feature_dim = expected
    if feature_dim % heads != 0:
        raise ValueError(f"feature dimension {feature_dim} must be divisible by heads={heads}")
    head_dim = feature_dim // heads

    def split_heads(tensor: np.ndarray) -> np.ndarray:
        return tensor.reshape(token_count, heads, head_dim).transpose(1, 0, 2)

    query_h = split_heads(query)
    key_h = split_heads(key)
    value_h = split_heads(value)

    scores = np.einsum("htd,hsd->hts", query_h, key_h) / math.sqrt(head_dim)
    weights = _stable_softmax(scores, axis=-1)
    output_h = np.einsum("hts,hsd->htd", weights, value_h)
    output = output_h.transpose(1, 0, 2).reshape(token_count, feature_dim)
    return {
        "backend": "numpy",
        "op_name": schema.name,
        "result": output,
        "intermediates": {"weights": weights},
        "notes": ["executed via explicit head split, score contraction, softmax, and value contraction"],
    }


def execute_numpy(schema: BroadcastedOperationSchema, tensors: list[Any]) -> dict[str, Any]:
    if schema.name == "convolution":
        return _execute_convolution_numpy(schema, tensors)
    if schema.name == "self_attention":
        return _execute_attention_numpy(schema, tensors)
    raise NotImplementedError(f"no numpy executor for op {schema.name!r}")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def execute_reference_backend(
    plan: TorchLoweringPlan,
    tensors: list[Any],
    schema: BroadcastedOperationSchema | None = None,
) -> dict[str, Any]:
    """
    Execute the explicit reference plan.

    The authoritative runtime here is NumPy. When torch is available, we only
    convert the verified NumPy result into a torch tensor so downstream callers
    can inspect tensor-shaped artifacts without claiming native torch lowering.
    """

    if schema is None:
        raise ValueError("schema is required for execution")

    if _torch_available():
        import torch

        numpy_result = execute_numpy(schema, tensors)
        return {
            "backend": "torch-tensorized-reference",
            "op_name": plan.op_name,
            "explicit_broadcast_dims": plan.explicit_broadcast_dims,
            "weave_orders": plan.weave_orders,
            "result": torch.as_tensor(numpy_result["result"]),
            "notes": plan.notes + ["executed through the explicit NumPy reference path, then converted to torch"],
        }

    numpy_result = execute_numpy(schema, tensors)
    return {
        "backend": "numpy-reference",
        "op_name": plan.op_name,
        "explicit_broadcast_dims": plan.explicit_broadcast_dims,
        "weave_orders": plan.weave_orders,
        "result": numpy_result["result"],
        "intermediates": numpy_result.get("intermediates", {}),
        "notes": plan.notes + numpy_result.get("notes", []) + ["torch not installed; NumPy reference executor used"],
    }


def maybe_execute_torch(
    plan: TorchLoweringPlan,
    tensors: list[Any],
    schema: BroadcastedOperationSchema | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for the reference executor surface."""

    return execute_reference_backend(plan, tensors, schema)
