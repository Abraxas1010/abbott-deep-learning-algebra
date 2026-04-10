from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from schema import DType, FragmentGraph, Node, RankedOp, ShapeDim, ShapeExpr, TensorTy, ValueDecl


@dataclass(slots=True)
class UnsupportedNodeError(RuntimeError):
    code: str
    detail: str
    op_type: str | None = None

    def __post_init__(self) -> None:
        op = f" op={self.op_type}" if self.op_type is not None else ""
        RuntimeError.__init__(self, f"{self.code}:{op} {self.detail}")


ONNX_OP_MAP: dict[str, RankedOp] = {
    "Reshape": RankedOp.RESHAPE,
    "Transpose": RankedOp.TRANSPOSE,
    "Expand": RankedOp.EXPAND,
    "Unsqueeze": RankedOp.UNSQUEEZE,
    "Squeeze": RankedOp.SQUEEZE,
    "Add": RankedOp.ADD,
    "Mul": RankedOp.MUL,
    "MatMul": RankedOp.MATMUL,
    "Gemm": RankedOp.GEMM,
    "Conv": RankedOp.CONV,
    "Attention": RankedOp.ATTENTION,
    "Flatten": RankedOp.FLATTEN,
    "Concat": RankedOp.CONCAT,
    "Slice": RankedOp.SLICE,
    "Gather": RankedOp.GATHER,
    "Shape": RankedOp.SHAPE,
    "ConstantOfShape": RankedOp.CONSTANT_OF_SHAPE,
    "BatchNormalization": RankedOp.BATCH_NORMALIZATION,
    "Relu": RankedOp.RELU,
    "Clip": RankedOp.CLIP,
    "Softmax": RankedOp.SOFTMAX,
    "ReduceSum": RankedOp.REDUCE_SUM,
    "ReduceMean": RankedOp.REDUCE_MEAN,
    "MaxPool": RankedOp.MAX_POOL,
    "AveragePool": RankedOp.AVERAGE_POOL,
    "Where": RankedOp.WHERE,
    "Identity": RankedOp.IDENTITY,
    "Cast": RankedOp.CAST,
    "Pad": RankedOp.PAD,
    "Resize": RankedOp.RESIZE,
}


def dtype_of(elem_type: int) -> DType:
    mapping = {
        TensorProto.FLOAT: DType.FLOAT32,
        TensorProto.DOUBLE: DType.FLOAT64,
        TensorProto.INT64: DType.INT64,
        TensorProto.BOOL: DType.BOOL,
    }
    if elem_type not in mapping:
        raise UnsupportedNodeError("unsupported_dtype", f"unsupported dtype: {elem_type}")
    return mapping[elem_type]


def shape_expr_from_tensor_type(tensor_type: Any) -> ShapeExpr:
    dims: list[ShapeDim] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(ShapeDim(known=int(dim.dim_value)))
        elif dim.HasField("dim_param"):
            dims.append(ShapeDim(symbolic=dim.dim_param))
        else:
            dims.append(ShapeDim(symbolic="?"))
    return ShapeExpr(dims)


def value_decl_from_value_info(value_info: Any) -> ValueDecl:
    tensor_type = value_info.type.tensor_type
    return ValueDecl(
        name=value_info.name,
        ty=TensorTy(dtype=dtype_of(tensor_type.elem_type), shape=shape_expr_from_tensor_type(tensor_type)),
    )


def value_decl_from_initializer(initializer: TensorProto) -> ValueDecl:
    return ValueDecl(
        name=initializer.name,
        ty=TensorTy(dtype=dtype_of(initializer.data_type), shape=ShapeExpr.from_known(list(initializer.dims))),
    )


def _attribute_value(attr: AttributeProto) -> Any:
    if attr.type == AttributeProto.INT:
        return int(attr.i)
    if attr.type == AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if attr.type == AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == AttributeProto.TENSOR:
        array = numpy_helper.to_array(attr.t)
        return array.tolist()
    raise UnsupportedNodeError("unsupported_attr", f"unsupported attribute type for {attr.name}")


def _tensor_values(initializers: dict[str, TensorProto], name: str) -> list[int] | list[float] | None:
    tensor = initializers.get(name)
    if tensor is None:
        return None
    return numpy_helper.to_array(tensor).tolist()


def _shape_attr_from_dims(dims: list[int]) -> dict[str, Any]:
    return {"dims": [{"known": int(dim), "symbolic": None} for dim in dims]}


def lower_model(path: str | Path) -> FragmentGraph:
    model = onnx.load(Path(path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    initializer_decls = [value_decl_from_initializer(init) for init in graph.initializer]
    nodes: list[Node] = []

    for raw in graph.node:
        if raw.op_type not in ONNX_OP_MAP:
            raise UnsupportedNodeError("unsupported_op", f"unsupported op: {raw.op_type}", op_type=raw.op_type)
        attrs = {attr.name: _attribute_value(attr) for attr in raw.attribute}
        inputs = list(raw.input)

        if raw.op_type in {"Reshape", "Expand"} and len(inputs) == 2 and inputs[1] in initializers:
            dims = numpy_helper.to_array(initializers[inputs[1]]).astype("int64").tolist()
            attrs["shape"] = _shape_attr_from_dims(dims)
            inputs = [inputs[0]]
        if raw.op_type in {"Unsqueeze", "Squeeze", "ReduceSum", "ReduceMean"} and len(inputs) >= 2:
            axes = _tensor_values(initializers, inputs[1])
            if axes is not None:
                attrs["axes"] = [int(axis) for axis in axes]
                inputs = [inputs[0], *inputs[2:]]
        if raw.op_type == "Slice" and len(inputs) >= 3:
            starts = _tensor_values(initializers, inputs[1])
            ends = _tensor_values(initializers, inputs[2])
            axes = _tensor_values(initializers, inputs[3]) if len(inputs) >= 4 else None
            steps = _tensor_values(initializers, inputs[4]) if len(inputs) >= 5 else None
            if starts is not None and ends is not None:
                attrs["starts"] = [int(value) for value in starts]
                attrs["ends"] = [int(value) for value in ends]
                if axes is not None:
                    attrs["axes"] = [int(value) for value in axes]
                if steps is not None:
                    attrs["steps"] = [int(value) for value in steps]
                inputs = [inputs[0]]
        if raw.op_type == "Pad" and len(inputs) >= 2:
            pads = _tensor_values(initializers, inputs[1])
            if pads is not None:
                attrs["pads"] = [int(value) for value in pads]
                if len(inputs) >= 3:
                    pad_value = _tensor_values(initializers, inputs[2])
                    if isinstance(pad_value, list) and len(pad_value) == 1:
                        attrs["value"] = pad_value[0]
                inputs = [inputs[0]]
        if raw.op_type == "Resize":
            if len(inputs) >= 3:
                scales = _tensor_values(initializers, inputs[2])
                if scales is not None:
                    attrs["scales"] = [float(value) for value in scales]
            if len(inputs) >= 4:
                sizes = _tensor_values(initializers, inputs[3])
                if sizes is not None:
                    attrs["sizes"] = _shape_attr_from_dims([int(value) for value in sizes])
        if raw.op_type == "Cast" and "to" in attrs:
            attrs["to"] = dtype_of(int(attrs["to"])).value

        nodes.append(
            Node(
                op=ONNX_OP_MAP[raw.op_type],
                inputs=inputs,
                outputs=list(raw.output),
                attrs=attrs,
            )
        )

    return FragmentGraph(
        inputs=[value_decl_from_value_info(value) for value in graph.input if value.name not in initializers],
        initializers=initializer_decls,
        nodes=nodes,
        outputs=[value_decl_from_value_info(value) for value in graph.output],
    )
