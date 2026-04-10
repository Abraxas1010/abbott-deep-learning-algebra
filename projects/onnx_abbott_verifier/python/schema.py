from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class RankedOp(str, Enum):
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    EXPAND = "expand"
    UNSQUEEZE = "unsqueeze"
    SQUEEZE = "squeeze"
    ADD = "add"
    MUL = "mul"
    MATMUL = "matMul"
    GEMM = "gemm"
    CONV = "conv"
    ATTENTION = "attention"
    FLATTEN = "flatten"
    CONCAT = "concat"
    SLICE = "slice"
    GATHER = "gather"
    SHAPE = "shape"
    CONSTANT_OF_SHAPE = "constantOfShape"
    BATCH_NORMALIZATION = "batchNormalization"
    RELU = "relu"
    CLIP = "clip"
    SOFTMAX = "softmax"
    REDUCE_SUM = "reduceSum"
    REDUCE_MEAN = "reduceMean"
    MAX_POOL = "maxPool"
    AVERAGE_POOL = "averagePool"
    WHERE = "where"
    IDENTITY = "identity"
    CAST = "cast"
    PAD = "pad"
    RESIZE = "resize"


RANKED_OPS = [
    RankedOp.RESHAPE,
    RankedOp.TRANSPOSE,
    RankedOp.EXPAND,
    RankedOp.UNSQUEEZE,
    RankedOp.SQUEEZE,
    RankedOp.ADD,
    RankedOp.MUL,
    RankedOp.MATMUL,
    RankedOp.GEMM,
    RankedOp.CONV,
    RankedOp.ATTENTION,
    RankedOp.FLATTEN,
    RankedOp.CONCAT,
    RankedOp.SLICE,
    RankedOp.GATHER,
    RankedOp.SHAPE,
    RankedOp.CONSTANT_OF_SHAPE,
    RankedOp.BATCH_NORMALIZATION,
    RankedOp.RELU,
    RankedOp.CLIP,
    RankedOp.SOFTMAX,
    RankedOp.REDUCE_SUM,
    RankedOp.REDUCE_MEAN,
    RankedOp.MAX_POOL,
    RankedOp.AVERAGE_POOL,
    RankedOp.WHERE,
    RankedOp.IDENTITY,
    RankedOp.CAST,
    RankedOp.PAD,
    RankedOp.RESIZE,
]


class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT64 = "int64"
    BOOL = "bool"


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


SUPPORT_STATUS: dict[RankedOp, SupportStatus] = {
    RankedOp.RESHAPE: SupportStatus.SUPPORTED,
    RankedOp.TRANSPOSE: SupportStatus.SUPPORTED,
    RankedOp.EXPAND: SupportStatus.SUPPORTED,
    RankedOp.UNSQUEEZE: SupportStatus.SUPPORTED,
    RankedOp.SQUEEZE: SupportStatus.SUPPORTED,
    RankedOp.ADD: SupportStatus.SUPPORTED,
    RankedOp.MUL: SupportStatus.SUPPORTED,
    RankedOp.MATMUL: SupportStatus.SUPPORTED,
    RankedOp.GEMM: SupportStatus.SUPPORTED,
    RankedOp.CONV: SupportStatus.SUPPORTED,
    RankedOp.ATTENTION: SupportStatus.PARTIAL,
    RankedOp.FLATTEN: SupportStatus.SUPPORTED,
    RankedOp.CONCAT: SupportStatus.SUPPORTED,
    RankedOp.SLICE: SupportStatus.SUPPORTED,
    RankedOp.GATHER: SupportStatus.SUPPORTED,
    RankedOp.SHAPE: SupportStatus.SUPPORTED,
    RankedOp.CONSTANT_OF_SHAPE: SupportStatus.SUPPORTED,
    RankedOp.BATCH_NORMALIZATION: SupportStatus.PARTIAL,
    RankedOp.RELU: SupportStatus.SUPPORTED,
    RankedOp.CLIP: SupportStatus.PARTIAL,
    RankedOp.SOFTMAX: SupportStatus.SUPPORTED,
    RankedOp.REDUCE_SUM: SupportStatus.SUPPORTED,
    RankedOp.REDUCE_MEAN: SupportStatus.PARTIAL,
    RankedOp.MAX_POOL: SupportStatus.SUPPORTED,
    RankedOp.AVERAGE_POOL: SupportStatus.SUPPORTED,
    RankedOp.WHERE: SupportStatus.SUPPORTED,
    RankedOp.IDENTITY: SupportStatus.SUPPORTED,
    RankedOp.CAST: SupportStatus.SUPPORTED,
    RankedOp.PAD: SupportStatus.PARTIAL,
    RankedOp.RESIZE: SupportStatus.UNSUPPORTED,
}


@dataclass(slots=True)
class ShapeDim:
    known: int | None = None
    symbolic: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ShapeExpr:
    dims: list[ShapeDim]

    @classmethod
    def from_known(cls, dims: list[int]) -> "ShapeExpr":
        return cls([ShapeDim(known=d) for d in dims])

    def known_dims(self) -> list[int] | None:
        dims: list[int] = []
        for dim in self.dims:
            if dim.known is None:
                return None
            dims.append(dim.known)
        return dims

    def to_json(self) -> dict[str, Any]:
        return {"dims": [dim.to_json() for dim in self.dims]}


@dataclass(slots=True)
class TensorTy:
    dtype: DType
    shape: ShapeExpr

    def to_json(self) -> dict[str, Any]:
        return {"dtype": self.dtype.value, "shape": self.shape.to_json()}


@dataclass(slots=True)
class ValueDecl:
    name: str
    ty: TensorTy

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "ty": self.ty.to_json()}


@dataclass(slots=True)
class Node:
    op: RankedOp
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
        }


@dataclass(slots=True)
class FragmentGraph:
    inputs: list[ValueDecl]
    initializers: list[ValueDecl]
    nodes: list[Node]
    outputs: list[ValueDecl]

    def to_json(self) -> dict[str, Any]:
        return {
            "inputs": [decl.to_json() for decl in self.inputs],
            "initializers": [decl.to_json() for decl in self.initializers],
            "nodes": [node.to_json() for node in self.nodes],
            "outputs": [decl.to_json() for decl in self.outputs],
        }
