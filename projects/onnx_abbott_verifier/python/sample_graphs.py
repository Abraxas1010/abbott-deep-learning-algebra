from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def make_add_model(path: str | Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"])
    graph = helper.make_graph([node], "add_model", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)


def make_reshape_model(path: str | Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [3, 2])
    node = helper.make_node("Reshape", ["x", "shape"], ["y"])
    graph = helper.make_graph([node], "reshape_model", [x], [out], [shape_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)


def make_identity_add_model(path: str | Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    identity = helper.make_node("Identity", ["x"], ["x_id"])
    add = helper.make_node("Add", ["x_id", "y"], ["out"])
    graph = helper.make_graph([identity, add], "identity_add_model", [x, y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)


def make_matmul_add_model(path: str | Path) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 2])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2])
    mm = helper.make_node("MatMul", ["a", "b"], ["tmp"])
    add = helper.make_node("Add", ["tmp", "bias"], ["out"])
    graph = helper.make_graph([mm, add], "matmul_add_model", [a, b, bias], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)


def make_sub_model(path: str | Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Sub", ["x", "y"], ["z"])
    graph = helper.make_graph([node], "sub_model", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)


def make_identity_mul_model(path: str | Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    identity = helper.make_node("Identity", ["x"], ["x_id"])
    mul = helper.make_node("Mul", ["x_id", "y"], ["out"])
    graph = helper.make_graph([identity, mul], "identity_mul_model", [x, y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return Path(path)
