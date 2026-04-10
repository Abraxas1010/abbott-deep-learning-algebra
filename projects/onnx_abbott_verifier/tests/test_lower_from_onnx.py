from __future__ import annotations

from pathlib import Path

import pytest

from lower_from_onnx import UnsupportedNodeError, lower_model
from sample_graphs import make_add_model, make_matmul_add_model, make_reshape_model, make_sub_model


def test_lower_add_model(tmp_path: Path) -> None:
    model_path = make_add_model(tmp_path / "add.onnx")
    graph = lower_model(model_path)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].op.value == "add"
    assert graph.nodes[0].inputs == ["x", "y"]


def test_lower_reshape_promotes_shape_initializer(tmp_path: Path) -> None:
    model_path = make_reshape_model(tmp_path / "reshape.onnx")
    graph = lower_model(model_path)
    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.op.value == "reshape"
    assert node.inputs == ["x"]
    assert node.attrs["shape"]["dims"] == [
        {"known": 3, "symbolic": None},
        {"known": 2, "symbolic": None},
    ]


def test_lower_matmul_add_graph(tmp_path: Path) -> None:
    model_path = make_matmul_add_model(tmp_path / "matmul_add.onnx")
    graph = lower_model(model_path)
    assert [node.op.value for node in graph.nodes] == ["matMul", "add"]
    assert graph.nodes[0].outputs == ["tmp"]
    assert graph.nodes[1].inputs == ["tmp", "bias"]


def test_lower_rejects_unsupported_op_with_typed_reason(tmp_path: Path) -> None:
    model_path = make_sub_model(tmp_path / "sub.onnx")
    with pytest.raises(UnsupportedNodeError) as excinfo:
        lower_model(model_path)
    err = excinfo.value
    assert err.code == "unsupported_op"
    assert err.op_type == "Sub"
