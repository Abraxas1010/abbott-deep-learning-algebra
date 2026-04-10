from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ort_diff import assert_outputs_close, diff_graphs, optimize_with_ort, ort_semantic_report, run_outputs
from sample_graphs import make_identity_add_model, make_identity_mul_model


def test_ort_outputs_match_after_optimization(tmp_path: Path) -> None:
    model_path = make_identity_add_model(tmp_path / "identity_add.onnx")
    optimized_path = optimize_with_ort(model_path, tmp_path / "identity_add_optimized.onnx")
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "y": np.ones((2, 3), dtype=np.float32),
    }
    before = run_outputs(model_path, feeds)
    after = run_outputs(optimized_path, feeds)
    assert_outputs_close(before, after)
    graph_diff = diff_graphs(model_path, optimized_path)
    assert graph_diff["before_node_count"] == 2
    assert graph_diff["after_node_count"] <= graph_diff["before_node_count"]
    assert graph_diff["after_ops"][-1] == "add"
    report = ort_semantic_report(model_path, optimized_path, feeds)
    assert report["before_output_digest"] == report["after_output_digest"]
    assert report["graph_diff"]["before_node_count"] == 2


def test_ort_diff_rejects_perturbed_wrong_graph(tmp_path: Path) -> None:
    model_path = make_identity_add_model(tmp_path / "identity_add.onnx")
    wrong_path = make_identity_mul_model(tmp_path / "identity_mul.onnx")
    feeds = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3),
        "y": np.full((2, 3), 2.0, dtype=np.float32),
    }
    before = run_outputs(model_path, feeds)
    wrong = run_outputs(wrong_path, feeds)
    with pytest.raises(AssertionError):
        assert_outputs_close(before, wrong)
    wrong_diff = diff_graphs(model_path, wrong_path)
    assert wrong_diff["changed"] is True
    assert wrong_diff["after_ops"][-1] == "mul"
