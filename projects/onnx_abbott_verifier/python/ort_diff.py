from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from lower_from_onnx import lower_model


def optimize_with_ort(model_path: str | Path, optimized_path: str | Path) -> Path:
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(optimized_path)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    return Path(optimized_path)


def run_outputs(model_path: str | Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def output_digest(outputs: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in outputs:
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def diff_graphs(before_path: str | Path, after_path: str | Path) -> dict[str, object]:
    before = lower_model(before_path)
    after = lower_model(after_path)
    before_ops = [node.op.value for node in before.nodes]
    after_ops = [node.op.value for node in after.nodes]
    return {
        "before_node_count": len(before.nodes),
        "after_node_count": len(after.nodes),
        "before_ops": before_ops,
        "after_ops": after_ops,
        "node_delta": len(after.nodes) - len(before.nodes),
        "changed": before.to_json() != after.to_json(),
    }


def ort_semantic_report(model_path: str | Path, optimized_path: str | Path, feeds: dict[str, np.ndarray]) -> dict[str, object]:
    before = run_outputs(model_path, feeds)
    after = run_outputs(optimized_path, feeds)
    assert_outputs_close(before, after)
    return {
        "before_output_digest": output_digest(before),
        "after_output_digest": output_digest(after),
        "graph_diff": diff_graphs(model_path, optimized_path),
    }


def assert_outputs_close(lhs: list[np.ndarray], rhs: list[np.ndarray]) -> None:
    if len(lhs) != len(rhs):
        raise AssertionError("output arity mismatch")
    for before, after in zip(lhs, rhs):
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
