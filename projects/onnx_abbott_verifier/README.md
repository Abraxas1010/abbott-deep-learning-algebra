# ONNX Abbott Verifier

This project mirrors the Lean `HeytingLean.Bridge.Abbott.ONNX` fragment in Python.
It owns:
- canonical fragment schema classes
- honest coverage-matrix emission
- lowering from supported ONNX graphs into the canonical fragment
- typed rejection for unsupported imported nodes
- ONNX Runtime output comparison and lowered-graph diffing for supported models

Local workflow:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
pytest
python python/coverage_ledger.py --out artifacts/coverage_matrix.json
python python/optimize.py path/to/model.onnx artifacts/model_optimized.onnx
```

Key Python surfaces:
- `lower_from_onnx.py`: canonical lowering plus `UnsupportedNodeError(code, detail, op_type)`
- `ort_diff.py`: `optimize_with_ort`, `run_outputs`, `diff_graphs`, and `ort_semantic_report`

Current honest boundary:
- supported lowering/status surface: `Reshape`, `Transpose`, `Expand`, `Unsqueeze`, `Squeeze`, `Add`, `Mul`, `MatMul`, `Gemm`, `Conv`, `Flatten`, `Concat`, `Slice`, `Gather`, `Shape`, `ConstantOfShape`, `Relu`, `Softmax`, `ReduceSum`, `MaxPool`, `AveragePool`, `Where`, `Identity`, `Cast`
- partial lowering/status surface: `Attention`, `BatchNormalization`, `Clip`, `ReduceMean`, `Pad`
- unsupported in this tranche: `Resize`
- initializer-driven attrs are promoted for `Reshape`, `Expand`, `Unsqueeze`, `Squeeze`, `ReduceSum`, `ReduceMean`, `Slice`, `Pad`, and `Resize` when the relevant inputs are static initializers
- graphs outside the canonical fragment are rejected explicitly by `lower_from_onnx.py`
