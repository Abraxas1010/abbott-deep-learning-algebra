from __future__ import annotations

import argparse
from pathlib import Path

from ort_diff import optimize_with_ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize an ONNX model with ONNX Runtime.")
    parser.add_argument("model", type=Path, help="Input ONNX model path")
    parser.add_argument("optimized", type=Path, help="Output optimized ONNX model path")
    args = parser.parse_args()
    optimize_with_ort(args.model, args.optimized)


if __name__ == "__main__":
    main()
