from __future__ import annotations

import argparse
import json
from pathlib import Path

from schema import RANKED_OPS, SUPPORT_STATUS


def build_coverage_rows() -> list[dict[str, str]]:
    return [
        {"op": op.value, "status": SUPPORT_STATUS[op].value}
        for op in RANKED_OPS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the ONNX Abbott coverage matrix.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()
    rows = build_coverage_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
