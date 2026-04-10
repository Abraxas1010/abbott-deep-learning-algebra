from __future__ import annotations

from collections import Counter

from coverage_ledger import build_coverage_rows


def test_coverage_rows_cover_ranked_core() -> None:
    rows = build_coverage_rows()
    assert len(rows) == 30
    counts = Counter(row["status"] for row in rows)
    assert counts["supported"] == 24
    assert counts["partial"] == 5
    assert counts["unsupported"] == 1
