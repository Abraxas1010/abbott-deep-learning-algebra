from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from .uid_config import AxisSpec, SlotRef, UIDConfig
except ImportError:
    from uid_config import AxisSpec, SlotRef, UIDConfig


@dataclass(slots=True)
class ArrayObjectSchema:
    dtype: str
    axes: list[AxisSpec]


@dataclass(slots=True)
class ReindexingSchema:
    dom_axes: list[int]
    cod_axes: list[int]
    linear: list[list[int]]
    offset: list[int]


@dataclass(slots=True)
class BroadcastedOperationSchema:
    name: str
    inputs: list[ArrayObjectSchema]
    output: ArrayObjectSchema
    uid_config: UIDConfig
    reindexings: list[ReindexingSchema] = field(default_factory=list)
    input_weaves: list[list[int]] = field(default_factory=list)
    output_weaves: list[list[int]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def slot_ref(uid: int, kind: str, label: str = "") -> SlotRef:
    return SlotRef(uid=uid, kind=kind, label=label)
