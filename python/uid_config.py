from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SlotKind = Literal["axis", "dtype", "op_param", "placeholder"]


@dataclass(slots=True)
class SlotRef:
    uid: int
    kind: SlotKind
    label: str = ""


@dataclass(slots=True)
class AxisSpec:
    uid: int
    label: str
    size: int


@dataclass(slots=True)
class UIDConfig:
    slots: list[SlotRef] = field(default_factory=list)
    axes: list[AxisSpec] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    op_params: dict[int, float | int | str] = field(default_factory=dict)

    def slot_map(self) -> dict[int, SlotRef]:
        return {slot.uid: slot for slot in self.slots}

    def axis_map(self) -> dict[int, AxisSpec]:
        return {axis.uid: axis for axis in self.axes}

    def validate_unique_uids(self) -> None:
        seen: set[int] = set()
        for uid in [slot.uid for slot in self.slots] + [axis.uid for axis in self.axes]:
            if uid in seen:
                raise ValueError(f"duplicate UID detected: {uid}")
            seen.add(uid)
