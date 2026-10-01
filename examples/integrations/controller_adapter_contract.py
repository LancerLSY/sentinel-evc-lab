"""Shape required by Executor; this file deliberately contains no hardware driver."""
from __future__ import annotations

from typing import Protocol


class ExecutorControllerPort(Protocol):
    cancel_acked: bool | None

    def free_slots(self) -> int: ...
    def submit(self, action, generation: int, gripper_event=None) -> bool: ...
    def cancel(self) -> None: ...
    def tick(self) -> None: ...
    def cursors(self) -> dict: ...
    @property
    def is_drained(self) -> bool: ...
    def read_feedback(self, now_ns: int) -> dict: ...


class UnimplementedHardwareAdapter:
    """Explicit placeholder: a reviewed device adapter must implement every method."""

    cancel_acked = None

    def __getattr__(self, name):
        raise NotImplementedError(f"hardware adapter capability is not implemented: {name}")
