from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pyBaba

_OBJECT_TYPES = pyBaba.ObjectType.__members__


def resolve_object_type(value: int | str | pyBaba.ObjectType) -> pyBaba.ObjectType:
    """Resolve an object specifier to a `pyBaba.ObjectType` value."""
    if isinstance(value, str):
        key = value.strip().upper()
        if key not in _OBJECT_TYPES:
            raise KeyError(f"Unknown object type: {value}")
        return _OBJECT_TYPES[key]
    if isinstance(value, int):
        return pyBaba.ObjectType(value)
    return value


def write_level(path: str | Path, rows: Sequence[Sequence[int | str | pyBaba.ObjectType]]) -> Path:
    """Write a numeric Baba level file from symbolic rows.

    Each cell may be an `int`, an enum name such as `"ICON_BABA"` or `"IS"`,
    or a `pyBaba.ObjectType` value.
    """
    if not rows or not rows[0]:
        raise ValueError("rows must be a non-empty rectangular grid")

    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("rows must be rectangular")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{width} {len(rows)}\n")
        for row in rows:
            encoded_row = [str(int(resolve_object_type(cell))) for cell in row]
            handle.write("  ".join(encoded_row))
            handle.write("\n")

    return path.resolve()


def read_level(path: str | Path) -> np.ndarray:
    """Read a numeric Baba level file into a `(height, width)` integer array."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        width, height = map(int, handle.readline().split())
        data = [
            [int(token) for token in handle.readline().split()]
            for _ in range(height)
        ]

    array = np.asarray(data, dtype=np.int64)
    if array.shape != (height, width):
        raise ValueError(f"Level {path} does not match declared shape {(height, width)}")
    return array
