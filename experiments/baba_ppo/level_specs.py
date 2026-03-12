from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyBaba

from Extensions.BabaRL import write_level


_COMMENT_PREFIX = "#"
_DEFAULT_SPEC_SUFFIX = ".level.txt"
_SENTINEL_NAMES = {"NOUN_TYPE", "OP_TYPE", "PROPERTY_TYPE", "ICON_TYPE"}
_OBJECT_MEMBERS = pyBaba.ObjectType.__members__


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {
        ".": "ICON_EMPTY",
        "_": "ICON_EMPTY",
        "empty": "ICON_EMPTY",
    }

    for name in _OBJECT_MEMBERS:
        if name in _SENTINEL_NAMES:
            continue

        lowered = name.lower()
        aliases[lowered] = name

        if name.startswith("ICON_"):
            base = name.removeprefix("ICON_").lower()
            aliases[base] = name
            aliases[f"icon_{base}"] = name
        else:
            aliases[f"text_{lowered}"] = name
            aliases[f"{lowered}_text"] = name

    return aliases


ALIASES = _build_aliases()


def resolve_token(token: str) -> pyBaba.ObjectType:
    normalized = token.strip().lower()
    if not normalized:
        raise ValueError("Encountered an empty cell token.")
    if "+" in normalized:
        raise ValueError(
            "Natural-language specs currently support one initial object per cell."
        )

    enum_name = ALIASES.get(normalized, token.strip().upper())
    try:
        return _OBJECT_MEMBERS[enum_name]
    except KeyError as exc:
        raise KeyError(f"Unknown level token: {token}") from exc


def parse_level_spec(text: str) -> list[list[pyBaba.ObjectType]]:
    rows: list[list[pyBaba.ObjectType]] = []

    for raw_line in text.splitlines():
        line = raw_line.split(_COMMENT_PREFIX, maxsplit=1)[0].strip()
        if not line:
            continue
        rows.append([resolve_token(token) for token in line.split()])

    if not rows:
        raise ValueError("Level spec is empty.")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Level spec must be rectangular.")

    return rows


def compile_level_spec(spec_path: str | Path, output_path: str | Path) -> Path:
    spec_path = Path(spec_path)
    output_path = Path(output_path)
    rows = parse_level_spec(spec_path.read_text(encoding="utf-8"))
    return write_level(output_path, rows)


def compile_level_directory(
    spec_dir: str | Path,
    output_dir: str | Path,
    *,
    suffix: str = _DEFAULT_SPEC_SUFFIX,
) -> list[Path]:
    spec_dir = Path(spec_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compiled: list[Path] = []
    for spec_path in sorted(spec_dir.glob(f"*{suffix}")):
        output_name = spec_path.name[: -len(suffix)] + ".txt"
        compiled.append(compile_level_spec(spec_path, output_dir / output_name))
    return compiled


def iter_supported_tokens() -> Iterable[str]:
    return sorted(ALIASES)
