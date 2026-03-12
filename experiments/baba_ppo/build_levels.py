from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.baba_ppo.config import EXPERIMENT_ROOT
from experiments.baba_ppo.level_specs import compile_level_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile natural-language Baba levels into numeric map files."
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=EXPERIMENT_ROOT / "level_specs",
        help="Directory containing train/ and test/ level specs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "levels",
        help="Directory where compiled numeric maps will be written.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
        help="Which level split to compile.",
    )
    return parser.parse_args()


def compile_split(spec_root: Path, output_root: Path, split: str) -> list[Path]:
    compiled = compile_level_directory(spec_root / split, output_root / split)
    print(f"Compiled {len(compiled)} {split} level(s).")
    for path in compiled:
        print(path)
    return compiled


def main() -> None:
    args = parse_args()

    if args.split in {"train", "all"}:
        compile_split(args.spec_root, args.output_root, "train")
    if args.split in {"test", "all"}:
        compile_split(args.spec_root, args.output_root, "test")


if __name__ == "__main__":
    main()
