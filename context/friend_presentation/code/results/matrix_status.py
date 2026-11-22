"""Print progress of the nine-cell training and ablation archive."""

from __future__ import annotations

import json
from pathlib import Path


RUNS = Path(__file__).resolve().parent / "runs"


def last_update(run: Path) -> str:
    history = run / "training.jsonl"
    if not history.exists():
        return ""
    with history.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - 2048))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return f"{row['stage']} {row['update']}"
        except (ValueError, KeyError):
            continue
    return ""


def main() -> None:
    for run in sorted(RUNS.glob("20261011_*")):
        manifest = run / "manifest.json"
        if not manifest.exists():
            continue
        status = json.loads(manifest.read_text())["status"]
        print(f"{run.name:58} {status:34} {last_update(run)}")


if __name__ == "__main__":
    main()
