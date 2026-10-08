"""Run with an existing Python; optional dependencies stay inside this project."""
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent
if (PROJECT / ".deps").is_dir():
    sys.path.insert(0, str(PROJECT / ".deps"))

if __name__ == "__main__":
    from vtlab.cli import main
    raise SystemExit(main())
