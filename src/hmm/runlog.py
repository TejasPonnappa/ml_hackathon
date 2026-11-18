"""Print to the terminal AND save the same text to a UTF-8 log file (proof of the run)."""
from __future__ import annotations

import sys
from pathlib import Path


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def tee_output(path: Path) -> None:
    """From now on, everything printed (and any error traceback) also goes to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
