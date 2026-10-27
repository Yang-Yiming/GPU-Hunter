"""Compact progress output used by the default (non-verbose) mode.

In verbose mode gpu-hunter keeps logging every status line.  Without -v the
polling loops only emit one ``x`` per failed attempt (without a newline) and a
normal text line when something succeeds, so long waits do not flood the
terminal.
"""
from __future__ import annotations

import sys
from typing import TextIO


class Progress:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self._pending = False

    def break_line(self) -> None:
        """Finish an unfinished x-line, e.g. before a log warning is printed."""
        if self._pending:
            self.stream.write("\n")
            self.stream.flush()
            self._pending = False

    def failed(self) -> None:
        """Record one failed attempt without scrolling the terminal."""
        self.stream.write("x")
        self.stream.flush()
        self._pending = True

    def say(self, message: str) -> None:
        """Print a normal line, first closing any pending x-line."""
        self.break_line()
        self.stream.write(message.rstrip("\n") + "\n")
        self.stream.flush()

    def finish(self) -> None:
        self.break_line()
