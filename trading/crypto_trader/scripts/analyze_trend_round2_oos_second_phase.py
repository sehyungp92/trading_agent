"""Compatibility wrapper for the generic trend round-2 second-phase OOS repair sweep."""

from __future__ import annotations

import sys

from analyze_oos_repair import main


if __name__ == "__main__":
    main(["--strategy", "trend", "--round", "2", "--phase", "second", *sys.argv[1:]])
