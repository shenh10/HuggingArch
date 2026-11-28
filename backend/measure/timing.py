"""Optional JSONL timing for an offline warm.

The file is named by ``HA_WARM_TIMING_FILE``; when it is unset every call is
a no-op, so production requests pay nothing. A line is one event with the
fields the caller names, plus the process and wall-clock time. Workers inherit
the environment and append under an exclusive ``flock`` so a pool of processes
writes one ordered JSONL file.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_PATH = os.environ.get("HA_WARM_TIMING_FILE", "")
_LOCK = threading.Lock()


def timing_enabled() -> bool:
    return bool(_PATH)


def log_timing(event: str, **fields: Any) -> None:
    if not _PATH:
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        "time": time.time(),
        **fields,
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":"),
                      default=str) + "\n"
    path = Path(_PATH)
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    stream.write(line)
                    stream.flush()
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
