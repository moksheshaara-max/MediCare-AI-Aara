"""Optional no-dependency RSS/cgroup memory diagnostics for MediCare AI.

Enabled only with MEDICARE_MEMORY_LOGGING=1. Emits no query text,
file names, personal health data, request bodies, or secrets.
Observations are process/container measurements, NOT per-request allocations.
"""

import functools
import os
from pathlib import Path
import re
import threading
import time

_lock = threading.Lock()
_sampler_started = False
_active_requests = 0


def _enabled():
    return os.getenv("MEDICARE_MEMORY_LOGGING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _mb(value):
    try:
        return f"{int(value) / (1024 * 1024):.1f}"
    except (ValueError, TypeError, OverflowError):
        return "na"


def _read_memory():
    # Linux worker RSS and high-water mark in kilobytes; cgroup values in bytes.
    fields = {"rss_mb": "na", "rss_peak_mb": "na", "cgroup_mb": "na",
              "cgroup_peak_mb": "na", "cgroup_limit_mb": "na", "threads": "na"}
    proc = Path("/proc/self/status")
    if proc.is_file():
        status = proc.read_text(encoding="ascii", errors="replace")
        for key, target in (("VmRSS", "rss_mb"), ("VmHWM", "rss_peak_mb")):
            match = re.search(rf"^{key}:\s*(\d+)\s+kB", status, flags=re.M)
            if match:
                fields[target] = _mb(int(match.group(1)) * 1024)
        match = re.search(r"^Threads:\s*(\d+)", status, flags=re.M)
        if match:
            fields["threads"] = match.group(1)

    # cgroup memory includes other processes and cached file pages; do not
    # interpret it as this Python process's unique memory allocation.
    root = Path("/sys/fs/cgroup")
    cg_files = (("memory.current", "cgroup_mb"),
                ("memory.peak", "cgroup_peak_mb"),
                ("memory.max", "cgroup_limit_mb"))
    for name, target in cg_files:
        path = root / name
        if path.is_file():
            raw = path.read_text(encoding="ascii").strip()
            fields[target] = "unlimited" if raw == "max" else _mb(raw)
    return fields


def log_memory(stage, since=None):
    """Log a small diagnostic snapshot. Never affect clinical processing."""
    if not _enabled():
        return
    try:
        values = _read_memory()
        elapsed = f" elapsed_s={time.monotonic() - since:.1f}" if since is not None else ""
        metrics = " ".join(f"{key}={value}" for key, value in values.items())
        print(f"[MEM] stage={stage} pid={os.getpid()} tid={threading.get_ident()}"
              f" {metrics}{elapsed}", flush=True)
    except Exception:
        # Monitoring cannot be allowed to break a real medical request.
        pass


def _sampler_loop():
    while True:
        time.sleep(5)
        with _lock:
            active = _active_requests
        if active:
            log_memory("sample.active")


def trace_request(label):
    """Optional 5s sampler while a monitored endpoint is handling requests."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not _enabled():
                return fn(*args, **kwargs)
            global _sampler_started, _active_requests
            with _lock:
                _active_requests += 1
                if not _sampler_started:
                    threading.Thread(target=_sampler_loop, name="medicare-memory-sampler",
                                     daemon=True).start()
                    _sampler_started = True
            started = time.monotonic()
            log_memory(f"{label}.start")
            try:
                return fn(*args, **kwargs)
            finally:
                log_memory(f"{label}.finish", since=started)
                with _lock:
                    _active_requests -= 1
        return wrapped
    return decorate
