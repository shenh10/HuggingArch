"""Exclusive lock on the GPU a measurement run is about to use.

Measured latency IS the product. Two processes sharing a device slow each
other down, and nothing in the emitted row records that it happened — so a
contended sweep is indistinguishable from a clean one after the fact and the
whole batch has to be thrown away. This has cost real corpus: five sweeps once
ran concurrently on one H100 for 22 hours, and 16k landed rows had to be
quarantined.

`--skip-existing` cannot prevent it: that asks what the CORPUS already holds,
not what another process is measuring right now.

The lock is per PHYSICAL DEVICE, keyed on the CUDA UUID rather than the device
index — `CUDA_VISIBLE_DEVICES` remaps indices, so index 0 in two shells can be
two different cards, or the same one. A multi-GPU box therefore runs one sweep
per card concurrently, which is correct and wanted; only a second sweep on the
SAME card is refused.

Default is fail-fast, not wait: the caller is a long GPU sweep, and silently
queueing for hours behind another run is rarely what anyone meant. The refusal
names the holder (pid + command) so the operator can decide.
"""
from __future__ import annotations

import errno
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


class GpuBusy(RuntimeError):
    """Raised when another process already holds this GPU's lock."""


@dataclass(frozen=True)
class ComputeProcess:
    """One process already resident on the physical GPU."""

    pid: int
    command: str
    used_memory_mib: Optional[int]


def _query_compute_apps(uuid: str) -> Tuple[List[ComputeProcess], Optional[str]]:
    """Return compute processes already resident on ``uuid``.

    The warning is non-empty when the box cannot provide a trustworthy
    ``nvidia-smi`` answer. The flock remains useful in that case, but the
    caller must not mistake it for proof that an older uncoordinated process
    is absent.
    """
    cmd = [
        "nvidia-smi", f"--id={uuid}",
        "--query-compute-apps=pid,used_gpu_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False)
    except FileNotFoundError:
        return [], "nvidia-smi not found; external GPU occupancy was not checked"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"nvidia-smi occupancy probe failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        return [], (f"nvidia-smi occupancy probe exited {proc.returncode}"
                    + (f": {detail}" if detail else ""))

    rows: List[ComputeProcess] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            return [], f"unparseable nvidia-smi compute-apps row: {line!r}"
        try:
            pid = int(fields[0])
        except ValueError:
            return [], f"unparseable nvidia-smi compute-apps pid: {fields[0]!r}"
        try:
            used_memory_mib = int(fields[1])
        except ValueError:
            used_memory_mib = None
        rows.append(ComputeProcess(
            pid=pid, command=fields[2], used_memory_mib=used_memory_mib))
    return rows, None


def gpu_uuid(device: int = 0) -> Optional[str]:
    """Stable identity of the physical device index `device` maps to.

    None when the box exposes no device identity (CPU-only — nothing to
    lock).

    CUDA first, then NPU. The NPU arm exists because this function returned
    None on an Ascend box, and `GpuLock.acquire` treats None as "nothing to
    lock" — so every 910C run was unprotected. Two sweeps on one die would
    both have measured, both recorded, and neither row would say the other
    was there, which is the exact contamination this module exists to
    prevent. Found while scoping multi-process collective measurement:
    that work makes several processes share a box by design, and it would
    have made this silent gap load-bearing.

    `torch.npu.get_device_properties(n).uuid` is a device UUID of the same
    shape as CUDA's (`00cc0800-d001-0000-0000-f89a252b6ce2`), so it keys
    the same way. Prefixed `NPU-` rather than `GPU-` so one box that
    somehow exposed both could not collide.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            uid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            if not uid:
                return None
            value = str(uid)
            return value if value.startswith("GPU-") else f"GPU-{value}"
    except Exception:
        pass
    # Imported here, not at module scope: a CUDA-only box has no torch_npu
    # and must not pay an ImportError on every lock.
    try:
        import torch_npu  # noqa: F401  (registers the `npu` namespace)
        if not torch.npu.is_available():
            return None
        uid = getattr(torch.npu.get_device_properties(device), "uuid", None)
        if not uid:
            return None
        value = str(uid)
        return value if value.startswith("NPU-") else f"NPU-{value}"
    except Exception:
        return None


def _lock_dir() -> str:
    # /var/lock is the conventional home and is tmpfs on modern systems (so a
    # stale file cannot outlive a reboot); fall back to /tmp when it is not
    # writable, e.g. an unprivileged container.
    for d in ("/var/lock", "/tmp"):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    return "/tmp"


def lock_path_for(uuid: str) -> str:
    return os.path.join(_lock_dir(), f"kernel_bench.gpu.{uuid}.lock")


#: How long a run that holds the lock waits for a process already on the
#: card to leave before refusing. The keepalive's burst is such a process:
#: it starts in the gaps between a sweep's envs (minutes of `uv` resolving,
#: no CUDA context), and it yields to a held lock (`held`) within seconds.
#: Without the wait, every env that started inside a burst was refused.
FOREIGN_GRACE_S = 30.0

#: What the preflight says on a device with no occupancy probe. A constant
#: rather than a literal so a caller can test for it, and honest rather than
#: silent: the flock still protects, but it only protects processes that
#: take it.
_NPU_NO_OCCUPANCY_PROBE = (
    "no occupancy probe for this device: the flock protects only processes "
    "that take it, so a foreign process on the same die is not detected")


def held(uuid: str) -> bool:
    """Whether a run holds `uuid`'s lock right now — probed without taking
    it for longer than the probe."""
    import fcntl
    try:
        fh = open(lock_path_for(uuid), "a+", encoding="utf-8")  # noqa: SIM115
    except OSError:
        return False
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


def _holder_of(path: str) -> str:
    """Best-effort description of the process holding `path`."""
    try:
        with open(path, encoding="utf-8") as f:
            info = f.read().strip()
        return info or "(holder wrote no info)"
    except OSError:
        return "(holder unknown)"


class GpuLock:
    """flock-based exclusive lock on one physical GPU, or on several at once.

    A no-op when the box exposes no device identity, so CPU-only test runs
    and dry runs are unaffected. `flock` is released by the kernel when the
    process exits — including SIGKILL — so a crashed sweep never leaves the
    card locked.

    `devices` names several dies for a measurement that spans them (an
    N-rank collective). They are acquired in ASCENDING IDENTITY
    ORDER, never in the order given: two processes asking for {die 0, die 1}
    and {die 1, die 0} would otherwise each take one and wait forever for
    the other. One order globally is what makes multi-die locking
    deadlock-free, and it has to live here — a caller that sorts its own
    list is a second statement of the rule, and the second caller will not.
    """

    def __init__(self, device: int = 0, *, devices: Optional[List[int]] = None,
                 wait: bool = False, enabled: bool = True):
        self._wait = wait
        self._enabled = enabled
        self._devices = [device] if devices is None else list(devices)
        # Ascending, and de-duplicated: the order is the deadlock-freedom
        # argument, so it is imposed rather than trusted.
        uuids: List[str] = []
        if enabled:
            for index in sorted(set(self._devices)):
                uid = gpu_uuid(index)
                if uid and uid not in uuids:
                    uuids.append(uid)
            uuids.sort()
        self._uuids = uuids
        self._paths = [lock_path_for(u) for u in uuids]
        # The first device keeps the single-device accessors' meaning.
        self._uuid = uuids[0] if uuids else None
        self._path = self._paths[0] if self._paths else None
        self._handles: List[Tuple[str, object]] = []
        self._fh = None
        self._preflight_warning = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def active(self) -> bool:
        return bool(self._handles)

    @property
    def uuids(self) -> List[str]:
        """Every identity this lock holds, in the order it took them."""
        return list(self._uuids)

    @property
    def preflight_warning(self) -> Optional[str]:
        return self._preflight_warning

    def acquire(self) -> "GpuLock":
        if not self._paths:
            return self                       # nothing to lock
        import fcntl
        # One identity at a time, in the order `__init__` imposed. A partial
        # acquisition is released before raising: holding die 0 while
        # refusing die 1 would block every other process on a lock this run
        # is not going to use.
        # `strict`: the two are built together in `__init__`, so a length
        # mismatch is a bug here rather than a die to skip silently.
        for uuid, path in zip(self._uuids, self._paths, strict=True):
            fh = open(path, "a+", encoding="utf-8")  # noqa: SIM115
            flags = (fcntl.LOCK_EX if self._wait
                     else (fcntl.LOCK_EX | fcntl.LOCK_NB))
            try:
                fcntl.flock(fh.fileno(), flags)
            except OSError as e:
                fh.close()
                self._release_handles()
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    holder = _holder_of(path)
                    raise GpuBusy(
                        f"GPU {uuid} is already being measured by: {holder}\n"
                        f"  A second run on the same card makes BOTH sets of "
                        f"latencies wrong, and nothing in the emitted rows "
                        f"would record that it happened.\n"
                        f"  Wait for it, pick another card "
                        f"(CUDA_VISIBLE_DEVICES=…), or pass --wait-for-gpu "
                        f"to queue behind it."
                    ) from e
                raise
            self._handles.append((uuid, fh))
        self._fh = self._handles[0][1] if self._handles else None

        # Holding the lock is the signal a yielding process (the keepalive)
        # watches for, so wait out its exit before calling the card busy.
        # Asked of EVERY identity: one foreign process on the second die is
        # as contaminating as one on the first.
        warnings: List[str] = []
        foreign: List[ComputeProcess] = []
        for uuid, _fh in self._handles:
            waited = 0.0
            # The occupancy probe is nvidia-smi's. On an NPU it would answer
            # "nvidia-smi not found", which reads as "could not check" and
            # trains the operator to ignore the warning. Say what is true
            # instead: there is no probe for this device, and the flock
            # alone is the protection.
            if uuid.startswith("NPU-"):
                processes, warning = [], _NPU_NO_OCCUPANCY_PROBE
            else:
                processes, warning = _query_compute_apps(uuid)
            while True:
                here = [p for p in processes if p.pid != os.getpid()]
                if not here or waited >= FOREIGN_GRACE_S:
                    break
                time.sleep(2.0)
                waited += 2.0
                if not uuid.startswith("NPU-"):
                    processes, warning = _query_compute_apps(uuid)
            if warning:
                warnings.append(warning)
            foreign.extend(here)
        self._preflight_warning = "; ".join(warnings) or None
        if foreign:
            self._release_handles()
            details = "; ".join(
                f"pid={proc.pid} cmd={proc.command}"
                + (f" memory={proc.used_memory_mib} MiB"
                   if proc.used_memory_mib is not None else "")
                for proc in foreign)
            raise GpuBusy(
                f"{' / '.join(self._uuids)} already has compute process(es): "
                f"{details}\n"
                f"  They did not acquire kernel_bench's lock, but they can "
                f"still contaminate every measured latency. Stop them or "
                f"select another card before running the sweep."
            )
        # Record who holds it, for the next process's error message.
        try:
            with open("/proc/self/cmdline", "rb") as c:
                cmd = c.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        for _uuid, fh in self._handles:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()} cmd={cmd}\n")
            fh.flush()
        return self

    def _release_handles(self) -> None:
        import fcntl
        for _uuid, fh in self._handles:
            try:
                fh.seek(0)
                fh.truncate()
                fh.flush()
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                fh.close()
        self._handles = []
        self._fh = None

    def release(self) -> None:
        self._release_handles()

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
