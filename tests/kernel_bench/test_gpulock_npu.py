"""The lock has to exist on a device that is not CUDA.

`gpu_uuid` asked `torch.cuda` and returned None everywhere else, and
`GpuLock.acquire` reads None as "nothing to lock". On an Ascend box that
made every protection in this module a no-op: two sweeps on one die both
measured, both wrote rows, and neither row said the other was there. The
module's own docstring is the cost of exactly that — 16k rows quarantined
after five sweeps shared an H100.

The multi-die arm exists for collective measurement: one measurement
spanning N dies takes N locks, and it must take them in one global order
or two runs asking for the same set in different orders deadlock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel_bench import gpulock

#: Deliberately NOT ascending by index: the lock must sort by identity, and
#: an identity order that happened to match the index order would pass while
#: sorting nothing.
_UUIDS = {0: "NPU-ccc", 1: "NPU-aaa", 2: "NPU-bbb"}


@pytest.fixture
def lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: _UUIDS.get(device))
    return tmp_path


def _npu_present() -> bool:
    """Whether this box has an NPU this process can see.

    `torch.npu` does not exist at all without `torch_npu`, so the attribute
    has to be probed — `torch.npu.is_available()` raises AttributeError on
    a CUDA-only box rather than answering False.
    """
    try:
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


def test_no_cuda_means_no_lock_was_the_bug(monkeypatch):
    """`gpu_uuid` must not answer None on a box that has a device.

    Pinned as a shape rather than a value: whichever device is present, an
    identity exists. The CUDA-only version failed here.
    """
    if not _npu_present():
        pytest.skip("no NPU on this box")
    assert gpulock.gpu_uuid(0) is not None


def test_the_npu_identity_is_namespaced():
    """A UUID from another backend must not be mistakable for CUDA's.

    The flock path is built from this string, so a collision would have two
    different physical devices share one lock.
    """
    if not _npu_present():
        pytest.skip("no NPU on this box")
    assert gpulock.gpu_uuid(0).startswith("NPU-")


def test_multi_device_is_acquired_in_ascending_identity_order(lockdir):
    """The order is the deadlock-freedom argument, so it is imposed.

    Two runs asking for {0, 1} and {1, 0} each take one lock and wait
    forever for the other. Sorting inside the lock is what makes that
    impossible; a caller that sorts its own list is a second statement of
    the rule and the next caller will not make it.
    """
    lock = gpulock.GpuLock(devices=[2, 0, 1]).acquire()
    try:
        assert lock.uuids == ["NPU-aaa", "NPU-bbb", "NPU-ccc"]
        assert lock.active
    finally:
        lock.release()


def test_repeated_devices_take_one_lock_each(lockdir):
    lock = gpulock.GpuLock(devices=[1, 1, 0]).acquire()
    try:
        assert lock.uuids == ["NPU-aaa", "NPU-ccc"]
    finally:
        lock.release()


def test_a_partial_acquisition_is_rolled_back(lockdir):
    """Refusing die 1 must not leave die 0 held.

    Die 0 is acquired first (ascending), so a run that cannot get die 1 has
    already taken die 0 — and holding it would block every process that
    wants die 0 alone, for a measurement this run will never make.
    """
    with gpulock.GpuLock(devices=[0]):
        with pytest.raises(gpulock.GpuBusy):
            gpulock.GpuLock(devices=[0, 1]).acquire()
        # The rollback is what this asserts: die 1 is free again.
        other = gpulock.GpuLock(devices=[1]).acquire()
        other.release()


def test_single_device_call_sites_are_unchanged(lockdir):
    """`GpuLock()` keeps meaning one card, keyed the same way."""
    lock = gpulock.GpuLock().acquire()
    try:
        assert lock.active
        assert lock.path.endswith(".lock")
        assert lock.uuids == ["NPU-ccc"]
    finally:
        lock.release()


def test_no_device_at_all_is_still_a_no_op(monkeypatch):
    """A CPU-only box locks nothing and must not raise."""
    monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: None)
    lock = gpulock.GpuLock(devices=[0, 1]).acquire()
    try:
        assert not lock.active
        assert lock.path is None
        assert lock.uuids == []
    finally:
        lock.release()


def test_the_npu_preflight_does_not_claim_nvidia_smi_could_not_be_found(
        lockdir):
    """Saying "nvidia-smi not found" on an NPU trains the operator to
    ignore the warning. Say there is no probe instead."""
    lock = gpulock.GpuLock(devices=[0]).acquire()
    try:
        assert "nvidia-smi" not in (lock.preflight_warning or "")
        assert lock.preflight_warning == gpulock._NPU_NO_OCCUPANCY_PROBE
    finally:
        lock.release()
