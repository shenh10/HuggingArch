"""The one place that knows how to talk to a compute device.

`torch.cuda.*` used to be spelled out at every site that synchronized, took
a timing event, asked for free memory or sized the L2 sweep — about thirty
of them across `timing.py`, `driver.py` and `env.py`. Each was correct and
each assumed CUDA, so a box with another accelerator did not fail: it fell
through to the CPU branch and produced a wall-clock number that looked like
a measurement. That is the failure `ARCHITECTURE.md` opens with — true in
one place, assumed in another, nothing crashes, the number is wrong.

So the device is named once and read everywhere else. The vocabulary is the
sheet's (`gpu_specs.csv` `device`: `cuda`, `npu`), because a provider's
dispatch, a framework binding's `devices:` and a measured row's card all
already speak it; a second spelling here would be a second fact.

What a backend does NOT paper over: where an accelerator has no equivalent
of a CUDA primitive, the backend says so (`spin` returns False) rather than
emulating it, and the caller decides. Two devices that reach an answer by
different routes must not claim the same answer — which is why the timing
method a row records is the backend's to name (`timing_method`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

#: The device names this harness can measure on, in the sheet's vocabulary
#: (`gpu_specs.csv` `device`). Order is probe order.
DEVICES: Tuple[str, ...] = ("cuda", "npu")


class NoDeviceBackend(RuntimeError):
    """Asked for a device backend on a box that has no accelerator, or for a
    device this harness does not know. Raised rather than falling back to the
    CPU: a CPU number recorded under an accelerator's name is the defect this
    module exists to prevent."""


@dataclass(frozen=True)
class DeviceBackend:
    """One accelerator's primitives, as the harness uses them.

    `name` is the sheet's device name; `module` is the torch namespace that
    implements it (`torch.cuda`, `torch.npu`). Everything else is read off
    `module`, so adding a device is naming it in `DEVICES` and checking the
    primitives below exist — not another branch at thirty call sites.
    """

    name: str
    module: Any

    # -- presence ---------------------------------------------------------
    @property
    def tensor_device(self) -> str:
        """The string `torch.tensor(..., device=)` takes for this device."""
        return self.name

    def is_available(self) -> bool:
        return bool(self.module.is_available())

    # -- the primitives the harness uses ----------------------------------
    def synchronize(self) -> None:
        self.module.synchronize()

    def event(self) -> Any:
        """A timing event. `elapsed_time` between two is milliseconds."""
        return self.module.Event(enable_timing=True)

    def device_count(self) -> int:
        return int(self.module.device_count())

    def current_device(self) -> int:
        fn = getattr(self.module, "current_device", None)
        return int(fn()) if fn is not None else 0

    def set_device(self, index: int) -> None:
        """Bind this process to `index`. A collective rank calls this
        before it puts tensors on the device; without it every rank
        lands on die 0 and the group is the identity of one buffer."""
        self.module.set_device(int(index))

    def properties(self, index: Optional[int] = None) -> Any:
        if index is None:
            index = self.current_device()
        return self.module.get_device_properties(index)

    def device_name(self, index: Optional[int] = None) -> Optional[str]:
        if index is None:
            index = self.current_device()
        try:
            return self.module.get_device_name(index)
        except Exception:
            return None

    def mem_get_info(self) -> Optional[Tuple[int, int]]:
        """`(free, total)` bytes, or None where the device cannot say."""
        try:
            return tuple(self.module.mem_get_info())  # type: ignore[return-value]
        except Exception:
            return None

    def empty_cache(self) -> None:
        try:
            self.module.empty_cache()
        except Exception:
            pass

    def l2_bytes(self) -> Optional[int]:
        """The L2 the sweep must exceed to evict a launch's residents.

        Read off the device, not off a table: both torch namespaces spell it
        `L2_cache_size` (an Ascend 910C die reports 176160768 = 168 MiB), so
        a constant here would be a second statement of a fact the driver
        already makes, free to drift from it.
        """
        props = self.properties()
        value = (getattr(props, "L2_cache_size", 0)
                 or getattr(props, "l2_cache_size", 0))
        return int(value) or None

    def spin(self, cycles: int) -> bool:
        """Hold the device busy for `cycles`, so the start event cannot fire
        before the caller's kernels are queued.

        Returns False when this device has no such primitive, and the caller
        must keep the queue full another way. Not emulated with a loop of
        real kernels: those carry their own memory footprint, which evicts
        the very lines the sweep just arranged (measured on a 910C: a 64 MiB
        matmul hold moved a 32 MiB copy from 533 to 946 GB/s).

        Ascend is the False case twice over — `torch.npu._sleep` exists but
        its operator is not built for the `ascend910_93` SoC (CANN:
        "SoC version ascend910_93 verification failed"), and there is no
        other zero-footprint spin.
        """
        sleep = getattr(self.module, "_sleep", None)
        if sleep is None:
            return False
        try:
            sleep(int(cycles))
            return True
        except Exception:
            # Present but not implemented for this SoC.
            return False


def backend_named(name: str) -> DeviceBackend:
    """The backend for a device name, whether or not a device is present.

    Importing the namespace is enough: `torch.npu` needs `torch_npu`
    imported first, which is why that import is here and not at a call site.
    """
    import torch

    if name == "cuda":
        return DeviceBackend("cuda", torch.cuda)
    if name == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise NoDeviceBackend(
                "npu needs `torch_npu` installed to reach the device") from exc
        return DeviceBackend("npu", torch.npu)
    raise NoDeviceBackend(
        f"no device backend named {name!r}; this harness measures on "
        f"{list(DEVICES)}")


def detect_backend() -> Optional[DeviceBackend]:
    """The accelerator this box has, or None.

    Probed in `DEVICES` order and answered by the device itself
    (`is_available`), so a box with neither is told apart from a box whose
    device the caller named wrong.
    """
    for name in DEVICES:
        try:
            backend = backend_named(name)
        except NoDeviceBackend:
            continue
        try:
            if backend.is_available():
                return backend
        except Exception:
            continue
    return None
