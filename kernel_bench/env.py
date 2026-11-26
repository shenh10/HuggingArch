"""Environment detection — what THIS provider env can actually run.

The harness core is one thin package installed into many isolated envs
(one per pinned library stack). `detect_env` reports the ground truth of
the current env: torch/cuda, the GPU, and which kernel libraries are
importable + their versions. Providers consult it for availability +
provenance so every emitted row is stamped with what ACTUALLY produced
it, never what was requested.
"""
from __future__ import annotations

import re
import importlib
from dataclasses import dataclass, field
from typing import Dict, Optional

from backend.inference import benchmarks as _benchmarks

# Kernel libraries we probe for. Import name → the pip/display name we
# record. A given env will have only a subset installed.
@dataclass
class Env:
    torch_version: Optional[str]
    cuda_available: bool
    device_name: Optional[str]
    gpu_sku: Optional[str]           # our canonical SKU (gpu_specs), if resolvable
    peak_bw_gbps: Optional[float]
    sm: Optional[str] = None         # compute-capability tag, e.g. "sm_89" (Ada)
    #: The NVIDIA KERNEL-mode driver this machine runs, e.g. "575.57.08" —
    #: the version string `nvidia-smi` prints. A property of the BOX: every
    #: env on it sees the same one, and nothing short of loading a different
    #: kernel module changes it. Read from `/proc/driver/nvidia/version`,
    #: which is the kernel module reporting itself and so cannot be moved by
    #: `LD_LIBRARY_PATH`. None when there is no NVIDIA kernel module.
    kmd_version: Optional[str] = None
    #: The NVIDIA USER-mode driver (`libcuda.so`) THIS PROCESS loaded, in the
    #: same version form, e.g. "580.95.05". A property of the PROCESS: it is
    #: whichever `libcuda.so.1` the loader resolved, which `LD_LIBRARY_PATH`
    #: decides — so two envs on one box can differ, and that is exactly the
    #: case this field exists for.
    #:
    #: Normally the box's own, and then it EQUALS `kmd_version`. It differs
    #: whenever something else is ahead of it on `LD_LIBRARY_PATH` — on this
    #: repo's H100 box that is NVIDIA's forward-compatibility package, which
    #: is how a CUDA-13 wheel runs on a CUDA-12 kernel driver with no reboot
    #: (`kernel_bench/envs/boxes/h100-sxm-cu128.toml`, `[library_env]`). So
    #: the PAIR says what drove a measurement, and
    #: `benchmarks.is_driver_substituted` reads off it that the driver was not
    #: the box's own — not its direction, and not whether the pairing is
    #: supported — without anyone keeping a list of which envs are set up
    #: that way.
    #:
    #: None when no `libcuda` is mapped, which is also the answer when this
    #: env has no CUDA at all.
    umd_version: Optional[str] = None
    libs: Dict[str, str] = field(default_factory=dict)   # import_name -> version
    #: The library this env was built for (`run --library`, which `run-all`
    #: passes), or None for an env nobody built for one library (a bare
    #: `run`). Importable is not owned: every env carries torch, and the
    #: sglang / vllm envs carry flashinfer, so without this each env measured
    #: those providers again under its own torch / flashinfer release — one
    #: cell timed three times in three minutes. An env measures the providers
    #: registered under its library (`Provider.library`, the same field
    #: `dependency_manifest` groups an env's `# Providers:` line by) and no
    #: others.
    library: Optional[str] = None
    #: How many cards this process can drive together: its visible device
    #: count at run time; the same count, read off the box and the
    #: environment a run will get, when a plan judges ahead of time
    #: (`BoxEnvs.devices`). A collective cell needs one card per rank of its
    #: group (`n_comm_ranks`), and this is the other half of that question
    #: (`Provider.refusals`). None when nobody said, which judges nothing.
    devices: Optional[int] = None

    def peak_tflops(self, dtype: str) -> Optional[float]:
        """The card's peak (TFLOPs) at the MMA family `dtype` runs on —
        `gpu_specs.peak_flops_for_dtype`, the one lookup the estimate also
        reads. An FP8 GEMM judged against the BF16 peak reads 2× its real
        utilization, and the SOL gate refused honest rows above 1088 TF on an
        H100. None when the SKU is unknown or its sheet has no column for
        that family: an unstated peak, which the gate does not judge."""
        from backend.inference.gpu_specs import get_gpu, peak_flops_for_dtype
        flops = peak_flops_for_dtype(get_gpu(self.gpu_sku) if self.gpu_sku else None, dtype)
        return flops / 1e12 if flops else None

    def has(self, import_name: str) -> bool:
        return import_name in self.libs

    def version_of(self, import_name: str) -> Optional[str]:
        return self.libs.get(import_name)

    def provenance(self) -> Dict[str, str]:
        """Reproducibility blob for the emitted row (destined for a future
        `env_json` schema column; for now surfaced in logs / contributor)."""
        p = {"torch": self.torch_version or "", "gpu": self.device_name or ""}
        p.update(self.libs)
        return p


def _probe_version(import_name: str) -> Optional[str]:
    """Version of the library behind `import_name`, or None if absent.

    An import gate may name a SUBMODULE — `fla.ops`, `flash_attn.cute` — when
    that is what the build actually needs and the top-level package imports
    without it. A submodule carries no `__version__`, so reading only its
    attribute recorded `framework_version=unknown` for 63 rows, which defeats
    pinning the dependency to a release in the first place: the corpus is meant
    to say which version produced a number. Fall back to the top-level package,
    then to the installed distribution.
    """
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    v = getattr(mod, "__version__", None)
    if v:
        return v
    root = import_name.split(".")[0]
    if root != import_name:
        try:
            v = getattr(importlib.import_module(root), "__version__", None)
        except Exception:
            v = None
        if v:
            return v
    try:
        from importlib.metadata import version as _dist
        return _dist(root)
    except Exception:
        # Not "unknown": None is the honest answer, and the emit side refuses
        # to write a corpus row without a provider version — the version is
        # half of the row's identity.
        return None


#: What a driver version looks like — the SCHEMA's grammar, imported rather
#: than restated, so what this reads out of `/proc` cannot drift from what
#: the column it is written to will accept (`benchmarks._check_driver_
#: version`). Three-or-more-digit branch, which is what separates a driver
#: (`575.57.08`) from a CUDA version (`13.0`).
_DRIVER_VERSION = re.compile(_benchmarks.DRIVER_VERSION_PATTERN)

#: The whole basename of a mapped user-mode driver, version captured. ONE
#: pattern rather than a prefix test plus a fixed-length slice: `libcudart.so.
#: 13.0` is the CUDA RUNTIME and must not be read as a driver, and under a
#: prefix test it is only excluded by the slice landing mid-name — an
#: accident rather than a decision, and one that leaves the rejection
#: untested (a mutation loosening the prefix did not fail a single
#: assertion). Here the name and the version are one match, so loosening
#: either is visible.
_MAPPED_UMD = re.compile(
    rf"^libcuda\.so\.({_benchmarks.DRIVER_VERSION_PATTERN})$")

_NVRM_PROC = "/proc/driver/nvidia/version"
_SELF_MAPS = "/proc/self/maps"


def kernel_mode_driver(nvrm_text: str) -> Optional[str]:
    """The kernel-mode driver version in `/proc/driver/nvidia/version`'s text.

    The file's first line names the module and then its version, and the
    wording varies by build — "NVIDIA UNIX x86_64 Kernel Module  575.57.08"
    against "NVIDIA UNIX Open Kernel Module for x86_64  595.84" — so the
    version is taken as the first driver-shaped token on the NVRM line rather
    than by counting words. `x86_64` cannot be mistaken for it: the schema's
    grammar wants a three-or-more-digit branch and at least one dotted
    component, so neither `86` nor `64` can match.
    """
    for line in nvrm_text.splitlines():
        if "NVRM" not in line:
            continue
        found = _DRIVER_VERSION.search(line)
        return found.group(0) if found else None
    return None


def loaded_user_mode_driver(maps_text: str) -> Optional[str]:
    """The user-mode driver version this process has mapped, from
    `/proc/self/maps`' text.

    The loader resolves `libcuda.so.1` through to the real file and maps THAT
    path, so the mapped basename carries the version — the system driver's
    `libcuda.so.595.84`, or a compat package's `libcuda.so.580.95.05` out of
    its own directory. Reading the mapping is
    what makes this the driver that is actually in force rather than the one a
    box profile meant to supply: an `LD_LIBRARY_PATH` that failed to take
    effect reports the system driver here, correctly.

    None when no versioned `libcuda` is mapped — no CUDA in this env, or a
    `libcuda.so.1` that is itself the real file and names no version. Both are
    honestly unknown and must not be filled in with the kernel driver, which
    would assert the very thing this pair exists to check.
    """
    for line in maps_text.splitlines():
        path = line.rsplit(" ", 1)[-1].strip()
        found = _MAPPED_UMD.match(path.rsplit("/", 1)[-1])
        if found:
            return found.group(1)
    return None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def detect_env() -> Env:
    torch_version = None
    cuda_available = False
    device_name = None
    peak_bw_gbps = None
    sm = None
    devices = 0
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            devices = int(torch.cuda.device_count())
            device_name = torch.cuda.get_device_name(0)
            gpu_sku, peak_bw_gbps = _resolve_peaks(device_name)
            try:
                major, minor = torch.cuda.get_device_capability(0)
                sm = f"sm_{major}{minor}"        # (8,9) → "sm_89"
            except Exception:
                sm = None
        else:
            gpu_sku = None
    except Exception:
        gpu_sku = None

    # Probe exactly the modules the registered providers declare (+ torch).
    # No hand-maintained list — the providers ARE the source of truth.
    from .providers import probe_import_names
    libs: Dict[str, str] = {}
    for import_name in probe_import_names():
        v = _probe_version(import_name)
        if v is not None:
            libs[import_name] = v

    # What drove this measurement. Read AFTER the torch import above, because
    # that is what loads `libcuda` — asked first, `/proc/self/maps` would
    # answer "nothing mapped" for every env and the pair would never differ.
    #
    # Both unconditionally, not behind `cuda_available`: the box's kernel
    # driver is a fact about the machine whether or not this env can reach a
    # device, and the compat case is precisely one where a lower gate passes
    # while the call does not (that box profile's "FALSE GREEN" note).
    return Env(
        torch_version=torch_version,
        cuda_available=cuda_available,
        device_name=device_name,
        gpu_sku=gpu_sku if cuda_available else None,
        peak_bw_gbps=peak_bw_gbps,
        sm=sm if cuda_available else None,
        kmd_version=kernel_mode_driver(_read_text(_NVRM_PROC)),
        umd_version=loaded_user_mode_driver(_read_text(_SELF_MAPS)),
        libs=libs,
        devices=devices,
    )


def _resolve_peaks(device_name: str):
    """Map the torch device name to canonical peaks from the GPU spec DB:
    (sku, mem BW GB/s). Best effort — returns (None, None) when the card is
    unknown. The compute peak is not one number per card: it depends on the
    MMA a kernel runs, so it is read per row (`Env.peak_tflops`)."""
    name_l = device_name.lower()
    name_compact = name_l.replace(" ", "")
    try:
        # load_gpu_specs() is a LIST of GPUSpec. Match a spec's `.sku` two ways:
        #  * datacenter SKUs carry a specific family token ("a100" in
        #    "H100-SXM" -> "h100") that appears verbatim in the device name
        #    ("h100" ∈ "nvidia h100 80gb hbm3"). Require the token to mix a
        #    letter AND a digit so a generic prefix ("rtx") can't over-match.
        #  * consumer SKUs ("RTX-4090") disambiguate only on the compact model
        #    form ("rtx4090" ∈ "nvidiageforcertx4090").
        from backend.inference.gpu_specs import load_gpu_specs
        candidates = []
        for spec in load_gpu_specs():
            sku_l = (spec.sku or "").lower()
            if not sku_l:
                continue
            family = sku_l.split("-")[0]
            family_specific = (any(c.isalpha() for c in family)
                               and any(c.isdigit() for c in family))
            # Neither token may be followed by another digit: "h20" is a
            # substring of "h200", so a bare `in` check made every H200 look
            # like an H20 too (previously masked by first-match-wins).
            def _hit(token: str, hay: str) -> bool:
                return bool(token) and re.search(
                    re.escape(token) + r"(?!\d)", hay) is not None

            if (family_specific and _hit(family, name_l)) or \
                    _hit(sku_l.replace("-", ""), name_compact):
                candidates.append(spec)
        if not candidates:
            return None, None
        if len(candidates) == 1:
            spec = candidates[0]
            return spec.sku, spec.mem_bw_gbps

        # A family token can be shared by several variants (H100-SXM vs
        # H100-PCIe both reduce to "h100"). Taking the first match silently
        # picked whichever the CSV happened to list first — on an "NVIDIA H100
        # 80GB HBM3" (SXM) box that was H100-PCIe, recording 2000 GB/s instead
        # of 3350 and inflating every %-of-peak derived from those rows by
        # ~1.7×. Disambiguate on the variant markers the device name actually
        # carries: the form factor ("sxm" / "pcie"), the capacity ("80gb" —
        # what separates A100-40G-SXM from A100-80G-SXM), and the memory
        # generation ("hbm3" vs "hbm2e"), which separates exactly the H100 pair
        # the other two don't. Each round narrows; a round that matches nothing
        # is not evidence, so it leaves the candidate set alone.
        def _marks(spec) -> list:
            out = [(getattr(spec, "form_factor", "") or "").lower()[:4],
                   (getattr(spec, "mem_type", "") or "").lower()]
            mem_gib = getattr(spec, "mem_gib", None)
            if mem_gib:
                out.insert(1, f"{int(mem_gib)}gb")
            return out

        for i in range(3):
            marked = [s for s in candidates
                      if (_marks(s)[i] or "\0") in name_compact]
            if len(marked) == 1:
                spec = marked[0]
                return spec.sku, spec.mem_bw_gbps
            if marked:
                candidates = marked

        # Still ambiguous: report unknown rather than guess. An empty peak is a
        # visible gap; a wrong peak silently corrupts every MFU computed from it.
        return None, None
    except Exception:
        pass
    return None, None
