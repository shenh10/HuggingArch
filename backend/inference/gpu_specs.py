"""Loader for backend/inference/gpu_specs.csv.

The CSV is the source of truth for NVIDIA SKUs supported in the
Inference tab. Numeric columns are parsed float, empty cells → None.

What a compute column means: the card's dense peak at **the clock it
actually runs tensor-core work at** — a speed-of-light bound, so no kernel
may exceed it. For a part specified at a clock it holds (every datacenter
SKU here) that is the vendor's published peak, and the cell is the sheet
value. A GeForce part is different: its published boost clock is a floor
that GPU Boost routinely clears, so the sheet peak is *not* a bound and the
cell is the sheet value scaled to the clock measured under a sustained
tensor-core GEMM (the `notes` cell records that clock). Writing the sheet
number there instead puts every consumer of this file below the hardware:
the roofline under-predicts throughput and MFU reads above 100%, which is
how 1,439 RTX-4090 rows came to claim up to 105% of peak.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, asdict, fields
from types import MappingProxyType
from typing import Dict, FrozenSet, Mapping, Optional, Tuple, get_type_hints

from backend.arch.dtypes import DTYPES, peak_kind
from backend.code_trace.memo import memoize

_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_specs.csv")

@dataclass
class GPUSpec:
    sku: str
    arch: str
    #: The platform a serving framework's dispatch checks (`cuda`): sglang
    #: gates CUDA-only kernels on `_is_cuda`. `arch` is the generation name,
    #: not the vendor, so the device is its own column.
    device: str
    form_factor: str
    fp32_tflops: Optional[float]
    tf32_tflops: Optional[float]
    bf16_tflops: Optional[float]
    fp16_tflops: Optional[float]
    int8_tops: Optional[float]
    fp8_tflops: Optional[float]
    fp4_tflops: Optional[float]
    #: The per-direction NVLink bandwidth. A value also declares the
    #: topology: every GPU of the card's NVLink domain (`gpus_per_node` of
    #: them — an HGX baseboard, a whole NVL72 rack) reaches every other
    #: over NVLink, which is what a framework's peer-to-peer all-reduce
    #: kernels require. Empty: the GPUs talk over PCIe.
    nvlink_gbps: Optional[float]
    #: `yes` when the NVLink domain's switches reduce in the network and
    #: multicast (NVLink SHARP, "NVLS": third-generation NVSwitch and later,
    #: Hopper/Blackwell HGX and NVL72), `no` otherwise (A100's NVSwitch, a
    #: PCIe card). What the NVLS all-reduce needs (`comm_cost`).
    nvls: str
    pcie_gbps: Optional[float]
    rdma_gbps: Optional[float]
    #: HBM capacity as the vendor sheet prints it ("80GB"). The sheet says
    #: GB but the figure is binary: HBM stacks come in power-of-two sizes,
    #: so H100's "80GB" is 80 GiB (nvidia-smi: 81559 MiB). Bandwidth columns
    #: are the opposite — signalling rates, decimal GB/s.
    mem_gib: Optional[float]
    mem_type: str
    mem_bw_gbps: Optional[float]
    tdp_w: Optional[float]
    gpus_per_node: Optional[float] = None
    notes: str = ""
    #: The CUDA compute capability (`major.minor`, "9.0" for Hopper, "10.0" /
    #: "10.3" for the data-centre Blackwells): what a framework's kernel
    #: selection asks (vLLM `has_device_capability(100)`,
    #: `is_device_capability_family(100)`). A generation name (`arch`) spans
    #: several capabilities, so it is its own column. Empty off CUDA.
    compute_capability: str = ""
    #: Streaming multiprocessors available to runtime tile/split schedulers.
    #: Unknown is not inferred from architecture or peak FLOPs.
    sm_count: Optional[float] = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    @memoize(maxsize=None)
    def numeric_columns(cls) -> frozenset:
        """The CSV columns parsed as float: the fields typed `Optional[float]`."""
        hints = get_type_hints(cls)
        return frozenset(f.name for f in fields(cls)
                         if hints[f.name] == Optional[float])


def _parse_row(row: dict) -> GPUSpec:
    numeric = GPUSpec.numeric_columns()
    kwargs: dict = {}
    for k, v in row.items():
        # csv.DictReader files a row's OVERFLOW columns (more fields than the
        # header — e.g. a half-written row seen mid-write) under a None restkey.
        # Skip non-string keys so that never reaches GPUSpec(**kwargs) as the
        # cryptic "keywords must be strings"; a racing read is retried below.
        if not isinstance(k, str):
            continue
        if v is None or v == "":
            kwargs[k] = None if k in numeric else ""
            continue
        if k in numeric:
            try:
                kwargs[k] = float(v)
            except ValueError:
                kwargs[k] = None
        else:
            kwargs[k] = str(v).strip()
    return GPUSpec(**kwargs)


@memoize(maxsize=1)
def load_gpu_specs() -> list[GPUSpec]:
    # The CSV sits in a shared working tree other processes may rewrite; a read
    # that races a write can see a truncated/half-written file. Retry a few
    # times so a transient race doesn't become a hard failure. @lru_cache means
    # only the first (successful) load per process pays this.
    import time
    last: Exception | None = None
    for attempt in range(4):
        try:
            with open(_CSV_PATH, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            specs = [_parse_row(row) for row in rows]
            if specs:
                break
            last = ValueError("gpu_specs.csv parsed to zero rows")
        except Exception as e:
            last = e
        time.sleep(0.1 * (attempt + 1))
    else:
        raise RuntimeError(f"could not load gpu_specs.csv after retries: {last}")
    for spec in specs:
        check_compute_capability(spec)
        check_nvls(spec)
        check_memory_bandwidth(spec)
    return specs


def check_nvls(spec: GPUSpec) -> None:
    """A row says whether its NVLink domain reduces in the switch, `yes` or
    `no`: the NVLS all-reduce is priced off it, and a blank would be read
    as one or the other by whoever asked."""
    if spec.nvls not in ("yes", "no"):
        raise ValueError(
            f"gpu_specs.csv: {spec.sku} states nvls {spec.nvls!r}; every row "
            "says `yes` or `no`")
    if spec.nvls == "yes" and not spec.nvlink_gbps:
        raise ValueError(
            f"gpu_specs.csv: {spec.sku} states nvls yes without an NVLink "
            "domain (`nvlink_gbps`)")


def has_nvlink_domain(gpu: GPUSpec) -> bool:
    """Every GPU of the card's node reaches every other over NVLink
    (`GPUSpec.nvlink_gbps`)."""
    return bool(gpu is not None and gpu.nvlink_gbps)


def has_nvls(gpu: GPUSpec) -> bool:
    """The card's NVLink domain reduces in the switch (`GPUSpec.nvls`)."""
    return bool(gpu is not None and gpu.nvls == "yes")


def check_compute_capability(spec: GPUSpec) -> None:
    """A CUDA row states its compute capability as `major.minor`: a
    framework binding's `when.min_compute_capability` reads it, and a row
    without it would silently lose every kernel selected by it."""
    if spec.device != "cuda":
        return
    major, dot, minor = spec.compute_capability.partition(".")
    if not (dot and major.isdigit() and minor.isdigit()):
        raise ValueError(
            f"gpu_specs.csv: {spec.sku} (device cuda) states compute_capability "
            f"{spec.compute_capability!r}; every CUDA row needs one as "
            "`major.minor` (e.g. 9.0)")


def get_gpu(sku: str) -> Optional[GPUSpec]:
    for g in load_gpu_specs():
        if g.sku == sku:
            return g
    return None


def _peak_columns() -> Mapping[str, str]:
    """Each MMA family to the sheet column holding the card's peak for it.

    Both halves are declared already, so neither is spelled here: the
    families are the `peak` each `arch.dtypes` row states (the one place a
    dtype says which MMA it runs on), and the columns are `GPUSpec`'s own
    numeric fields, which the CSV header is checked against at load. The
    join is the naming law the sheet follows — one numeric column per
    family, named for it (`int8` prints integer ops, so the suffix is the
    sheet's, not this map's). A family with no column, or two, is refused
    HERE, at import, naming both sides: before, a dtype row declaring a new
    `peak` raised `KeyError` in the middle of pricing a leaf.
    """
    numeric = GPUSpec.numeric_columns()
    out: Dict[str, str] = {}
    for family in sorted({row.peak for row in DTYPES.values() if row.peak}):
        columns = sorted(c for c in numeric if c.startswith(family + "_"))
        if len(columns) != 1:
            raise ValueError(
                f"MMA family {family!r} (declared by an `arch.dtypes` row's "
                f"`peak`) needs exactly one numeric `GPUSpec` column named "
                f"`{family}_*` to read the card's peak from; found "
                f"{columns or 'none'}")
        out[family] = columns[0]
    return MappingProxyType(out)


#: Each MMA family `arch.dtypes.peak_kind` names, mapped to the `GPUSpec`
#: column that lists the card's peak for it — the one table every peak
#: reader (`peak_flops_for_dtype`, `peak_source`, `peak_families`) reads,
#: derived from the two declarations it joins (`_peak_columns`).
PEAK_COLUMNS: Mapping[str, str] = _peak_columns()


#: The sheet columns a card's PRICE is read from — the values that reach a
#: number and never a row. Each is read here, through the accessor named
#: beside it, and lands in one of the two types a card's price is
#: (`capacity._CardStamp` and `comm_cost.Fabric`):
#:
#:   * `mem_gib`     — `hbm_bytes`, the rank's budget;
#:   * `mem_bw_gbps` — `peak_hbm_bw`, every row's memory term;
#:   * the peak columns (`PEAK_COLUMNS`) — `peak_flops_for_dtype`, each
#:     row's compute term at its own dtype. Their PRESENCE is a walk fact
#:     too (`peak_families`): a leaf whose dtype has no column is walked
#:     weight-only. Value priced, presence walked;
#:   * `nvlink_gbps` / `pcie_gbps` — `peak_intra_bw_bytes_per_sec` and
#:     `intra_bw_label`; `nvlink_gbps` also says the card has an NVLink
#:     domain (`has_nvlink_domain`);
#:   * `rdma_gbps`   — `peak_inter_bw_bytes_per_sec`, `inter_bw_label`;
#:   * `nvls`        — `has_nvls`, which all-reduce the switch admits;
#:   * `gpus_per_node` — `gpus_per_node`, whether a group stays in a node;
#:   * `compute_capability` — a collective rung's own window
#:     (`comm_cost.CollectiveRung.runs_on`, off `Fabric`). It is a WALK
#:     fact as well, through the dispatch (`DispatchCondition.CARD_KEYS`).
#:
#: Exhaustive over the sheet with `DESCRIPTIVE_COLUMNS` and the walk
#: columns: a column in none of the three is refused at load
#: (`framework_bindings._check_card_fact_classification`).
PRICE_COLUMNS: FrozenSet[str] = frozenset({
    "mem_gib", "mem_bw_gbps", "nvlink_gbps", "pcie_gbps", "rdma_gbps",
    "nvls", "gpus_per_node", "compute_capability", *PEAK_COLUMNS.values()})

#: The sheet columns no walk and no number reads — what a row says about
#: itself, for a reader. Explicit, one entry per column with why: the third
#: bucket of the classification above is not where an unexamined column
#: lands. A column here that some pricing path starts reading is a column
#: to reclassify, not to leave.
DESCRIPTIVE_COLUMNS: Mapping[str, str] = MappingProxyType({
    "sku": "names the row — `get_gpu` finds a card by it and a peak says "
           "which cell it was read from (`peak_source`). No row and no "
           "number moves with the name",
    "arch": "the generation name. A generation spans several compute "
            "capabilities and several form factors, so it answers neither "
            "a dispatch nor a price: the facts a framework reads are "
            "`device` and `compute_capability`, which are their own "
            "columns (CLAUDE.md §3)",
    "form_factor": "SXM or PCIe, for a reader. What the form factor "
                   "actually changes is already in the columns it changes: "
                   "the NVLink and PCIe bandwidths and `gpus_per_node`",
    "mem_type": "HBM3e, GDDR7 — the part number behind `mem_bw_gbps` and "
                "`mem_gib`, which are what is priced",
    "tdp_w": "board power. Nothing this project estimates is per-watt",
    "tf32_tflops": "a peak column no peak reads. The sheet's peak columns "
                   "reach a row through `PEAK_COLUMNS`, which is the MMA "
                   "families `arch.dtypes` rows declare a `peak` for — and "
                   "no dtype declares `tf32`, so nothing computes in it "
                   "here and `peak_flops_for_dtype` can never select this "
                   "column. The day a dtype row does declare it, "
                   "`PEAK_COLUMNS` picks the column up and the "
                   "descriptive-only claim here is refused at load",
    "notes": "the sheet's provenance for a cell a reader would otherwise "
             "query — the clock a GeForce peak was scaled to (module "
             "docstring). Prose; nothing parses it",
})


def peak_flops_for_dtype(gpu: GPUSpec, dtype: str) -> Optional[float]:
    """Peak compute throughput (FLOPs/s) for a dtype **name**.

    The roofline needs the GPU's advertised peak at the MMA the kernel
    actually runs. Column choice is `arch.dtypes.peak_kind`, not a byte
    width: BF16 and FP16 are both 2 bytes and different columns.
    Returns None when the SKU sheet has no column for that family.
    """
    if gpu is None:
        return None
    t = getattr(gpu, PEAK_COLUMNS[peak_kind(dtype)])
    if t is None:
        return None
    return float(t) * 1e12


def peak_source(gpu: GPUSpec, dtype: str) -> str:
    """Where a peak came from: `"<sku>.<column>"` when the SKU sheet has the
    column for this dtype's MMA family, `"unknown"` when it does not. The
    response carries it beside every peak so a number is never shown
    without the cell it was read from."""
    if gpu is None:
        return "unknown"
    if peak_flops_for_dtype(gpu, dtype) is None:
        return "unknown"
    return f"{gpu.sku}.{PEAK_COLUMNS[peak_kind(dtype)]}"


def gbps_to_bytes_per_sec(gbps: Optional[float]) -> Optional[float]:
    """CSV bandwidth columns are decimal GB/s → bytes/s."""
    return float(gbps) * 1e9 if gbps else None


def inter_node_bw_gbps(rdma_gbps: Optional[float]) -> Optional[float]:
    """Cross-node link in CSV GB/s: the RDMA NIC, or nothing.

    PCIe is an intra-node fabric; a collective or a KV handoff that
    crosses node boundaries has no PCIe path. A SKU without an RDMA
    column therefore has no cross-node bandwidth, and the answer is None
    rather than a number borrowed from a different fabric.
    """
    return rdma_gbps or None


def hbm_bytes(gpu: GPUSpec) -> Optional[float]:
    """HBM capacity in bytes. The CSV column is GiB (see `GPUSpec.mem_gib`)."""
    if gpu is None or not gpu.mem_gib:
        return None
    return gpu.mem_gib * 1024 ** 3


def peak_hbm_bw(gpu: GPUSpec) -> Optional[float]:
    """HBM bandwidth in bytes/s for roofline `t_memory = bytes / bw`.
    CSV is GB/s (decimal: 1e9 bytes), so multiply accordingly.

    `None` means one thing only: NO CARD was chosen, so no time was asked
    for. A card always carries a bandwidth — `check_memory_bandwidth`
    refuses a sheet row that leaves the column empty at load — so a reader
    never has to tell "this card has no bandwidth" from "there is no
    card"."""
    if gpu is None:
        return None
    check_memory_bandwidth(gpu)
    return gbps_to_bytes_per_sec(gpu.mem_bw_gbps)


def peak_intra_bw_bytes_per_sec(gpu: GPUSpec) -> Optional[float]:
    """Intra-node interconnect bandwidth (bytes/s), per-direction.

    Collectives whose `comm_world <= gpus_per_node` ride the high-speed
    fabric inside a single node — NVLink on modern SKUs, PCIe on older
    or PCIe-form-factor cards. None when neither column is filled
    (degenerate; t_comm then degrades to 0).

    CSV convention: all bandwidth columns store the *unidirectional*
    per-direction value, i.e. the throughput a single ring step can
    actually drive in `bytes_sent / bw`. NVIDIA marketing publishes
    NVLink as a bidirectional aggregate (NVLink 4 H100 = 900 GB/s sum
    over both directions) — the CSV halves that down to the unidir
    figure (450 GB/s for H100) so consumers don't have to remember
    which fabric used which convention. PCIe Gen5 x16 = 64 GB/s/dir,
    PCIe Gen6 x16 = 128 GB/s/dir, NVLink 5 (B/GB-series) = 900 GB/s/dir.
    """
    if gpu is None:
        return None
    bw = gpu.nvlink_gbps if gpu.nvlink_gbps else gpu.pcie_gbps
    return gbps_to_bytes_per_sec(bw)


def peak_inter_bw_bytes_per_sec(gpu: GPUSpec) -> Optional[float]:
    """Inter-node interconnect bandwidth (bytes/s), per-direction.

    Collectives whose `comm_world > gpus_per_node` must cross node
    boundaries — RDMA (InfiniBand / RoCE / NVIDIA's CX-class NICs).
    None when the SKU lists no RDMA column: there is no cross-node path
    to price, and PCIe is not one.

    CSV convention: per-direction GB/s. CX-7 = 400 Gbps unidirectional
    = 50 GB/s; CX-8 = 800 Gbps = 100 GB/s; GB200/GB300 NDR/XDR =
    1800 Gbps aggregate per node = 225 GB/s.
    """
    if gpu is None:
        return None
    return gbps_to_bytes_per_sec(inter_node_bw_gbps(gpu.rdma_gbps))


def gpus_per_node(gpu: GPUSpec) -> int:
    """Number of GPUs in one NVLink domain / one physical node — the CSV's
    `gpus_per_node` column, the one place it is declared.

    Used by the per-op timing pass to decide whether a collective's
    `comm_world` fits inside one node (intra fabric) or crosses node
    boundaries (inter fabric). A SKU without the column is a sheet bug,
    not "8": GB200 NVL72 is 72, and 8 read as an answer priced its
    collectives on the wrong fabric.
    """
    if gpu is None:
        raise ValueError("gpus_per_node needs a GPU spec; none was given")
    raw = gpu.gpus_per_node
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"gpu_specs.csv: {gpu.sku!r} has no usable gpus_per_node "
            f"({raw!r}); every SKU declares its node size") from None
    if value < 1:
        raise ValueError(f"gpu_specs.csv: {gpu.sku!r} gpus_per_node={value}")
    return value


def intra_bw_label(gpu: GPUSpec) -> str:
    """Display label for intra-node BW: `"NVLink"` / `"PCIe"` / `""`."""
    if gpu is None:
        return ""
    if gpu.nvlink_gbps:
        return "NVLink"
    if gpu.pcie_gbps:
        return "PCIe"
    return ""


def inter_bw_label(gpu: GPUSpec) -> str:
    """Display label for inter-node BW: `"RDMA"` / `""`."""
    if gpu is None or inter_node_bw_gbps(gpu.rdma_gbps) is None:
        return ""
    return "RDMA"


class NoPeakForDtype(ValueError):
    """The card lists no peak throughput for the dtype a step computes in,
    so that step has no time on it. Its FLOPs and bytes do not depend on
    the card and still stand; a caller that reports them alongside time
    (the playground) shows them and says why the time is missing."""

    def __init__(self, gpu_sku: str, dtype: str, kind: str):
        self.gpu_sku, self.dtype, self.kind = gpu_sku, dtype, kind
        super().__init__(
            f"{gpu_sku} lists no {kind.upper()} peak throughput, so a step "
            f"computing in {kind.upper()} has no time on this card. Its FLOPs "
            f"and bytes do not depend on the card; to time it, pick a "
            f"precision that computes in a format {gpu_sku} lists.")


class NoPeakBandwidth(ValueError):
    """The card lists no HBM bandwidth (`mem_bw_gbps`), so a step that reads
    bytes has no time on it.

    The sibling of `NoPeakForDtype` on the memory side, and refused for the
    same reason: `t_memory = bytes / bw`, so a blank column read as "no
    limit" prices every byte at zero seconds and the card reads as
    infinitely fast — the one answer a roofline must never give. Raised
    where a row's bytes meet a card: at load for a sheet row that leaves the
    column empty (`check_memory_bandwidth`), and where a step's form is
    bound to a card (`row_roofline.CardForm.bind`)."""

    def __init__(self, gpu_sku: str):
        self.gpu_sku, self.field = gpu_sku, "mem_bw_gbps"
        super().__init__(
            f"{gpu_sku} lists no HBM bandwidth (gpu_specs.csv column "
            f"`mem_bw_gbps`), so a step that reads bytes has no time on this "
            f"card. Its FLOPs and bytes do not depend on the card; to time "
            f"it, state {gpu_sku}'s memory bandwidth on its sheet row")


def check_memory_bandwidth(spec: GPUSpec) -> None:
    """Every row states a positive `mem_bw_gbps`: every roofline divides a
    step's access bytes by it (`peak_hbm_bw`), so a row that left the column
    empty would price its memory term at zero seconds on every consumer of
    this file. Refused here, at load, where the sheet is the subject — so
    `peak_hbm_bw` returning None means exactly "no card was chosen"."""
    bw = spec.mem_bw_gbps
    if bw is None or not math.isfinite(bw) or bw <= 0:
        raise NoPeakBandwidth(f"gpu_specs.csv: {spec.sku}")


def peak_families(gpu: GPUSpec) -> Tuple[str, ...]:
    """The MMA families the card lists a peak column for (`PEAK_COLUMNS`),
    sorted — the sheet's half of a card's walk class
    (`framework_bindings.CardClass`): a leaf whose dtype has no column is
    walked weight-only and a deployment-chosen dtype without one is refused
    (`NoPeakForDtype`), so two cards listing different families walk
    different rows."""
    return tuple(sorted(kind for kind, col in PEAK_COLUMNS.items()
                        if getattr(gpu, col) is not None))
