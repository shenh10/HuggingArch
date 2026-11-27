"""Invocation — one kernel measurement request.

Parsed from a plan job (which itself derives from the analyze pipeline /
suggest-shapes). Carries everything a provider needs to build inputs at
the exact per-rank shape a real serving run hits, plus the identity +
roofline numerators (flops / access_bytes) needed to emit a CSV row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from backend.arch.eval.routed import RoutedLaunch
from backend.inference.op_signatures import PAGE_AXIS

from .signatures import op_to_signature


@dataclass(frozen=True)
class Invocation:
    # --- what to measure ---
    signature: str                 # op-signature name, e.g. "gqa_attention"
    op: str                        # our fused op, e.g. "flash_attention"
    # The IDENTITY: the shapes projected onto scalars. Lossy by design — a
    # corpus key must be scalars so that two models landing on one kernel
    # geometry share one row.
    dims: Dict[str, int]           # {B,S,N,Nk,D,H,I,M,K,Nout,...}
    dtype_env: Dict[str, str]      # {weight,activation,kv,output,metadata}
    phase: str                     # prefill | decode
    tp: int                        # tensor-parallel degree (per-rank shapes)

    # The engine's own per-rank operand shapes, carried whole. Construction
    # reads these (a permutation of the signature contract) rather than
    # rebuilding from `dims`: `dims` is the lossy corpus key, and the two
    # layouts disagree on axis order (engine Q is [B,N,Sq,D]; canonical is
    # [B,Sq,N,D]). Older jobs without them still build from the scalars.
    #
    # `shape_in` is the ACTIVATION operands only; a stored weight's shape is
    # `OpSpec.weight_shape`, and the output width comes from `shape_out`.
    # All three are the graph's, which is the point.
    #
    # Optional because an older jobs.jsonl has neither. A job without them can
    # still be measured the old way; it just cannot be built from its shapes.
    shape_in: Optional[Any] = None
    shape_out: Optional[Any] = None
    # The PARAM's shape. `shape_in` is activations only, so a builder without
    # this rebuilds the weight from the identity scalars — the same duplication
    # the two fields above exist to remove, one operand class over. Per-rank:
    # under TP a kernel allocates its shard, not the whole tensor.
    shape_weight: Optional[Any] = None
    # The ports of its plan each `shape_in` operand of a fused launch fills
    # (the row's `operand_ports`, `FusedNode.operand_ports`). A signature that
    # names a multi-port launch's operands by port reads them by these
    # (`OpSignature.at_ports`); a job without them — planned before
    # launches named their operands — is refused for such a signature and
    # re-planned, since where its operands stand is no contract.
    operand_ports: Optional[Any] = None

    # The measurement key this cell is, as the one key derivation stated it
    # (`calibration_overlay.point_keys` → `PointRow.key`, carried by the plan
    # job): signature, projected shape, dtype environment, phase. The card is
    # the run's, not the plan's, so it is stamped at run time
    # (`driver.physical_key_for`). A job without it cannot be keyed here —
    # assembling one from `dims` and `dtype_env` is the second derivation this
    # field exists to remove — so the driver refuses it and asks for a re-plan.
    #: How far this launch's collective group physically reached, when it
    #: reached past one chassis (`comm_cost.comm_reach`). Not a dim: a dim
    #: is an extent and this is a placement, which is why it rides beside
    #: them rather than in them. `emit.build_row` joins it to the cell the
    #: row states, so a crossing measurement reads as one when the corpus
    #: is read back. A TIER code (`comm_cost.COMM_REACHES`), not a rank
    #: count: `scaleout` has no fixed extent to count. 0 is intra-host,
    #: the default every row carried before the axis existed.
    comm_domain: int = 0
    physical_key: Optional[Tuple[str, str, str, str]] = None

    # --- roofline numerators (for achieved_* on emit) ---
    flops_per_rank: Optional[float] = None
    access_bytes_per_rank: Optional[float] = None

    # --- kernel parameters ---
    # Configuration that changes what the kernel DOES, as opposed to `dims`,
    # which is the shape the engine derived. It travels with the invocation
    # because a provider that picks it instead measures a configuration nobody
    # asked for, and the row cannot say which one it was. Two sources, both
    # outside the shape:
    #   * the MODEL's config — e.g. `n_expert_groups`, the router's expert
    #     grouping, which picks WHICH experts win and not how much work it is.
    #   * the model's graph — the constants an absorbed op binds (a clamp
    #     bound, SiTU's betas; `OpSignature.config_params`).
    # A value that changes how much WORK the kernel does is not one of these:
    # it is an axis of the cell and travels in `dims`, even when the
    # deployment rather than the model chooses it — the Mamba-2 scan chunk
    # (`ssm_chunk_size`, declared by that signature, because the launch runs
    # `ceil(s_query / chunk)` tiles of it; its value is
    # `Program.scan_chunk_size`) and the paged-KV page
    # (`kv_block_size`, an axis of every signature that reads a paged cache;
    # its value is `FrameworkBinding.kv_block_size` or the request's).
    # A parameter that is ABSENT is not a parameter with a default: the provider
    # that needs it declines and says so.
    kernel_params: Dict[str, Any] = field(default_factory=dict)
    # --- the routed boundary ---
    # The MoE dispatch boundary a routed cell's launch is read off, as the
    # walk published it (`eval.routed.RoutedLaunch.wire`, the row's
    # `routed`): None for a cell on no routed buffer, and for a cell no walk
    # produced (a smoke or hand-built cell, whose dims are its whole
    # statement).
    routed: Optional[Dict[str, Any]] = None
    # The leaf carries a recurrent state (the job's `carries_state`): its
    # numerators bill the state's read and write-back, so only a kernel that
    # takes the state is this cell (`OpSignature.carries_state`).
    carries_state: bool = False
    # This process's rank in the measurement group. 0 is a single process.
    # A collective oracle reads this so each rank's payload is a different,
    # reconstructable draw; `refs` must not mix `rank` into the seed
    # privately, because then the same cell on two ranks would disagree
    # about which numbers the other ranks held.
    rank: int = 0

    # --- provenance carried through to the CSV identity ---
    model: str = ""
    spec_hash: str = ""
    node_spec_id: str = ""
    parallel: str = "tp=1"
    count_per_step: int = 1
    moe_block_member: bool = False

    @property
    def input_seed(self) -> int:
        """The seed every input tensor of this cell is drawn at.

        Derived from the four fields of the measurement key
        (`physical_key`: signature, shape_json, dtype_env_json, phase), in
        the bytes the plan already stated them in — not a re-serialization,
        so the seed moves exactly when the cell does. `\x1f` (unit
        separator) joins them: it occurs in neither a signature name nor
        canonical JSON, so the concatenation is unambiguous. The digest is
        8 bytes, which is the range `torch.Generator.manual_seed` takes.

        The card is NOT in it: the same cell on an H100 and a B200 draws at
        the same seed (the card is on the row). Neither is the provider: the
        kernels of one cell must be timed on one batch of inputs or their
        latencies do not compare, which is why the driver builds the inputs
        once and shares them.

        Two cells whose keys differ anywhere — one axis, one dtype slot, the
        phase, the signature — get different seeds, so a shape's data is not
        a prefix of a bigger shape's.
        """
        from hashlib import blake2b
        if self.physical_key is None:
            raise ValueError(
                f"{self.op} [{self.node_spec_id}]: this job carries no "
                f"measurement key, so its inputs have no seed; re-run "
                f"`bench plan` to state the key")
        payload = b"\x1f".join(f.encode("utf-8") for f in self.physical_key)
        return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")

    def seed_for_rank(self, rank: int) -> int:
        """The draw this cell uses on `rank`.

        Mixes the physical key with the rank so two ranks of one all-reduce
        do not hold the same buffer (a silent identity reduction) while
        remaining reconstructable: the oracle on any rank calls this for
        every rank and reduces what it would have drawn. GEMM cells keep
        `input_seed`; only a collective fill reads this.
        """
        from hashlib import blake2b
        if self.physical_key is None:
            raise ValueError(
                f"{self.op} [{self.node_spec_id}]: this job carries no "
                f"measurement key, so its inputs have no seed; re-run "
                f"`bench plan` to state the key")
        payload = b"\x1f".join(f.encode("utf-8") for f in self.physical_key)
        payload += b"\x1frank=" + str(int(rank)).encode("utf-8")
        return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")

    @property
    def rank_seed(self) -> int:
        return self.seed_for_rank(self.rank)

    def at_rank(self, rank: int) -> "Invocation":
        """The same cell as seen from `rank`. Frozen, so a copy."""
        return replace(self, rank=int(rank))

    @property
    def weight_granularity(self) -> Optional[str]:
        """Which weight elements share a scale in the checkpoint this cell
        was planned from (`TensorFormat.weight_granularity`); None for a
        weight that stores no scales."""
        return self.dtype_env.get("weight_granularity")

    @property
    def activation_granularity(self) -> Optional[str]:
        """Which activation elements share one dynamic scale in the format
        this cell runs under (`ActivationFormat.granularity`): `per-token`,
        or `per-group:<g>`. None for a cell whose activation carries no
        scales — every cell but a quantised-activation cast, today.

        Read rather than derived, for the reason the weight's is: per-token,
        per-128-group and per-32-group are three different kernels between
        the same two dtypes, so it is part of what the row measured
        (`ActivationCast.of`)."""
        return self.dtype_env.get("activation_granularity")

    @property
    def output_format(self) -> Optional[str]:
        """The cache format the cell's primary output is stored in, by
        registry name (`LeafAccess.output_format`): the cached record a
        launch writes, which the `output` dtype alone cannot state. None
        for an activation and for a cache stored plain."""
        return self.dtype_env.get("output_format")

    @property
    def kv_format(self) -> Optional[str]:
        """The cache format of the cache the cell's `kv` slot names, by
        registry name (`CachedTensor.format_name`): the record a launch
        reads, or writes beside its output. None for a cell without a cache
        and for a cache stored plain."""
        return self.dtype_env.get("kv_format")

    @property
    def kv_block_size(self) -> Optional[int]:
        """Paged-KV page the deployment asks for, or None when the cell has
        none.

        The cell's own axis (`op_signatures.PAGE_AXIS`, in `dims`), so the
        page a provider builds the cache at is the page the row is keyed
        and written under. Absent is absent: this used to answer 16 (vLLM's
        `CacheConfig.DEFAULT_BLOCK_SIZE`) for a job that named no page, so
        a job planned under a framework that declares none — sglang
        resolves its page per attention backend — would have been measured
        at a page nobody chose, and the row could not say which. The
        provider declines instead
        (`providers.attention.resolve_kv_block_size`).
        """
        page = self.dims.get(PAGE_AXIS)
        return None if page is None else int(page)

    def dtype_for(self, role: str) -> str:
        """The dtype an operand of this contract ROLE is in.

        `TensorLayout.role` names the `dtype_env` slot a tensor is in
        (activation / weight / kv / metadata), and the cell carries one
        dtype per slot — the KV cache's own format is a declared fact, not
        the activation's. Nothing read the `kv` slot until Gemma-2, whose
        cache ships in fp32 while the harness allocated bf16: the row was
        billed four bytes an element and the provider moved two, so every
        one of its attention rows reported twice the bandwidth it moved.
        The reverse (an fp8 cache allocated as bf16) is the same error in
        the direction that never crosses a ceiling and so never shows.

        A slot the cell does not carry falls back to the activation: the row
        has no such tensor of its own (a kernel that reads no cache, a
        weight without scales), so an operand the provider still builds for
        that role is carried at the dtype the kernel runs in.
        """
        return self.dtype_env.get(role) or self.dtype

    @property
    def kv_dtype(self) -> str:
        """The dtype the KV cache is kept in — `dtype_for("kv")`."""
        return self.dtype_for("kv")

    @property
    def dtype(self) -> str:
        """Primary compute dtype (activation slot) — the dtype the kernel
        runs in. Providers use this to allocate inputs. A job without the
        slot is a malformed job, not a bf16 one."""
        try:
            return self.dtype_env["activation"]
        except KeyError:
            raise ValueError(
                f"{self.op} job for {self.model} has no activation dtype in "
                f"dtype_env {sorted(self.dtype_env)}; the plan writes one per "
                "row") from None

    @property
    def moe_token_counts(self) -> tuple[int, ...]:
        """Rows per expert of a grouped MoE invocation.

        `balanced` is every expert of the cell at the cell's own per-expert
        width. A routed cell's experts are the ones its launch touches — the
        experts that receive rows, `min(ceil(R), E_local)` of the rank's —
        each padded to `ceil(R / touched)` rows (`RoutedLaunch`, which the
        cell's `n_experts` / `s_expert_rows` / `n_tokens` are read off), so
        the launch reads exactly the weights the walk bills and an expert
        that receives nothing is not in it, as a grouped kernel launches no
        blocks for one. The width divides the total and there is no
        remainder to hand out. A total that does not divide is a cell from
        some other convention, and it raises rather than being spread into a
        distribution nothing declared.
        """
        experts = int(self.dims.get("n_experts", 0) or 0)
        maximum = int(self.dims.get("s_expert_rows", 0) or 0)
        expected = self.dims.get("n_tokens")
        raw = self.kernel_params.get("moe_token_counts")
        distribution = self.kernel_params.get("moe_token_distribution")
        if raw is not None:
            counts = tuple(int(value) for value in raw)
        elif distribution == "balanced" and self.routed is not None:
            # The walk's launch, expert by expert (`RoutedLaunch.
            # expert_rows`); the cell's dims are its projection, so they
            # must agree, which the checks below state.
            counts = tuple(int(n) for n in
                           RoutedLaunch.from_wire(self.routed).expert_rows)
        elif distribution == "balanced":
            if experts <= 0 or maximum <= 0 or expected is None:
                raise ValueError(
                    "balanced MoE token distribution requires positive "
                    "n_experts, s_expert_rows and n_tokens")
            if int(expected) != experts * maximum:
                raise ValueError(
                    f"balanced MoE token distribution over {experts} experts "
                    f"of {maximum} rows each is {experts * maximum} rows, "
                    f"but the cell states n_tokens={expected}")
            counts = (maximum,) * experts
        elif distribution is not None:
            raise ValueError(
                f"unknown moe_token_distribution {distribution!r}")
        else:
            counts = (maximum,) * experts
        if len(counts) != experts or any(
                value < 0 or value > maximum for value in counts):
            raise ValueError(
                f"invalid moe_token_counts for E={experts}, "
                f"max_m={maximum}: {counts}")
        if expected is not None and sum(counts) != int(expected):
            raise ValueError(
                f"moe_token_counts sum to {sum(counts)}, "
                f"expected n_tokens={expected}")
        return counts

    @property
    def token_rows(self) -> int:
        """The token side a dispatch boundary moves between (the rows a
        `combine` writes back): the walk's `RoutedLaunch.source_rows` where
        the job carries its boundary, which the cell's own token axes
        (`B` x `s_query`) are the projection of — refused where they
        disagree — and those axes for a cell no walk produced."""
        cell = int(self.dims.get("B", 1)) * int(self.dims.get("s_query", 1))
        if self.routed is None:
            return cell
        source = RoutedLaunch.from_wire(self.routed).source_rows
        if source != cell:
            raise ValueError(
                f"{self.op} job for {self.model}: the routed boundary holds "
                f"{source} tokens and the cell states B x s_query = {cell}")
        return int(source)

    @classmethod
    def from_job(cls, job: Dict[str, Any]) -> "Invocation":
        """Build from a plan job dict (one line of jobs.jsonl)."""
        dims = job["dims"]
        if isinstance(dims, str):
            dims = json.loads(dims)
        dtype_env = job["dtype_env"]
        if isinstance(dtype_env, str):
            dtype_env = json.loads(dtype_env)
        op = job["op"]
        sig = job.get("signature") or op_to_signature(op)
        if sig is None:
            raise ValueError(
                f"op {op!r} has no op-signature mapping — not measurable "
                f"(map it in backend.inference.benchmark_identity."
                f"the signature catalog's binds_ops, or drop the job)."
            )
        return cls(
            signature=sig,
            op=op,
            dims={k: int(v) for k, v in dims.items()},
            dtype_env=dtype_env,
            phase=job.get("phase", "prefill"),
            tp=int(job.get("tp", 1)),
            shape_in=job.get("shape_in"),
            operand_ports=job.get("operand_ports"),
            shape_out=job.get("shape_out"),
            shape_weight=job.get("shape_weight"),
            flops_per_rank=job.get("flops_per_rank"),
            access_bytes_per_rank=job.get("access_bytes_per_rank"),
            model=job.get("model", ""),
            spec_hash=job.get("spec_hash", ""),
            node_spec_id=job.get("node_spec_id", ""),
            parallel=job.get("parallel", f"tp={job.get('tp', 1)}"),
            count_per_step=int(job.get("count_per_step", 1)),
            moe_block_member=bool(job.get("moe_block_member", False)),
            comm_domain=int(job.get("comm_domain") or 0),
            physical_key=(tuple(job["physical_key"])
                          if job.get("physical_key") else None),
            kernel_params=dict(job.get("kernel_params") or {}),
            routed=job.get("routed"),
            carries_state=bool(job.get("carries_state", False)),
            rank=int(job.get("rank", 0) or 0),
        )
