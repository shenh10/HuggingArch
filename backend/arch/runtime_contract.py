"""The declared facts of a serving runtime's sparse-DCP contract.

A framework binding (`inference/framework_bindings/<fw>.yaml`) states these
once; the loader builds one instance and threads it into the architecture
walk and its materialization as plain data. Nothing here imports `inference`
— the dependency direction is binding -> arch, never the reverse — so the
arch/layout consumers read the declared numbers rather than restating them.

The role at a structural position (`role_ops`) is the part of the contract
that says WHAT OP the runtime requires at that position; the spec owns the
node this model's graph puts there (`SparseDcpSpec`, a graph position). Two
declarations, one fact each, kept apart on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

#: The structural positions a sparse-DCP graph names. One declaration, read
#: by the spec schema (which position a component's graph binds) and by the
#: binding (which op the runtime requires there); a new position is added
#: here, not invented by a consumer.
SPARSE_DCP_ROLES = (
    "query_projection",
    "absorbed_query",
    "stored_kv_projection",
    "value_projection",
    "score",
    "softmax",
    "apply",
    "indexer_logits",
    "indexer_topk",
)


@dataclass(frozen=True)
class SparseDcpRuntime:
    """One framework's sparse-DCP runtime contract, as plain data.

    `role_ops` maps each structural position to the op the runtime requires
    there (`score` -> `dsa_attention_score`). The constants size, phase and
    load the kernels the audited launch runs: candidate certificate width,
    its pair bytes, the direct-A2A push row capacity with its LSE slot
    widths, the pushed partial width, and the windowed scheduler's block /
    per-query overhead. `sources` maps a stage name to the permalink the
    binding's `source_anchor` resolves to at the audited revision.
    """

    role_ops: Mapping[str, str] = field(hash=False)
    candidate_count: int
    pair_bytes: int
    push_max_rows: int
    push_lse_slot_bytes: int
    overflow_lse_slot_bytes: int
    push_value_dim: int
    flashmla_block_size: int
    flashmla_overhead: int
    #: The audited reference profile's parallel group; a deployment that
    #: runs a sparse-DCP component at `dcp > 1` outside it has no binding
    #: match, which is a refusal stated at the public entry, not a layout
    #: construction exception.
    require_dcp: Optional[int] = 8
    require_tp_attention: Optional[int] = 8
    require_cp: Optional[int] = 1
    sources: Mapping[str, str] = field(default_factory=dict, hash=False)

    def require_mesh(self, parallel) -> None:
        """Check the audited striped deployment before creating layout events."""
        from backend.arch.eval.refusal import DeploymentRefusal
        if parallel.dcp <= 1:
            return
        for knob, required in (("dcp", self.require_dcp),
                               ("tp_attention", self.require_tp_attention),
                               ("cp", self.require_cp)):
            if getattr(parallel, knob) != required:
                raise DeploymentRefusal(
                    f"sparse DCP requires {knob}={required}", knobs=(knob,))

    def __post_init__(self) -> None:
        for name in SPARSE_DCP_ROLES:
            if name not in self.role_ops:
                raise ValueError(
                    f"sparse-DCP contract names no op for structural "
                    f"position {name!r}")
        for name in self.role_ops:
            if name not in SPARSE_DCP_ROLES:
                raise ValueError(
                    f"sparse-DCP contract names unregistered position "
                    f"{name!r}")
        if self.candidate_count < 1 or self.pair_bytes < 1:
            raise ValueError("sparse-DCP contract needs positive candidate "
                             "count and pair bytes")
        if self.push_max_rows < 1 or self.push_value_dim < 1:
            raise ValueError("sparse-DCP contract needs a positive push "
                             "capacity and pushed partial width")
        if self.flashmla_block_size < 1 or self.flashmla_overhead < 1:
            raise ValueError("sparse-DCP contract needs positive FlashMLA "
                             "block size and overhead")