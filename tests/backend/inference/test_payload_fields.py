"""The estimate's published fields are pinned by path, not only by value.

The number goldens (`test_capacity_golden`, `test_sol_surface_golden`) catch
a value that moves. They do not catch a field that appears: a name the walk
starts binding reaches every per-op row's `node_dims` and the public payload
grows on every model while every pinned number stays put. This gate pins the
SET of leaf paths an estimate publishes (list indices collapsed to `[*]`),
per model, against `tests/golden/payload_fields.json`.

A path that disappears fails unless it is registered in `REMOVED`, and a
path that appears fails unless it is registered in `ADDED`, each with the
change that made it and why; a path registered as removed that the
estimate still publishes fails too, and so does a path registered as added
that no model the gate walks publishes. `ADDED` holds every model's
additions, so a model publishes the ones its attention has.

Regenerate after an intentional change to the payload's shape:
    REGEN_GOLDEN=1 .venv/bin/python -m pytest tests/backend/inference/test_payload_fields.py
and register what is new in `ADDED` instead, where it is a field to keep.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Set

import pytest

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, snapshot_config

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "payload_fields.json"

#: Paths the payload gained on purpose: `{path: why}`.
ADDED = {
    "gpu.sm_count": "observed scheduler hardware geometry, unknown when the sheet has no measurement",
    "parallel.dcp": "the explicit decode-context parallel degree of the reported deployment",
    "per_op_timing[*].point_law.point.dcp_failed_owners": "the declared candidate-certificate fallback scenario used to reread this law",
    "per_op_timing[*].point_law.point.rho_dcp": "the declared selected-position ownership scenario used to reread this law",
    "per_op_timing_decode[*].point_law.point.dcp_failed_owners": "the decode law's candidate-certificate fallback scenario",
    "per_op_timing_decode[*].point_law.point.rho_dcp": "the decode law's selected-position ownership scenario",
    "per_op_timing[*].roofline_expr.comm_dependency_rounds": "serial transfers inside one runtime kernel; independent of kernel launch count",
    "per_op_timing_decode[*].roofline_expr.comm_dependency_rounds": "decode serial transfers inside one runtime kernel",
    "phases.decode.flops_by_module[*].flops_law.point.dcp_failed_owners": "complete workload point accompanying a module arithmetic law",
    "phases.decode.flops_by_module[*].flops_law.point.rho_dcp": "selected-position workload point accompanying a module arithmetic law",
    "phases.decode.flops_by_module[*].shape_law.point.dcp_failed_owners": "complete workload point accompanying a module shape law",
    "phases.decode.flops_by_module[*].shape_law.point.rho_dcp": "selected-position workload point accompanying a module shape law",
    "phases.prefill.flops_by_module[*].flops_law.point.dcp_failed_owners": "shared workload point, retaining fallback scenario identity across phases",
    "phases.prefill.flops_by_module[*].flops_law.point.rho_dcp": "shared workload point, retaining selected-position scenario identity across phases",
    "phases.prefill.flops_by_module[*].shape_law.point.dcp_failed_owners": "shared workload point accompanying the prefill shape law",
    "phases.prefill.flops_by_module[*].shape_law.point.rho_dcp": "shared workload point accompanying the prefill shape law",
    "phases.decode.flops_by_module[*].roofline_expr.comm_dependency_rounds": "module row's serial-transfer law, without extra kernel launches",
    "phases.prefill.flops_by_module[*].roofline_expr.comm_dependency_rounds": "module row's serial-transfer law, defaulting to one for ordinary collectives",
    "phases.decode.throughput.concurrent_requests":
        "the requests the cluster decodes at once (B per group × groups), "
        "the dimension a decode Pareto frontier is held at "
        "(`parallel_sweep.frontier_view`)",
    "per_op_timing[*].node_dims.s_pairs":
        "9475b2abb: the phase declares `s_pairs`, the step's live (query, "
        "key) pairs, which the attention cost is written in; a row's frame "
        "shows it as it shows `s_past`",
    "per_op_timing_decode[*].node_dims.s_pairs":
        "9475b2abb: as for prefill",
    "per_op_timing[*].point_law.dims.s_pairs":
        "9475b2abb: the row's law of the same frame name",
    "per_op_timing_decode[*].point_law.dims.s_pairs":
        "9475b2abb: as for prefill",
    "per_op_timing[*].point_law.in_terms[*].index":
        "which input a read term's fan-out belongs to, by its index in "
        "`shape_in`: the measurement key reads a head-broadcast K/V as "
        "stored (`benchmark_identity.stored_operands`), and terms skip "
        "identity-source edges, so their order is not the inputs'",
    "per_op_timing_decode[*].point_law.in_terms[*].index":
        "as for prefill",
    "gpu.nvls":
        "the GPU sheet states whether a SKU's NVLink domain reduces in the "
        "switch (NVLink SHARP), which decides whether a framework's NVLS "
        "all-reduce runs on it (`comm_cost.collective_ladder`); the SKU "
        "block publishes every sheet column",
    "gpu.compute_capability":
        "the GPU sheet states each SKU's CUDA compute capability, which a "
        "framework binding's `when.min_compute_capability` asks (a kernel "
        "the framework enables only from a capability on); the SKU block "
        "publishes every sheet column",
    "per_op_timing[*].t_launch_s":
        "the launch floor a row's kernel takes however little it does (`row_roofline`, `t_launch`), the fourth term of its roofline",
    "per_op_timing[*].roofline_expr.t_launch":
        "the row's law of `t_launch_s` (`n x LAUNCH_S`)",
    "per_op_timing_decode[*].t_launch_s":
        "the launch floor a row's kernel takes however little it does (`row_roofline`, `t_launch`), the fourth term of its roofline",
    "per_op_timing_decode[*].roofline_expr.t_launch":
        "the row's law of `t_launch_s` (`n x LAUNCH_S`)",
    "phases.prefill.flops_by_module[*].t_launch_s":
        "the launch floor a row's kernel takes however little it does (`row_roofline`, `t_launch`), the fourth term of its roofline",
    "phases.prefill.flops_by_module[*].roofline_expr.t_launch":
        "the row's law of `t_launch_s` (`n x LAUNCH_S`)",
    "phases.decode.flops_by_module[*].t_launch_s":
        "the launch floor a row's kernel takes however little it does (`row_roofline`, `t_launch`), the fourth term of its roofline",
    "phases.decode.flops_by_module[*].roofline_expr.t_launch":
        "the row's law of `t_launch_s` (`n x LAUNCH_S`)",
    "phases.prefill.t_sol_by_bound.launch":
        "the step's time under the launch floor, the fourth bound",
    "phases.prefill.t_sol_by_bound_share.launch":
        "its share of t_sol",
    "phases.decode.t_sol_by_bound.launch":
        "the step's time under the launch floor, the fourth bound",
    "phases.decode.t_sol_by_bound_share.launch":
        "its share of t_sol",
    "formulas.per_request_kv.subterms[*].note":
        "1b563d9fe: a per-request KV class term names the cache format(s) "
        "its bytes are stored in (`CacheView.request_formula`); the formula "
        "popover prints a subterm's note as its footnote",
    "per_op_timing[*].point_law.in_terms[*].headless_store":
        "aa3834b08: whether a head broadcast expanded a store with no head "
        "axis (`traffic._Fanout.headless_store`); the measurement key reads "
        "it off the row (`benchmark_identity._headless_stores`), since the "
        "fan-out ratio alone cannot tell one latent from one head",
    "per_op_timing_decode[*].point_law.in_terms[*].headless_store":
        "aa3834b08: as for prefill",
}

# `s_pairs` writes the pair-priced FLOPs (`binding.STEP_LAWS`, the node
# rebindings): the phase binding publishes it in place of `causal_factor`,
# a row's FLOPs record binds it, and a row's declared definitions and its
# where column's definition line are new.
_PAIRS = "the attention cost is written in `s_pairs`; `causal_factor` is gone"
_DEFINED = ("a row carries its declared definitions (`point_law.definitions`) "
            "and its where column prints them (`defs[*].definition`)")
for _phase in ("per_op_timing", "per_op_timing_decode"):
    ADDED[f"{_phase}[*].point_law.flops.record.s_pairs"] = _PAIRS
    ADDED[f"{_phase}[*].flops_formula.defs[*].definition"] = _DEFINED
    for _name, _reads in (("s_pairs", ("s_past", "s_query", "s_kv", "s_window",
                                      "compress_ratio", "n_topk_keys",
                                      "n_verify")),
                          ("s_past", ("s_kv", "s_query"))):
        ADDED[f"{_phase}[*].point_law.definitions.{_name}.law"] = _DEFINED
        for _read in _reads:
            if _read == "n_verify" and _phase == "per_op_timing":
                continue    # the query axis is `n_verify` at decode only
            ADDED[f"{_phase}[*].point_law.definitions.{_name}.bindings."
                  f"{_read}"] = _DEFINED
for _phase in ("per_op_timing", "per_op_timing_decode"):
    for _field in ("output_dtype", "output_cache", "output_format"):
        ADDED[f"{_phase}[*].{_field}"] = (
            "the dtype a row writes its primary output at, as the walk "
            "priced it, the cached tensor that output is, and that "
            "tensor's format by registry name: the measured cell's "
            "`output` and `output_format` slots on a cached tensor's writer")
for _phase in ("prefill", "decode"):
    for _field in ("output_dtype", "output_cache", "output_format"):
        ADDED[f"phases.{_phase}.flops_by_module[*].{_field}"] = (
            "4fe1fa9e4/b89c49b0e: a module row carries the same primary "
            "output dtype, cached tensor, and cache format as its measured "
            "leaf, so a re-read can key the row by the output it walks")
for _phase in ("prefill", "decode"):
    ADDED[f"phases.{_phase}.binding.s_pairs"] = _PAIRS
for _phase in ("prefill", "decode"):
    for _field in ("t_partial_s", "partial_measured_share", "throughput.partial"):
        ADDED[f"phases.{_phase}.{_field}"] = (
            "08b29819b: the partial-measured blend (`Folded.partial`), each "
            "measured op at its Kernel Bench time and the rest at SOL, with "
            "the share of the step's time it measured, shown behind the "
            "Partial measured checkbox")
    ADDED[f"phases.{_phase}.flops_by_module[*].flops_formula.defs[*]"
          f".definition"] = _DEFINED

#: Paths the payload lost on purpose: `{path: why}`.
# The matrices a row's kernel multiplies, partition by partition, as the grid
# a scale tiles unsharded and on one rank, with the weight each is packed
# into and its own scale layout (`Analysis.weight_shards`): what a
# block-scaled weight's shard rule reads (`kernel_bench.plan.
# _scale_block_refusal`) — a heads cut, a packed `gate|up`'s partitions and a
# borrowed matrix included, which no activation width says.
_SHARDS = ("each matrix the kernel multiplies, unsharded and on one rank, so "
           "the block-scale shard rule reads which axes a world cut")
for _phase in ("per_op_timing", "per_op_timing_decode"):
    # Leaf paths only: the walker publishes a list's items, never the list.
    for _path in ("weight_shards[*].pack[*]", "weight_shards[*].rows",
                  "weight_shards[*].rows_local", "weight_shards[*].width",
                  "weight_shards[*].width_local",
                  "weight_shards[*].granularity"):
        ADDED[f"{_phase}[*].{_path}"] = _SHARDS

# A row's weight axes by identity, beside `shape_weight`'s extents — the
# `shape_in_dims` of the weight: which of them is an expert axis is what a
# launch over the experts a step's rows reach rewrites
# (`benchmark_identity._launch_weight`), asked by identity, not by where the
# matrix sits.
_WEIGHT_DIMS = ("the weight's axes by identity, so a launch's expert bank is "
                "found by `is_expert_axis` and not by position")
for _phase in ("per_op_timing", "per_op_timing_decode"):
    for _path in ("shape_weight_dims", "shape_weight_dims[*].expr",
                  "shape_weight_dims[*].value"):
        ADDED[f"{_phase}[*].{_path}"] = _WEIGHT_DIMS

REMOVED = {}
for _phase in ("per_op_timing", "per_op_timing_decode"):
    for _where in ("node_dims", "point_law.dims", "point_law.flops.record"):
        REMOVED[f"{_phase}[*].{_where}.causal_factor"] = _PAIRS
    REMOVED[f"{_phase}[*].point_law.flops.record.s_window"] = (
        "a sliding band's FLOPs read its pairs (`s_pairs`), not `s_window`")
for _phase in ("prefill", "decode"):
    REMOVED[f"phases.{_phase}.binding.causal_factor"] = _PAIRS

# The batch's routing skew is a workload input the walk reads as `rho`
# (`Workload.ep_load_factor`): a point names its value, and a routed row's
# per-rank shapes are the busiest rank's, so no row publishes a load factor
# to be multiplied in beside them.
_RHO = ("routing skew is a workload input (`rho`) the walk reads into the "
        "busiest rank's shapes")
for _phase in ("per_op_timing", "per_op_timing_decode"):
    ADDED[f"{_phase}[*].point_law.point.rho"] = _RHO
    REMOVED[f"{_phase}[*].rank_load_factor"] = _RHO
for _phase in ("prefill", "decode"):
    for _law in ("flops_law", "shape_law"):
        ADDED[f"phases.{_phase}.flops_by_module[*].{_law}.point.rho"] = _RHO

# The activation-write term is now a term and not just a width: a fused
# boundary may publish a tensor its launch only rewrote a region of, and the
# popover must print the region or it states a charge the row does not make
# (`access.WriteTerm`, `phase_formulas.act_out_from_terms`). So the width
# list becomes a `[width, elements-or-null]` pair per written tensor, beside
# the read side's terms, and the row carries each region as a law for a
# reading at another point (`batch_sweep._access_at`).
_WRITTEN = ("a write term is its width AND the elements written, which a "
            "narrowed fused boundary needs (`traffic._PortWrite`)")
for _phase in ("per_op_timing", "per_op_timing_decode"):
    REMOVED[f"{_phase}[*].point_law.access.out_widths[*]"] = _WRITTEN
    ADDED[f"{_phase}[*].point_law.access.out_terms[*][*]"] = _WRITTEN
    for _field in ("elements", "spanned"):
        ADDED[f"{_phase}[*].point_law.out_terms[*].{_field}"] = _WRITTEN

# A leaf that writes a quantised activation (the grafted cast) states the
# format it writes on its row, by name (`LeafAccess.output_activation_format`), so the measured row it is keyed by names what the launch
# writes and which elements share a scale — the cast cells were keyed
# `output: bf16` and no provider served one.
_WRITES = ("a leaf writing a quantised activation states its format, the "
           "rest of its dtype environment (`LeafAccess.output_activation_format`)")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    ADDED[f"{_rows}[*].output_activation_format"] = _WRITES

# A binding's audited source anchor is now split into text and the places it
# cites, each with a GitHub permalink at the audited commit (`AuditedAt.cite`).
_ANCHOR = ("e5516b4f7: a source badge links every cited place to the audited "
           "commit instead of publishing one unlinked anchor string")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module",
              "phases.prefill.flops_by_module"):
    REMOVED[f"{_rows}[*].source_badge.source_anchor"] = _ANCHOR
    ADDED[f"{_rows}[*].source_badge.source_anchor.text"] = _ANCHOR
    ADDED[f"{_rows}[*].source_badge.source_anchor.segments[*].text"] = _ANCHOR
    ADDED[f"{_rows}[*].source_badge.source_anchor.segments[*].url"] = _ANCHOR

# A multi-output row (a fused gate_up) states its exact output rows and their
# axes per port, aligned with `shape_outs`, where it used to state None — so a
# routed port's mean rows (1/2 per expert) are not floored to 0 in a job's
# shape (ae6dca52d). A single-output row is unchanged.
_PORTS = ("ae6dca52d: a multi-output row states `shape_out_exact` / "
          "`shape_out_dims` per port, in `shape_outs` order, instead of None")
for _rows in ("per_op_timing", "per_op_timing_decode"):
    REMOVED[f"{_rows}[*].shape_out_dims"] = _PORTS
    ADDED[f"{_rows}[*].shape_out_dims[*][*].expr"] = _PORTS
    ADDED[f"{_rows}[*].shape_out_dims[*][*].value"] = _PORTS
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    REMOVED[f"{_rows}[*].shape_out_exact"] = _PORTS
    ADDED[f"{_rows}[*].shape_out_exact[*][*]"] = _PORTS

# A leaf on a dispatch's routed buffer carries its dispatch's routed launch
# (`RoutedLaunch.wire`, `phases._routed_columns`) and the laws of its rows in
# `point_law`, re-read at another batch (`batch_sweep`): the one record the
# launch geometry, the overlay numerator and the plan's refusals read, in
# place of each consumer re-identifying the routed axis (589b295ab). None on
# a row off any routed buffer.
_ROUTED = ("589b295ab: a routed leaf carries its dispatch's `RoutedLaunch` "
           "and the laws of its rows")
_ROUTED_FIELDS = ("dispatch", "local_experts", "source_rows", "routed_rows",
                  "inputs[*]", "output", "bytes_per_row", "weight_rows_per_row")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    ADDED[f"{_rows}[*].routed"] = _ROUTED
    for _field in _ROUTED_FIELDS:
        ADDED[f"{_rows}[*].routed.{_field}"] = _ROUTED
for _rows in ("per_op_timing", "per_op_timing_decode"):
    ADDED[f"{_rows}[*].point_law.routed"] = _ROUTED
    for _field in ("source_rows", "routed_rows"):
        ADDED[f"{_rows}[*].point_law.routed.{_field}"] = _ROUTED

# A row carries the ordered framework selector's outcome beside its launch:
# the singular gap is what a report prints and the list is every candidate
# skipped, so an unmeasurable cell cannot be queued as if operand semantics
# alone had named a kernel (b01bbee14).
_LAUNCH_GAPS = ("b01bbee14: every launch candidate the selector skipped, "
                "as the row's closed coverage outcome")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    ADDED[f"{_rows}[*].launch_gap"] = _LAUNCH_GAPS
    ADDED[f"{_rows}[*].launch_gaps[*]"] = _LAUNCH_GAPS

# The selected launch's input format is part of its identity when it reads a
# quantised operand: group-32 and group-128 are different launches at one
# dtype, and the row carries the scale dtype with the granularity
# (f34d4dd02).
_INPUT_FORMAT = ("f34d4dd02: the quantised input format a selected launch "
                 "reads, by granularity and scale dtype")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    ADDED[f"{_rows}[*].activation_granularity"] = _INPUT_FORMAT
    ADDED[f"{_rows}[*].activation_scale_dtype"] = _INPUT_FORMAT

# A served response no longer carries expressions inside a fused kernel's
# absorbed parts. `shape_out` remains the point shape; only the in-process
# law is removed (28a7930fa).
_NO_SERVED_LAWS = ("28a7930fa: a response serves a fused kernel's absorbed "
                   "output shape, not the in-process law it was read from")
for _rows in ("per_op_timing", "per_op_timing_decode",
              "phases.decode.flops_by_module", "phases.prefill.flops_by_module"):
    REMOVED[f"{_rows}[*].source_badge.absorbed[*].shape_out_law"] = (
        _NO_SERVED_LAWS)

CASES = [
    pytest.param("meta-llama/Llama-3.2-3B", id="llama"),
    pytest.param("deepseek-ai/DeepSeek-V4-Flash", id="dsv4-flash",
                 marks=pytest.mark.regression),
]


def _paths(value: Any, at: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _paths(inner, f"{at}.{key}" if at else str(key))
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _paths(inner, f"{at}[*]")
    else:
        yield at


def _published(model_id: str) -> Set[str]:
    estimate = estimate_capacity(
        config=snapshot_config(model_id), model_id=model_id,
        resolved=resolve_snapshot(model_id), gpu_sku="B200",
        parallel=ParallelConfig(), num_gpus=1, gpu_memory_fraction=0.9,
        avg_input_tokens=4096, avg_output_tokens=256,
        prefill_batch_size=2, decode_batch_size=2,
        selection_mode="theoretical_only", measurement_scope=None)
    return set(_paths(estimate))


@pytest.fixture(scope="module")
def published():
    """Each model's published paths, walked once for both tests."""
    cache = {}

    def of(model_id):
        if model_id not in cache:
            cache[model_id] = _published(model_id)
        return cache[model_id]
    return of


@pytest.mark.regression  # walks every model `CASES` names
def test_every_registered_addition_is_published(published):
    """A path registered as added is published by at least one model the
    gate walks: a registration nothing publishes would let the field it
    vouches for disappear unseen."""
    seen = set().union(*(published(case.values[0]) for case in CASES))
    assert sorted(path for path in ADDED if path not in seen) == [], (
        "registered in ADDED and published by no model this gate walks: "
        "drop it from ADDED, or walk a model that publishes it")


@pytest.mark.parametrize("model_id", CASES)
def test_the_published_fields_are_the_pinned_ones(model_id, published):
    now = published(model_id)
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("REGEN_GOLDEN"):
        golden[model_id] = sorted((now - set(ADDED)) | (
            set(golden.get(model_id, ())) & set(REMOVED)))
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    pinned = set(golden[model_id])
    assert sorted(pinned - now - set(REMOVED)) == [], (
        "published fields disappeared: register each in REMOVED with why")
    assert sorted(now - pinned - set(ADDED)) == [], (
        "new published fields: register each in ADDED with why it belongs "
        "in the payload, or keep it out of the payload")
    assert sorted(path for path in REMOVED if path in now) == [], (
        "a field registered as removed is published; drop it from REMOVED")
