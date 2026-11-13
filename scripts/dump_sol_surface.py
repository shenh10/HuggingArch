#!/usr/bin/env python3
"""Dump the public SOL numeric surface for every architecture snapshot.

Run this file from the repository whose corpus and implementation should be
measured.  ``--reference-snapshot-dir`` restricts the run to the intersection
of the two corpora without ever loading one tree's snapshots in the other.

``measure_case`` is also the corpus golden's instrument: it is imported by
``tests/backend/inference/test_sol_surface_golden.py`` so the surface under test and the surface
this file dumps for a cross-tree comparison are the same columns, declared once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEPLOYMENTS = {
    "eight": {
        "num_gpus": 8,
        "parallel": {
            "tp_attention": 8,
            "tp_dense_ffn": 8,
            "tp_moe_expert": 8,
            "ep_moe": 1,
            "cp": 1,
            "dp_attention": 1,
        },
    },
    "single": {
        "num_gpus": 1,
        "parallel": {
            "tp_attention": 1,
            "tp_dense_ffn": 1,
            "tp_moe_expert": 1,
            "ep_moe": 1,
            "cp": 1,
            "dp_attention": 1,
        },
    },
}

FIXED_INPUTS = {
    "gpu_sku": "H100-SXM",
    # Every model as it ships, stated: the instrument carries no precision
    # default of its own. The profile's key is written into the output.
    "precision": {"preset": "as_shipped"},
    "avg_input_tokens": 4096,
    "avg_output_tokens": 1024,
    "selection_mode": "theoretical_only",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="tree to import and measure (default: current directory)",
    )
    parser.add_argument(
        "--reference-snapshot-dir",
        type=Path,
        help="restrict snapshots to names present in this second corpus",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        help="measure only this snapshot filename (repeatable)",
    )
    parser.add_argument(
        "--deployment",
        action="append",
        choices=sorted(DEPLOYMENTS),
        default=[],
        help="measure only this deployment (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: stdout)",
    )
    return parser.parse_args()


def _snapshot_names(
    snapshot_dir: Path,
    reference_dir: Path | None,
    requested: list[str],
) -> tuple[list[str], list[str], list[str]]:
    local = {p.name for p in snapshot_dir.glob("*.json")}
    reference = (
        {p.name for p in reference_dir.glob("*.json")}
        if reference_dir is not None
        else set(local)
    )
    included = local & reference
    if requested:
        missing = set(requested) - included
        if missing:
            raise ValueError(
                "requested snapshots are absent from the corpus intersection: "
                + ", ".join(sorted(missing))
            )
        included &= set(requested)
    return sorted(included), sorted(local - reference), sorted(reference - local)


def _total_params(weight_breakdown: dict[str, Any]) -> int:
    """Sum the non-overlapping root rows exposed by weight_breakdown."""
    roots = list(weight_breakdown.get("groups") or [])
    roots.extend(weight_breakdown.get("model_level") or [])
    return sum(int(row.get("params") or 0) for row in roots)


def _kv_metrics(result: dict[str, Any]) -> tuple[float, float, float]:
    """The cache's law as the estimate budgets with it, per rank.

    `kv_cache.per_token_bytes` and `kv_cache.constant_bytes` are the two
    terms `max_tokens` divides by and subtracts: a request stores
    `per_token_bytes × L + constant_bytes` past every window (a full window
    and a recurrent state are the second term, not the first). Static scales
    are held once per rank, whatever the requests.

    This used to re-derive the slope from presentation structures — the
    per-class popover rows whose law was not `1`, or the per-request formula
    card's first subterm — so a sliding window still counted as growth, and
    for an all-window model the "slope" was a whole request's bytes.
    """
    kv = result["kv_cache"]
    return (float(kv["per_token_bytes"]), float(kv["constant_bytes"]),
            float(kv["static_scale_bytes"]))


def _row(row: dict[str, Any]) -> dict[str, Any]:
    """One per-op row, as four published columns that add to the fifth.

    This used to re-derive the weight read — `min(B x s_ctx x per_token_active,
    total)` off the overlay batch — and take activation as `access - weight -
    metadata`. Two things were wrong with that. The engine's own min carries
    `ep_load_factor` inside it and this one did not, so under an imbalanced
    routing trace the two disagreed; and because activation was a REMAINDER,
    the disagreement was published as activation traffic rather than as a
    mismatch. The engine now publishes `weight_read_bytes_per_rank` and
    `act_in_bytes_per_rank`, so every term is read and the identity
    `access = weight_read + metadata + act_in + act_out + convert` holds by
    construction — which `backend.inference.sol_closure` asserts rather
    than assumes (`measure_case` checks the closure before reading a row).
    The last term is the quantised copy of its own operand a matmul's kernel
    needs when the edge does not carry it, and it is published on its own
    rather than folded into the activation halves: it is neither the read the
    edge carried nor the write the output is.
    """
    access = int(row.get("access_bytes_per_rank") or 0)
    metadata = int(row.get("metadata_bytes_per_rank") or 0)
    weight_read = int(row.get("weight_read_bytes_per_rank") or 0)
    act_in = int(row.get("act_in_bytes_per_rank") or 0)
    act_out = int(row.get("act_out_bytes_per_rank") or 0)
    convert = int(row.get("convert_bytes_per_rank") or 0)
    activation = act_in + act_out
    return {
        "path": row.get("aggregation_id"),
        "op": row.get("op"),
        "count_per_step": int(row.get("count_per_step") or 0),
        "flops_per_rank": int(row.get("flops_per_rank") or 0),
        "access_bytes_per_rank": access,
        "weight_read_bytes_per_rank": weight_read,
        "metadata_read_bytes_per_rank": metadata,
        "activation_bytes_per_rank": activation,
        # Both halves are folds of published columns now. They were a remainder
        # and a None while the weight read was re-derived here; with the
        # engine's own `act_in` on the row there is nothing left to subtract
        # against.
        "activation_read_bytes_per_rank": act_in,
        "activation_write_bytes_per_rank": act_out,
        # The input conversion, named: `access` is the five terms above.
        "convert_bytes_per_rank": convert,
        # Still None, and for a different reason than the two above: the cache
        # a step reads arrives as an INPUT to the attention leaf in the
        # post-swap tree, so it is inside `act_in` rather than beside it.
        # Publishing a number here would mean splitting that input by
        # provenance, which the tree does not currently carry.
        "kv_read_bytes_per_rank": None,
        "t_sol_s": float(row.get("t_sol_s") or 0.0),
    }


def _phase(
    phase: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = [_row(row) for row in rows]
    normalized.sort(key=lambda row: (str(row["path"]), str(row["op"])))

    sums = {
        key: sum(int(row[key]) for row in normalized)
        for key in (
            "flops_per_rank",
            "access_bytes_per_rank",
            "weight_read_bytes_per_rank",
            "metadata_read_bytes_per_rank",
            "activation_bytes_per_rank",
            "activation_read_bytes_per_rank",
            "activation_write_bytes_per_rank",
            "convert_bytes_per_rank",
        )
    }
    # The closure (`backend.inference.sol_closure.check_closure`) has held
    # by the time a phase is read: the rows' FLOPs, access and write bytes
    # equal the phase's, and every row partitions.
    access = int(phase.get("total_bytes") or 0)
    flops = int(phase.get("flops_per_step") or 0)

    return {
        "batch": int(phase.get("batch") or 0),
        "flops_per_step": flops,
        "access_bytes_per_rank": access,
        "mem_read_bytes": int(phase.get("mem_read_bytes") or 0),
        "mem_write_bytes": int(phase.get("mem_write_bytes") or 0),
        "weight_read_bytes_per_rank": sums["weight_read_bytes_per_rank"],
        "metadata_read_bytes_per_rank": sums["metadata_read_bytes_per_rank"],
        "activation_bytes_per_rank": sums["activation_bytes_per_rank"],
        "activation_read_bytes_per_rank": sums["activation_read_bytes_per_rank"],
        "activation_write_bytes_per_rank": sums["activation_write_bytes_per_rank"],
        # See `_row`: the cache read is inside the attention leaf's input.
        "kv_read_bytes_per_rank": None,
        # `or 0.0` erased the one distinction this column exists to carry. A
        # phase that cannot fit a single request withholds its time and says
        # why; a phase that ran and took no measurable time would report 0.0.
        # Coercing the first into the second made the golden blind to the
        # withhold gate — GLM-5.2 on eight H100s, whose weights alone are
        # 175.65 GB against 72.00 GB usable, was recorded as a 4.08 ms decode.
        "t_sol_s": (None if phase.get("t_sol_s") is None
                    else float(phase["t_sol_s"])),
        "time_unpublished_reason": (
            str(phase.get("time_unpublished_reason") or "") or None),
    }, normalized


def runtime_framework_inputs(resolved: Any) -> dict[str, str]:
    """Preserve the corpus default unless its declared runtime requires another.

    A runtime-specific graph must be measured under its owning framework;
    choosing by model name or silently switching to THEORY would test a
    different execution contract. Ordinary corpus entries retain their input.
    """
    from backend.arch.eval.program import Program
    framework = Program.of(resolved).required_runtime_framework
    return {"framework": framework} if framework is not None else {}


def measure_case(config: dict[str, Any], deployment: dict[str, Any],
                 resolved: Any = None) -> dict[str, Any]:
    from backend.inference.capacity import estimate_pending
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.sol_closure import check_closure

    if resolved is None:
        from backend.arch.loader import get_default_loader
        resolved = get_default_loader().match(config)
    runtime_inputs = runtime_framework_inputs(resolved) if resolved is not None else {}

    # The walk, not the published estimate: a deployment that cannot hold
    # the weights publishes no step (`capacity.without_step`), and half the
    # cases here are such deployments of the corpus's largest models. Their
    # walk is still what every other card and batch is re-priced from, so it
    # is what this surface pins; the time it withholds is pinned as None
    # beside its reason, as before.
    result = estimate_pending(
        config=config,
        resolved=resolved,
        num_gpus=deployment["num_gpus"],
        parallel=ParallelConfig(**deployment["parallel"]),
        **FIXED_INPUTS,
        **runtime_inputs,
    ).finish()
    check_closure(result)

    per_token_kv, constant_kv, static_kv = _kv_metrics(result)
    phases: dict[str, Any] = {}
    ops: dict[str, Any] = {}
    for phase_name, rows_key in (
        ("prefill", "per_op_timing"),
        ("decode", "per_op_timing_decode"),
    ):
        phases[phase_name], ops[phase_name] = _phase(
            result["phases"][phase_name], result.get(rows_key) or []
        )

    wb = result["weight_breakdown"]
    return {
        "model": {
            "weight_bytes_per_rank": float(wb["total_bytes_per_rank"]),
            "total_params": _total_params(wb),
            "per_token_kv_bytes_per_rank": per_token_kv,
            "constant_kv_bytes_per_rank": constant_kv,
            "static_kv_bytes_per_rank": static_kv,
            "max_batch_per_gpu": int(result.get("max_batch_per_gpu") or 0),
            # None is unbounded — a cache that never grows runs out of room
            # for requests, not for tokens — and is kept apart from the 0 of
            # a rank whose weights leave no cache at all.
            "max_tokens_per_gpu": (
                None if result.get("max_tokens_per_gpu") is None
                else int(result["max_tokens_per_gpu"])),
        },
        "phases": phases,
        "ops": ops,
    }


def main() -> int:
    args = _args()
    repo_root = args.repo_root.resolve()
    snapshot_dir = repo_root / "backend" / "arch" / "snapshots"
    names, local_only, reference_only = _snapshot_names(
        snapshot_dir,
        args.reference_snapshot_dir.resolve()
        if args.reference_snapshot_dir is not None
        else None,
        args.snapshot,
    )
    deployments = args.deployment or sorted(DEPLOYMENTS)

    sys.path.insert(0, str(repo_root))
    from backend.arch.loader import SpecLoader
    cases: dict[str, Any] = {}
    for name in names:
        config = json.loads((snapshot_dir / name).read_text())["config"]
        model_id = Path(name).stem.replace("__", "/", 1)
        resolved = SpecLoader().match(config, model_id=model_id)
        cases[name] = {
            deployment_name: measure_case(
                config, DEPLOYMENTS[deployment_name], resolved)
            for deployment_name in deployments
        }

    from backend.arch.eval.precision import PrecisionProfile
    payload = {
        "schema_version": 1,
        "fixed_inputs": FIXED_INPUTS,
        "profile_key": PrecisionProfile.from_request(
            FIXED_INPUTS["precision"]).key,

        "deployments": {name: DEPLOYMENTS[name] for name in deployments},
        "snapshot_selection": {
            "included": names,
            "local_only_skipped": local_only,
            "reference_only_skipped": reference_only,
        },
        "column_sources": {
            "model.weight_bytes_per_rank": "weight_breakdown.total_bytes_per_rank",
            "model.total_params": "sum(weight_breakdown.groups/model_level root params)",
            "model.per_token_kv_bytes_per_rank": "kv_cache.per_token_bytes",
            "model.constant_kv_bytes_per_rank": "kv_cache.constant_bytes",
            "model.static_kv_bytes_per_rank": "kv_cache.static_scale_bytes",
            "model.max_tokens_per_gpu": "max_tokens_per_gpu (None = unbounded)",
            "phase.flops_per_step": "phases.<phase>.flops_per_step",
            "phase.access_bytes_per_rank": "phases.<phase>.total_bytes",
            "phase.mem_read_bytes": "phases.<phase>.mem_read_bytes",
            "phase.mem_write_bytes": "phases.<phase>.mem_write_bytes",
            "op.flops_per_rank": "per_op_timing*.flops_per_rank",
            "op.access_bytes_per_rank": "per_op_timing*.access_bytes_per_rank",
            "op.weight_read_bytes_per_rank": "per_op_timing*.weight_read_bytes_per_rank",
            "op.metadata_read_bytes_per_rank": "per_op_timing*.metadata_bytes_per_rank",
            "op.activation_bytes_per_rank": "per_op_timing*.act_in_bytes_per_rank plus act_out_bytes_per_rank",
            "*.activation_read_bytes_per_rank": "per_op_timing*.act_in_bytes_per_rank",
            "*.activation_write_bytes_per_rank": "per_op_timing*.act_out_bytes_per_rank",
            "*.kv_read_bytes_per_rank": None,
        },
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
