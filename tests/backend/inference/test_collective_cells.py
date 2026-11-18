"""A collective is a kernel somebody has to measure, and its cell is the row's.

Nothing bound the collectives layout grafts, so `all_reduce`, `all_gather`,
`all_to_all`, `reduce_scatter` and the pipeline's `send_recv` only ever
reached the plan's dropped list: communication time had never been
calibrated on any multi-GPU deployment, and because a measured total is
all-or-nothing over a step's ops, one unmeasured collective denied every
multi-GPU layout a measured line at all.

What is pinned here: that every op the registry CALLS a collective is
measurable (so a sixth primitive cannot be added without a cell), that the
cell is read off the facts the walk published about the launch rather than
rebuilt from a deployment label, that a row re-read at another batch has no
cell rather than the walked batch's, and that the one collective which was
already measured — the fused all-reduce+RMSNorm — is untouched.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.ops import OP_REGISTRY
from backend.inference.benchmark_identity import (
    CommLaunch, identity_shape_from_shape_in, resolve_op, row_invocation_dims,
)
from backend.inference.op_signatures import all_signatures, get_signature
from kernel_bench.signatures import op_to_signature


def _row(**over):
    """A collective row as `phases.py` exports one, with the comm facts a
    cell is built from."""
    row = {"is_comm": True, "comm_payload_shape": [8, 1, 2880],
           "comm_world": 8, "comm_base_stride": 1,
           "comm_payload_dtype_bytes": 2.0, "act_dtype": "BF16"}
    row.update(over)
    return row


class TestEveryDeclaredCollectiveIsMeasurable(unittest.TestCase):
    def test_a_collective_op_has_a_signature(self):
        """The coverage gate. Which ops are collectives is one declaration
        (`OpSpec.is_comm_op`), so this asks it rather than listing names —
        a sixth collective primitive fails here until it has a cell."""
        declared = sorted(name for name, spec in OP_REGISTRY.items()
                          if spec.is_comm_op)
        self.assertEqual(
            declared,
            ["all_gather", "all_reduce", "all_to_all", "reduce_scatter",
             "send_recv"],
            "the set of collective primitives moved; every one needs a cell")
        unmeasurable = [op for op in declared if op_to_signature(op) is None]
        self.assertEqual(unmeasurable, [], (
            "a collective with no signature never becomes a job, so its time "
            "is never calibrated and no deployment that runs it can have a "
            "measured total"))

    def test_each_collective_is_its_own_signature(self):
        """One kernel family each: `ncclAllReduce` is not `ncclAllToAll`, and
        a framework that fuses one into a norm does not fuse the others."""
        bound = {op: op_to_signature(op) for op, spec in OP_REGISTRY.items()
                 if spec.is_comm_op}
        self.assertEqual(len(set(bound.values())), len(bound), bound)

    def test_the_view_op_called_broadcast_is_not_one(self):
        """`broadcast` reads like a collective and is not: it inserts an axis
        with `expand()`, zero FLOPs and zero HBM. It shows up beside the real
        collectives in the plan's dropped list, which is correct — a view op
        has no kernel to measure."""
        spec = OP_REGISTRY["broadcast"]
        self.assertTrue(spec.is_view_op)
        self.assertFalse(spec.is_comm_op)
        self.assertIsNone(op_to_signature("broadcast"))


class TestTheCellIsTheLaunchTheRowPublished(unittest.TestCase):
    def test_the_payload_is_the_buffer_not_the_operand(self):
        """An all-gather's `shape_in` is one rank's shard while the launch
        moves the gathered buffer. The walk states which it priced
        (`comm_payload_shape`), so the cell reads that and does not pick a
        side of the node."""
        cell = json.loads(identity_shape_from_shape_in(
            "all_gather", [8, 1, 25136], shape_out=[8, 1, 201088],
            comm=CommLaunch.from_row(_row(comm_payload_shape=[8, 1, 201088]))))
        self.assertEqual(cell, {"n_comm_elems": 8 * 201088,
                                "n_comm_ranks": 8, "n_comm_stride": 1})

    def test_the_payload_enters_flattened(self):
        """One count, not a shape: these kernels take a pointer, a count, a
        dtype and a communicator, so two payloads of equal size are one
        measurement however they were shaped."""
        a = identity_shape_from_shape_in(
            "all_reduce", [8, 1, 2880],
            comm=CommLaunch.from_row(_row(comm_payload_shape=[8, 1, 2880])))
        b = identity_shape_from_shape_in(
            "all_reduce", [2880, 8], shape_out=[2880, 8],
            comm=CommLaunch.from_row(_row(comm_payload_shape=[2880, 8])))
        self.assertEqual(a, b)
        self.assertEqual(json.loads(a)["n_comm_elems"], 23040)

    def test_the_group_and_its_stride_are_both_in_the_cell(self):
        """An 8-rank all-reduce inside one node is not the measurement an
        8-rank all-reduce across eight nodes is, and what decides that is
        the group's spacing on the mesh (`comm_base_stride`) against the
        box's own `gpus_per_node`. So the stride is part of the cell; the
        box is part of the hardware key."""
        near = identity_shape_from_shape_in(
            "all_to_all", [8, 1, 2880], comm=CommLaunch.from_row(_row()))
        far = identity_shape_from_shape_in(
            "all_to_all", [8, 1, 2880],
            comm=CommLaunch.from_row(_row(comm_base_stride=8)))
        self.assertNotEqual(near, far)
        self.assertEqual(json.loads(far)["n_comm_stride"], 8)

    def test_a_group_of_one_is_no_collective(self):
        """At one rank an all-reduce is the identity — nothing grafts one,
        and the fused collective's cell refuses it on the same grounds."""
        self.assertIsNone(CommLaunch.from_row(_row(comm_world=1)))

    def test_a_row_that_is_not_a_collective_has_no_launch(self):
        """`is_comm` is the row's own flag, the walk's projection of
        `OpSpec.is_comm_op`; no op name is tested here."""
        self.assertIsNone(CommLaunch.from_row(_row(is_comm=False)))

    def test_a_re_read_row_keys_the_payload_it_carries(self):
        """A row re-read at another batch is keyed by the payload it
        carries, because `batch_sweep.rows_at` now re-reads the
        collective columns from the same law as the shapes.

        It used to be refused outright: the payload stayed at the walked
        batch while the shapes moved, so keying off it would hand two
        batches one cell and bill the smaller an eighth of the bytes. The
        blanket refusal that bought safety cost every collective its cell at
        every batch but the walked one. What decides now is whether the row
        HAS a payload — the rescale drops it where no law can state it (a
        gather widens an axis; the element width is unpublished) — so this
        asks for both halves of that.
        """
        # The payload the rescale wrote at the target batch, not the
        # anchor's: this fixture's walked payload is [8,1,2880].
        re_read = _row(shape_at_batch=True, comm_payload_shape=[1, 1, 2880])
        launch = CommLaunch.from_row(re_read)
        self.assertIsNotNone(launch)
        self.assertEqual(launch.payload_shape, (1, 1, 2880))
        self.assertIsNotNone(identity_shape_from_shape_in(
            "all_reduce", [1, 1, 2880], comm=launch))
        # A re-read row whose payload the rescale could not state carries
        # none, and no payload is still no cell.
        for dropped in ({**re_read}, {**_row(shape_at_batch=False)}):
            dropped.pop("comm_payload_shape", None)
            self.assertIsNone(CommLaunch.from_row(dropped))

    def test_a_payload_of_another_width_has_no_cell(self):
        """The cell counts elements, and the buffer gets built and billed at
        the width this row computes at. DCP's log-sum-exp all-gather moves
        an fp32 statistic beside bf16 activations
        (`materialize_rows._LSE_BYTES = 4.0`): as a cell it would be
        measured at half its width and billed at full. The row publishes a
        width, not a dtype name, and a width cannot name a dtype — 1 byte is
        fp8 or int8 — so there is nothing to route it into and the row is
        refused instead."""
        self.assertIsNone(CommLaunch.from_row(
            _row(comm_payload_dtype_bytes=4.0)))
        # Consistent at another width is a cell; it is the DISAGREEMENT that
        # is refused, not fp32 itself.
        launch = CommLaunch.from_row(
            _row(comm_payload_dtype_bytes=4.0, act_dtype="F32"))
        self.assertIsNotNone(launch)
        self.assertEqual(launch.payload_dtype_bytes, 4.0)

    def test_the_width_is_carried_so_a_consumer_need_not_re_derive_it(self):
        launch = CommLaunch.from_row(_row())
        self.assertEqual(launch.payload_dtype_bytes, 2.0)
        self.assertEqual(launch.elements * launch.payload_dtype_bytes,
                         8 * 2880 * 2.0)

    def test_a_missing_fact_is_a_refusal_not_a_default(self):
        for over in ({"comm_payload_shape": []},
                     {"comm_payload_shape": [8, 0, 2880]},
                     {"comm_world": None},
                     {"comm_base_stride": 0},
                     {"comm_payload_dtype_bytes": None},
                     {"act_dtype": None}):
            with self.subTest(over=over):
                self.assertIsNone(CommLaunch.from_row(_row(**over)))


class TestTheChannelDisturbsNothingElse(unittest.TestCase):
    def test_the_fused_collective_keeps_its_cell(self):
        """The one collective that was already measured — correctly: its
        launch shape is one call, its group is read from the row, its count
        is the layer count. It keys off the per-rank layout and the group,
        NOT the flat payload, and passing a launch to it changes nothing."""
        sig = get_signature("allreduce_fused_rmsnorm")
        self.assertEqual(sig.shape_projection, "fold_rmsnorm")
        self.assertEqual(sig.cell_dims,
                         ("B", "s_query", "d_model", "n_comm_ranks"))
        spec_dims = row_invocation_dims({}, signature=sig, comm_world=8)
        without = identity_shape_from_shape_in(
            "allreduce_fused_rmsnorm", [8, 1, 7168],
            shape_out=[[8, 1, 7168]], spec_dims=spec_dims)
        with_launch = identity_shape_from_shape_in(
            "allreduce_fused_rmsnorm", [8, 1, 7168],
            shape_out=[[8, 1, 7168]], spec_dims=spec_dims,
            comm=CommLaunch.from_row(_row(comm_payload_shape=[8, 1, 7168])))
        self.assertEqual(without, with_launch)
        self.assertEqual(json.loads(without)["n_comm_ranks"], 8)

    def test_no_other_signature_reads_the_launch(self):
        """Nothing that already produced a job may change. A launch is
        offered to every projection and only the collective one takes it,
        so no existing key can move."""
        launch = CommLaunch.from_row(_row())
        cases = (("linear", [1, 2048, 3072], [1, 2048, 4096]),
                 ("rmsnorm", [1, 2048, 3072], None),
                 ("silu_and_mul", [1, 2048, 512], None),
                 ("flash_attention", [[1, 32, 2048, 128], [1, 8, 2048, 128]],
                  None))
        for op, shape_in, shape_out in cases:
            with self.subTest(op=op):
                self.assertEqual(
                    identity_shape_from_shape_in(op, shape_in,
                                                 shape_out=shape_out),
                    identity_shape_from_shape_in(op, shape_in,
                                                 shape_out=shape_out,
                                                 comm=launch))

    def test_a_routed_exchange_is_its_own_cell(self):
        """A dispatch's exchange is not the flat `all_to_all_single` the
        plain cell describes: it sends a token once per distinct destination
        its top-k picks reach, so its per-peer messages are ragged and their
        total a fraction of the buffer. The row says so (`comm_route_fanout`,
        a fact of the graph), so it keys as its own kernel with the three
        routing axes beside the buffer and the group."""
        fanout = {"picks": 8, "targets": 256, "placed": 256}
        row = _row(comm_route_fanout=fanout)
        self.assertEqual(resolve_op("all_to_all", route_fanout=True),
                         "routed_all_to_all")
        cell = json.loads(identity_shape_from_shape_in(
            "routed_all_to_all", [8, 1, 2880],
            comm=CommLaunch.from_row(row)))
        self.assertEqual(cell, {"n_comm_elems": 8 * 2880, "n_comm_ranks": 8,
                                "n_comm_stride": 1, "n_experts_active": 8,
                                "n_route_targets": 256,
                                "n_route_experts": 256})

    def test_an_exchange_with_no_fan_out_keys_as_it_did(self):
        """A Ulysses CP exchange carries no fan-out: the same kernel, the
        same cell, the same three axes as before — the routed cell takes
        nothing that was already keyed."""
        self.assertEqual(resolve_op("all_to_all"), "all_to_all")
        self.assertIsNone(CommLaunch.from_row(_row()).route_fanout)
        cell = json.loads(identity_shape_from_shape_in(
            "all_to_all", [8, 1, 2880], comm=CommLaunch.from_row(_row())))
        self.assertEqual(cell, {"n_comm_elems": 8 * 2880, "n_comm_ranks": 8,
                                "n_comm_stride": 1})

    def test_only_the_collective_signatures_declare_the_projection(self):
        """The projection name is the declaration the rest of the code asks
        — the ref registry builds payload buffers for exactly these — so it
        must name exactly the collective signatures."""
        by_projection = sorted(s.name for s in all_signatures()
                               if s.shape_projection == "collective")
        self.assertEqual(by_projection,
                         ["all_gather", "all_reduce", "all_to_all",
                          "reduce_scatter", "routed_all_to_all",
                          "send_recv"])


if __name__ == "__main__":
    unittest.main()
