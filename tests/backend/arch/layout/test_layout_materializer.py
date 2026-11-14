"""Typed communication materialization and public-tree contracts."""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import Dim, value_of
from backend.arch.eval.occurrence import Reference, ReferenceTable
from backend.arch.eval.program import Program
from backend.arch.eval.transition import RedistributeSite
from backend.arch.eval.values import Shape
from tests.snapshots import own_walk, shared_loader
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.layout.layout import physical_local_extent, replicated
from backend.arch.layout.materialize_comm import (
    FlowRef,
    LogicalBindingGap,
    PayloadRole,
    Placement,
    PostTransform,
    RewriteTarget,
)
from backend.arch.layout.materialize_graph import validate_event_graph
from backend.arch.layout.materializer import (
    collect_events,
    materialize_comm_events,
    materialize_comm_tree,
    dcp_events,
    emit_event,
    lower_record,
    node_order,
    reduction_event,
)
from backend.arch.layout.value import ONLINE_ATTENTION, Partial
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.step_eval import evaluate_step

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
PHASES = ("prefill", "decode")


def _resolved(snapshot):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    if resolved is None:
        raise AssertionError(f"snapshot {snapshot} did not resolve")
    return resolved


class TestUnknownCollectivesRefuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resolved = _resolved("meta-llama__Llama-3.2-3B.json")
        cls.step = evaluate_step(
            resolved,
            ParallelConfig(tp_attention=8, tp_dense_ffn=8),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )

    def _probe_record(self, **overrides):
        analysis = self.step.analysis
        base = analysis.redistributes[1]  # a real edge record for scaffolding
        fields = dict(
            node=base.node,
            input_index=base.input_index,
            producer=base.producer,
            arrived=base.arrived,
            required=base.required,
            collectives=base.collectives,
            producer_port=base.producer_port,
            site=base.site,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_an_unlowered_collective_raises_by_name(self):
        probe = self._probe_record(collectives=(("cp", "gather_scatter"),))
        with self.assertRaisesRegex(ValueError, "gather_scatter"):
            lower_record(self.step.analysis, probe)

    def test_a_record_without_collectives_raises(self):
        with self.assertRaisesRegex(ValueError, "no collectives"):
            lower_record(self.step.analysis, self._probe_record(collectives=()))

    def test_a_generic_multi_collective_record_requires_an_explicit_chain(self):
        probe = self._probe_record(
            collectives=(("tp_attention", "all_reduce"),) * 2
        )
        with self.assertRaisesRegex(ValueError, "explicit physical event chain"):
            lower_record(self.step.analysis, probe)

    def test_event_constructor_requires_axis_exact_collective_geometry(self):
        record = self._probe_record()
        common = dict(
            record=record,
            anchor_uid=record.node,
            group_size=8,
            count_uid=record.node,
            world="tp_attention",
            source=FlowRef(record.producer),
            rewrites=(),
            placement_uid=record.node,
            placement=Placement.BEFORE,
        )
        malformed = (
            (
                "all_reduce",
                (2, 6, 1, 128),
                (1, 1, 768),
                (2, 6, 1, 128),
            ),
            ("all_gather", (1, 2), (2, 8), (2, 8)),
            ("reduce_scatter", (2, 8), (1, 2), (2, 8)),
            ("all_to_all", (2, 4), (8,), (2, 4)),
        )
        for index, (op, before, after, payload) in enumerate(malformed):
            with self.subTest(op=op), self.assertRaisesRegex(
                ValueError, "invalid collective geometry"
            ):
                emit_event(
                    self.step.analysis,
                    **common,
                    event_id=f"malformed{index}",
                    op=op,
                    flow_in_shape=before,
                    collective_in_shape=before,
                    collective_out_shape=after,
                    flow_out_shape=after,
                    payload_shape=payload,
                )

    def test_derived_value_family_owns_its_exact_world_label(self):
        step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(tp_attention=16, tp_dense_ffn=16),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        record = next(
            item
            for item in step.analysis.redistributes
            if item.arrived.value is not None
            and item.arrived.value.group.id == "tp_attention^cross"
        )
        event = reduction_event(step.analysis, record, "tp_attention^cross")
        self.assertEqual((event.group_size, event.base_stride), (8, 2))
        for bad_world in ("tp_attention", "bogus_world"):
            with self.subTest(world=bad_world), self.assertRaisesRegex(
                ValueError, "does not name its value family"
            ):
                reduction_event(step.analysis, record, bad_world)

    def test_event_graph_rejects_dangling_and_inconsistent_topology(self):
        events = collect_events(self.step.analysis)
        rewritten = next(event for event in events if event.rewrites)
        with self.assertRaisesRegex(ValueError, "unknown source port"):
            validate_event_graph(self.step.analysis, (
                replace(
                    rewritten,
                    source=replace(rewritten.source, port="wrong_port"),
                ),
            ))
        with self.assertRaisesRegex(ValueError, "missing or later event"):
            validate_event_graph(self.step.analysis, (
                replace(
                    events[0], source=None, source_event="later_event"
                ),
            ))
        with self.assertRaisesRegex(ValueError, "duplicate or empty"):
            validate_event_graph(self.step.analysis, (
                events[0], replace(events[1], event_id=events[0].event_id)
            ))

        cp_step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        chain = [
            event
            for event in collect_events(cp_step.analysis)
            if event.site == RedistributeSite.MODEL_OUTPUT
        ]
        with self.assertRaisesRegex(ValueError, "unknown source port"):
            validate_event_graph(cp_step.analysis, (
                replace(
                    chain[0],
                    source=replace(chain[0].source, port="wrong_port"),
                ),
            ))
        with self.assertRaisesRegex(ValueError, "placements differ"):
            second = replace(
                chain[0],
                event_id="synthetic_second",
                source=None,
                source_event=chain[0].event_id,
                tree_anchor_event=chain[0].event_id,
                flow_in_shape=chain[0].flow_out_shape,
                rewrites=(),
                placement=Placement.BEFORE,
            )
            validate_event_graph(cp_step.analysis, (
                chain[0], second,
            ))

    def test_real_events_preserve_site_placement_and_payload_formula(self):
        events = materialize_comm_events(self.step.analysis)

        stage_close = [
            event for event in events
            if event.site == RedistributeSite.STAGE_CLOSE
        ]
        model_output = [
            event for event in events
            if event.site == RedistributeSite.MODEL_OUTPUT
        ]
        edges = [
            event for event in events
            if event.site == RedistributeSite.EDGE
        ]
        self.assertEqual(len(stage_close), 1)
        self.assertEqual(len(model_output), 1)
        self.assertTrue(edges)
        self.assertEqual(
            (stage_close[0].placement, stage_close[0].rewrites),
            (Placement.AFTER, ()),
        )
        self.assertEqual(
            (model_output[0].placement, model_output[0].rewrites),
            (Placement.AFTER, ()),
        )
        self.assertTrue(all(
            event.placement == Placement.BEFORE and event.rewrites
            for event in edges
        ))
        for event in events:
            expected = (
                math.prod(event.payload_shape)
                * event.payload_dtype_bytes
            )
            self.assertEqual(event.payload_bytes, expected)
            if event.payload_role == PayloadRole.FLOW_BUFFER:
                basis = (
                    event.collective_out_shape
                    if event.op == "all_gather"
                    else event.collective_in_shape
                )
                self.assertEqual(event.payload_shape, basis)

    def test_decode_dcp_uses_the_full_reduce_scatter_input_buffer(self):
        resolved = _resolved("meta-llama__Llama-3.2-3B.json")
        parallel = ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2
        )
        step = evaluate_step(
            resolved,
            parallel,
            phase="decode",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        events = materialize_comm_events(step.analysis)
        scattered = next(
            event
            for event in events
            if event.op == "reduce_scatter" and event.world == "dcp"
        )
        lse = next(
            event
            for event in events
            if event.op == "all_gather"
            and event.world == "dcp"
            and event.payload_role == PayloadRole.AUXILIARY_STATISTIC
        )

        self.assertEqual(lse.flow_in_shape, (1, 12, 1, 128))
        self.assertEqual(lse.flow_out_shape, lse.flow_in_shape)
        self.assertEqual(lse.collective_in_shape, (1, 12, 1, 1))
        self.assertEqual(lse.collective_out_shape, (1, 12, 1, 2))
        self.assertEqual(lse.bytes_per_rank, 1344.0)
        self.assertEqual(
            math.prod(scattered.flow_in_shape),
            scattered.group_size * math.prod(scattered.flow_out_shape),
        )
        self.assertEqual(scattered.payload_shape, scattered.flow_in_shape)
        self.assertEqual(scattered.flow_in_shape, (1, 12, 1, 128))
        self.assertEqual(scattered.flow_out_shape, (1, 6, 1, 128))
        self.assertEqual(scattered.bytes_per_rank, 43008.0)

        public = analyze(
            resolved,
            parallel=parallel,
            phase="decode",
            batch_size=1,
            step_eval=step,
            breakdown=True,
        )
        public_rs = next(
            row
            for row in public.per_op_timing
            if row.get("op") == "reduce_scatter"
            and row.get("comm_world") == 2
        )
        self.assertEqual(
            public_rs["comm_bytes_per_rank"], scattered.bytes_per_rank
        )

    def test_dcp_auxiliary_and_chained_payloads_are_typed_by_their_emitter(self):
        """The LSE statistic is the apply output's typed shape less its
        feature axis, and the output collective chained after it
        (`source_event`) carries its own typed payload: both grow with the
        request batch and their values are the bytes the event moves."""
        resolved = _resolved("meta-llama__Llama-3.2-3B.json")
        parallel = ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2
        )
        step = evaluate_step(
            resolved, parallel, phase="decode", batch=3, input_tokens=128, output_tokens=0,
        )
        events = materialize_comm_events(step.analysis)
        lse = next(
            event for event in events
            if event.world == "dcp"
            and event.payload_role == PayloadRole.AUXILIARY_STATISTIC
        )
        chained = next(
            event for event in events if event.source_event == lse.event_id
        )
        for event in (lse, chained):
            self.assertEqual(float(event.payload_extent), event.payload_bytes)
            self.assertIn("B", event.payload_extent.names)
            self.assertEqual(
                event.payload_bytes,
                math.prod(int(d) for d in event.payload_shape)
                * event.payload_dtype_bytes,
            )
        self.assertEqual(lse.payload_shape[0], 3)
        self.assertEqual(lse.payload_shape[-1], 2)

    def test_materialized_tree_preserves_event_topology_and_program(self):
        step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(
                tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2
            ),
            phase="decode",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        source_tree = step.analysis.program.ir
        source_nodes = tuple(step.analysis.program.nodes())
        materialized = materialize_comm_tree(step.analysis)

        def walk(tree):
            def visit(node):
                yield node
                for child in node.children or ():
                    yield from visit(child)

            for root in tree.top_level:
                yield from visit(root)
            for _meta, roots in tree.blocks:
                for root in roots:
                    yield from visit(root)

        copied_nodes = tuple(walk(materialized.tree))
        copied_by_uid = {node.uid: node for node in copied_nodes}
        source_uids = {node.uid for node in source_nodes}
        self.assertIsNot(materialized.tree, source_tree)
        self.assertTrue(all(node.provenance == "spec" for node in source_nodes))
        self.assertEqual(
            len(copied_nodes), len(source_nodes) + len(materialized.events)
        )
        self.assertEqual(
            set(materialized.node_uid_by_event),
            {event.event_id for event in materialized.events},
        )
        self.assertTrue(
            source_uids.isdisjoint(materialized.node_uid_by_event.values())
        )

        events = {event.event_id: event for event in materialized.events}
        for event_id, event in events.items():
            node = copied_by_uid[materialized.node_uid_by_event[event_id]]
            self.assertEqual((node.op, node.provenance), (event.op, "comm"))
            self.assertEqual(node.dims["__comm_world__"], event.group_size)
            self.assertEqual(
                node.dims["__comm_base_stride__"], event.base_stride
            )
            self.assertEqual(
                node.dims["__anchor_spec_id__"],
                copied_by_uid[event.report_anchor_uid].display_path,
            )
            uid = node.uid
            shape = materialized.shapes.values[uid]
            self.assertEqual(shape.ins, (tuple(event.flow_in_shape),))
            self.assertEqual(shape.out, tuple(event.flow_out_shape))
            self.assertEqual(materialized.instances.values[uid], event.count)
            self.assertEqual(
                materialized.comm_bytes.values[uid] * event.count,
                event.bytes_per_rank,
            )
            meta = materialized.comm_meta.values[uid]
            self.assertEqual(meta["tensor_bytes"], event.payload_bytes)
            self.assertEqual(meta["tensor_shape"], list(event.payload_shape))
            self.assertEqual(
                meta["parent_aggregation_id"],
                copied_by_uid[event.report_anchor_uid].aggregation_id,
            )
            source = node.inputs_obj[0].src
            if event.source is not None:
                self.assertEqual(source.uid, event.source.node_uid)
            else:
                self.assertEqual(
                    source.uid,
                    materialized.node_uid_by_event[event.source_event],
                )
            for target in event.rewrites:
                consumer = copied_by_uid[target.node_uid]
                self.assertIs(
                    consumer.inputs_obj[target.input_index].src, node
                )

            placement = copied_by_uid[event.placement_uid]
            structural_anchor = (
                copied_by_uid[
                    materialized.node_uid_by_event[event.tree_anchor_event]
                ]
                if event.tree_anchor_event is not None
                else placement
            )
            self.assertIs(node.parent, structural_anchor.parent)
            siblings = (
                node.parent.children
                if node.parent is not None
                else next(
                    roots
                    for roots in (
                        materialized.tree.top_level,
                        *(roots for _meta, roots in materialized.tree.blocks),
                    )
                    if any(candidate is node for candidate in roots)
                )
            )
            if event.tree_anchor_event is not None:
                self.assertGreater(
                    siblings.index(node), siblings.index(structural_anchor)
                )
            elif event.placement == Placement.BEFORE:
                self.assertLess(siblings.index(node), siblings.index(placement))
            else:
                self.assertGreater(
                    siblings.index(node), siblings.index(placement)
                )

    def test_materialized_tree_uses_the_selected_stage_counts(self):
        from backend.arch.layout.stages import StagePlacement

        step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(tp_attention=8, tp_dense_ffn=8, pp=4),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        analysis = step.analysis
        # 28 layers at pp = 4: stage 0 runs 7 of them and the embedding.
        stage = StagePlacement.of(analysis).view(0)
        materialized = materialize_comm_tree(
            analysis,
            stage=stage,
        )
        runs = stage.rows(analysis.program.included_stack_names())
        block_uids = set()
        for contribution in analysis.program.contributions:
            if contribution.block_index is None:
                continue
            pending = [contribution.node]
            while pending:
                node = pending.pop()
                block_uids.add(node.uid)
                pending.extend(node.children or ())

        self.assertTrue(materialized.events)
        for event in materialized.events:
            if event.op == "send_recv":
                expected = 1
            elif event.count_uid in block_uids:
                expected = 7
            else:
                expected = runs.values[event.count_uid]
            self.assertEqual(event.count, expected)
        self.assertIn(1, {e.count for e in materialized.events
                          if e.count_uid not in block_uids})

    def test_comm_artifacts_are_valid_fusion_inputs(self):
        from backend.arch.eval.fusion import (
            Framework,
            fusion_matches,
            materialize_tree,
        )
        from backend.arch.eval.layer import Layer
        from backend.inference.framework_bindings import get_binding
        from backend.inference.fusion_plans import FUSION_SCHEMES
        from backend.inference.phases import compute_per_op_timing
        from tests.serving_fixture import numbers_as_laws

        analysis = self.step.analysis
        materialized = materialize_comm_tree(analysis)
        post_program, rewrite = Framework.of(
            get_binding("vllm"), FUSION_SCHEMES
        ).rewrite(materialized.program, analysis)
        matches = Counter(
            match["fused_op"] for match in fusion_matches(
                materialized.program, post_program, rewrite, FUSION_SCHEMES
            )
        )
        self.assertGreaterEqual(
            matches.get("allreduce_fused_rmsnorm", 0), 1
        )
        self.assertTrue(rewrite.origins)

        post = materialize_tree(
            materialized.program, post_program, rewrite
        )

        def walk(node):
            yield node
            for child in node.children or ():
                yield from walk(child)

        post_nodes = tuple(
            node
            for roots in (
                post.top_level,
                *(roots for _meta, roots in post.blocks),
            )
            for root in roots
            for node in walk(root)
        )
        self.assertEqual(
            len({node.uid for node in post_nodes}), len(post_nodes)
        )

        timing = compute_per_op_timing(
            materialized.program,
            materialized.cost,
            materialized.shapes,
            Layer(materialized.program, {}),
            instances=materialized.instances,
            # What this checks is the collectives' rows; no row's weight
            # shards are read.
            weight_shards=lambda packs: (),
            comm_meta=materialized.comm_meta,
            # The layers at the step's point are their own laws there.
            laws=numbers_as_laws(
                cost=materialized.cost, shapes=materialized.shapes,
                access=Layer(materialized.program, {}),
                access_shapes=materialized.shapes,
                comm_meta=materialized.comm_meta, step_rows=1),
            pre=False,
            at=self.step.point,
            intra_bw_bytes_per_sec=1e12,
            inter_bw_bytes_per_sec=1e11,
            gpus_per_node=None,
            breakdown=True,
        )
        by_sid = {row["aggregation_id"]: row for row in timing}
        for event in materialized.events:
            node = materialized.program.by_uid(
                materialized.node_uid_by_event[event.event_id]
            )
            row = by_sid[node.aggregation_id]
            self.assertEqual(row["shape_in"], list(event.flow_in_shape))
            self.assertEqual(row["shape_out"], list(event.flow_out_shape))
            self.assertEqual(
                row["comm_bytes_per_rank"], event.bytes_per_rank
            )

    def test_graft_preserves_a_named_source_port(self):
        events = materialize_comm_events(self.step.analysis)
        event = next(item for item in events if not item.rewrites)
        self.assertIsNotNone(event.source)
        source_uid = event.source.node_uid
        source_port = "stats"
        named = replace(
            event,
            source=FlowRef(source_uid, source_port),
            source_event=None,
            tree_anchor_event=None,
        )
        with patch(
            "backend.arch.layout.materializer.collect_events",
            return_value=[named],
        ):
            materialized = materialize_comm_tree(self.step.analysis)
        uid = materialized.node_uid_by_event[named.event_id]
        ref = materialized.program.by_uid(uid).inputs_obj[0]
        self.assertEqual((ref.src.uid, ref.port), (source_uid, source_port))

    def test_graft_allocates_around_an_authored_name_collision(self):
        event = next(
            item
            for item in materialize_comm_events(self.step.analysis)
            if not item.rewrites
        )
        tree = copy.deepcopy(self.step.analysis.program.ir)
        program = Program.of(self.step.analysis.program.resolved, ir=tree)
        placement = program.by_uid(event.placement_uid)
        baseline = materialize_comm_tree(self.step.analysis)
        baseline_uid = baseline.node_uid_by_event[event.event_id]
        base = baseline.program.by_uid(baseline_uid).name
        authored = tree.make_node(
            name=base,
            op="add",
            is_component=False,
            children=[],
            provenance="spec",
        )
        if placement.parent is not None:
            siblings = placement.parent.children
        else:
            siblings = next(
                roots
                for roots in (
                    tree.top_level,
                    *(roots for _meta, roots in tree.blocks),
                )
                if any(candidate is placement for candidate in roots)
            )
        authored.parent = placement.parent
        siblings.insert(siblings.index(placement), authored)
        analysis = replace(
            self.step.analysis,
            program=Program.of(self.step.analysis.program.resolved, ir=tree),
        )
        with patch(
            "backend.arch.layout.materializer.collect_events",
            return_value=[event],
        ):
            materialized = materialize_comm_tree(analysis)
        uid = materialized.node_uid_by_event[event.event_id]
        grafted = materialized.program.by_uid(uid)
        self.assertEqual(grafted.name, f"{base}__2")
        if grafted.parent is not None:
            grafted_siblings = grafted.parent.children
        else:
            grafted_siblings = next(
                roots
                for roots in (
                    materialized.tree.top_level,
                    *(roots for _meta, roots in materialized.tree.blocks),
                )
                if any(candidate is grafted for candidate in roots)
            )
        self.assertEqual(
            [node.name for node in grafted_siblings].count(base), 1
        )
        self.assertEqual(
            [node.name for node in grafted_siblings].count(f"{base}__2"), 1
        )

    def test_mla_dcp_finalizes_at_the_apply_consumer(self):
        step = evaluate_step(
            _resolved("deepseek-ai__DeepSeek-V3.json"),
            ParallelConfig(
                tp_attention=8,
                tp_dense_ffn=8,
                tp_moe_expert=8,
                ep_moe=2,
                cp=2,
                dcp=2,
            ),
            phase="decode",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        direct = next(
            record
            for record in step.analysis.redistributes
            if record.arrived.value is not None
            and record.arrived.value.reducer == ONLINE_ATTENTION
            and record.producer
            == record.arrived.value.provenance.producer
        )
        apply_uid = direct.arrived.value.provenance.producer
        events = materialize_comm_events(step.analysis)
        scattered = next(
            event
            for event in events
            if event.op == "reduce_scatter"
            and event.world == "dcp"
            and event.report_anchor_uid == direct.node
        )
        direct_target = RewriteTarget(direct.node, direct.input_index)
        self.assertEqual(scattered.placement_uid, direct.node)
        self.assertIn(direct_target, scattered.rewrites)
        self.assertNotIn(
            RewriteTarget(apply_uid, direct.input_index), scattered.rewrites
        )

        materialized = materialize_comm_tree(step.analysis)
        def find(uid):
            def visit(node):
                if node.uid == uid:
                    return node
                for child in node.children or ():
                    found = visit(child)
                    if found is not None:
                        return found
                return None

            for root in materialized.tree.top_level:
                found = visit(root)
                if found is not None:
                    return found
            for _meta, roots in materialized.tree.blocks:
                for root in roots:
                    found = visit(root)
                    if found is not None:
                        return found
            raise AssertionError(uid)

        scattered_node = find(
            materialized.node_uid_by_event[scattered.event_id]
        )
        consumer = find(direct.node)
        self.assertIs(
            consumer.inputs_obj[direct.input_index].src, scattered_node
        )

    def test_dcp_trio_coalesces_one_physical_chain_across_fanout(self):
        from backend.arch.layout.materialize_comm import _coalesce_events

        step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(
                tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2
            ),
            phase="decode",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        record = next(
            item
            for item in step.analysis.redistributes
            if item.arrived.value is not None
            and item.arrived.value.reducer == ONLINE_ATTENTION
        )
        query, lse, scattered = dcp_events(step.analysis, record)
        self.assertTrue(all(
            event.dedup_key is not None
            for event in (query, lse, scattered)
        ))
        group = record.arrived.value.group
        reordered_group = ProcessGroupFamily(
            group.id, tuple(reversed(group.groups))
        )
        self.assertEqual(group, reordered_group)
        reordered_record = replace(
            record,
            arrived=replace(
                record.arrived,
                value=replace(
                    record.arrived.value, group=reordered_group
                ),
            ),
        )
        reordered = dcp_events(step.analysis, reordered_record)
        self.assertEqual(
            [event.dedup_key for event in reordered],
            [event.dedup_key for event in (query, lse, scattered)],
        )

        self.assertGreaterEqual(len(scattered.rewrites), 2)
        first_target, second_target = scattered.rewrites[:2]
        first = replace(scattered, rewrites=(first_target,))
        query2 = replace(query, event_id="query_fanout")
        lse2 = replace(lse, event_id="lse_fanout")
        scattered2 = replace(
            scattered,
            event_id="output_fanout",
            source_event=lse2.event_id,
            tree_anchor_event=lse2.event_id,
            rewrites=(second_target,),
        )
        merged = _coalesce_events(
            (query, lse, first, query2, lse2, scattered2),
            node_order(step.analysis),
        )
        self.assertEqual(len(merged), 3)
        self.assertEqual(
            set(merged[2].rewrites), {first_target, second_target}
        )
        self.assertEqual(merged[2].source_event, merged[1].event_id)
        self.assertEqual(merged[2].tree_anchor_event, merged[1].event_id)
        validate_event_graph(step.analysis, merged)

    def test_mla_dcp_over_cp_replicas_has_explicit_local_transforms(self):
        resolved = _resolved("deepseek-ai__DeepSeek-V3.json")
        parallel = ParallelConfig(
            tp_attention=2,
            tp_dense_ffn=2,
            tp_moe_expert=2,
            ep_moe=4,
            cp=4,
            dcp=8,
        )
        prefill = evaluate_step(
            resolved, parallel, phase="prefill", batch=1,
            input_tokens=128, output_tokens=0,
        )
        cache_events = [
            event for event in materialize_comm_events(prefill.analysis)
            if event.site == RedistributeSite.CACHE_ENDPOINT
        ]
        self.assertFalse(cache_events)
        producer = [
            event for event in materialize_comm_events(prefill.analysis)
            if event.site == RedistributeSite.INTERNAL_OUT
            and event.world == "cp" and event.op == "all_gather"
        ]
        self.assertTrue(producer)

        decode = evaluate_step(
            resolved, parallel, phase="decode", batch=1,
            input_tokens=4096, output_tokens=0,
        )
        dcp_events = [
            event for event in materialize_comm_events(decode.analysis)
            if event.world == "dcp"
        ]
        query = next(
            event for event in dcp_events
            if event.op == "all_gather"
            and event.post_transform == PostTransform.REPLICA_SELECT
        )
        output = next(
            event for event in dcp_events
            if event.op == "all_reduce"
            and event.post_transform == PostTransform.LOCAL_SLICE
        )
        self.assertGreater(
            query.collective_out_shape[1], query.flow_out_shape[1]
        )
        self.assertGreater(
            output.collective_out_shape[1], output.flow_out_shape[1]
        )

    def test_generic_gather_and_embedding_use_physical_rank_buffers(self):
        minimax = evaluate_step(
            _resolved("MiniMaxAI__MiniMax-M2.7.json"),
            TP8,
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        gather = next(
            event
            for event in materialize_comm_events(minimax.analysis)
            if minimax.analysis.program.by_uid(
                event.report_anchor_uid
            ).display_path == "self_attn.q_norm"
        )
        self.assertEqual(gather.flow_in_shape, (1, 128, 768))
        self.assertEqual(gather.collective_out_shape, (1, 128, 6144))
        self.assertEqual(gather.payload_shape, gather.collective_out_shape)

        cp_step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        embedding = next(
            event
            for event in materialize_comm_events(cp_step.analysis)
            if event.site == RedistributeSite.STAGE_CLOSE
        )
        self.assertEqual(embedding.flow_in_shape, (1, 64, 3072))
        self.assertEqual(embedding.payload_shape, embedding.flow_in_shape)

    def test_a_column_parallel_output_split_at_declared_offsets_is_gathered(self):
        """DeepSeek-V4.1's Engram `wkv` is column-parallel with its output
        all-gathered before the key/value split (vLLM
        `ColumnParallelLinear(gather_output=True)`, deepseek_v41/common/
        engram.py:1555-1563). The split's slices declare their offsets; a
        proper subinterval of the cut axis is taken of the axis gathered
        whole, so each engram layer runs the row-cut lookup's all_reduce
        before `wkv` and one all_gather of its output, and the pending sum
        is no longer carried through `wkv` to two all_reduces downstream."""
        step = evaluate_step(
            _resolved("deepseek-ai__DeepSeek-V4.1-Flash.json"), TP8,
            phase="decode", batch=1, input_tokens=128, output_tokens=0)
        program = step.analysis.program
        engram = Counter(
            (event.op, program.by_uid(event.report_anchor_uid)
             .display_path.split("__")[0])
            for event in materialize_comm_events(step.analysis)
            if program.by_uid(event.report_anchor_uid).display_path
            .startswith("engram."))
        # Two engram layers (1 and 14), one of each per layer.
        self.assertEqual(engram, Counter({("all_reduce", "engram.embed"): 2,
                                          ("all_gather", "engram.key_flat"): 2}))

    def test_sampled_model_output_only_gathers_vocab(self):
        step = evaluate_step(
            _resolved("meta-llama__Llama-3.2-3B.json"),
            ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0,
        )
        events = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.site == RedistributeSite.MODEL_OUTPUT
        ]

        self.assertEqual([event.world for event in events], ["tp_attention"])
        output = events[0]
        self.assertIsNotNone(output.source)
        self.assertIsNone(output.source_event)
        self.assertEqual(output.flow_in_shape, (1, 1, 32064))
        self.assertEqual(output.collective_out_shape, (1, 1, 128256))
        self.assertEqual(output.payload_shape, output.collective_out_shape)
        self.assertEqual(output.post_transform, PostTransform.NONE)
        self.assertEqual(output.flow_out_shape[-1], 128256)
        cache_events = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.site == RedistributeSite.CACHE_ENDPOINT
        ]
        self.assertEqual(len(cache_events), 0)
        producer = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.site == RedistributeSite.INTERNAL_OUT
            and event.world == "cp" and event.op == "all_gather"
        ]
        self.assertEqual(len(producer), 2)

    def test_a_route_exchange_moves_the_busiest_ranks_rows(self):
        step = evaluate_step(
            _resolved("Qwen__Qwen3-30B-A3B.json"),
            ParallelConfig(ep_moe=8, dp_attention=8),
            phase="prefill",
            batch=1,
            input_tokens=128, output_tokens=0, ep_load_factor=1.25,
        )
        events = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.op == "all_to_all" and event.world == "ep_moe"
        ]

        # rho = 1.25 is per expert; the rank holding the hot expert dilutes it
        # over its k = 128 / 8 = 16 experts:
        # (rho + (k - 1)(E - rho)/(E - 1)) / k. The mean rank exchanges 1024
        # rows; the busiest one the next whole row above 1024 x that.
        rho, e, k = Fraction(5, 4), 128, 16
        busiest_rank = (rho + (k - 1) * (e - rho) / (e - 1)) / k
        rows = math.ceil(1024 * busiest_rank)
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event.meta, ())
            self.assertEqual(event.flow_in_shape, (rows, 2048))
            self.assertEqual(event.flow_out_shape, event.flow_in_shape)
            self.assertEqual(
                event.payload_bytes,
                math.prod(event.payload_shape) * event.payload_dtype_bytes,
            )
            # One typed payload: its value is the bytes, and its rows grow
            # with the request batch.
            self.assertEqual(float(event.payload_extent), event.payload_bytes)
            self.assertIn("B", event.payload_extent.names)

    def test_a_route_exchange_under_attention_tp_holds_the_next_whole_row(self):
        # Attention tensor-parallel 4 ways inside each of 2 data-parallel
        # replicas, experts over all 8 ranks: one decode token per replica
        # routed to top-6 is 12 rows over 8 ranks, 1.5 a rank on average.
        # What a replica sends is whole; what a rank receives is an
        # expectation, and its buffer holds the next whole row.
        step = evaluate_step(
            _resolved("deepseek-ai__DeepSeek-V2-Lite.json"),
            ParallelConfig(tp_attention=4, tp_dense_ffn=4, dp_attention=2,
                           ep_moe=8),
            phase="decode",
            batch=1,
            input_tokens=1024, output_tokens=256,
        )
        events = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.op == "all_to_all" and event.world == "ep_moe"
        ]
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event.flow_in_shape[0], 2)

    def test_cp_exchange_is_not_a_route_exchange_on_the_same_ranks(self):
        step = evaluate_step(
            _resolved("Qwen__Qwen3-30B-A3B.json"),
            ParallelConfig(cp=4, ep_moe=4, dp_attention=1,
                           cp_scheme="ulysses"),
            phase="prefill",
            batch=8,
            input_tokens=2048, output_tokens=0,
        )

        events = materialize_comm_events(step.analysis)

        self.assertTrue(any(
            event.op == "all_to_all" and event.world == "cp"
            for event in events
        ))
        self.assertTrue(any(
            event.op == "all_to_all" and event.world == "ep_moe"
            for event in events
        ))

    def test_route_exchange_packs_fractional_per_expert_mean(self):
        step = evaluate_step(
            _resolved("XiaomiMiMo__MiMo-V2.5.json"),
            ParallelConfig(
                tp_attention=4,
                tp_dense_ffn=4,
                tp_moe_expert=4,
                ep_moe=2,
                dp_attention=2,
            ),
            phase="decode",
            batch=4248,
            input_tokens=4224, output_tokens=0,
        )
        events = [
            event
            for event in materialize_comm_events(step.analysis)
            if event.op == "all_to_all" and event.world == "ep_moe"
        ]

        self.assertTrue(events)
        self.assertTrue(all(
            event.flow_in_shape == (33984, 4096)
            for event in events
        ))

    def test_an_empty_route_exchange_has_no_point(self):
        """A route exchange of zero tokens needs a step of zero requests, and
        the batch counts at least one (`Workload.symbolic`): the walk's laws
        are written for that domain, so a batch of zero is refused at the
        point rather than read off laws that do not hold there."""
        with self.assertRaises(ValueError):
            _ = evaluate_step(
                _resolved("Qwen__Qwen3-30B-A3B.json"),
                ParallelConfig(ep_moe=8, dp_attention=8),
                phase="prefill", batch=0, input_tokens=128,
                output_tokens=0, ep_load_factor=1.25).analysis

    def test_model_output_preserves_request_ownership(self):
        resolved = _resolved("meta-llama__Llama-3.2-3B.json")
        deployments = (
            ParallelConfig(
                tp_attention=2, tp_dense_ffn=2, dp_attention=2
            ),
            ParallelConfig(
                tp_attention=2, tp_dense_ffn=2, dp_attention=2, cp=2
            ),
        )
        for phase in PHASES:
            for parallel in deployments:
                with self.subTest(phase=phase, parallel=repr(parallel)):
                    step = evaluate_step(
                        resolved,
                        parallel,
                        phase=phase,
                        batch=6,
                        input_tokens=128, output_tokens=0,
                    )
                    events = [
                        event
                        for event in materialize_comm_events(step.analysis)
                        if event.site == RedistributeSite.MODEL_OUTPUT
                    ]
                    keys = {
                        key
                        for event in events
                        for key, enabled in event.meta
                        if enabled is True and key.startswith("logical_owner:")
                    }
                    self.assertIn("logical_owner:request_B", keys)
                    self.assertFalse(
                        any(event.world == "dp_attention" for event in events)
                    )
                    if parallel.cp > 1 and phase == "prefill":
                        self.assertFalse(
                            any(event.world == "cp" for event in events)
                        )
                        self.assertNotIn(
                            "logical_owner:sampled_position", keys
                        )

    def test_dsa_indexer_score_cannot_impersonate_the_main_score(self):
        from backend.arch.layout.sparse_dcp import SparseDcpPlans
        def ref(node):
            return SimpleNamespace(src=node, port="")

        def node(uid, op, path, *inputs):
            return SimpleNamespace(
                uid=uid,
                op=op,
                display_path=path,
                inputs_obj=tuple(ref(item) for item in inputs),
                children=(),
            )

        query = node(1, "head_split", "self_attn.query")
        key = node(2, "head_split", "self_attn.key")
        index_query = node(3, "head_split", "self_attn.index_query")
        index_key = node(4, "head_broadcast", "self_attn.index_key")
        index_score = node(
            5, "attention_score", "self_attn.index_score", index_query, index_key
        )
        index_topk = node(6, "dsa_topk_mask", "self_attn.index_topk", index_score)
        main_score = node(
            7, "dsa_attention_score", "self_attn.qk_score", query, key, index_topk
        )
        softmax = node(8, "dsa_softmax", "self_attn.softmax", main_score)
        value = node(9, "head_split", "self_attn.value")
        apply = node(10, "dsa_attention_apply", "self_attn.attn_apply", softmax, value)
        merge = node(11, "head_merge", "self_attn.o_merge", apply)
        o_proj = node(12, "linear", "self_attn.o_proj", merge)
        nodes = (
            query,
            key,
            index_query,
            index_key,
            index_score,
            index_topk,
            main_score,
            softmax,
            value,
            apply,
            merge,
            o_proj,
        )

        class Program:
            def __init__(self):
                self._nodes = {item.uid: item for item in nodes}

            def by_uid(self, uid):
                return self._nodes[uid]

            def nodes(self):
                return nodes

        pending = Partial(
            ONLINE_ATTENTION,
            ProcessGroupFamily("dcp", ((0, 1),)),
        ).with_provenance(apply.uid, score=main_score.uid, query=query.uid)
        ranks = frozenset({0, 1})
        whole = (replicated(ranks),) * 4
        query_shape = (
            Dim(sp.Symbol("B"), 1),
            Dim(sp.Symbol("n_heads"), 8),
            1,
            128,
        )
        apply_shape = (
            Dim(sp.Symbol("B"), 1),
            Dim(sp.Symbol("n_heads"), 4),
            1,
            64,
        )
        query_state = SimpleNamespace(value=None, layout=whole, workload=None)
        query_required = SimpleNamespace(value=None, layout=whole, workload=None)
        apply_state = SimpleNamespace(value=pending, layout=whole, workload=None)
        exit_required = SimpleNamespace(value=None, layout=whole, workload=None)
        states = {item.uid: SimpleNamespace(value=None, layout=whole)
                  for item in nodes}
        states[query.uid] = query_state
        states[apply.uid] = apply_state
        states[merge.uid] = SimpleNamespace(value=pending, layout=whole)
        analysis = SimpleNamespace(
            program=Program(),
            states=SimpleNamespace(values=states),
            shapes=SimpleNamespace(values={
                query.uid: SimpleNamespace(out=query_shape, ins=()),
                apply.uid: SimpleNamespace(out=apply_shape, ins=()),
                merge.uid: SimpleNamespace(out=apply_shape, ins=(apply_shape,)),
            }),
            kernel_shapes=SimpleNamespace(values={
                main_score.uid: Shape(
                    out=(1, 16, 1, 64),
                    ins=(
                        (1, 16, 1, 128),
                        (1, 2, 64, 128),
                        (1, 1, 8),
                    ),
                ),
                apply.uid: Shape(
                    out=(1, 8, 1, 64),
                    ins=((1, 16, 1, 64), (1, 2, 64, 64)),
                ),
            }),
            input_states={
                (main_score.uid, 0): query_required,
                (merge.uid, 0): exit_required,
            },
            instances=SimpleNamespace(
                values={main_score.uid: 1, apply.uid: 1}
            ),
            rule=SimpleNamespace(
                parallel=ParallelConfig(
                    tp_attention=4, tp_dense_ffn=4, cp=2, dcp=2
                ),
                route_family=None,
                sparse_dcp=SparseDcpPlans(),
                families={"dcp": pending.group},
                project_local_extent=lambda extent, layout: (
                    physical_local_extent(layout, int(value_of(extent)))
                ),
            ),
        )

        record = SimpleNamespace(
            node=o_proj.uid,
            input_index=0,
            producer=merge.uid,
            producer_port="out",
            arrived=states[merge.uid],
            required=exit_required,
            collectives=(
                ("dcp", "all_gather_lse"),
                ("dcp", "all_reduce"),
            ),
            site=RedistributeSite.EDGE,
        )
        events = dcp_events(analysis, record)

        self.assertEqual(
            (events[0].report_anchor_uid, events[0].count_uid),
            (main_score.uid, main_score.uid),
        )
        self.assertEqual(events[0].payload_bytes, 4096.0)
        self.assertEqual(events[0].flow_out_shape, (1, 16, 1, 128))
        self.assertEqual(
            events[0].binding_gap, LogicalBindingGap.DCP_HEAD_FLOW
        )
        self.assertEqual(events[0].payload_role, PayloadRole.FLOW_BUFFER)
        self.assertEqual(events[0].source, FlowRef(query.uid))
        self.assertEqual(
            events[0].rewrites, (RewriteTarget(main_score.uid, 0),)
        )
        self.assertEqual(events[1].source, FlowRef(apply.uid))
        self.assertEqual(events[1].rewrites, ())
        self.assertEqual(
            events[1].payload_role, PayloadRole.AUXILIARY_STATISTIC
        )
        self.assertEqual(events[1].flow_in_shape, (1, 8, 1, 64))
        self.assertEqual(events[1].flow_out_shape, (1, 8, 1, 64))
        self.assertEqual(events[1].collective_in_shape, (1, 8, 1, 1))
        self.assertEqual(events[1].collective_out_shape, (1, 8, 1, 2))
        self.assertEqual(events[1].payload_bytes, 64.0)
        self.assertIsNone(events[2].source)
        self.assertEqual(events[2].source_event, events[1].event_id)
        self.assertEqual(
            events[2].rewrites, (RewriteTarget(merge.uid, 0),)
        )
        self.assertEqual(events[2].flow_in_shape, (1, 8, 1, 64))
        self.assertEqual(events[2].flow_out_shape, (1, 4, 1, 64))
        self.assertEqual(
            [
                (event.report_anchor_uid, event.count_uid)
                for event in events[1:]
            ],
            [(merge.uid, apply.uid), (merge.uid, apply.uid)],
        )

        malformed = SimpleNamespace(**{
            **vars(record),
            "collectives": (*record.collectives, ("dcp", "all_reduce")),
        })
        with self.assertRaisesRegex(ValueError, "exact physical signature"):
            dcp_events(analysis, malformed)


#: A backbone whose exit three repetitions of a projection read by name.
#: Under sequence parallelism the exit is cut on the sequence and each
#: projection needs it whole.
_NAMED_READ_SPEC = """\
match: { model_type: [named_read], model_id: [named_read] }
dims:
  d_model: config.hidden_size
  d_ffn: config.intermediate_size
  n_layers: "4"
  n_deepstack: "3"
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: add, in: [x, n], dim: d_model }
    outputs: [o]
  tap:
    graph:
      fc1: { op: linear, in: x, dims: { in: d_model, out: d_ffn },
             weight_axes: [{ world: dense_ffn, parts: d_ffn, fit: exact }, { replicated: true }] }
      fc2: { op: linear, in: fc1, dims: { in: d_ffn, out: d_model },
             weight_axes: [{ replicated: true }, { world: dense_ffn, parts: d_ffn, fit: exact }] }
    outputs: [fc2]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: layers, repeat: n_layers, template: blk, inputs: [embed],
      streams: { hidden: { carries: true } } }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: d_model }, inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: heads, repeat: n_deepstack, template: tap, inputs: [layers] }
"""


#: The backbone block also projects its own output, column-split: the exit
#: of every layer is read inside its iteration, and the last one by name.
_INNER_READER = (
    "      p: { op: linear, in: o, dims: { in: d_model, out: d_ffn },\n"
    "           weight_axes: [{ world: dense_ffn, parts: d_ffn, fit: exact }, "
    "{ replicated: true }] }\n"
    "      q: { op: add, in: [o, o], dim: d_model }\n")


def _named_read_model(*, inner_reader=False):
    import tempfile
    from backend.arch.loader import SpecLoader
    spec = _NAMED_READ_SPEC
    if inner_reader:
        spec = spec.replace(
            "      o: { op: add, in: [x, n], dim: d_model }\n    outputs: [o]\n",
            "      o: { op: add, in: [x, n], dim: d_model }\n" + _INNER_READER
            + "    outputs: [o]\n", 1)
        assert spec != _NAMED_READ_SPEC
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "named_read.yaml").write_text(spec)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": "named_read", "torch_dtype": "bfloat16", "hidden_size": 8,
             "intermediate_size": 16}, model_id="named_read")


class TestValueSettledPerProducerOccurrence(unittest.TestCase):
    def test_a_value_read_by_name_from_three_iterations_is_gathered_once(self):
        """The three projections read one value, the backbone's exit: it is
        gathered once, after its producer, however many readers read it."""
        from backend.arch.eval.transforms import Deployment
        analysis = own_walk(Deployment(
            Program.of(_named_read_model()),
            config=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                  sequence_parallel=True),
        ).at(batch=1, input_tokens=4), why="a program of this file's own, and one test here overwrites `analysis.input_references` to build a broken graph")
        fc1 = next(n for n in analysis.program.nodes() if n.name == "fc1")
        (record,) = [r for r in analysis.redistributes if r.node == fc1.uid]
        (event,) = [
            e for e in materialize_comm_events(analysis)
            if e.site == RedistributeSite.EDGE
            and e.source.node_uid == record.producer
        ]
        self.assertEqual((event.op, event.count), ("all_gather", 1))
        self.assertEqual((event.placement, event.placement_uid, event.rewrites),
                         (Placement.AFTER, record.producer, ()))

    @staticmethod
    def _sp(**extra):
        from backend.arch.eval.transforms import Deployment
        return own_walk(Deployment(
            Program.of(_named_read_model(**extra.pop("model", {}))),
            config=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                  sequence_parallel=True, **extra),
        ).at(batch=1, input_tokens=4), why="a program of this file's own, and one test here overwrites `analysis.input_references` to build a broken graph")

    @staticmethod
    def _gathers_of(events, producer):
        return [e for e in events if e.site == RedistributeSite.EDGE
                and e.op == "all_gather" and e.source.node_uid == producer]

    def test_a_pipeline_stage_counts_the_readers_it_runs(self):
        """Under two pipeline stages the projections all run on the last;
        that stage gathers the exit once, the first not at all, and the
        whole pipeline once."""
        from backend.arch.layout.materializer import collect_events
        from backend.arch.layout.stages import StagePlacement
        analysis = self._sp(pp=2)
        fc1 = next(n for n in analysis.program.nodes() if n.name == "fc1")
        (record,) = [r for r in analysis.redistributes if r.node == fc1.uid]
        whole = self._gathers_of(collect_events(analysis),
                                 record.producer)
        self.assertEqual([e.count for e in whole], [1])
        placement = StagePlacement.of(analysis)
        per_stage = [
            [e.count for e in self._gathers_of(
                collect_events(analysis, stage=placement.view(stage)),
                record.producer)]
            for stage in range(2)]
        self.assertEqual(per_stage, [[0], [1]])

    def test_readers_that_read_no_producer_occurrence_raise(self):
        """A reader on this view whose input resolves to no occurrence of
        its producer has nothing to settle; that is a broken graph, not a
        zero count."""
        analysis = self._sp()
        table = analysis.input_references
        analysis.input_references = ReferenceTable(
            program=table.program, arity=table.arity,
            references={key: Reference.fixed() for key in table.references})
        with self.assertRaisesRegex(ValueError, "no occurrence"):
            materialize_comm_events(analysis)

    def test_readers_inside_and_outside_the_iteration_share_one_count(self):
        """Every layer's exit is gathered for its own projection; the last
        one is also read by name. The name read adds no occurrence: four
        gathers, before the in-iteration reader, which is rewritten."""
        analysis = self._sp(model={"inner_reader": True})
        p = next(n for n in analysis.program.nodes() if n.name == "p")
        fc1 = next(n for n in analysis.program.nodes() if n.name == "fc1")
        (inner,) = [r for r in analysis.redistributes if r.node == p.uid]
        (outer,) = [r for r in analysis.redistributes if r.node == fc1.uid]
        self.assertEqual(inner.producer, outer.producer)
        (event,) = self._gathers_of(materialize_comm_events(analysis),
                                    inner.producer)
        self.assertEqual(event.count, 4)
        self.assertEqual((event.placement, event.placement_uid),
                         (Placement.BEFORE, p.uid))
        self.assertEqual(event.rewrites, (RewriteTarget(p.uid, 0),))


class TestExpertBankFinalize(unittest.TestCase):
    """Qwen3-30B-A3B under sequence parallelism gathers the routed rows
    before the bank's gate and up projections. The bank reads the whole
    `[n_expert_copies, M, d_model]` buffer once per layer, so the gather is
    one per layer of that buffer: 128 experts x 4 rows x 2048 x 2 bytes."""

    @classmethod
    def setUpClass(cls):
        from backend.arch.eval.transforms import Deployment
        cls.analysis = own_walk(Deployment(
            Program.of(_resolved("Qwen__Qwen3-30B-A3B.json")),
            config=ParallelConfig(tp_attention=4, tp_dense_ffn=4,
                                  tp_moe_expert=4, sequence_parallel=True),
        ).at(batch=1, input_tokens=64),
            why="its own resolution, not the shared layer's")
        program = cls.analysis.program
        cls.gate = next(n for n in program.nodes()
                        if n.display_path == "mlp.experts.gate_proj")
        cls.up = next(n for n in program.nodes()
                      if n.display_path == "mlp.experts.up_proj")
        (cls.record,) = [r for r in cls.analysis.redistributes
                         if r.node == cls.gate.uid]

    def test_the_bank_settles_once_per_layer(self):
        instances = int(value_of(
            self.analysis.instances.values[self.gate.uid]))
        self.assertEqual(instances, 48)
        (event,) = [e for e in materialize_comm_events(self.analysis)
                    if e.site == RedistributeSite.EDGE
                    and e.source.node_uid == self.record.producer]
        self.assertEqual((event.op, event.count), ("all_gather", instances))
        self.assertEqual(event.payload_bytes, 128 * 4 * 2048 * 2)

    def test_readers_pricing_one_value_differently_raise(self):
        from backend.arch.layout.materializer import _Session
        real = _Session._count

        def count(session, uid):
            return 2 * real(session, uid) if uid == self.up.uid \
                else real(session, uid)
        with patch.object(_Session, "_count", count), \
                self.assertRaisesRegex(ValueError, "different instances"):
            materialize_comm_events(self.analysis)


if __name__ == "__main__":
    unittest.main()
