"""A fused op's output port carries the tensor its name says, on every model.

A fusion plan names its outputs (`fused_qk_rope`'s `q_out` / `k_out`) after
pattern nodes. Nothing about the port name is checked by the numbers: the
fused kernel's cost is the sum of its members and its data flow follows the
real refs, so a port bound to the wrong member computes the same totals and
still shows the wrong name. The only thing that can catch it is a fact that
says which member is which, independent of the order the matcher met them.

That fact is the operand each member's output becomes (`OperandRoles`): the
attention score declares that it reads its query at position 0 and its keys
at position 1, and the plan declares which operand each rope `feeds`. For
every fusion across the corpus — every snapshot, both phases, TP1 and TP8,
both framework bindings — each output port must sit on the member bound to
the pattern node the port names, and a port whose node declares `feeds`
must sit on the one member that reaches that operand.

Parallel-form fusions (`qkv_parallel_linear`, …) declare their siblings
interchangeable and name their outputs by position only (`output_0..N`), so
the port name there claims no identity to check beyond that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.fusion import OperandRoles
from backend.arch.eval.launch import FusedNode
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_loader

SNAPSHOTS = sorted(iter_snapshot_ids())
FRAMEWORKS = ("vllm", "sglang")
MESHES = {"tp1": {}, "tp8": {"tp_attention": 8, "tp_dense_ffn": 8}}
PHASES = ("prefill", "decode")


def _port_name(io_name: str) -> str:
    """The name `FusedNode.output_ports` carries: `output` is the primary."""
    return "" if io_name == "output" else io_name


def _violations(result, framework: str) -> list:
    operands = OperandRoles.of(result.pre_program)
    out = []
    for node in result.program.nodes():
        if not isinstance(node, FusedNode):
            continue
        pattern = FUSION_SCHEMES[node.op].naive_pattern
        where = f"{framework} {node.op} @ {node.aggregation_id}"
        if pattern.is_parallel():
            expected = {f"output_{i}" for i in range(len(node.output_ports))}
            if set(node.output_ports.values()) != expected:
                out.append(f"{where}: positional ports "
                           f"{sorted(node.output_ports.values())}")
            continue
        declared = pattern.nodes
        members = {pattern_name: uid
                   for pattern_name, (uid, _port) in node.ports.items()}
        ported = {}
        for io_name, target in pattern.io_mapping.items():
            if target.startswith("external."):
                continue
            pattern_name = pattern.port_node(target.partition(".")[0], members)
            if pattern_name is not None:
                ported[io_name] = pattern_name
        for io_name, pattern_name in ported.items():
            uid = members[pattern_name]
            name = result.pre_program.by_uid(uid).name
            bound = node.output_ports.get(uid)
            # One member may be named by two io entries (a plan exposing a
            # tensor under two operand names); the port must be one of them.
            names_here = {_port_name(n) for n, p in ported.items()
                          if members[p] == uid}
            if bound not in names_here:
                out.append(f"{where}: {io_name!r} names {pattern_name} "
                           f"({name}) but that member carries port {bound!r}")
            feeds = declared[pattern_name].feeds
            if feeds is None:
                continue
            reaching = sorted(
                result.pre_program.by_uid(members[other]).name
                for other in ported.values()
                if feeds in operands.of_node(members[other]))
            if reaching != [name]:
                out.append(f"{where}: {io_name!r} declares it feeds "
                           f"{feeds!r}; members reaching {feeds!r}: "
                           f"{reaching}, port sits on {name}")
    return out


@pytest.mark.regression  # every snapshot
@pytest.mark.parametrize("model_id", SNAPSHOTS)
def test_every_fused_port_sits_on_the_member_its_name_declares(model_id):
    resolved = resolve_snapshot(model_id, shared_loader())
    problems = []
    for framework in FRAMEWORKS:
        binding = get_binding(framework)
        for mesh, parallel in MESHES.items():
            for phase in PHASES:
                result = analyze(
                    resolved, parallel=ParallelConfig.from_dict(parallel),
                    framework_binding=binding, batch_size=1, avg_input_tokens=16, avg_output_tokens=0,
                    # Inspect graph identities even when the selected loader
                    # refuses this deployment; loadability has its own gates.
                    phase=phase, breakdown=False, allow_load_refusals=True)
                problems += [f"{mesh} {phase} {p}"
                             for p in _violations(result, framework)]
    assert not problems, "\n".join(sorted(set(problems)))


def test_the_rope_pair_is_told_apart_by_what_it_feeds():
    """DeepSeek-V3's MLA ropes: a head-shared `rope_emb` for K beside a
    per-head `rope` for Q. `q_out` is `q_pe_rope`, `k_out` is `k_pe_rope`."""
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V3", shared_loader())
    result = analyze(resolved, parallel=ParallelConfig.from_dict({}),
                     framework_binding=get_binding("vllm"),
                     batch_size=1, avg_input_tokens=16, avg_output_tokens=0, phase="decode", breakdown=False)
    pairs = {
        tuple(sorted((port, result.pre_program.by_uid(uid).name)
                     for uid, port in node.output_ports.items()))
        for node in result.program.nodes()
        if isinstance(node, FusedNode) and node.op == "fused_qk_rope"
    }
    assert pairs == {(("k_out", "k_pe_rope"), ("q_out", "q_pe_rope"))}


def test_two_ropes_that_each_reach_q_and_k_are_refused():
    """`feeds` breaks the pattern's symmetry, not the graph's. Two ropes that
    each feed one score as the query and another as the key both satisfy
    `feeds: q` and `feeds: k`, so neither says which port it is. Picking the
    first would name the ports by match order; the matcher refuses instead —
    by binding neither, since a candidate whose operand roles also hold a role
    another pattern node feeds is no answer for either slot. (It used to get
    as far as two bindings of one member set and raise; the rule that rejects
    such a candidate up front now stops it earlier. Same refusal, no match.)"""
    from types import SimpleNamespace

    from backend.arch.eval.fusion import Fusion
    from backend.arch.ops import OP_REGISTRY

    def node(uid, name, op, inputs=(), declaration=None):
        return SimpleNamespace(
            uid=uid, name=name, op=op, children=(), template="",
            declaration=declaration,
            inputs_obj=tuple(SimpleNamespace(src=src, port="", external_sid=None)
                             for src in inputs))

    score = OP_REGISTRY["attention_score"]
    a_src, b_src = node(1, "a_proj", "linear"), node(2, "b_proj", "linear")
    a_rope, b_rope = node(3, "a_rope", "rope", [a_src]), node(4, "b_rope", "rope", [b_src])
    nodes = [a_src, b_src, a_rope, b_rope,
             node(5, "score_ab", "attention_score", [a_rope, b_rope], score),
             node(6, "score_ba", "attention_score", [b_rope, a_rope], score)]
    program = SimpleNamespace(nodes=lambda: iter(nodes),
                              walked_nodes=lambda: iter(nodes),
                              ir=SimpleNamespace(next_uid=len(nodes) + 1))

    operands = OperandRoles.of(program)
    assert operands.of_node(3) == operands.of_node(4) == {"q", "k"}
    # no dispatch condition: the walk's dims are not consulted
    fused, rewrite = Fusion.named("fused_qk_rope", FUSION_SCHEMES).rewrite(
        program, None)
    assert rewrite.origins == {}
    assert [n.uid for n in fused.nodes()] == [n.uid for n in nodes]
