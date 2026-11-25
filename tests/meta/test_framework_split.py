"""D1 read from the framework source that decides the split
(`tools/corpus_pins/decisions/framework_split.py`).

Every node's split is the class the framework assigns its module, through the
framework's packed modules and the node's nearest assigned ancestor, the
model's own files before the ones it imports; a stack is written only when
every node is settled and split as measured.

Two kinds of source say it and they are read differently. vLLM's layer
classes are not in the staged files, so a class name is resolved through
`VllmSplit.CLASSES`. A publisher's own reference implementation *defines* its
classes in the staged file, so `ReferenceSplit` reads what the class does
with its weight: the rank window it narrows to, the collective its forward
passes the result through, the extent it divides by the number of ranks, and
what its caller passes for an extent. A class it cannot read that way — a
plain `nn.Linear`, whose per-rank-ness lives in the caller's config and in no
definition — leaves the node a person's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.corpus_pins.decisions import tp_split
from tools.corpus_pins.decisions.framework_split import (
    MANIFEST, SOURCE_KINDS, FrameworkSplit, Oracle, ReferenceSplit, SglangSplit,
    VllmSplit)
from tools.corpus_pins.decisions.tp_split import FRAMEWORK, VERDICTS, framework_entry

# ── vLLM: the class name, through the table ─────────────────────────────

OWN = '''
class Attention(nn.Module):
    packed_modules_mapping = {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]}
    def __init__(self):
        self.fused_qkv_a_proj = MergedColumnParallelLinear(h, [q, kv], disable_tp=True)
        self.q_b_proj = ColumnParallelLinear(q, heads * d)
        self.o_proj = RowParallelLinear(heads * d, h)
        self.sink = nn.Parameter(torch.empty(self.num_local_heads))
        self.hc_scale = nn.Parameter(torch.empty(2))
        self.norm = RMSNorm(h) if first else PPMissingLayer()
        self.experts = FusedMoEFactory(n_experts, h, d)
        self.indexer = Indexer(h)
        self.lm_head = self.lm_head.tie_weights(self.embed_tokens)
'''
IMPORTED = '''
class Base(nn.Module):
    def __init__(self):
        self.q_b_proj = q_proj_cls(q, heads * d)
        self.lm_head = ParallelLMHead(vocab, h)
'''


def _staged(root: Path, files, manifest) -> FrameworkSplit:
    for rel, text in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / MANIFEST).write_text(json.dumps(manifest))
    return FrameworkSplit.read(root)


def _vllm(root: Path) -> FrameworkSplit:
    return _staged(root, (("vllm/models/m/own.py", OWN),
                          ("vllm/model_executor/models/base.py", IMPORTED)),
                   {"kind": "vllm", "revision": "abc",
                    "imported": ["vllm/model_executor/models/base.py"]})


def test_each_node_takes_the_class_vllm_assigns_it(tmp_path):
    v = _vllm(tmp_path)
    assert v.answer("blk|self_attn.o_proj").placement == "split"
    assert v.answer("blk|self_attn.o_proj").citation == "vllm/models/m/own.py:7 RowParallelLinear"
    assert v.answer("blk|hc_scale").placement == "replicated"
    assert v.answer("blk|self_attn.sink").placement == "split"  # sized by local heads
    assert v.answer("norm").placement == "replicated"  # the PP stub is no answer


def test_a_packed_module_answers_for_its_shards(tmp_path):
    assert _vllm(tmp_path).answer("blk|self_attn.q_a_proj").placement == "replicated"


def test_a_parameter_a_fused_module_owns_takes_that_module(tmp_path):
    assert _vllm(tmp_path).answer("blk|mlp.experts.gate_up_proj").placement == "split"


def test_the_models_own_files_answer_before_an_imported_one(tmp_path):
    v = _vllm(tmp_path)
    assert v.answer("blk|self_attn.q_b_proj").placement == "split"
    assert v.answer("lm_head").placement == "split"  # only the imported base assigns it


def test_a_module_the_table_does_not_know_is_no_answer(tmp_path):
    answer = _vllm(tmp_path).answer("blk|self_attn.indexer.wk")
    assert answer.placement is None and "Indexer" in answer.why


def test_a_stack_is_written_only_when_every_node_agrees(tmp_path):
    oracle = Oracle([_vllm(tmp_path)])
    rows = {"blk|self_attn.o_proj": {"placement": "split", "bytes": "split"},
            "blk|hc_scale": {"placement": "replicated", "bytes": "replicated"}}
    answers = {n: oracle.answer(n) for n in rows}
    entry = framework_entry(rows, answers, False, oracle)
    assert entry["verdict"] == "framework_split"
    assert entry["nodes"]["blk|hc_scale"] == {FRAMEWORK: "replicated", "decided_by": "vllm"}
    assert entry["citations"]["blk|self_attn.o_proj"].endswith("RowParallelLinear")
    assert "vLLM abc" in entry["source"]
    disagree = {**rows, "blk|hc_scale": {"placement": "split", "bytes": "split"}}
    assert framework_entry(disagree, answers, False, oracle) is None
    rows["blk|self_attn.indexer.wk"] = {"placement": "replicated", "bytes": "replicated"}
    assert framework_entry(rows, {n: oracle.answer(n) for n in rows}, False, oracle) is None


# ── a reference implementation: the class definition itself ─────────────
#
# The shapes below are the ones the publishers actually ship: Step-5's
# `inference/model.py` (ColumnParallelLinear / RowParallelLinear /
# VocabParallelLinear / the stacked expert bank, `self.rank = tp_rank()`) and
# DeepSeek's module-level `world_size`.

REFERENCE = '''
import torch
import torch.nn as nn
import torch.distributed as dist

_TP_GROUP = None
world_size = dist.get_world_size()


def tp_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(_TP_GROUP)


def tp_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank(_TP_GROUP)


def _local_size(total, tp):
    return total // tp


def _all_reduce(x):
    dist.broadcast(x, src=0)
    return x


def _all_gather(x):
    out = [torch.empty_like(x) for _ in range(tp_size())]
    dist.all_gather(out, x)
    return torch.cat(out, dim=-1)


class ColumnParallelLinear(nn.Module):
    """Output-sliced; all-gathers its output when asked to."""

    def __init__(self, in_features, out_features, tp_size=1, gather_output=False):
        self.weight = nn.Parameter(
            torch.empty((_local_size(out_features, tp_size), in_features)))
        self.gather_output = gather_output

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight)
        if self.gather_output:
            out = _all_gather(out)
        return out


class RowParallelLinear(nn.Module):
    """Input-sliced; all-reduces the partial sum."""

    def __init__(self, in_features, out_features, tp_size=1):
        self.weight = nn.Parameter(
            torch.empty((out_features, _local_size(in_features, tp_size))))

    def forward(self, x):
        return _all_reduce(torch.nn.functional.linear(x, self.weight))


class VocabParallelLinear(nn.Module):
    """Registers the whole vocabulary, then keeps this rank's window."""

    def __init__(self, in_features, out_features, tp_size=1):
        self._per_rank = _local_size(out_features, tp_size)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

    def shard_(self):
        rank = tp_rank()
        local = self._per_rank
        start = rank * local
        self.weight = nn.Parameter(self.weight.narrow(0, start, local).contiguous())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class WorldSizedLinear(nn.Module):
    """Divides by the world size it read at module level."""

    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(
            torch.empty((out_features // world_size, in_features)))


class RMSNorm(nn.Module):
    def __init__(self, hidden):
        self.weight = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return x * self.weight


class StackedExperts(nn.Module):
    """Keeps the full shape of whatever expert count it is given."""

    def __init__(self, n_local_experts, out_features, in_features):
        self.weight = nn.Parameter(
            torch.empty((n_local_experts, out_features, in_features)))


class Block(nn.Module):
    def __init__(self, config):
        self.rank = tp_rank()
        n_local = config.n_experts // config.tp
        self.experts_start = self.rank * n_local
        self.q_proj = ColumnParallelLinear(
            config.hidden, config.heads * config.head_dim, tp_size=config.tp)
        self.o_proj = RowParallelLinear(
            config.heads * config.head_dim, config.hidden, tp_size=config.tp)
        self.k_proj = nn.Linear(config.hidden, config.kv_groups_per_rank * config.head_dim)
        self.norm = RMSNorm(config.hidden)
        self.lm_head = VocabParallelLinear(config.hidden, config.vocab, tp_size=config.tp)
        self.gate_proj = StackedExperts(n_local, config.moe_inter, config.hidden)
        self.router_bias = nn.Parameter(torch.zeros(config.n_experts))
        self.embed_proj = WorldSizedLinear(config.hidden, config.hidden)
        self.register_buffer("ssmax_s", torch.zeros(config.heads))
'''


def _reference(root: Path) -> FrameworkSplit:
    return _staged(root, (("model.py", REFERENCE),),
                   {"kind": "reference", "revision": "snapshot sha256:dead",
                    "model_id": "pub/Model-1"})


def test_a_class_that_all_reduces_or_all_gathers_holds_a_slice(tmp_path):
    reference = _reference(tmp_path)
    assert reference.answer("blk|self_attn.o_proj").placement == "split"
    assert reference.answer("blk|self_attn.q_proj").placement == "split"
    assert reference.answer("blk|self_attn.o_proj").citation.startswith("model.py:")


def test_a_class_that_narrows_its_weight_at_its_rank_holds_a_slice(tmp_path):
    assert _reference(tmp_path).answer("lm_head").placement == "split"


def test_an_extent_divided_by_the_world_size_holds_a_slice(tmp_path):
    assert _reference(tmp_path).answer("embed_proj").placement == "split"


def test_an_extent_the_caller_passes_one_ranks_worth_for_holds_a_slice(tmp_path):
    # StackedExperts keeps whatever count it is handed; `Block` hands it
    # `n_experts // tp`, which is the count it pairs with its own rank
    assert _reference(tmp_path).answer("mlp.experts.gate_proj").placement == "split"


def test_a_class_that_cannot_know_the_world_holds_the_whole_tensor(tmp_path):
    reference = _reference(tmp_path)
    assert reference.answer("blk|input_layernorm.norm").placement == "replicated"
    assert reference.answer("blk|moe.router_bias").placement == "replicated"
    assert reference.answer("blk|self_attn.ssmax_s").placement == "replicated"


def test_a_class_the_source_does_not_define_is_no_answer(tmp_path):
    answer = _reference(tmp_path).answer("blk|self_attn.k_proj")
    assert answer.placement is None
    assert "Linear" in answer.why and "does not define" in answer.why


def test_a_node_no_assignment_reaches_is_no_answer(tmp_path):
    answer = _reference(tmp_path).answer("blk|self_attn.nothing_like_this")
    assert answer.placement is None and "no module on this path" in answer.why


def test_a_reference_entry_names_the_publishers_own_implementation(tmp_path):
    oracle = Oracle([_reference(tmp_path)])
    rows = {"blk|self_attn.o_proj": {"placement": "split", "bytes": "split"},
            "blk|moe.router_bias": {"placement": "replicated", "bytes": "replicated"}}
    entry = framework_entry(rows, {n: oracle.answer(n) for n in rows}, False, oracle)
    assert {row["decided_by"] for row in entry["nodes"].values()} == {"reference"}
    assert entry["verdict"] == "framework_split"
    assert "pub/Model-1's own implementation at snapshot sha256:dead" in entry["source"]


def test_a_source_that_is_not_a_tp_implementation_decides_nothing(tmp_path):
    """A publisher's `modeling_*.py` is written for one rank and says nothing
    about TP; reading "every rank holds the whole tensor" out of that silence
    would invent a fact."""
    single_rank = '''
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden):
        self.weight = nn.Parameter(torch.zeros(hidden))


class Block(nn.Module):
    def __init__(self, config):
        self.norm = RMSNorm(config.hidden)
        self.q_proj = nn.Linear(config.hidden, config.hidden)
'''
    reference = _staged(tmp_path, (("modeling_m.py", single_rank),),
                        {"kind": "reference", "revision": "r1", "model_id": "pub/M"})
    assert reference.decides is False
    answer = reference.answer("blk|norm")
    assert answer.placement is None
    assert "not a tensor-parallel implementation" in answer.why


# ── which source answers when ───────────────────────────────────────────

def test_vllm_answers_first_when_it_has_the_model(tmp_path):
    oracle = Oracle([_reference(tmp_path / "ref"), _vllm(tmp_path / "vllm")])
    assert [source.kind for source in oracle.sources] == ["vllm", "reference"]
    # vLLM's table settles `norm` as replicated; the reference never assigns
    # the name, so vLLM's answer stands and is recorded as vLLM's
    answer = oracle.answer("norm")
    assert answer.placement == "replicated" and answer.decided_by == "vllm"


def test_a_node_vllm_does_not_settle_takes_the_references_answer(tmp_path):
    """vLLM keeps its precedence where it speaks; where it has no answer at
    all, the publisher's own implementation is evidence about the same model
    and overrides nothing."""
    oracle = Oracle([_reference(tmp_path / "ref"), _vllm(tmp_path / "vllm")])
    answer = oracle.answer("embed_proj")  # vLLM's files never assign it
    assert answer.placement == "split" and answer.decided_by == "reference"


def test_the_reference_answers_when_vllm_has_nothing(tmp_path):
    oracle = Oracle([_reference(tmp_path)])
    answer = oracle.answer("embed_proj")
    assert answer.placement == "split" and answer.decided_by == "reference"


def test_a_node_no_source_settles_reports_every_reason(tmp_path):
    oracle = Oracle([_reference(tmp_path / "ref"), _vllm(tmp_path / "vllm")])
    answer = oracle.answer("blk|self_attn.k_proj")
    assert answer.placement is None
    assert "vllm:" in answer.why and "reference:" in answer.why


def test_two_sources_that_disagree_leave_the_node_to_a_person(tmp_path):
    # vLLM's table reads `q_b_proj` (an unknown factory in its own file, a
    # ParallelLMHead in the imported base) as split; a reference that assigns
    # the same name a plain norm reads it as replicated
    conflicting = '''
import torch
import torch.nn as nn
import torch.distributed as dist


def tp_rank():
    return dist.get_rank()


class RMSNorm(nn.Module):
    def __init__(self, hidden):
        self.weight = nn.Parameter(torch.zeros(hidden))


class Head(nn.Module):
    def __init__(self, hidden):
        self.rank = tp_rank()
        self.o_proj = RMSNorm(hidden)
'''
    reference = _staged(tmp_path / "ref", (("model.py", conflicting),),
                        {"kind": "reference", "revision": "r1", "model_id": "pub/M"})
    oracle = Oracle([_vllm(tmp_path / "vllm"), reference])
    answer = oracle.answer("blk|self_attn.o_proj")
    assert answer.placement is None
    assert "disagree" in answer.why
    assert "vllm says split" in answer.why and "reference says replicated" in answer.why


def test_every_kind_is_defined_once(tmp_path):
    assert list(SOURCE_KINDS) == [VllmSplit.kind, SglangSplit.kind,
                                 ReferenceSplit.kind]
    assert all(kind == impl.kind for kind, impl in SOURCE_KINDS.items())


def test_sglang_staging_preserves_framework_and_refuses_unknown_groups(tmp_path):
    source = _staged(tmp_path, (("python/sglang/srt/models/test.py", '''
class Model:
    def __init__(self):
        self.embed_tokens = VocabParallelEmbedding(vocab, hidden)
        self.lm_head = ParallelLMHead(vocab, hidden)
        self.norm = RMSNorm(hidden)
        self.output_attn_res_proj = ReplicatedLinear(hidden, 1)
        self.custom = ColumnParallelLinear(hidden, hidden, parallel_group=group)
        self.unknown = SomeExpertFactory(hidden)
'''),), {"kind": "sglang", "revision": "sglang-test-revision"})
    oracle = Oracle([source])
    for node in ("embed_tokens", "lm_head"):
        answer = oracle.answer(node)
        assert answer.placement == "split"
        assert answer.decided_by == "sglang"
    for node in ("norm", "output_attn_res_proj"):
        assert oracle.answer(node).placement == "replicated"
    for node in ("custom", "unknown"):
        assert oracle.answer(node).placement is None
    assert source.provenance == "SGLang sglang-test-revision"


# ── the migrated record ─────────────────────────────────────────────────

def test_every_recorded_entry_reads_in_the_framework_vocabulary():
    doc = tp_split.load()
    assert doc, "D1 has no recorded entries"
    for stem, stacks in doc.items():
        for stack, entry in stacks.items():
            where = f"{stem} {stack or '(backbone)'}"
            assert entry["verdict"] in VERDICTS, where
            assert entry.get("source"), where
            assert "decided_by" not in entry, f"{where}: who decided is a per-node fact"
            if entry.get("nodes") is None:
                continue
            for node, row in entry["nodes"].items():
                assert row[FRAMEWORK] in ("split", "replicated"), f"{where} {node}"
                assert row["decided_by"] in SOURCE_KINDS, f"{where} {node}"
                assert "vllm" not in row, f"{where} {node}"
                if "spec" in row:
                    assert row.get("reason"), f"{where} {node} departs with no reason"


def test_the_migrated_record_still_judges_its_models():
    """`judge_stack` reads the migrated entries: a stack whose nodes match
    the record has no problem, and a node registered under the old key would
    raise rather than pass silently."""
    entry = {"verdict": "framework_split", "encoder": False, "source": "vLLM abc",
             "nodes": {"a": {FRAMEWORK: "split", "decided_by": "vllm"},
                       "b": {FRAMEWORK: "replicated", "decided_by": "reference",
                             "spec": "split", "reason": "the spec splits it"}}}
    measured = {"a": {"placement": "split", "bytes": "split",
                      "tp1_bytes": 8, "tp8_bytes": 1},
                "b": {"placement": "split", "bytes": "split",
                      "tp1_bytes": 8, "tp8_bytes": 1}}
    assert tp_split.judge_stack(entry, measured) == []
    old = {**entry, "nodes": {"a": {"vllm": "split"}, "b": {"vllm": "split"}}}
    try:
        tp_split.judge_stack(old, measured)
    except KeyError as exc:
        assert FRAMEWORK in str(exc)
    else:
        raise AssertionError("an entry under the old key must not read as a verdict")
