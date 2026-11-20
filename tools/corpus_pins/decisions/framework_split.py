"""D1 checked against the framework source that decides the split.

D1 (`tp_split.py`) records how an external framework splits each weight node
a model brings, the external fact the measured placement is held to. A
framework loads a checkpoint by module attribute name, so a node's leaf name
is the attribute that framework's model assigns it (`self.o_proj =
RowParallelLinear(...)`), and the assignment says whether a TP rank holds a
slice or the whole tensor. Three source kinds can say it; they differ in
how an assigned class becomes a split:

  * **vLLM** — the deployment target the rest of the corpus is held to. Its
    layer classes live in vLLM, not in the files the admitter stages, so a
    class name is resolved through `VllmSplit.CLASSES`: the one place those
    classes' TP behaviour is written down for the corpus.
  * **SGLang** — an explicitly SGLang-sourced snapshot stages its frozen
    Python files without probing vLLM. `SglangSplit.CLASSES` describes its
    audited TP layer classes; custom parallel groups remain unsettled.
  * **the model's own reference implementation** — what a publisher ships
    beside the checkpoint, and what the spec was written from (DeepSeek, Kimi
    and StepFun all ship one; `backend/hf/forward_source.py` is where
    HuggingArch already reads it). Here the class is *defined* in the staged
    file, so `ReferenceSplit` reads what the class does with its weight
    rather than trusting that it is spelled like vLLM's.

A node whose leaf the source loads into a packed module (vLLM's
`packed_modules_mapping`: `q_a_proj` into `fused_qkv_a_proj`) takes the packed
module's; one the source never assigns (a parameter a fused module owns, such
as `experts.gate_up_proj` inside vLLM's `FusedMoE`) takes its nearest
assigned ancestor. The model's own files answer first; a module it imports
from elsewhere answers only a name the own files never assign. A node the
source does not settle — no assignment on its path, a class it cannot read,
or two assignments that disagree — has no answer and stays a person's to
write.

Which source answers when: `SOURCE_KINDS` lists the kinds in the order they
answer, and is the one place they are defined. vLLM comes first, because it
is the default deployment target; explicitly staged SGLang is next, and the
publisher's own implementation comes after it. Only staged sources participate:
an explicitly SGLang-sourced snapshot does not stage vLLM at all. A node's answer
is the first source in that order that settles it, and `decided_by` records which one did. That keeps
vLLM's precedence where vLLM speaks: a node it does not settle has no vLLM
answer at all, so asking the publisher's implementation — evidence about the
same model — only adds evidence and overrides nothing. A node two sources
settle *differently* is nobody's: `Oracle.answer` hands back both answers for
a person to read rather than letting the earlier one quietly win.

A staged source that never reaches `torch.distributed` decides no split and
answers nothing. That is the difference between a publisher's serving
implementation and its `modeling_*.py`: the first builds a rank's slice and
says so, the second is written for one rank and is silent about TP, and
reading "this class holds the whole tensor" out of silence would invent a
fact. `ReferenceSplit.decides` is that gate.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tools.corpus_pins.decisions.tp_split import REPLICATED, SPLIT

#: the file the admitter writes beside a staged source: which kind of source
#: it is, the revision it was read at, and what was read
MANIFEST = "MANIFEST.json"

#: `torch.distributed`'s surface, as far as reading a layer definition needs
#: it: who asks how many ranks there are, who asks which rank this is, and
#: which calls move a tensor between ranks. Every other attribute of the
#: module is contact without one of those meanings (`is_initialized`,
#: `new_group`). No other declaration in this repository covers that API, so
#: this is its definition here.
TOUCH, WORLD, RANK, MOVES = "touch", "world", "rank", "moves"
DISTRIBUTED = {
    "get_world_size": WORLD,
    "get_rank": RANK,
    "all_reduce": MOVES, "all_gather": MOVES, "all_gather_into_tensor": MOVES,
    "all_gather_single": MOVES, "all_gather_object": MOVES,
    "all_to_all": MOVES, "all_to_all_single": MOVES,
    "reduce": MOVES, "reduce_scatter": MOVES, "reduce_scatter_tensor": MOVES,
    "broadcast": MOVES, "gather": MOVES, "scatter": MOVES,
}

#: torch's two forms for "a tensor this module owns", as opposed to a layer
#: class it builds: the extents are then the module's own expressions
REGISTRATIONS = ("Parameter", "register_buffer")

#: what a rank that holds no weight there assigns instead (another pipeline
#: stage's layer, a disabled module): not an answer
_STUBS = ("PPMissingLayer", "Identity")
#: how far a name is followed to the expression that defines it
_DEPTH = 12


@dataclass(frozen=True)
class Assignment:
    """`self.<attr> = <cls>(...)` at `path:line` in a framework's source."""

    path: str
    line: int
    cls: str
    placement: Optional[str]

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line} {self.cls}"


@dataclass(frozen=True)
class NodeAnswer:
    """The split of one node, with the assignment that says so and — once
    `Oracle` has asked in order — which source said it; `placement` is None
    when no source settles it (`why`)."""

    placement: Optional[str]
    citation: Optional[str] = None
    why: Optional[str] = None
    decided_by: Optional[str] = None


class FrameworkSplit:
    """The staged files of one framework source, indexed by attribute name."""

    #: the name `MANIFEST.json` and the written entry's `decided_by` use
    kind = ""

    def __init__(self, manifest: Dict) -> None:
        self.manifest = manifest
        self.revision = str(manifest["revision"])
        #: files only an import brought; they answer what the own files do not
        self.imported = frozenset(manifest.get("imported") or ())
        #: a checkpoint module's name -> the module it is packed into
        self.packed: Dict[str, str] = {}
        self.assignments: Dict[str, List[Assignment]] = {}

    # ── reading the staged files ─────────────────────────────────────────
    @classmethod
    def read(cls, root: Path) -> Optional["FrameworkSplit"]:
        """The source the admitter staged under `root` (paths as in the
        source it came from, `MANIFEST.json` beside them), or None when none
        was staged."""
        manifest_file = Path(root) / MANIFEST
        if not manifest_file.is_file():
            return None
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        source = SOURCE_KINDS[manifest["kind"]](manifest)
        source.load([(path.relative_to(root).as_posix(), path)
                     for path in sorted(Path(root).rglob("*.py"))])
        return source

    def load(self, files: Sequence[Tuple[str, Path]]) -> None:
        """Index every assignment the staged files make, with this kind of
        source's reading of each."""
        trees: List[Tuple[str, ast.Module]] = []
        for rel, path in files:
            try:
                trees.append((rel, ast.parse(path.read_text(encoding="utf-8"))))
            except (SyntaxError, OSError):
                continue
        self.index(trees)
        for rel, tree in trees:
            self.packed.update(_packing(tree))
            for owner, assign in _self_assignments(tree):
                for call in _calls(assign.value):
                    if _on_self(call.func):
                        continue  # `self.lm_head.tie_weights(…)`: a method of a module it has
                    name = _called(call)
                    if name in _STUBS:
                        continue
                    for target in assign.targets:
                        if _is_self_attribute(target):
                            self.assignments.setdefault(target.attr, []).append(
                                Assignment(rel, assign.lineno, name,
                                           self.placement(name, call, owner)))
            for owner, attr, call in self.declared(tree):
                self.assignments.setdefault(attr, []).append(
                    Assignment(rel, call.lineno, "register_buffer",
                               self.placement("register_buffer", call, owner)))

    def index(self, trees: Sequence[Tuple[str, ast.Module]]) -> None:
        """Whatever this kind of source needs before it can place a class."""

    def declared(self, tree: ast.Module
                 ) -> Iterable[Tuple[Optional[ast.ClassDef], str, ast.Call]]:
        """Tensors the source declares other than by `self.<attr> = …`.
        None by default: a source read through a class-name table has nothing
        to say about a shape."""
        return ()

    def placement(self, cls_name: str, call: ast.Call,
                  owner: Optional[ast.ClassDef]) -> Optional[str]:
        """How a TP rank holds what this assignment registers, or None when
        this source does not settle it."""
        raise NotImplementedError

    def unsettled(self, classes: Sequence[str]) -> str:
        """Why this source could not place them."""
        raise NotImplementedError

    @property
    def provenance(self) -> str:
        """Who decided, as a written entry's `source` names them."""
        raise NotImplementedError

    # ── answering a node ─────────────────────────────────────────────────
    def answer(self, node: str) -> NodeAnswer:
        """This source's split of a D1 node (`<template>|<dotted path>` or a
        root entry's name): its leaf's assignment, else its nearest
        ancestor's."""
        path = node.split("|")[-1].split(".")
        for attr in reversed(path):
            hits = self.assignments.get(attr) or self.assignments.get(self.packed.get(attr, ""))
            if not hits:
                continue
            own = [a for a in hits if a.path not in self.imported]
            hits = own or hits
            placements = {a.placement for a in hits}
            if None in placements:
                unknown = sorted({a.cls for a in hits if a.placement is None})
                return NodeAnswer(None, why=f"`self.{attr}` is a {', '.join(unknown)}, "
                                            + self.unsettled(unknown))
            if len(placements) > 1:
                return NodeAnswer(None, why=f"`self.{attr}` is assigned both split and "
                                            "replicated: "
                                            + "; ".join(a.citation for a in hits))
            return NodeAnswer(placements.pop(), citation=hits[0].citation)
        return NodeAnswer(None, why=f"{self.kind} assigns no module on this path ("
                                    + ".".join(path) + ")")


class VllmSplit(FrameworkSplit):
    """vLLM's files for one model.

    vLLM's layer classes are not among the staged files — they live in vLLM —
    so a class name is resolved through `CLASSES`, which is the one place
    their TP behaviour is written down for the corpus.
    """

    kind = "vllm"

    #: vLLM layer classes and how a TP rank holds their weight; a class named
    #: `FusedMoE…` / `SharedFusedMoE…` (the expert bank and its factories)
    #: splits its experts over TP / EP
    CLASSES: ClassVar[Dict[str, str]] = {
        "ColumnParallelLinear": SPLIT, "RowParallelLinear": SPLIT,
        "MergedColumnParallelLinear": SPLIT, "QKVParallelLinear": SPLIT,
        "VocabParallelEmbedding": SPLIT, "ParallelLMHead": SPLIT,
        "ReplicatedLinear": REPLICATED, "Linear": REPLICATED, "Embedding": REPLICATED,
        "RMSNorm": REPLICATED, "GemmaRMSNorm": REPLICATED, "LayerNorm": REPLICATED,
        # vLLM's input-embedding factory: a VocabParallelEmbedding, replicated
        # only under the VLLM_REPLICATE_EMBED switch
        "make_input_embedding": SPLIT,
    }
    _EXPERT_BANKS = ("FusedMoE", "SharedFusedMoE")

    @property
    def provenance(self) -> str:
        return f"vLLM {self.revision}"

    def unsettled(self, classes: Sequence[str]) -> str:
        return "whose TP split this table does not know"

    def placement(self, cls_name: str, call: ast.Call,
                  owner: Optional[ast.ClassDef]) -> Optional[str]:
        if any(k.arg == "disable_tp" and isinstance(k.value, ast.Constant) and k.value.value
               for k in call.keywords):
            return REPLICATED
        if cls_name == "Parameter":
            # A parameter is the rank's own: sized by a TP-local count, a slice
            args = ast.unparse(call)
            return SPLIT if ("local" in args or "tp_size" in args) else REPLICATED
        if cls_name.startswith(self._EXPERT_BANKS):
            return SPLIT
        return self.CLASSES.get(cls_name)


class SglangSplit(FrameworkSplit):
    """SGLang's explicitly named TP layers (default parallel group only).

    Audited at 1b2870d33de7b5f31844fd74e97eea6d7896c53a:
    python/sglang/srt/layers/linear.py, vocab_parallel_embedding.py,
    layernorm.py. Unknown factories and custom groups remain unsettled;
    a class name alone cannot determine their rank-local ownership.
    """

    kind = "sglang"
    CLASSES: ClassVar[Dict[str, str]] = {
        "ColumnParallelLinear": SPLIT, "RowParallelLinear": SPLIT,
        "MergedColumnParallelLinear": SPLIT, "QKVParallelLinear": SPLIT,
        "VocabParallelEmbedding": SPLIT, "ParallelLMHead": SPLIT,
        "ReplicatedLinear": REPLICATED, "RMSNorm": REPLICATED,
        "LayerNorm": REPLICATED,
    }

    @property
    def provenance(self) -> str:
        return f"SGLang {self.revision}"

    def unsettled(self, classes: Sequence[str]) -> str:
        return "whose SGLang TP split is not declared for the default group"

    def placement(self, cls_name: str, call: ast.Call,
                  owner: Optional[ast.ClassDef]) -> Optional[str]:
        if any(k.arg in ("parallel_group", "tp_size", "tp_rank")
               for k in call.keywords):
            return None
        return self.CLASSES.get(cls_name)


@dataclass
class _Class:
    """One class of a reference implementation, as reading its weight needs
    it: where its names get their values, the extents of the tensors it
    registers, and the products it writes (one of which may be a rank
    window)."""

    name: str
    #: `self.<attr>` -> every value assigned to it, in any method
    attributes: Dict[str, List[ast.expr]] = field(default_factory=dict)
    #: a plain local name -> every value assigned to it, in any method
    names: Dict[str, List[ast.expr]] = field(default_factory=dict)
    #: the extents of every tensor the class registers, by attribute
    extents: Dict[str, List[ast.expr]] = field(default_factory=dict)
    #: `__init__`'s parameters: all of them, and the ones it takes by position
    parameters: List[str] = field(default_factory=list)
    positional: List[str] = field(default_factory=list)
    methods: Dict[str, ast.FunctionDef] = field(default_factory=dict)
    #: every product the class writes; `ReferenceSplit._windows` keeps the
    #: ones whose other side is the rank it runs on
    products: List[ast.BinOp] = field(default_factory=list)
    #: unparsed expressions the class pairs with its rank, once computed
    windows: Optional[Set[str]] = None


class ReferenceSplit(FrameworkSplit):
    """The model's own reference implementation.

    The class a node is assigned is defined in the staged file, so what a
    rank holds is read from that definition, in this order:

      1. the class registers the weight from an expression that involves the
         rank it runs on (`self.weight = nn.Parameter(self.weight.narrow(0,
         rank * local, local))`) — a rank window, so a slice;
      2. the class, or one of the source's helpers it calls, moves a tensor
         between ranks: an all-reduce of what it computed from the weight
         means the contraction it summed was a slice, an all-gather of its
         output means its rows were — so a slice;
      3. an extent of a tensor it registers divides a feature count by the
         number of ranks (`out_features // get_world_size()`, directly or
         through one of the source's helpers) — a slice;
      4. an extent is one of its own constructor parameters, and the class
         that constructs it passes one rank's worth there
         (`_StackedExpertWeight(n_experts // tp, …)`) — a slice;
      5. the class and every helper it calls never touch `torch.distributed`
         and no extent is per-rank: it cannot know the world it runs in, so
         every rank holds the whole tensor — replicated.

    Anything else has no answer: a class the staged source does not define
    (`nn.Linear`, whose per-rank-ness lives in the caller's config and in no
    definition at all), or one that does reach `torch.distributed` without
    any of 1-4 being visible.

    A node that is not a layer class at all but a tensor the module owns
    (`nn.Parameter`, `register_buffer`) is read from its extents alone: they
    are written in the owning class's own names, so 1 and 3 answer it and
    there is nothing else to ask. A router bias of the full expert count in a
    class that does know its rank is still replicated — the extent is the
    whole count.

    All of that rests on the staged source being a tensor-parallel
    implementation at all: reading 5 out of a file that never mentions
    `torch.distributed` would turn "written for one rank" into "every rank
    holds the whole tensor", which the file never says. `decides` is that
    precondition, and a source without it answers nothing.
    """

    kind = "reference"

    @property
    def provenance(self) -> str:
        return (f"{self.manifest.get('model_id') or 'the model'}'s own implementation "
                f"at {self.revision}")

    def unsettled(self, classes: Sequence[str]) -> str:
        undefined = [c for c in classes if c not in self.classes and c not in REGISTRATIONS]
        if undefined:
            return "which this source does not define"
        return "whose definition here does not say which dim a rank holds"

    def index(self, trees: Sequence[Tuple[str, ast.Module]]) -> None:
        #: names bound to `torch.distributed` by an import
        self.aliases: Set[str] = set()
        #: whether the staged source reaches `torch.distributed` anywhere: a
        #: source that never does is not a tensor-parallel implementation and
        #: decides no split
        self.decides = False
        #: module-level functions and assignments, by name
        self.helpers: Dict[str, ast.FunctionDef] = {}
        self.globals: Dict[str, List[ast.expr]] = {}
        self.classes: Dict[str, _Class] = {}
        #: `_reaches` per class, computed once
        self._reached: Dict[str, Set[str]] = {}
        for _rel, tree in trees:
            self.aliases |= _dist_aliases(tree)
            self.decides = self.decides or _names_distributed(tree)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.helpers.setdefault(node.name, node)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.globals.setdefault(target.id, []).append(node.value)
                elif isinstance(node, ast.ClassDef):
                    self.classes.setdefault(node.name, _read_class(node))

    def declared(self, tree: ast.Module
                 ) -> Iterable[Tuple[Optional[ast.ClassDef], str, ast.Call]]:
        return _registered_buffers(tree)

    def answer(self, node: str) -> NodeAnswer:
        if not self.decides:
            return NodeAnswer(None, why=(
                "this source never reaches torch.distributed, so it is not a "
                "tensor-parallel implementation and decides no split"))
        return super().answer(node)

    def placement(self, cls_name: str, call: ast.Call,
                  owner: Optional[ast.ClassDef]) -> Optional[str]:
        caller = self.classes.get(owner.name) if owner is not None else None
        if cls_name in REGISTRATIONS:
            # a tensor the class owns: its extents are the class's own
            extents = _extents(call)
            if caller is None or not extents:
                return None
            if self._involves(call, RANK, caller) or any(
                    self._per_rank(e, caller) for e in extents):
                return SPLIT  # 1 / 3
            return REPLICATED  # 5
        defined = self.classes.get(cls_name)
        if defined is None:
            return None  # a class this source does not define
        if caller is not None and self._involves(call, RANK, caller):
            return SPLIT  # 1
        if MOVES in self._reaches(defined):
            return SPLIT  # 2
        if any(self._per_rank(e, defined)
               for extents in defined.extents.values() for e in extents):
            return SPLIT  # 3
        if caller is not None and any(
                self._per_rank(arg, caller)
                for arg in _bound(call, defined, _extent_parameters(defined))):
            return SPLIT  # 4
        if not self._reaches(defined):
            return REPLICATED  # 5
        return None

    # ── what the definitions say ─────────────────────────────────────────
    def _reaches(self, defined: _Class) -> Set[str]:
        """The `torch.distributed` meanings the class's own methods reach,
        through the source's helpers they call. Constructing a submodule is
        not calling into it, so a class is not made rank-aware by the layers
        it builds."""
        if defined.name not in self._reached:
            self._reached[defined.name] = set()  # while computing: a cycle reaches nothing
            self._reached[defined.name] = self._called_meanings(
                list(defined.methods.values()), defined, set())
        return self._reached[defined.name]

    def _called_meanings(self, bodies: Sequence[ast.AST], defined: Optional[_Class],
                         seen: Set[str]) -> Set[str]:
        found: Set[str] = set()
        for body in bodies:
            for node in ast.walk(body):
                if not isinstance(node, ast.Call):
                    continue
                meaning = self._distributed(node.func)
                if meaning:
                    found.add(meaning)
                found |= self._into(node.func, defined, seen)
        return found

    def _into(self, func: ast.AST, defined: Optional[_Class], seen: Set[str]) -> Set[str]:
        """The meanings a call into the source's own code reaches."""
        if isinstance(func, ast.Name) and func.id in self.helpers and func.id not in seen:
            return self._called_meanings([self.helpers[func.id]], defined, seen | {func.id})
        key = f"{defined.name}.{getattr(func, 'attr', '')}" if defined else ""
        if (defined is not None and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == "self"
                and func.attr in defined.methods and key not in seen):
            return self._called_meanings([defined.methods[func.attr]], defined, seen | {key})
        return set()

    def _distributed(self, func: ast.AST) -> Optional[str]:
        """What a call on `torch.distributed` means, or None when the call is
        not on it."""
        if not isinstance(func, ast.Attribute):
            return None
        base = func.value
        if isinstance(base, ast.Name) and base.id in self.aliases:
            return DISTRIBUTED.get(func.attr, TOUCH)
        if (isinstance(base, ast.Attribute) and base.attr == "distributed"
                and isinstance(base.value, ast.Name) and base.value.id == "torch"):
            return DISTRIBUTED.get(func.attr, TOUCH)
        return None

    def _windows(self, scope: _Class) -> Set[str]:
        """What the class pairs with the rank it runs on: `self.rank *
        n_local` says `n_local` is one rank's worth."""
        if scope.windows is None:
            scope.windows = set()
            for product in scope.products:
                for side, other in ((product.left, product.right),
                                    (product.right, product.left)):
                    if self._involves(side, RANK, scope):
                        scope.windows.add(ast.unparse(other))
        return scope.windows

    def _per_rank(self, expr: Optional[ast.expr], scope: _Class, depth: int = _DEPTH,
                  seen: Tuple[int, ...] = ()) -> bool:
        """Whether an extent is one rank's worth: a feature count divided by
        the number of ranks, or a value the class pairs with its rank."""
        if expr is None or depth <= 0 or id(expr) in seen:
            return False
        seen = (*seen, id(expr))
        if ast.unparse(expr) in self._windows(scope):
            return True
        if (isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.FloorDiv, ast.Div))
                and self._involves(expr.right, WORLD, scope, depth - 1, seen)):
            return True
        if isinstance(expr, ast.Call) and any(
                self._per_rank(value, scope, depth - 1, seen)
                for value in self._returns(expr, scope)):
            return True
        if any(self._per_rank(value, scope, depth - 1, seen)
               for value in self._definitions(expr, scope)):
            return True
        return any(self._per_rank(child, scope, depth - 1, seen)
                   for child in ast.iter_child_nodes(expr) if isinstance(child, ast.expr))

    def _involves(self, expr: Optional[ast.AST], meaning: str, scope: Optional[_Class],
                  depth: int = _DEPTH, seen: Tuple[int, ...] = ()) -> bool:
        """Whether an expression asks `torch.distributed` one of its two
        questions about the world — how many ranks, or which one is this."""
        if expr is None or depth <= 0 or id(expr) in seen:
            return False
        seen = (*seen, id(expr))
        if isinstance(expr, ast.Call):
            if self._distributed(expr.func) == meaning:
                return True
            if any(self._involves(value, meaning, scope, depth - 1, seen)
                   for value in self._returns(expr, scope)):
                return True
        if any(self._involves(value, meaning, scope, depth - 1, seen)
               for value in self._definitions(expr, scope)):
            return True
        return any(self._involves(child, meaning, scope, depth - 1, seen)
                   for child in ast.iter_child_nodes(expr) if isinstance(child, ast.expr))

    def _returns(self, call: ast.Call, scope: Optional[_Class]) -> List[ast.expr]:
        """What a call into one of the source's own helpers evaluates to."""
        func = call.func
        name = func.id if isinstance(func, ast.Name) else None
        body: Optional[ast.FunctionDef] = self.helpers.get(name) if name else None
        if (body is None and scope is not None and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == "self"):
            body = scope.methods.get(func.attr)
        if body is None:
            return []
        return [node.value for node in ast.walk(body)
                if isinstance(node, ast.Return) and node.value is not None]

    def _definitions(self, expr: ast.AST, scope: Optional[_Class]) -> List[ast.expr]:
        """Where a name gets its value: the class's own assignments, else the
        module's."""
        if isinstance(expr, ast.Name):
            if scope is not None and expr.id in scope.names:
                return scope.names[expr.id]
            return self.globals.get(expr.id, [])
        if scope is not None and _is_self_attribute(expr):
            return scope.attributes.get(expr.attr, [])
        return []


#: the framework sources that decide a split, in the order they answer: vLLM
#: first, because it is the deployment target the corpus is held to; the
#: publisher's own implementation when vLLM has nothing for the architecture.
#: This is the one place the kinds are defined — a staged manifest's `kind`,
#: a written entry's `decided_by`, and the staging in `tools/corpus_admit.py`
#: all name them from here.
SOURCE_KINDS: Dict[str, type] = {
    kind.kind: kind for kind in (VllmSplit, SglangSplit, ReferenceSplit)}


class Oracle:
    """The framework sources staged for one model, in the order they answer."""

    def __init__(self, sources: Sequence[FrameworkSplit]) -> None:
        order = list(SOURCE_KINDS)
        self.sources = tuple(sorted(sources, key=lambda source: order.index(source.kind)))

    def __bool__(self) -> bool:
        return bool(self.sources)

    @property
    def staged(self) -> List[Dict[str, str]]:
        """What was asked, in the order it was asked."""
        return [{"kind": source.kind, "revision": source.revision}
                for source in self.sources]

    def source(self, kinds: Sequence[str]) -> str:
        """The `source` line a written entry carries: the provenance of the
        sources that decided one of its nodes."""
        return ("; ".join(source.provenance for source in self.sources
                          if source.kind in kinds)
                + ": each node's assignment, read by "
                  "tools/corpus_pins/decisions/framework_split.py")

    def answer(self, node: str) -> NodeAnswer:
        """The first source that settles the node, in the order they answer,
        with the kind that settled it. A node two sources settle differently
        is nobody's to resolve, so it comes back with every answer for a
        person to read."""
        answers = [(source, source.answer(node)) for source in self.sources]
        settled = [(source, answer) for source, answer in answers
                   if answer.placement is not None]
        if not settled:
            return NodeAnswer(None, why="; ".join(
                f"{source.kind}: {answer.why}" for source, answer in answers))
        placements = {answer.placement for _source, answer in settled}
        if len(placements) > 1:
            return NodeAnswer(None, why="the sources disagree: " + "; ".join(
                f"{source.kind} says {answer.placement} ({answer.citation})"
                for source, answer in settled))
        source, answer = settled[0]
        return NodeAnswer(answer.placement, citation=answer.citation,
                          decided_by=source.kind)


# ── reading the source text ─────────────────────────────────────────────

def _packing(tree: ast.Module) -> Dict[str, str]:
    """`{"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]}` as
    `{shard: packed}`."""
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "packed_modules_mapping"
                        for t in node.targets)):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(value, (ast.List, ast.Tuple)):
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out[elt.value] = str(key.value)
    return out


def _within_classes(tree: ast.AST, want: type) -> List[Tuple[Optional[ast.ClassDef], ast.AST]]:
    """Every node of `want` in a tree, with the class it is written in."""
    out: List[Tuple[Optional[ast.ClassDef], ast.AST]] = []

    def walk(node: ast.AST, owner: Optional[ast.ClassDef]) -> None:
        for child in ast.iter_child_nodes(node):
            here = child if isinstance(child, ast.ClassDef) else owner
            if isinstance(child, want):
                out.append((owner, child))
            walk(child, here)

    walk(tree, tree if isinstance(tree, ast.ClassDef) else None)
    return out


def _self_assignments(tree: ast.AST) -> List[Tuple[Optional[ast.ClassDef], ast.Assign]]:
    """Every `self.<attr> = …`, with the class it is written in."""
    return [(owner, node) for owner, node in _within_classes(tree, ast.Assign)
            if any(_is_self_attribute(t) for t in node.targets)]


def _registered_buffers(tree: ast.AST
                        ) -> List[Tuple[Optional[ast.ClassDef], str, ast.Call]]:
    """Every `self.register_buffer("<name>", <tensor>)`: a checkpoint entry a
    class declares without assigning an attribute."""
    out = []
    for owner, node in _within_classes(tree, ast.Call):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "register_buffer"
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and isinstance(node.args[1], ast.Call)):
            out.append((owner, node.args[0].value, node.args[1]))
    return out


def _read_class(node: ast.ClassDef) -> _Class:
    """One class, as `ReferenceSplit` reads it."""
    found = _Class(name=node.name)
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.methods[item.name] = item
            if item.name == "__init__":
                args = item.args
                found.positional = [a.arg for a in (*args.posonlyargs, *args.args)
                                    if a.arg != "self"]
                found.parameters = [*found.positional, *(a.arg for a in args.kwonlyargs)]
    for _owner, assign in _within_classes(node, ast.Assign):
        for target in assign.targets:
            if _is_self_attribute(target):
                found.attributes.setdefault(target.attr, []).append(assign.value)
                for call in _calls(assign.value):
                    if _called(call) == "Parameter":
                        found.extents.setdefault(target.attr, []).extend(_extents(call))
            elif isinstance(target, ast.Name):
                found.names.setdefault(target.id, []).append(assign.value)
    for _owner, attr, call in _registered_buffers(node):
        found.extents.setdefault(attr, []).extend(_extents(call))
    found.products = [sub for sub in ast.walk(node)
                      if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult)]
    return found


def _extent_parameters(defined: _Class) -> Set[str]:
    """The constructor parameters the class writes a tensor extent in."""
    out: Set[str] = set()
    for extents in defined.extents.values():
        for extent in extents:
            out |= {node.id for node in ast.walk(extent)
                    if isinstance(node, ast.Name) and node.id in defined.parameters}
    return out


def _bound(call: ast.Call, defined: _Class, wanted: Set[str]) -> List[ast.expr]:
    """What an assignment passes for each parameter in `wanted`."""
    out = [arg for index, arg in enumerate(call.args)
           if index < len(defined.positional) and defined.positional[index] in wanted]
    return out + [kw.value for kw in call.keywords if kw.arg in wanted]


def _extents(call: ast.Call) -> List[ast.expr]:
    """The shape a tensor registration is given: through the factory it wraps
    (`nn.Parameter(torch.empty(…))`) and through a method chained onto it
    (`.narrow(0, start, local).contiguous()`), then the first positional
    argument when that is a tuple (`torch.empty((out, in), dtype=…)`), else
    every positional argument (`torch.zeros(hidden, dtype=…)`)."""
    if not call.args:
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
            return _extents(call.func.value)
        return []
    first = call.args[0]
    if isinstance(first, (ast.Tuple, ast.List)):
        return list(first.elts)
    if isinstance(first, ast.Call):
        return _extents(first)
    return list(call.args)


def _names_distributed(tree: ast.Module) -> bool:
    """Whether a file reaches `torch.distributed` at all — by importing it
    under any name, or by writing it out."""
    if _dist_aliases(tree):
        return True
    return any(isinstance(node, ast.Attribute) and node.attr == "distributed"
               and isinstance(node.value, ast.Name) and node.value.id == "torch"
               for node in ast.walk(tree))


def _dist_aliases(tree: ast.Module) -> Set[str]:
    """The names a file binds `torch.distributed` to."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.distributed" and alias.asname:
                    out.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if ((node.module == "torch" and alias.name == "distributed")
                        or node.module == "torch.distributed"):
                    out.add(alias.asname or alias.name)
    return out


def _calls(value: ast.AST) -> List[ast.Call]:
    """The constructions an assigned value may be: the call, or each branch's
    of a conditional (`RMSNorm(h) if first else PPMissingLayer()`)."""
    if isinstance(value, ast.Call):
        return [value]
    if isinstance(value, ast.IfExp):
        return _calls(value.body) + _calls(value.orelse)
    return []


def _is_self_attribute(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self")


def _on_self(func: ast.AST) -> bool:
    while isinstance(func, ast.Attribute):
        func = func.value
    return isinstance(func, ast.Name) and func.id == "self"


def _called(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "?")
