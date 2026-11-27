"""KV cache marks on the rendered compute graph.

Every graph the site draws — the Inspector's blocks, the Playground's
isolated component, the inference page's fused serving graph — carries the
same two cache facts, projected here from `eval.cache` and nowhere else:

* ``kv`` on a node: the cached tensors it writes and reads, each with its
  byte size per rank and per layer. A compound (a component, a fused kernel,
  a ``block_stack``) carries the union of what its descendants do, so a
  collapsed block still shows that it writes and reads the cache.
* ``kind: "cache"`` on an edge whose source is cache-resident data. It rides
  the existing ``output_kind`` transport (`_shapes.build_edges` copies it
  onto each out-edge; `strip_transport_keys` drops it), per output port when
  the source is a fused kernel publishing several tensors — fused qk-rope
  writes K to the cache and hands Q to attention, and only K's edge is a
  cache edge.

The byte figures are the cache layer's own: ``bytes_per_position`` is one
cached token's row (elements × width ÷ the ranks the positions are spread
over), and the per-request and per-step figures are the tensor's storage and
read laws evaluated at the cache length and batch the graph was analysed at.
A graph drawn for the model rather than for a run (the Inspector's,
`workload_symbolic`) carries no length or batch: its entries keep the
per-token rows, the laws and a recurrent state's fixed size, and leave the
per-request and per-step figures of a growing cache out.
The frontend formats them; it computes none of them.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.launch import FusedNode
from backend.arch.eval.layer import Layer
from backend.arch.layout.physical_cache import cache_geometry


def _number(value: Any) -> Any:
    number = float(value_of(value))
    return int(number) if number.is_integer() else number


def _total(entries: List[Dict[str, Any]], key: str) -> Any:
    values = [e[key] for e in entries if e.get(key) is not None]
    return _number(sum(values)) if values else None


def kv_payload(writes: Iterable[Dict[str, Any]],
               reads: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The ``kv`` field of one node, from catalogue entries.

    Entries are deduplicated by ``ref``: a compound that contains a writer
    twice over (a fused kernel and its expanded member) still writes the
    tensor once. The totals are sums over the entries, per layer; a
    recurrent state is a per-request amount and is kept apart from the
    per-token ones.
    """
    w = sorted({e["ref"]: e for e in writes}.values(), key=lambda e: e["ref"])
    r = sorted({e["ref"]: e for e in reads}.values(), key=lambda e: e["ref"])
    if not w and not r:
        return None
    growing_w = [e for e in w if not e["recurrent"]]
    growing_r = [e for e in r if not e["recurrent"]]
    by_law: Dict[str, List[Dict[str, Any]]] = {}
    for e in growing_r:
        by_law.setdefault(e["read"], []).append(e)
    first = (w or r)[0]
    return {
        "writes": w,
        "reads": r,
        "write_bytes_per_token": _total(growing_w, "bytes_per_position"),
        "state_bytes_per_request": _total(
            [e for e in (w or r) if e["recurrent"]], "bytes_per_request"),
        "read_bytes_per_token": _total(growing_r, "bytes_per_position"),
        # The per-token reads grouped by the law each follows: the read is
        # the sum of `bytes_per_token × law`. One term when every read
        # follows one law; several where they differ (a DSA indexer's keys
        # read over `s_kv` beside the `Min(topk, s_kv)` rows it selects,
        # DeepSeek-V4's window beside its compressed stream).
        "read_terms": [{"law": law, "bytes_per_token":
                        _total(entries, "bytes_per_position")}
                       for law, entries in sorted(by_law.items())],
        "read_bytes_per_step": _total(r, "bytes_per_step"),
        "cache_len": first["cache_len"],
        "batch": first["batch"],
    }


def merge_kv(payloads: Iterable[Optional[Mapping[str, Any]]]
             ) -> Optional[Dict[str, Any]]:
    """The ``kv`` of a group of nodes — a compound, an L1 box."""
    writes: List[Dict[str, Any]] = []
    reads: List[Dict[str, Any]] = []
    for payload in payloads:
        if payload:
            writes.extend(payload["writes"])
            reads.extend(payload["reads"])
    return kv_payload(writes, reads)


class KVMarks:
    """The cache catalogue of one analysis, and who touches which tensor.

    ``access`` is keyed by the uids the rendered nodes carry. On the serving
    graph a fused kernel's uid is in the rebased ``access`` (its members'
    union) while its expanded children keep their pre-fusion uids, found in
    ``pre_access`` — the pair `pipeline.analyze` carries.
    """

    def __init__(self, catalogue: Mapping[str, Dict[str, Any]],
                 access: Layer[cache_mod.KVAccess],
                 pre_access: Optional[Layer[cache_mod.KVAccess]] = None,
                 program: Any = None) -> None:
        self.catalogue = catalogue
        self.access = access
        self.pre_access = pre_access
        self.program = program

    # -- construction ----------------------------------------------------

    @classmethod
    def of_analysis(cls, analysis: Any, *,
                    pre_analysis: Any = None,
                    workload_symbolic: bool = False) -> "KVMarks":
        """Marks for a graph drawn straight off one walk (Inspector,
        Playground): one rank, the walk's own shapes, each cached tensor at
        its own format.

        A fused walk may also name its pre-fusion analysis: the live fused
        kernel reads the rebased access layer while its expanded children read
        the members' original entries. Without that pair, opening a fused
        compound would lose the cache facts on the children it absorbed.
        """
        return cls(cls._catalogue(analysis, None, workload_symbolic),
                   cache_mod.kv_access(analysis),
                   cache_mod.kv_access(pre_analysis) if pre_analysis is not None else None,
                   analysis.program)

    @classmethod
    def of_serving(cls, result: Any) -> "KVMarks":
        """Marks for the inference page's serving graph: per-rank widths
        from the deployment's physical cache geometry (the one capacity
        budgets with), fused kernels from the rebased access layer."""
        geometry = cache_geometry(result.analysis)
        return cls(cls._catalogue(result.analysis, geometry),
                   result.kv_access, result.pre_kv_access, result.program)

    @staticmethod
    def _catalogue(analysis: Any, geometry: Any,
                   workload_symbolic: bool = False
                   ) -> Dict[str, Dict[str, Any]]:
        """ref → the entry fields every node touching that tensor shows."""
        program = analysis.program
        resident = cache_mod.resident(analysis, geometry=geometry)
        out: Dict[str, Dict[str, Any]] = {}
        # A walk that keeps no cache — an encode walk, which runs no language
        # step and binds no `s_kv` — has nothing to catalogue.
        if not resident:
            return out
        cache_len = int(value_of(cache_mod.workload_dim(analysis, "s_kv")))
        batch = int(value_of(cache_mod.workload_dim(analysis, "B")))
        for ref, (holder, entry, tensor) in resident.items():
            dtype = tensor.dtype
            try:
                graph = program.spec.get_block(holder.op or "")
            except (KeyError, AttributeError):
                graph = None
            stored = tensor.bytes_at(cache_len)
            read = tensor.bytes_at(cache_len, read=True)
            per_position = (None if tensor.recurrent else _number(
                tensor.position_bytes / tensor.positions_over))
            # Without a workload, only a recurrent state has a size per
            # request: a growing cache's is its law at a length.
            sized = tensor.recurrent or not workload_symbolic
            out[str(ref)] = {
                "ref": str(ref),
                "tensor": tensor.name,
                "family": entry.label,
                "display": graph.attn_display if graph is not None else None,
                "dtype": dtype,
                "recurrent": tensor.recurrent,
                "bytes_per_position": per_position,
                "stored": str(tensor.stored),
                "read": str(tensor.read),
                "stored_bytes_per_request": _number(stored) if sized else None,
                "read_bytes_per_request": _number(read) if sized else None,
                "cache_len": None if workload_symbolic else cache_len,
                "batch": None if workload_symbolic else batch,
            }
        return out

    # -- projection ------------------------------------------------------

    def _write_entry(self, ref: str) -> Dict[str, Any]:
        e = self.catalogue[ref]
        return {k: e[k] for k in (
            "ref", "tensor", "family", "display", "dtype", "recurrent",
            "bytes_per_position", "stored", "cache_len", "batch")} | {
            "bytes_per_request": e["stored_bytes_per_request"]}

    def _read_entry(self, ref: str) -> Dict[str, Any]:
        e = self.catalogue[ref]
        return {k: e[k] for k in (
            "ref", "tensor", "family", "display", "dtype", "recurrent",
            "bytes_per_position", "read", "cache_len", "batch")} | {
            "bytes_per_request": e["read_bytes_per_request"],
            "bytes_per_step": (None if e["batch"] is None else _number(
                e["read_bytes_per_request"] * e["batch"]))}

    def _access_of(self, uid: Optional[int]) -> Optional[cache_mod.KVAccess]:
        if uid is None:
            return None
        if uid in self.access:
            return self.access[uid]
        if self.pre_access is not None:
            return self.pre_access.get(uid)
        return None

    def stamp(self, nodes: List[Dict[str, Any]]) -> None:
        """Write ``kv`` and the cache ``output_kind`` onto a node tree.

        Must run on the finished tree and before `build_edges` — the edge
        builder reads ``output_kind`` off the source node — and so before
        `strip_transport_keys`.
        """
        for node in nodes:
            self._stamp(node)

    def _stamp(self, node: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """Stamp one node and its subtree; return the refs the subtree
        writes and reads."""
        own = self._access_of(node.get("uid"))
        writes: Set[str] = set()
        reads: Set[str] = set()
        if own is not None:
            writes |= {str(r) for r in own.writes}
            reads |= {str(r) for r in own.reads}
            self._mark_output(node, own)
        for child in node.get("children") or ():
            child_writes, child_reads = self._stamp(child)
            writes |= child_writes
            reads |= child_reads
        payload = kv_payload((self._write_entry(r) for r in writes),
                             (self._read_entry(r) for r in reads))
        if payload is not None:
            node["kv"] = payload
        return writes, reads

    def _mark_output(self, node: Dict[str, Any],
                     own: cache_mod.KVAccess) -> None:
        """Mark the outputs that carry cache-resident data."""
        engine = (self.program.by_uid(node["uid"])
                  if self.program is not None else None)
        if isinstance(engine, FusedNode):
            ports = {}
            for origin, _origin_port, port in engine.publishers():
                member = (self.pre_access.get(origin)
                          if self.pre_access is not None else None)
                if member is not None and member.publishes:
                    ports[port] = "cache"
            if ports.pop("", None):
                self._set_kind(node, "cache")
            if ports:
                node["port_kinds"] = ports
            return
        if own.publishes:
            self._set_kind(node, "cache")

    @staticmethod
    def _set_kind(node: Dict[str, Any], kind: str) -> None:
        existing = node.get("output_kind")
        if existing is not None and existing != kind:
            raise ValueError(
                f"{node['id']}: its output is cache-resident data but its op "
                f"declares output_kind {existing!r}; an edge has one kind")
        node["output_kind"] = kind
