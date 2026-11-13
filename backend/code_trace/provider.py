"""Which code a computation ran: the code half of a cached result's version.

A cached result is a function of its inputs and of the code that computed
it. The inputs are in its key (spec fingerprint, data footprint, request
content); the code is answered here, by one of two providers behind one
small interface:

  * `GlobalHash` — today's answer: the code is every engine `.py`, so the
    trace of any computation is the one global hash
    (`backend.fingerprint.engine_code_version`), and any edit anywhere makes
    every trace stale.
  * `TracedFunctions` — a verifying trace (Shake, Salsa; "Build Systems à la
    Carte"): the functions the computation actually executed, each with the
    hash of its normalized source (`source.FileIdentity`), recorded with
    `sys.monitoring` PY_START. A trace stays valid while each function it
    names hashes the same; an edit to code a computation never ran leaves
    its result valid.

The interface: `begin()` a recording, `end()` it into a `Trace`, ask
`valid(trace)` against the code this process runs. A memo that skips code
on a hit hands the trace it recorded on its miss to `merge` (`memo.py`); a
memo whose value keeps being filled after it is built holds a `Growing` and
`attach`es it on every read instead (`backend.inference.walk_registry`). Key
derivation that must not count as a dependency runs under `untraced()`.

Soundness of `TracedFunctions` rests on four rules, each a test in
`tests/backend/code_trace/`:

  1. Every repo code object first started or resumed while a recording is
     active is added to every active recording (PY_START and PY_RESUME: a
     generator made before `begin` and resumed inside is code the
     computation runs). The callback DISABLEs itself per code location;
     `begin` re-arms all of them with `restart_events`. A computation's
     recording also starts with the code of the frame that opened it — it
     keeps executing inside it; recordings are lexical, so no caller above
     it does. `untraced` is per thread: that thread's code lands only in
     recordings begun inside the window, and stays armed while another
     computation has a recording open (with none, it is disabled as usual
     and re-armed when the window closes, for this computation's own
     recordings around it). Supersets are allowed, misses are not.
  2. What ran before any recording — module top levels, class bodies, and
     every function called while importing — is one import-time trace
     (`global_trace`), named by one item in every trace. The traced trees
     are imported in full before the first recording, so nothing imports
     later; a repo module imported after that is refused (strict) or logged.
     The import-time trace holds every module's skeleton (`source.
     FileIdentity`): adding, removing or rebinding a def anywhere, or editing
     a module's top level, makes every trace stale. That is the price of
     soundness for what import time decides — a subclass override, a
     module-level rebinding, a registration (`register_op`) — and a
     body-only edit, the common bug fix or formula tweak, stays precise.
     The recording machinery itself (this package) is in it too: it decides
     what a trace holds.
  3. Third-party machinery can memoize calls into our code where no memo of
     ours sees it (SymPy's `cacheit` calling an `_eval_is_*` of a repo `Expr`
     subclass). So every method of a repo class with a base class from
     outside both the repo and the standard library is in the import-time
     trace too; and what such a method runs when that machinery calls it is
     the foreign closure (`_watch_foreign`), named by every trace. That
     cache is this process's, so whatever it skips ran here before, inside
     such a call. The closure grows with what a process ran, so two
     processes may record one computation with traces that differ by it; a
     trace is checked item by item, so a wider one from elsewhere holds.
  4. Every memo in the traced trees goes through `memo.py` (the AST gate in
     `tests/meta/test_code_trace_gate.py`).

Third-party code is one item: the hash of `uv.lock`.
Stdlib only, and imports nothing from `backend`: it sits below every package
whose code it names.

Deploying and switching
-----------------------
`HA_CODE_TRACE` picks the provider per process, at `backend` import:
`global` (the default) or `functions`. Every process of a deployment — the
server, its pool workers (they inherit the environment), the warm job —
must run the same one: a key under `global` carries the engine code hash,
under `functions` a constant (`fingerprint.result_version`), so the two
never read each other's entries. Switching back is setting it back.

What changes on deploy, whichever provider: every cached value is stored
beside its trace id (`measure.cache.KeyedCache`), the SOL pool's schema
moved to 5 with a trace per record, and the validate replay memo keys on
the data half — so the result caches (memory, disk, the Supabase tier, the
Gallery artifact), the SOL pool and the replay memo are cold once and
refill. Under `global` nothing else moves: keys, `engine_version()` and the
artifact paths are byte for byte what they were.

Under `functions`, a process seals at start (the server's lifespan, each
pool worker): it imports the traced trees and reads the canonical
import-time trace and each file's identity from `HA_CODE_TRACE_CACHE`
(computed once per source). Trace bodies live in `HA_CODE_TRACE_DIR`, the
remote tier, and inside each SOL bundle and the Gallery artifact.
`HA_CODE_TRACE_STRICT` (tests) makes a module imported after the seal, and
a value stored with no recording open, fail loudly.

Before deploying an edit, `python -m tools.code_trace_dryrun --from <old>
--to <new>` reads the stored traces against the functions the edit moved
and says which SOL records and cache entries would go stale.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import importlib
import importlib.machinery
import json
import logging
import os
import pkgutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple

from . import import_state
from .source import FileIdentity, digest

logger = logging.getLogger(__name__)

#: Which provider a process runs: "global" (default) or "functions". Read
#: once, at the first `provider()` — or at `backend` import when it names
#: "functions", so the import-time trace starts before any engine import.
ENV = "HA_CODE_TRACE"
#: Set (to anything) to refuse, rather than log, a repo module imported after
#: the traced trees were (rule 2). Tests set it.
STRICT_ENV = "HA_CODE_TRACE_STRICT"

#: `backend/`: the code a trace names lives under it.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: This package: the machinery that records is not what a result depends on.
_SELF = os.path.dirname(os.path.abspath(__file__)) + os.sep

#: The packages imported in full before the first recording (rule 2).
TRACED_TREES = ("backend.arch", "backend.inference", "backend.measure",
                "backend.hf")

#: Storage infrastructure inside the traced trees: their code runs when a
#: result is READ through them, but it does not compute the result. A trace
#: that includes them means every storage edit invalidates every record —
#: the exact coarse-granularity `functions` mode exists to avoid.
TRACED_EXCLUDE = frozenset({
    "backend.inference.sol_pool",      # the SOL record store
})

#: The fresh interpreter `_canonical_import_time` runs: it records importing
#: the traced trees and nothing else, and writes the trace.
_CANONICAL_CHILD = """
import json, os, sys
spec = json.loads(sys.argv[1])
from backend.code_trace.provider import TracedFunctions
p = TracedFunctions(root=spec["root"], trees=tuple(spec["trees"]),
                    lock_file=spec["lock"], canonical=True)
p.seal()
tmp = spec["out"] + f".{os.getpid()}.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump({"trace": p.global_trace.to_json(),
               "foreign": list(p.foreign_methods)}, f)
os.replace(tmp, spec["out"])
"""

_EVENTS = (sys.monitoring.events.PY_START, sys.monitoring.events.PY_RESUME)
_EVENT_SET = sys.monitoring.events.PY_START | sys.monitoring.events.PY_RESUME

ENGINE_ITEM = "<engine>"
_UNSEEN = object()
IMPORT_ITEM = "<import-time>"
LOCK_ITEM = "<uv.lock>"


@dataclass(frozen=True)
class Trace:
    """What one computation's result depends on in code: (item, hash)
    pairs, sorted. Content-addressed by `id`."""

    items: Tuple[Tuple[str, str], ...]

    @functools.cached_property
    def id(self) -> str:
        """Hashed once: a trace is merged, stored and looked up by it, and
        hashing every item each time was a cost of its own."""
        return digest("\n".join(f"{k}\t{h}" for k, h in self.items))

    def names(self) -> FrozenSet[str]:
        return frozenset(k for k, _h in self.items)

    def covers(self, other: "Trace") -> bool:
        """Whether this trace names every item `other` names, at its hash."""
        return set(other.items) <= set(self.items)

    def to_json(self) -> List[List[str]]:
        return [list(pair) for pair in self.items]

    @classmethod
    def from_json(cls, data: Any) -> "Trace":
        return cls(tuple(sorted((str(k), str(h)) for k, h in data)))


class Fragment:
    """A memo miss's trace, not yet assembled: the code objects it ran and
    the traces (or fragments) it merged. A memo keeps one beside its value
    and hands it to the computations around each hit; only a computation's
    own trace is assembled into items (`TracedFunctions._assemble`), once,
    however many misses it reached. Assembling every miss's — thousands a
    computation — was most of what recording cost. In-process only: pickled
    (a value crossing to a worker with its memos), it becomes the trace of
    its code and parts, flattened once (`provider.flatten`) however many
    values carry it."""

    __slots__ = ("codes", "flat", "parts")

    def __init__(self, codes: FrozenSet[Any], parts: Tuple[Any, ...]) -> None:
        self.codes = codes
        self.parts = parts
        self.flat: Optional[Trace] = None

    @property
    def id(self) -> str:
        """Its identity among a recording's merged traces: the object's."""
        return f"fragment:{id(self)}"

    def __reduce__(self) -> Any:
        if self.flat is None:
            self.flat = provider().flatten(self)
        return (Trace, (self.flat.items,))


@dataclass(eq=False)
class Recording:
    """One computation's recording in progress."""

    codes: set = field(default_factory=set)
    #: Traces of memo hits, by id.
    traces: Dict[str, Trace] = field(default_factory=dict)
    #: Growing memo entries this recording read (`attach`).
    growing: List["Growing"] = field(default_factory=list)
    #: The last `snapshot` taken, and what it was taken of.
    snap: Any = None


@dataclass(eq=False)
class Growing:
    """A memo entry whose value is filled after it is built (a registry of
    walks, lazily walked): its trace is everything every computation that
    read it ran, and grows as they end."""

    codes: set = field(default_factory=set)
    traces: Dict[str, Trace] = field(default_factory=dict)
    #: Recordings reading it right now: what they ran so far is folded into
    #: any other reader's trace at its end.
    readers: List[Recording] = field(default_factory=list)


class GlobalHash:
    """Today's code version: one hash of every engine `.py`."""

    traced = False
    #: No recordings to change (`TracedFunctions.epoch`).
    epoch = 0
    #: Whether a value stored with no recording open fails loudly: never
    #: here, where every store has the one trace.
    strict = False
    #: Whether a cache key carries the code (`fingerprint.result_version`):
    #: here it does, so keys are exactly the ones they were before traces,
    #: and a trace (the same hash) is only ever valid.
    keys_code = True

    def __init__(self, code_version: Callable[[], str]) -> None:
        self._code_version = code_version

    def trace(self) -> Trace:
        return Trace(((ENGINE_ITEM, self._code_version()),))

    def begin(self, *, nested: bool = False) -> Recording:
        return Recording()

    def end(self, rec: Recording, *, nested: bool = False) -> Trace:
        return self.trace()

    def flatten(self, fragment: Any) -> Trace:
        return self.trace()

    def snapshot(self, rec: Optional[Recording] = None) -> Optional[Trace]:
        return self.trace()

    def seal(self) -> None:
        return None

    def merge(self, trace: Optional[Trace],
              into: Optional[Tuple[Recording, ...]] = None) -> None:
        return None

    def merged(self, slot: Any) -> None:
        return None

    def attach(self, entry: Growing) -> None:
        return None

    def valid(self, trace: Trace) -> bool:
        return trace == self.trace()

    @contextlib.contextmanager
    def untraced(self) -> Iterator[None]:
        yield


class TracedFunctions:
    """The functions a computation executed, each by its source hash."""

    traced = True
    #: The code is each entry's trace, so a key names only data and request:
    #: an edit leaves a key where it was and the trace decides.
    keys_code = False

    def __init__(self, *, strict: bool = False,
                 trees: Tuple[str, ...] = TRACED_TREES,
                 root: str = _BACKEND,
                 lock_file: Optional[str] = None,
                 canonical: bool = False) -> None:
        #: Refuse — rather than log — a repo module imported after the
        #: traced trees were, and a value stored with no recording open
        #: (`HA_CODE_TRACE_STRICT`; tests set it).
        self.strict = strict
        #: This process is the fresh interpreter that records the canonical
        #: import-time trace (`_canonical_import_time`).
        self._canonical = canonical
        self._trees = trees
        #: Files of the excluded modules, resolved once; the event callback
        #: checks co_filename against this set.
        self._excluded_files: frozenset = frozenset()
        self._resolve_excludes()
        #: Code under `root` is what traces name, by path relative to it.
        self._root = os.path.abspath(root)
        self._prefix = self._root + os.sep
        self._lock_file = (lock_file if lock_file is not None else
                           os.path.join(os.path.dirname(self._root), "uv.lock"))
        self._lock = threading.RLock()
        self._import = Recording()
        #: The active recordings, replaced (never mutated) under the lock, so
        #: the callback iterates a stable tuple without taking it.
        self._active: Tuple[Recording, ...] = (self._import,)
        self._sealed = False
        #: The traced trees' static import graph (`import_state`), built
        #: at `seal`: the import-time state a trace's named code reaches.
        self._imports: Optional[import_state.ImportGraph] = None
        #: Moves whenever the set of active recordings does: a memo hit
        #: whose trace was merged in the same epoch was merged into every
        #: one of them already (`merged`).
        self.epoch = 0
        #: Set while `seal` imports: a memo miss at import time records, but
        #: must not seal again.
        self._sealing = False
        self._files: Dict[str, FileIdentity] = {}
        #: `current`'s files, by the path a trace names them by (None: no
        #: such file in the code this process runs).
        self._by_rel: Dict[str, Optional[FileIdentity]] = {}
        self._item_of: Dict[Any, Tuple[str, str]] = {}
        self._valid: Dict[str, bool] = {}
        self._modules: FrozenSet[str] = frozenset()
        self.global_trace: Optional[Trace] = None
        self._global_items: Dict[str, str] = {}
        #: The methods rule 3 put in the import-time trace.
        self.foreign_methods: Tuple[str, ...] = ()
        #: Per thread, while it runs `untraced` code: the recordings begun on
        #: it inside that window, the only ones its code may land in; and
        #: how deep it is in a watched foreign method (`_watch_foreign`).
        self._thread = threading.local()
        #: The foreign closure: every repo code object that ran inside a
        #: watched foreign method in this process, ever (`_watch_foreign`).
        #: Every trace names it.
        self._closure: set = set()
        mon = sys.monitoring
        self._tool = next(t for t in range(6) if mon.get_tool(t) is None)
        mon.use_tool_id(self._tool, "code_trace")
        callback = self._callback()
        for event in _EVENTS:
            mon.register_callback(self._tool, event, callback)
        mon.set_events(self._tool, _EVENT_SET)
        #: The canonical import-time trace's interpreter, when this source has
        #: none on disk yet: started now, read at `seal`.
        self._canonical_child: Optional[Tuple[Any, Any]] = None
        if not canonical and not os.path.exists(self._canonical_path()):
            try:
                self._start_canonical()
            except OSError:
                logger.debug("code_trace: canonical child not started early",
                             exc_info=True)

    # -- recording ---------------------------------------------------------

    def _callback(self) -> Callable[[Any, int], Any]:
        """The PY_START / PY_RESUME callback: a closure over everything it
        reads, which outlives module globals at interpreter shutdown.

        A code object first executed (or a generator first resumed) lands in
        every active recording and is then DISABLEd until the next `begin`
        re-arms it. On a thread inside `untraced`, it lands only in the
        recordings begun inside that window and is NOT disabled: another
        thread's recording, or this one's once the window closes, must
        still see it the first time it runs there."""
        prefix, own, disable = self._prefix, _SELF, sys.monitoring.DISABLE
        provider, thread = self, self._thread
        excluded = self._excluded_files

        def on_event(code: Any, _offset: int) -> Any:
            name = code.co_filename
            if not name.startswith(prefix) or name.startswith(own):
                return disable
            if excluded and os.path.abspath(name) in excluded:
                return disable
            window = getattr(thread, "window", None)
            if window is not None:
                for rec in window:
                    rec.codes.add(code)
                # Armed for another computation's recording, which must see
                # this code the first time it runs there; with none, disabled
                # as usual, and re-armed when the window closes (`untraced`)
                # for this computation's own (open in this context, whatever
                # thread runs it).
                mine = _COMPUTATIONS.get()
                if any(rec not in mine for rec in provider._active):
                    return None
                thread.disabled_in_window = True
                return disable
            for rec in provider._active:
                rec.codes.add(code)
            return disable

        return on_event

    def close(self) -> None:
        """Stop recording and free the monitoring tool id."""
        mon = sys.monitoring
        mon.set_events(self._tool, 0)
        for event in _EVENTS:
            mon.register_callback(self._tool, event, None)
        mon.free_tool_id(self._tool)

    def _targets(self) -> Any:
        """The recordings what this thread runs now lands in."""
        window = getattr(self._thread, "window", None)
        return self._active if window is None else tuple(window)

    def begin(self, *, nested: bool = False) -> Recording:
        """A new recording.

        A computation's (`nested=False`) first imports the traced trees
        (`seal`), and starts with the code of the frame that opened it: that
        frame's body goes on executing inside it, and no PY_START will say
        so. Only that one — a recording is lexical (`recording` is a with
        block, opened and closed in one frame), so the callers above it are
        suspended until it closes and run none of their body inside it; a
        caller's code in the trace would make one computation's trace
        depend on the path that reached it. A memo miss's (`nested=True`)
        does neither: it can run in the middle of an import, where importing
        everything else would meet that module half-built, and what it
        depends on is the memoised call, not who called it."""
        if not nested and not self._sealed:
            self.seal()
        rec = Recording()
        if not nested:
            code = _opening_code()
            if (code is not None and code.co_filename.startswith(self._prefix)
                    and not code.co_filename.startswith(_SELF)):
                rec.codes.add(code)
        with self._lock:
            self._active = (*self._active, rec)
            self.epoch += 1
        window = getattr(self._thread, "window", None)
        if window is not None:
            window.append(rec)
        sys.monitoring.restart_events()
        return rec

    def end(self, rec: Recording, *, nested: bool = False) -> Any:
        """`rec`'s trace — for a memo miss's (`nested`), a `Fragment`,
        assembled only inside the computation that reaches it."""
        window = getattr(self._thread, "window", None)
        if window is not None and rec in window:
            window.remove(rec)
        with self._lock:
            self._active = tuple(r for r in self._active if r is not rec)
            self.epoch += 1
        if nested and not rec.growing:
            return Fragment(frozenset(rec.codes), tuple(rec.traces.values()))
        return self._trace_of(rec, final=True)

    def flatten(self, fragment: Fragment) -> Trace:
        """`fragment` as a trace: the items of its code objects and of every
        part it reaches. No more — the receiving computation adds the
        foreign closure and the import-time trace when it forms its own."""
        codes, items = self._reach({fragment.id: fragment})
        items = {**dict(self._items(codes)), **items}
        return Trace(tuple(sorted(items.items())))

    def _reach(self, traces: Dict[str, Any]) -> Tuple[set, Dict[str, str]]:
        """The code objects and items `traces` stand for: each trace's
        items, each fragment's code objects and its parts', each once."""
        codes: set = set()
        items: Dict[str, str] = {}
        pending = list(traces.values())
        seen: set = set()
        while pending:
            part = pending.pop()
            if id(part) in seen:
                continue
            seen.add(id(part))
            if type(part) is Fragment:
                codes |= part.codes
                pending.extend(part.parts)
            else:
                items.update(part.items)
        return codes, items

    def snapshot(self, rec: Optional[Recording] = None) -> Optional[Trace]:
        """The trace `rec` (default: the innermost computation open in this
        context, `current`) has recorded so far, leaving it open: what a value
        computed inside it so far depends on. None when no computation is
        open."""
        if rec is None:
            open_ = current()
            if not open_:
                return None
            rec = open_[-1]
        # A computation handing out many jobs in a row is snapshotted once
        # while it records nothing new in between.
        seen = (len(rec.codes), len(rec.traces), len(self._closure),
                tuple((len(e.codes), len(e.traces),
                       sum(len(r.codes) + len(r.traces) for r in e.readers))
                      for e in rec.growing))
        if rec.snap is not None and rec.snap[0] == seen:
            return rec.snap[1]
        trace = self._trace_of(rec, final=False)
        rec.snap = (seen, trace)
        return trace

    def _trace_of(self, rec: Recording, *, final: bool) -> Trace:
        """`rec`'s trace: what it ran, the traces it merged, and for each
        growing entry it read, what every reader of the entry ran — save
        its own enclosing computations in this context. Those run none of
        their body while `rec` is open (a recording is lexical), so nothing
        they ran filled the entry for `rec`; what they ran reaches the entry
        when they end, as theirs."""
        codes = set(rec.codes)
        traces = dict(rec.traces)
        open_ = current()
        enclosing = open_[:open_.index(rec)] if rec in open_ else ()
        with self._lock:
            for entry in rec.growing:
                codes |= entry.codes
                traces.update(entry.traces)
                for other in entry.readers:
                    if other is not rec and other not in enclosing:
                        codes |= set(other.codes)
                        traces.update(other.traces)
            if final:
                for entry in rec.growing:
                    entry.readers = [r for r in entry.readers if r is not rec]
                    entry.codes |= codes
                    entry.traces.update(traces)
        return self._assemble(codes, traces)

    def _assemble(self, codes: set, traces: Dict[str, Any]) -> Trace:
        """A trace's items: of the code objects, of every trace and
        fragment merged (a fragment's own merges included, each once), of
        the foreign closure; then the import-time state of the traced
        trees the named code reaches (`import_state`)."""
        reached, items = self._reach(traces)
        codes |= reached
        # The foreign closure, whatever this computation ran: a third-party
        # cache it hit may have skipped any of it (`_watch_foreign`).
        codes |= set(self._closure)
        items = {**dict(self._items(codes)), **items}
        if self._sealed:
            self._check_modules()
            # The import-time state the named code reaches, module by
            # module (`import_state.ImportGraph.state_items`).
            items = {**items,
                     **self._imports.state_items(items, self._lock_file)}
        # Unsealed: a memo miss while the trees import. Its code is in the
        # import-time trace too, which every later trace names.
        return Trace(tuple(sorted(items.items())))

    def merge(self, trace: Optional[Trace],
              into: Optional[Tuple[Recording, ...]] = None) -> None:
        """`trace` into the recordings `into` (default: every recording what
        this thread runs lands in)."""
        if trace is None:
            return
        for rec in (self._targets() if into is None else into):
            if trace.id not in rec.traces:
                rec.traces[trace.id] = trace

    def merged(self, slot: Any) -> None:
        """A memo hit: `slot.trace` into every recording this thread's code
        lands in — unless it went into all the active ones in this epoch
        already, which covers a thread inside `untraced` too (its targets
        are among them). Memos test the epoch before calling."""
        if slot.epoch == self.epoch:
            return
        self.merge(slot.trace)
        if getattr(self._thread, "window", None) is None:
            slot.epoch = self.epoch

    def attach(self, entry: Growing) -> None:
        with self._lock:
            for rec in self._targets():
                if rec is self._import or entry in rec.growing:
                    continue
                rec.growing.append(entry)
                entry.readers.append(rec)

    @contextlib.contextmanager
    def untraced(self) -> Iterator[None]:
        """Run code on this thread that no recording open around it counts:
        key derivation. Other threads' recordings are untouched, and a memo
        miss inside the window still records its own call."""
        outer = getattr(self._thread, "window", None)
        self._thread.window = []
        self._thread.disabled_in_window = False
        try:
            yield
        finally:
            self._thread.window = outer
            if getattr(self._thread, "disabled_in_window", False):
                # What the window disabled, this thread's own recordings
                # (around it) must still see when it runs there.
                self._thread.disabled_in_window = False
                sys.monitoring.restart_events()

    # -- the import-time trace ---------------------------------------------

    def seal(self) -> None:
        """Import the traced trees in full, close the import-time recording
        and fix the code this process runs (rules 2 and 3)."""
        with self._lock:
            if self._sealed or self._sealing:
                return
            self._sealing = True
            failed = self._import_trees()
            with self._lock:
                self._active = tuple(r for r in self._active
                                     if r is not self._import)
                self.epoch += 1
            self._modules = self._repo_modules()
            for path in self._module_files():
                self._identity(path)
            self._imports = import_state.ImportGraph(
                self._root, self._trees, self._identity,
                frozenset(
                    os.path.relpath(path, self._root).replace(os.sep, "/")
                    for path in self._excluded_files))
            if self._canonical:
                self.global_trace, self.foreign_methods = self._import_time()
            else:
                self.global_trace, self.foreign_methods = self._canonical_import_time()
                self._watch_foreign()
            self._global_items = dict(self.global_trace.items)
            # Every trace names this one by id: keep its body where a trace's
            # reader (and `tools.code_trace_dryrun`) can find it.
            TRACES.put(self.global_trace)
            self._sealed = True
            self._sealing = False
            if failed:
                logger.warning("code_trace: %d modules failed to import: %s",
                               len(failed), ", ".join(failed))

    def _import_time(self) -> Tuple[Trace, Tuple[str, ...]]:
        """This process's import-time trace: what ran while it imported, the
        skeleton of every module it imported, rule 3's methods, this
        package's own source and `uv.lock`. Canonical only in a process
        that imported nothing but the traced trees (`_canonical_import_time`)."""
        items: Dict[str, str] = dict(self._items(self._import.codes))
        for path in self._module_files():
            items.update((self._identity(path).module_item(),))
        foreign = self._foreign_methods()
        items.update(foreign)
        # The recording machinery is not recorded, but it decides what a
        # trace holds: an edit to it moves every trace.
        for name in sorted(os.listdir(_SELF)):
            if name.endswith(".py"):
                with open(os.path.join(_SELF, name), encoding="utf-8") as f:
                    own = FileIdentity(f"code_trace/{name}", f.read())
                items.update(own.items())
        if os.path.exists(self._lock_file):
            with open(self._lock_file, "rb") as f:
                items[LOCK_ITEM] = digest(f.read().decode("utf-8", "replace"))
        return Trace(tuple(sorted(items.items()))), tuple(sorted(foreign))

    def _source_key(self) -> str:
        """What the canonical import-time trace is a function of: every
        `.py` under the root and this package's, `uv.lock`, the traced trees
        and the interpreter."""
        h = hashlib.sha256(repr((sys.version, self._trees)).encode())
        for base in (self._root, _SELF):
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames.sort()
                for name in sorted(filenames):
                    if name.endswith(".py"):
                        path = os.path.join(dirpath, name)
                        h.update(os.path.relpath(path, base).encode())
                        with open(path, "rb") as f:
                            h.update(f.read())
        if os.path.exists(self._lock_file):
            with open(self._lock_file, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:24]

    def _canonical_import_time(self) -> Tuple[Trace, Tuple[str, ...]]:
        """The import-time trace as a fresh interpreter that imports exactly
        the traced trees records it (`_CANONICAL_CHILD`), once per source
        (`_source_key`), kept on disk. What a process happened to import
        before it sealed — an API server's routes, a warm job's script —
        then moves no trace: a trace recorded in a pool worker, a warm job
        or another machine reads the same here."""
        path = self._canonical_path()
        if not os.path.exists(path):
            child, errors = self._canonical_child or self._start_canonical()
            child.wait()
            if child.returncode != 0 or not os.path.exists(path):
                errors.seek(0)
                logger.warning("code_trace: the canonical import-time trace "
                               "could not be computed (%s); this process's "
                               "own stands in", errors.read()[-2000:])
                return self._import_time()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return Trace.from_json(data["trace"]), tuple(data["foreign"])

    def _canonical_path(self) -> str:
        return os.path.join(DERIVED, "import-time", f"{self._source_key()}.json")

    def _start_canonical(self) -> Tuple[Any, Any]:
        """Start the fresh interpreter `_canonical_import_time` reads, and
        do not wait for it: it needs only the source, so it runs while this
        process imports (`__init__`), and leaves each file's identity in
        `DERIVED` behind it."""
        os.makedirs(os.path.join(DERIVED, "import-time"), exist_ok=True)
        spec = json.dumps({"root": self._root, "trees": list(self._trees),
                           "lock": self._lock_file, "out": self._canonical_path()})
        env = dict(os.environ, HA_CODE_TRACE="global",
                   PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        # Read at `seal`, after this returns: kept open beside the child.
        errors = tempfile.TemporaryFile("w+")  # noqa: SIM115
        child = subprocess.Popen([sys.executable, "-c", _CANONICAL_CHILD, spec],
                                 env=env, stdout=subprocess.DEVNULL,
                                 stderr=errors, text=True)
        self._canonical_child = (child, errors)
        return self._canonical_child

    def _resolve_excludes(self) -> None:
        """Map each excluded module name to its file, once at start —
        found, not imported: the provider is the first thing a process
        installs, and importing a module here would run repo code (and
        bind `memo` to a provider) before this one exists and records."""
        files = set()
        for name in TRACED_EXCLUDE:
            path: Optional[List[str]] = None
            spec = None
            for depth in range(1, name.count(".") + 2):
                spec = importlib.machinery.PathFinder.find_spec(
                    ".".join(name.split(".")[:depth]), path)
                if spec is None:
                    break
                path = spec.submodule_search_locations
            if spec is None or not spec.origin:
                raise ValueError(f"TRACED_EXCLUDE names {name!r}, which has "
                                 f"no source file on sys.path")
            files.add(os.path.abspath(spec.origin))
        self._excluded_files = frozenset(files)

    def _import_trees(self) -> List[str]:
        names: List[str] = []
        for tree in self._trees:
            package = importlib.import_module(tree)
            names.extend(info.name for info in pkgutil.walk_packages(
                package.__path__, tree + ".", onerror=lambda _name: None))
        names = [n for n in names if n not in TRACED_EXCLUDE]
        failed = names
        # A module that fails first can be an import cycle that resolves
        # once its partner is in (the order a run would import them in).
        for _attempt in range(3):
            retry = []
            for name in failed:
                try:
                    importlib.import_module(name)
                except Exception:
                    retry.append(name)
            if len(retry) == len(failed):
                break
            failed = retry
        return failed

    def _repo_modules(self) -> FrozenSet[str]:
        return frozenset(
            name for name, mod in list(sys.modules.items())
            if (getattr(mod, "__file__", None) or "").startswith(self._prefix)
            and not mod.__file__.startswith(_SELF))

    def _module_files(self) -> List[str]:
        return sorted({os.path.abspath(sys.modules[name].__file__)
                       for name in self._modules if name in sys.modules})

    def _foreign_classes(self) -> Iterator[Tuple[Any, type]]:
        """(module, class) of every repo class with a base from outside the
        repo and the standard library (rule 3). From the classes themselves
        (their MRO), not a list of names."""
        seen: set = set()
        for name in sorted(self._modules):
            mod = sys.modules.get(name)
            if mod is None:
                continue
            pending = [v for v in list(vars(mod).values())
                       if isinstance(v, type)]
            while pending:
                cls = pending.pop()
                if id(cls) in seen or getattr(cls, "__module__", None) != name:
                    continue
                seen.add(id(cls))
                pending.extend(v for v in vars(cls).values()
                               if isinstance(v, type))
                if any(self._foreign(base) for base in cls.__mro__[1:]):
                    yield mod, cls

    def _foreign_methods(self) -> Dict[str, str]:
        """Rule 3's items: every method of a `_foreign_classes` class."""
        out: Dict[str, str] = {}
        for mod, cls in self._foreign_classes():
            identity = self._identity(os.path.abspath(mod.__file__))
            prefix = f"{identity.rel}::{cls.__qualname__}."
            out.update((key, h) for key, h in identity.items().items()
                       if key.startswith(prefix)
                       and "." not in key[len(prefix):])
        return out

    def _watch_foreign(self) -> None:
        """Wrap every method of a `_foreign_classes` class, so the code it
        runs joins the foreign closure (`_closure`).

        Rule 3 names those methods; it cannot name what they call. A
        third-party cache that answers a call without making it (SymPy's
        `cacheit` skipping a repo `Function.eval`) skips their callees too.
        That cache is this process's, so whatever it skips ran in this
        process before, inside one of these methods called back by such
        machinery: everything that ran inside one is kept, and every trace
        names it (`_trace_of`). The closure grows with what this process
        happened to run, so two processes can name one computation with
        traces that differ by it; a trace is checked item by item
        (`valid`), so a wider one recorded elsewhere still holds here."""
        for _mod, cls in self._foreign_classes():
            for name, member in list(vars(cls).items()):
                wrapped = self._watched(member)
                if wrapped is not None:
                    try:
                        setattr(cls, name, wrapped)
                    except (AttributeError, TypeError):
                        logger.debug("code_trace: %s.%s not watched",
                                     cls.__qualname__, name)

    def _watched(self, member: Any) -> Any:
        """`member` (a function, classmethod or staticmethod of repo code)
        wrapped to run as foreign code; None for anything else."""
        kind = type(member)
        fn = member.__func__ if kind in (classmethod, staticmethod) else member
        code = getattr(fn, "__code__", None)
        if (not callable(fn) or code is None
                or not code.co_filename.startswith(self._prefix)
                or getattr(fn, "__code_trace_watched__", False)):
            return None
        thread, prefix, own = self._thread, self._prefix, _SELF
        closure = self._closure

        def watch(frame: Any, event: str, _arg: Any) -> None:
            if event == "call":
                name = frame.f_code.co_filename
                if name.startswith(prefix) and not name.startswith(own):
                    closure.add(frame.f_code)

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            # Only a call made by code outside the repo — the third-party
            # machinery, whose cache can skip it — is a callback; a call from
            # repo code is that code's, recorded or skipped with it. And
            # inside one callback, the next is already watched.
            if (sys._getframe(1).f_code.co_filename.startswith(prefix)
                    or getattr(thread, "foreign", 0)):
                return fn(*args, **kwargs)
            # A profile hook, this thread's alone and only while the
            # callback runs, sees every call in it — code this process ran
            # before (DISABLEd for `sys.monitoring`) included — with nothing
            # re-armed for anyone else.
            thread.foreign = 1
            outer = sys.getprofile()
            sys.setprofile(watch)
            try:
                return fn(*args, **kwargs)
            finally:
                sys.setprofile(outer)
                thread.foreign = 0

        run.__code_trace_watched__ = True  # type: ignore[attr-defined]
        return kind(run) if kind in (classmethod, staticmethod) else run

    def _foreign(self, cls: type) -> bool:
        """A class defined outside both the traced code and the standard
        library: third-party machinery (SymPy, pydantic) whose own caches
        can call into a subclass's methods unseen. The standard library's
        classes (exceptions, `dict`, `Enum`) memoise no such calls."""
        module = getattr(cls, "__module__", "") or ""
        if module.partition(".")[0] in sys.stdlib_module_names:
            return False
        mod = sys.modules.get(module)
        return not (getattr(mod, "__file__", None) or "").startswith(self._prefix)

    def _check_modules(self) -> None:
        now = self._repo_modules()
        new = now - self._modules
        if not new:
            return
        message = (f"code_trace: repo modules imported after the traced trees "
                   f"were: {sorted(new)} — their import-time code is in no "
                   f"trace; add their package to TRACED_TREES")
        if self.strict:
            raise RuntimeError(message)
        logger.warning(message)
        with self._lock:
            self._modules = now

    # -- items and validity ------------------------------------------------

    def _identity(self, path: str) -> FileIdentity:
        """`path`'s identity: normalized once per content, ever — kept in
        `DERIVED` by the content's hash, so a process that seals reads the
        files it has seen before instead of parsing them again."""
        identity = self._files.get(path)
        if identity is None:
            with open(path, encoding="utf-8") as f:
                src = f.read()
            rel = os.path.relpath(path, self._root).replace(os.sep, "/")
            kept = os.path.join(DERIVED, "files", f"{digest(src)}.json")
            try:
                with open(kept, encoding="utf-8") as f:
                    identity = FileIdentity.from_json(rel, json.load(f))
            except (OSError, ValueError, KeyError, TypeError):
                identity = FileIdentity(rel, src)
                _write_json(kept, identity.to_json())
            self._files[path] = identity
        return identity

    def _item_for(self, code: Any) -> Tuple[str, str]:
        item = self._item_of.get(code)
        if item is None:
            item = self._item_of[code] = self._identity(
                code.co_filename).item_of(code.co_firstlineno)
        return item

    def _items(self, codes: Any) -> Iterator[Tuple[str, str]]:
        """Each code's item, and its module's import-time state — per
        top-level binding (`FileIdentity`), not as the one whole-file
        skeleton: a def, alias or method added to the module is a new
        binding's new item, where the skeleton hash moved on any addition
        and made every trace of the module's code stale with it."""
        for code in list(codes):
            yield self._item_for(code)
            identity = self._identity(code.co_filename)
            yield from identity.bindings().items()
            if not identity.bindings():
                # Unparsable source: the raw skeleton is all it has.
                yield identity.module_item()

    def current(self, item: str) -> Optional[str]:
        """The hash `item` has in the code this process runs; None when it
        names nothing (a removed def or file)."""
        if not self._sealed:
            self.seal()
        if item == IMPORT_ITEM:
            return self.global_trace.id
        if item == ENGINE_ITEM:
            return None
        if item == LOCK_ITEM:
            return import_state.lock_hash(self._lock_file)
        rel, _, _qual = item.partition("::")
        identity = self._by_rel.get(rel, _UNSEEN)
        if identity is _UNSEEN:
            # Once per file: whether it is there, and what it holds.
            path = os.path.abspath(os.path.join(self._root, rel))
            identity = self._by_rel[rel] = (
                self._identity(path) if os.path.exists(path) else None)
        return None if identity is None else identity.items().get(item)

    def valid(self, trace: Trace) -> bool:
        known = self._valid.get(trace.id)
        if known is None:
            known = self._valid[trace.id] = all(
                self.current(item) == h for item, h in trace.items)
        return known




# ---------------------------------------------------------------------------
# Trace bodies, by id.
# ---------------------------------------------------------------------------


class TraceStore:
    """Trace bodies, content-addressed: an entry of any cache stores its
    trace's id, the body is kept once here. In memory, and in `directory`
    (one JSON file per trace) when there is one, so a trace outlives the
    process that recorded it — what a later process checks an entry
    against, and what the dry run (`tools.code_trace_dryrun`) reads."""

    def __init__(self, directory: Optional[str]) -> None:
        self.directory = directory
        self._mem: Dict[str, Trace] = {}
        self._lock = threading.Lock()

    def put(self, trace: Trace) -> str:
        tid = trace.id
        with self._lock:
            if tid in self._mem:
                return tid
            self._mem[tid] = trace
        if self.directory:
            path = os.path.join(self.directory, f"{tid}.json")
            if not os.path.exists(path):
                try:
                    os.makedirs(self.directory, exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(trace.to_json(), f)
                    os.replace(tmp, path)
                except OSError:
                    logger.debug("code trace %s not persisted", tid,
                                 exc_info=True)
        return tid

    def get(self, tid: str) -> Optional[Trace]:
        with self._lock:
            hit = self._mem.get(tid)
        if hit is not None or not self.directory:
            return hit
        try:
            with open(os.path.join(self.directory, f"{tid}.json"),
                      encoding="utf-8") as f:
                trace = Trace.from_json(json.load(f))
        except (OSError, ValueError, TypeError):
            return None
        if trace.id != tid:
            return None
        with self._lock:
            self._mem[tid] = trace
        return trace

    def ids(self) -> List[str]:
        """Every trace id kept here."""
        on_disk: set = set()
        if self.directory and os.path.isdir(self.directory):
            on_disk = {name[:-5] for name in os.listdir(self.directory)
                       if name.endswith(".json")}
        with self._lock:
            return sorted(on_disk | set(self._mem))


#: What is derived from source alone — each file's identity, the canonical
#: import-time trace — by a hash of the source (`HA_CODE_TRACE_CACHE`):
#: shareable by every process, session and checkout, never stale.
DERIVED = os.environ.get(
    "HA_CODE_TRACE_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingarch",
                 "code-trace-derived"))


def _write_json(path: str, value: Any) -> None:
    """`value` at `path`, atomically; a cache that cannot be written is
    only slower."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f)
        os.replace(tmp, path)
    except OSError:
        logger.debug("code_trace: %s not kept", path, exc_info=True)


#: Where trace bodies are kept. A cache entry on disk is served only while
#: its trace body can be read, so this must persist as long as the caches
#: whose entries name it (in production: on the same volume).
TRACE_DIR_ENV = "HA_CODE_TRACE_DIR"
#: The process's trace bodies (`TRACE_DIR_ENV`; default beside the other
#: local caches).
TRACES = TraceStore(os.environ.get(
    TRACE_DIR_ENV,
    os.path.join(os.path.expanduser("~"), ".cache", "huggingarch",
                 "code-traces")))


# ---------------------------------------------------------------------------
# The process's provider.
# ---------------------------------------------------------------------------

_PROVIDER: Any = None
_ENGINE_CODE_VERSION: Optional[Callable[[], str]] = None


def use_engine_code_version(fn: Callable[[], str]) -> None:
    """Name the global code hash `GlobalHash` traces are (`backend.
    fingerprint.engine_code_version`, which reads the engine trees this
    package must not import)."""
    global _ENGINE_CODE_VERSION
    _ENGINE_CODE_VERSION = fn


def _engine_code_version() -> str:
    """The engine's code hash, or — in a process that never imported the
    engine (`backend.fingerprint` names the hash when it is) — a marker that
    no engine code ran: what such a process computed depends on none, and a
    process that has the engine reads it as stale."""
    if _ENGINE_CODE_VERSION is None:
        return "engine-not-imported"
    return _ENGINE_CODE_VERSION()


def install() -> Any:
    """The process's provider, chosen by `HA_CODE_TRACE` once."""
    global _PROVIDER
    if _PROVIDER is None:
        mode = os.environ.get(ENV, "global")
        if mode == "functions":
            _PROVIDER = TracedFunctions(strict=bool(os.environ.get(STRICT_ENV)))
        elif mode == "global":
            _PROVIDER = GlobalHash(_engine_code_version)
        else:
            raise ValueError(f"{ENV}={mode!r}: expected 'global' or 'functions'")
    return _PROVIDER


provider = install


#: The computations open in this context, outermost first. A context, not a
#: thread: requests are coroutines sharing the event loop's thread, and a
#: thread a computation hands work to (`asyncio.to_thread`) runs in a copy
#: of its context.
_COMPUTATIONS: contextvars.ContextVar[Tuple[Recording, ...]] = \
    contextvars.ContextVar("code_trace_computations", default=())


def _opening_code() -> Any:
    """The code of the frame that opened the recording being begun: the
    first frame outward from here outside this package and `contextlib`
    (`recording`'s with block)."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if not (name.startswith(_SELF) or name == contextlib.__file__):
            return frame.f_code
        frame = frame.f_back
    return None


def seal() -> None:
    """Fix the code this process runs (`TracedFunctions.seal`: import the
    traced trees, read the import-time trace) now rather than on the first
    recorded computation: a server's lifespan, a pool worker's start."""
    provider().seal()


def current() -> Tuple[Recording, ...]:
    """The computations open in this context, outermost first."""
    return _COMPUTATIONS.get()


@contextlib.contextmanager
def recording() -> Iterator[Dict[str, Trace]]:
    """Record one computation: `with recording() as out: ...`, then
    `out["trace"]`. Inside it, a cache `put` that names no trace stores the
    trace recorded so far (`snapshot`)."""
    p = provider()
    rec = p.begin()
    token = _COMPUTATIONS.set((*_COMPUTATIONS.get(), rec))
    out: Dict[str, Trace] = {}
    try:
        yield out
    finally:
        try:
            # Ended while still open in this context: its trace is formed
            # knowing which computations enclose it (`_trace_of`).
            out["trace"] = p.end(rec)
        finally:
            _COMPUTATIONS.reset(token)


def run_recorded(inherited: Optional[Trace], fn: Callable[..., Any],
                 *args: Any, **kwargs: Any) -> Tuple[Any, Trace]:
    """`fn(*args, **kwargs)` as one computation in this (worker) process,
    and its trace. `inherited` is the trace of the computation that handed
    the job over, as far as it had got: what the job's arguments were
    computed by, so a value the job stores names it too."""
    with recording() as rec:
        provider().merge(inherited, into=current()[-1:])
        value = fn(*args, **kwargs)
    return value, rec["trace"]
