"""The code half of a cached result's version (`backend.code_trace.provider`)
and the memos that keep it honest (`backend.code_trace.memo`).

`TracedFunctions` is exercised on a throwaway package in a fresh process —
the provider has to be the first thing the process installs, and what it
names must not move when `backend/` does. Each process prints what it saw as
one JSON line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PACKAGE = {
    "tpkg/__init__.py": "",
    "tpkg/a.py": '''
        from backend.code_trace.memo import Memo, memo_property, memoize


        def helper(x):
            return x + 1


        def other(x):
            return x * 2


        def plain():
            return 0


        def keyed(x):
            return x


        @memoize(maxsize=None)
        def cached(x):
            return helper(x)


        class C:
            @memo_property
            def p(self):
                return other(2)


        TABLE = Memo(maxsize=4)


        def via_table(k):
            return TABLE.get(k, lambda: other(k))


        def fresh_b():
            return 1


        def keyed_a():
            return 2


        def helper2(x):
            return x - 1


        @memoize(maxsize=None)
        def cached2(x):
            return helper2(x)


        def opener(record):
            with record() as rec:
                pass
            return rec


        def deep(x):
            return x + 3


        @memoize(maxsize=None)
        def inner(x):
            return deep(x)


        @memoize(maxsize=None)
        def outer(x):
            return inner(x) * 2


        def from_a_route(record):
            return opener(record)


        def from_a_warm_job(record):
            return opener(record)


        def helper3(x):
            return x * 3


        def gen():
            yield 1
            yield helper3(2)
    ''',
    "tpkg/b.py": '''
        import sympy


        class Quotient(sympy.Expr):
            def reason(self):
                return "no"


        class Refused(Exception):
            def why(self):
                return "stdlib"


        def halved(x):
            return x / 2


        class Half(sympy.Function):
            """SymPy calls `eval` from `Function.__new__`, which its own cache
            (`cacheit`) answers on a second call without calling it — nor
            `halved`, which rule 3 does not name."""

            @classmethod
            def eval(cls, x):
                if x.is_Integer:
                    return halved(x)
    ''',
    "late.py": '''
        def f():
            return 1
    ''',
}

DRIVER = '''
import json, sys
sys.path[:0] = [{src!r}, {repo!r}]
from backend.code_trace import provider as P
p = P._PROVIDER = P.TracedFunctions(root={src!r}, trees=("tpkg",),
                                    strict=True, lock_file="")
from backend.code_trace.provider import Growing, Trace, recording
mode = sys.argv[1]
if mode == "check":
    trace = Trace.from_json(json.loads(sys.argv[2]))
    print(json.dumps({{"valid": p.valid(trace)}}))
    raise SystemExit
import sympy
import tpkg.a as a
import tpkg.b as b
out = {{}}

def names(rec):
    return sorted(k for k, _ in rec["trace"].items)

with recording() as r:
    a.cached(1)
out["miss"] = names(r)
with recording() as r:
    a.cached(1)
out["hit"] = names(r)
c = a.C()
with recording() as r:
    c.p
with recording() as r2:
    c.p
out["property_hit"] = names(r2)
with recording() as r:
    a.via_table(3)
with recording() as r2:
    a.via_table(3)
out["table_hit"] = names(r2)
with recording() as r:
    with p.untraced():
        a.keyed(1)
out["untraced"] = names(r)
g = Growing()
with recording() as r:
    p.attach(g)
    a.plain()
with recording() as r2:
    p.attach(g)
out["growing"] = names(r2)
with recording() as r:
    b.Quotient().reason()
    b.Refused().why()
out["foreign_in_trace"] = names(r)
# Rule 3's route: the same SymPy call cold, then answered by SymPy's cache.
with recording() as cold:
    b.Half(sympy.Integer(6))
with recording() as hot:
    b.Half(sympy.Integer(6))
out["sympy_cold"] = [list(i) for i in cold["trace"].items]
out["sympy_hot"] = [list(i) for i in hot["trace"].items]
out["foreign_in_global"] = [k for k, _ in p.global_trace.items
                            if "Quotient." in k or "Refused." in k]
# Bug A: thread A's untraced window must not hide what thread B runs.
import threading
b_began, a_inside, b_called = (threading.Event() for _ in range(3))
seen = {{}}

def thread_a():
    with recording() as ra:
        b_began.wait(10)
        with p.untraced():
            a.keyed_a()
            a_inside.set()
            b_called.wait(10)
    seen["a"] = names(ra)

def thread_b():
    with recording() as rb:
        b_began.set()
        a_inside.wait(10)
        a.fresh_b()
        b_called.set()
    seen["b"] = names(rb)

threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
for t in threads:
    t.start()
for t in threads:
    t.join()
out["thread_a"], out["thread_b"] = seen["a"], seen["b"]
# A memo missed inside an untraced window still records its own call.
with recording() as r:
    with p.untraced():
        a.cached2(1)
with recording() as r2:
    a.cached2(1)
out["untraced_memo_hit"] = names(r2)
# A memo whose miss hit another memo: a later hit names both misses' code.
with recording() as r:
    a.inner(7)
with recording() as r:
    a.outer(7)
with recording() as r:
    a.outer(7)
out["nested_memo_hit"] = names(r)
# Bug B: the frame that opened a recording, and a generator made before it.
out["opener"] = names(a.opener(recording))
# One computation, reached from two callers: one trace.
out["via_route"] = names(a.from_a_route(recording))
out["via_warm"] = names(a.from_a_warm_job(recording))
g = a.gen()
next(g)
with recording() as r:
    next(g)
out["resumed"] = names(r)
with recording() as r:
    a.helper(1); a.other(1)
out["trace"] = r["trace"].to_json()
try:
    with recording() as r:
        import late
    out["late"] = "accepted"
except RuntimeError as e:
    out["late"] = "refused"
print(json.dumps(out))
'''


def _write(tmp: Path, files: dict) -> None:
    """The traced code goes under `tmp/src`; the driver beside it, outside
    what is traced."""
    for rel, src in files.items():
        path = tmp / "src" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src).lstrip(), encoding="utf-8")


def _run(tmp: Path, *args: str) -> dict:
    script = tmp / "driver.py"
    script.write_text(DRIVER.format(src=str(tmp / "src"), repo=str(ROOT)),
                      encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "HA_CODE_TRACE"}
    out = subprocess.run([sys.executable, str(script), *args], cwd=tmp,
                         env=env, capture_output=True, text=True, timeout=300,
                         check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("traced")
    _write(tmp, PACKAGE)
    return tmp, _run(tmp, "record")


def test_a_memo_hit_names_the_code_its_miss_ran(traced):
    _tmp, out = traced
    assert {"tpkg/a.py::cached", "tpkg/a.py::helper"} <= set(out["miss"])
    assert {"tpkg/a.py::cached", "tpkg/a.py::helper"} <= set(out["hit"])
    assert "tpkg/a.py::other" in out["property_hit"]
    assert "tpkg/a.py::other" in out["table_hit"]


def test_a_memo_hit_names_the_memos_its_miss_hit(traced):
    """`outer`'s miss hit `inner`, filled before; a hit on `outer` names
    what `inner`'s miss ran too (`provider.Fragment`'s parts)."""
    _tmp, out = traced
    assert {"tpkg/a.py::outer", "tpkg/a.py::inner", "tpkg/a.py::deep"} <= set(
        out["nested_memo_hit"])


def test_untraced_code_is_in_no_trace(traced):
    _tmp, out = traced
    assert "tpkg/a.py::keyed" not in out["untraced"]


def test_a_growing_entry_carries_what_its_earlier_readers_ran(traced):
    _tmp, out = traced
    assert "tpkg/a.py::plain" in out["growing"]


def test_a_method_of_a_class_with_a_foreign_base_is_import_time(traced):
    # Rule 3: third-party machinery may memoise calls into such a method,
    # so it is named by every trace, now by its own item where the one
    # import-time aggregate used to cover it (that aggregate moved on any
    # addition anywhere). A standard-library base (an exception) memoises
    # no such calls: its methods stay per recording.
    _tmp, out = traced
    assert out["foreign_in_global"] == ["tpkg/b.py::Quotient.reason"]
    assert "tpkg/b.py::Quotient.reason" in out["foreign_in_trace"]
    assert "tpkg/b.py::Refused.why" in out["foreign_in_trace"]
    assert "<import-time>" not in out["foreign_in_trace"]


def test_another_threads_untraced_window_hides_nothing_this_thread_runs(traced):
    # Thread B's first call to `fresh_b` lands inside thread A's untraced
    # window; B's recording must still name it, and A's must not name what
    # A ran inside the window.
    _tmp, out = traced
    assert "tpkg/a.py::fresh_b" in out["thread_b"]
    assert "tpkg/a.py::keyed_a" not in out["thread_a"]


def test_a_memo_missed_inside_an_untraced_window_records_its_call(traced):
    _tmp, out = traced
    assert {"tpkg/a.py::cached2", "tpkg/a.py::helper2"} <= set(
        out["untraced_memo_hit"])


def test_code_already_running_when_a_recording_opens_is_in_it(traced):
    # The function that opened the recording keeps executing inside it; a
    # generator made before it resumes inside it.
    _tmp, out = traced
    assert "tpkg/a.py::opener" in out["opener"]
    assert {"tpkg/a.py::gen", "tpkg/a.py::helper3"} <= set(out["resumed"])


def test_a_third_party_cache_hit_hides_no_code_of_ours(traced):
    """SymPy's own cache answers the second `Half(6)` without calling
    `Half.eval` or the `halved` it calls: what the cold call named, the
    cached one must too — the method through rule 3 (the import-time
    trace), what it calls through the foreign closure."""
    _tmp, out = traced
    cold = {tuple(i) for i in out["sympy_cold"]}
    hot = {tuple(i) for i in out["sympy_hot"]}
    assert "tpkg/b.py::halved" in {k for k, _h in cold}
    assert sorted(cold - hot) == []


def test_a_computation_names_the_same_code_whoever_called_it(traced):
    """The callers above the frame that opened a recording are suspended
    while it is open: none of them is in it, so one computation reached
    from a route and from a warm job is one trace."""
    _tmp, out = traced
    assert out["via_route"] == out["via_warm"]
    assert "tpkg/a.py::opener" in out["via_route"]


def test_a_module_imported_after_the_traced_trees_is_refused(traced):
    _tmp, out = traced
    assert out["late"] == "refused"


@pytest.mark.parametrize("edit, valid", [
    # A body the trace names: stale.
    (("tpkg/a.py", "return x + 1", "return x + 2"), False),
    # A body it does not name: still valid.
    (("tpkg/a.py", "return 0", "return 5"), True),
    # A comment or docstring is no semantic edit.
    (("tpkg/a.py", "def other(x):\n", "def other(x):  # doubled\n"), True),
    # A new def binds a new name at import, but an old trace names no
    # such item: it stays valid. (The one import-time aggregate this
    # replaced moved on the addition and took every record with it.)
    (("tpkg/a.py", "def plain():", "def extra():\n    return 1\n\n\ndef plain():"),
     True),
    # A decorated def may write into a registry old code reads: the
    # module's decorator aggregate moves.
    (("tpkg/a.py", "def plain():",
      "@memoize(maxsize=None)\ndef extra():\n    return 1\n\n\ndef plain():"),
     False),
    # A new top-level binding runs at import but under a name no old
    # code reads: a new item, not a move.
    (("tpkg/a.py", "def helper(x):", "TABLE2 = Memo(maxsize=4)\n\n\ndef helper(x):"),
     True),
    # One more import alias is a new item; the old aliases do not move.
    (("tpkg/a.py", "from backend.code_trace.memo import",
      "from contextlib import suppress\n\nfrom backend.code_trace.memo import"),
     True),
    # A top-level statement that binds no name runs at import all the
    # same: the module's statement aggregate moves.
    (("tpkg/a.py", "def helper(x):", "TABLE.warm = True\n\n\ndef helper(x):"),
     False),
    # So does an edit to a method of a class with a foreign base (rule 3).
    (("tpkg/b.py", 'return "no"', 'return "yes"'), False),
    # An undecorated method is an item of its own: adding one is an
    # addition, not a move.
    (("tpkg/a.py", "class C:",
      "class C:\n    def added(self):\n        return 9\n"),
     True),
])
def test_a_trace_is_stale_exactly_when_code_it_names_moved(traced, tmp_path,
                                                             edit, valid):
    src_tmp, out = traced
    files = {rel: (src_tmp / "src" / rel).read_text(encoding="utf-8")
             for rel in PACKAGE}
    rel, old, new = edit
    assert old in files[rel], edit
    files[rel] = files[rel].replace(old, new, 1)
    for path, text in files.items():
        dest = tmp_path / "src" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    # Same relative paths, so the trace's names read against the copy.
    assert _run(tmp_path, "check", json.dumps(out["trace"]))["valid"] is valid


def test_global_hash_is_todays_engine_code_version():
    """Under `GlobalHash` every trace is the one global hash, byte for byte
    the one cache keys folded in before traces existed."""
    from backend.code_trace.provider import ENGINE_ITEM, provider
    from backend.fingerprint import engine_code_version

    p = provider()
    if p.traced:
        pytest.skip("this process runs TracedFunctions")
    rec = p.begin()
    trace = p.end(rec)
    assert trace.items == ((ENGINE_ITEM, engine_code_version()),)
    assert p.valid(trace)
