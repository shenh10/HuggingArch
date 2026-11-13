"""Source identity: what counts as a semantic edit of Python source.

`normalized_py_source` is the one definition of it. The global engine hash
(`backend.fingerprint.engine_code_version`) hashes whole files through it, and
the per-function hashes of a verifying trace (`function_table`) hash each
def through it, so a comment, docstring or formatting edit moves neither.
Stdlib only: this sits below every package it describes.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, Tuple


def _docstring_spans(tree: Any, lines: list) -> list:
    """(start, end) (lineno, CHAR col) spans of every module/class/function
    docstring. Docstrings are documentation, not semantics — a doc typo fix
    must not invalidate every cached result. (One engine file reads
    ``__doc__`` for a CLI help string — no cached *result* depends on it.)

    ast col offsets are UTF-8 BYTE offsets while tokenize / str indexing are
    char-based — without the conversion, a non-ASCII docstring over-blanks
    and erases the semantic code beside it from the hash."""
    import ast

    def _char_col(lineno: int, byte_col: int) -> int:
        return len(lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8"))

    spans = []
    # Only a Module / ClassDef / FunctionDef holds a docstring, and Python's
    # grammar puts no statement inside an expression — so no def or class is
    # reachable through an `ast.expr`, and an expression subtree cannot hold
    # a docstring. Descending into one only costs: `ast.walk` visited all
    # 523k AST nodes of the engine trees to find 3,028 docstrings, where the
    # statements alone are 58k. Order is not part of the answer —
    # `normalized_py_source` only ever blanks these spans, and blanking is
    # order-independent.
    pending = [tree]
    while pending:
        node = pending.pop()
        pending.extend(child for child in ast.iter_child_nodes(node)
                       if not isinstance(child, ast.expr))
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            e = body[0].value
            spans.append((
                (e.lineno, _char_col(e.lineno, e.col_offset)),
                (e.end_lineno, _char_col(e.end_lineno, e.end_col_offset)),
            ))
    return spans


def normalized_py_source(src: str) -> str:
    """Semantic form of a Python source file: comments, docstrings, trailing
    whitespace and blank lines removed. Two files that differ only in those
    normalize identical — so a comment typo fix or a reformat no longer
    invalidates every cached engine result, while any change to names,
    literals, or structure still does. Lines that are (partly) inside a
    multi-line STRING LITERAL are content, not formatting: their text is
    kept byte-exact (a blank line inside ``X = \"\"\"a\\n\\nb\"\"\"`` is a
    different value than without it).

    Version-stable by construction: positions come from ``ast.parse`` +
    ``tokenize`` of the same source on the one supported interpreter
    (``requires-python``), and both agree on docstring spans and COMMENT
    tokens — a hard requirement now that cache keys are shared across
    machines over the Supabase tier. Unparseable source falls back to the raw bytes (a
    broken file still hashes distinctly)."""
    lines = semantic_lines(src)
    if lines is None:
        return "\x00raw\x00" + src
    return "\n".join(text for text in lines if text is not None)


def semantic_lines(src: str, tree: Any = None) -> "list | None":
    """`normalized_py_source`, line for line: each source line's semantic
    text, or None where the line normalizes away. None for source that does
    not parse. A def's lines are its slice of this, so a file is normalized
    once however many of its defs are asked for."""
    import ast
    import io
    import tokenize

    try:
        if tree is None:
            tree = ast.parse(src)
        raw_lines = src.splitlines()
        spans = _docstring_spans(tree, raw_lines)
        doc_spans = set(spans)
        # Interior of a multi-line string literal: those lines' whitespace IS
        # the value. Covers plain strings and f-strings (which lex as
        # FSTRING_START…FSTRING_END).
        protected: set = set()
        fstring_start = None
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                spans.append((tok.start, tok.end))
            elif tok.type == tokenize.STRING:
                if (tok.start, tok.end) not in doc_spans and tok.start[0] < tok.end[0]:
                    # From the opening line (its tail is literal content) to
                    # the line before the closing quotes.
                    protected.update(range(tok.start[0], tok.end[0]))
            elif tok.type == tokenize.FSTRING_START:
                fstring_start = tok.start[0]
            elif tok.type == tokenize.FSTRING_END and fstring_start is not None:
                if fstring_start < tok.end[0]:
                    protected.update(range(fstring_start, tok.end[0]))
                fstring_start = None
    except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
        return None

    # Spanned text becomes spaces (so a docstring or comment contributes
    # nothing but the layout around it). Only the lines a span touches are
    # rewritten: of the engine's 105k lines, 21k are docstring and the rest
    # are returned as read, so splitting every line into a list of single
    # characters to blank those did the work of 4.9 MB to edit 0.8 MB.
    blanked: Dict[int, str] = {}
    for (l1, c1), (l2, c2) in spans:
        for ln in range(l1, l2 + 1):
            text = blanked.get(ln)
            if text is None:
                text = blanked[ln] = raw_lines[ln - 1]
            lo = c1 if ln == l1 else 0
            # Blanking never changes a line's length, so `len(text)` is the
            # raw line's length however many spans already landed on it.
            hi = min(c2 if ln == l2 else len(text), len(text))
            if hi > lo:
                blanked[ln] = text[:lo] + " " * (hi - lo) + text[hi:]
    out: list = []
    for i, raw in enumerate(raw_lines):
        text = blanked.get(i + 1, raw)
        if (i + 1) in protected:
            out.append(text)  # literal content — byte-exact, even blank
            continue
        stripped = text.rstrip()
        out.append(stripped or None)
    return out


# ---------------------------------------------------------------------------
# Per-function identity: what a verifying trace names code by.
# ---------------------------------------------------------------------------


def digest(text: str) -> str:
    """The short content hash every identity here is written in."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class FileIdentity:
    """One source file as a verifying trace names it.

    Each outermost def — a function or method not nested in another def —
    is an item: its qualname, and the hash of its dedented source through
    `normalized_py_source` (decorators, signature and body; the defs,
    lambdas and comprehensions inside it are part of it). What the
    module's import leaves behind is named per top-level binding
    (`<binding>:NAME`), with three aggregates — `<decorators>`, `<flow>`,
    `<stmt>` — for what runs at import without binding a name a statement
    span can name. Adding a def, an import alias or a class is then a new
    item: a corpus recorded before the addition stays valid, where the
    one whole-file skeleton it replaces moved on any addition and took
    every record with it. Editing what an existing binding binds, or
    removing a def a trace names, still moves what that trace names; so
    does adding a decorated def, whose decorator may write into a
    registry old code reads. The whole-file skeleton remains one legacy
    item, for traces recorded before this and as the fallback for code no
    statement span names."""

    def __init__(self, rel: str, src: str) -> None:
        import ast
        import textwrap

        self.rel = rel
        self._starts: list = []
        self._defs: list = []
        self._bindings: dict = {}
        self._top: list = []       # (start, end, item) of top-level stmts
        self._imports: list = []   # (level, module, alias names)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            self.skeleton = digest("\x00raw\x00" + src)
            return
        lines = src.splitlines()
        found: list = []

        def visit(body: list, prefix: str) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno]
                                + [d.lineno for d in node.decorator_list])
                    found.append((start, node, prefix + node.name))
                elif isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")
                else:
                    for name in ("body", "orelse", "finalbody"):
                        visit(getattr(node, name, None) or [], prefix)
                    for part in (getattr(node, "handlers", None) or []) \
                            + (getattr(node, "cases", None) or []):
                        visit(part.body, prefix)

        visit(tree.body, "")
        found.sort(key=lambda f: f[0])
        semantic = semantic_lines(src, tree)
        if semantic is None:
            self.skeleton = digest("\x00raw\x00" + src)
            return
        for start, node, qual in found:
            snippet = textwrap.dedent("\n".join(
                text for text in semantic[start - 1:node.end_lineno]
                if text is not None))
            self._starts.append(start)
            self._defs.append((node.end_lineno, qual, digest(snippet)))
        # The import-time state names what the module's import leaves
        # behind — `import_state`, imported here to keep this module the
        # one `backend.code_trace` still imports with the stdlib alone.
        from backend.code_trace.import_state import module_state
        self._bindings, self._top, self._imports = module_state(
            self.rel, tree, lines, semantic)
        skeleton = list(semantic)
        for _start, node, _qual in reversed(found):
            header = lines[node.lineno - 1]
            indent = header[:len(header) - len(header.lstrip())]
            skeleton[node.lineno - 1:node.end_lineno] = [
                f"{indent}def {node.name}(): ..."]
        self.skeleton = digest("\n".join(
            text for text in skeleton if text is not None))

    def to_json(self) -> Dict[str, Any]:
        """What this identity is, apart from where the file lives."""
        return {"v": 2, "starts": self._starts, "defs": self._defs,
                "skeleton": self.skeleton, "bindings": self._bindings,
                "top": self._top, "imports": self._imports}

    @classmethod
    def from_json(cls, rel: str, data: Dict[str, Any]) -> "FileIdentity":
        if data.get("v") != 2:
            # A cache of the one-skeleton format: the binding items it
            # lacks are not optional, so it is a miss, not an identity.
            raise ValueError("a file identity of an unknown format")
        identity = cls.__new__(cls)
        identity.rel = rel
        identity._starts = list(data["starts"])
        identity._defs = [tuple(d) for d in data["defs"]]
        identity.skeleton = data["skeleton"]
        identity._bindings = dict(data["bindings"])
        identity._top = [tuple(t) for t in data["top"]]
        identity._imports = [tuple(i) for i in data["imports"]]
        return identity

    def item_of(self, line: int) -> Tuple[str, str]:
        """The item of code that starts at `line`: its outermost def's, or
        the top-level binding statement's for code outside every def (a
        module-level lambda, a class body); the whole-file skeleton is the
        fallback."""
        i = bisect.bisect_right(self._starts, line) - 1
        if i >= 0:
            end, qual, h = self._defs[i]
            if line <= end:
                return f"{self.rel}::{qual}", h
        for start, end, item in self._top:
            if start <= line <= end and item in self._bindings:
                return item, self._bindings[item]
        return self.module_item()

    def module_item(self) -> Tuple[str, str]:
        return f"{self.rel}::<module>", self.skeleton

    def bindings(self) -> Dict[str, str]:
        """The module's import-time state, per top-level binding
        (`backend.code_trace.import_state`)."""
        return self._bindings

    def items(self) -> Dict[str, str]:
        """Every item of the file — built once; read, never written."""
        out = self.__dict__.get("_items")
        if out is None:
            out = {f"{self.rel}::{qual}": h for _end, qual, h in self._defs}
            out.update(self._bindings)
            out.update((self.module_item(),))
            self.__dict__["_items"] = out
        return out
