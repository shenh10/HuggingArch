"""The import-time state a trace names, apart from the recording that
names it (`provider`) and the source identity it reads (`source`).

A computation inherits what the traced trees' import left behind —
constants, registries, registrations — without running any code of its
own at import. One aggregate item over the whole import-time trace
covered that once; it also moved on any addition anywhere in the trees (a
skeleton hash counts a def), and with it every record of a warm corpus.
The whole concern lives here instead:

* `module_state` — one module's import-time state as per-top-level-binding
  items (`BINDING`): an addition is a new item, an edit to what an
  existing binding binds is a move;
* `ImportGraph` — the traced trees' static import graph, and the binding
  items of the named modules' closure: state a registry read reaches;
* `lock_hash` — the lock file, one item every trace names.

`source` stays below this (it is stdlib only); `FileIdentity` asks here
for its bindings, lazily, to keep that layering one-way.
"""
from __future__ import annotations

import ast
import os
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .source import FileIdentity, digest

#: How a module's import-time state is named: `rel::<binding>:NAME`, one
#: item per top-level binding (plus the aggregates `module_state` keeps).
BINDING = "<binding>:"

State = Tuple[Dict[str, str], List[Tuple[int, int, str]],
              List[Tuple[int, Optional[str], List[str]]]]


def module_state(rel: str, tree: ast.Module, lines: list, semantic: list) -> State:
    """(bindings, top-level statement spans, imports) of one parsed module.

    A binding's hash is the semantic source of every top-level statement
    that binds the name, in order (a rebinding joins the first). A class's
    methods are excised from its item — they are defs, items of their own —
    while its decorators stay out of it and in the `<decorators>` aggregate
    with every other decorator at import scope: a decorator may write into
    module state a computation reads without running the def it decorates.
    Imports name each alias by where it is imported from, so one more alias
    on a line is a new item and a different module for an old one moves it.
    `if`/`try`/`for` and expression statements bind nothing a span can
    name; each kind shares one aggregate item, moved by any change or
    addition among them."""
    bindings: Dict[str, str] = {}
    top: List[Tuple[int, int, str]] = []
    imports: List[Tuple[int, Optional[str], List[str]]] = []
    statements: Dict[str, List[str]] = {}
    aggregates: Dict[str, List[str]] = {"<decorators>": [], "<flow>": [],
                                        "<stmt>": []}

    def text_of(node) -> str:
        return "\n".join(
            text for text in semantic[node.lineno - 1:node.end_lineno]
            if text is not None)

    def names_of(target) -> list:
        return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # its def item is the binding; an addition is new
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            level = getattr(node, "level", 0) or 0
            module = getattr(node, "module", None)
            imports.append((level, module, [a.name for a in node.names]))
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                origin = ("." * level
                          + (module + "." if module else "") + alias.name)
                item = f"{rel}::<binding>:{name}"
                bindings[item] = digest("import " + origin)
                top.append((node.lineno, node.end_lineno, item))
            continue
        if isinstance(node, ast.ClassDef):
            body = [text_of(child) for child in node.body
                    if not isinstance(child, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))]
            header = lines[node.lineno - 1]
            indent = header[:len(header) - len(header.lstrip())]
            item = f"{rel}::<binding>:{node.name}"
            bindings[item] = digest("\n".join(
                [indent + "class " + node.name + ":", *body]))
            top.append((node.lineno, node.end_lineno, item))
            continue
        names: List[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names += names_of(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names += names_of(node.target)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                names += names_of(target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names += names_of(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item_ in node.items:
                if item_.optional_vars is not None:
                    names += names_of(item_.optional_vars)
        elif isinstance(node, ast.Expr):
            aggregates["<stmt>"].append(text_of(node))
        else:
            aggregates["<flow>"].append(text_of(node))
        if names:
            for name in sorted(set(names)):
                statements.setdefault(name, []).append(text_of(node))
            top.append((node.lineno, node.end_lineno,
                        f"{rel}::<binding>:{sorted(set(names))[0]}"))

    for name, parts in statements.items():
        bindings[f"{rel}::<binding>:{name}"] = digest("\n".join(parts))

    def decorated(scope, into: list) -> None:
        for child in scope.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                into.extend(text_of(d) for d in child.decorator_list)
            if isinstance(child, ast.ClassDef):
                decorated(child, into)
    decorated(tree, aggregates["<decorators>"])
    for name, parts in aggregates.items():
        bindings[f"{rel}::<binding>:{name}"] = digest("\n".join(parts))
    return bindings, top, imports


def lock_hash(path: Optional[str]) -> Optional[str]:
    """The lock file's hash, or None without one."""
    if not (path and os.path.exists(path)):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return digest(f.read())


def tree_rels(root: str, trees: Tuple[str, ...],
              excluded: FrozenSet[str]) -> FrozenSet[str]:
    """The traced trees' files, as items name them (relative to the backend
    root), the excluded stores aside."""
    prefix = os.path.basename(root)
    rels: set = set()
    for tree in trees:
        parts = tree.split(".")
        if len(parts) < 2 or parts[0] != prefix:
            continue
        directory = os.path.join(root, "/".join(parts[1:]))
        if os.path.isdir(directory):
            for found, _dirs, files in os.walk(directory):
                for name in files:
                    if name.endswith(".py"):
                        rels.add(os.path.relpath(
                            os.path.join(found, name),
                            root).replace(os.sep, "/"))
    return frozenset(rels - excluded)


class ImportGraph:
    """The traced trees' static import graph, file to file.

    `FileIdentity` keeps each file's imports as written; here each is
    resolved against the trees' module names (an edge for the module
    imported and for the name imported from it, whichever is a file of the
    trees). `loader` reads a file's identity by absolute path — the
    provider's own `_identity`, so the two share its cache."""

    def __init__(self, root: str, trees: Tuple[str, ...],
                 loader: Callable[[str], FileIdentity],
                 excluded: FrozenSet[str] = frozenset()) -> None:
        self._root = os.path.abspath(root)
        self._loader = loader
        self.rels = tree_rels(self._root, trees, excluded)
        self._by_name = self._names_of(self.rels, os.path.basename(self._root))
        self._edges = {
            rel: frozenset(self._targets(rel))
            for rel in self.rels}

    @staticmethod
    def _names_of(rels: FrozenSet[str], prefix: str) -> Dict[str, str]:
        names: Dict[str, str] = {}
        for rel in rels:
            module = prefix + "." + rel[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            names[module] = rel
        return names

    def _module_of(self, rel: str) -> str:
        prefix = os.path.basename(self._root)
        module = prefix + "." + rel[:-3].replace("/", ".")
        return module.removesuffix(".__init__")

    def _targets(self, rel: str) -> list:
        identity = self.identity(rel)
        here = self._module_of(rel)
        out = []
        for level, module, aliases in ((identity._imports if identity else ()) or ()):
            base = module or ""
            if level:
                package = here
                for _ in range(level):
                    package = package.rsplit(".", 1)[0]
                base = (package + ("." + module if module else "")).strip(".")
            if base in self._by_name:
                out.append(self._by_name[base])
            for alias in aliases:
                name = f"{base}.{alias}" if base else alias
                out.extend(self._by_name[candidate]
                           for candidate in (name, name.rsplit(".", 1)[0])
                           if candidate in self._by_name)
        return [target for target in out if target != rel]

    def identity(self, rel: str) -> Optional[FileIdentity]:
        """The file's identity, or None where the tree has none."""
        path = os.path.abspath(os.path.join(self._root, rel))
        return self._loader(path) if os.path.exists(path) else None

    def closure(self, seeds: Iterable[str]) -> FrozenSet[str]:
        """The files reachable from `seeds` through the graph's edges."""
        seen, pending = set(seeds), list(seeds)
        while pending:
            for target in self._edges.get(pending.pop(), ()):
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        return frozenset(seen)

    def state_items(self, items: Dict[str, str],
                    lock_file: Optional[str]) -> Dict[str, str]:
        """The import-time state the code `items` names can reach: the
        binding items of every tree module in the named modules' import
        closure, and the lock file.

        The closure is static and narrow: code that reads a module's
        import-time state (a registry its import builds) imports that
        module, and state nothing the computation names imports or runs
        cannot reach its result."""
        seeds = {item.split("::", 1)[0] for item in items
                 if "::" in item} & self.rels
        state: Dict[str, str] = {}
        for rel in self.closure(seeds):
            identity = self.identity(rel)
            if identity is not None:
                state.update(identity.bindings())
        held = lock_hash(lock_file)
        if held is not None:
            state["<uv.lock>"] = held
        return state
