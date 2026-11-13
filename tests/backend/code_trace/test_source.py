"""What a verifying trace names code by (`backend.code_trace.source.
FileIdentity`): each outermost def by its normalized source, and the
module's import-time state per top-level binding, with the whole-file
skeleton kept as one legacy item."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.code_trace.source import FileIdentity

SRC = '''"""Module doc."""
import os

LIMIT = 3


def helper(x):
    """Doc."""
    def inner(y):
        return y + 1
    return inner(x) + LIMIT


class K:
    field: int = 1

    @staticmethod
    def make():
        return [i for i in range(LIMIT)]

    def other(self):
        return lambda: 2
'''


def _items(src: str) -> dict:
    return FileIdentity("m.py", src).items()


def test_each_def_and_binding_and_the_legacy_skeleton_is_one_item():
    assert set(_items(SRC)) == {
        "m.py::helper", "m.py::K.make", "m.py::K.other",
        "m.py::<binding>:os", "m.py::<binding>:LIMIT", "m.py::<binding>:K",
        "m.py::<binding>:<decorators>", "m.py::<binding>:<flow>",
        "m.py::<binding>:<stmt>", "m.py::<module>"}


def test_code_inside_a_def_belongs_to_it_and_code_outside_to_the_skeleton():
    ident = FileIdentity("m.py", SRC)
    lines = SRC.splitlines()
    inner = next(i for i, l in enumerate(lines, 1) if "def inner" in l)
    lam = next(i for i, l in enumerate(lines, 1) if "lambda" in l)
    decorated = next(i for i, l in enumerate(lines, 1) if "@staticmethod" in l)
    assert ident.item_of(inner)[0] == "m.py::helper"
    assert ident.item_of(lam)[0] == "m.py::K.other"
    # A decorated def's code starts at its first decorator.
    assert ident.item_of(decorated)[0] == "m.py::K.make"
    # Module-level code outside every def names the binding statement it
    # is, not the whole file.
    assert ident.item_of(4)[0] == "m.py::<binding>:LIMIT"


def _moved(edited: str) -> set:
    before, after = _items(SRC), _items(edited)
    return {k for k in before.keys() | after.keys()
            if before.get(k) != after.get(k)}


def test_a_body_edit_moves_only_its_def():
    assert _moved(SRC.replace("return y + 1", "return y + 2")) == {"m.py::helper"}


def test_comments_docstrings_and_blank_lines_move_nothing():
    edited = (SRC.replace('"""Doc."""', '"""Other doc."""')
              .replace("LIMIT = 3", "LIMIT = 3  # three")
              .replace("class K:\n", "class K:\n\n"))
    assert _moved(edited) == set()


def test_a_class_body_edit_moves_its_binding_and_a_new_def_nothing():
    # The class is one binding item with its methods excised; a field edit
    # moves it. A new def is a new item: the legacy skeleton alone moves.
    assert _moved(SRC.replace("field: int = 1", "field: str = 1")) == {
        "m.py::<binding>:K", "m.py::<module>"}
    assert _moved(SRC + "\n\ndef extra():\n    return 0\n") == {
        "m.py::<module>", "m.py::extra"}


def test_a_signature_or_decorator_edit_moves_its_def():
    assert _moved(SRC.replace("def helper(x):", "def helper(x, y=0):")) == {
        "m.py::helper"}
    assert "m.py::K.make" in _moved(SRC.replace("@staticmethod", "@classmethod"))
