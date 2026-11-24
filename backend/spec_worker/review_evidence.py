"""Mechanical check of a review report's evidence against the run's snapshot.

A review is an agent's reading of the model. Its claims about numbers (a
window, a layer count, a tensor shape) and about source (what a line of the
modeling file does) can be checked without another agent: the run already
holds config.json, the checkpoint tensor index and the staged source files.
This module parses the report, checks every claim it can, and classifies
each finding before the verdict is stored and before Agent Fix sees it:

  verified      every tag it carries agrees with the snapshot, it carries at
                least one source quote found at the cited lines, and every
                number it states appears in one of its verified tags;
  unverified    nothing contradicts it, but that bar is not met;
  contradicted  a tag disagrees with the snapshot (wrong value, wrong shape,
                a quote that is not in the file).

The unit of classification is one `### Defects` top-level bullet (with its
sub-bullets) or one Table F row. The review prompt asks for one defect per
bullet — one repair whose evidence stands on its own — because a single
contradicted tag withholds the whole bullet from Agent Fix.

Evidence is written as backtick tags (the review prompt defines them):

  `src: modeling_x.py:120-122 "self.scaling = self.head_dim**-0.5"`
  `cfg: text_config.sliding_window = 128`
  `ckpt: model.layers.X.mlp.gate_proj.weight = [2048, 6144]`
  `count: model.mtp.layers = 3`

The prompt asks for exactly that form. The checker also reads the variants
review agents write anyway, and checks them the same way: `:L120` /
`:L120-L122` line numbers; several quotes joined by `/`, read as pieces
that must appear in order (as with `...`); `\"` escapes inside a quote; and
the tags without their backticks (`src:` with its quote, the others with
`= value`). A Table F status is one of the statuses the template names,
in any letter case; a mark or another word (`✅`, `matches`) is no status.

Reports written before the tags existed still parse: a bare `file.py:L`
citation is a source check without a quote, and `key=value` / `` `tensor`
[shape] `` phrases whose key or tensor is in the snapshot are checked the
same way. Only an explicit tag can contradict: an untagged phrase is parsed
heuristically, so its disagreement with the snapshot makes the finding
unverified, never contradicted. Such reports can never be verified either.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import math
from enum import Enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

from backend.hf.header import SafetensorsHeader

# Line numbers of a citation: `120`, `120-122`, and the `L120` / `L120-L122`
# spelling review agents also write.
_LINES = r"L?(\d+)(?:\s*[-–]\s*L?(\d+))?"
_CITE_RE = re.compile(rf"(?<![\w/.-])([\w./-]*?\.py|modeling):{_LINES}")
_SRC_TAG_RE = re.compile(rf"^(\S+?):{_LINES}\s*(.*?)\s*$", re.S)
# A quoted excerpt with backslash escapes, and a run of them joined by `/`
# (`"gate = ..." / "up = ..."`): pieces that must appear in this order.
_QUOTE = r'"(?:[^"\\]|\\.)*"'
_QUOTES_RE = re.compile(rf"{_QUOTE}(?:\s*/\s*{_QUOTE})*")
# A backticked tag. A quote ends a `src:` tag only after its closing `"`:
# quoted code may hold backticks of its own (a docstring's `pixel_values`).
_TAG_RE = re.compile(
    r"`(src|cfg|ckpt|count):\s*"
    rf"([^\s`]+?:L?\d+(?:\s*[-–]\s*L?\d+)?\s*{_QUOTES_RE.pattern}|[^`]+?)\s*`")
# Tags written without backticks. Only forms whose end is unambiguous in
# running text are tags: `src:` with its quote, and `cfg:` / `ckpt:` /
# `count:` with `= value` (a number, a boolean, `[...]` or `absent`).
_BARE_SRC_RE = re.compile(
    rf"(?<![\w`])src:\s*([\w./-]+\.py):{_LINES}\s*({_QUOTES_RE.pattern})")
_BARE_VALUE_RE = re.compile(
    r"(?<![\w`])(cfg|ckpt|count):\s*([A-Za-z_][\w.]*\s*=\s*"
    r"(?:\[[^\]\n]*\]|(?:absent|true|false|True|False)\b"
    r"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)))")
_LEGACY_CFG_RE = re.compile(
    r"`?([A-Za-z_][\w.]*)`?\s*(?:=|:)\s*`?(\[[-\d.,\s]*\]|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|true|false|True|False)`?(?![\w.])"
    r"|([A-Za-z_]\w*)\((-?\d+(?:\.\d+)?)\)")
_LEGACY_SHAPE_RE = re.compile(
    r"`([A-Za-z_][\w.]*)`\s*(?:as\s+|is\s+|:\s*|\(\s*)?`?\[(\d+(?:\s*,\s*\d+)+)\]")
_NUMBER_RE = re.compile(
    r"(?<![\w.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|[.,]\d)")
_NOT_A_CLAIM_RE = re.compile(
    rf"{_TAG_RE.pattern}"                        # tags: checked on their own
    rf"|{_BARE_SRC_RE.pattern}|{_BARE_VALUE_RE.pattern}"  # unbackticked tags
    r"|[\w./-]*:L?\d+(?:\s*[-–]\s*L?\d+)?(?:\s*,\s*L?\d+(?:\s*[-–]\s*L?\d+)?)*"  # file:line refs
    r"|\b(?:steps?|rows?|layers? index|#)\s*\d+(?:\s*[-–,]\s*\d+)*"
    r"|\[library:[^\]]*\]", re.I)
_BULLET_RE = re.compile(r"^(\s*)[-*] (.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s")
# `None.` alone on its line: the whole of a Defects section with no defects.
_NONE_RE = re.compile(r"^\s*(?:[-*]\s*)?none\.?\s*$", re.I)

VERIFIED, UNVERIFIED, CONTRADICTED = "verified", "unverified", "contradicted"

class RowStatus(str, Enum):
    """A Table F row's status, as the review template names it. `passing`
    is whether a PASS may carry it."""
    MATCH = "match"
    NOT_EXPRESSIBLE = "constant not expressible"
    NOT_APPLICABLE = "n/a"
    MISMATCH = "MISMATCH"
    UNMAPPED = "UNMAPPED"

    @property
    def passing(self) -> bool:
        return self not in (RowStatus.MISMATCH, RowStatus.UNMAPPED)

    @classmethod
    def read(cls, head: str) -> Optional["RowStatus"]:
        """The status `head` names (any letter case), or None."""
        for status in cls:
            if head.lower() == status.value.lower():
                return status
        return None


def _cited_name(path: str) -> str:
    """The name a staged source file is cited by: its path under the run's
    `source/` dir (`modeling_x.py`, `vllm/models/hy_v4/nvidia/mtp.py`)."""
    _head, sep, rel = str(path).partition("/source/")
    return rel if sep else Path(path).name


def _quote(text: str) -> Optional[str]:
    """The code a `src:` tag quotes, from what follows its line numbers: one
    `"..."`, or several joined by `/` (read as `...`-elided pieces, in
    order); inside backticks a single quote may also hold bare `"`. None when
    nothing is quoted or the text is not a quote."""
    text = text.strip()
    if _QUOTES_RE.fullmatch(text):
        return " ... ".join(q[1:-1] for q in re.findall(_QUOTE, text))
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return None


def _captured(fact: Optional[Mapping]) -> Optional[Mapping]:
    """A fetched mapping, or None when the fetch captured nothing."""
    return fact if fact else None


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text)]


def _as_float(value: Any) -> Optional[float]:
    """The number a config / YAML value spells, or None when it spells none
    (config.json writes some sizes as strings: `"speech_vocab_size": "1280"`)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.fullmatch(value.strip()):
        return float(value.strip().replace(",", ""))
    return None


def _same(claimed: Any, actual: Any) -> bool:
    """Value equality across YAML/JSON spellings: `128` == "128", `1.0` == 1,
    `true` == True, lists element-wise."""
    if isinstance(actual, (list, tuple)) or isinstance(claimed, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and isinstance(claimed, (list, tuple))
                and len(actual) == len(claimed)
                and all(_same(c, a) for c, a in zip(claimed, actual, strict=True)))
    if isinstance(actual, bool) or isinstance(claimed, bool):
        return str(claimed).lower() == str(actual).lower()
    c, a = _as_float(claimed), _as_float(actual)
    if c is not None and a is not None:
        return math.isclose(c, a, rel_tol=1e-9, abs_tol=0.0)
    return str(claimed).strip() == str(actual).strip()


def _flat_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for v in value:
            yield from _flat_numbers(v)
        return
    f = _as_float(value)
    if f is not None:
        yield f


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _in_order(pieces: Sequence[str], text: str) -> bool:
    """Every piece occurs in `text`, each after the end of the one before."""
    at = 0
    for piece in pieces:
        at = text.find(piece, at)
        if at < 0:
            return False
        at += len(piece)
    return True


@dataclass(frozen=True)
class Check:
    """One claim compared with the snapshot. `backs` are the numbers a
    verified check vouches for (the values it read from the snapshot);
    `lines` is the (file, first, last) a source check cited."""
    kind: str
    claim: str
    status: str
    detail: str
    backs: tuple = ()
    lines: tuple = ()

    def covers(self, name: Optional[str], first: int, last: int) -> bool:
        """Whether this is a verified source quote whose cited lines include
        all of `name`:`first`-`last`."""
        if self.status != VERIFIED or not self.lines or name is None:
            return False
        own, lo, hi = self.lines
        return own == name and lo <= first and last <= hi

    def where(self) -> str:
        """Where the claim can be checked — the file and lines, config key,
        tensor or prefix it names — without the value the review asserted."""
        body = re.sub(rf"^{self.kind}:\s*", "", self.claim.replace("`", "").strip())
        m = re.match(r"[\w.:/-]+", body)
        subject = m.group(0).rstrip(":") if m else body[:60]
        return {"src": f"source `{subject}`", "cfg": f"config.json `{subject}`",
                "ckpt": f"tensor `{subject}`",
                "count": f"layer indices under `{subject}`"}[self.kind]


class SnapshotFacts:
    """What a review claim can be checked against: config.json, the
    checkpoint's safetensors header (`hf.header.SafetensorsHeader`) and the
    staged source files (by file name). A fact this run did not capture is
    None: a claim about it cannot be checked, so it is unverified, never
    contradicted."""

    def __init__(self, config: Optional[Mapping], header: Optional[Any],
                 sources: Mapping[str, str]):
        self.config = config
        self.header = header
        self.sources = {name: text.splitlines() for name, text in sources.items()}

    @classmethod
    def empty(cls) -> "SnapshotFacts":
        """Nothing captured: every claim is unverified."""
        return cls(None, None, {})

    @classmethod
    def from_run(cls, snapshot_data: Optional[Mapping],
                 source_paths: Sequence[str]) -> "SnapshotFacts":
        """From a prep's `_snapshot_data` and the staged source paths. The
        fetchers report a failed fetch as an empty mapping
        (`get_ground_truth`, `fetch_metadata_raw`); a model's config.json and
        tensor index are never empty, so an empty one is not captured."""
        snap = snapshot_data or {}
        meta = snap.get("metadata") or {}
        sources = {_cited_name(p): Path(p).read_text(encoding="utf-8")
                   for p in source_paths}
        header = (SafetensorsHeader.from_json(meta["header"])
                  if meta.get("header") else None)
        return cls(_captured(snap.get("config")), header or None, sources)

    # ── config ────────────────────────────────────────────────────────────
    def _config_paths(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.config is None:
            return out

        def walk(prefix: str, node: Any) -> None:
            if isinstance(node, Mapping):
                for k, v in node.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            else:
                out[prefix] = node
        walk("", self.config)
        return out

    def config_matches(self, key: str) -> dict[str, Any]:
        """Config paths the key may name. `config.<path>` is anchored at the
        root of config.json and names that one path; a bare key names the
        path it spells and every nested path ending in `.key` (`rope_theta`
        is both the top-level θ and `audio_config.rope_theta`)."""
        key = key.strip()
        paths = self._config_paths()
        if key.startswith("config."):
            key = key.removeprefix("config.")
            return {key: paths[key]} if key in paths else {}
        return {p: v for p, v in paths.items() if p == key or p.endswith("." + key)}

    def check_config(self, key: str, claimed: Any, raw: str) -> Check:
        if self.config is None:
            return Check("cfg", raw, UNVERIFIED,
                         "this run captured no config.json; the claim cannot be checked")
        found = self.config_matches(key)
        if isinstance(claimed, str) and claimed.strip().lower() == "absent":
            if found:
                return Check("cfg", raw, CONTRADICTED,
                             f"config.json has {self._show(found)}")
            return Check("cfg", raw, VERIFIED, f"config.json has no `{key}`")
        if not found:
            return Check("cfg", raw, CONTRADICTED, f"config.json has no `{key}`")
        hits = {p: v for p, v in found.items() if _same(claimed, v)}
        if len(hits) == len(found):
            backs = tuple(n for v in hits.values() for n in _flat_numbers(v))
            return Check("cfg", raw, VERIFIED, self._show(hits), backs)
        if len(found) > 1:
            return Check("cfg", raw, UNVERIFIED,
                         f"`{key}` names several config paths ({self._show(found)}); "
                         "write the full path (`config.<path>` for a top-level key)")
        return Check("cfg", raw, CONTRADICTED,
                     f"review says `{key}` = {claimed}; config.json has {self._show(found)}")

    @staticmethod
    def _show(found: Mapping[str, Any]) -> str:
        return ", ".join(f"`{p}` = {v!r}" for p, v in list(found.items())[:4])

    # ── checkpoint ────────────────────────────────────────────────────────
    def tensor_matches(self, name: str) -> dict[str, list]:
        """Header tensor groups the name denotes (an index written as a
        digit names that index, `{0-2,5}` each index listed, all of which
        must be stored, `X` any; a trailing `.weight` may be
        omitted; a dotted suffix names every tensor ending in it) → their
        stored shapes."""
        name = name.strip()
        out: dict[str, list] = {}
        for wanted in (name, name + ".weight"):
            tail = wanted.split(".")
            for group in self.header or ():
                if len(tail) > len(group.segments):
                    continue
                head = group.query_segments[:len(group.segments) - len(tail)]
                selection = self.header.select(".".join([*head, *tail]))
                if selection.complete and any(g is group for g, _n in selection.parts):
                    out.setdefault(group.display(), []).append(list(group.shape))
        return out

    def check_tensor(self, name: str, shape: Any, raw: str) -> Check:
        module = re.sub(r"(?:^|\.)(?:weight|bias)$", "", name.strip())
        if not module:
            # `weight` alone names every weight in the checkpoint
            return Check("ckpt", raw, UNVERIFIED,
                         f"`{name}` does not name a module; write the tensor's full path")
        if self.header is None:
            return Check("ckpt", raw, UNVERIFIED,
                         "this run captured no checkpoint tensor index; the claim "
                         "cannot be checked")
        found = self.tensor_matches(name)
        if isinstance(shape, str) and shape.strip().lower() == "absent":
            if found:
                return Check("ckpt", raw, CONTRADICTED,
                             f"the checkpoint stores {sorted(found)[:3]}")
            return Check("ckpt", raw, VERIFIED, f"no `{name}` in the checkpoint")
        if not found:
            return Check("ckpt", raw, CONTRADICTED, f"the checkpoint has no tensor `{name}`")
        if shape is None:
            return Check("ckpt", raw, VERIFIED, f"`{name}` is stored")
        matching = [p for p, shapes in found.items() if any(_same(shape, s) for s in shapes)]
        stored = "; ".join(f"`{p}` {s}" for p, s in list(found.items())[:3])
        if len(matching) == len(found):
            return Check("ckpt", raw, VERIFIED, f"`{name}` stored as {shape}",
                         tuple(_flat_numbers(shape)))
        if len(found) > 1:
            return Check("ckpt", raw, UNVERIFIED,
                         f"`{name}` names several tensors ({stored}); write the full path")
        return Check("ckpt", raw, CONTRADICTED,
                     f"review says `{name}` is {shape}; the checkpoint stores {stored}")

    def layer_counts(self) -> dict[str, int]:
        """Prefix → number of layer indices stored under it (`model.mtp.layers`
        → 3), read from the header: the indices at the first numeric segment
        after that prefix, the most any one tensor group holds."""
        out: dict[str, int] = {}
        for group in self.header or ():
            if not group.slot_positions:
                continue
            head = ".".join(group.segments[:group.slot_positions[0]])
            firsts = {i for box in group.boxes for i in box[0]}
            out[head] = max(out.get(head, 0), len(firsts))
        return dict(sorted(out.items()))

    def check_count(self, prefix: str, claimed: Any, raw: str) -> Check:
        prefix = re.sub(r"\.(?:X|\d+)$", "", prefix.strip())
        if self.header is None:
            return Check("count", raw, UNVERIFIED,
                         "this run captured no checkpoint layer counts; the claim "
                         "cannot be checked")
        counts = self.layer_counts()
        if prefix not in counts:
            if _same(claimed, 0):
                return Check("count", raw, VERIFIED,
                             f"the checkpoint stores no `{prefix}.<i>.*` tensors", (0.0,))
            return Check("count", raw, CONTRADICTED,
                         f"review says `{prefix}` has {claimed}; the checkpoint stores "
                         f"no `{prefix}.<i>.*` tensors")
        if _same(claimed, counts[prefix]):
            return Check("count", raw, VERIFIED, f"`{prefix}` has {counts[prefix]} indices",
                         (float(counts[prefix]),))
        return Check("count", raw, CONTRADICTED,
                     f"review says `{prefix}` has {claimed}; the checkpoint stores "
                     f"{counts[prefix]}")

    def render_layer_counts(self) -> str:
        if self.header is None:
            return ("(this run captured no checkpoint tensor index; "
                    "`count:` tags cannot be checked)")
        counts = self.layer_counts()
        if not counts:
            return "(no indexed tensors in the checkpoint index)"
        return "\n".join(f"  {p}: {n}" for p, n in counts.items())

    # ── source ────────────────────────────────────────────────────────────
    def _sources(self, cited: str) -> list[str]:
        """The staged files a citation can name: its path under `source/`
        (`vllm/models/hy_v4/nvidia/mtp.py`), else the files whose path ends
        with it (`nvidia/mtp.py`), else the files with its file name, else
        the files whose name it is the start of. Several when the citation
        does not tell same-named files apart (vLLM's `amd/model.py` and
        `nvidia/model.py`); `check_source` lets the quote choose."""
        cited = cited.strip()
        cited = cited[len("source/"):] if cited.startswith("source/") else cited
        if cited in self.sources:
            return [cited]
        tail = [n for n in self.sources if n.endswith("/" + cited)]
        if tail:
            return tail
        name = Path(cited).name
        named = [n for n in self.sources if Path(n).name == name]
        return named or [n for n in self.sources if Path(n).name.startswith(name)]

    def check_source(self, cited: str, first: int, last: int,
                     quote: Optional[str], raw: str) -> Check:
        """A citation of several same-named files is the one whose cited
        lines hold the quote: name, lines and verbatim code together identify
        the file. When none or more than one does, the citation does not say
        which file it means."""
        names = self._sources(cited)
        if not names:
            return Check("src", raw, UNVERIFIED,
                         f"`{cited}` is not among this run's staged source files")
        checks = [self._check_in(name, first, last, quote, raw) for name in names]
        if len(checks) == 1:
            return checks[0]
        verified = [c for c in checks if c.status == VERIFIED]
        if len(verified) == 1:
            return verified[0]
        listed = ", ".join(f"`{n}`" for n in names)
        held = (f"{len(verified)} of them hold" if verified else "none holds")
        return Check("src", raw, UNVERIFIED,
                     f"`{cited}` names {len(names)} staged files ({listed}) and "
                     f"{held} the quote at lines {first}-{last}; cite the file by "
                     "its path under source/")

    def _check_in(self, name: str, first: int, last: int,
                  quote: Optional[str], raw: str) -> Check:
        lines = self.sources[name]
        if first < 1 or last < first or first > len(lines):
            return Check("src", raw, CONTRADICTED,
                         f"`{name}` has {len(lines)} lines; the citation points past them")
        if not quote:
            return Check("src", raw, UNVERIFIED, "no quoted code to compare with the cited lines")
        # The quote as written, and with `\"` / `\\` unescaped: agents escape
        # the quotes of a string literal they quote inside a quote.
        readings = list(dict.fromkeys([quote, re.sub(r'\\(["\\])', r"\1", quote)]))
        cuts = [[_squash(p) for p in re.split(r"\.\.\.|…", q) if _squash(p)]
                for q in readings]
        lo, hi = max(0, first - 4), min(len(lines), last + 3)
        near, whole = _squash("".join(lines[lo:hi])), _squash("".join(lines))
        if any(pieces and _in_order(pieces, near) for pieces in cuts):
            return Check("src", raw, VERIFIED, f"{name}:{first}-{last}",
                         tuple(float(n) for n in _numbers(quote)), (name, first, last))
        if any(pieces and all(p in whole for p in pieces) for pieces in cuts):
            return Check("src", raw, UNVERIFIED,
                         f"the quoted pieces are in `{name}` but not in this order "
                         f"at lines {first}-{last}")
        return Check("src", raw, CONTRADICTED, f"the quoted code is not in `{name}`")

    # ── dispatch ──────────────────────────────────────────────────────────
    def check_tag(self, kind: str, body: str) -> Check:
        raw = f"{kind}: {body}"
        if kind == "src":
            m = _SRC_TAG_RE.match(body)
            quote = _quote(m.group(4)) if m else None
            if not m or (m.group(4) and quote is None):
                return Check("src", raw, UNVERIFIED, "not `file:line \"quote\"`")
            first = int(m.group(2))
            return self.check_source(m.group(1), first, int(m.group(3) or first),
                                     quote, raw)
        subject, sep, value = body.partition("=")
        subject = subject.strip()
        if kind == "ckpt" and not sep:
            return self.check_tensor(subject, None, raw)
        if not sep:
            return Check(kind, raw, UNVERIFIED, "no `= value`")
        try:
            claimed = load_yaml(value.strip())
        except yaml.YAMLError:
            claimed = value.strip()
        if kind == "cfg":
            return self.check_config(subject, claimed, raw)
        if kind == "ckpt":
            return self.check_tensor(subject, claimed, raw)
        return self.check_count(subject, claimed, raw)

    def checks_in(self, text: str) -> list[Check]:
        """Every claim in `text` this snapshot can check: tags first, then the
        untagged forms older reports use (bare citations, `key=value` whose
        key is in config.json, `` `tensor` [shape] `` whose tensor is stored).
        A bare citation whose file and lines a verified `src:` tag of the same
        text includes (a Table F source cell next to its tag) is that tag's
        citation, not a claim of its own; any other bare citation is checked
        as a source claim without a quote."""
        tagged = [self.check_tag(k, b) for k, b in _TAG_RE.findall(text)]
        rest = _TAG_RE.sub(" ", text)
        for m in _BARE_SRC_RE.finditer(rest):
            first = int(m.group(2))
            tagged.append(self.check_source(m.group(1), first, int(m.group(3) or first),
                                            _quote(m.group(4)), m.group(0)))
        tagged += [self.check_tag(k, b) for k, b in _BARE_VALUE_RE.findall(rest)]
        rest = _BARE_VALUE_RE.sub(" ", _BARE_SRC_RE.sub(" ", rest))
        untagged = []
        for m in _CITE_RE.finditer(rest):
            first = int(m.group(2))
            last = int(m.group(3) or first)
            if any(c.covers(n, first, last) for c in tagged
                   for n in self._sources(m.group(1))):
                continue
            untagged.append(self.check_source(m.group(1), first, last, None, m.group(0)))
        for m in _LEGACY_CFG_RE.finditer(rest):
            if re.search(r"(?:\bnot|[=!<>+*/-])\s*`?$", rest[:m.start()]):
                continue  # an operand inside an expression, not a key = value claim
            key, value = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            if self.config_matches(key):
                untagged.append(self.check_config(key, load_yaml(value), m.group(0)))
        for m in _LEGACY_SHAPE_RE.finditer(rest):
            if self.tensor_matches(m.group(1)):
                shape = [int(x) for x in m.group(2).split(",")]
                untagged.append(self.check_tensor(m.group(1), shape, m.group(0)))
        return tagged + [
            Check(c.kind, c.claim, UNVERIFIED, f"untagged phrase: {c.detail}")
            if c.status == CONTRADICTED else c for c in untagged]


@dataclass
class Finding:
    """One claim of the report: a `### Defects` bullet or a Table F row."""
    kind: str                      # "defect" | "row"
    text: str
    checks: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    unbacked: list = field(default_factory=list)   # numbers no verified check carries

    @property
    def status(self) -> str:
        """contradicted if any check is; else unverified if any check is, if
        nothing grounds it, or if a number it states is not backed; verified
        only when every check verified. A verified quote grounds a finding
        but never vouches for its other claims."""
        if any(c.status == CONTRADICTED for c in self.checks):
            return CONTRADICTED
        if self.problems or self.unbacked or any(c.status != VERIFIED for c in self.checks):
            return UNVERIFIED
        return VERIFIED

    @property
    def title(self) -> str:
        if self.kind == "row":
            cells = [c.strip() for c in self.text.strip().strip("|").split("|")]
            return cells[1] if len(cells) > 1 else self.text[:100]
        bold = re.search(r"\*\*(.+?)\*\*", self.text)
        if bold:
            return bold.group(1).strip()[:80]
        first = self.text.strip().splitlines()[0] if self.text.strip() else ""
        return first.removeprefix("- ")[:80]

    def reasons(self) -> list[str]:
        bad = [c.detail for c in self.checks if c.status == CONTRADICTED]
        if bad:
            return bad
        out = list(self.problems)
        out += [c.detail for c in self.checks if c.status == UNVERIFIED]
        if self.unbacked:
            out.append("states numbers no verified config/checkpoint/source tag carries: "
                       + ", ".join(self.unbacked[:8]))
        return out

    def allows_pass(self) -> bool:
        """A Table F row whose status a PASS may carry: `match`, `constant
        not expressible` or `n/a`, optionally followed by a reason after `(`,
        `:`, `.`, `;`, `,` or a dash."""
        status = RowStatus.read(self.canonical_status())
        return status is not None and status.passing

    def canonical_status(self) -> str:
        """The row's status without its reason (and without markdown bold),
        as the template spells it when it names one."""
        status = self.row_status().replace("*", "").strip()
        head = re.split(r"\s*[(:;,.—–]|\s+-\s", status, maxsplit=1)[0].strip()
        status = RowStatus.read(head)
        return status.value if status is not None else head

    def judge(self, facts: SnapshotFacts) -> "Finding":
        """Check every claim against `facts` and record what keeps the finding
        from being verified. A Table F row needs a verified source quote (an
        n/a row may rest on config or the checkpoint instead); a defect needs
        a verified source quote, and every number it states must be carried
        by one of its verified checks. 0, 1, 2 and -1 are structural (axis
        indices, input counts, halves), not claims about the model."""
        self.checks = facts.checks_in(self.text)
        verified = [c for c in self.checks if c.status == VERIFIED]
        if self.kind == "row" and self.row_status().startswith("n/a"):
            grounded = bool(verified)
        else:
            grounded = any(c.kind == "src" for c in verified)
        if not grounded:
            self.problems.append("no quoted source line was found at its citation")
        if self.kind == "row":
            return self
        backed = [n for c in verified for n in c.backs]
        for n in _numbers(_NOT_A_CLAIM_RE.sub(" ", self.text)):
            f = float(n)
            if f in (0.0, 1.0, 2.0, -1.0) or any(math.isclose(f, b) for b in backed):
                continue
            if n not in self.unbacked:
                self.unbacked.append(n)
        return self

    def row_status(self) -> str:
        cells = [c.strip() for c in self.text.strip().strip("|").split("|")]
        return cells[-1].lower() if cells else ""


class ReviewEvidence:
    """A review report with every claim checked against one snapshot."""

    def __init__(self, findings: list[Finding], rows: list[Finding],
                 table_s: Sequence[str] = (), after_none: Sequence[str] = ()):
        self.findings = findings
        self.rows = rows
        self.table_s = list(table_s)       # Table S (spec → forward) rows
        self.after_none = list(after_none)  # Defects lines after "None"

    @classmethod
    def audit(cls, report: str, facts: SnapshotFacts) -> "ReviewEvidence":
        body = _agent_part(report or "")
        table_s = _table_rows(body, lambda h: "forward step" in h and "status" not in h)
        defects, after_none = _defects_section(body)
        return cls([Finding("defect", d).judge(facts) for d in defects],
                   [Finding("row", r).judge(facts)
                    for r in _table_rows(body, lambda h: "forward step" in h and "status" in h)],
                   table_s, after_none)

    def by_status(self, status: str) -> list[Finding]:
        return [f for f in self.findings if f.status == status]

    def pass_blockers(self) -> list[str]:
        """Why a PASS verdict cannot be recorded from this report: no Table F;
        a Table F row that is not verified, or whose status is not one a
        PASS allows (match / constant not expressible / n/a); a Table S row
        marked UNSOURCED, or no Table S; any listed defect, verified or not (a
        PASS that lists a defect contradicts itself); a Defects section that
        says "None" and then goes on."""
        out = []
        if not self.rows:
            out.append("the report has no Table F (forward → spec) rows")
        if not self.table_s:
            out.append("the report has no Table S (spec → forward) rows")
        for i, f in enumerate(self.rows, 1):
            if f.status != VERIFIED:
                out.append(f"Table F row {i} ({f.title[:60]}): {f.status} — "
                           + "; ".join(f.reasons()))
            if not f.allows_pass():
                out.append(f"Table F row {i} ({f.title[:60]}): status "
                           f"`{f.row_status()}` is not match / constant not "
                           "expressible / n/a")
        out.extend(f"Table S row marked UNSOURCED: {r[:80]}"
                   for r in self.table_s if "unsourced" in r.lower())
        out.extend(f"the report lists a defect: {f.title[:60]} ({f.status})"
                   for f in self.findings)
        if self.after_none:
            out.append("the Defects section says None but continues: "
                       + self.after_none[0].strip()[:80])
        return out

    def render(self, agent_verdict: str) -> str:
        """The `## Evidence check` section stored with the report."""
        lines = [
            "## Evidence check",
            "Every claim was compared with this run's config.json, checkpoint "
            "tensor index and staged source files.",
            f"Defects: {len(self.by_status(VERIFIED))} verified, "
            f"{len(self.by_status(UNVERIFIED))} unverified, "
            f"{len(self.by_status(CONTRADICTED))} contradicted. "
            f"Table F: {len(self.rows)} rows, "
            f"{sum(r.status == VERIFIED for r in self.rows)} verified.",
        ]
        lines.extend(f"- [{f.status}] {f.title} — " + "; ".join(f.reasons())
                     for f in self.findings if f.status != VERIFIED)
        blockers = self.pass_blockers()
        if agent_verdict == "pass" and blockers:
            lines.append("")
            lines.append("The agent's PASS is not recorded: "
                         + " | ".join(blockers[:6]))
        return "\n".join(lines)

    def gives_work_order(self) -> bool:
        """Whether any finding's evidence checked out. A report with none
        (a validator-failure stub, a list of uncited leads) orders no edit."""
        return bool(self.by_status(VERIFIED))

    def work_order(self) -> str:
        """The fix run's view of the review. Verified findings are the work
        order, in full. An unverified finding is a lead: its title and the
        places its evidence points at, never its text or suggested fix. A
        contradicted finding is withheld: only its false tags and what the
        snapshot says instead."""
        out = ["### Work order — findings whose evidence checked out"]
        out += [f.text for f in self.by_status(VERIFIED)] or ["(none)"]
        out += ["", "### Unverified leads — confirm before changing anything",
                "A review raised these, but their evidence did not check out, so "
                "only their title and where to look are shown. Change the spec for "
                "one only if the staged source, config.json or the tensor table "
                "shows the defect; take every value from there."]
        out += [self._lead(f) for f in self.by_status(UNVERIFIED)] or ["(none)"]
        out += ["", "### Withheld — contradicted by config.json, the checkpoint or the source",
                "The review's evidence below is false. Do not act on it."]
        out += [f"- {c.where()}: {c.detail[:200]}"
                for f in self.by_status(CONTRADICTED) for c in f.checks
                if c.status == CONTRADICTED] or ["(none)"]
        return "\n".join(out)

    @staticmethod
    def _lead(f: Finding) -> str:
        places = list(dict.fromkeys(c.where() for c in f.checks))
        why = list(f.problems) + (
            ["its numbers are not backed by config.json or the checkpoint"]
            if f.unbacked else [])
        where = "; ".join(places) if places else "nothing checkable was cited"
        return f"- {f.title} — check: {where} ({'; '.join(why)})"


def _agent_part(report: str) -> str:
    """The agent's findings: after the stored `## Agent review` heading when
    present, and never the derived `## Evidence check` section."""
    head = report.split("## Agent review", 1)
    body = head[1] if len(head) == 2 else report
    return body.split("## Evidence check", 1)[0]


def _table_rows(body: str, header) -> list[str]:
    """Rows of the markdown tables whose (lower-cased) header line satisfies
    `header`."""
    rows: list[str] = []
    in_table, started = False, 0
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("|") and header(s.lower()):
            in_table, started = True, len(rows)
            continue
        if in_table:
            if not s.startswith("|"):
                in_table = len(rows) == started   # blank lines before the first row
                continue
            if re.fullmatch(r"\|[\s:|-]+\|?", s):
                continue
            rows.append(s)
    return rows


def _defects_section(body: str) -> tuple[list[str], list[str]]:
    """(defect bullets, unaccounted lines) of the `### Defects` section.

    The report template fixes the structure: a `Defects` heading, its
    bullets, then the next section (`### Notes`) under a heading of the same
    or a higher level. That heading ends the section; nothing inside it —
    a label, a bold line, a deeper sub-heading — does, so every bullet
    under it is a defect. Bullets are top-level, with their sub-bullets.
    A section with no defects is `None.` alone on its first line; every
    further non-blank line is returned as unaccounted: the report says there
    are no defects and then says more, which a PASS cannot carry.

    A report without that section (older free-form reviews) contributes
    every leaf bullet instead, each a claim of its own."""
    lines = body.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if _HEADING_RE.match(ln)
                  and ln.lstrip("#").strip().lower().startswith("defects")), None)
    if start is None:
        return _bullets(lines, leaves=True), []
    level = len(_HEADING_RE.match(lines[start]).group(1))
    section = []
    for ln in lines[start + 1:]:
        heading = _HEADING_RE.match(ln)
        if heading and len(heading.group(1)) <= level:
            break
        section.append(ln)
    filled = [ln for ln in section if ln.strip()]
    if filled and _NONE_RE.match(filled[0]):
        return [], filled[1:]
    return _bullets(section, leaves=False), []


def _bullets(lines: list[str], *, leaves: bool) -> list[str]:
    items: list[tuple[int, list[str]]] = []
    for ln in lines:
        m = _BULLET_RE.match(ln)
        if m:
            items.append((len(m.group(1)), [m.group(2)]))
        elif items and ln.strip() and not _HEADING_RE.match(ln):
            items[-1][1].append(ln.strip())
    if not leaves:
        out: list[str] = []
        for indent, text in items:
            if indent == 0 or not out:
                out.append("- " + "\n".join(text))
            else:
                out[-1] += "\n  - " + "\n".join(text)
        return out
    leaf = []
    for i, (indent, text) in enumerate(items):
        nxt = items[i + 1][0] if i + 1 < len(items) else -1
        if nxt <= indent:
            leaf.append("- " + "\n".join(text))
    return leaf
