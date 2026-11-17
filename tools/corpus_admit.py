"""Admit a model into the corpus-wide pins.

A model enters the corpus with its snapshot, its model yaml and the drafts
that yaml lists. Every pin (`tools/corpus_pins`) needs the model's keys; the
corpus checks must hold for it; and the decisions only a person can make
(how a framework splits a stack, which copy contracts it brings, why a
checkpoint residual exists) must be written and agree with what is measured.
This program measures everything, checks everything, reports what is missing,
and — in write mode — inserts the model's keys into every pin it can. It
never changes an existing key and never writes a file no pin owns. The one
decision it writes is D1 for a stack a framework's own source settles:
before the pinned run it stages the sources that decide the split
(`--framework-source`, one directory per source) — vLLM's files for the
model (`backend.hf.vllm_source`), and the model's own reference
implementation where the publisher ships one (`backend.hf.forward_source`,
the code the spec was written from). A snapshot explicitly recorded from
SGLang instead stages that frozen SGLang source offline, with its exact revision,
and never probes vLLM. A stack whose every node the leading
source splits as measured is recorded with that source's revision and each
node's cited assignment (`tools/corpus_pins/decisions/framework_split.py`);
any other stack, and every other decision, stays a person's.

    python -m tools.corpus_admit <model_id> [--dry-run [--preview-dir DIR]]
                                 [--report PATH] [--base-sha SHA]
                                 [--base-kind main|deployment]
                                 [--base-lock-sha256 HEX] [--env-root DIR]
                                 [--orchestrator SHA]
                                 [--bundle PATH ...] [--cache-dir DIR]
    python -m tools.corpus_admit verify [--changed --paths-file FILE]
    python -m tools.corpus_admit carry <model_id> --old DIR --new DIR

The service runs it in a tree it extracted from a commit's archive, which has
no `.git`: it names that commit (`--base-sha`, `--base-kind`), the sha256 of
the commit's `uv.lock` (`--base-lock-sha256`, otherwise read through git),
the project the running interpreter was installed from (`--env-root`, the
deployment; otherwise this tree), and its own revision (`--orchestrator`).

The run happens in a fixed process environment (offline hub, an empty HOME,
`PYTHONHASHSEED=0`, `TZ=UTC`, `LC_ALL=C.UTF-8`): the command re-executes
itself under it. Writing also needs the interpreter `ADMIT_PYTHON` with the
lock's runtime distributions installed; any other interpreter only previews
(`--dry-run`, the report says `preview: true`).

The order is fixed: fix the inputs (G0), check the base can be extended
(G1), measure every pin, run the checks (G2), judge the decisions (G4, D3,
D4), write the eligible keys, report. See `tools/corpus_pins/states.py` for
how each key's state and the run's status follow from the gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.corpus_pins import REGISTRY
from tools.corpus_pins.corpus import ROOT, CorpusModel, relative
from tools.corpus_pins.decision import DecisionState, Verdict
from tools.corpus_pins.decisions import tp_split
from tools.corpus_pins.decisions.tp_split import TP_SPLIT
from tools.corpus_pins.decisions.framework_split import (
    MANIFEST, SOURCE_KINDS, FrameworkSplit, Oracle, ReferenceSplit, SglangSplit,
    VllmSplit)
from tools.corpus_pins.env import Environment, lock_sha256_at
from tools.corpus_pins.pin import Pin, PinBase, Scope, canonical
from tools.corpus_pins.pins.encoder_weight import ENCODER_WEIGHT
from tools.corpus_pins.pins.gate_surfaces import SurfacePin
from tools.corpus_pins.pins.checkpoint_reconciliation import (
    CHECKPOINT_RECONCILIATION, reconcile, split)
from tools.corpus_pins.pins.index_refusals import INDEX_REFUSALS
from tools.corpus_pins.pins.readme_counts import PHRASES, README_COUNTS, cell, counts, is_family
from tools.corpus_pins.pins.sol_surface import SOL_SURFACE
from tools.corpus_pins.sol_selfcheck import selfcheck
from tools.corpus_pins.states import (
    KeyFacts, KeyState, RunFacts, Status, failed_gates, key_state, overall)

SCHEMA = 1
DRY_RUN, PR = "dry_run", "pr"
#: decisions whose open state holds the G4 dependency
G4_DECISIONS = tuple(d.id for d in REGISTRY.decisions if d.gates_g4)
OPEN = (DecisionState.PENDING, DecisionState.INCONSISTENT)
CHILD_ENV = "CORPUS_ADMIT_CHILD"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256(files: Sequence[Path]) -> str:
    """One hash over the bundle: each file's path under `backend/arch` and
    its content hash, sorted."""
    return sha256_bytes("\n".join(sorted(
        f"{relative(p)}\0{sha256_bytes(Path(p).read_bytes())}" for p in files)).encode())


def git_head(root: Path = ROOT) -> Optional[str]:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=False)
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


@dataclass
class Gates:
    g0: List[str] = field(default_factory=list)
    #: pin id -> why its base cannot be extended
    g1: Dict[str, List[str]] = field(default_factory=dict)
    g2: List[str] = field(default_factory=list)

    def failed(self) -> List[str]:
        return failed_gates(not self.g0, sorted(self.g1), not self.g2)


class Admission:
    """One admission run over the tree this module lives in."""

    def __init__(self, model_id: str, mode: str = DRY_RUN, *,
                 base_sha: Optional[str] = None,
                 bundle: Optional[Sequence[Path]] = None,
                 cache_dir: Optional[Path] = None,
                 orchestrator: Optional[str] = None,
                 preview_dir: Optional[Path] = None,
                 base_kind: str = "main",
                 base_lock_sha256: Optional[str] = None,
                 env_root: Optional[Path] = None,
                 framework_sources: Sequence[Path] = ()) -> None:
        self.model_id = model_id
        self.model = CorpusModel(model_id.replace("/", "__", 1))
        self.mode = mode
        self.base_sha = base_sha
        self.bundle_files = tuple(Path(p).resolve() for p in bundle) if bundle else None
        self.cache_dir = cache_dir or default_cache_dir()
        self.orchestrator = orchestrator
        #: `main` (the default branch's tip) or `deployment` (the service's
        #: own revision, when main's lock differs from the deployed one)
        self.base_kind = base_kind
        #: the base commit's `uv.lock` sha256, for a tree without `.git`
        self.base_lock_sha256 = base_lock_sha256
        #: the project the running interpreter's environment was installed
        #: from; its `uv.lock` is the environment's lock
        self.env_root = env_root
        #: a dry run writes the files a write would produce here, never
        #: into the tree
        self.preview_dir = preview_dir
        self.gates = Gates()
        self.errors: List[str] = []
        #: whether the READMEs already count this model's bundle, and the
        #: count deltas the bundle explains
        self.readme_written = True
        self.readme_deltas: Dict[str, int] = {}
        self.applies: Dict[str, bool] = {}
        self.rows: Dict[str, Dict[str, Any]] = {}
        #: the framework sources for this model, staged before the pinned run
        self.oracle = Oracle([source for root in framework_sources
                              for source in (FrameworkSplit.read(root),) if source])

    # ── S0: inputs ───────────────────────────────────────────────────────
    def fix_inputs(self) -> Dict[str, Any]:
        sha = self.base_sha or git_head()
        if sha is None:
            self.gates.g0.append("base SHA unknown")
        self.base_sha = sha
        self.env = Environment.here(self.env_root or ROOT)
        base_lock = self.base_lock_sha256 or (lock_sha256_at(sha) if sha else None)
        refusals = self.env.write_refusals(base_lock)
        self.preview = bool(refusals)
        if self.mode == PR and self.preview:
            self.gates.g0 += [f"write refused: {r}" for r in refusals]
        if not self.model.path.exists():
            self.gates.g0.append(f"no snapshot {relative(self.model.path)}")
            return {}
        files = self.bundle_files or self.model.bundle()
        missing = [str(p) for p in files if not Path(p).exists()]
        if missing:
            self.gates.g0.append(f"bundle files missing: {missing}")
            return {}
        self.bundle_files = tuple(files)
        return {
            "model_id": self.model.model_id,
            "snapshot_stem": self.model.stem,
            "bundle": sorted(relative(p) for p in files),
            "bundle_sha256": bundle_sha256(files),
            "snapshot_sha256": sha256_bytes(self.model.path.read_bytes()),
        }

    # ── S1: baseline ─────────────────────────────────────────────────────
    def check_baseline(self) -> None:
        for pin in REGISTRY.pins:
            if isinstance(pin, Pin):
                data = pin.read()
                if not pin.round_trips(data):
                    self.gates.g1.setdefault(pin.id, []).append(
                        f"{pin.artifact} does not read back byte for byte")
        for pin in (p for p in REGISTRY.pins if isinstance(p, SurfacePin)):
            if pin.id not in self.gates.g1:
                drift = self.surface_drift(pin)
                if drift:
                    self.gates.g1[pin.id] = drift
        self.readme_baseline()

    def surface_drift(self, pin: SurfacePin) -> List[str]:
        """The existing blocks whose measurement no longer equals the file
        (the candidate's own block excluded). Cached per (base SHA,
        environment), since only those decide it."""
        doc = pin.load()
        stems = [s for s in doc if s != self.model.stem]
        cache = self.cache_dir / (f"{pin.id}_{self.base_sha}_"
                                  f"{self.env.dists_sha256}_{self.env.python}.json")
        measured = json.loads(cache.read_text()) if self.base_sha and cache.exists() else {}
        missing = [stem for stem in stems if stem not in measured]
        measured.update({stem: pin.rows(CorpusModel(stem)) for stem in missing})
        if self.base_sha and missing:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(measured, sort_keys=True))
        moved = sorted(s for s in stems if measured[s] != doc[s])
        return [f"{pin.artifact}: {len(moved)} blocks differ from the measurement: {moved[:10]}"] if moved else []

    def readme_baseline(self) -> None:
        """The READMEs state either the repository's counts without this
        model's bundle or with it; anything else is a base the bundle cannot
        explain."""
        after = counts()
        self.readme_deltas = self.bundle_deltas()
        before = {name: after[name] - self.readme_deltas.get(name, 0) for name in after}
        texts = README_COUNTS.texts()
        values, problems = README_COUNTS.cells(texts)
        if problems:
            self.gates.g1[README_COUNTS.id] = problems
            return
        stated = {name: {values[cell(r, name)] for r in README_COUNTS.readmes} for name in after}
        if all(stated[name] == {after[name]} for name in after):
            self.readme_written = True
        elif all(stated[name] == {before[name]} for name in after):
            self.readme_written = False
        else:
            self.gates.g1[README_COUNTS.id] = [
                f"README states {name}={sorted(stated[name])}; the repository has "
                f"{before[name]} without the bundle, {after[name]} with it"
                for name in after if stated[name] - {before[name], after[name]}
                or len(stated[name]) != 1] or ["README counts are partly updated"]

    def bundle_deltas(self) -> Dict[str, int]:
        models = [p for p in self.bundle_files if relative(p).startswith("models/")]
        library = [p for p in self.bundle_files if relative(p).startswith("components/library/")]
        return {"families": sum(1 for p in models if is_family(p.read_text(encoding="utf-8"))),
                "components": len(library)}

    # ── S2: measure ──────────────────────────────────────────────────────
    def measure(self) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for pin in REGISTRY.pins:
            if not isinstance(pin, Pin):
                continue
            try:
                applies = pin.applies(self.model)
                rows[pin.id] = pin.measure(self.model) if applies else {}
            except Exception as exc:  # a measurement that raises is a G2 failure, reported
                self.gates.g2.append(f"{pin.id} measurement raised {type(exc).__name__}: {exc}")
                self.errors.append(traceback.format_exc())
                applies, rows[pin.id] = False, {}
            self.applies[pin.id] = applies
            if isinstance(pin, SurfacePin):
                failed = pin.failures(rows[pin.id])
                if failed:
                    self.gates.g2.append(f"{pin.id} rows report a failed analysis: {failed}")
        return rows

    # ── S3: checks ───────────────────────────────────────────────────────
    def run_checks(self) -> List[Dict[str, Any]]:
        out = []
        for check in REGISTRY.checks:
            if check.kind.value != "admit":
                continue
            try:
                found = check.violations(self.model)
            except Exception as exc:  # a check that raises fails, reported
                found = [f"raised {type(exc).__name__}: {exc}"]
                self.errors.append(traceback.format_exc())
            if found:
                self.gates.g2.append(f"{check.id} {check.title}")
            out.append({"id": check.id, "title": check.title, "kind": check.kind.value,
                        "state": "failed" if found else "passed", "violations": found})
        return out

    # ── S4: decisions ────────────────────────────────────────────────────
    def judge(self, rows: Mapping[str, Any]) -> Dict[str, Verdict]:
        return {decision.id: self.judge_one(decision, rows) for decision in REGISTRY.decisions}

    def judge_one(self, decision: Any, rows: Mapping[str, Any]) -> Verdict:
        try:
            if decision is TP_SPLIT:
                return decision.judge(self.model, dict(rows), oracle=self.oracle)
            return decision.judge(self.model, dict(rows))
        except Exception as exc:  # a decision that cannot be judged is open, reported
            self.errors.append(traceback.format_exc())
            return Verdict(DecisionState.INCONSISTENT, [f"raised {type(exc).__name__}: {exc}"])

    def unmet(self, pin: Pin, key: str, verdicts: Mapping[str, Verdict],
              rows: Mapping[str, Any]) -> Tuple[str, ...]:
        out = []
        for dependency in pin.key_waits_on(key):
            if dependency == "G4":
                if any(verdicts[d].state in OPEN for d in G4_DECISIONS):
                    out.append("G4")
            elif verdicts[dependency].state in OPEN:
                out.append(dependency)
        if pin is INDEX_REFUSALS and verdicts["D4"].state in OPEN:
            out.append("D4")
        return tuple(out)

    # ── S5 / S6 ──────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        candidate = self.fix_inputs()
        items: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        decisions: Dict[str, Verdict] = {}
        writes: Dict[str, bytes] = {}
        if not self.gates.g0:
            try:
                self.check_baseline()
                rows = self.rows = self.measure()
                checks = self.run_checks()
                decisions = self.judge(rows)
                items, writes = self.classify(rows, decisions)
                settled = (decisions[TP_SPLIT.id].suggestion
                           or {}).get("written_from_framework")
                if settled:
                    writes[tp_split.ARTIFACT] = tp_split.render_with(
                        self.model, settled["entries"])
            except Exception as exc:  # a run that fails is unavailable, reported
                self.gates.g0.append(f"run failed: {type(exc).__name__}: {exc}")
                self.errors.append(traceback.format_exc())
        status, blocked_kind = overall(RunFacts(
            g0_ok=not self.gates.g0, g1_failures=tuple(sorted(self.gates.g1)),
            g2_ok=not self.gates.g2,
            keys=[(item["scope"] == Scope.REQUIRED_AGGREGATE.value, KeyState(v["state"]))
                  for item in items for v in item["key_states"].values()],
            decisions_open=any(v.state in OPEN for v in decisions.values())))
        written: List[str] = []
        writable = status in (Status.READY, Status.NEEDS_REVIEW) and not self.gates.failed()
        if self.mode == DRY_RUN and self.preview_dir is not None and writable:
            for artifact, data in writes.items():
                target = self.preview_dir / artifact
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        if self.mode == PR and writable:
            written = self.write(writes)
            for item in items:
                for verdict in item["key_states"].values():
                    if verdict["state"] == KeyState.READY.value:
                        verdict["state"] = KeyState.WRITTEN.value
        return self.report(candidate, status, blocked_kind, items, checks, decisions, written,
                           sorted(writes) if writable else [])

    def classify(self, rows: Mapping[str, Any], verdicts: Mapping[str, Verdict]
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, bytes]]:
        failed = self.gates.failed()
        items, writes = [], {}
        for pin in REGISTRY.pins:
            if pin is README_COUNTS:
                item, new_texts = self.classify_readme(failed)
                items.append(item)
                writes.update({name: text.encode("utf-8") for name, text in new_texts.items()})
                continue
            row = rows.get(pin.id, {})
            doc = pin.load()
            key_states, ready = {}, {}
            for key in sorted(row):
                facts = KeyFacts(applies=self.applies.get(pin.id, False), present=key in doc,
                                 waits_on=self.unmet(pin, key, verdicts, rows),
                                 own_g1_failed=pin.id in self.gates.g1)
                verdict = key_state(facts, failed)
                key_states[key] = verdict.as_dict()
                if verdict.state is KeyState.READY:
                    ready[key] = row[key]
            if not row:
                key_states[self.model.stem] = key_state(KeyFacts(applies=False), failed).as_dict()
            if ready:
                old = pin.read()
                new = pin.render(pin.merge(doc, ready))
                problems = pin.check_diff(old, new, ready)
                if problems:
                    raise RuntimeError(f"{pin.id} write refused: {problems}")
                writes[pin.artifact] = new
            items.append(self.item(pin, key_states, row))
        return items, writes

    def classify_readme(self, failed: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        key_states = {}
        deltas = {} if self.readme_written else self.readme_deltas
        for readme in README_COUNTS.readmes:
            for name in PHRASES[readme]:
                facts = KeyFacts(aggregate=True, own_g1_failed=README_COUNTS.id in self.gates.g1,
                                 delta_zero=not deltas.get(name))
                key_states[cell(readme, name)] = key_state(facts, failed).as_dict()
        new_texts: Dict[str, str] = {}
        if any(deltas.values()) and README_COUNTS.id not in self.gates.g1:
            old = README_COUNTS.texts()
            new = README_COUNTS.rewrite(old, deltas)
            problems = README_COUNTS.check_rewrite(old, new, deltas)
            if problems:
                raise RuntimeError(f"P8 write refused: {problems}")
            new_texts = {name: text for name, text in new.items() if text != old[name]}
        return self.item(README_COUNTS, key_states, {"deltas": deltas}), new_texts

    def item(self, pin: PinBase, key_states: Dict[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
        withheld = any(v["state"] == KeyState.WITHHELD.value for v in key_states.values())
        found = evidence(pin, row)
        if pin is SOL_SURFACE and row:
            found["selfcheck"] = selfcheck(
                self.model, row, self.rows.get(ENCODER_WEIGHT.id, {}).get(self.model.stem, {}),
                SOL_SURFACE.load())
        return {
            "id": pin.id, "title": pin.title, "scope": pin.scope.value,
            "keys": sorted(key_states), "key_states": key_states,
            "evidence": found,
            "expected_failing_tests": list(pin.owner_tests) if withheld else [],
        }

    def write(self, writes: Mapping[str, bytes]) -> List[str]:
        for artifact, data in writes.items():
            (ROOT / artifact).write_bytes(data)
        return sorted(writes)

    def report(self, candidate: Dict[str, Any], status: Status, blocked_kind: Any,
               items: List[Dict[str, Any]], checks: List[Dict[str, Any]],
               decisions: Mapping[str, Verdict], written: List[str],
               would_write: List[str]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema": SCHEMA,
            "status": status.value,
            "blocked_kind": blocked_kind.value if blocked_kind else None,
            "preview": self.preview,
            "mode": self.mode,
            "base": {"sha": self.base_sha, "kind": self.base_kind},
            "measurement_tree": self.base_sha,
            "orchestrator": self.orchestrator,
            "env": self.env.summary(),
            "candidate": candidate,
            "gates": {"G0": self.gates.g0, "G1": self.gates.g1, "G2": self.gates.g2},
            "items": items,
            "checks": checks,
            "decisions": [{"id": d, "state": v.state.value, "detail": v.detail,
                           "suggestion": v.suggestion} for d, v in sorted(decisions.items())],
            "base_notes": [{"id": pin_id, "state": KeyState.BASE_DRIFT.value, "detail": detail}
                           for pin_id, detail in sorted(self.gates.g1.items())],
            "would_write": would_write,
            "written": written,
            "errors": self.errors,
        }
        body["report_sha256"] = sha256_bytes(canonical(body).encode())
        return body


def evidence(pin: PinBase, row: Mapping[str, Any]) -> Dict[str, Any]:
    """Small facts about a measured row, never a golden's body: the SOL
    cells' model-level fields and digests, the encoder stacks' bytes, a
    gate surface block's row count and hash; the row itself otherwise."""
    if pin is SOL_SURFACE:
        return {key: {"model": cell_["model"],
                      "digests": {phase: ops["digest"] for phase, ops in cell_["ops"].items()}}
                for key, cell_ in row.items()}
    if pin is ENCODER_WEIGHT:
        return {stem: {tp: {stack: v["bytes_per_rank"] for stack, v in stacks.items()}
                       for tp, stacks in value.items()} for stem, value in row.items()}
    if isinstance(pin, SurfacePin):
        return {stem: {"rows": len(block), "sha256": sha256_bytes("\n".join(block).encode())}
                for stem, block in row.items()}
    if pin is CHECKPOINT_RECONCILIATION and row:
        result = reconcile(split(next(iter(row)))[1])
        return {**row, "delta": int(result.delta), "weight_delta": int(result.weight_delta),
                "metadata_delta": int(result.metadata_delta), "anchored": result.is_anchored,
                "measured_delta": int(result.measured_delta) if result.is_anchored else None}
    if pin is INDEX_REFUSALS and row:
        return {stem: {**value, "explained_by": refusal_reasons(stem, value["families"])}
                for stem, value in row.items()}
    return dict(row)


def refusal_reasons(stem: str, families: Mapping[str, int]) -> Dict[str, str]:
    """Why each refusal family needs no note: the models that already bring
    it."""
    doc = INDEX_REFUSALS.load()
    out = {}
    for fam in families:
        others = sorted(s for s, entry in doc.items() if s != stem and fam in entry["families"])
        out[fam] = f"brought by {others}" if others else "unexplained (D4)"
    return out


def default_cache_dir() -> Path:
    """Outside the tree, so a run leaves the work tree as it found it."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingarch" / "corpus_admit"


def pinned_environment() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "HA_SPEC_OVERLAY_DIR"}
    env.update({"HF_HUB_OFFLINE": "1", "HOME": tempfile.mkdtemp(prefix="corpus_admit_home_"),
                "PYTHONHASHSEED": "0", "TZ": "UTC", "LC_ALL": "C.UTF-8", CHILD_ENV: "1"})
    return env


def verify_main(argv: Sequence[str]) -> int:
    from tools.corpus_pins import verify
    parser = argparse.ArgumentParser(prog="python -m tools.corpus_admit verify",
                                     description=verify.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--paths-file", type=Path)
    args = parser.parse_args(argv)
    if args.changed and args.paths_file is None:
        parser.error("--changed needs --paths-file")
    found = verify.verify(verify.read_paths(args.paths_file) if args.changed else ())
    for section, problems in found.items():
        print(f"{section}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for line in problems:
            print(f"  {line}")
    return 1 if any(found.values()) else 0


def carry_main(argv: Sequence[str]) -> int:
    from tools.corpus_pins.carry import carry
    parser = argparse.ArgumentParser(prog="python -m tools.corpus_admit carry",
                                     description=carry.__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--old", type=Path, required=True,
                        help="the decision files as the bot's last commit holds them")
    parser.add_argument("--new", type=Path, required=True,
                        help="the decision files as the branch head holds them")
    args = parser.parse_args(argv)
    carried, problems = carry(CorpusModel(args.model_id.replace("/", "__", 1)),
                              args.old, args.new)
    sys.stdout.write(json.dumps({"carried": carried, "problems": problems},
                                indent=2, sort_keys=True) + "\n")
    return 1 if problems else 0


def stage_sources(argv: Sequence[str]) -> List[Path]:
    """Stage every framework source that can decide the split for the model
    the arguments name, under the cache dir, each with the manifest
    `FrameworkSplit.read` reads. Empty when the model has no snapshot; a
    source that cannot be read is simply absent, and a stack no staged source
    settles waits on a person."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model_id")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args, _rest = parser.parse_known_args(list(argv))
    model = CorpusModel(args.model_id.replace("/", "__", 1))
    if not model.path.exists():
        return []
    source = model.payload.get("forward_source") or {}
    if source.get("source") == SglangSplit.kind:
        # This snapshot explicitly names its serving framework. Its frozen
        # source is sufficient offline; probing another framework could fetch
        # unrelated code and cannot establish this deployment's split.
        staged = {SglangSplit.kind: stage_sglang(model, args.cache_dir)}
    else:
        staged = {VllmSplit.kind: stage_vllm(model, args.cache_dir),
                  ReferenceSplit.kind: stage_reference(model, args.cache_dir)}
    return [staged[kind] for kind in SOURCE_KINDS if staged.get(kind) is not None]


def stage_vllm(model: CorpusModel, cache_dir: Path) -> Optional[Path]:
    """vLLM's files for the model, at their path in the vLLM repo; None when
    vLLM has none of the model's architectures or cannot be read."""
    from backend.hf.vllm_source import VllmModelSource
    source = VllmModelSource.for_config(model.config)
    if source is None:
        return None
    return stage_files(
        cache_dir / VllmSplit.kind / source.commit / model.stem,
        [(f["path"], f["text"]) for f in source.files],
        {"kind": VllmSplit.kind, "revision": source.commit,
         "architectures": list(source.architectures),
         "imported": list(source.imported)})


def stage_sglang(model: CorpusModel, cache_dir: Path) -> Optional[Path]:
    """Stage explicitly recorded SGLang Python sources without any fetch.

    Paths and revision stay exactly as recorded. Native kernel evidence
    remains in the snapshot; the split oracle consumes only Python ASTs.
    Hashing the snapshot for the staging directory keeps different frozen
    file sets at the same upstream commit from sharing stale staged files.
    """
    source = model.payload.get("forward_source") or {}
    if source.get("source") != SglangSplit.kind:
        return None
    files = [(f["path"], f["text"]) for f in source.get("files", ())
             if f.get("text") and f.get("path", "").endswith(".py")]
    if not files:
        return None
    revision = source.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"{model.model_id}: recorded SGLang source needs its revision")
    return stage_files(
        cache_dir / SglangSplit.kind / sha256_bytes(model.path.read_bytes()) / model.stem,
        files,
        {"kind": SglangSplit.kind, "revision": revision,
         "model_id": model.model_id,
         "architectures": model.config.get("architectures", []),
         "imported": list(source.get("imported") or ()),
         "read_from": [path for path, _text in files]})


def stage_reference(model: CorpusModel, cache_dir: Path) -> Optional[Path]:
    """The model's own reference implementation, as the snapshot recorded it
    (`backend.hf.forward_source`: the code the spec was written from, frozen
    in the snapshot, so a pinned run reads the same bytes offline). Its
    revision is the snapshot those bytes came from. None when the model ships
    no forward source."""
    from backend.hf.forward_source import recorded_forward_files
    files = [(f["path"], f["text"]) for f in recorded_forward_files(model.model_id)
             if f.get("text") and f.get("path", "").endswith(".py")]
    if not files:
        return None
    revision = f"snapshot sha256:{sha256_bytes(model.path.read_bytes())}"
    return stage_files(
        cache_dir / ReferenceSplit.kind / revision.rsplit(":", 1)[-1] / model.stem,
        [(PurePosixPath(path).name, text) for path, text in files],
        {"kind": ReferenceSplit.kind, "revision": revision,
         "model_id": model.model_id, "read_from": [path for path, _text in files]})


def stage_files(root: Path, files: Sequence[Tuple[str, str]],
                manifest: Mapping[str, Any]) -> Path:
    """Write the staged source and its manifest, so the pinned run — which
    has no network — reads what this process read."""
    for path, text in files:
        target = (root / path).resolve()
        if root.resolve() not in target.parents:
            raise ValueError(f"staged source path {path!r} leaves {root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / MANIFEST).write_text(json.dumps(dict(manifest), indent=1), encoding="utf-8")
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get(CHILD_ENV) != "1":
        # The cache lives under the caller's HOME, which the pinned
        # environment replaces.
        if "--cache-dir" not in argv and argv[:1] not in (["verify"], ["carry"]):
            argv += ["--cache-dir", str(default_cache_dir())]
        if "--framework-source" not in argv and argv[:1] not in (["verify"], ["carry"]):
            # A framework's files may be read over the network, which the
            # pinned run does not have; staged here, read there.
            for staged in stage_sources(argv):
                argv += ["--framework-source", str(staged)]
        return subprocess.run([sys.executable, "-m", "tools.corpus_admit", *argv],
                              cwd=ROOT, env=pinned_environment(), check=False).returncode
    if argv[:1] == ["verify"]:
        return verify_main(argv[1:])
    if argv[:1] == ["carry"]:
        return carry_main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m tools.corpus_admit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--base-sha")
    parser.add_argument("--base-kind", choices=("main", "deployment"), default="main")
    parser.add_argument("--base-lock-sha256",
                        help="sha256 of the base commit's uv.lock (default: read through git)")
    parser.add_argument("--env-root", type=Path,
                        help="the project this interpreter's environment was installed from")
    parser.add_argument("--orchestrator", help="the revision of the program that ran this")
    parser.add_argument("--bundle", type=Path, action="append")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--framework-source", type=Path, action="append",
                        help="one framework source's files for the model, staged with a "
                             "MANIFEST.json naming its kind; repeatable (default: every "
                             "source readable before the pinned run)")
    parser.add_argument("--preview-dir", type=Path,
                        help="with --dry-run: write the files a write would produce here")
    args = parser.parse_args(argv)
    report = Admission(args.model_id, DRY_RUN if args.dry_run else PR,
                       base_sha=args.base_sha, bundle=args.bundle,
                       cache_dir=args.cache_dir, preview_dir=args.preview_dir,
                       orchestrator=args.orchestrator, base_kind=args.base_kind,
                       base_lock_sha256=args.base_lock_sha256,
                       env_root=args.env_root,
                       framework_sources=args.framework_source or ()).run()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if report["status"] in (Status.READY.value, Status.ALREADY_ADMITTED.value) else 1


if __name__ == "__main__":
    sys.exit(main())
