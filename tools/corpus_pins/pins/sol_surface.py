"""P2: the public SOL surface, per (snapshot x deployment) cell.

`measure_case` lives in `scripts/dump_sol_surface.py`, the script that also
measures the other side of a cross-tree comparison; this pin shapes its
output into the golden's cells: model-level fields, both phases, and per-op
rows reduced to family sums plus per-row digests.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from tools.corpus_pins.corpus import ROOT, CorpusModel, shared_loader
from tools.corpus_pins.pin import JsonMapPin, Row, Scope


def _dump_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "dump_sol_surface", ROOT / "scripts" / "dump_sol_surface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DUMP = _dump_module()
DEPLOYMENTS = DUMP.DEPLOYMENTS
measure_case = DUMP.measure_case

#: The per-op columns summed per operator family.
OP_COLUMNS = (
    "count_per_step",
    "flops_per_rank",
    "access_bytes_per_rank",
    "weight_read_bytes_per_rank",
    "metadata_read_bytes_per_rank",
    "activation_bytes_per_rank",
    # The two halves the row above is the sum of. Summed per family for the
    # same reason as everything else here: `activation_bytes_per_rank` holding
    # still while the split inside it moves is a redistribution between reads
    # and writes, and the total cannot see it.
    "activation_read_bytes_per_rank",
    "activation_write_bytes_per_rank",
    # The quantised copy of its own operand a matmul's kernel needs when the
    # edge does not carry it: the fifth term of `access`, summed like the
    # rest so a family's total says whether it moved.
    "convert_bytes_per_rank",
)


#: Integers compare exactly. The floats here (`t_sol_s`, the byte figures the
#: result reports as floats) are deterministic arithmetic over exact integers,
#: so the allowance is last-ulp only, not a drift budget: two interpreters
#: (or two numpy builds) may round the last digit differently.
FLOAT_TOL = 1e-12


def same_value(recorded: Any, measured: Any) -> bool:
    """Equal, floats within `FLOAT_TOL` relative."""
    if isinstance(recorded, dict) and isinstance(measured, dict):
        return recorded.keys() == measured.keys() and all(
            same_value(recorded[k], measured[k]) for k in recorded)
    if isinstance(recorded, float) or isinstance(measured, float):
        return (recorded is not None and measured is not None
                and abs(measured - recorded) <= abs(recorded) * FLOAT_TOL)
    return recorded == measured


def cell_key(filename: str, deployment: str) -> str:
    return f"{filename}|{deployment}"


def reduce_ops(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Family sums plus a digest over the full row list.

    The sums say *what* moved in terms an operator name explains. The digest
    says whether anything moved at all — including a path rename or a shift
    between two ops of the same family, which the sums cannot see.
    """
    families: Dict[str, List[int]] = collections.defaultdict(
        lambda: [0] * len(OP_COLUMNS))
    for row in rows:
        acc = families[str(row["op"])]
        for i, column in enumerate(OP_COLUMNS):
            acc[i] += int(row[column])
    canonical = json.dumps(rows, sort_keys=True)
    # One short digest per row, keyed by the row's path and op (a repeated
    # key gets a running index), so a failure names the rows that moved
    # instead of reporting that a whole-surface digest changed.
    seen: Dict[str, int] = collections.defaultdict(int)
    row_digests: Dict[str, str] = {}
    for row in rows:
        base = f"{row['path']}|{row['op']}"
        seen[base] += 1
        key = base if seen[base] == 1 else f"{base}#{seen[base]}"
        row_digests[key] = hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "rows": len(rows),
        "digest": hashlib.sha256(canonical.encode()).hexdigest(),
        "row_digests": row_digests,
        "by_op": {
            op: dict(zip(OP_COLUMNS, sums, strict=True))
            for op, sums in sorted(families.items())
        },
    }


@dataclass(frozen=True)
class SolSurfacePin(JsonMapPin):
    def missing_keys(self, doc, model: CorpusModel) -> list:
        return [k for k in (cell_key(model.filename, d) for d in sorted(DUMP.deployments_for(model.filename)))
                if k not in doc]

    def key_exists(self, key: str, stems, components) -> bool:
        return key.split("|", 1)[0][:-len(".json")] in stems

    def value_matches(self, recorded: Any, measured: Any) -> bool:
        """As the golden test holds it: floats to the last digit only."""
        return same_value(recorded, measured)

    def measure(self, model: CorpusModel) -> Row:
        resolved = model.resolved(shared_loader())
        row: Row = {}
        for deployment in sorted(DUMP.deployments_for(model.filename)):
            cell = measure_case(model.config, DEPLOYMENTS[deployment], resolved)
            row[cell_key(model.filename, deployment)] = {
                "model": cell["model"],
                "phases": cell["phases"],
                "ops": {
                    phase: reduce_ops(rows)
                    for phase, rows in sorted(cell["ops"].items())
                },
            }
        return row


_TEST = "tests/backend/inference/test_sol_surface_golden.py::TestSolSurfaceGolden::"
SOL_SURFACE = SolSurfacePin(
    id="P2",
    title="SOL surface",
    scope=Scope.REQUIRED,
    artifact="tests/golden/sol_surface.json",
    owner_tests=tuple(_TEST + name for name in (
        "test_case_matrix_is_the_committed_one",
        "test_model_level_surface",
        "test_phase_surface",
        "test_per_op_family_totals",
        "test_per_op_rows_are_identical",
    )),
    waits_on=("G4",),
)
