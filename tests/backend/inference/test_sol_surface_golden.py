"""The corpus-wide SOL golden — every public number, every model, both phases.

Two commits in the eval swap claimed the numbers did not move and were wrong.
`aa42ac8` stopped charging decode for the cache read and shipped; the phase that
is *defined* by being cache-read-bound reported a step that never touched the
cache, and it survived four days until a person read a curve. `9af5f72` said
"No number moves" on the strength of a cached-element check, and moved decode
access in 48 cells across 24 models — the check was structurally blind to
traffic, so it could not have failed. The full accounting was
`docs/sol_reconciliation/report.md`, retired after commit f357dd547 (git history keeps it).

Neither commit was reckless. Both were checked against the only baselines that
existed, and both baselines were narrow in a direction nobody had written down:
`tests/test_capacity_golden.py` pins seven hand-calibrated cells on GPUs,
dtypes and parallel layouts of their own, and pins per-op work only through the
*sum* of its timings. Nothing pinned a phase's FLOPs, its access decomposition,
a KV growth slope, or any per-op row, anywhere in the corpus.

So this golden is deliberately the opposite shape: no calibration, no measured
seeds, no hand-picked deployments — one fixed request against every snapshot,
under a one-GPU and an eight-GPU layout, with every column the public result
exposes written down. It is wide and shallow where the capacity golden is
narrow and deep, and the two are meant to be read together.

What is pinned, per (snapshot × deployment)
-------------------------------------------
* model: weight footprint, parameter count, the cache law the budget uses —
  per-rank growth per token of length and per-request constant (a full
  window, a recurrent state), as `kv_cache` publishes them — static KV
  scales, max batch, max tokens (None when the cache never grows);
* phase (prefill AND decode): batch, FLOPs, access bytes and the weight-read /
  metadata / activation+KV decomposition it closes into, `mem_read_bytes`,
  `mem_write_bytes`, `t_sol_s`;
* per-op: the six numeric columns summed per operator family — so a failure
  names the op that moved — plus a digest over the full row list, which is what
  catches a change that redistributes work inside a family or renames a path.

The cells come from the pin (`tools/corpus_pins/pins/sol_surface.py`), which
shapes `scripts/dump_sol_surface.py`'s `measure_case` — the script that
measures the *other* side of a cross-tree comparison — into the golden's
cells. One declaration of the columns, and the admitter measures a new
model's cells with the same code.

What a cell pins when the deployment cannot load the weights
------------------------------------------------------------
63 of the 126 cells are deployments that refuse: the weights alone exceed
usable HBM on the card, so the estimate publishes no step at any batch
(`infeasibility_reason`). Since `d2ab068b` such a card runs no phase walk
either — there is nothing measured at one request for this file to pin, and
the walk it used to take (which this file pinned on the grounds that every
other card and batch was re-priced from it) is not taken, because a card
class is re-priced from its own representative's walk and a class whose
roomiest card refuses has no card that does not.

So a refused cell pins what the budget answers without a step: the weight
footprint module by module, the parameter count, the cache law, the
ceilings. Its phase and per-op columns are `None` beside the reason, not
zero — a withheld step is not a step that moved no bytes, and writing the
zero would make a step reappearing here read as growth from nothing.

Those cells are therefore not watching per-op accounting, and for 22 of the
63 models — DeepSeek-V3 / R1 / V3.1 / V3.2 / V4-Pro, Kimi-K2 / K3, GLM-4.5
and 5 through 5.3, Ling-2.6-1T, Qwen3.8-2.4T-A95B, Nemotron-3-Ultra,
Step-5-Preview, Hy4-preview, both LongCats, MiMo-V2.5-Pro — neither cell
is, because they refuse on one H100 and on eight alike. Closing that needs
a layout this matrix does not have (a wide one that holds a 1T-parameter
model), and adding a deployment is a case-matrix change: its own commit, so
the diff on every other cell stays reviewable.

One of the three `*_read`/`*_write` splits the report asks for is still absent,
and absent for a reason rather than by omission: the cache a step reads arrives
as an INPUT to the attention leaf in the post-swap tree, so a KV-read column
would mean splitting that input by provenance, which the tree does not carry.

The other two are pinned. The write half used to be the literal zero in every
cell — a leaf's access is `weight + metadata + act_in + act_out`, so the write
was never missing from the total, only published under a field named "read".
The read half used to be `None`, because the dump re-derived the weight read
from the overlay batch — `min(B x s_ctx x per_token_active, total)` — and took
activation as the remainder, so subtracting the write out of it would have
reported the re-derivation's error as a read. That re-derivation used the
step's token count where the engine's own min reads the leaf's rows (the
busiest rank's, for a routed leaf); the two agree only while routing is
balanced and attention is not data-parallel, and a remainder cannot say
when they stop.
Both halves are folds of published engine columns now — `act_in_bytes_per_rank`
and `act_out_bytes_per_rank` — and the dump refuses to write a cell where
`weight_read + metadata + act_in + act_out` does not equal the row's access.

When this surface is measured, and what that costs you
------------------------------------------------------
Nothing runs it unless you ask. The class is `regression`-marked (one
measurement of 126 cells across 63 models, ~35 min), `pyproject.toml`'s
`addopts` deselects that mark, and `.github/workflows/regression.yml` is
`workflow_dispatch` only — so the default run, the push / PR lane and the
deploy gate all go green without this file executing once. It said the
opposite here until 2026-10-31 ("95 seconds … an alarm that rings without
being asked to"), written when there were 78 cells and left standing after
`6fc159d47` gave the class its half hour two days before six engine commits
landed on top of it.

Those six (`a53fd4f38`, `455ba648f`, `9f3d870e8`, `f34d4dd02`, `a41780b24`,
`098803f40`, re-pinned together in `c69952da4`) each said no number moved,
and each was right about every check that ran. Together they moved 56 cells
across 29 models, including 96 row keys this file would have named —
`test_per_op_rows_are_identical` prints
`- <path>|<op>` for a key the corpus stopped producing, which is exactly
what `455ba648f` left behind. The alarm was not blind. It was switched off,
by a commit that had a good reason and did not say what it was suspending.

The pin was next re-pinned for the DeepSeek-V4 router-width declaration
(`24eb51bf3`, corrected by `6a95878b2`): exactly the six V4 / V4-Pro /
V4.1 cells moved. The router's FP32 output doubles the logits write the
`linear_param` family reports (and the hash router's activation reads beside
it); FLOPs, weights, row identities outside those router families and every
other model stayed on the old surface.

So: **if your change can move any model, run this lane before you claim it
did not.** `-m regression -k sol_surface` locally, or Actions → regression on
your branch. A green default suite is not evidence about this surface.

Regenerating
------------
    REGEN_GOLDEN=1 PYTHONPATH=. python -m pytest tests/backend/inference/test_sol_surface_golden.py

Then read the diff before committing it: that diff is the record of what moved,
and a commit that regenerates this file without explaining every changed line
is the exact move that let both regressions above pass.
"""
from __future__ import annotations

import os
import sys
import unittest
import pytest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.pins.sol_surface import FLOAT_TOL, SOL_SURFACE

pytestmark = [
    pytest.mark.regression,  # walks every snapshot
    # The lane's 900 s is per test; this class's one measurement (`setUpClass`
    # walks every model, ~35 min) lands in the first test's setup and is the
    # surface itself, not a hang.
    pytest.mark.timeout(3600),
]

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
GOLDEN_PATH = SOL_SURFACE.path


#: How many mismatching cells a failure prints before summarizing the rest.
REPORT_LIMIT = 40


def _mismatches(got: Dict[str, Any], want: Dict[str, Any], prefix: str) -> List[str]:
    out: List[str] = []
    for key in sorted(set(got) | set(want)):
        new, old = got.get(key), want.get(key)
        if new is None and old is None:
            continue
        if isinstance(new, float) or isinstance(old, float):
            if new is None or old is None:
                out.append(f"{prefix}.{key}: {old} -> {new}")
            elif abs(new - old) > abs(old) * FLOAT_TOL:
                ratio = f" (x{new / old:.6g})" if old else ""
                out.append(f"{prefix}.{key}: {old} -> {new}{ratio}")
        elif new != old:
            ratio = ""
            if isinstance(new, int) and isinstance(old, int) and old:
                ratio = f" (x{new / old:.6g})"
            out.append(f"{prefix}.{key}: {old} -> {new}{ratio}")
    return out


def _render(mismatches: List[str]) -> str:
    head = "\n".join("  " + line for line in mismatches[:REPORT_LIMIT])
    if len(mismatches) > REPORT_LIMIT:
        head += f"\n  ... and {len(mismatches) - REPORT_LIMIT} more"
    return f"{len(mismatches)} column(s) moved:\n{head}"


def _maybe_regen() -> bool:
    if os.environ.get("REGEN_GOLDEN") != "1":
        return False
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_bytes(SOL_SURFACE.render(SOL_SURFACE.measure_all()))
    return True


class TestSolSurfaceGolden(unittest.TestCase):
    """Every cell reproduces the committed surface, column by column.

    A failure is not a request to regenerate. It is the alarm the two swap
    commits did not have: find the commit that predicts the size of what
    moved, and if no commit does, that is the finding.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        if _maybe_regen():
            raise unittest.SkipTest(
                "REGEN_GOLDEN=1 — surface rewritten, not asserted")
        if not GOLDEN_PATH.exists():
            raise unittest.SkipTest(
                f"golden missing — run REGEN_GOLDEN=1 pytest {Path(__file__).name}")
        cls.golden = SOL_SURFACE.load()
        cls.measured = SOL_SURFACE.measure_all()

    def test_case_matrix_is_the_committed_one(self) -> None:
        missing = sorted(set(self.golden) - set(self.measured))
        added = sorted(set(self.measured) - set(self.golden))
        self.assertEqual(
            (missing, added), ([], []),
            "the corpus changed under the golden (absent, new). A snapshot "
            "added or removed is a legitimate regeneration — do it in the "
            "commit that changes the corpus, so the diff on every OTHER cell "
            "is reviewable on its own",
        )

    def test_model_level_surface(self) -> None:
        bad: List[str] = []
        for key in sorted(set(self.golden) & set(self.measured)):
            bad += _mismatches(
                self.measured[key]["model"], self.golden[key]["model"], key)
        self.assertEqual(bad, [], _render(bad))

    def test_phase_surface(self) -> None:
        bad: List[str] = []
        for key in sorted(set(self.golden) & set(self.measured)):
            for phase in ("prefill", "decode"):
                bad += _mismatches(
                    self.measured[key]["phases"][phase],
                    self.golden[key]["phases"][phase],
                    f"{key}.{phase}",
                )
        self.assertEqual(bad, [], _render(bad))

    def test_per_op_family_totals(self) -> None:
        bad: List[str] = []
        for key in sorted(set(self.golden) & set(self.measured)):
            for phase in ("prefill", "decode"):
                new = self.measured[key]["ops"][phase]["by_op"]
                old = self.golden[key]["ops"][phase]["by_op"]
                for op in sorted(set(new) | set(old)):
                    bad += _mismatches(
                        new.get(op, {}), old.get(op, {}),
                        f"{key}.{phase}.{op}",
                    )
        self.assertEqual(bad, [], _render(bad))

    def test_per_op_rows_are_identical(self) -> None:
        """The per-row digests catch what the family sums are blind to.

        If this fails alone — every family total identical, rows not — the
        per-op work did not change size; a path was renamed, or it moved
        between two rows of the same operator. The failure lists the rows
        added (+), removed (-) and changed (~), so the commit that regenerates
        the golden can say which ones and why.
        """
        bad: List[str] = []
        for key in sorted(set(self.golden) & set(self.measured)):
            for phase in ("prefill", "decode"):
                new = self.measured[key]["ops"][phase]
                old = self.golden[key]["ops"][phase]
                if new["digest"] == old["digest"]:
                    continue
                new_rows, old_rows = new["row_digests"], old["row_digests"]
                for row in sorted(set(new_rows) | set(old_rows)):
                    if row not in old_rows:
                        bad.append(f"{key}.{phase}: + {row}")
                    elif row not in new_rows:
                        bad.append(f"{key}.{phase}: - {row}")
                    elif new_rows[row] != old_rows[row]:
                        bad.append(f"{key}.{phase}: ~ {row}")
        self.assertEqual(bad, [], _render(bad))


if __name__ == "__main__":
    unittest.main()
