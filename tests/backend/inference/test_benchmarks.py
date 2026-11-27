"""Tests for the benchmark CSV strict validator.

Covers both positive cases (well-formed CSV passes, round-trip byte-identity,
two rows at different spec_hash are distinct) and negative cases (missing
column, unknown column, bad type, unknown shape symbol, unknown dtype slot,
future schema_version, duplicate identity).

Schema reference: `backend/inference/BENCHMARKS_CSV.md`.
"""
from __future__ import annotations


import csv
import io
import json
import os
import pathlib
from typing import Any, ClassVar

import pytest

from tests.snapshots import snapshot_config

import tests.corpus_row as _corpus
from backend.inference.benchmarks import (
    ALL_COLUMNS,
    IDENTITY_COLUMNS,
    MEASUREMENT_COLUMNS,
    PROVENANCE_COLUMNS,
    KNOWN_SCHEMA_VERSION,
    BenchmarkValidationError,
    assert_round_trip,
    emit_csv,
    parse_rows,
    validate_text,
)


def _base_row(**over) -> dict:
    """One shared row, not a sixth copy — see `tests/corpus_row.py`."""
    return _corpus.row(**over)


def _emit(rows: list) -> str:
    return _corpus.to_csv(rows)



class TestColumnSets:
    def test_total_column_count_is_43(self):
        assert len(ALL_COLUMNS) == 43

    def test_identity_columns_are_12(self):
        assert len(IDENTITY_COLUMNS) == 12

    def test_measurement_columns_are_9(self):
        assert len(MEASUREMENT_COLUMNS) == 9

    def test_provenance_columns_are_22(self):
        assert len(PROVENANCE_COLUMNS) == 22

    def test_no_duplicates_across_groups(self):
        combined = list(IDENTITY_COLUMNS) + list(MEASUREMENT_COLUMNS) + list(PROVENANCE_COLUMNS)
        assert len(combined) == len(set(combined))


class TestWellFormedPasses:
    def test_single_row_passes(self):
        text = _emit([_base_row()])
        header, rows = validate_text(text, source="ok.csv")
        assert len(rows) == 1
        assert rows[0]["op"] == "flash_attention"
        assert set(header) == set(ALL_COLUMNS)

    def test_two_rows_at_different_spec_hash_pass(self):
        r1 = _base_row()
        r2 = _base_row()
        r2["spec_hash"] = "llama32_3b_v2"
        text = _emit([r1, r2])
        _, rows = validate_text(text, source="ok2.csv")
        assert len(rows) == 2
        assert rows[0]["spec_hash"] != rows[1]["spec_hash"]


class TestRoundTrip:
    def test_emit_then_parse_equals_input(self, tmp_path):
        text = _emit([_base_row()])
        path = tmp_path / "rt.csv"
        path.write_text(text)
        assert_round_trip(str(path))

    def test_parse_then_emit_byte_identical(self):
        text = _emit([_base_row()])
        header, rows = parse_rows(text)
        emitted = emit_csv(header, rows)
        assert emitted == text


# -------------------- negative cases --------------------


class TestMissingRequiredColumns:
    """Missing peak_tflops_at_test, peak_bw_at_test, spec_hash,
    node_spec_id, count_per_step, framework_version, or kernel_impl is
    rejected with named field.
    """

    @pytest.mark.parametrize("missing_col", [
        "peak_tflops_at_test",
        "peak_bw_at_test",
        "spec_hash",
        "node_spec_id",
        "count_per_step",
        "framework_version",
        "kernel_impl",
    ])
    def test_header_missing_column_is_rejected(self, missing_col):
        # Construct a CSV header that omits the chosen column.
        kept_cols = [c for c in ALL_COLUMNS if c != missing_col]
        row = _base_row()
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(kept_cols)
        w.writerow([row[c] for c in kept_cols])
        text = buf.getvalue()
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        # Named field appears in the error.
        assert missing_col in str(exc.value), (
            f"error must name the missing column {missing_col!r}: "
            f"{exc.value}"
        )


class TestExtraColumnRejected:
    def test_unknown_top_level_column_is_rejected(self):
        row = _base_row()
        row["secret_sauce"] = "💎"
        cols = [*list(ALL_COLUMNS), "secret_sauce"]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        w.writerow([row[c] for c in cols])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(buf.getvalue(), source="bad.csv")
        assert "secret_sauce" in str(exc.value)


class TestHeaderMismatchErrorNamesTrackedDoc:
    """Regression: header-mismatch error must point users at the
    tracked canonical schema doc (`backend/inference/BENCHMARKS_CSV.md`),
    not the gitignored `.claude/rules/benchmarks-csv.md` pointer."""

    def test_header_mismatch_error_cites_tracked_canonical_doc(self):
        cols = [c for c in ALL_COLUMNS if c != "spec_hash"]
        row = _base_row()
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        w.writerow([row[c] for c in cols])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(buf.getvalue(), source="bad.csv")
        msg = str(exc.value)
        assert "backend/inference/BENCHMARKS_CSV.md" in msg, (
            f"error must direct users to the tracked schema doc, got: {msg}"
        )
        assert ".claude/rules/benchmarks-csv.md" not in msg, (
            f"error must not cite the gitignored pointer, got: {msg}"
        )


class TestOptionalColumnsAndTheMigrationToolAgree:
    """`OPTIONAL_COLUMNS` is the set an older CSV may omit. The migration
    tool must declare a fill for exactly those, because it REFUSES any
    column it has no fill for — a right default that also means a schema
    bump which forgets one leaves the tool unable to migrate anything.

    This is how v13 shipped: `xcheck_typical` / `xcheck_worst` had no fill,
    so every v12 CSV handed to `migrate_schema` was refused, and nothing
    noticed — the optional set was a local variable inside `validate_header`,
    so no test could compare the two halves.
    """

    def test_every_optional_column_has_a_declared_migration_fill(self):
        from backend.inference.benchmarks import OPTIONAL_COLUMNS
        import tools.migrate_schema as migrate
        undeclared = sorted(OPTIONAL_COLUMNS - set(migrate.FILL))
        assert undeclared == [], (
            f"these columns may be absent from an older CSV but the "
            f"migration tool declares no fill, so it refuses every such "
            f"file: {undeclared}")

    def test_every_optional_column_is_a_real_column(self):
        from backend.inference.benchmarks import OPTIONAL_COLUMNS
        assert set(ALL_COLUMNS) >= OPTIONAL_COLUMNS

    def test_no_identity_column_is_optional(self):
        """An identity column may never be omitted: a row missing part of its
        key is not a row whose key is partly unknown, it is unindexable."""
        from backend.inference.benchmarks import OPTIONAL_COLUMNS
        assert OPTIONAL_COLUMNS.isdisjoint(IDENTITY_COLUMNS)

    def test_the_header_gate_reads_that_set(self):
        """Excludes a second copy of the optional set inside the validator:
        a column named optional here must actually be omittable."""
        from backend.inference.benchmarks import OPTIONAL_COLUMNS
        for column in sorted(OPTIONAL_COLUMNS):
            cols = [c for c in ALL_COLUMNS if c != column]
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(cols)
            w.writerow([_base_row()[c] for c in cols])
            validate_text(buf.getvalue(), source="old.csv")  # no raise


class TestTheSchemaDocsCountsAreTheSchemas:
    """Every CLAIM the schema doc makes about the schema, against the code.

    The doc is declared the single source of truth for external producers, so
    each of its restatements is a second copy of something the code owns
    (CLAUDE.md §1.4). They stay, because a reader needs them — so they get a
    gate, and the gate covers CLAIMS, not just counts.

    It started as counts only, which was too narrow and provably so: the same
    change that added it shipped the document wrong in three further places
    that it did not read — the current schema version (12 against the code's
    14), the list of nullable columns (four missing), and a blank line that
    cut the last two rows out of their own Markdown table. Counts were simply
    the drift that had already been noticed. So: versions, the optional and
    nullable sets, and table structure, each derived from the code or from
    the validator rather than from a second list kept here.
    """

    def _doc(self) -> str:
        path = (pathlib.Path(__file__).resolve().parents[3]
                / "backend" / "inference" / "BENCHMARKS_CSV.md")
        return path.read_text(encoding="utf-8")

    def _cited(self, text: str) -> set:
        """The backticked column names a paragraph of prose lists."""
        import re
        return {m for m in re.findall(r"`([a-z][a-z0-9_]*)`", text)
                if m in ALL_COLUMNS}

    def test_the_group_sizes_the_doc_states_are_the_code_s(self):
        import re
        text = self._doc()
        found = re.search(
            r"CSV 共 \*\*(\d+) 列\*\*.*?(\d+) 个 identity 列、(\d+) 个 "
            r"measurement 列、(\d+) 个 provenance 列", text)
        assert found, "the doc no longer states its column counts in the form this reads"
        assert [int(g) for g in found.groups()] == [
            len(ALL_COLUMNS), len(IDENTITY_COLUMNS),
            len(MEASUREMENT_COLUMNS), len(PROVENANCE_COLUMNS)]

    def test_the_whitelist_paragraph_states_the_same_total(self):
        import re
        counts = {int(m) for m in re.findall(
            r"CSV header 必须是这 (\d+) 列", self._doc())}
        assert counts == {len(ALL_COLUMNS)}

    def test_every_column_appears_in_the_doc(self):
        """A column the schema has and the doc never mentions is undocumented,
        which for this file is the whole point of it existing."""
        text = self._doc()
        missing = [c for c in ALL_COLUMNS if f"`{c}`" not in text]
        assert missing == []

    def test_every_column_has_a_row_in_a_field_TABLE(self):
        """Mentioned somewhere is not documented. Each column needs a row in
        one of the field tables, and that row has to be IN the table.

        A blank line ends a Markdown table, so the two v14 rows were added
        just past the end of the provenance table and rendered as literal
        pipes — present to a grep, invisible to a reader. The old gate saw
        the backticked names and stayed green, which is how it shipped.
        """
        lines = self._doc().splitlines()
        rows = {}
        for i, line in enumerate(lines):
            for c in ALL_COLUMNS:
                if line.startswith(f"| `{c}` |"):
                    rows[c] = i
        missing = sorted(set(ALL_COLUMNS) - set(rows))
        assert missing == [], f"no field-table row for: {missing}"
        # A row is in a table only if the run of `|` lines it belongs to also
        # holds the `|---|` separator that opens one.
        for column, i in sorted(rows.items(), key=lambda kv: kv[1]):
            start = i
            while start > 0 and lines[start - 1].startswith("|"):
                start -= 1
            block = lines[start:i + 1]
            assert any(set(ln) <= set("|- :") and "-" in ln for ln in block), (
                f"`{column}`'s row at line {i + 1} is not inside a Markdown "
                f"table — a blank line above it ended the table, so it "
                f"renders as literal text")

    def test_the_nullable_and_required_prose_is_what_the_validator_does(self):
        """The doc says which provenance columns may be EMPTY. Derived by
        asking the validator, column by column, so there is no second list
        here either.

        It was wrong in two directions at once: it claimed the first five are
        required when `sm` (the fourth) may be empty, and its nullable list
        omitted `launch_signature`, `outcome`, `xcheck_typical` and
        `xcheck_worst`. A contributor following it would expect the validator
        to reject rows it accepts.
        """
        import re
        actually_required = []
        for column in PROVENANCE_COLUMNS:
            try:
                from backend.inference.benchmarks import validate_row
                validate_row(_base_row(**{column: ""}))
            except BenchmarkValidationError:
                actually_required.append(column)
        text = self._doc()
        said = re.search(r"\*\*必须有值\*\*（(\d+) 列）：([^\n]+)", text)
        assert said, "the doc no longer states its required-value set in the form this reads"
        assert int(said.group(1)) == len(actually_required)
        assert self._cited(said.group(2)) == set(actually_required), (
            f"doc says {sorted(self._cited(said.group(2)))}, the validator "
            f"requires {sorted(actually_required)}")
        # And the count of the ones that may be empty.
        nullable = [c for c in PROVENANCE_COLUMNS if c not in actually_required]
        empties = {int(m) for m in re.findall(r"其余 (\d+) 列值可为空", text)}
        assert empties == {len(nullable)}

    def test_the_omittable_prose_is_OPTIONAL_COLUMNS(self):
        """The doc's "may be absent from the header" list, against the set the
        header gate actually exempts. This is the claim an external producer
        acts on, and it is a different claim from "may be empty" — writing
        them as one sentence is what let both halves rot."""
        import re
        from backend.inference.benchmarks import OPTIONAL_COLUMNS
        text = self._doc()
        said = re.search(
            r"\*\*可从 header 缺列\*\*（(\d+) 列，`OPTIONAL_COLUMNS`）：([^\n]+)",
            text)
        assert said, "the doc no longer states its omittable set in the form this reads"
        assert int(said.group(1)) == len(OPTIONAL_COLUMNS)
        assert self._cited(said.group(2)) == set(OPTIONAL_COLUMNS), (
            f"doc says {sorted(self._cited(said.group(2)))}, the gate exempts "
            f"{sorted(OPTIONAL_COLUMNS)}")

    def test_the_schema_version_the_doc_states_is_the_code_s(self):
        """The counts were gated and the VERSION STRING was not, so it rotted
        next — it sat at 12 while the code said 14, in the same change that
        added the count gate.

        This is the worst of the drifts, not the most cosmetic: the validator
        rejects only HIGHER schema versions, so an external producer
        following this document stamps driver-bearing rows `schema_version=12`
        and the validator accepts them. The row's self-reported version then
        permanently disagrees with its own column set.
        """
        import re
        stated = {int(m) for m in re.findall(
            r"KNOWN_SCHEMA_VERSION\s*=\s*(\d+)", self._doc())}
        assert stated == {KNOWN_SCHEMA_VERSION}, (
            f"the doc states schema version(s) {sorted(stated)}; the code "
            f"says {KNOWN_SCHEMA_VERSION}")

    def test_the_doc_cites_no_schema_symbol_that_no_longer_exists(self):
        """Backticked ALL-CAPS names in this document are `benchmarks`
        module constants, and the document is where an external producer
        learns them. `_OPTIONAL_COLUMNS` outlived its own removal here:
        hoisting it to `OPTIONAL_COLUMNS` left the canonical reference
        naming a symbol that no longer exists.

        Generic on purpose — it catches the next renamed constant without
        anyone remembering to add it.
        """
        import re
        from backend.inference import benchmarks
        cited = {m for m in re.findall(r"`([A-Z][A-Z0-9_]{3,})`", self._doc())}
        # `_`-prefixed names are caught too: the doc writes them backticked
        # the same way, and a private name is exactly the kind that moves.
        cited |= {m for m in re.findall(r"`(_[A-Z][A-Z0-9_]{3,})`", self._doc())}
        absent = sorted(n for n in cited if not hasattr(benchmarks, n))
        assert absent == [], (
            f"the doc backticks these as bare ALL-CAPS names and they are "
            f"not `benchmarks` constants: {absent}. Either the symbol was "
            f"renamed and the doc still names the old one, or this is an "
            f"environment variable — write those `$NAME`, which is the "
            f"convention this check relies on to tell the two apart.")


class TestShapeJsonSymbolRestriction:
    """shape_json keys outside the dim registry + phase-binding extras
    (s_query, s_kv, s_cache, n_verify, s_past, s_pairs) are rejected.
    """

    def test_unknown_shape_symbol_is_rejected(self):
        row = _base_row()
        row["shape_json"] = json.dumps({"B": 1, "s_ctx": 2048, "FOOBAR": 7})
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "FOOBAR" in str(exc.value)

    def test_phase_binding_extras_are_allowed(self):
        row = _base_row()
        row["shape_json"] = json.dumps(
            {"B": 1, "s_query": 1, "s_kv": 2048, "s_cache": 2048, "n_heads": 24, "d_head": 128}
        )
        text = _emit([row])
        validate_text(text, source="ok.csv")

    def test_engine_supplied_dims_are_allowed(self):
        row = _base_row()
        row["shape_json"] = json.dumps({"B": 1, "s_ctx": 128})
        text = _emit([row])
        validate_text(text, source="ok.csv")

    def test_an_axis_a_signature_declares_is_allowed_outside_the_registry(self):
        # The Mamba-2 scan chunk is the framework binding's, not a model
        # dim, so `dims.yaml` does not name it; the `mamba2_scan` signature
        # declares it as an axis of its cell, and that declaration is what
        # lets a measured scan row keep it.
        from backend.arch import dims
        from backend.inference.op_signatures import get_signature
        assert "ssm_chunk_size" not in dims.base_symbols()
        assert "ssm_chunk_size" in get_signature("mamba2_scan").cell_dims
        row = _base_row()
        row["shape_json"] = json.dumps(
            {"B": 1, "d_ssm_head": 64, "d_ssm_state": 128, "n_ssm_groups": 8,
             "n_ssm_heads": 64, "s_query": 1024, "ssm_chunk_size": 128})
        validate_text(_emit([row]), source="ok.csv")

    def test_shape_value_must_be_positive_int(self):
        row = _base_row()
        row["shape_json"] = json.dumps({"B": 1, "s_ctx": 0})
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "positive integer" in str(exc.value)


class TestDtypeEnvSlotRestriction:
    def test_unknown_slot_is_rejected(self):
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "bf16", "gradient": "fp32"}
        )
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "gradient" in str(exc.value)

    def test_unknown_dtype_value_is_rejected(self):
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "tf32", "weight": "bf16"}
        )
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "tf32" in str(exc.value)

    def test_missing_activation_slot_is_rejected(self):
        row = _base_row()
        row["dtype_env_json"] = json.dumps({"weight": "bf16"})
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "activation" in str(exc.value)

    def test_every_kernel_states_its_activation_and_output(self):
        """Every kernel has an activation and an output, so a row without
        one names no environment: it is rejected, not hashed as BF16."""
        row = _base_row()
        row["dtype_env_json"] = json.dumps({"activation": "bf16"})
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "output" in str(exc.value)

    def test_a_slot_that_does_not_apply_is_left_out(self):
        """No weight, no cache: an activation kernel states neither."""
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "bf16", "output": "bf16"})
        validate_text(_emit([row]), source="ok.csv")

    def test_scales_need_the_weight_they_scale(self):
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "bf16", "output": "bf16", "metadata": "fp32"})
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "weight" in str(exc.value)

    def test_a_weight_without_scales_omits_metadata(self):
        """`metadata` is the dtype of the weight's own scales: a row whose
        weight stores none leaves the slot out (rule 3)."""
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "bf16", "weight": "bf16", "kv": "bf16",
             "output": "bf16"})
        validate_text(_emit([row]), source="ok.csv")

    def test_a_weight_with_scales_states_their_dtype(self):
        """A granularity says the weight stores scales, so their dtype is
        required beside it."""
        row = _base_row()
        row["dtype_env_json"] = json.dumps(
            {"activation": "fp8_e4m3", "weight": "fp8_e4m3", "kv": "bf16",
             "output": "bf16", "weight_granularity": "per-tile:128x128"})
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "metadata" in str(exc.value)


class TestSmColumn:
    """`sm` records the compute-capability tag of the card a row was measured
    on. It is PROVENANCE, not identity — `gpu` already keys the physical
    corpus, so adding `sm` must not perturb any calibration match."""

    def test_sm_is_provenance_not_identity(self):
        from backend.inference.benchmarks import (
            IDENTITY_COLUMNS, PROVENANCE_COLUMNS,
        )
        assert "sm" in PROVENANCE_COLUMNS
        assert "sm" not in IDENTITY_COLUMNS

    def test_sm_column_round_trips(self):
        row = _base_row()
        row["sm"] = "sm_89"
        _header, rows = validate_text(_emit([row]), source="ok.csv")
        assert rows[0]["sm"] == "sm_89"

    def test_sm_is_required_like_other_columns(self):
        # Dropping it is a header mismatch (strict closed column set).
        cols = [c for c in ALL_COLUMNS if c != "sm"]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        w.writerow([_base_row()[c] for c in cols])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(buf.getvalue(), source="bad.csv")
        assert "sm" in str(exc.value)


class TestDriverColumns:
    """v14: WHICH DRIVER RAN THE ROW — the box's kernel-mode driver and the
    user-mode driver the measuring process actually loaded.

    The pair exists because one box can measure some envs through NVIDIA's
    forward-compatibility user-mode driver and others natively, and before
    these columns nothing on a row said which.
    """

    def test_the_pair_is_declared_once_and_every_consumer_spreads_it(self):
        """`DRIVER_COLUMNS` is the one declaration; the schema tuple, the
        optional set and the lookup's wire all spread it.

        Three separate spellings of the same two names is the §1.4 name-list
        pattern, which this branch of all branches should not ship.
        """
        from backend.inference.benchmarks import (
            DRIVER_COLUMNS, OPTIONAL_COLUMNS,
        )
        from backend.inference.op_calibration import ROW_COLUMNS
        assert set(DRIVER_COLUMNS) <= set(PROVENANCE_COLUMNS)
        assert set(DRIVER_COLUMNS) <= OPTIONAL_COLUMNS
        assert set(DRIVER_COLUMNS) <= set(ROW_COLUMNS)

    def test_the_derivation_reads_the_columns_the_pair_declares(self):
        """Excludes the silent failure a rename would cause.

        `is_driver_substituted` reads the two keys by name. If the columns
        were renamed in `DRIVER_COLUMNS` and the function missed, it would
        find neither key, read both as absent, and answer `None` — UNKNOWN —
        for every row in the corpus, while every other test here still
        passed. So this builds its row from the declaration.
        """
        from backend.inference.benchmarks import (
            DRIVER_COLUMNS, is_driver_substituted,
        )
        kmd, umd = DRIVER_COLUMNS
        assert is_driver_substituted({kmd: "575.57.08", umd: "580.95.05"}) is True
        assert is_driver_substituted({kmd: "595.84", umd: "595.84"}) is False

    def test_drivers_are_provenance_not_identity(self):
        # Keyed, they would split the corpus permanently rather than
        # supersede: every pre-v14 row has them empty, and an unknown-driver
        # key would never again meet a known-driver one.
        for c in ("kmd_version", "umd_version"):
            assert c in PROVENANCE_COLUMNS
            assert c not in IDENTITY_COLUMNS

    def test_both_drivers_round_trip(self):
        # Distinct values, so a reader that confused the two columns is red
        # here rather than silently reporting the wrong driver.
        row = _base_row(kmd_version="575.57.08", umd_version="580.95.05")
        _header, rows = validate_text(_emit([row]), source="ok.csv")
        assert rows[0]["kmd_version"] == "575.57.08"
        assert rows[0]["umd_version"] == "580.95.05"

    @pytest.mark.parametrize("value", [
        "575.57.08",    # a packaged kernel module's three-part form
        "595.84",       # an open kernel module's two-part form
        "1000.12.34",   # a four-digit branch, if NVIDIA ever ships one
    ])
    def test_driver_shaped_values_accepted(self, value):
        for column in ("kmd_version", "umd_version"):
            validate_text(_emit([_base_row(**{column: value})]), source="ok.csv")

    @pytest.mark.parametrize("value", [
        "13.0",         # A CUDA RUNTIME VERSION — see the test below
        "12.9",
        "13.0.1.2",     # a four-part CUDA version
        "13.0.1.2.x",   # trailing junk
        "sm_90",        # the neighbouring column's vocabulary
        "575",          # a bare branch number: no dot, so not a driver version
        "r575.57.08",   # the prose form, not what nvidia-smi prints
        "575.57.08-ubuntu1",
    ])
    def test_non_driver_shaped_values_rejected(self, value):
        """Excludes the "accept it as free text" implementation.

        The shape matters because the two columns are compared to EACH OTHER
        as strings: a CUDA version written into `umd_version` beside a driver
        version in `kmd_version` would read as a substituted driver and flag
        an honest row.
        """
        for column in ("kmd_version", "umd_version"):
            with pytest.raises(BenchmarkValidationError) as exc:
                validate_text(_emit([_base_row(**{column: value})]),
                              source="bad.csv")
            assert column in str(exc.value)

    def test_a_cuda_version_cannot_pass_as_a_substituted_driver(self):
        """REGRESSION. The contract these columns document is that a CUDA
        version does not belong in them; `^\\d+(\\.\\d+)+$` accepted `13.0`,
        so for one review cycle the contract was false and the validator was
        a second, laxer reading of the same fact.

        The exact upload that got through: a real kernel driver beside a
        runtime version, which validated and then read as a substituted
        driver — presenting `13.0` to a user as the driver that ran a kernel.
        Both halves are asserted here, because rejecting the value is only
        useful if it is the thing that stops the bad verdict.
        """
        row = _base_row(kmd_version="575.57.08", umd_version="13.0")
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "umd_version" in str(exc.value)
        assert "13.0" in str(exc.value)
        # Were it ever to validate, this is the verdict it would carry — so
        # the rejection above is load-bearing, not cosmetic.
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(row) is True

    def test_the_schema_owns_the_driver_grammar(self):
        """The producer reads `/proc` with the same pattern the column is
        validated against, imported rather than restated — otherwise the
        harness can measure a value its own validator rejects."""
        from backend.inference.benchmarks import DRIVER_VERSION_PATTERN
        import kernel_bench.env as env
        assert DRIVER_VERSION_PATTERN in env._MAPPED_UMD.pattern
        assert env._DRIVER_VERSION.pattern == DRIVER_VERSION_PATTERN

    def test_empty_is_accepted_on_both(self):
        """Excludes the "make them required" implementation, which would
        reject every CSV written before v14."""
        row = _base_row(kmd_version="", umd_version="")
        _header, rows = validate_text(_emit([row]), source="ok.csv")
        assert rows[0]["kmd_version"] == ""
        assert rows[0]["umd_version"] == ""

    @pytest.mark.parametrize("dropped", ["kmd_version", "umd_version"])
    def test_a_pre_v14_csv_without_the_column_still_validates(self, dropped):
        """Excludes the "forgot `_OPTIONAL_COLUMNS`" implementation: the whole
        point of an additive column is that the 295k rows already written
        keep loading."""
        cols = [c for c in ALL_COLUMNS if c != dropped]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        w.writerow([_base_row()[c] for c in cols])
        _header, rows = validate_text(buf.getvalue(), source="old.csv")
        assert rows[0].get(dropped, "") == ""


class TestIsDriverSubstituted:
    """The ONE place a driver substitution is read off a row. Three-valued,
    and the third value is the point."""

    def test_same_driver_is_native(self):
        """Excludes an implementation that reports every row as substituted
        (e.g. one keyed on the mere presence of the columns)."""
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(
            {"kmd_version": "595.84", "umd_version": "595.84"}) is False

    def test_different_driver_is_driver_substituted(self):
        """Excludes an implementation that always reports native — which is
        what no column at all already did."""
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(
            {"kmd_version": "575.57.08", "umd_version": "580.95.05"}) is True

    def test_an_older_user_mode_driver_is_a_substitution_too(self):
        """The function claims SUBSTITUTION, not forward compatibility, and
        this is the row that separates the two claims.

        A stale overlay puts an OLDER user-mode driver ahead of the system
        one. The pair is just as unequal, so the answer is the same `True` —
        but nothing may call it "forward" anything. This asserted the old
        name's bug: `is_forward_compat` returned `True` here and the UI said
        "forward-compat driver", which is false, and in the same breath said
        the configuration was outside NVIDIA's support matrix, which does not
        follow from inequality at all.
        """
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(
            {"kmd_version": "580.95.05", "umd_version": "575.57.08"}) is True

    def test_it_reports_no_direction_and_no_support_verdict(self):
        """The answer is a bool about one question. Excludes any widening of
        the return into a direction or a support claim — which is what a
        consumer would then render, and what no row has the facts for: a row
        carries no `box`, so the specific pairing a support verdict depends
        on is not available to anyone reading this."""
        from backend.inference.benchmarks import is_driver_substituted
        newer = is_driver_substituted(
            {"kmd_version": "575.57.08", "umd_version": "580.95.05"})
        older = is_driver_substituted(
            {"kmd_version": "580.95.05", "umd_version": "575.57.08"})
        assert newer is older is True

    @pytest.mark.parametrize("row", [
        {"kmd_version": "", "umd_version": ""},             # a pre-v14 row
        {"kmd_version": "575.57.08", "umd_version": ""},    # one side observed
        {"kmd_version": "", "umd_version": "580.95.05"},
        {},                                                 # column absent
    ])
    def test_either_side_unknown_is_unknown(self, row):
        """THE assertion of this module, and the one the obvious wrong
        implementation fails.

        Excludes `return (row.get("kmd_version") or "") != (row.get(
        "umd_version") or "")` — a plain string comparison. On a pre-v14 row
        that returns `False`, i.e. it tells ~295,000 rows that measured a
        NATIVE driver, which nobody observed; and on a half-observed row it
        returns `True`, inventing a substitution. Absent must read as
        unknown, so this asserts `is None`, not falsiness.
        """
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(row) is None

    def test_unknown_is_not_merely_falsy(self):
        """`is None` above would also hold for an implementation returning
        `False`, if a reader wrote `== False`. Pin the distinction: native and
        unknown are different answers and a consumer must be able to tell."""
        from backend.inference.benchmarks import is_driver_substituted
        native = is_driver_substituted({"kmd_version": "595.84",
                                    "umd_version": "595.84"})
        unknown = is_driver_substituted({"kmd_version": "", "umd_version": ""})
        assert native is False and unknown is None
        assert native is not unknown

    def test_whitespace_only_is_unknown_not_a_driver(self):
        """A cell holding a space is not an observation. Excludes the
        implementation that compares raw strings, under which `" "` vs `""`
        reads as a substitution."""
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(
            {"kmd_version": " ", "umd_version": ""}) is None

    def test_the_fixture_row_is_native(self):
        """`tests/corpus_row.py` must describe the ORDINARY case, so a test
        that starts from it is not silently asserting the unsupported one."""
        from backend.inference.benchmarks import is_driver_substituted
        assert is_driver_substituted(_base_row()) is False


class TestMeasuredLoadFactorColumn:
    """v7: max/mean expert load of the routing distribution a row was
    measured under. Empty when no trace was carried — a fabricated 1.0 would
    read as "provably balanced", so no default is ever written."""

    def test_empty_means_no_trace(self):
        row = _base_row()
        row["measured_load_factor"] = ""
        validate_text(_emit([row]), source="ok.csv")

    def test_a_real_factor_round_trips(self):
        row = _base_row()
        row["measured_load_factor"] = "2.5"
        _header, rows = validate_text(_emit([row]), source="ok.csv")
        assert rows[0]["measured_load_factor"] == "2.5"

    def test_balanced_is_exactly_one(self):
        row = _base_row()
        row["measured_load_factor"] = "1.0"
        validate_text(_emit([row]), source="ok.csv")

    def test_below_one_is_not_a_load_factor(self):
        # max/mean of any distribution is >= 1; below that is not a load
        # factor, it is a bug in whatever produced the trace.
        row = _base_row()
        row["measured_load_factor"] = "0.5"
        with pytest.raises(BenchmarkValidationError):
            validate_text(_emit([row]), source="bad.csv")

    def test_a_word_is_not_a_load_factor(self):
        row = _base_row()
        row["measured_load_factor"] = "balanced"
        with pytest.raises(BenchmarkValidationError):
            validate_text(_emit([row]), source="bad.csv")

class TestSchemaVersionForwardIncompatible:
    """schema_version newer than the validator is rejected."""

    def test_future_schema_version_is_rejected(self):
        row = _base_row()
        row["schema_version"] = str(KNOWN_SCHEMA_VERSION + 1)
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "forward-incompatible" in str(exc.value)

    def test_known_schema_version_passes(self):
        row = _base_row()
        row["schema_version"] = str(KNOWN_SCHEMA_VERSION)
        text = _emit([row])
        validate_text(text, source="ok.csv")

    def test_old_grouped_gemm_identity_requires_migration(self):
        row = _base_row()
        row.update(
            op="grouped_gemm",
            shape_json=json.dumps({
                "d_out": 128, "d_reduce": 64,
                "n_experts": 4, "s_query": 2,
            }, separators=(",", ":"), sort_keys=True),
            schema_version="5",
        )
        with pytest.raises(BenchmarkValidationError, match="n_tokens"):
            validate_text(_emit([row]), source="old-grouped.csv")


def _row_for_op(op: str) -> dict:
    """A row spelled `op`, carrying the routed token total when the op's
    signature declares one (asked of the catalog, as the validator asks)."""
    from backend.inference.benchmark_identity import declares_routed_token_total
    from backend.inference.op_signatures import op_signature
    row = _base_row(op=op)
    if declares_routed_token_total(op_signature(op)):
        shape = json.loads(row["shape_json"])
        shape["n_tokens"] = 8
        row["shape_json"] = json.dumps(shape, separators=(",", ":"), sort_keys=True)
    return row


class TestOpColumnIsTheCatalogsVocabulary:
    """The `op` column holds only what the kernel catalog binds — the whole of
    what `RowIdentity.op_of` can produce. Any other spelling validates into
    a row no lookup can ever reach."""

    def test_fusion_plan_name_is_refused(self):
        # `grouped_gemm_moe` is a fusion plan's fused_op, not a measurement op;
        # the grouped expert GEMM's one spelling is `grouped_gemm`.
        with pytest.raises(BenchmarkValidationError, match="kernel catalog"):
            validate_text(_emit([_row_for_op("grouped_gemm_moe")]), source="bad.csv")

    def test_every_catalog_op_passes(self):
        from backend.inference.op_signatures import op_bindings
        rows = [_row_for_op(op) | {"node_spec_id": f"n_{op}"} for op in op_bindings()]
        _, parsed = validate_text(_emit(rows), source="ok.csv")
        assert {r["op"] for r in parsed} == set(op_bindings())


def _lake_op_spellings() -> set:
    """Every `op` value stored in the local lake at `BENCH_LAKE_DATA`, read
    straight off its parquet files (read-only; no catalog attach, so the
    serving process's lock is not contended). Deleted rows are included —
    a superset, which only makes the check stricter."""
    data = os.environ.get("BENCH_LAKE_DATA", "")
    if not data or not pathlib.Path(data).is_dir():
        pytest.skip("BENCH_LAKE_DATA does not name a local lake data directory")
    import duckdb
    con = duckdb.connect()
    spellings: set = set()
    for path in sorted(pathlib.Path(data).rglob("*.parquet")):
        if path.name.endswith("-delete.parquet"):
            continue    # a DuckLake deletion vector, not rows
        cols = {r[0] for r in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
        if "op" in cols:
            spellings |= {r[0] for r in con.execute(
                "SELECT DISTINCT op FROM read_parquet(?)", [str(path)]).fetchall()}
    if not spellings:
        pytest.skip(f"no measurement rows under {data}")
    return spellings


@pytest.mark.regression
def test_every_spelling_in_the_lake_passes_the_op_gate():
    """Run with `BENCH_LAKE_DATA=<lake>/data`: a stored spelling the gate
    refuses is a row no lookup reaches."""
    spellings = _lake_op_spellings()
    assert None not in spellings
    rows = [_row_for_op(op) | {"node_spec_id": f"n_{op}"} for op in sorted(spellings)]
    validate_text(_emit(rows), source="lake-spellings.csv")


class TestColumnTypeEnforcement:
    @pytest.mark.parametrize("col,bad_value", [
        ("latency_us", "fast"),
        ("p50_us", "n/a"),
        ("samples", "many"),
        ("count_per_step", "1.5"),     # floats not allowed for int col
        ("achieved_tflops", "high"),
        ("peak_tflops_at_test", "nan_ish"),
    ])
    def test_bad_numeric_value_is_rejected(self, col, bad_value):
        row = _base_row()
        row[col] = bad_value
        text = _emit([row])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert col in str(exc.value)


class TestPhaseAndParallelVocabulary:
    def test_unknown_phase_is_rejected(self):
        row = _base_row()
        row["phase"] = "warmup"
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "phase" in str(exc.value)

    def test_malformed_parallel_is_rejected(self):
        row = _base_row()
        row["parallel"] = "tp four"
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "parallel" in str(exc.value)

    def test_parallel_keys_must_be_sorted(self):
        row = _base_row()
        row["parallel"] = "tp=4,ep=2"  # ep should come before tp
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "lexicographic" in str(exc.value)


class TestDateFormat:
    def test_non_iso_date_is_rejected(self):
        row = _base_row()
        row["date"] = "May 16, 2026"
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "date" in str(exc.value)

    def test_impossible_calendar_date_is_rejected(self):
        """Regex-pass / calendar-fail: 2026-99-99 looks like YYYY-MM-DD but
        is not a real date. `datetime.date.fromisoformat` catches it."""
        row = _base_row()
        row["date"] = "2026-99-99"
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert "calendar date" in str(exc.value)


class TestNonFiniteFloatsRejected:
    """`float("inf") > 0` is True, so the `positive` guard alone lets
    infinities slip through. `math.isfinite` closes the gap."""

    @pytest.mark.parametrize("col,bad_value", [
        ("latency_us", "inf"),
        ("p95_us", "-inf"),
        ("achieved_tflops", "nan"),
        ("peak_bw_at_test", "Infinity"),
    ])
    def test_non_finite_value_is_rejected(self, col, bad_value):
        row = _base_row()
        row[col] = bad_value
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(_emit([row]), source="bad.csv")
        assert col in str(exc.value)


class TestDuplicateIdentityRejected:
    """Same 12-column identity tuple cannot appear twice."""

    def test_duplicate_identity_is_rejected(self):
        text = _emit([_base_row(), _base_row()])
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text(text, source="bad.csv")
        assert "duplicate" in str(exc.value)

    def test_same_node_spec_id_different_shape_is_kept_distinct(self):
        r1 = _base_row()
        r2 = _base_row()
        r2["shape_json"] = json.dumps(
            {"B": 1, "s_ctx": 4096, "n_heads": 24, "d_head": 128}, separators=(",", ":")
        )
        _, rows = validate_text(_emit([r1, r2]), source="ok.csv")
        assert len(rows) == 2


class TestEmptyCSVRejected:
    def test_empty_csv_is_rejected(self):
        with pytest.raises(BenchmarkValidationError) as exc:
            validate_text("", source="empty.csv")
        assert "empty" in str(exc.value)


class TestRoundTripWithJsonFields:
    """JSON columns contain commas; csv.writer must quote them; the
    validator's round-trip must reproduce the exact quoted form.
    """

    def test_json_fields_round_trip_byte_identical(self, tmp_path):
        text = _emit([_base_row()])
        path = tmp_path / "json.csv"
        path.write_text(text)
        assert_round_trip(str(path))


class TestDocumentedParallelExamplePasses:
    """The `parallel` example in
    `backend/inference/BENCHMARKS_CSV.md` must actually validate —
    the key-sort rule rejects forms like `tp=4,ep=2,pp=1,dp=1`."""

    def test_documented_sorted_example_passes(self):
        from backend.inference.benchmarks import _check_parallel
        assert _check_parallel("dp=1,ep=2,pp=1,tp=4") == "dp=1,ep=2,pp=1,tp=4"

    def test_single_key_documented_example_passes(self):
        from backend.inference.benchmarks import _check_parallel
        assert _check_parallel("tp=1") == "tp=1"


class TestSuggestShapesCsvAlignment:
    """`emit_suggest_shapes` must use `csv.writer` rather than a
    manual `,`.join — `shape_json` contains commas, which would
    break column alignment under naive joining. Lock in: every
    emitted row parses cleanly with exactly `len(ALL_COLUMNS)`
    fields."""

    def _patch_pipeline(self, monkeypatch, per_op_timing_rows):
        """Stub the pipeline call chain so the test doesn't need a real
        model + config + spec on disk."""

        class _StubProgram:
            # S5a(3/3): RowIdentity keys nodes by aggregation id off the
            # program. These stub rows carry none — identity is read off
            # `shape_in` — so an empty program reproduces the pre-S5 path
            # (every row's node resolves to None, kind comes from the shape).
            components: ClassVar[dict] = {}

            def nodes(self):
                return ()

        class _StubAnalysis:
            # A bf16 model: `WorkloadShape.of_walk` asks whether it stores a
            # weight at all. A row's key reads only the row's own dtypes.
            class resolved_precision:
                model_dtype = "bf16"

            # A model with no Mamba-2 scan chunk in its config, and no
            # Sinkhorn round count either: both are serving choices a key
            # reads, and a config that states neither serves neither.
            class program:
                @staticmethod
                def scan_chunk_size(_fields):
                    return None

                @staticmethod
                def sinkhorn_iters(_fields):
                    return None

            def dominant_weight_dtype(self):
                return "bf16"

            def dominant_cache_dtype(self):
                return "bf16"

            def dominant_weight_scale_dtype(self):
                return "fp32"

        class _StubAnalysisResult:
            def __init__(self, rows):
                self.per_op_timing = rows
                self.program = _StubProgram()
                self.analysis = _StubAnalysis()
                self.identities = {}

        class _StubSpec:
            name = "stub"
            yaml_path = "/dev/null"

        class _StubResolved:
            spec = _StubSpec()
            dim_values: ClassVar[dict] = {"d_model": 4096, "n_heads": 32}
            config: ClassVar[dict] = {"model_type": "stub"}

            def identity_hash(self) -> str:
                return "stub-spec-hash"

        class _StubLoader:
            def match(self, _config, model_id=""):
                return _StubResolved()

        # The three the template fetches, resolves and walks with, patched
        # where it reads them (its own module's names, imported at its top).
        import backend.cli_bench as cli_bench
        monkeypatch.setattr(cli_bench, "fetch_config",
                            lambda mid, overlay_dir=None: {"model_type": "stub"})
        monkeypatch.setattr(cli_bench, "SpecLoader", _StubLoader)
        monkeypatch.setattr(
            cli_bench, "analyze",
            lambda *_a, **_kw: _StubAnalysisResult(per_op_timing_rows),
        )

    #: Rows shaped the way `compute_per_op_timing` shapes them — the
    #: identity is read off `shape_in`, so a row without one is not a
    #: row any kernel was launched for.
    #: Whether a row costs nothing, which `point_keys` reads off the row
    #: (`row_roofline.law_columns`): these rows cost something.

    #: A row's launches are its count (`count_per_step`), and the constants
    #: its key reads beyond its shapes are its own node's (`node_dims`).
    SHAPED_ROWS: ClassVar[list[dict[str, Any]]] = [
        {
            "spec_id": "self_attn", "op": "flash_attention", "count_per_step": 32,
            "shape_in": [[8, 32, 4096, 128], [8, 8, 4096, 128], [8, 8, 4096, 128]],
            "shape_out": [[8, 32, 4096, 128]],
            "operand_ports": [["q"], ["k"], ["v"]],
            "node_dims": {"n_heads": 32, "n_kv_heads": 8, "d_head": 128},
            "act_dtype": "BF16",
            "zero_cost": False,
        },
        {
            "spec_id": "mlp.gate_proj", "op": "linear", "count_per_step": 32,
            "shape_in": [[8, 4096, 4096], [11008, 4096]],
            "shape_out": [[8, 4096, 11008]],
            "node_dims": {"d_model": 4096, "d_ffn": 11008},
            "act_dtype": "BF16", "weight_dtype": "BF16",
            "zero_cost": False,
        },
        {
            "spec_id": "input_layernorm", "op": "rmsnorm", "count_per_step": 32,
            "shape_in": [[8, 4096, 4096]],
            "shape_out": [[8, 4096, 4096]],
            "node_dims": {"d_model": 4096},
            "act_dtype": "BF16", "weight_dtype": "BF16",
            "zero_cost": False,
        },
    ]

    def test_emit_suggest_shapes_csv_parses_with_correct_column_count(self, monkeypatch):
        from backend.cli_bench import emit_suggest_shapes

        per_op_timing = self.SHAPED_ROWS
        self._patch_pipeline(monkeypatch, per_op_timing)
        text = emit_suggest_shapes(
            model_id="stub/stub", batch_size=1, seq_len=2048,
        )
        header, rows = parse_rows(text)
        assert header == list(ALL_COLUMNS), (
            "header must match ALL_COLUMNS in order"
        )
        assert len(rows) == len(per_op_timing)
        for i, row in enumerate(rows):
            assert set(row.keys()) == set(ALL_COLUMNS), (
                f"row {i} key set {set(row.keys())} does not match "
                f"ALL_COLUMNS {set(ALL_COLUMNS)}"
            )

    def test_emit_suggest_shapes_shape_json_round_trips_as_json(self, monkeypatch):
        from backend.cli_bench import emit_suggest_shapes

        per_op_timing = self.SHAPED_ROWS[:1]
        self._patch_pipeline(monkeypatch, per_op_timing)
        text = emit_suggest_shapes(
            model_id="stub/stub", batch_size=1, seq_len=2048,
        )
        _, rows = parse_rows(text)
        # The shape_json cell must decode as a JSON object, not split
        # across multiple CSV fields — it is full of commas. A paged
        # attention row carries the deployment's page (the framework
        # binding's default, vLLM's 16).
        decoded = json.loads(rows[0]["shape_json"])
        assert decoded == {
            "B": 8, "n_heads": 32, "n_kv_heads": 8,
            "s_query": 4096, "s_kv": 4096, "d_head": 128, "kv_block_size": 16,
        }

    def test_emit_suggest_shapes_omits_rows_it_cannot_identify(self, monkeypatch):
        """A row with no `shape_in` is left out, not templated.

        The template used to fill such a row with the model-wide
        `{B, s_ctx, …dims}`, which reads as an invitation: a contributor
        measuring it in good faith contributes a kernel the model never
        launches, under a key another model's estimate can hit. `bench
        validate` naming the gap is the fix; filling it is not."""
        from backend.cli_bench import emit_suggest_shapes

        self._patch_pipeline(monkeypatch, [
            {"spec_id": "self_attn", "op": "flash_attention", "count_per_step": 32,
             "operand_ports": [["q"], ["k"], ["v"]],
             "node_dims": {"n_heads": 32, "n_kv_heads": 8, "d_head": 128},
             "act_dtype": "BF16",
             "zero_cost": False},
            {"spec_id": "self_attn.q", "op": "transpose", "count_per_step": 32,
             "shape_in": [[8, 32, 4096, 128]], "node_dims": {"d_head": 128}, "act_dtype": "BF16",
             "zero_cost": False},
            self.SHAPED_ROWS[1],
        ])
        text = emit_suggest_shapes(model_id="stub/stub", batch_size=1, seq_len=2048)
        _, rows = parse_rows(text)
        assert [r["op"] for r in rows] == ["linear"], (
            "the shapeless flash_attention row and the unbenchable "
            "transpose must both be omitted"
        )

    def test_emit_suggest_shapes_uses_default_framework_binding(self):
        """`emit_suggest_shapes` must invoke
        `pipeline.analyze(framework_binding=get_binding('vllm'))`
        so the framework rewrite runs and per_op_timing carries FUSED op
        names (`flash_attention` at `self_attn.attn_apply` anchor sid)
        matching what production `/inference/estimate` looks up at
        request time. Without the binding, the CLI emits naive op
        names like `attention_score`/`attention_apply` and contributor
        CSV rows miss the route's default-mode lookup."""
        from backend.cli_bench import emit_suggest_shapes

        snap = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "backend", "arch", "snapshots",
            "meta-llama__Llama-3.2-3B.json",
        )
        snapshot_config(pathlib.Path(snap).name)

        text = emit_suggest_shapes(
            model_id="meta-llama/Llama-3.2-3B",
            batch_size=1, seq_len=2048,
        )
        _, rows = parse_rows(text)
        ops = [r["op"] for r in rows]
        # Post-fix the FlashAttention scheme must show up as a single
        # fused row; the naive attention chain must NOT appear.
        assert "flash_attention" in ops, (
            f"emit_suggest_shapes must emit the fused `flash_attention` "
            f"op name (the default vllm rewrite ran via its "
            f"binding); got ops: {sorted(set(ops))!r}"
        )
        for naive in ("attention_score", "attention_apply"):
            assert naive not in ops, (
                f"emit_suggest_shapes must NOT emit the pre-fusion "
                f"naive op {naive!r} when the default binding fuses "
                f"the attention chain; got ops: {sorted(set(ops))!r}."
            )
        # The fused row must carry the anchor sibling's spec_id
        # (`self_attn.attn_apply`); Option A appends a `__<component>`
        # scope suffix to compute-only primitives, so the leaf segment
        # is e.g. `attn_apply__gqa_attention`. Strip the scope when
        # comparing the anchor sibling name.
        flash_rows = [r for r in rows if r["op"] == "flash_attention"]
        def _leaf_no_scope(sid: str) -> str:
            return sid.rsplit(".", 1)[-1].split("__", 1)[0]
        assert any(_leaf_no_scope(r["node_spec_id"]) == "attn_apply"
                   for r in flash_rows), (
            f"fused FlashAttention row's node_spec_id must end with "
            f"`.attn_apply` (the anchor sibling that the vllm binding binds); "
            f"got node_spec_ids: "
            f"{[r['node_spec_id'] for r in flash_rows]!r}"
        )

    def test_emit_suggest_shapes_ignores_the_rows_own_shape_json(
        self, monkeypatch,
    ):
        """The identity is derived from `shape_in`, never copied out of
        `row["shape_json"]`.

        Every per_op_timing row carries a `shape_json` of its own, and
        without a `calibration_context` it holds the phases.py defaults
        (`{"B": 1, "s_ctx": 1}`). Copied verbatim it would key the
        contributor's measurement at batch 1, sequence 1 — a kernel
        nobody ran, spelled in the model-wide vocabulary no measured row
        uses. The CLI must key off the shape the op was actually
        walked with.
        """
        from backend.cli_bench import emit_suggest_shapes

        pipeline_default_shape = json.dumps(
            {"B": 1, "s_ctx": 1, "d_model": 4096, "n_heads": 32}, sort_keys=True,
        )
        per_op_timing = [
            dict(
                self.SHAPED_ROWS[0],
                shape_json=pipeline_default_shape,
                dtype_env_json='{"weight":"bf16"}',
            ),
        ]
        self._patch_pipeline(monkeypatch, per_op_timing)

        text = emit_suggest_shapes(
            model_id="stub/stub", batch_size=8, seq_len=4096,
        )
        _, rows = parse_rows(text)
        decoded = json.loads(rows[0]["shape_json"])
        assert "s_ctx" not in decoded, (
            f"the model-wide spelling must not reach the CSV; got {decoded!r} "
            f"(pre-fix this was row['shape_json']={pipeline_default_shape!r})"
        )
        assert (decoded.get("s_query"), decoded.get("d_head")) == (4096, 128), (
            f"emitted key must come from the row's shape_in, not its "
            f"shape_json. got: {decoded!r}"
        )


class TestRowCheck:
    """`validate_text`'s per-row gate over a stream, answering instead of
    raising: what an owner's calibration corpus keeps and reports."""

    def test_a_failing_row_is_answered_and_the_rest_pass(self):
        from backend.inference.benchmarks import RowCheck, parse_rows
        bad = _corpus.row(latency_us="not-a-number")
        good = _corpus.row()
        _h, rows = parse_rows(_corpus.to_csv([bad, good, good]))
        check = RowCheck()
        answers = [check.reason(r, i) for i, r in enumerate(rows, start=2)]
        assert "latency_us" in answers[0]
        assert answers[1] is None
        assert answers[2].startswith("duplicate identity tuple (first seen at row 3)")

    def test_the_revision_names_the_vocabularies_read(self, monkeypatch):
        import backend.inference.benchmarks as B
        rev = B.validator_revision()
        assert len(rev) == 16 and B.validator_revision() == rev
        B.validator_revision.cache_clear()
        monkeypatch.setattr(B, "REGISTERED_SYMBOLS", B.REGISTERED_SYMBOLS | {"a_symbol_added"})
        try:
            assert B.validator_revision() != rev
        finally:
            B.validator_revision.cache_clear()


class TestTimingMethod:
    """`timing_method` (v16): which clock timed the row. Optional on ingest —
    every row and file written before it validates and reads empty, the
    pre-drain clock — and a closed vocabulary when present."""

    def test_a_row_without_it_validates(self):
        from backend.inference.benchmarks import validate_row
        validate_row(_base_row(timing_method=""))

    def test_a_header_without_it_validates(self):
        from backend.inference.benchmarks import validate_header, OPTIONAL_COLUMNS
        assert "timing_method" in OPTIONAL_COLUMNS
        validate_header([c for c in ALL_COLUMNS if c != "timing_method"])

    def test_each_clock_of_the_vocabulary_validates(self):
        from backend.inference.benchmarks import TIMING_METHODS, validate_row
        for method in TIMING_METHODS:
            validate_row(_base_row(timing_method=method))

    def test_a_clock_outside_the_vocabulary_is_refused(self):
        from backend.inference.benchmarks import validate_row
        with pytest.raises(BenchmarkValidationError, match="timing_method"):
            validate_row(_base_row(timing_method="l2_flush"))

    def test_a_row_that_is_not_a_measurement_names_no_clock(self):
        from backend.inference.benchmarks import MEASUREMENT_COLUMNS, validate_row
        oom = _base_row(outcome="oom", timing_method="l2_drain",
                        cross_check="unknown",
                        **{c: "" for c in (*MEASUREMENT_COLUMNS,
                                           "cross_check_rel_err", "cross_check_cosine",
                                           "xcheck_typical", "xcheck_worst")})
        validate_row({**oom, "timing_method": ""})   # the same row, clockless, is fine
        with pytest.raises(BenchmarkValidationError, match="timing_method"):
            validate_row(oom)
