"""The record-level measured fold keeps one formula and one coverage owner."""
from __future__ import annotations

import numpy as np

import pytest

from backend.measure.measure_matrix import MeasureMatrix
from backend.inference.sol_law import _measurePointsToJson
from backend.inference.card import Card
from backend.measure.auto_sweep import (
    MEASURED, MISSING, MeasuredFormula, PointMeasure,
)
from backend.measure.lookup import Key


def _measure(formula: str, launches: tuple[float, ...],
             signatures: tuple[str, ...] = ("op",)) -> PointMeasure:
    count = len(launches)
    keys = tuple(
        Key(signatures[i % len(signatures)], '{"x":1}', "{}",
            "B200", "decode") for i in range(count))
    return PointMeasure(
        keys, launches,
        tuple(None for _ in launches), tuple("op" for _ in launches),
        tuple("" for _ in launches),
        MeasuredFormula(formula, tuple(False for _ in launches)),
        {"framework": "vllm", "gpu": "B200"},
        tuple({"anchor": None, "dispatch": None} for _ in launches),
        aggregation_ids=tuple("op" for _ in launches))


def test_the_same_formula_law_reads_a_matrix_of_points():
    """`at_many` is the same compiled declaration as `at`, including `Max`."""
    measures = [
        _measure("T_0*n_0", (2.0,), ("left",)),
        _measure("Max(T_0*n_0, T_1*n_1)", (2.0, 3.0),
                 ("left", "right")),
    ]
    matrix = MeasureMatrix.of_measures(measures)
    times = {measures[1].keys[0]: 0.002, measures[1].keys[1]: 0.003}

    def select(key, _context, _op, _read):
        return times.get(key, 0.001), MEASURED, None

    folds = matrix.fold(select)
    assert [fold.total for fold in folds] == pytest.approx([0.004, 0.009])
    assert [fold.track.total_rows for fold in folds] == [1, 2]
    assert [fold.subtotal((0,)) for fold in folds] == [0.004, 0.004]


def test_padding_between_records_is_not_a_measured_row():
    measures = [_measure("T_0*n_0", (2.0,)),
                _measure("T_0*n_0 + T_1*n_1", (2.0, 3.0))]
    matrix = MeasureMatrix.of_measures(measures)
    states = np.array([[0, -1], [1, 0]], dtype=np.int8)
    tracks = matrix._tracks(states, np.array([False, False]),
                            matrix.slots >= 0,
                            np.array([0.0, 0.0]))
    assert tracks[0].total_rows == 1
    assert tracks[1].total_rows == 2
    assert tracks[1].coverage()["states"]["missing"]


def test_one_key_selected_once_answers_every_point_that_names_it():
    measures = [_measure("T_0*n_0", (2.0,)),
                _measure("T_0*n_0", (4.0,))]
    matrix = MeasureMatrix.of_measures(measures)
    asked = 0

    def select(_key, _context, _op, _read):
        nonlocal asked
        asked += 1
        return None, MISSING, None

    folds = matrix.fold(select)
    assert asked == 1
    assert all(fold.track.total_s is None for fold in folds)


def test_record_k_views_share_keys_but_keep_their_formula_variants():
    """Recursive draft launches repeat K times; the launch is selected once."""
    measures = [_measure("T_0*n_0", (2.0,)),
                _measure("3*T_0*n_0", (2.0,))]
    table = _measurePointsToJson([
        ("B200", 1024, 1, 0, measures[0]),
        ("B200", 1024, 1, 1, measures[1]),
    ])
    assert table["schema"] == "measure-points-v7"
    assert [(p["draftTokens"], p["variant"]) for p in table["points"]] == [
        (0, 0), (1, 1)]
    assert len(table["keyColumns"][0]) == 1

    matrix = MeasureMatrix.of_record(
        table, card=Card.of("B200"), wanted=[(1024, 1, 0), (1024, 1, 1)])
    assert matrix.formulas == ("T_0*n_0", "3*T_0*n_0")
    assert len(matrix.keys) == 1
