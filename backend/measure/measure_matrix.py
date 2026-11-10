"""Record-level folds over a matrix of measured terms."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from backend.measure.auto_sweep import (
    AMBIGUOUS,
    DECLINED,
    MEASURED,
    MISSING,
    MeasuredFormula,
    MeasuredTrack,
    PointMeasure,
    STATES,
    UNPLACED,
    UNPLACED_REASONS,
    blend,
    key_label,
)
from backend.measure.lookup import Key


_STATE_CODES = {MEASURED: 0, MISSING: 1, AMBIGUOUS: 2, UNPLACED: 3, DECLINED: 4}
_CODE_STATES = tuple(_STATE_CODES)


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _frozen(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _cached_questions(matrix: "MeasureMatrix") -> tuple[Any, ...]:
    """A matrix's selection questions, frozen once for repeated concats."""
    if not hasattr(matrix, "_questionKeys"):
        matrix._questionKeys = tuple(
            (key, _frozen(context), label, _frozen(read))
            for key, context, label, read in zip(
                matrix.keys, matrix.contexts, matrix.labels, matrix.inputs,
                strict=True))
    return matrix._questionKeys


@dataclass(frozen=True)
class _RowGroup:
    labels: tuple[str, ...]
    reasons: tuple[str, ...]
    unkeyed: tuple[Optional[str], ...]
    aggregation_ids: tuple[str, ...]


@dataclass
class MeasureMatrix:
    """A measured formula over points × terms, with one key per column.

    `slots` is a sparse key matrix: row `p`, term `i` names descriptor
    `slots[p, i]`, or -1 when that term is zero-cost/unkeyed. `launches` is
    the matching L matrix. Descriptors hold everything selection asks, so a
    fold selects each physical question once before evaluating the formula
    over whole rows.
    """

    formulas: tuple[str, ...]
    zero: np.ndarray
    launches: np.ndarray
    slots: np.ndarray
    keys: tuple[Optional[Key], ...]
    contexts: tuple[Any, ...]
    labels: tuple[str, ...]
    inputs: tuple[Any, ...]
    row_groups: tuple[_RowGroup, ...]
    group_of: np.ndarray
    point_rows: dict[tuple[int, int, int], int]
    #: Each term's SOL leaf (`PointMeasure.sol`), NaN where its row states
    #: none — every term of a record's row: a pool point's SOL is a function
    #: of the floors its reader picks (`FloorForm.terms_at`). None: no row
    #: states any.
    sol: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "MeasureMatrix":
        return cls(
            (), np.empty((0, 0), dtype=bool), np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=np.int32), (), (), (), (), (),
            np.empty(0, dtype=np.int32), {})

    @classmethod
    def of_measures(cls, measures: Sequence[PointMeasure]) -> "MeasureMatrix":
        width = max((len(measure.formula.zero_cost) for measure in measures),
                    default=0)
        rows = len(measures)
        zero = np.zeros((rows, width), dtype=bool)
        launches = np.zeros((rows, width), dtype=float)
        slots = np.full((rows, width), -1, dtype=np.int32)
        keys: list[Optional[Key]] = []
        contexts: list[Any] = []
        labels: list[str] = []
        inputs: list[Any] = []
        descriptors: dict[Any, int] = {}
        groups: dict[_RowGroup, int] = {}
        row_groups: list[_RowGroup] = []
        group_of = np.empty(rows, dtype=np.int32)

        sol = np.full((rows, width), np.nan, dtype=float)
        for row, measure in enumerate(measures):
            count = len(measure.formula.zero_cost)
            zero[row, :count] = measure.formula.zero_cost
            if measure.sol:
                sol[row, :count] = [np.nan if v is None else float(v)
                                    for v in measure.sol]
            launches[row, :count] = measure.launches
            group = _RowGroup(
                tuple(measure.labels), tuple(measure.reasons),
                tuple(measure.unkeyed), tuple(measure.aggregation_ids))
            if group not in groups:
                groups[group] = len(row_groups)
                row_groups.append(group)
            group_of[row] = groups[group]
            for term in range(count):
                if zero[row, term] or measure.keys[term] is None:
                    continue
                context = (measure.contexts[term]
                           if measure.contexts else measure.context)
                question = (
                    measure.keys[term], _frozen(context), measure.labels[term],
                    _frozen(measure.inputs[term]
                            if measure.inputs else None))
                if question not in descriptors:
                    descriptors[question] = len(keys)
                    keys.append(measure.keys[term])
                    contexts.append(context)
                    labels.append(measure.labels[term])
                    inputs.append(measure.inputs[term]
                                  if measure.inputs else None)
                slots[row, term] = descriptors[question]

        return cls(
            tuple(measure.formula.text for measure in measures), zero,
            launches, slots, tuple(keys), tuple(contexts), tuple(labels),
            tuple(inputs), tuple(row_groups), group_of, {},
            sol if np.any(~np.isnan(sol)) else None)

    @classmethod
    def of_record(cls, table: Mapping[str, Any], *, card: Any,
                  wanted: Sequence[tuple[int, int, int]]) -> "MeasureMatrix":
        """One canonical record's compact measure table as a matrix."""
        cardless = bool(table.get("cardless"))
        strings = table["strings"]
        keyColumns = table["keyColumns"]
        physical: list[Key] = []
        anchors: list[Any] = []
        dispatches: list[Any] = []
        for entry in range(len(keyColumns[0]) if keyColumns else 0):
            fields = [column[entry] for column in keyColumns]
            shape_row = table["shapes"][fields[1]]
            template = table["shapeTemplates"][shape_row[0]]
            shape = {
                table["shapeAxes"][axis]: value
                for axis, value in zip(template, shape_row[1:], strict=True)
            }
            shape_json = json.dumps(shape, sort_keys=True,
                                    separators=(",", ":"))
            gpu = card.sku if cardless else strings[fields[3]]
            physical.append(Key(strings[fields[0]], shape_json,
                                strings[fields[2]], gpu, strings[fields[4]]))
            anchors.append(table["anchors"][fields[5]])
            dispatches.append(table["dispatches"][fields[6]])

        contexts = [dict(value) for value in table["contexts"]]
        if cardless:
            for context in contexts:
                context["gpu"] = card.sku
        variants = table["variants"]
        width = max((len(variant["zero_cost"]) for variant in variants),
                    default=0)

        # A physical key is one selection question under a context and label.
        # Splitting on the latter two is structural: the same signature can
        # name two ops, and a selector is allowed to read the op and anchor.
        keys: list[Optional[Key]] = []
        selection_contexts: list[Any] = []
        labels: list[str] = []
        inputs: list[Any] = []

        rows_for_points: dict[tuple[int, int, int], int] = {}
        selected: list[Mapping[str, Any]] = []
        for context, batch, draft_tokens in wanted:
            for point in table["points"]:
                if (point.get("gpu", card.sku) == card.sku
                        and int(point["context"]) == int(context)
                        and int(point["batch"]) == int(batch)
                        and int(point.get("draftTokens", 0)) == int(draft_tokens)):
                    rows_for_points[
                        (int(context), int(batch), int(draft_tokens))
                    ] = len(selected)
                    selected.append(point)
                    break
            else:
                raise KeyError(
                    f"the serving law has no measured point for {card.sku} "
                    f"at {context} tokens, batch {batch}, K={draft_tokens}")

        # Fill this context's key-entry × op-label slot table once. The same
        # entry under another op label is another selection question; the same
        # question at another point names the same descriptor.
        descriptors: dict[tuple[int, int, int], int] = {}
        slot_by_context = [
            np.full((len(physical), len(strings)), -1, dtype=np.int32)
            for _context in contexts
        ]
        for point in selected:
            context_index = int(point["contextIndex"])
            variant = variants[int(point["variant"])]
            for term, entry in zip(point["keyedTerms"], point["keyedKeys"],
                                   strict=True):
                label_index = int(variant["labels"][term])
                question = (context_index, int(entry), label_index)
                if question not in descriptors:
                    descriptors[question] = len(keys)
                    keys.append(physical[int(entry)])
                    selection_contexts.append(contexts[context_index])
                    labels.append(strings[label_index])
                    inputs.append({
                        "anchor": anchors[int(entry)],
                        "dispatch": dispatches[int(entry)],
                    })
                slot_by_context[context_index][
                    int(entry), label_index] = descriptors[question]

        rows = len(selected)
        zero = np.zeros((rows, width), dtype=bool)
        launches = np.zeros((rows, width), dtype=float)
        slots = np.full((rows, width), -1, dtype=np.int32)
        groups: dict[_RowGroup, int] = {}
        row_groups: list[_RowGroup] = []
        group_of = np.empty(rows, dtype=np.int32)
        for row, point in enumerate(selected):
            variant = variants[int(point["variant"])]
            count = len(variant["zero_cost"])
            zero[row, :count] = variant["zero_cost"]
            launches[row, :count] = variant["launches"]
            group = _RowGroup(
                tuple(strings[i] for i in variant["labels"]),
                tuple(strings[i] for i in variant["reasons"]),
                tuple(strings[i] if i is not None else None
                      for i in variant["unkeyed"]),
                ())
            if group not in groups:
                groups[group] = len(row_groups)
                row_groups.append(group)
            group_of[row] = groups[group]
            terms = np.asarray(point["keyedTerms"], dtype=np.int32)
            entries = np.asarray(point["keyedKeys"], dtype=np.int32)
            term_labels = np.asarray(variant["labels"], dtype=np.int32)[terms]
            slots[row, terms] = slot_by_context[int(point["contextIndex"])][
                entries, term_labels]

        return cls(
            tuple(strings[variants[int(point["variant"])]["formula"]]
                  for point in selected),
            zero, launches, slots, tuple(keys),
            tuple(selection_contexts), tuple(labels), tuple(inputs),
            tuple(row_groups), group_of, rows_for_points)

    def concat(self, other: "MeasureMatrix") -> "MeasureMatrix":
        if not len(self.formulas):
            return other
        if not len(other.formulas):
            return self
        width = max(self.zero.shape[1], other.zero.shape[1])
        left_questions = _cached_questions(self)
        right_questions = _cached_questions(other)

        stated = self.sol is not None or other.sol is not None

        def padded(matrix: MeasureMatrix) -> MeasureMatrix:
            sol = (matrix.sol if matrix.sol is not None
                   else np.full(matrix.zero.shape, np.nan))
            return matrix.__class__(
                matrix.formulas,
                np.pad(matrix.zero, ((0, 0), (0, width - matrix.zero.shape[1]))),
                np.pad(matrix.launches,
                       ((0, 0), (0, width - matrix.launches.shape[1]))),
                np.pad(matrix.slots,
                       ((0, 0), (0, width - matrix.slots.shape[1]))),
                matrix.keys, matrix.contexts, matrix.labels, matrix.inputs,
                matrix.row_groups, matrix.group_of, matrix.point_rows,
                np.pad(sol, ((0, 0), (0, width - sol.shape[1])),
                       constant_values=np.nan))

        left, right = padded(self), padded(other)
        keys = list(left.keys)
        contexts = list(left.contexts)
        labels = list(left.labels)
        inputs = list(left.inputs)
        questions = {question: i for i, question in enumerate(left_questions)}
        remap = np.empty(len(right.keys), dtype=np.int32)
        for old, (question, key, context, label, read) in enumerate(zip(
                right_questions, right.keys, right.contexts, right.labels,
                right.inputs, strict=True)):
            if question not in questions:
                questions[question] = len(keys)
                keys.append(key)
                contexts.append(context)
                labels.append(label)
                inputs.append(read)
            remap[old] = questions[question]
        right_slots = np.where(right.slots >= 0,
                               remap[np.where(
                                   right.slots >= 0, right.slots, 0)],
                               -1).astype(np.int32)
        group_offset = len(left.row_groups)
        result = self.__class__(
            (*left.formulas, *right.formulas),
            np.vstack((left.zero, right.zero)),
            np.vstack((left.launches, right.launches)),
            np.vstack((left.slots, right_slots)),
            tuple(keys), tuple(contexts), tuple(labels), tuple(inputs),
            (*left.row_groups, *other.row_groups),
            np.concatenate((left.group_of, right.group_of + group_offset)),
            {**left.point_rows, **other.point_rows},
            np.vstack((left.sol, right.sol)) if stated else None)
        result._questionKeys = tuple(questions)
        return result

    def fold(self, select: Any) -> list["MatrixFold"]:
        """Select one question per descriptor, then fold all rows together."""
        times: list[Optional[float]] = []
        states: list[str] = []
        details: list[Any] = []
        unchecked: list[bool] = []
        for key, context, label, read in zip(
                self.keys, self.contexts, self.labels, self.inputs,
                strict=True):
            time, state, detail = select(key, context, label, read)
            if state not in (MEASURED, MISSING, AMBIGUOUS) or (
                    (time is None) == (state == MEASURED)):
                raise ValueError(
                    f"selection of {key_label(key)}: {time!r} with state "
                    f"{state}")
            times.append(time)
            states.append(state)
            details.append(detail)
            unchecked.append(bool(
                state == MEASURED and isinstance(detail, dict)
                and ((detail.get("source_badge") or {}).get("cross_check")
                     or "passed") != "passed"))

        selected_times = np.array(
            [0.0 if value is None else float(value) for value in times],
            dtype=float)
        selected_states = np.array(
            [_CODE_STATES.index(value) for value in states], dtype=np.int8)
        selected_unchecked = np.array(unchecked, dtype=bool)
        selected_details = tuple(details)
        keyed = self.slots >= 0
        safe_slots = np.where(keyed, self.slots, 0)
        state_matrix = np.where(
            keyed, selected_states[safe_slots],
            np.where(self.zero, 0, 3)).astype(np.int8)
        # Unkeyed rows retain their own state; -1 terms are padding.
        for row, group_index in enumerate(self.group_of):
            group = self.row_groups[group_index]
            for term, state in enumerate(group.unkeyed):
                if (term < state_matrix.shape[1]
                        and not self.zero[row, term]
                        and not keyed[row, term]):
                    state_matrix[row, term] = (
                        0 if state is None else _STATE_CODES.get(state, 3))
            for term in range(len(group.unkeyed), state_matrix.shape[1]):
                state_matrix[row, term] = -1
        launch_times = np.where(keyed, selected_times[safe_slots], 0.0)
        totals = np.zeros(len(self.formulas), dtype=float)
        complete = np.all((state_matrix == 0) | (state_matrix == -1), axis=1)
        for text in dict.fromkeys(self.formulas):
            rows = [i for i, value in enumerate(self.formulas)
                    if value == text and complete[i]]
            if not rows:
                continue
            count = max((len(self.row_groups[int(self.group_of[i])].labels)
                         for i in rows), default=0)
            formula = MeasuredFormula(text, tuple(False for _ in range(count)))
            totals[rows] = formula.at_many(
                self.launches[rows, :count], launch_times[rows, :count])
        totals = np.where(complete, totals, np.nan)

        tracks = self._tracks(
            state_matrix, selected_unchecked, keyed, totals)
        return [
            MatrixFold(
                self.formulas[row], self.zero[row], self.launches[row],
                self.slots[row], self.keys, self.row_groups[
                    int(self.group_of[row])], state_matrix[row],
                launch_times[row], float(totals[row]), tracks[row],
                selected_details,
                None if self.sol is None else self.sol[row])
            for row in range(len(self.formulas))
        ]

    def _tracks(self, states: np.ndarray, unchecked_by_slot: np.ndarray,
                keyed: np.ndarray, totals: np.ndarray) -> list[MeasuredTrack]:
        key_labels = np.array(
            ["" if key is None else key_label(key) for key in self.keys],
            dtype=object)
        patterns: dict[tuple[int, bytes], MeasuredTrack] = {}
        tracks: list[MeasuredTrack] = []
        for row in range(states.shape[0]):
            pattern = (int(self.group_of[row]), states[row].tobytes(),
                       float(totals[row]))
            if pattern not in patterns:
                patterns[pattern] = self._track(
                    row, states[row], unchecked_by_slot, keyed,
                    key_labels, totals[row])
            tracks.append(patterns[pattern])
        return tracks

    def _track(self, row: int, states: np.ndarray,
               unchecked_by_slot: np.ndarray, keyed: np.ndarray,
               key_labels: np.ndarray, total: float) -> MeasuredTrack:
        group = self.row_groups[int(self.group_of[row])]
        counts = np.bincount(states[states >= 0],
                             minlength=len(_CODE_STATES))
        row_keyed = keyed[row]
        safe = np.where(row_keyed, self.slots[row], 0)
        unchecked = int(np.count_nonzero(
            row_keyed & (states == 0) & unchecked_by_slot[safe]))
        missing = list(dict.fromkeys(
            key_labels[safe][(states == 1) & row_keyed]))
        ambiguous = list(dict.fromkeys(
            key_labels[safe][(states == 2) & row_keyed]))
        unplaced = list(dict.fromkeys(
            label for label, state in zip(group.labels, group.unkeyed,
                                          strict=True)
            if state == UNPLACED))
        declined = list(dict.fromkeys(
            (group.labels[i], group.reasons[i])
            for i in np.flatnonzero(states == 4)))
        details = [
            {"op": label, "aggregation_id": aggregation, "reason": reason}
            for label, aggregation, reason in zip(
                group.labels, group.aggregation_ids, group.reasons,
                strict=False)
            if group.aggregation_ids and reason in UNPLACED_REASONS
        ]
        total = (None if np.any((states != 0) & (states != -1))
                 else float(total))
        return MeasuredTrack(
            int(counts[0]), len(group.labels),
            tuple(missing), tuple(ambiguous), tuple(unplaced),
            tuple(declined), unchecked_rows=unchecked,
            unplaced_details=tuple(details),
            rows_by_state=tuple(
                (state, int(counts[_STATE_CODES[state]]))
                for state in STATES if state != MEASURED and counts[
                    _STATE_CODES[state]]),
            total_s=total,
        )



@dataclass
class MatrixFold:
    """One row of a record-level fold, with the same readings as `Folded`."""

    formula_text: str
    zero: np.ndarray
    launches: np.ndarray
    slots: np.ndarray
    keys: tuple[Optional[Key], ...]
    group: _RowGroup
    states: np.ndarray
    launch_times: np.ndarray
    total: float
    track: MeasuredTrack
    details: tuple[Any, ...]
    #: The row's SOL leaves (`MeasureMatrix.sol`), or None.
    sol: Optional[np.ndarray] = None

    @property
    def rows(self) -> tuple[tuple[str, Optional[float], str, Any], ...]:
        out = []
        keyed = self.slots >= 0
        for term in range(len(self.group.labels)):
            state = _CODE_STATES[int(self.states[term])]
            if state == MEASURED:
                time = float(self.launch_times[term] * self.launches[term]) \
                    if not self.zero[term] else 0.0
                if keyed[term]:
                    label = key_label(self.keys[int(self.slots[term])])
                    detail = self.details[int(self.slots[term])]
                else:
                    label = self.group.labels[term]
                    detail = self.group.reasons[term]
            elif keyed[term]:
                time = None
                label = key_label(self.keys[int(self.slots[term])])
                detail = None
            else:
                time = None
                label = self.group.labels[term]
                detail = self.group.reasons[term]
            out.append((state, time, label, detail))
        return tuple(out)

    @property
    def times(self) -> tuple[Optional[float], ...]:
        return tuple(
            None if self.states[term] != 0 else
            (0.0 if self.zero[term] else float(self.launch_times[term]))
            for term in range(len(self.group.labels)))

    def subtotal(self, terms: Iterable[int]) -> Optional[float]:
        terms = list(terms)
        if not terms or any(self.states[i] != 0 for i in terms):
            return None
        return MeasuredFormula(
            self.formula_text, tuple(bool(v) for v in self.zero)).at(
            tuple(float(v) for v in self.launches), self.times, terms)

    @property
    def measured_terms(self) -> tuple[int, ...]:
        """The terms with a measured time (a zero-cost term among them)."""
        return tuple(int(k) for k in np.flatnonzero(self.states == 0))

    @property
    def measured_s(self) -> float:
        """The formula's share of the measured terms: what the partial
        track puts in place of their SOL (`auto_sweep.blend`)."""
        return MeasuredFormula(
            self.formula_text, tuple(bool(v) for v in self.zero)).at(
            tuple(float(v) for v in self.launches),
            tuple(0.0 if v is None else v for v in self.times),
            self.measured_terms)

    def partial(self, t_sol: Optional[float]) -> Optional[dict[str, float]]:
        """`Folded.partial` on the matrix row: the partial track
        (`auto_sweep.blend`) off the row's own SOL leaves, weighted by the
        formula as the SOL total weights them. None when the step has no
        SOL total or the row states no SOL — a record's row, whose SOL its
        reader holds as a function of the floors (`FloorForm.terms_at`)."""
        if t_sol is None or self.sol is None or np.all(np.isnan(self.sol)):
            return None
        live = [k for k in self.measured_terms if not self.zero[k]]
        unpriced = [k for k in live if np.isnan(self.sol[k])]
        if unpriced:
            raise ValueError(
                f"measured terms {unpriced[:5]} have no SOL leaf: a measured "
                "term is a priced row of this walk")
        formula = MeasuredFormula(
            self.formula_text, tuple(bool(v) for v in self.zero))
        ones = [1.0] * len(self.states)
        leaves = [0.0 if np.isnan(v) else float(v) for v in self.sol]
        return blend(t_sol, formula.at(ones, leaves, live), self.measured_s)
