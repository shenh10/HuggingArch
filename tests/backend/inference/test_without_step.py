"""A card that admits no request publishes no step (`capacity.without_step`):
no batch, no per-op rows, no module table, no encode step, no fusion delta."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.inference.capacity import withheld_phase, without_step
from backend.inference.phases import WITHHELD_TIMES

ROWS = ("per_op_timing", "per_op_timing_decode", "per_op_timing_draft")


def _result(reason):
    phase = {"batch": 1, "t_sol_s": None, "flops_by_module": [{"uid": 1}],
             "time_unpublished_reason": reason or ""}
    return {
        "infeasibility_reason": reason, "max_batch_per_gpu": 0 if reason else 4,
        "phases": {"prefill": {"phase": "prefill", **phase},
                   "decode": {"phase": "decode", **phase}},
        "per_op_timing": [{"uid": 1}], "per_op_timing_decode": [{"uid": 1}],
        "vs_naive_delta": {"prefill": {}, "decode": {}},
        "encode": {"batch": 1, "stacks": [{"stack": "vision", "per_op_timing": [{"uid": 2}],
                                           "flops_by_module": [{"uid": 2}]}]},
        "weight_breakdown": {"total": 1},
    }


def test_a_card_that_admits_no_request_publishes_no_step():
    out = without_step(_result("weights do not fit"), ROWS)
    assert out["phases"] == {
        name: withheld_phase(name, "weights do not fit")
        for name in ("prefill", "decode")}
    for key in (*ROWS, "vs_naive_delta", "encode"):
        assert key not in out, key
    # What is not a step stays: the card's verdict and its weights.
    assert out["max_batch_per_gpu"] == 0 and out["weight_breakdown"] == {"total": 1}


def test_the_no_step_phase_states_no_time_under_any_name():
    """The pin on the shape itself, read off the OTHER declaration.

    The assertion above compares the published phase with the producer that
    wrote it, so a key added to `withheld_phase` would appear on both sides
    and stay green. The field it must never grow is a time, and which fields
    are times is stated in `phases.WITHHELD_TIMES`: a step with no batch
    publishes none of them under any name — zero is a duration, and every
    consumer divides tokens by it and calls the quotient a rate.
    """
    phase = without_step(_result("weights do not fit"), ROWS)["phases"]["prefill"]
    assert phase["batch"] is None
    assert not set(phase) & set(WITHHELD_TIMES), sorted(set(phase) & set(WITHHELD_TIMES))


def test_a_card_that_serves_is_published_whole():
    result = _result(None)
    assert without_step(result, ROWS) is result
