"""A step's rows added once (`row_roofline.StepSum`), with the leaf swappable,
and its laws as text that survive a process boundary and the disk.

Falsification targets:
  - with the leaf the rows' `t_op` law, the step's law at a batch and card is
    the SOL phase total a direct estimate there publishes — every summed
    column, per bound too, pp stages and a draft's forwards included;
  - with the leaf `n_k x T_k`, the measured total is the same weighted sum
    over the same terms, and a zero-cost row is 0 and names no symbol;
  - a law read back from its text (`read_row_law`) is the law, in this
    process or another one, and a name outside the vocabulary is refused.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.transforms import Workload
from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from backend.inference.comm_cost import Fabric
from backend.inference.parallel_config import ParallelConfig
from backend.inference.row_roofline import (
    FloorForm, LawBook, StepSum, build_row_exprs, card_binding, comm_factor,
    evaluate_law, law_columns, read_row_law, RowCard)
from tests.sweep_parity import REL, Deployment

CTX = 4096
B = Workload.symbols()["batch"]
#: The workload the synthetic rows stand at, but for the batch.
_POINT = {"s_input": CTX, "s_output": 0, "s_prefix": 0, "s_verify": 1, "rho": 1}
_FABRIC = Fabric(8, 1.0e11, 1.0e10)


def _row(uid, *, flops=200, act=4, weight=100, payload=0.0, world=1,
         dtype="bf16", peak=1.0e15, count=1):
    """A synthetic per-op row: its law and the facts the step reads."""
    is_comm = world > 1
    return {
        "uid": uid, "act_dtype": dtype, "is_comm": is_comm,
        "comm_op": "all_reduce" if is_comm else "", "comm_world": world,
        "count_per_step": count, "comm_base_stride": 1,
        "peak_flops_per_sec_for_op": peak,
        **law_columns(build_row_exprs(
            count=count, flops=B * flops, act_in=B * act, act_out=0,
            demand=8 * B * weight,
            weight_total=weight, metadata_total=0, comm_payload=B * payload,
            comm_factor=comm_factor("all_reduce", world) if is_comm else 0.0,
            comm_priced=is_comm), LawBook()),
    }


def _at(text: str, batch: int, rows) -> float:
    return evaluate_law(text, {"B": batch, **card_binding(
        rows, peak_flops=None, peak_bw=1.0e12, fabric=_FABRIC)})


def _t_op(row, batch) -> float:
    return LawBook().row_at(row, {**_POINT, "B": batch}, RowCard.of(
        row, peak_flops=None, peak_bw=1.0e12, fabric=_FABRIC)).times.t_op


class TestTheSum:
    def test_rows_of_two_dtypes_and_a_collective_add(self):
        rows = [_row(0), _row(1, dtype="fp8_e4m3", peak=2.0e15, flops=10**9),
                _row(2, flops=0, act=0, weight=0, payload=1000.0, world=4)]
        law = sp.sstr(StepSum.of(rows).law())
        assert {s.name for s in read_row_law(law).free_symbols} == {
            "B", "PEAK_BW", "PEAK_FLOPS_bf16", "PEAK_FLOPS_fp8_e4m3",
            "COMM_S_PER_BYTE_all_reduce_4_1_1", "COMM_LATENCY_S_all_reduce_4_1_1",
            "LAUNCH_S"}
        for batch in (1, 3, 64):
            assert _at(law, batch, rows) == pytest.approx(
                sum(_t_op(r, batch) for r in rows), rel=REL)

    def test_a_draft_runs_its_forwards_and_pp_keeps_the_rank_account(self):
        rows, draft = [_row(0), _row(1)], [_row(7, flops=10**8)]
        step = StepSum.of(rows, draft_rows=draft, draft_forwards=3)
        assert _at(sp.sstr(step.law()), 5, rows + draft) == pytest.approx(
            _t_op(rows[0], 5) + _t_op(rows[1], 5) + 3 * _t_op(draft[0], 5),
            rel=REL)
        pp = StepSum.of(stages=[[_row(0)], [_row(1, flops=10**7)]], rank_stage=1)
        assert pp.law("flops", account=True) == read_row_law(
            _row(1, flops=10**7)["roofline_expr"]["flops"])
        assert len(pp.law().args) == 2

    def test_measured_total_and_zero_cost_rows(self):
        view = _row(3, flops=0, act=0, weight=0)
        rows = [_row(0), view, _row(1, count=4)]
        step = StepSum.of(rows, draft_rows=[_row(9)], draft_forwards=2)
        text, names = step.measured_law()
        # The view row costs 0 on both tracks and names no symbol.
        assert sp.sympify(StepSum.of([view]).law()) == 0
        assert {v["uid"] for v in names.values()} == {0, 1, 9}
        assert names["T_3"] == {"uid": 9, "stage": None, "draft": True,
                                "symbol": "launch_time"}
        values = {name: 1.0 if v["symbol"] == "launches" else 2.0
                  for name, v in names.items()}
        # Σ w x n x T: three step rows at 1 x 2.0 (the view is 0), the draft's
        # one row twice.
        assert evaluate_law(text, values) == pytest.approx(2.0 + 2.0 + 2 * 2.0)


class TestTheTextIsTheLaw:
    def test_round_trip_in_process_on_disk_and_in_another_process(self, tmp_path):
        rows = [_row(0), _row(1, dtype="fp8_e4m3", peak=2.0e15, flops=10**9),
                _row(2, flops=0, act=0, weight=0, payload=1000.0, world=4)]
        step = StepSum.of(rows, draft_rows=[_row(5)], draft_forwards=2)
        laws = step.laws()
        laws["measured"], names = step.measured_law()
        for text in laws.values():
            # Reading evaluates (a coefficient distributes over a sum), so the
            # text settles after one read and is a fixed point from there.
            once = sp.sstr(read_row_law(text))
            assert sp.sstr(read_row_law(once)) == once
        values = {"B": 7, **card_binding(rows, peak_flops=None, peak_bw=1.0e12,
                                         fabric=_FABRIC),
                  **{name: 1.5 for name in names}}
        here = {key: float(evaluate_law(text, values)) for key, text in laws.items()}
        path = tmp_path / "laws.json"
        path.write_text(json.dumps({"laws": laws, "values": values}))
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "from backend.inference.row_roofline import evaluate_law\n"
            f"d = json.loads(open({str(path)!r}).read())\n"
            "print(json.dumps({k: float(evaluate_law(t, d['values'])) "
            "for k, t in d['laws'].items()}))\n")
        there = json.loads(subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True,
            text=True).stdout)
        assert there == here
        assert here["t_sol_s"] == pytest.approx(
            sum(here[f"t_sol_{b}_s"] for b in ("compute", "memory", "comm")), rel=REL)

    def test_the_vocabulary_is_closed(self):
        for text in ("B*s_ctx", "PEAK_FLOPS_bf16*E", "T_x*n_1"):
            with pytest.raises(ValueError, match="may not read"):
                read_row_law(text)
        for text in ("__import__('os')", "exp(B)", "Symbol('B')"):
            with pytest.raises(ValueError, match="does not have"):
                read_row_law(text)


def _on_card(rows, card: str):
    """`rows` priced on `card`: each row's own peak re-read for its dtype,
    and the card's bandwidth and fabric (`capacity._stamp_gpu_on_result`)."""
    from backend.inference.gpu_specs import get_gpu, peak_hbm_bw
    from backend.inference.pipeline import leaf_peak
    gpu = get_gpu(card)
    priced = [{**r, "peak_flops_per_sec_for_op":
               float(leaf_peak(gpu, r["act_dtype"]) or 0.0) if r.get("act_dtype")
               else 0.0} for r in rows]
    return (priced, peak_hbm_bw(gpu),
            Fabric.of(gpu, COMM_LATENCY_US_DEFAULT * 1e-6),
            LAUNCH_FLOOR_US_DEFAULT * 1e-6)


def _step(estimate: dict, phase: str, card: str) -> tuple:
    """The phase's step from the estimate's rows (every stage's row laws under
    pp), on `card`, and the card's binding of its laws."""
    rows = estimate["per_op_timing" if phase == "prefill" else "per_op_timing_decode"]
    pipeline = estimate["phases"][phase]["pipeline_time"]
    stages = pipeline["stage_laws"] if pipeline else None
    every = [r for s in stages for r in s] if stages else rows
    priced, bw, fabric, launch_s = _on_card(every, card)
    if stages:
        it = iter(priced)
        stages = [[next(it) for _ in s] for s in stages]
        step = StepSum.of(stages=stages, rank_stage=pipeline["stage"])
    else:
        step = StepSum.of(priced)
    return step, card_binding(priced, peak_flops=None, peak_bw=bw,
                              fabric=fabric, launch_s=launch_s)


_CASES = [
    pytest.param(Deployment("meta-llama/Llama-3.2-3B"), id="llama-tp1"),
    pytest.param(Deployment("Qwen/Qwen3-30B-A3B",
                            parallel=ParallelConfig(dp_attention=8, ep_moe=8)),
                 id="qwen3-moe-dp8ep8", marks=pytest.mark.regression),
    pytest.param(Deployment("meta-llama/Llama-3.2-3B",
                            parallel=ParallelConfig(pp=2)),
                 id="llama-pp2", marks=pytest.mark.regression),
]


@pytest.mark.parametrize("dep", _CASES)
def test_the_sol_law_is_the_phase_total(dep):
    """Walk once at B=1; the step's laws (`StepSum.laws`) at B and the card's
    binding are the totals a direct estimate on that card at B publishes —
    FLOPs, bytes, writes, each time and each bound's share of it."""
    anchor = dep.direct(avg_input_tokens=CTX, prefill_batch=1, decode_batch=1)
    for card in ("H100-SXM", "B200"):
        for batch in (1, 16):
            direct = dep.on(card).direct(avg_input_tokens=CTX,
                                         prefill_batch=batch, decode_batch=batch)
            for phase in ("prefill", "decode"):
                step, binding = _step(anchor, phase, card)
                published = direct["phases"][phase]
                for key, text in step.laws().items():
                    want = (published["t_sol_by_bound"][key[6:-2]]
                            if key.startswith("t_sol_") and key != "t_sol_s"
                            else published[key])
                    got = evaluate_law(text, {**_point(step), "B": batch,
                                              **binding})
                    assert float(got) == pytest.approx(want, rel=REL, abs=0), (
                        card, batch, phase, key)


def test_the_measured_law_is_the_same_sum():
    """With `n_k = 1` and `T_k` each row's own time at the point, the
    measured law is the SOL total: the same terms, the same weights."""
    dep = Deployment("meta-llama/Llama-3.2-3B")
    anchor = dep.direct(avg_input_tokens=CTX, prefill_batch=1, decode_batch=1)
    step, binding = _step(anchor, "decode", "H100-SXM")
    text, names = step.measured_law()
    values = {}
    for name, where in names.items():
        k = int(name.split("_")[1])
        term = step.terms[k]
        values[name] = (1 if where["symbol"] == "launches" else
                        _t_op_at(term.row, {**_point(step), "B": 16},
                                 binding))
    sol = evaluate_law(sp.sstr(step.law()),
                       {**_point(step), "B": 16, **binding})
    assert evaluate_law(text, values) == pytest.approx(sol, rel=REL)


def _point(step) -> dict:
    """The workload the step's rows were walked at: a full row's
    `point_law`, a pipeline stage's law row's `point`."""
    row = step.terms[0].row
    return dict(row["point_law"]["point"] if "point_law" in row
                else row["point"])


def _t_op_at(row, point, binding) -> float:
    from backend.inference.row_roofline import lifted_law
    return float(evaluate_law(sp.sstr(lifted_law(row, "t_op")),
                              {**point, **binding}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


class TestTheFloorFormIsTheLaw:
    """A step's floor form (`FloorForm`), read off one estimate's rows, is
    the step at any other link latency and launch floor: what an estimate
    priced at those floors publishes."""

    def test_another_pair_of_floors_is_an_estimate_at_them(self):
        from backend.inference.capacity import estimate_capacity
        from backend.inference.gpu_specs import get_gpu
        from backend.inference.row_roofline import FloorForm, LawBook
        from tests.snapshots import resolve_snapshot, snapshot_config
        model = "Qwen/Qwen3-30B-A3B"
        common = dict(config=snapshot_config(model), model_id=model,
                      resolved=resolve_snapshot(model),
                      gpu_sku="H100-SXM",
                      # all-reduces inside the attention TP group, all-to-alls
                      # across all four ranks.
                      parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                              dp_attention=2, ep_moe=4),
                      avg_input_tokens=1024, avg_output_tokens=256,
                      prefill_batch_size=1, decode_batch_size=2)
        anchor = estimate_capacity(**common)
        form = FloorForm.of(anchor["per_op_timing_decode"], book=LawBook(),
                            peak_flops=anchor["peak_flops_per_sec"],
                            peak_bw=anchor["peak_hbm_bw_bytes_per_sec"],
                            fabric=Fabric.of(get_gpu("H100-SXM")))
        assert any(c > 0 for _a, _b, c, _n in form.terms)
        assert any(n > 0 for _a, _b, _c, n in form.terms)
        for comm_us, launch_us in ((COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT),
                                   (5.0, 0.5), (0.0, 0.0), (20.0, 4.0)):
            there = estimate_capacity(comm_latency_us=comm_us,
                                      launch_floor_us=launch_us, **common)
            want = there["phases"]["decode"]["t_sol_s"]
            assert form.at(comm_us * 1e-6, launch_us * 1e-6) == pytest.approx(
                want, rel=1e-9, abs=0), (comm_us, launch_us)

    def test_a_counted_draft_is_its_forwards_in_the_form(self):
        """Speculative decoding with the MTP draft counted: the step is the
        verify's rows once and the draft's one-forward rows once per
        forward. The form weighted that way is the step at any other floors
        too — the gallery frontier reads every K's step off it."""
        from backend.inference.capacity import estimate_capacity
        from backend.inference.gpu_specs import get_gpu
        from backend.inference.phases import Speculation
        from backend.inference.row_roofline import FloorForm, LawBook
        from tests.snapshots import resolve_snapshot, snapshot_config
        model = "XiaomiMiMo/MiMo-7B-RL"  # a small model with an MTP stack
        common = dict(config=snapshot_config(model), model_id=model,
                      resolved=resolve_snapshot(model), gpu_sku="H100-SXM",
                      parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2),
                      avg_input_tokens=1024, avg_output_tokens=256,
                      prefill_batch_size=1, decode_batch_size=4,
                      s_verify=Speculation(3).s_verify, acceptance_rate=0.8,
                      speculative_draft="recursive")
        anchor = estimate_capacity(**common)
        draft = anchor["phases"]["decode"]["speculative_draft"]
        assert draft["forwards"] >= 1 and anchor["per_op_timing_draft"]
        form = FloorForm.of(anchor["per_op_timing_decode"], book=LawBook(),
                            peak_flops=anchor["peak_flops_per_sec"],
                            peak_bw=anchor["peak_hbm_bw_bytes_per_sec"],
                            fabric=Fabric.of(get_gpu("H100-SXM")),
                            draft_rows=anchor["per_op_timing_draft"],
                            draft_forwards=draft["forwards"])
        plain = FloorForm.of(anchor["per_op_timing_decode"], book=LawBook(),
                             peak_flops=anchor["peak_flops_per_sec"],
                             peak_bw=anchor["peak_hbm_bw_bytes_per_sec"],
                             fabric=Fabric.of(get_gpu("H100-SXM")))
        for comm_us, launch_us in ((COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT),
                                   (5.0, 0.5), (0.0, 0.0), (20.0, 4.0)):
            there = estimate_capacity(comm_latency_us=comm_us,
                                      launch_floor_us=launch_us, **common)
            want = there["phases"]["decode"]["t_sol_s"]
            got = form.at(comm_us * 1e-6, launch_us * 1e-6)
            assert got == pytest.approx(want, rel=1e-9, abs=0), (comm_us, launch_us)
            # The draft's forwards are in it: without them the step is short.
            assert plain.at(comm_us * 1e-6, launch_us * 1e-6) < got


_FLOORS_US = ((COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT),
              (5.0, 0.5), (0.0, 0.0), (20.0, 4.0))


def _estimates(model, gpu_sku, parallel, **knobs):
    """`{(comm_us, launch_us): estimate}` of one deployment on `gpu_sku`
    at every pair of `_FLOORS_US`."""
    from backend.inference.capacity import estimate_capacity
    from tests.snapshots import resolve_snapshot, snapshot_config
    common = dict(config=snapshot_config(model), model_id=model,
                  resolved=resolve_snapshot(model), gpu_sku=gpu_sku,
                  parallel=parallel, avg_input_tokens=1024,
                  avg_output_tokens=256, prefill_batch_size=1, **knobs)
    return {floors: estimate_capacity(comm_latency_us=floors[0],
                                      launch_floor_us=floors[1], **common)
            for floors in _FLOORS_US}


def _card_form(est):
    from backend.inference.gpu_specs import get_gpu
    from backend.inference.row_roofline import CardForm
    draft = est["phases"]["decode"].get("speculative_draft") or {}
    return CardForm.of(est["per_op_timing_decode"], book=LawBook(),
                       fabric=Fabric.of(get_gpu(est["gpu"]["sku"])),
                       draft_rows=est.get("per_op_timing_draft") or (),
                       draft_forwards=int(draft.get("forwards") or 0))


def _assert_form_is_the_estimates(form, gpu_sku, estimates):
    from backend.inference.capacity import floor_form_on
    from backend.inference.gpu_specs import get_gpu
    bound = floor_form_on(form, get_gpu(gpu_sku))
    for (comm_us, launch_us), est in estimates.items():
        want = est["phases"]["decode"]["t_sol_s"]
        assert bound.at(comm_us * 1e-6, launch_us * 1e-6) == pytest.approx(
            want, rel=1e-9, abs=0), (gpu_sku, comm_us, launch_us)


class TestTheCardFormIsTheLaw:
    """A step's card form (`CardForm`) is its rows' laws with the card left
    free: bound to a card (`capacity.floor_form_on`) it is the step an
    estimate on that card publishes, at any floors — the walk's own card,
    and another card of its class (`framework_bindings.card_class`)."""

    def test_a_counted_draft_weighs_its_collectives_hops_too(self):
        """The MTP draft's forwards weigh every term of a draft row — the
        collective's link traversals too, the range of its payload chosen
        on ONE forward's bytes, as the row's law chooses it. A form that
        weighs the payload and not the traversals is short by the draft's
        hops at any non-zero link latency."""
        from backend.inference.phases import Speculation
        estimates = _estimates(
            "XiaomiMiMo/MiMo-7B-RL", "H100-SXM",
            ParallelConfig(tp_attention=2, tp_dense_ffn=2),
            decode_batch_size=4, s_verify=Speculation(3).s_verify,
            acceptance_rate=0.8, speculative_draft="recursive")
        anchor = next(iter(estimates.values()))
        draft = anchor["phases"]["decode"]["speculative_draft"]
        assert draft["forwards"] > 1 and anchor["per_op_timing_draft"]
        form = _card_form(anchor)
        # The draft's collectives are in the form, at its forwards.
        assert any(w == draft["forwards"] and si >= 0
                   for *_r, si, _p, w in form.terms)
        _assert_form_is_the_estimates(form, "H100-SXM", estimates)

    def test_on_its_walks_card_it_is_the_estimate_and_survives_json(self):
        from backend.inference.row_roofline import CardForm
        parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                  dp_attention=2, ep_moe=4)
        estimates = _estimates("Qwen/Qwen3-30B-A3B", "H100-SXM", parallel,
                               decode_batch_size=2)
        form = _card_form(next(iter(estimates.values())))
        assert any(si >= 0 for *_r, si, _p, _w in form.terms)
        _assert_form_is_the_estimates(form, "H100-SXM", estimates)
        again = CardForm.from_json(json.loads(json.dumps(form.to_json())))
        assert again == form

    def test_on_another_card_of_its_class_it_is_that_cards_estimate(self):
        """The rows walked on H100 bound to H200 — another HBM bandwidth,
        the collectives' ranges re-decided on its fabric — are what an
        estimate walked on H200 publishes."""
        from backend.inference.framework_bindings import card_class
        from backend.inference.gpu_specs import get_gpu
        assert card_class(get_gpu("H100-SXM")) == card_class(get_gpu("H200"))
        parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                  dp_attention=2, ep_moe=4)
        walked = _estimates("Qwen/Qwen3-30B-A3B", "H100-SXM", parallel,
                            decode_batch_size=2)
        there = _estimates("Qwen/Qwen3-30B-A3B", "H200", parallel,
                           decode_batch_size=2)
        _assert_form_is_the_estimates(_card_form(next(iter(walked.values()))),
                                      "H200", there)

    def test_a_dtype_the_card_has_no_peak_for_is_refused_not_free(self):
        from backend.inference.gpu_specs import NoPeakForDtype
        from backend.inference.row_roofline import CardForm
        form = CardForm(((1.0e12, 1.0e9, 0, 1.0, -1, 0.0, 1),), ("FP4_E2M1",))
        with pytest.raises(NoPeakForDtype):
            form.bind({"FP4_E2M1": None}, peak_bw=1.0e12, fabric=_FABRIC)

    def test_a_card_with_no_bandwidth_is_refused_not_infinitely_fast(self):
        """A row that reads bytes needs the card's HBM bandwidth. Pricing
        `bytes / bw` at zero seconds when `bw` is missing says the card
        moves bytes instantly — so the memory side refuses with the card
        and the column named, as the compute side does. A row with no bytes
        needs no bandwidth and still prices."""
        from backend.inference.gpu_specs import NoPeakBandwidth
        from backend.inference.row_roofline import CardForm
        form = CardForm(((0.0, 2.0e9, 0, 0.0, -1, 0.0, 1),), ("BF16",))
        for missing in (None, 0.0, float("nan"), float("inf")):
            with pytest.raises(NoPeakBandwidth, match="mem_bw_gbps"):
                form.bind({"BF16": 1.0e15}, peak_bw=missing, fabric=_FABRIC,
                          gpu_sku="MadeUp-1")
            with pytest.raises(NoPeakBandwidth, match="MadeUp-1"):
                form.bind({"BF16": 1.0e15}, peak_bw=missing, fabric=_FABRIC,
                          gpu_sku="MadeUp-1")
        priced = form.bind({"BF16": 1.0e15}, peak_bw=1.0e12, fabric=_FABRIC,
                           gpu_sku="MadeUp-1")
        assert priced.at(0.0, 0.0) == pytest.approx(2.0e9 / 1.0e12)
        # No bytes, no bandwidth needed: the FLOPs still price.
        flops_only = CardForm(((1.0e12, 0.0, 0, 0.0, -1, 0.0, 1),), ("BF16",))
        assert flops_only.bind({"BF16": 1.0e15}, peak_bw=None,
                               fabric=_FABRIC).at(0.0, 0.0) > 0


class TestTheDispatchIsCasesOverThePayload:
    """A collective whose framework declares rungs is priced in its law as
    cases over its payload (`row_roofline.build_row_exprs`): the step's law
    read at another batch lands in that batch's algorithm, as the row read
    there does; at one point the time is linear in the link latency."""

    def _rows(self):
        from backend.inference.framework_bindings import get_binding
        from types import SimpleNamespace
        rungs = [r.as_dict() for r in get_binding("vllm").collectives_for(
            SimpleNamespace(architectures=()), None)["all_reduce"]]
        row = _row(0, flops=0, act=0, weight=0, payload=14336.0, world=8,
                   count=3)
        law = law_columns(build_row_exprs(
            count=3, flops=0, act_in=0, act_out=0, demand=0, weight_total=0,
            metadata_total=0, comm_payload=B * 14336,
            comm_factor=comm_factor("all_reduce", 8), comm_priced=True,
            comm_rungs=len(rungs)), LawBook())
        return [{**row, **law, "comm_rungs": rungs}]

    def _fabric(self, alpha=COMM_LATENCY_US_DEFAULT * 1e-6):
        from backend.inference.gpu_specs import get_gpu
        return Fabric.of(get_gpu("H100-SXM"), alpha)

    def test_the_step_law_switches_algorithm_with_the_batch(self):
        rows = self._rows()
        fabric = self._fabric()
        law = sp.sstr(StepSum.of(rows).law())
        binding = card_binding(rows, peak_flops=None, peak_bw=1.0e12,
                               fabric=fabric)
        a = fabric.step_latency
        for batch, want in (
                # B=1: one-shot — one traversal, 7 payloads in.
                (1, 3 * (a + 7 * 14336 / 450e9)),
                # B=16: two-shot.
                (16, 3 * (2 * a + 1.75 * 16 * 14336 / 450e9)),
                # B=1024: NVLS — half the ring's bytes.
                (1024, 3 * (2 * a + 0.875 * 1024 * 14336 / 450e9))):
            got = float(evaluate_law(law, {**_POINT, "B": batch, **binding}))
            row_t = LawBook().row_at(rows[0], {**_POINT, "B": batch}, RowCard.of(
                rows[0], peak_flops=None, peak_bw=1.0e12, fabric=fabric)
                ).times.t_comm
            assert got == pytest.approx(want, rel=1e-12), batch
            assert row_t == pytest.approx(want, rel=1e-12), batch

    def test_at_one_point_the_time_is_linear_in_the_link_latency(self):
        rows = self._rows()
        for batch in (1, 16, 1024):
            t = [LawBook().row_at(rows[0], {**_POINT, "B": batch}, RowCard.of(
                rows[0], peak_flops=None, peak_bw=1.0e12,
                fabric=self._fabric(alpha))).times.t_comm
                for alpha in (0.0, 1e-6, 7e-6)]
            slope = (t[1] - t[0]) / 1e-6
            assert t[2] == pytest.approx(t[0] + slope * 7e-6, rel=1e-12)
            # The slope is the instances times the algorithm's hops.
            assert slope == pytest.approx(3 * (1 if batch == 1 else 2), rel=1e-9)


def test_a_floor_form_collapses_proportional_terms_exactly():
    """Terms equal up to a positive weight are one term (`w·max(a, b+cα, nL)
    = max(wa, wb+wcα, wnL)`); terms that are not exactly proportional stay
    apart; the form is the same function at any floors, and its columnar
    JSON reads back as itself."""
    from backend.inference.row_roofline import FloorForm

    terms = [(0.25, 0.0, 0.0, 1.0), (0.75, 0.0, 0.0, 3.0),
             (0.25, 0.0, 0.0, 1.0000000000000002), (1e-6, 3e-6, 4.0, 0.0),
             (0.0, 0.0, 0.0, 0.0)]
    raw = FloorForm(0.5, tuple(terms))
    form = FloorForm.collapsed(0.5, terms)
    assert len(form.terms) == 3
    for floors in ((0.0, 0.0), (2.3e-6, 3.7e-6), (1e-5, 1e-7), (0.0, 0.5)):
        assert form.at(*floors) == pytest.approx(raw.at(*floors), rel=1e-15)
    assert FloorForm.from_json(json.loads(json.dumps(form.to_json()))) == form
    assert set(form.to_json()) == {"constant", "a", "b", "c", "n"}


def test_a_termwise_form_prices_only_the_step_terms_it_is_asked():
    """`FloorForm.terms_at`: the SOL of the named step terms alone at the
    floors given; an index past the step's width, or a collapsed form
    (whose terms are rays, not step terms), refuses."""
    form = FloorForm.termwise([(0, (3.0, 0.0, 0.0, 0.0)),
                               (2, (0.0, 1.0, 2.0, 0.0)),
                               (5, (0.0, 0.0, 0.0, 0.0))], 6)
    assert form.index == (0, 2)  # an all-zero term is left out
    assert form.at(1.0, 1.0) == pytest.approx(3.0 + 3.0)
    assert form.terms_at(1.0, 1.0, [0]) == pytest.approx(3.0)
    assert form.terms_at(4.0, 1.0, [2, 5]) == pytest.approx(9.0)
    assert form.terms_at(1.0, 1.0, []) == 0
    assert FloorForm.from_json(form.to_json()) == form
    with pytest.raises(ValueError):
        form.terms_at(1.0, 1.0, [6])
    with pytest.raises(ValueError):
        FloorForm.collapsed(0.0, [(1.0, 0.0, 0.0, 0.0)]).terms_at(0, 0, [0])
