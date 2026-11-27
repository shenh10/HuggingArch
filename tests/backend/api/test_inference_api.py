"""Inference API contracts: disaggregation, request validation and rank views.

Calibration matching itself is tested at its typed overlay boundary in
``test_calibration_overlay_contracts.py``.
"""
from __future__ import annotations

from typing import ClassVar
from unittest import mock

import pytest

from tests.snapshots import resolve_snapshot, snapshot_config

import tests.corpus_row as _corpus

from backend.api._models import InferenceEstimateRequest


def _make_req(**overrides) -> InferenceEstimateRequest:
    return InferenceEstimateRequest(
        model_id="meta-llama/Llama-3.2-3B",
        gpu_sku="H100-SXM",
        avg_input_tokens=2048,
        avg_output_tokens=128,
        **overrides,
    )


def _run_route(req: InferenceEstimateRequest):
    """The `/inference/estimate` answer, as the route computes it (in a
    compute worker, before the worker renders it to JSON)."""
    from backend.api.routes import inference as _route
    return _route._estimate_payload(req)


class TestGpusEndpoint:
    def test_audit_parallel_is_the_declared_reference_deployment(self):
        """The pickers seed their default parallelism from this field —
        it is the one declaration in `sweep_grid.AUDIT_PARALLEL`, and a
        rename there must fail here rather than drift the pages."""
        from fastapi.testclient import TestClient

        from backend.api import app
        from backend.inference.sweep_grid import AUDIT_PARALLEL

        body = TestClient(app).get("/inference/gpus").json()
        assert body.get("audit_parallel") == AUDIT_PARALLEL


class TestParallelShapeEndpoint:
    """The GPU count of a deployment and what makes it invalid are
    `ParallelConfig`'s answers; the page that edits one asks."""

    def test_cluster_and_validation_come_from_the_config(self):
        from fastapi.testclient import TestClient

        from backend.api import app
        from backend.inference.parallel_config import ParallelConfig

        client = TestClient(app)
        par = {"tp_attention": 2, "ep_moe": 8, "dp_attention": 4, "pp": 2}
        body = client.post("/inference/parallel-shape",
                           json={"parallel": par}).json()
        cfg = ParallelConfig.from_dict(par)
        assert body == {"cluster_size": cfg.cluster_size,
                        "per_stage_size": cfg.per_stage_size,
                        "errors": cfg.validate()}
        assert body["errors"], "dp_attention × tp_attention ≠ ep_moe is invalid"
        bad = client.post("/inference/parallel-shape",
                          json={"parallel": {"tp_attention": "two"}})
        assert bad.status_code == 400


class TestDefaultsEndpoint:
    def test_defaults_are_the_backend_declarations(self):
        """Every product default the frontend seeds from rides one
        payload; the values are the declarations in
        `inference/defaults.py`, not a second copy."""
        from fastapi.testclient import TestClient

        from backend.api import app
        from backend.inference import defaults as d
        from backend.inference.sweep_grid import (
            AUDIT_PARALLEL,
            SCALING_CONTEXTS,
            SCALING_DEGREES,
            SPEC_PRESETS,
        )

        body = TestClient(app).get("/inference/defaults").json()
        assert body["audit_parallel"] == AUDIT_PARALLEL
        assert body["scaling_contexts"] == SCALING_CONTEXTS
        assert body["scaling_degrees"] == SCALING_DEGREES
        assert body["gpu_memory_fraction"] == d.GPU_MEMORY_FRACTION_DEFAULT
        # One speculative-preset table: the endpoint serves the presets the
        # Scaling tracks plan and web/src/compare.ts mirrors.
        assert body["spec_decode_presets"] == SPEC_PRESETS
        # The opening deployment is the backend's answer per model and card
        # (`/inference/default-deployment`), not a table the page applies.
        assert "moe_dp_ep_degree" not in body

    def test_parallel_preset_resolves_through_config_for_degree(self):
        """A preset is a mode + degree resolved server-side; a mode the
        registry does not know is a 400, not a guessed config."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        ok = client.post("/inference/estimate-sweep", json={
            "model_id": "meta-llama/Llama-3.2-1B", "gpu_sku": "H200",
            "seq_lens": [256], "avg_output_tokens": 64,
            "parallel_preset": {"mode": "tp_tp", "degree": 8},
        })
        assert ok.status_code == 200
        body = ok.json()
        results = body.get("results_by_batch") or body.get("results")
        first = results[0] if isinstance(results, list) else (
            next(iter(results.values()))[0])
        assert first["parallel"]["tp_attention"] == 8
        assert first["parallel"]["ep_moe"] == 1

        bad = client.post("/inference/estimate-sweep", json={
            "model_id": "meta-llama/Llama-3.2-1B", "gpu_sku": "H200",
            "seq_lens": [256], "avg_output_tokens": 64,
            "parallel_preset": {"mode": "nonesuch", "degree": 8},
        })
        assert bad.status_code == 400

    def test_a_malformed_parallel_is_the_callers_400(self):
        """/inference/estimate resolves the deployment inside its error
        mapping, as /inference/estimate-sweep does: a `parallel` that does
        not parse is a 400, not a 500."""
        from fastapi.testclient import TestClient

        from backend.api import app

        bad = TestClient(app, raise_server_exceptions=False).post("/inference/estimate", json={
            "model_id": "meta-llama/Llama-3.2-1B", "gpu_sku": "H200",
            "parallel": {"tp_attention": "eight"},
        })
        assert bad.status_code == 400

    def test_parallel_sweep_modes_batches_axes(self):
        """One POST with `modes` returns every axis under `by_mode`."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        r = client.post("/inference/estimate-parallel-sweep", json={
            "model_id": "meta-llama/Llama-3.2-3B",
            "gpu_sku": "B300",
            "modes": ["dp_ep", "tp_tp"],
            "contexts": [1024],
            "gpu_skus": ["B300"],
            "selection_mode": "theoretical_only",
            "avg_output_tokens": 64,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["by_mode"]) == {"dp_ep", "tp_tp"}
        assert body["by_mode"]["dp_ep"]["series"] == []
        assert body["by_mode"]["tp_tp"]["series"]


    def test_compare_fronts_read_each_rows_sweep(self):
        """`/inference/estimate-sweep/frontier`: every point is the row's own
        sweep answer at its (context, batch key) — read back through the
        sweep endpoint the page posts — and each front keeps exactly the
        points of its (context, batch key) no other row beats on both axes."""
        from fastapi.testclient import TestClient

        from backend.api import app

        client = TestClient(app)
        grid = [{"key": "b1", "prefill_batch_size": 1, "decode_batch_size": 1},
                {"key": "max", "prefill_batch_size": 1, "decode_batch_size": None}]
        rows = [{"model_id": m, "gpu_sku": "H200", "seq_lens": [1024, 8192],
                 "batch_grid": grid, "avg_output_tokens": 64,
                 "selection_mode": "theoretical_only"}
                for m in ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B")]
        r = client.post("/inference/estimate-sweep/frontier", json={"sweeps": rows})
        assert r.status_code == 200, r.text
        out = r.json()
        assert {p["row"] for p in out["points"]} == {0, 1}
        for p in out["points"]:
            ans = client.post("/inference/estimate-sweep", json=rows[p["row"]]).json()
            si = rows[p["row"]]["seq_lens"].index(p["context"])
            dec = ans["results_by_batch"][p["batch_key"]][si]["phases"]["decode"]
            assert p["tok_s_per_gpu"] == dec["throughput"]["sol"]["per_gpu"]
            assert p["tpot_s"] == dec["throughput"]["sol"]["time_per_output_token_s"]
        # Every view's line keeps exactly the points of its group nothing
        # else in the group beats on both axes: across the rows at a batch
        # point, along a row's batch points, and over everything at a context.
        key = lambda p: (p["row"], p["context"], p["batch_key"])
        views = out["fronts"]["sol"]
        assert len(views["by_batch"]) == 4 and len(views["by_row"]) == 4
        assert len(views["envelope"]) == 2
        for lines in views.values():
            for line in lines:
                at = [p for p in out["points"]
                      if all(p[d] == v for d, v in line["group"].items())]
                beaten = lambda p, at=at: any(
                    q["tok_s_per_user"] >= p["tok_s_per_user"]
                    and q["tok_s_per_gpu"] >= p["tok_s_per_gpu"] and q != p for q in at)
                assert sorted(map(key, line["frontier"])) == sorted(
                    key(p) for p in at if not beaten(p))
        bad = client.post("/inference/estimate-sweep/frontier", json={"sweeps": []})
        assert bad.status_code == 422

    def test_frontier_view_reads_the_sweeps_cloud(self):
        """`/inference/parallel-sweep/frontier` is a view of the sweep the
        same body posts: its lines are `cloud_view` of that sweep's
        cached `throughput_cloud` (kept on the server: the sweep endpoint does
        not ship it) — its frontier by default, its lines along an X when
        asked — and a dimension outside the cloud's is refused."""
        import asyncio

        from fastapi.testclient import TestClient

        from backend.api import app
        from backend.api._models import InferenceParallelSweepRequest
        from backend.api.sweep_runtime import compute_parallel_sweep_cached
        from backend.inference.parallel_sweep import cloud_view

        client = TestClient(app)
        sweep = {"model_id": "meta-llama/Llama-3.2-3B", "gpu_sku": "B300",
                 "modes": ["tp_tp"], "contexts": [1024], "gpu_skus": ["B300"],
                 "selection_mode": "theoretical_only", "avg_output_tokens": 64}
        shipped = client.post("/inference/estimate-parallel-sweep", json=sweep).json()
        assert "throughput_cloud" not in shipped
        cloud = asyncio.run(compute_parallel_sweep_cached(
            InferenceParallelSweepRequest(**sweep, sol_only=True)))["throughput_cloud"]
        fix, by = {"num_gpus": [2]}, ["gpu_sku", "batch_key"]
        r = client.post("/inference/parallel-sweep/frontier",
                        json={"sweep": sweep, "fix": fix, "group_by": by})
        assert r.status_code == 200, r.text
        assert r.json() == cloud_view(cloud, fix=fix, group_by=by)
        assert len(r.json()["series"]) > 1
        # The same holds as lines along the per-GPU batch: one view of the
        # one cloud, with the frontier's control values.
        line = client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "fix": fix, "group_by": ["gpu_sku"],
            "form": "line", "x": "batch"})
        assert line.status_code == 200, line.text
        assert line.json() == cloud_view(cloud, form="line", x="batch",
                                          fix=fix, group_by=["gpu_sku"])
        assert line.json()["values"] == r.json()["values"]
        # The other phase of the same cloud, and an X the decode cloud has
        # had all along: one more view of one cloud, no second request shape.
        pre = client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "phase": "prefill", "fix": fix,
            "group_by": ["gpu_sku"], "form": "line", "x": "context"})
        assert pre.status_code == 200, pre.text
        assert pre.json() == cloud_view(cloud, phase="prefill", form="line",
                                        x="context", fix=fix,
                                        group_by=["gpu_sku"])
        drawn = [q for line in pre.json()["series"] for q in line["points"]]
        assert drawn and all(q["prefill_tok_s_per_gpu"] > 0 for q in drawn)
        assert pre.json()["axes"]["sol"] == ["prefill_tok_s_per_user",
                                             "prefill_tok_s_per_gpu"]
        # Which (phase, form) pairs this cloud answers is its own answer,
        # the same with every view: prefill runs one request per point, so a
        # line of it along the decode batch is not a view of it.
        forms = pre.json()["forms"]
        assert forms == r.json()["forms"]
        assert forms["decode"]["batch"] is None
        assert "batch" in forms["prefill"]["batch"]
        assert forms["prefill"]["frontier"] is None
        assert forms["prefill"]["num_gpus"] is None
        assert client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "phase": "encode"}).status_code == 422
        for missing_x in ({"form": "line"}, {"x": "batch"}):
            assert client.post("/inference/parallel-sweep/frontier", json={
                "sweep": sweep, **missing_x}).status_code == 422
        bad = client.post("/inference/parallel-sweep/frontier",
                          json={"sweep": sweep, "group_by": ["parallel"]})
        assert bad.status_code == 422
        # One line per deployment, and the card's envelope over them: the
        # front the card alone draws.
        per = client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "group_by": ["gpu_sku", "layout"],
            "envelope_by": ["gpu_sku"]}).json()
        alone = client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "group_by": ["gpu_sku"]}).json()
        assert per["envelopes"] == alone["series"]
        assert len(per["series"]) > len(alone["series"])
        assert client.post("/inference/parallel-sweep/frontier", json={
            "sweep": sweep, "form": "line", "x": "batch",
            "envelope_by": ["gpu_sku"]}).status_code == 422


class TestKvBudgetWhenWeightsDoNotFit:
    def test_overflowing_weights_publish_zero_budget_and_the_deficit(self):
        """A rank whose weights exceed usable HBM still 200s; the published
        KV budget is 0 and `weight_deficit_bytes` is the shortfall. DeepSeek-V3
        on one B200 is the overflowing case the health sweep saw as −499e9."""
        from fastapi.testclient import TestClient

        from backend.api import app

        r = TestClient(app).post("/inference/estimate", json={
            "model_id": "deepseek-ai/DeepSeek-V3", "gpu_sku": "B200",
            "num_gpus": 1,
            "avg_input_tokens": 1024, "avg_output_tokens": 1024,
            "parallel": {
                "tp_attention": 1, "tp_dense_ffn": 1, "tp_moe_expert": 1,
                "ep_moe": 1, "cp": 1, "dp_attention": 1, "dp_replicas": 1,
                "pp": 1,
            },
        })
        assert r.status_code == 200, r.text
        body = r.json()
        mem = body["memory"]
        assert mem["fits"] is False
        assert mem["kv_budget_bytes"] == 0
        assert mem["weight_deficit_bytes"] > 0
        assert mem["weight_deficit_bytes"] == pytest.approx(
            mem["weight_bytes_per_gpu"] - mem["usable_bytes"])
        kv_formula = body["formulas"]["kv_budget"]
        assert kv_formula["result_bytes"] == 0
        assert body["formulas"]["weight_deficit"]["result_bytes"] == pytest.approx(
            mem["weight_deficit_bytes"])


class TestModelIdValidation:
    def test_invalid_repo_id_is_json_4xx_not_server_error(self):
        """Typing a truncated id reaches the same config-fetch failure channel
        everywhere; it must not escape huggingface_hub as a 500 HTML page."""
        from fastapi.testclient import TestClient
        from huggingface_hub.errors import HFValidationError

        from backend.api import app

        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=HFValidationError("Repo id must use owner/name"),
        ):
            deployment = client.post(
                "/precision/effective", json={"model_id": "glm-5."})
            analyze = client.post("/analyze", json={"model_id": "glm-5."})
            estimate = client.post(
                "/inference/estimate",
                json={"model_id": "glm-5.", "gpu_sku": "B200"},
            )

        for resp in (deployment, analyze, estimate):
            assert 400 <= resp.status_code < 500, (resp.status_code, resp.text)
            assert resp.headers["content-type"].startswith("application/json")
            assert "not a valid Hugging Face repo id" in resp.json()["detail"]


class TestAscendSupernodeEdge:
    """The Ascend 910C sheet row states the fabric inside its 768-die
    supernode and leaves the NIC that leaves it empty -- that link has
    never been observed. Driven through the route, not a hand-built
    fabric: what has to hold is what an estimate answers."""

    _MODEL = "zai-org/GLM-5.3-Flash"
    _TP16: ClassVar[dict] = {
        "tp_attention": 16, "tp_dense_ffn": 16, "tp_moe_expert": 16}

    def _req(self, **kw):
        snapshot_config(self._MODEL)  # missing snapshot -> skip
        return InferenceEstimateRequest(
            model_id=self._MODEL, gpu_sku="Ascend-910C-A3",
            framework="vllm_ascend", avg_input_tokens=2048,
            avg_output_tokens=128, selection_mode="theoretical_only", **kw)

    def test_a_group_wider_than_the_supernode_is_refused_by_name(self):
        """1024 ranks leave the 768-die domain over a link the row does not
        describe. The answer is a refusal naming the column -- not a step
        whose cross-domain bytes moved in zero seconds."""
        from fastapi import HTTPException
        wide = {name: 1024 for name in self._TP16}
        with pytest.raises(HTTPException) as refused:
            _run_route(self._req(parallel=wide))
        assert refused.value.status_code == 400
        assert "rdma_gbps" in refused.value.detail
        assert "Ascend-910C-A3" in refused.value.detail

    @pytest.mark.parametrize("world", [32, 8, 16, 64])
    def test_a_group_inside_the_supernode_is_priced(self, world):
        r = _run_route(self._req(parallel={n: world for n in self._TP16}))
        for phase in ("prefill", "decode"):
            assert r["phases"][phase]["t_comm_s"] > 0
            assert not r["phases"][phase]["time_unpublished_reason"]
        links = {row["comm_bw_label"]
                 for row in r["per_op_timing_decode"] if row["comm_world"]}
        assert links == {"HCCS"}

    # Each case: prefill mesh, the request's stated link, and what the
    # handoff must report (link, GB/s). The decode cluster is one chassis.
    # First: the handoff that leaves the supernode with nothing stated --
    # the one that would divide by an absent bandwidth.
    @pytest.mark.parametrize("prefill_extra, stated, link, gbps", [
        ({"dp_replicas": 48}, None, "RDMA", None),
        ({}, None, "HCCS", 196.0),
        ({}, 81.8, "stated", 81.8),
        ({"dp_replicas": 48}, 25.0, "stated", 25.0),
    ])
    def test_the_kv_handoff_crosses_the_link_the_placement_gives(
            self, prefill_extra, stated, link, gbps):
        r = _run_route(self._req(
            parallel={**self._TP16, **prefill_extra},
            decode={"parallel": self._TP16}, kv_transfer_bw_gbps=stated))
        kv = r["disagg_handoff"]["kv_transfer"]
        assert (kv["link"], kv["link_bw_gbps"]) == (link, gbps)
        if gbps is None:
            assert r["prefill_deployment"]["cluster_size"] == 768
            assert kv["handoff_latency_s"] is None
            assert "rdma_gbps" in kv["unavailable_reason"]
        else:
            assert kv["handoff_latency_s"] == pytest.approx(
                kv["per_request_kv_bytes"] / (gbps * 1e9))
            assert kv["unavailable_reason"] == ""


class TestDisaggregation:
    """Disaggregated serving: an optional `decode` deployment block runs
    estimate_capacity a SECOND time and stitches — prefill/TTFT from the
    top-level (prefill) deployment, decode/TPOT + KV capacity from the decode
    deployment. Absent `decode` → single deployment, byte-identical to today."""

    _MODEL = "meta-llama/Llama-3.2-3B"

    def _base(self, **kw):
        return InferenceEstimateRequest(
            model_id=self._MODEL, gpu_sku="H100-SXM", num_gpus=2,
            avg_input_tokens=2048, avg_output_tokens=128,
            selection_mode="theoretical_only", **kw)

    def _skip_if_absent(self):
        # Missing snapshot → skip; present but broken → the helper fails.
        snapshot_config(self._MODEL)

    def test_single_deployment_has_no_disagg(self):
        self._skip_if_absent()
        r = _run_route(self._base(parallel={"tp_attention": 2, "tp_dense_ffn": 2}))
        assert r.get("disagg") is not True
        assert "prefill_deployment" not in r and "decode_deployment" not in r

    def test_disagg_stitches_prefill_and_decode(self):
        self._skip_if_absent()
        # Prefill deployment: TP=2 (wide, compute-bound). Decode deployment:
        # CP=2 DCP (KV sharded 2-way, memory-bound) on the same 2 GPUs.
        prefill_par = {"tp_attention": 2, "tp_dense_ffn": 2}
        decode_par = {"cp": 2}
        prefill_only = _run_route(self._base(parallel=prefill_par))
        decode_only = _run_route(self._base(parallel=decode_par))
        disagg = _run_route(self._base(
            parallel=prefill_par, decode={"parallel": decode_par}))

        assert disagg["disagg"] is True
        # Prefill phase == the prefill-only run.
        assert disagg["phases"]["prefill"]["t_sol_s"] == pytest.approx(
            prefill_only["phases"]["prefill"]["t_sol_s"])
        # Decode phase + KV capacity == the decode-only run.
        assert disagg["phases"]["decode"]["t_sol_s"] == pytest.approx(
            decode_only["phases"]["decode"]["t_sol_s"])
        assert disagg["max_tokens_per_gpu"] == decode_only["max_tokens_per_gpu"]
        assert disagg["max_batch_per_gpu"] == decode_only["max_batch_per_gpu"]
        # The two deployments genuinely differ (decode CP≠prefill TP), so the
        # stitched capacity is NOT the prefill deployment's.
        assert disagg["max_tokens_per_gpu"] != prefill_only["max_tokens_per_gpu"]
        # Per-deployment summaries carry each cluster's parallel.
        assert disagg["prefill_deployment"]["parallel"]["tp_attention"] == 2
        assert disagg["decode_deployment"]["parallel"]["cp"] == 2

    def test_decode_block_falls_back_to_top_level(self):
        self._skip_if_absent()
        # decode block omits gpu_sku → inherits the top-level GPU; its GPU
        # count is its own mesh's. Should still stitch (no error).
        r = _run_route(self._base(
            parallel={"tp_attention": 2, "tp_dense_ffn": 2},
            decode={"parallel": {"cp": 2}}))
        assert r["disagg"] is True
        assert r["decode_deployment"]["num_gpus"] == 2

    def test_each_deployment_counts_its_own_mesh(self):
        """The GPU count is each deployment's mesh's rank count. The page
        sends none: a 2-rank prefill beside a 4-rank decode is two meshes,
        not a request for one GPU (or for the prefill's count) that both
        then contradict."""
        self._skip_if_absent()
        r = _run_route(InferenceEstimateRequest(
            model_id=self._MODEL, gpu_sku="H100-SXM",
            avg_input_tokens=2048, avg_output_tokens=128,
            selection_mode="theoretical_only",
            parallel={"tp_attention": 2, "tp_dense_ffn": 2},
            decode={"parallel": {"tp_attention": 4, "tp_dense_ffn": 4}}))
        assert (r["prefill_deployment"]["num_gpus"],
                r["prefill_deployment"]["cluster_size"]) == (2, 2)
        assert (r["decode_deployment"]["num_gpus"],
                r["decode_deployment"]["cluster_size"]) == (4, 4)
        # Each summary is one cluster's whole budget: its weights per GPU
        # are its own mesh's split, beside its own memory and formulas.
        pre, dec = r["prefill_deployment"], r["decode_deployment"]
        for dep in (pre, dec):
            assert (dep["memory"]["weight_bytes_per_gpu"]
                    == dep["weight_breakdown"]["total_bytes_per_rank"])
            assert dep["formulas"]["weight_per_rank"]
        assert (dec["memory"]["weight_bytes_per_gpu"]
                < pre["memory"]["weight_bytes_per_gpu"])

    def test_a_stated_count_the_mesh_disagrees_with_is_refused(self):
        from fastapi import HTTPException
        self._skip_if_absent()
        with pytest.raises(HTTPException) as e:
            _run_route(self._base(  # num_gpus=2 against a 4-rank decode mesh
                parallel={"tp_attention": 2, "tp_dense_ffn": 2},
                decode={"parallel": {"tp_attention": 4, "tp_dense_ffn": 4},
                        "num_gpus": 2}))
        assert e.value.status_code == 400
        assert "asks for 2 GPUs" in e.value.detail

    def test_disagg_handoff_reports_kv_transfer_and_rate_match(self):
        self._skip_if_absent()
        r = _run_route(self._base(
            parallel={"tp_attention": 2, "tp_dense_ffn": 2},
            decode={"parallel": {"cp": 2}}))
        h = r["disagg_handoff"]
        kv, rm = h["kv_transfer"], h["rate_match"]
        # KV handoff: positive bytes, positive latency at a real link bandwidth.
        assert kv["per_request_kv_bytes"] > 0
        assert kv["link_bw_gbps"] > 0
        assert kv["handoff_latency_s"] > 0
        # Rate match: decode is many-token, so decode per-request time >> prefill.
        assert rm["decode_time_per_request_s"] > rm["prefill_time_per_request_s"]
        assert rm["decode_to_prefill_time_ratio"] > 1

    def test_kv_transfer_bw_override_scales_handoff_latency(self):
        self._skip_if_absent()
        base = dict(parallel={"tp_attention": 2, "tp_dense_ffn": 2},
                    decode={"parallel": {"cp": 2}})
        slow = _run_route(self._base(kv_transfer_bw_gbps=50.0, **base))
        fast = _run_route(self._base(kv_transfer_bw_gbps=900.0, **base))
        s = slow["disagg_handoff"]["kv_transfer"]
        f = fast["disagg_handoff"]["kv_transfer"]
        assert s["link_bw_gbps"] == 50.0 and f["link_bw_gbps"] == 900.0
        # Same KV bytes, 18× the bandwidth → ~18× lower latency.
        assert s["per_request_kv_bytes"] == pytest.approx(f["per_request_kv_bytes"])
        assert s["handoff_latency_s"] == pytest.approx(
            f["handoff_latency_s"] * (900.0 / 50.0), rel=1e-6)


class TestRoutingImbalanceRequest:
    """`parallel.ep_load_factor` (ρ = max/mean expert load) travels the
    estimate request into pricing: on a MoE model spread over EP ranks it
    lengthens both phases, and the balanced-routing assumption the estimate
    states at ρ = 1 is withdrawn once ρ is supplied."""

    _MODEL = "Qwen/Qwen3-30B-A3B"

    def _run(self, **par):
        snapshot_config(self._MODEL)
        return _run_route(InferenceEstimateRequest(
            model_id=self._MODEL, gpu_sku="H200", num_gpus=4,
            avg_input_tokens=2048, avg_output_tokens=128,
            selection_mode="theoretical_only",
            parallel={"ep_moe": 4, "dp_attention": 4, **par}))

    def test_rho_moves_both_phases_and_retires_the_assumption(self):
        balanced = self._run()
        skewed = self._run(ep_load_factor=1.5)
        for phase in ("prefill", "decode"):
            assert (skewed["phases"][phase]["t_sol_s"]
                    > balanced["phases"][phase]["t_sol_s"]), phase
        def knobs(r):
            return [a["knob"] for a in r["assumptions"]]
        assert "ep_load_factor" in knobs(balanced)
        assert "ep_load_factor" not in knobs(skewed)

    def test_disagg_states_both_deployments_assumptions(self):
        from backend.api.routes.inference import _stitch_disagg

        def dep(assumptions):
            return {"phases": {"prefill": {}, "decode": {}},
                    "num_gpus": 1, "cluster_size": 1,
                    "assumptions": assumptions}

        shared = {"id": "joined_stream_residency", "text": "a"}
        routing = {"id": "moe_routing_balance", "knob": "ep_load_factor"}
        out = _stitch_disagg(dep([shared]), dep([shared, routing]))
        assert out["assumptions"] == [shared, routing]

    def test_a_decode_cluster_without_a_step_shows_none_of_the_other_one(self):
        """A decode cluster that cannot hold the weights publishes no step
        (`capacity.without_step`): the stitched response has no decode rows
        and no decode fusion delta — not the prefill cluster's own — and says
        which cluster admits no request."""
        from backend.api.routes.inference import _stitch_disagg
        prefill = {"phases": {"prefill": {"batch": 1}, "decode": {"batch": 4}},
                   "per_op_timing": [{"uid": 1}], "per_op_timing_decode": [{"uid": 2}],
                   "per_op_timing_decode_measured_by_declaration": {"p": 1.0},
                   "vs_naive_delta": {"prefill": {"delta_s": 1}, "decode": {"delta_s": 2}},
                   "infeasibility_reason": None, "cluster_size": 8}
        decode = {"phases": {"prefill": {"batch": None}, "decode": {"batch": None}},
                  "infeasibility_reason": "weights do not fit", "cluster_size": 8}
        out = _stitch_disagg(prefill, decode)
        assert out["phases"]["decode"] == {"batch": None}
        assert "per_op_timing_decode" not in out
        assert "per_op_timing_decode_measured_by_declaration" not in out
        assert out["per_op_timing"] == [{"uid": 1}]
        assert out["vs_naive_delta"] == {"prefill": {"delta_s": 1}}
        assert out["infeasibility_reason"] == "Decode cluster: weights do not fit"
        # Both clusters: two sentences, not "….; …".
        both = _stitch_disagg({**prefill, "infeasibility_reason": "Too big."},
                              {**decode, "infeasibility_reason": "Also too big."})
        assert both["infeasibility_reason"] == (
            "Prefill cluster: Too big. Decode cluster: Also too big.")
        assert out["decode_deployment"]["infeasibility_reason"] == "weights do not fit"


class TestRealSeedCSVEndToEnd:
    """Helper-computed identity from the REAL Llama-3.2-3B snapshot must
    round-trip to a measured hit in exact-match lookup.

    The shared rows live in Kernel Bench, not in the repository, so this
    authors
    a row via the SAME identity helpers the route uses, loads it through
    the production merge path, and proves a freshly helper-computed key
    matches it byte-for-byte. Guards against a helper change silently
    breaking the identity contract between authored rows and lookups.
    """

    def test_helper_identity_round_trips_to_measured_hit(self):
        from backend.inference.benchmark_identity import (
            compute_per_row_dtype_env_json,
            compute_per_row_shape_json,
        )
        from backend.inference.op_calibration import (
            BenchmarkKey, CalibrationRegistry, measured_time_for,
            merge_validated_rows,
        )

        # Resolve the real Llama-3.2-3B snapshot through the production
        # loader path — no stubs for the identity inputs.
        resolved = resolve_snapshot("meta-llama/Llama-3.2-3B")

        spec_hash = resolved.identity_hash()
        shape_json = compute_per_row_shape_json(
            batch_size=1, seq_len=2048, dim_values=resolved.dim_values,
        )
        dtype_env_json = compute_per_row_dtype_env_json(
            weight_dtype="bf16", activation_dtype="bf16", kv_dtype="bf16",
            metadata_dtype=None)

        # Author a row carrying exactly the helper-computed identity, then
        # load it through the production merge path (validate + index).
        row = _corpus.row(
            model="meta-llama/Llama-3.2-3B", spec_hash=spec_hash,
            parallel="tp=1", node_spec_id="self_attn.attn_apply",
            shape_json=shape_json, dtype_env_json=dtype_env_json,
            contributor="@t", date="2026-07-27",
            cross_check_rel_err="0.012", cross_check_cosine="0.999997")
        registry = CalibrationRegistry()
        merge_validated_rows(registry, [row], source="<test>")

        # A freshly helper-computed key must exact-match the authored row.
        key = BenchmarkKey(
            model="meta-llama/Llama-3.2-3B", spec_hash=resolved.identity_hash(),
            phase="prefill", parallel="tp=1", framework="vllm",
            framework_version="0.21.0", gpu="H100-SXM", op="flash_attention",
            node_spec_id="self_attn.attn_apply",
            shape_json=compute_per_row_shape_json(
                batch_size=1, seq_len=2048, dim_values=resolved.dim_values,
            ),
            dtype_env_json=compute_per_row_dtype_env_json(
                weight_dtype="bf16", activation_dtype="bf16", kv_dtype="bf16",
                metadata_dtype=None),
            kernel_impl="flash_attention_v3",
        )
        result = measured_time_for(
            registry, key, current_peak_tflops_at=lambda _dtype: 989.0, current_peak_bw_gbps=3350.0,
        )
        assert result is not None, (
            f"helper-computed identity must round-trip to a measured hit; "
            f"got miss for key {key}"
        )
        assert result.source == "measured"
        assert result.t_op_s == pytest.approx(387.5e-6, abs=0.0)
        assert result.source_badge["framework"] == "vllm"


class TestKernelMixIsolation:
    """A `kernel_mix` payload in the request body must NOT affect
    lookups when `selection_mode != "kernel_mix"`. The model validator
    on `InferenceEstimateRequest` rejects the bad combination at the
    request boundary so stale UI state from a previous experiment
    can't silently mutate a default measured request — the
    user gets a clear 422 instead of a quiet wrong-answer."""

    def test_out_of_mode_kernel_mix_rejected_at_request_boundary(self):
        """Constructing the request with kernel_mix populated AND
        selection_mode != 'kernel_mix' must raise. This is the
        stricter contract — the old server-side mode gate would
        silently ignore the payload; the request validator surfaces
        the contradiction to the client."""
        import pytest as _pytest
        with _pytest.raises(Exception) as exc:
            _make_req(
                selection_mode="latest",
                framework="vllm",
                kernel_mix={"flash_attention": "this_should_be_ignored"},
            )
        # Pydantic wraps the ValueError into a ValidationError.
        msg = str(exc.value)
        assert "kernel_mix" in msg and "kernel_mix" in msg, (
            f"validation error must name the offending field "
            f"(kernel_mix) and the required selection_mode value; "
            f"got: {msg}"
        )

    def test_empty_kernel_mix_accepted_in_any_mode(self):
        """`kernel_mix={}` and `kernel_mix=None` carry no intent and
        must not trigger the validator — UI state often serializes an
        empty dict even when the editor was never touched."""
        _make_req(selection_mode="latest", kernel_mix={})
        _make_req(selection_mode="latest", kernel_mix=None)
        _make_req(selection_mode="theoretical_only", kernel_mix={})

    def test_kernel_mix_accepted_in_kernel_mix_mode(self):
        """The validator only fires on the contradiction; the
        well-formed combination must still be accepted."""
        _make_req(
            selection_mode="kernel_mix",
            kernel_mix={"flash_attention": "fa_v3_quirky"},
        )


class TestPrecisionRequestBoundary:
    """Precision is one field, `precision`, validated and written in its one
    spelling at the request boundary; the knobs it replaced are refused by
    name, never translated."""

    def test_retired_fields_are_refused_by_name(self):
        for field, value in (("kv_dtype", "fp8"), ("act_dtype_bytes", 1),
                             ("weight_dtype_bytes", 1.0),
                             ("compute_dtype_bytes", 1.0)):
            with pytest.raises(Exception) as exc:
                _make_req(**{field: value})
            assert f"`{field}`" in str(exc.value) and "precision" in str(exc.value)

    def test_a_preset_and_its_rules_are_one_profile(self):
        from backend.arch.eval.precision import PRESETS
        preset = _make_req(precision={"preset": "kv_fp8"})
        spelled = _make_req(precision={
            "canonical": PRESETS["kv_fp8"].canonical()})
        assert preset.precision == spelled.precision
        assert preset.precision_profile().key == PRESETS["kv_fp8"].key

    def test_an_unknown_preset_is_refused_with_the_vocabulary(self):
        with pytest.raises(Exception) as exc:
            _make_req(precision={"preset": "fp8_everything"})
        assert "kv_fp8" in str(exc.value)

class TestRankSlicing:
    """7.7: the compute-graph route's per-rank fragment helpers."""

    def _graph(self):
        nodes = [
            {"id": "embed", "op": "embedding", "inputs": []},
            {"id": "__block__a", "op": "block_stack", "count": 4,
             "label": "a × 4", "inputs": ["embed"], "children": []},
            {"id": "__block__b", "op": "block_stack", "count": 2,
             "label": "b × 2", "inputs": ["__block__a"], "is_moe": True,
             "children": [{"op": "grouped_gemm", "children": []}]},
            {"id": "__block__mtp", "op": "block_stack", "aux_kind": "mtp",
             "count": 1, "inputs": ["__block__b"], "children": []},
            {"id": "norm", "op": "rmsnorm", "inputs": ["__block__b"]},
        ]
        edges = [{"src": "embed", "dst": "__block__a"},
                 {"src": "__block__a", "dst": "__block__b"},
                 {"src": "__block__b", "dst": "norm"}]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _stage_graph(pp, rank, model="meta-llama/Llama-3.2-3B"):
        from fastapi.testclient import TestClient
        from backend.api import app
        resp = TestClient(app).get("/inference/compute-graph", params={
            "model_id": model, "pp": pp, "pp_rank": rank})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        return {**body["graph"], "stage": body["stage"]}

    def test_each_stage_draws_what_placement_puts_on_it(self):
        """Llama-3.2-3B at pp = 2 (layers 0-13 | 14-27): the embedding runs
        with layer 0, the norm and the head with layer 27, and each stage's
        block stack is drawn with that stage's layers."""
        first, last = self._stage_graph(2, 0), self._stage_graph(2, 1)

        def ids(g):
            return {n["id"] for n in g["nodes"]}
        assert any("embed_tokens" in i for i in ids(first))
        assert not any("lm_head" in i for i in ids(first))
        assert any("lm_head" in i for i in ids(last))
        # The last stage does not run the embedding; it receives the first
        # stage's output across the cut.
        assert "embed_tokens" not in ids(last)
        assert [n["label"] for n in last["nodes"] if n["op"] == "recv"] == [
            "recv from stage 0"]
        for g, (lo, hi) in ((first, (0, 14)), (last, (14, 28))):
            assert g["stage"]["layers"] == [lo, hi]
            (stack,) = [n for n in g["nodes"] if n["op"] == "block_stack"]
            assert stack["count"] == 14
            assert "× 14 of 28" in stack["label"]
            # the fragment's edges are internal only
            assert all(e["src"] in ids(g) and e["dst"] in ids(g)
                       for e in g["edges"])

    def test_the_partition_follows_the_canonical_split(self):
        """28 layers over 8 stages: 3,3,3,4,4,4,4,3."""
        from backend.arch.layout.parallel_config import stage_layer_ranges
        for rank, held in enumerate(stage_layer_ranges(28, 8)):
            g = self._stage_graph(8, rank)
            assert g["stage"]["layers"] == [held.start, held.stop]


class TestActivationPrecision:
    """The activation table prices the capacity peak, not the weight dtype.

    The precision profile's activation rule is the declaration of activation
    precision, and the peak-activation ceiling that bounds max batch reads
    it: F32 edges hold twice the BF16 peak on the same BF16 weights.
    """

    def test_peak_activation_follows_the_activation_rule(self):
        f32 = {"canonical": {"activations": [{"format": {"dtype": "F32"}}]}}
        peaks = {
            name: _run_route(_make_req(**kw))
            ["activation_peak"]["peak_activation_bytes_per_rank"]
            for name, kw in (("bf16", {}), ("f32", {"precision": f32}))
        }
        assert peaks["bf16"] > 0
        assert peaks["f32"] == pytest.approx(2 * peaks["bf16"], rel=1e-9)


class TestPerOpRowsByDeclaration:
    """`/inference/estimate` adds, beside each phase's per-op rows, the rows
    per declaration of any derived groups among them — and nothing when there
    are none. The rows themselves are returned as computed."""

    @pytest.fixture(scope="class")
    def rows(self):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        from tests.backend.arch.test_prefix_streams import _resolved
        return analyze(_resolved("decl_route_toy"),
                       parallel=ParallelConfig.from_dict({}),
                       batch_size=1, avg_input_tokens=4, avg_output_tokens=0, breakdown=True).per_op_timing

    def _estimate(self, prefill, decode):
        from backend.api.routes import inference as route
        result = {"per_op_timing": prefill, "per_op_timing_decode": decode}
        with mock.patch.object(route, "_resolve_estimate_environment",
                               return_value={}), \
                mock.patch.object(route, "estimate_pending",
                                  return_value=None), \
                mock.patch.object(route, "finish_estimates",
                                  return_value=[result]):
            return _run_route(_make_req())

    def test_derived_rows_get_their_declaration_rows(self, rows):
        from backend.views import declarations
        plain = [r for r in rows if "derived_group" not in r]
        from backend.inference.module_rows import served_rows
        out = self._estimate(rows, plain)
        # The rows as a response serves them: their laws stay in the process.
        assert out["per_op_timing"] == served_rows(rows)
        parents = out["per_op_timing_by_declaration"]
        assert parents == declarations.per_op_rows(rows)
        assert {p["declaration"]["base_template"] for p in parents} == {
            "start", "plain"}
        for p in parents:
            assert p["flops_per_rank"] == sum(
                rows[i]["flops_per_rank"] for i in p["children"])
            assert p["t_sol_s"] == sum(rows[i]["t_sol_s"] for i in p["children"])
        assert "per_op_timing_decode_by_declaration" not in out


class TestParallelShapeReadsTheModel:
    """`/inference/parallel-shape` with a `model_id` answers for the mesh
    that model runs: a dense model has no expert region, so expert knobs
    beside context parallelism are not an expert mesh to refuse."""

    _PAR: ClassVar[dict] = {"tp_attention": 2, "tp_dense_ffn": 2,
                            "tp_moe_expert": 2, "ep_moe": 1,
                            "dp_attention": 1, "cp": 2}

    def _shape(self, **body):
        from fastapi.testclient import TestClient

        from backend.api import app
        r = TestClient(app).post("/inference/parallel-shape", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_dense_model_has_no_expert_mesh(self):
        snapshot_config("Qwen/Qwen3-32B")
        bare = self._shape(parallel=self._PAR)
        assert any("MoE parallelism" in e for e in bare["errors"]), bare
        dense = self._shape(parallel=self._PAR, model_id="Qwen/Qwen3-32B")
        assert dense == {"cluster_size": 4, "per_stage_size": 4, "errors": []}

    def test_moe_model_keeps_its_expert_mesh(self):
        snapshot_config("Qwen/Qwen3-30B-A3B")
        moe = self._shape(parallel=self._PAR, model_id="Qwen/Qwen3-30B-A3B")
        assert any("MoE parallelism" in e for e in moe["errors"]), moe


def test_a_plan_export_carries_routing_skew_as_its_point(monkeypatch):
    """ρ leaves the mesh at the route and reaches the planner on PlanPoint."""
    import kernel_bench.plan as planner
    from backend.api._models import BenchPlanJobsRequest
    from backend.api.routes.inference import _plan_jobs

    asked = []

    def fake(**kwargs):
        point = kwargs["points"][0]
        asked.append(point)
        return {point: ([], [])}

    monkeypatch.setattr(planner, "build_jobs_over_points", fake)
    answer = _plan_jobs(BenchPlanJobsRequest(
        model_id="Qwen/Qwen3-30B-A3B", phase="prefill", gpu_sku="H100-SXM",
        parallel={"tp_attention": 2, "ep_load_factor": 1.5},
        batch_size=1, avg_input_tokens=1024, avg_output_tokens=0))
    assert answer["job_count"] == 0
    assert asked[0].ep_load_factor == 1.5
    assert asked[0].counts("prefill")["rho"] == 1.5


@pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5, 7.0])
def test_acceptance_rate_outside_its_range_is_refused_at_the_door(alpha):
    """α ∈ (0, 1]: a rejected draft keeps 1 token, an accepted one 1 + K, so
    a rate outside the range has no step it describes."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="acceptance_rate"):
        _make_req(speculative_draft_tokens=3, acceptance_rate=alpha)


@pytest.mark.parametrize("alpha", [0.05, 0.65, 1.0])
def test_acceptance_rate_inside_its_range_is_taken_as_stated(alpha):
    assert _make_req(speculative_draft_tokens=3,
                     acceptance_rate=alpha).acceptance_rate == alpha
