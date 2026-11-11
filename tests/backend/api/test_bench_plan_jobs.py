"""`/inference/bench/plan-jobs` — the Inference page's plan as bench jobs.

The contract worth a test is that there is ONE builder with two entry
points: whatever `hbench plan` writes for a set of inputs, the endpoint
answers with for the same inputs — the same bytes, not an equivalent list.
A second serialization (a route that rebuilt a job field, or a client that
re-dumped the dicts it was sent) would key measurements to shapes nobody
planned, and nothing else in the harness would notice.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from tests.snapshots import snapshot_config

TEXT_MODEL = "meta-llama/Llama-3.2-3B"
#: A model with an encoder, and media whose sides are whole multiples of its
#: processor's resize factor (`test_encode_estimate`).
ENCODER_MODEL = "moonshotai/Kimi-K2.5"
IMAGE = {"pixel_values": {"n_img": 2, "img_h": 672, "img_w": 1344}}


def _client() -> TestClient:
    return TestClient(app)


def _export(**body) -> dict:
    body.setdefault("gpu_sku", "H100-SXM")
    response = _client().post("/inference/bench/plan-jobs", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _attention_dim(jsonl: str, dim: str) -> int:
    """One dim of the exported attention job — the per-rank geometry a
    measurement of this plan would be built at."""
    for line in jsonl.splitlines():
        job = json.loads(line)
        if job["op"] == "flash_attention":
            return job["dims"][dim]
    raise AssertionError(f"no flash_attention job in {len(jsonl.splitlines())} jobs")


class TestSameBuilderAsTheCli:
    def test_endpoint_bytes_equal_what_hbench_plan_writes(self, tmp_path):
        """Both entry points, built here, byte for byte.

        The CLI side runs `hbench plan`'s own argv handler rather than
        calling the builder again — the file a user would get on the box is
        the thing the download has to match.
        """
        snapshot_config(TEXT_MODEL)
        from kernel_bench.cli import _cmd_plan

        out = tmp_path / "jobs.jsonl"
        assert _cmd_plan([
            TEXT_MODEL, "--batch", "1", "--seq", "2048",
            "--phase", "prefill", "--output-tokens", "0", "-o", str(out),
            "--gpu", "H100-SXM",
        ]) == 0
        written = out.read_text(encoding="utf-8")

        body = _export(model_id=TEXT_MODEL, phase="prefill", batch_size=1,
                       avg_input_tokens=2048, avg_output_tokens=0,
                       gpu_sku="H100-SXM")
        assert body["jsonl"] == written
        assert body["job_count"] == len(written.splitlines())
        assert body["job_count"] > 0
        assert body["filename"] == "meta-llama_Llama-3.2-3B_prefill_jobs.jsonl"

    def test_the_deployment_reaches_the_builder_whole(self):
        """The page sends the mesh it edited, not a flattened TP degree: a
        sharded export must carry the per-RANK geometry and the canonical
        parallel signature the estimate looks measurements up with."""
        snapshot_config(TEXT_MODEL)
        from backend.inference.calibration_overlay import parallel_signature
        from backend.inference.parallel_config import ParallelConfig

        mesh = dict(tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4)
        whole = _export(model_id=TEXT_MODEL, phase="prefill", batch_size=1,
                        avg_input_tokens=2048, avg_output_tokens=0)
        shard = _export(model_id=TEXT_MODEL, phase="prefill", batch_size=1,
                        avg_input_tokens=2048, avg_output_tokens=0,
                        parallel=mesh)
        want = parallel_signature(ParallelConfig(**mesh))
        jobs = [json.loads(line) for line in shard["jsonl"].splitlines()]
        assert {j["parallel"] for j in jobs} == {want}
        assert (_attention_dim(shard["jsonl"], "n_heads")
                == _attention_dim(whole["jsonl"], "n_heads") // 4)

    def test_the_workload_reaches_the_builder(self):
        """Prompt, cached prefix and verify width are the point's fields: a
        step exported at another one is a measurement of another step.
        The speculative step needs a model with an MTP draft stack (the
        planner refuses speculative tokens on one without)."""
        MTP_MODEL = "Qwen/Qwen3.5-35B-A3B"
        snapshot_config(TEXT_MODEL)
        snapshot_config(MTP_MODEL)
        plain = _export(model_id=MTP_MODEL, phase="decode", batch_size=8,
                        avg_input_tokens=2048, avg_output_tokens=128)
        verify = _export(model_id=MTP_MODEL, phase="decode", batch_size=8,
                         avg_input_tokens=2048, avg_output_tokens=128,
                         speculative_draft_tokens=3)
        # K drafted tokens are verified in one forward over K + 1 positions.
        assert _attention_dim(plain["jsonl"], "s_query") == 1
        assert _attention_dim(verify["jsonl"], "s_query") == 4


class TestWhatIsNotExported:
    def test_dropped_ops_come_back_named(self):
        """A plan drops the ops no provider can measure (zero-cost layout
        views, families with no signature). They are part of the answer: an
        unreported drop reads exactly like full coverage."""
        snapshot_config(TEXT_MODEL)
        body = _export(model_id=TEXT_MODEL, phase="prefill", batch_size=1,
                       avg_input_tokens=2048, avg_output_tokens=0)
        assert isinstance(body["dropped"], list)
        assert body["dropped"], "a transformer plans no job for its views"
        assert all(isinstance(note, str) and note for note in body["dropped"])

    def test_a_bad_phase_is_refused(self):
        response = _client().post("/inference/bench/plan-jobs", json={
            "model_id": TEXT_MODEL, "phase": "warmup"})
        assert response.status_code == 422, response.text
        assert "phase" in response.text

    def test_a_model_with_no_spec_is_refused_not_guessed(self):
        response = _client().post("/inference/bench/plan-jobs", json={
            "model_id": "not-a-real-org/not-a-real-model", "phase": "prefill",
            "gpu_sku": "H100-SXM"})
        assert response.status_code == 400, response.text

    def test_a_queue_export_needs_the_displayed_card(self):
        """A card-free API export is not silently a box plan: mixed-precision
        launch dispatch can change the key the displayed estimate asks for."""
        snapshot_config(TEXT_MODEL)
        response = _client().post("/inference/bench/plan-jobs", json={
            "model_id": TEXT_MODEL, "phase": "prefill"})
        assert response.status_code == 422, response.text
        assert "gpu_sku" in response.text


@pytest.mark.regression  # resolves a snapshot with an encoder
class TestEncodePhase:
    def test_an_encoder_step_plans_over_its_media(self):
        snapshot_config(ENCODER_MODEL)
        body = _export(model_id=ENCODER_MODEL, phase="encode", batch_size=1,
                       media=IMAGE)
        assert body["job_count"] > 0
        assert body["filename"].endswith("_encode_jobs.jsonl")
        jobs = [json.loads(line) for line in body["jsonl"].splitlines()]
        assert {j["phase"] for j in jobs} == {"encode"}

    def test_an_encode_plan_with_no_media_says_so(self):
        snapshot_config(ENCODER_MODEL)
        response = _client().post("/inference/bench/plan-jobs", json={
            "model_id": ENCODER_MODEL, "phase": "encode", "batch_size": 1,
            "gpu_sku": "H100-SXM"})
        assert response.status_code == 400, response.text
        assert "media" in response.json()["detail"]
