"""The shared projection layer below Gallery and Scaling responses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from backend.inference.card import Card
from backend.inference.serving_projection import (
    PublicServingLawRequest,
    SERVING_PROJECTION,
    projectServingLawPoints,
)
from backend.inference.sol_data import ServingWalkKey
from backend.inference.sweep_grid import (
    FRONTIER_BATCH_GRID,
    GALLERY_SEQ_LENS,
    SCALING_BATCH_GRID,
)


def _request(**changes):
    request = PublicServingLawRequest(
        walk=ServingWalkKey.of_card(
            "rv", model="meta-llama/Llama-3.2-3B", card=Card.of("H100-SXM")),
        card="H100-SXM",
        layout={},
        environment={
            "avg_output_tokens": 256,
            "acceptance_rate": 1.0,
            "comm_latency_us": 2.0,
            "launch_floor_us": 1.0,
        },
        contexts=(4096,),
        decode_batches=(1,),
        prefill_batches=(1,),
        prefix_tokens=(0,),
        verify_widths=(1,),
    )
    return replace(request, **changes)


def _project(monkeypatch):
    calls = []

    def rows(record, *, contexts, decodeBatches, **_kwargs):
        calls.append((tuple(contexts), tuple(decodeBatches)))
        return [[{"context": context, "batch": batch}
                 for batch in decodeBatches]
                for context in contexts]

    monkeypatch.setattr(
        "backend.inference.serving_projection.servingLawPoints", rows)
    return calls


def test_the_same_axes_in_a_different_order_are_one_projection(monkeypatch):
    calls = _project(monkeypatch)
    request = _request(
        contexts=(4096, 1024),
        decode_batches=(None, 1),
        prefill_batches=(1, 1),
        prefix_tokens=(0, 0),
        verify_widths=(1, 1),
    )
    reordered = _request(
        contexts=(1024, 4096),
        decode_batches=(1, None),
        prefill_batches=(1, 1),
        prefix_tokens=(0, 0),
        verify_widths=(1, 1),
    )

    first = projectServingLawPoints(
        {}, request, recordDigest="record")
    second = projectServingLawPoints(
        {}, reordered, recordDigest="record")

    assert calls == [((1024, 4096), (1, None))]
    assert first == [
        [{"context": 4096, "batch": None}, {"context": 4096, "batch": 1}],
        [{"context": 1024, "batch": None}, {"context": 1024, "batch": 1}],
    ]
    assert second == [
        [{"context": 1024, "batch": 1}, {"context": 1024, "batch": None}],
        [{"context": 4096, "batch": 1}, {"context": 4096, "batch": None}],
    ]


def test_a_hit_returns_an_independent_copy(monkeypatch):
    calls = _project(monkeypatch)
    request = _request()

    first = projectServingLawPoints({}, request, recordDigest="record")
    second = projectServingLawPoints({}, request, recordDigest="record")

    assert calls == [((4096,), (1,))]
    assert first == second
    first[0][0]["caller_write"] = True
    assert "caller_write" not in second[0][0]


def test_the_durable_tier_serves_a_fresh_process():
    """A pool worker and the warm parent are two interpreters, one entry."""
    SERVING_PROJECTION.clear()
    child = """
import json, os

import backend.inference.serving_projection as sp
from backend.inference.card import Card
from backend.inference.sol_data import ServingWalkKey

if os.environ["CHILD_MODE"] == "reader":
    def recompute(*args, **kwargs):
        raise AssertionError("a fresh process re-projected a durable row")
    sp.servingLawPoints = recompute
else:
    sp.servingLawPoints = lambda *args, **kwargs: [
        [{"context": 4096, "batch": 1}]]

request = sp.PublicServingLawRequest(
    walk=ServingWalkKey.of_card(
        "rv", model="meta-llama/Llama-3.2-3B", card=Card.of("H100-SXM")),
    card="H100-SXM", layout={},
    environment={"avg_output_tokens": 256, "acceptance_rate": 1.0,
                 "comm_latency_us": 2.0, "launch_floor_us": 1.0},
    contexts=(4096,), decode_batches=(1,), prefill_batches=(1,),
    prefix_tokens=(0,), verify_widths=(1,))
print(json.dumps(sp.projectServingLawPoints(
    {}, request, recordDigest="record")))
"""
    answers = []
    for mode in ("writer", "reader"):
        env = {**os.environ, "CHILD_MODE": mode}
        run = subprocess.run([sys.executable, "-c", child], env=env,
                             text=True, capture_output=True, check=True)
        answers.append(json.loads(run.stdout))
    assert answers[0] == answers[1] == [[{"context": 4096, "batch": 1}]]


_KEY_CHANGES = [
    {"card": "H200"},
    {"layout": {"tp_attention": 2}},
    {"contexts": (2048,), "prefix_tokens": (0,), "verify_widths": (1,)},
    {"decode_batches": (2,), "prefill_batches": (1,)},
    {"decode_batches": (1,), "prefill_batches": (2,)},
    {"prefix_tokens": (1,)},
    {"verify_widths": (2,)},
    {"draft_tokens": 1},
    {"environment": {
        "avg_output_tokens": 128, "acceptance_rate": 1.0,
        "comm_latency_us": 2.0, "launch_floor_us": 1.0,
    }},
    {"environment": {
        "avg_output_tokens": 256, "acceptance_rate": 0.5,
        "comm_latency_us": 2.0, "launch_floor_us": 1.0,
    }},
    {"environment": {
        "avg_output_tokens": 256, "acceptance_rate": 1.0,
        "comm_latency_us": 3.0, "launch_floor_us": 1.0,
    }},
    {"environment": {
        "avg_output_tokens": 256, "acceptance_rate": 1.0,
        "comm_latency_us": 2.0, "launch_floor_us": 2.0,
    }},
]


def test_every_projection_input_is_in_the_key(monkeypatch):
    calls = _project(monkeypatch)
    base = _request()

    for change in _KEY_CHANGES:
        changed = _request(**change)
        assert (base.projection_identity(recordDigest="record").key
                != changed.projection_identity(recordDigest="record").key)
        SERVING_PROJECTION.clear()
        calls.clear()
        projectServingLawPoints({}, base, recordDigest="record")
        projectServingLawPoints({}, changed, recordDigest="record")
        assert len(calls) == 2


def test_the_callers_record_digest_is_in_the_key(monkeypatch):
    calls = _project(monkeypatch)
    request = _request()

    projectServingLawPoints({"record": 1}, request, recordDigest="one")
    projectServingLawPoints({"record": 2}, request, recordDigest="two")

    assert len(calls) == 2


def test_the_record_digest_is_explicit_not_rederived():
    request = _request()

    with pytest.raises(TypeError, match="recordDigest"):
        projectServingLawPoints({}, request)


def test_floor_forms_is_a_view_not_a_second_cache_axis(monkeypatch):
    request = _request()
    monkeypatch.setattr(
        "backend.inference.serving_projection.servingLawPoints",
        lambda *args, **kwargs: [[{
            "context": 4096, "batch": 1,
            "phases": {"decode": {"floors": "step form"}},
        }]])

    plain = projectServingLawPoints(
        {}, request, floorForms=False, recordDigest="record")
    rich = projectServingLawPoints(
        {}, request, floorForms=True, recordDigest="record")

    assert plain[0][0]["phases"]["decode"] == {}
    assert rich[0][0]["phases"]["decode"]["floors"] == "step form"


def test_public_gallery_and_scaling_axes_share_one_projection(monkeypatch):
    calls = _project(monkeypatch)
    gallery = _request(
        contexts=tuple(GALLERY_SEQ_LENS),
        prefix_tokens=(0,) * len(GALLERY_SEQ_LENS),
        verify_widths=(1,) * len(GALLERY_SEQ_LENS),
        decode_batches=tuple(
            point["decode_batch_size"] for point in FRONTIER_BATCH_GRID),
        prefill_batches=(1,) * len(FRONTIER_BATCH_GRID),
    )
    scaling = _request(
        contexts=tuple(GALLERY_SEQ_LENS),
        prefix_tokens=(0,) * len(GALLERY_SEQ_LENS),
        verify_widths=(1,) * len(GALLERY_SEQ_LENS),
        decode_batches=tuple(
            point["decode_batch_size"] for point in SCALING_BATCH_GRID),
        prefill_batches=(1,) * len(SCALING_BATCH_GRID),
    )

    galleryRows = projectServingLawPoints(
        {}, gallery, floorForms=True, publicEnvelope=True,
        recordDigest="record")
    scalingRows = projectServingLawPoints(
        {}, scaling, floorForms=False, publicEnvelope=True,
        recordDigest="record")

    assert calls == [(tuple(GALLERY_SEQ_LENS), tuple(
        point["decode_batch_size"] for point in FRONTIER_BATCH_GRID))]
    assert (len(galleryRows), len(galleryRows[0])) == (
        len(GALLERY_SEQ_LENS), len(FRONTIER_BATCH_GRID))
    assert (len(scalingRows), len(scalingRows[0])) == (
        len(GALLERY_SEQ_LENS), len(SCALING_BATCH_GRID))
    assert galleryRows[0][0]["context"] == GALLERY_SEQ_LENS[0]
    assert scalingRows[0][0]["context"] == GALLERY_SEQ_LENS[0]


def test_a_custom_public_caller_does_not_pay_for_the_whole_envelope(
        monkeypatch):
    calls = _project(monkeypatch)
    request = _request(contexts=(4096,), decode_batches=(1,))

    assert request.publicEnvelope() is None
    projectServingLawPoints(
        {}, request, publicEnvelope=True, recordDigest="record")

    assert calls == [((4096,), (1,))]


def test_a_record_without_envelope_contexts_stays_exact(monkeypatch):
    calls = _project(monkeypatch)
    request = _request(
        contexts=tuple(GALLERY_SEQ_LENS),
        prefix_tokens=(0,) * len(GALLERY_SEQ_LENS),
        verify_widths=(1,) * len(GALLERY_SEQ_LENS),
        decode_batches=tuple(
            point["decode_batch_size"] for point in FRONTIER_BATCH_GRID),
        prefill_batches=(1,) * len(FRONTIER_BATCH_GRID),
    )
    record = {"capacity": [
        {"gpu_sku": request.card, "context": context, "max_decode_batch": 1}
        for context in GALLERY_SEQ_LENS
    ]}

    projectServingLawPoints(
        record, request, publicEnvelope=True, recordDigest="record")

    assert calls == [(tuple(GALLERY_SEQ_LENS), tuple(
        point["decode_batch_size"] for point in FRONTIER_BATCH_GRID))]


def test_the_projection_is_machine_local():
    identity = _request().projection_identity(recordDigest="record")
    assert identity.remote_ok is False
