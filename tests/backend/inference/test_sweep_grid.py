"""The routine shape grid has one source of truth (backend.inference.sweep_grid).
The web Compare grid hard-codes the same points (there is no runtime config
channel for it), so these tests fail if `web/src` drifts from the backend —
that is the enforcement that keeps "the routine shapes" meaning one thing.
The Scaling panel's context / GPU-count grid is read from `/inference/defaults`
instead. The gallery's axes are generated into `web/src/gallery/axes.json`
instead (see `test_react_gallery_axes_json_matches_sweep_grid`).
"""
from __future__ import annotations

import pathlib
import re

from backend.inference.sweep_grid import (
    GALLERY_SEQ_LENS,
    ROUTINE_OUTPUT_TOKENS,
    ROUTINE_SEQ_LENS,
    SCALING_BATCH_GRID,
    SCALING_PANEL_BATCH_GRID,
    SPEC_PRESETS,
)

_WEB = pathlib.Path(__file__).resolve().parents[3] / "web" / "src"


def _int_array(ts: str, name: str) -> list[int]:
    m = re.search(rf"{re.escape(name)}\s*=\s*\[([^\]]*)\]", ts)
    assert m, f"{name} not found"
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def _batch_grid(ts: str) -> list[dict]:
    block = re.search(r"SWEEP_BATCH_GRID\b[^=]*=\s*\[(.*?)\]\s*;", ts, re.S)
    assert block, "SWEEP_BATCH_GRID not found"
    out = []
    for row in re.findall(r"\{([^}]*)\}", block.group(1)):
        key = re.search(r"key:\s*[\"']([^\"']+)[\"']", row).group(1)
        pre = re.search(r"prefill_batch_size:\s*(-?\d+)", row).group(1)
        dec = re.search(r"decode_batch_size:\s*(-?\d+|null)", row).group(1)
        out.append({
            "key": key,
            "prefill_batch_size": int(pre),
            "decode_batch_size": None if dec == "null" else int(dec),
        })
    return out


def test_web_sweep_grid_matches_backend():
    """`web/src/compare.ts` feeds both the Scaling panel and Compare."""
    ts = (_WEB / "compare.ts").read_text()
    assert _int_array(ts, "SWEEP_SEQ_LENS") == GALLERY_SEQ_LENS
    assert _batch_grid(ts) == SCALING_BATCH_GRID
    out = re.search(r"SWEEP_OUTPUT_TOKENS\s*=\s*(\d+)", ts)
    assert out and int(out.group(1)) == ROUTINE_OUTPUT_TOKENS


def test_the_panels_context_batch_grid_is_the_rungs_it_draws():
    """The Scaling panel's Context × Batch body asks the cloud's own ladder,
    in both languages by one rule."""
    ts = (_WEB / "compare.ts").read_text()
    # The panel asks the full grid (`docs/sweep_card_class.md` §2.B): the
    # trim to a few rungs was a cost decision the class machinery dissolved.
    assert [bp["key"] for bp in SCALING_PANEL_BATCH_GRID] == [
        "b1", "b8", "b16", "b32", "b64", "b128", "b256", "b512",
        "b1024", "max"]
    # The TS mirror asks the full grid too (no filter).
    assert "SCALING_PANEL_BATCH_GRID: SweepBatchPoint[] = SWEEP_BATCH_GRID" in ts


def test_serving_prefill_stays_at_one_request():
    """The public Context × Batch grid sweeps decode concurrency only."""
    assert all(point["prefill_batch_size"] == 1
               for point in SCALING_BATCH_GRID)
    assert [point["decode_batch_size"] for point in SCALING_BATCH_GRID] == [
        1, 8, 16, 32, 64, 128, 256, 512, 1024, None]


def _spec_presets(ts: str) -> dict:
    block = re.search(r"SPEC_PRESETS\b[^=]*=\s*\{(.*?)\n\};", ts, re.S)
    assert block, "SPEC_PRESETS not found"
    return {name: {"K": int(k), "alpha": float(a)} for name, k, a in re.findall(
        r"(\w+):\s*\{\s*K:\s*(\d+),\s*alpha:\s*([\d.]+)", block.group(1))}


def test_web_spec_presets_match_backend():
    """The Scaling panel's speculative presets are the ones the Kernel Bench
    routine tracks plan (`ROUTINE_TRACKS`' decode draft lengths). `custom`
    is the seed of the free-form input on the Inference page, not a preset."""
    ts = (_WEB / "compare.ts").read_text()
    presets = _spec_presets(ts)
    assert "custom" in presets
    del presets["custom"]
    assert presets == SPEC_PRESETS


def test_react_gallery_axes_json_matches_sweep_grid():
    """web/src/gallery/axes.json is generated from sweep_grid, not a third table."""
    import json
    from backend.inference.gallery_axes import axes_payload
    from backend.inference.parallel_sweep import DEFAULT_SWEEP_GPUS
    from backend.inference.sweep_grid import (
        AUDIT_PARALLEL,
        GALLERY_GPUS,
        GALLERY_LANDING_GPU,
        GALLERY_PARALLEL_PRESETS,
    )

    path = pathlib.Path(__file__).resolve().parents[3] / "web" / "src" / "gallery" / "axes.json"
    data = json.loads(path.read_text())
    live = axes_payload()
    assert data == live
    assert data["landing_gpu"] == GALLERY_LANDING_GPU
    # The picker's GPUs are the Gallery series, and the landing GPU is one.
    assert data["gpus"] == list(GALLERY_GPUS)
    assert GALLERY_LANDING_GPU in data["gpus"]
    # The Scaling series is every card the Inference page offers (the
    # sheet); the Gallery's curated series is within it, less the Kernel
    # Bench card the Scaling panel compares against.
    from backend.inference.gpu_specs import load_gpu_specs
    assert [g.sku for g in load_gpu_specs()] == DEFAULT_SWEEP_GPUS
    assert set(GALLERY_GPUS) <= set(DEFAULT_SWEEP_GPUS)
    assert "RTX-4090" in DEFAULT_SWEEP_GPUS and "RTX-4090" not in GALLERY_GPUS
    # The gallery shows every model as shipped: no precision axis.
    assert "weight_dtypes" not in data
    assert "weight_dtype_options" not in data
    assert [p["value"] for p in data["parallel_presets"]] == list(GALLERY_PARALLEL_PRESETS)
    assert data["audit_parallel"] == dict(AUDIT_PARALLEL)
    assert not (path.parent / "presets.ts").exists()


def test_gallery_and_kernel_bench_draw_from_canonical():
    # Consumers import from the canonical module (not their own copies).
    from backend.api.routes import gallery
    from backend.inference.sweep_grid import GALLERY_SEQ_LENS
    from kernel_bench.sweep import SWEEP_SEQ_LENS
    assert gallery.GALLERY_SEQ_LENS is GALLERY_SEQ_LENS
    assert SWEEP_SEQ_LENS is ROUTINE_SEQ_LENS
    # Gallery's curated subset is a real subset of the canonical grid.
    assert set(GALLERY_SEQ_LENS) <= set(ROUTINE_SEQ_LENS)
