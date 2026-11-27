"""The paged-KV page size is part of what a measurement measured.

A paged attention kernel's time depends on the page the cache manager
allocates in: how many pages one query gathers, and how coalesced each
gather is. The plan has always emitted one job per page a deployment might
run (16 / 64 / 128), because which of them a backend can serve is the
backend's own constraint. But the page travelled only in `kernel_params`,
and `BenchmarkKey` does not index that — so all three jobs carried ONE key,
three different measurements overwrote each other under it, and the
estimate got whichever was written last. The collision was live.

The page is a deployment parameter with one owner: the framework binding
declares its default (it owns the cache manager) and a request may override
it. It is an axis of the cell of every kernel that reads a paged cache
(`op_signatures.PAGE_AXIS` in `cell_dims`), and which kernels those are is
derived from the signature contract — an operand with `role="kv"` — not
listed. Being a declared axis is what lets one value reach the job's dims,
its physical key and the row a measurement writes: the page used to be
folded into the shape after the projection, so the physical key dropped it
again and the measured row was refused at persist for an axis no signature
declared.
"""
from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.framework_bindings import get_binding
from backend.inference.op_signatures import (
    PAGE_AXIS, all_signatures, get_signature)

MODEL = "meta-llama/Llama-3.2-3B"


def _jobs(*, phase="decode", binding=None, kv_block_sizes=None):
    """The plan for one small paged model, keyed under `binding` (vLLM's when
    None) at `kv_block_sizes` (the binding's page when None)."""
    from tests.snapshots import resolve_snapshot
    from kernel_bench.plan import build_jobs
    return build_jobs(
        avg_output_tokens=0, model_id=MODEL, batch_size=1, seq_len=512,
        phase=phase, resolved=resolve_snapshot(MODEL),
        binding=binding if binding is not None else get_binding("vllm"),
        kv_block_sizes=kv_block_sizes)


def _paged(jobs):
    return [j for j in jobs
            if get_signature(j["signature"]).reads_paged_cache]


class TestWhichKernelsPageTheirCacheIsDerived(unittest.TestCase):
    def test_a_kv_operand_is_what_makes_a_signature_paged(self):
        for sig in all_signatures():
            reads_kv = any(t.role == "kv"
                           for t in (sig.inputs + sig.variant_inputs))
            self.assertEqual(
                sig.reads_paged_cache, reads_kv,
                f"{sig.name}: paged-ness must follow the contract's kv "
                f"operand, not a list")

    def test_the_lightning_indexer_is_included(self):
        """What the list this replaced left out. DSA's indexer reads the
        same paged K as the attention it feeds, so its kernel has the same
        dependence on the page — and it had no page in its key at all."""
        sig = get_signature("indexer_logits")
        self.assertIsNotNone(sig)
        self.assertTrue(sig.reads_paged_cache)

    def test_the_page_is_an_axis_of_exactly_the_paged_cells(self):
        """Declared once, by the kv operand: every signature that reads a
        paged cache has the page among its cell axes (and so its identity),
        and no other signature has it."""
        for sig in all_signatures():
            with self.subTest(signature=sig.name):
                self.assertEqual(PAGE_AXIS in sig.cell_dims,
                                 sig.reads_paged_cache)
                self.assertEqual(PAGE_AXIS in sig.identity_dims,
                                 sig.reads_paged_cache)
        for name in ("gemm", "rmsnorm", "silu_and_mul", "moe_permute"):
            self.assertNotIn(PAGE_AXIS, get_signature(name).cell_dims, name)

    def test_a_row_may_name_the_page_because_a_signature_declares_it(self):
        """The measured row's vocabulary is the signatures' axes, so the
        page is allowed in a `shape_json` without being a symbol a model
        spec may bind."""
        from backend.arch import dims
        from backend.inference.benchmarks import SIGNATURE_AXES
        self.assertIn(PAGE_AXIS, SIGNATURE_AXES)
        self.assertNotIn(PAGE_AXIS, dims.base_symbols())


class TestThePageIsInTheCell(unittest.TestCase):
    """One declaration, read by every consumer: the job's dims, its
    physical key and the row a measurement writes carry the same page."""

    def test_a_measured_paged_row_survives_persist(self):
        """The bug: the measured row's `shape_json` carried the page, the
        validator allowed no such key, and `cli._persist` dropped every
        paged-attention row it measured."""
        from backend.inference.benchmarks import validate_row
        from kernel_bench.emit import build_row
        from kernel_bench.env import Env
        from kernel_bench.invocation import Invocation
        from kernel_bench.timing import LatencyStats
        jobs, _dropped = _jobs()
        paged = _paged(jobs)
        self.assertTrue(paged)
        provider = type("P", (), {
            "library": "flashinfer", "kernel_impl": "k",
            "computes_at": None,          # computes at the cell's precision
            "actual_computes_at": lambda self, inv: {},
            "version": lambda self, env: "1.0",
            "impl_id": lambda self, inv: "k"})()
        env = Env("2.11", None, "H100", "H100-SXM", 3350.0, sm="sm_90")
        for job in paged:
            inv = Invocation.from_job(job)
            row = build_row(inv, provider, LatencyStats(5.0, 5.0, 6.0, 100, "l2_drain"),
                            env, date="2026-10-17", contributor="@t")
            self.assertEqual(json.loads(row["shape_json"])[PAGE_AXIS], 16)
            validate_row(row)

    def test_the_job_and_its_key_carry_the_bindings_page(self):
        jobs, _dropped = _jobs()
        paged = _paged(jobs)
        self.assertTrue(paged)
        self.assertEqual(get_binding("vllm").kv_block_size, 16)
        for job in paged:
            self.assertEqual(job["dims"][PAGE_AXIS], 16)
            self.assertEqual(json.loads(job["physical_key"][1])[PAGE_AXIS], 16)
            # One statement of it: the provider reads the cell's axis.
            from kernel_bench.invocation import Invocation
            self.assertEqual(Invocation.from_job(job).kv_block_size, 16)
            self.assertNotIn(PAGE_AXIS, job["kernel_params"])

    def test_cells_that_read_no_paged_cache_carry_no_page(self):
        jobs, _dropped = _jobs()
        unpaged = [j for j in jobs if j not in _paged(jobs)]
        self.assertTrue(unpaged)
        for job in unpaged:
            self.assertNotIn(PAGE_AXIS, job["dims"], job["signature"])
            self.assertNotIn(PAGE_AXIS, json.loads(job["physical_key"][1]))

    def test_another_page_on_the_binding_is_another_cell(self):
        at16, _ = _jobs()
        at64, _ = _jobs(binding=replace(get_binding("vllm"), kv_block_size=64))
        keys16 = {tuple(j["physical_key"]) for j in _paged(at16)}
        keys64 = {tuple(j["physical_key"]) for j in _paged(at64)}
        self.assertTrue(keys16 and keys64)
        self.assertFalse(keys16 & keys64, "two pages are two cells")
        self.assertEqual({j["dims"][PAGE_AXIS] for j in _paged(at64)}, {64})
        # The cells that read no paged cache do not move with the page.
        self.assertEqual(
            sorted(tuple(j["physical_key"]) for j in at16 if j not in _paged(at16)),
            sorted(tuple(j["physical_key"]) for j in at64 if j not in _paged(at64)))

    def test_the_plan_fans_a_paged_cell_out_over_the_pages_it_is_asked(self):
        # Every page some backend of this kernel serves becomes its own cell,
        # and a page none serves becomes none. 128 is vLLM-Ascend's
        # (`providers.ascend`, the page its plugin allocates the cache in);
        # 32 is served too but is not asked here, so it is absent for the
        # other reason — the fan-out is over the pages ASKED, intersected
        # with the pages served.
        jobs, _dropped = _jobs(kv_block_sizes=(16, 64, 128))
        paged = _paged(jobs)
        self.assertEqual(sorted(j["dims"][PAGE_AXIS] for j in paged),
                         [16, 64, 128])
        self.assertEqual(len({tuple(j["physical_key"]) for j in paged}), 3,
                         "two pages used to share one physical key")
        # A page nobody serves is still dropped, which is what keeps the
        # intersection meaningful rather than "whatever was asked".
        unserved, _ = _jobs(kv_block_sizes=(16, 63))
        self.assertEqual(sorted(j["dims"][PAGE_AXIS] for j in _paged(unserved)),
                         [16])
        unpaged = [j for j in jobs if j not in paged]
        self.assertEqual(
            len(unpaged), len({(tuple(j["physical_key"]), j["node_spec_id"])
                               for j in unpaged}),
            "a cell that reads no paged cache is one job, not one per page")


class TestThePagedProvidersReadTheCellsPage(unittest.TestCase):
    """The backends that build a paged cache take the page off the cell
    (`Invocation.dims`), so the page they run is the page the row is keyed
    and written under. A page left only in `kernel_params` (the old route)
    is no page at all: they decline."""

    #: The providers whose launch builds a paged cache at the cell's page
    #: (each resolves it with `resolve_kv_block_size` and names it in its
    #: `impl_id`).
    PAGED = (("flashinfer_paged_decode", "gqa_attention", 16),
             ("flashmla_decode", "mla_attention", 64),
             ("vllm_flashmla_decode_fp8", "mla_attention", 64),
             ("flashinfer_mla_decode", "mla_attention", 64),
             ("cutlass_mla_decode", "mla_attention", 128))

    @staticmethod
    def _cell(signature, dims, **over):
        from kernel_bench.invocation import Invocation
        base = ({"B": 1, "s_query": 1, "s_kv": 128, "n_heads": 16,
                 "n_kv_heads": 2, "d_head": 128}
                if signature == "gqa_attention" else
                {"B": 1, "s_query": 1, "s_kv": 128, "n_heads": 16,
                 "d_head": 576, "d_v_head": 512})
        return Invocation(
            op=signature, signature=signature, phase="decode",
            dims={**base, **dims},
            dtype_env={"activation": "bf16", "kv": "bf16"}, tp=1, **over)

    def test_the_page_comes_from_the_cells_dims(self):
        from kernel_bench.providers import all_providers
        by_impl = {p.kernel_impl: p for p in all_providers()}
        for impl, signature, page in self.PAGED:
            with self.subTest(provider=impl):
                provider = by_impl[impl]
                self.assertIn(page, provider.serves.kv_block_sizes)
                inv = self._cell(signature, {PAGE_AXIS: page})
                self.assertEqual(inv.kv_block_size, page)
                self.assertIn(f"_b{page}", provider.impl_id(inv))
                stale = self._cell(signature, {},
                                   kernel_params={PAGE_AXIS: page})
                self.assertIsNone(stale.kv_block_size)
                with self.assertRaises(NotImplementedError):
                    provider.impl_id(stale)


class TestTheFrameworkOwnsTheDefault(unittest.TestCase):
    def test_every_binding_FILE_declares_the_page(self):
        """Declared, not defaulted — and the declaration is in the yaml.

        Asserting the dataclass field would have passed on a code-level
        default, which is what it did: sglang declares no page and the
        field answered 16 for it. So this reads the binding FILES.
        """
        import yaml
        files = sorted((PROJECT_ROOT / "backend" / "inference"
                        / "framework_bindings").glob("*.yaml"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(binding=path.name):
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIn(
                    "kv_block_size", raw,
                    f"{path.name}: the paged-KV page is part of the key of "
                    f"every kernel that reads the cache, so the binding "
                    f"states it — a positive int, or `null` for a framework "
                    f"that resolves it per backend")
                value = raw["kv_block_size"]
                if value is not None:
                    self.assertIsInstance(value, int, path.name)
                    self.assertGreater(value, 0, path.name)

    def test_a_missing_declaration_is_rejected_not_defaulted(self):
        from backend.inference.framework_bindings import (
            _load_framework_bindings_dir)
        import pathlib
        import tempfile
        src = (PROJECT_ROOT / "backend" / "inference" / "framework_bindings"
               / "vllm.yaml").read_text(encoding="utf-8")
        stripped = "\n".join(line for line in src.splitlines()
                              if not line.startswith("kv_block_size:"))
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "vllm.yaml").write_text(stripped, encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                _load_framework_bindings_dir(tmp)
            self.assertIn("kv_block_size", str(caught.exception))

    def test_every_binding_FILE_declares_the_scan_chunk(self):
        """The Mamba-2 scan chunk is the same kind of fact as the page: the
        kernel takes it as `chunk_size=`, so the framework chooses it, and
        it is an axis of the `mamba2_scan` cell. The framework reads it off
        the checkpoint config, so each file declares WHERE — `{from_config:
        [<field>, ...]}` — and a file that leaves it out, or states a number
        in its place, is refused rather than defaulted."""
        import pathlib
        import tempfile
        import yaml
        from backend.inference.framework_bindings import (
            _load_framework_bindings_dir)
        files = sorted((PROJECT_ROOT / "backend" / "inference"
                        / "framework_bindings").glob("*.yaml"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(binding=path.name):
                text = path.read_text(encoding="utf-8")
                raw = yaml.safe_load(text)
                self.assertEqual(set(raw.get("ssm_chunk_size")),
                                 {"from_config"}, path.name)
                lines = text.splitlines()
                at = lines.index("ssm_chunk_size:")
                block = lines[at:at + 2]
                for broken in ([], ["ssm_chunk_size: 128"],
                               ["ssm_chunk_size:", "  from_config: []"]):
                    patched = "\n".join(lines[:at] + broken + lines[at + 2:])
                    with tempfile.TemporaryDirectory() as tmp:
                        pathlib.Path(tmp, path.name).write_text(
                            patched, encoding="utf-8")
                        with self.assertRaises(ValueError) as caught:
                            _load_framework_bindings_dir(tmp)
                        self.assertIn("ssm_chunk_size", str(caught.exception))
                self.assertTrue(block[1].lstrip().startswith("from_config:"))

    def test_the_audited_scan_chunk_fields(self):
        """Where each shipped framework reads the chunk, as audited: vLLM
        `ModelConfig.get_mamba_chunk_size`, sglang's config classes —
        `mamba_chunk_size`, then `chunk_size`. The value is the checkpoint
        config's, read by eval at those fields (`Program.scan_chunk_size`),
        and a config that states neither has none."""
        import copy
        from backend.arch.eval.program import Program
        from tests.snapshots import resolve_snapshot
        nano = resolve_snapshot("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        for framework in ("vllm", "sglang"):
            fields = get_binding(framework).ssm_chunk_size_from
            self.assertEqual(fields, ("mamba_chunk_size", "chunk_size"),
                             framework)
            for stated, chunk in (({"chunk_size": 128}, 128),
                                  ({"chunk_size": 256}, 256),
                                  ({"mamba_chunk_size": 64,
                                    "chunk_size": 256}, 64),
                                  ({}, None)):
                resolved = copy.copy(nano)
                resolved.config = {
                    k: v for k, v in nano.config.items()
                    if k not in ("chunk_size", "mamba_chunk_size")} | stated
                self.assertEqual(Program.of(resolved).scan_chunk_size(fields),
                                 chunk, (framework, stated))
            resolved = copy.copy(nano)
            resolved.config = dict(nano.config, chunk_size="128")
            with self.assertRaises(ValueError):
                Program.of(resolved).scan_chunk_size(fields)

    def test_a_framework_that_declares_no_page_has_no_paged_cell(self):
        """sglang resolves the page per attention backend, so it has none to
        declare. A paged kernel has no cell without its page, so under sglang
        (with no page on the request) the paged cells have no key: the plan
        drops them and says why, rather than keying them on a guessed 16 or
        on the unpaged cell. Every other cell is keyed as under vLLM."""
        from backend.inference.calibration_overlay import (
            deployment_kv_block_size)
        self.assertIsNone(get_binding("sglang").kv_block_size)
        self.assertIsNone(deployment_kv_block_size(None))
        self.assertIsNone(deployment_kv_block_size(get_binding("sglang")))
        jobs, dropped = _jobs(binding=get_binding("sglang"))
        self.assertEqual(_paged(jobs), [])
        self.assertTrue(any("reads a paged cache" in note
                            and "kv_block_size: null" in note
                            for note in dropped), dropped)
        vllm_jobs, _ = _jobs()
        self.assertEqual(
            sorted(tuple(j["physical_key"]) for j in jobs),
            sorted(tuple(j["physical_key"]) for j in vllm_jobs
                   if j not in _paged(vllm_jobs)))
        # A request that names a page gives sglang its paged cells back.
        named, _ = _jobs(binding=get_binding("sglang"), kv_block_sizes=(64,))
        self.assertEqual({j["dims"][PAGE_AXIS] for j in _paged(named)}, {64})

    def test_vllm_declares_its_own_default(self):
        self.assertEqual(get_binding("vllm").kv_block_size, 16)

    def test_the_request_overrides_the_binding(self):
        from backend.inference.calibration_overlay import (
            build_deployment_context, deployment_kv_block_size)
        binding = get_binding("vllm")
        self.assertEqual(deployment_kv_block_size(binding), 16)
        self.assertEqual(deployment_kv_block_size(binding, 128), 128)
        # THEORY has no binding and runs no kernels, so there is no page —
        # and no code-level 16 standing in for one.
        self.assertIsNone(deployment_kv_block_size(None))
        ctx = build_deployment_context(
            framework="vllm", framework_binding=binding, parallel=None,
            gpu_sku="H100-SXM", phase="decode", kv_block_size=64)
        self.assertEqual(ctx.kv_block_size, 64)


@pytest.mark.regression
class TestOnePlanHasNoCollidingKeys(unittest.TestCase):
    """The end of the collision, asserted on a real plan: the three pages
    are three findable cells."""

    def test_no_two_jobs_share_a_key(self):
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from kernel_bench.plan import build_jobs
        from kernel_bench.providers import served_kv_block_sizes
        for model_id in ("meta-llama/Llama-3.2-3B", "deepseek-ai/DeepSeek-V3.2"):
            config = fetch_config(model_id)
            resolved = SpecLoader().match(config, model_id=model_id)
            jobs, _dropped = build_jobs(avg_output_tokens=0, 
                model_id=model_id, batch_size=1, seq_len=4096, phase="decode",
                resolved=resolved, kv_block_sizes=served_kv_block_sizes())
            seen = {}
            for job in jobs:
                # The physical key, which is what a measured row is stored
                # and looked up under; it used to drop the page.
                key = (job["node_spec_id"], tuple(job["physical_key"]))
                self.assertNotIn(
                    key, seen,
                    f"{model_id}: two jobs share one measurement key — "
                    f"{job['signature']} at page "
                    f"{job['dims'].get(PAGE_AXIS)} and "
                    f"{seen.get(key)}")
                seen[key] = job["dims"].get(PAGE_AXIS)
            paged = [j for j in jobs if "kv_block_size" in j["dims"]]
            self.assertTrue(
                paged, f"{model_id}: no cell carries a page in its key")


if __name__ == "__main__":
    unittest.main()
