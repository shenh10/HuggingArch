import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.analyzer import analyze_model, explain_config_fetch_failure, fetch_config
from backend.hf.task_gate import HubTask


def _diffusers_repo(task):
    """What the hub says about a diffusion repo: a page task, the
    diffusers library, no Transformers resolution."""
    return HubTask(page_tag=task, library="diffusers", resolved=False,
                   resolved_pipeline=None, registered=frozenset(),
                   architectures=(), files=frozenset())


class TestConfigFetchFailureMessages(unittest.TestCase):
    """`fetch_config` keeps its None contract; the failure REASON is classified
    on the failure path by `explain_config_fetch_failure`, and `analyze_model`
    surfaces that reason verbatim as the user-facing error."""

    MODEL = "some-org/this-model-does-not-exist-anywhere"

    def setUp(self):
        # Hermetic: the no-config branch of the classifier now asks the
        # task gate; these tests exercise the classifier's own words, so
        # the gate is pinned to "task unknown" (its pass-through answer).
        gate = mock.patch(
            "backend.hf.task_gate.fetch_hub_task", return_value=None
        )
        gate.start()
        self.addCleanup(gate.stop)

    @classmethod
    def _hf_error(cls, kind):
        import httpx2  # the hub's HTTP client; its errors subclass httpx2.HTTPError
        # HfHubHTTPError.__init__ reads response.headers and .request; httpx2
        # only exposes .request when the Response was built with one.
        # EntryNotFoundError is a plain Exception — no response kwarg.
        if issubclass(kind, httpx2.HTTPError):
            resp = httpx2.Response(404, request=httpx2.Request("GET", "https://hf.co"))
            return kind("probe", response=resp)
        return kind("probe")

    def _probe_raises(self, exc):
        """Both functions import hf_hub_download at call time."""
        return mock.patch("huggingface_hub.hf_hub_download",
                          side_effect=exc)

    def test_repo_not_found_says_check_the_id(self):
        from huggingface_hub import errors as hf_errors
        with self._probe_raises(self._hf_error(hf_errors.RepositoryNotFoundError)):
            self.assertIsNone(fetch_config(self.MODEL))
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("does not exist on Hugging Face", msg)
        self.assertIn(self.MODEL, msg)

    def test_gated_repo_is_distinguished_from_missing(self):
        from huggingface_hub import errors as hf_errors
        with self._probe_raises(self._hf_error(hf_errors.GatedRepoError)):
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("gated", msg)
        self.assertNotIn("does not exist", msg)

    def test_missing_config_json_is_distinguished_from_missing_repo(self):
        from huggingface_hub import errors as hf_errors
        with self._probe_raises(self._hf_error(hf_errors.EntryNotFoundError)):
            # EntryNotFoundError is not an OSError; it used to escape as a 500.
            self.assertIsNone(fetch_config(self.MODEL))
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("no config.json", msg)

    def test_no_config_on_wrong_task_names_the_task(self):
        """A repo like MiniMax-H3 (Image-Text-to-Video, no root
        config.json) must say WHY it is out of scope, not just "no
        config.json"."""
        from huggingface_hub import errors as hf_errors
        gate = mock.patch(
            "backend.hf.task_gate.fetch_hub_task",
            return_value=_diffusers_repo("image-text-to-video"),
        )
        gate.start()
        self.addCleanup(gate.stop)
        with self._probe_raises(self._hf_error(hf_errors.EntryNotFoundError)):
            msg = explain_config_fetch_failure("MiniMaxAI/MiniMax-H3")
        self.assertIn("Image-Text-to-Video model", msg)
        self.assertIn("text-generation and image-text-to-text models only", msg)

    def test_invalid_repo_id_is_distinguished_from_network_failure(self):
        from huggingface_hub import errors as hf_errors
        with self._probe_raises(hf_errors.HFValidationError("bad id")):
            self.assertIsNone(fetch_config(self.MODEL))
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("not a valid Hugging Face repo id", msg)
        self.assertIn("owner/name", msg)

    def test_network_failure_says_retry(self):
        with self._probe_raises(OSError("connection reset")):
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("Could not reach Hugging Face", msg)

    def test_a_hub_client_error_is_a_network_failure(self):
        """hf_hub_download wraps connect / timeout into an OSError, but a
        proxy or dropped-stream failure leaves as the hub client's own
        httpx2 error: `fetch_config` keeps its None contract and the
        classifier says retry, not "unknown"."""
        import httpx2
        with self._probe_raises(httpx2.ProxyError("proxy down")):
            self.assertIsNone(fetch_config(self.MODEL))
            msg = explain_config_fetch_failure(self.MODEL)
        self.assertIn("Could not reach Hugging Face", msg)

    def test_analyze_model_surfaces_the_classified_reason(self):
        from huggingface_hub import errors as hf_errors
        with self._probe_raises(self._hf_error(hf_errors.RepositoryNotFoundError)):
            result = analyze_model(model_id=self.MODEL)
        self.assertFalse(result["success"])
        self.assertIn("does not exist on Hugging Face", result["error"])

    def test_no_spec_path_refuses_wrong_task_before_offering_generation(self):
        """A readable config on an out-of-scope task (a diffusion repo
        can ship a root config.json) never matches a spec — refuse with
        the task instead of pointing at /generate-spec."""
        gate = mock.patch(
            "backend.hf.task_gate.fetch_hub_task",
            return_value=_diffusers_repo("text-to-video"),
        )
        gate.start()
        self.addCleanup(gate.stop)
        cfg = {"model_type": "nothing-any-spec-knows", "torch_dtype": "bfloat16", "hidden_size": 8}
        with mock.patch("backend.analyzer.fetch_config", return_value=cfg), \
             mock.patch("backend.analyzer.fetch_metadata", return_value={}), \
             self.assertRaises(ValueError) as raised:
            analyze_model(model_id="org/video-model")
        self.assertIn("Text-to-Video model", str(raised.exception))
        self.assertIn("text-generation and image-text-to-text", str(raised.exception))


if __name__ == "__main__":
    unittest.main()


# -- the headline weight total ----------------------------------------------

import pytest

# (model, total params, active params, parameter bytes): counts and the
# parameters-only byte figure, which counting scales in the weight total
# leaves alone.
_HEADLINE_MODELS = [
    # The MTP head's own copy of the vocabulary table is activated by the
    # row a token reads, not whole (`Evaluator._active_params`): 926,671,872
    # fewer than counting it whole.
    ("deepseek-ai/DeepSeek-V3", 684_489_845_504, 39_167_084_288,
     688_408_677_376),                                            # FP8
    ("moonshotai/Kimi-K2-Thinking", 1_026_408_232_448, 32_861_500_928,
     530_787_475_456),                                            # INT4
    ("Qwen/Qwen3-8B", 8_190_735_360, 8_190_735_360,
     16_381_470_720),                                             # BF16
]


def _node_bytes(nodes):
    return sum(int(n.get("weight_bytes") or 0)
               + sum(int(a.get("weight_bytes") or 0)
                     for a in (n.get("absorbed_params") or []))
               for n in nodes)


@pytest.mark.parametrize("model_id,params,active,param_bytes", _HEADLINE_MODELS)
def test_headline_weight_bytes_are_what_the_nodes_store(
        model_id, params, active, param_bytes, monkeypatch):
    """The Inspector's weight total is the weight ledger's `resident` —
    parameters, their quantisation scales and buffers — which is what the
    drawn nodes' weight_bytes add up to. On a quantized checkpoint it used
    to be the parameters alone and fell short of the per-op figures on the
    same page by every scale."""
    import backend.analyzer as analyzer
    from backend.api._models import AnalyzeRequest
    from backend.api.routes.analyze import _build_dashboard
    from backend.arch.eval import capacity
    from backend.arch.eval.projection import project_model
    from backend.rendering.builders import build_block_graph_data
    from tests.snapshots import resolve_snapshot, snapshot_config

    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    dashboard = _build_dashboard(summary, AnalyzeRequest(model_id=model_id))

    model = project_model(resolved, batch=1, input_tokens=1)
    ledger = capacity.weights(model.analysis)
    per_node = sum(held.resident for held in ledger.by_uid.values())
    assert summary["weight_bytes"] == per_node == ledger.resident
    assert dashboard["weight_bytes"] == ledger.resident

    # The drawn blocks are those nodes: each template's nodes, once per
    # layer it runs as, are the ledger's bytes for the blocks.
    graph, _ = build_block_graph_data(resolved, 1, 1)
    layers = {}
    for template, count, _report in model.block_results:
        layers[template] = layers.get(template, 0) + count
    drawn = sum(_node_bytes(graph[t]["nodes"]) * layers[t] for t in graph)
    by_block = ledger.by_block(model.analysis.program)
    assert drawn == sum(v for k, v in by_block.items() if k is not None)

    # Counts, and the parameters-only bytes the checkpoint header is
    # compared against, do not move.
    assert summary["total_params"] == params
    assert summary["total_active_params"] == active
    assert summary["param_bytes"] == dashboard["param_bytes"] == param_bytes


@pytest.mark.parametrize("model_id,params,active,param_bytes", _HEADLINE_MODELS)
def test_checkpoint_columns_follow_the_header_split(
        model_id, params, active, param_bytes, monkeypatch):
    """Weights and serialised buffers vs the header's non-sidecar tensors;
    metadata vs its quant-sidecar bucket; all tensors vs their sum. The
    Inspector rows are these three diffs, not a single weight-only Struct
    Diff."""
    import backend.analyzer as analyzer
    from backend.api._models import AnalyzeRequest
    from backend.api.routes.analyze import _build_dashboard
    from backend.arch.eval.checkpoint import Reconciliation
    from backend.arch.eval.projection import project_model
    from backend.hf.metadata import fetch_metadata
    from tests.snapshots import resolve_snapshot, snapshot_config

    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved,
        fetch_metadata(model_id), 1, 1, "bfloat16",
    )["summary"]
    dashboard = _build_dashboard(summary, AnalyzeRequest(model_id=model_id))
    recon = Reconciliation.of(
        project_model(resolved, batch=1, input_tokens=1).analysis,
        fetch_metadata(model_id), model_id)

    spec_meta = int(recon.metadata_bytes)
    spec_all = int(recon.total)
    assert summary["spec_metadata_bytes"] == spec_meta
    assert summary["spec_checkpoint_bytes"] == spec_all
    assert summary["metadata_storage_diff"] == (
        int(recon.header_metadata_bytes) - spec_meta)
    assert summary["all_tensor_diff"] == int(recon.header_total) - spec_all
    assert dashboard["metadata_storage_diff"] == summary["metadata_storage_diff"]
    assert dashboard["all_tensor_diff"] == summary["all_tensor_diff"]
    if recon.is_anchored:
        assert summary["checkpoint_anchored"] is True
        assert summary["hf_index_total_size"] == int(recon.measured_total)


class TestDerivedGroupLayers(unittest.TestCase):
    """The Inspector's per-layer entries of a derived group carry the group
    (`BlockMeta.derived_group`) so the page can bucket them under their
    declaration; every other entry is as before."""

    def test_each_layer_names_its_declaration_and_history(self):
        from backend.analyzer import _build_layer_groups
        from backend.arch.eval.program import Program
        from backend.arch.eval.projection.result import project_analysis
        from backend.arch.eval.transforms import Deployment
        from tests.backend.arch.test_prefix_streams import _resolved

        resolved = _resolved("decl_layers_toy")
        analysis = Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=4).analyze()
        groups = _build_layer_groups(project_analysis(analysis), resolved, 1, 4)
        blocks = [g for g in groups if g["layer_type"] == "transformer_block"]
        self.assertEqual(sorted(g["block_index"] for g in blocks), list(range(7)))
        seen = {}
        for g in blocks:
            d = g.get("derived_group")
            if g["template"] == "first":
                self.assertIsNone(d)
                continue
            self.assertEqual(d["base_template"], g["template"])
            self.assertIn(g["block_index"], d["layer_indices"])
            seen[g["block_index"]] = (d["declaration"]["base_template"],
                                      d["visible"])
        self.assertEqual(seen, {1: ("plain", 1), 2: ("plain", 1),
                                3: ("start", 1), 4: ("plain", 2),
                                5: ("plain", 2), 6: ("start", 2)})


# (model, main-model lookup tables): the MTP head's vocabulary table is the
# aux stack's, not a lookup table of the main model.
@pytest.mark.parametrize("model_id,main_lookup", [
    ("deepseek-ai/DeepSeek-V3", 0),     # MTP head with its own table
    ("tencent/HunyuanOCR", 0),          # vision encoder
    ("Qwen/Qwen3-8B", 0),
])
def test_the_breakdown_under_the_headline_sums_to_it(model_id, main_lookup,
                                                      monkeypatch):
    """Embeddings + lookup tables + transformer is the main model the
    headline states; MTP heads and encoders are their own rows. It summed to
    the checkpoint instead (684B under DeepSeek-V3's 671B), and listed the
    MTP head's table among the main model's lookup tables."""
    import backend.analyzer as analyzer
    from backend.api._models import AnalyzeRequest
    from backend.api.routes.analyze import _build_dashboard
    from tests.snapshots import resolve_snapshot, snapshot_config

    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    dash = _build_dashboard(summary, AnalyzeRequest(model_id=model_id))
    parts = (dash["embedding_params"] + (dash["lookup_table_params"] or 0)
             + dash["non_embedding_params"])
    assert parts == summary["main_total_params"]
    assert dash["non_embedding_params"] > 0
    assert (dash["lookup_table_params"] or 0) == main_lookup


def test_the_quant_methods_are_ordered_by_the_bytes_they_cover(monkeypatch):
    """DeepSeek-V4-Pro's experts are MXFP4 (`mxfp4-dot-suffix`, declared at
    its `per_role.ffn_moe` since 2d17bec2c) and hold most of the weights; its
    attention is FP8. `weight_methods` (what the quant badge reads) says so
    from the weight join, whatever order `per_role` lists them in."""
    import backend.analyzer as analyzer
    from tests.snapshots import resolve_snapshot, snapshot_config

    model_id = "deepseek-ai/DeepSeek-V4-Pro"
    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    methods = [row["method"] for row in summary["weight_methods"]]
    assert methods[:2] == ["mxfp4-dot-suffix", "fp8_ue8m0"]


def test_quant_metadata_bytes_sum_each_scale_at_its_own_width(monkeypatch):
    """MiMo-V2.6-Pro-RL stores 1-byte ue8m0 scales beside 4-byte F32 ones.
    The published metadata bytes are each tensor's own bytes summed per
    dtype (`iter_checkpoint`), not the scale count times the most common
    scale's width, which priced the F32 scales at one byte."""
    import backend.analyzer as analyzer
    from backend.api._models import AnalyzeRequest
    from backend.api.routes.analyze import _build_dashboard
    from tests.snapshots import resolve_snapshot, snapshot_config

    model_id = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    formats = {row["dtype"]: row["bytes"] for row in
               summary["quant_metadata_formats"]}
    assert set(formats) == {"F8_E8M0FNU", "F32"}
    assert summary["quant_metadata_bytes"] == sum(formats.values())
    # The same scales the count holds, each at its width.
    assert (formats["F8_E8M0FNU"] + formats["F32"] // 4
            == summary["quant_metadata_params"])
    dashboard = _build_dashboard(summary, AnalyzeRequest(model_id=model_id))
    assert dashboard["quant_metadata_bytes"] == summary["quant_metadata_bytes"]
    # The deprecated per-scale width has no single answer here: null.
    assert summary["quant_metadata_bytes_per_scale"] is None
    assert dashboard["quant_metadata_bytes_per_scale"] is None


def test_the_deprecated_per_scale_width_is_the_one_dtype_width(monkeypatch):
    """DeepSeek-V3's metadata is F32 alone: the compatibility field is that
    dtype's width, matching its bytes over its scale count."""
    import backend.analyzer as analyzer
    from tests.snapshots import resolve_snapshot, snapshot_config

    model_id = "deepseek-ai/DeepSeek-V3"
    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    assert [row["dtype"] for row in summary["quant_metadata_formats"]] == ["F32"]
    assert summary["quant_metadata_bytes_per_scale"] == 4
    assert (summary["quant_metadata_bytes"]
            == 4 * summary["quant_metadata_params"])


# (model, the tied copies its checkpoint stores): HunyuanOCR ties lm_head to
# the embedding and still ships lm_head.weight; Qwen3-8B unties them.
@pytest.mark.parametrize("model_id,copies", [
    ("tencent/HunyuanOCR", [{"tensor": "lm_head.weight",
                              "bytes": 120818 * 1024 * 2}]),
    ("Qwen/Qwen3-8B", []),
])
def test_the_checkpoint_says_which_tied_copies_it_stores(
        model_id, copies, monkeypatch):
    """The checkpoint's bytes exceed the loaded weights by the tied copies it
    stores and serving skips; the page names them where the two disagree."""
    import backend.analyzer as analyzer
    from backend.api._models import AnalyzeRequest
    from backend.api.routes.analyze import _build_dashboard
    from tests.snapshots import resolve_snapshot, snapshot_config

    resolved = resolve_snapshot(model_id)
    monkeypatch.setattr(analyzer, "_fetch_model_str", lambda *a, **k: "")
    summary = analyzer._analyze_from_spec(
        model_id, snapshot_config(model_id), resolved, {}, 1, 1, "bfloat16",
    )["summary"]
    dash = _build_dashboard(summary, AnalyzeRequest(model_id=model_id))
    assert dash["checkpoint_copies"] == copies
    # Neither model stores scales or buffers: the copies are the whole gap.
    assert (summary["param_bytes"] - summary["weight_bytes"]
            == sum(c["bytes"] for c in copies))


def test_attention_families_are_the_cache_families_not_the_display():
    """A DSA layer displays as DSA and runs an MLA cache: the families list
    says MLA, once, in first-appearance order beside the display line."""
    from types import SimpleNamespace

    from backend.analyzer import _build_attention_info

    headline = SimpleNamespace(heads=8, kv_heads=1, head_dim=64, kv_lora_rank=512,
                               q_lora_rank=None, qk_rope_dim=64)
    groups = [
        {"attn_type": "MLA", "attn_display": "DSA", "layer_indices": [0, 2],
         "stored_count": 2},
        {"attn_type": "MLA", "attn_display": "MLA", "layer_indices": [1, 3],
         "stored_count": 2},
    ]
    info = _build_attention_info(headline, groups, 4, is_hybrid=True)
    assert info["attention_type"] == "DSA + MLA"
    assert info["attention_families"] == ["MLA"]
