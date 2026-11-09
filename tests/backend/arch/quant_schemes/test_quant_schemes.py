"""quant_scheme yaml schema validation.

Each yaml under `backend/arch/quant_schemes/` declares the
materialization recipe consumed by `eval.formats` and the HF-side
classifiers (`is_quantization_metadata`, `packed_unpack_factor`). This
file tests:

  - All committed schemes load with the required fields populated
  - The validators trip on malformed yamls (synthesized in temp dirs)
  - `resolve_scheme` returns the right scheme + num_bits per config
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent


# The repository root, four parents up (tests/backend/arch/quant_schemes).
# One level short once inserted `tests/` itself at sys.path[0], and any later
# `import scripts.…` resolved to `tests/scripts` — the publish tests' import
# failed only under full-tree collection, where this module imports first.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


def _try_load_with_yaml(yaml_text: str):
    """Write yaml to a temp `arch/quant_schemes/` dir, redirect
    the loader at it, attempt load. Returns the resulting dict or
    raises on validation failure.
    """
    import backend.arch.quant_schemes as mod
    with tempfile.TemporaryDirectory() as td:
        scheme_dir = os.path.join(td, "backend", "arch", "quant_schemes")
        os.makedirs(scheme_dir)
        with open(os.path.join(scheme_dir, "test.yaml"), "w") as f:
            f.write(yaml_text)
        orig_file = mod.__file__
        # The registry reads the yaml files beside its own `__init__.py`.
        mod.__file__ = os.path.join(scheme_dir, "__init__.py")
        try:
            return mod._load_quant_schemes()
        finally:
            mod.__file__ = orig_file


class TestSchemeRegistryLoad(unittest.TestCase):
    """Every committed scheme yaml must load with the materialization
    recipe fields populated.
    """

    # `fp8_ue8m0` joined the registry for DSv4-style FP8: same F8_E4M3
    # weights as `fp8` but `.scale` siblings are UE8M0 (1 byte/elem)
    # instead of F32 `.weight_scale_inv` (4 bytes/elem). Different
    # metadata layout → different on-disk byte math → must be a separate
    # scheme. Triggered only via `quant_context.per_role.method=fp8_ue8m0`
    # (HF config still reports `quant_method: fp8`, so it can't be
    # autodetected from there).
    # `mxfp4-pack-quantized` is MXFP4 emitted through compressed-tensors:
    # compressed-tensors NAMING (`.weight_packed` / `.weight_scale`) over MXFP4
    # PACKING (2 FP4 per U8, not the I32/factor-8 default). Its own file says
    # so and names Kimi-K3 as its client; Kimi-K3 reconciles byte-exact against
    # its checkpoint through it. The scheme shipped and this list did not move.
    # `mxfp4-dot-suffix` (2d17bec2c) is `fp4`'s dot-suffix layout field for
    # field with `numeric_format: mx_e2m1`: DeepSeek-V4's routed experts are
    # declared MXFP4 at their own `per_role.ffn_moe`, and `fp4` stays the
    # generic FP4 scheme for the same layout without that claim.
    EXPECTED_SCHEMES = frozenset({"awq", "compressed-tensors", "fp4", "fp8", "fp8_ue8m0",
                                  "gptq", "modelopt", "mxfp4",
                                  "mxfp4-dot-suffix", "mxfp4-pack-quantized",
                                  "nvfp4"})

    def setUp(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        self.schemes = QUANT_SCHEMES

    def test_all_expected_schemes_loaded(self):
        self.assertEqual(set(self.schemes), self.EXPECTED_SCHEMES)

    def test_each_scheme_has_format_variants(self):
        for name, s in self.schemes.items():
            with self.subTest(scheme=name):
                self.assertGreater(
                    len(s.format_variants), 0,
                    f"{name}: format_variants must be non-empty",
                )

    def test_each_scheme_has_metadata_siblings(self):
        for name, s in self.schemes.items():
            with self.subTest(scheme=name):
                self.assertGreater(
                    len(s.metadata_siblings), 0,
                    f"{name}: metadata_siblings must be non-empty — "
                    "every quant linear emits at least one scale sibling",
                )

    def test_each_scheme_declares_numeric_format(self):
        for name, s in self.schemes.items():
            with self.subTest(scheme=name):
                self.assertGreater(
                    len(s.numeric_format), 0,
                    f"{name}: numeric_format must be non-empty",
                )
                self.assertGreater(
                    len(s.numeric_format_resolver), 0,
                    f"{name}: numeric_format_resolver must be non-empty",
                )
                # Reachability already enforced by loader; double-check here.
                targets = {fmt for _, fmt in s.numeric_format_resolver}
                unreachable = set(s.numeric_format) - targets
                self.assertFalse(
                    unreachable,
                    f"{name}: numeric_format {unreachable} not reachable",
                )

    def test_packing_factors_in_allowed_set(self):
        from backend.arch.quant_schemes import _ALLOWED_PACKING_FACTORS
        for name, s in self.schemes.items():
            for vname, fv in s.format_variants.items():
                with self.subTest(scheme=name, variant=vname):
                    if fv.packing_factor is not None:
                        self.assertIn(
                            fv.packing_factor, _ALLOWED_PACKING_FACTORS,
                            f"{name}/{vname}: bad packing_factor",
                        )
                    if fv.packing_factor_per_num_bits:
                        for nb, pf in fv.packing_factor_per_num_bits.items():
                            self.assertIn(
                                pf, _ALLOWED_PACKING_FACTORS,
                                f"{name}/{vname}: pf_per_num_bits[{nb}]={pf}",
                            )

    def test_resolve_scheme_routes_by_quant_method(self):
        """`resolve_scheme` picks the right scheme yaml from
        `config.quantization_config.quant_method` and resolves
        `num_bits` via the scheme's `num_bits:` block."""
        from backend.arch.quant_schemes import resolve_scheme
        cfg_awq = {"quantization_config": {"quant_method": "awq", "bits": 4}}
        s, n, unk = resolve_scheme(cfg_awq)
        self.assertEqual(s.name, "awq")
        self.assertEqual(n, 4)
        self.assertIsNone(unk)

    def test_packing_is_the_selected_schemes_fact(self):
        """A checkpoint whose config names no known method has no scheme to
        say a tensor is packed: a `.qweight` / `_blocks` / `.weight_packed`
        name packs nothing, and no bit width is assumed. A selected scheme
        that packs by bit width and is given none it packs at refuses."""
        from backend.arch.quant_schemes import (
            QUANT_SCHEMES, packed_unpack_factor, resolve_scheme)
        for config in ({}, {"quantization_config": {"quant_method": "nonesuch"}}):
            scheme, num_bits, _ = resolve_scheme(config)
            self.assertEqual(num_bits, 0)
            for name in ("a.qweight", "a_blocks", "a.weight_packed", "a.weight"):
                self.assertEqual(
                    packed_unpack_factor(name, "I32", num_bits, scheme), 1)
        scheme, num_bits, _ = resolve_scheme(
            {"quantization_config": {"quant_method": "gptq"}})
        self.assertEqual(num_bits, 0)
        with self.assertRaises(ValueError):
            packed_unpack_factor("a.qweight", "I32", num_bits, scheme)
        self.assertEqual(packed_unpack_factor(
            "a.qweight", "I32", 4, QUANT_SCHEMES["gptq"]), 8)


class TestSchemeYamlValidation(unittest.TestCase):
    """Validators trip on malformed yamls."""

    def test_name_field_is_rejected(self):
        """The stem names the scheme and is the key it registers under.
        A `name:` line can only repeat the stem or contradict it, and a
        contradiction registers the scheme under a key that no
        `quant_method` lookup will ever produce a file for.
        """
        bad = dedent("""\
            name: something_else
            format_variants:
              default: { packing_factor: 1, storage_dtype: I8 }
            """)
        with self.assertRaisesRegex(ValueError, "declares `name:`"):
            _try_load_with_yaml(bad)

    def test_packing_factor_must_be_in_allowed_set(self):
        bad = dedent("""\
            format_variants:
              default: { packing_factor: 3, storage_dtype: I8 }
            """)
        with self.assertRaisesRegex(ValueError, "packing_factor=3 not in"):
            _try_load_with_yaml(bad)

    def test_packing_factor_or_per_num_bits_not_both(self):
        bad = dedent("""\
            format_variants:
              default:
                packing_factor: 4
                packing_factor_per_num_bits: { 4: 8 }
                storage_dtype: I32
            """)
        with self.assertRaisesRegex(ValueError, "EITHER `packing_factor` OR"):
            _try_load_with_yaml(bad)

    def test_storage_dtype_or_per_num_bits_not_both(self):
        bad = dedent("""\
            format_variants:
              default:
                packing_factor: 1
                storage_dtype: F8_E4M3
                storage_dtype_per_num_bits: { 8: F8_E4M3 }
            """)
        with self.assertRaisesRegex(ValueError, "EITHER `storage_dtype` OR"):
            _try_load_with_yaml(bad)

    def test_unreachable_numeric_format_rejected(self):
        bad = dedent("""\
            numeric_format: [int_sym, int_asym, fp_e2m1]
            numeric_format_resolver:
              - { when: "true", format: int_sym }
            """)
        with self.assertRaisesRegex(ValueError, "no matching numeric_format_resolver"):
            _try_load_with_yaml(bad)

    def test_metadata_sibling_requires_suffix(self):
        bad = dedent("""\
            metadata_siblings:
              - cardinality: "in * out"
            """)
        with self.assertRaisesRegex(ValueError, "non-empty `suffix:`"):
            _try_load_with_yaml(bad)

    def test_metadata_sibling_requires_one_of_cardinality_forms(self):
        # Both cardinality and cardinality_per_strategy → reject.
        bad = dedent("""\
            metadata_siblings:
              - suffix: ".scale"
                cardinality: "in * out"
                cardinality_per_strategy: { group: "in * out" }
            """)
        with self.assertRaisesRegex(ValueError, "set ONE of"):
            _try_load_with_yaml(bad)
        # Neither → reject.
        bad2 = dedent("""\
            metadata_siblings:
              - suffix: ".scale"
            """)
        with self.assertRaisesRegex(ValueError, r"declare a `shape:`"):
            _try_load_with_yaml(bad2)
        # Both shape forms → reject, for the same reason as both count forms.
        bad3 = dedent("""\
            metadata_siblings:
              - suffix: ".scale"
                shape: ["in", "out"]
                shape_per_strategy: { group: ["in", "out"] }
            """)
        with self.assertRaisesRegex(ValueError, r"set ONE of `shape:`"):
            _try_load_with_yaml(bad3)
        # One strategy declared as BOTH a shape and a count → reject. A shape
        # already determines the count, so the two can only disagree.
        bad4 = dedent("""\
            metadata_siblings:
              - suffix: ".scale"
                cardinality_per_strategy: { group: "in * out" }
                shape_per_strategy: { group: ["in", "out"] }
            """)
        with self.assertRaisesRegex(ValueError, r"both a shape and a count"):
            _try_load_with_yaml(bad4)

    def test_a_sibling_shape_and_a_count_agree_where_both_are_legal(self):
        """A shape is the declaration and the count is its product, so a
        scheme migrating one strategy at a time must not shift a single byte.
        `compressed-tensors` is mid-migration — `tile_2d` is a shape while
        `group` / `channel` / `tensor` are still counts — and this is what
        makes that state safe rather than merely tolerated."""
        from backend.arch.eval.scheme_dialect import _shape_product, eval_formula
        bind = {"in_": 576, "out": 7168, "block_h": 128, "block_w": 128}
        shape = ["(in_ + block_h - 1) // block_h",
                 "(out + block_w - 1) // block_w"]
        count = ("((in_ + block_h - 1) // block_h) "
                 "* ((out + block_w - 1) // block_w)")
        self.assertEqual(_shape_product(shape, bind),
                         eval_formula(count, bind))
        self.assertEqual(_shape_product(shape, bind), 5 * 56)

    def test_invalid_quantization_status_rejected(self):
        bad = dedent("""\
            valid_status: [compressed, totally_invented_status]
            """)
        with self.assertRaisesRegex(ValueError, "totally_invented_status"):
            _try_load_with_yaml(bad)

    def test_resolver_rule_requires_when_and_format(self):
        bad = dedent("""\
            numeric_format: [int_sym]
            numeric_format_resolver:
              - { when: "true" }
            """)
        with self.assertRaisesRegex(ValueError, "both `when:` .*and `format:`"):
            _try_load_with_yaml(bad)


class TestPerSchemeSemantics(unittest.TestCase):
    """Sanity checks on individual schemes' materialization recipe — guards
    against accidental edits that break the materialization recipe.
    """

    def test_fp8_is_unpacked_e4m3(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["fp8"]
        fv = s.format_variants["default"]
        self.assertEqual(fv.packing_factor, 1)
        self.assertEqual(fv.storage_dtype, "F8_E4M3")
        self.assertEqual(fv.weight_storage_suffix, ".weight")
        self.assertIn("block_h", s.required_fields)
        self.assertIn("block_w", s.required_fields)

    def test_mxfp4_packs_two_per_byte(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["mxfp4"]
        fv = s.format_variants["default"]
        self.assertEqual(fv.packing_factor, 2)
        self.assertEqual(fv.storage_dtype, "U8")
        self.assertEqual(fv.weight_storage_suffix, "_blocks")

    def test_fp4_packs_two_per_byte_with_ue8m0_scale(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["fp4"]
        fv = s.format_variants["default"]
        self.assertEqual(fv.packing_factor, 2)
        self.assertEqual(fv.storage_dtype, "I8")
        # The single sibling should be `.scale` with UE8M0 dtype.
        self.assertEqual(len(s.metadata_siblings), 1)
        sib = s.metadata_siblings[0]
        self.assertEqual(sib.suffix, ".scale")
        self.assertEqual(sib.dtype, "F8_E8M0FNU")

    def test_compressed_tensors_has_three_format_variants(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["compressed-tensors"]
        self.assertEqual(
            set(s.format_variants),
            {"pack-quantized", "dense", "float-quantized"},
        )
        # pack-quantized: per-num_bits packing factors
        pq = s.format_variants["pack-quantized"]
        self.assertEqual(pq.packing_factor_per_num_bits, {2: 16, 4: 8, 8: 4})
        self.assertEqual(pq.storage_dtype, "I32")
        # dense: packing=1, storage per-num_bits
        dense = s.format_variants["dense"]
        self.assertEqual(dense.packing_factor, 1)

    def test_compressed_tensors_zero_point_conditional_on_symmetric(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["compressed-tensors"]
        zp = next(m for m in s.metadata_siblings
                  if m.suffix == ".weight_zero_point")
        self.assertIn("symmetric == false", zp.when)

    def test_gptq_g_idx_conditional_on_desc_act(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["gptq"]
        g = next(m for m in s.metadata_siblings if m.suffix == ".g_idx")
        self.assertIn("desc_act == true", g.when)

    def test_awq_qzeros_conditional_on_zero_point(self):
        from backend.arch.quant_schemes import QUANT_SCHEMES
        s = QUANT_SCHEMES["awq"]
        qz = next(m for m in s.metadata_siblings if m.suffix == ".qzeros")
        self.assertIn("zero_point == true", qz.when)


if __name__ == "__main__":
    unittest.main()
