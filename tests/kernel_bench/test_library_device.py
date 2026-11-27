"""A library names the accelerator it runs on; the box names its own.

The two facts used to meet only at `_sm_ok`, which refuses a stated CUDA
compute-capability floor and otherwise admits. flashinfer states no floor,
so an Ascend box that listed a flashinfer env marked it available, and
the page union took FlashInfer's 16/32/64 into an NPU sweep.
"""
from __future__ import annotations

from kernel_bench.device import DEVICES
from kernel_bench.env import Env
from kernel_bench.providers import (
    all_libraries, all_providers, get_library, served_kv_block_sizes,
)


def test_flashinfer_declares_cuda():
    """The fact that was missing. Delete `device=` on flashinfer and this
    is the first test that goes red."""
    assert get_library("flashinfer").device == "cuda"


def test_vllm_ascend_declares_npu():
    assert get_library("vllm_ascend").device == "npu"


def test_torch_is_not_bound_to_one_accelerator():
    assert get_library("torch").device is None


def test_every_library_device_is_the_sheet_vocabulary():
    for lib in all_libraries():
        assert lib.device is None or lib.device in DEVICES, lib.name


def test_every_library_but_torch_names_a_device():
    unbound = [lib.name for lib in all_libraries() if lib.device is None]
    assert unbound == ["torch"], unbound


def test_an_unnamed_device_does_not_invent_a_skip():
    env = Env("2.5.0", None, None, None, None, libs={"flashinfer": "0.2.0"})
    p = next(p for p in all_providers()
             if p.kernel_impl == "flashinfer_paged_decode")
    assert p.available(env)


def test_flashinfer_is_unavailable_on_npu_even_when_listed():
    env = Env("2.5.0", "npu", "Ascend910C", "Ascend-910C-A3", 1600.0,
              libs={"flashinfer": "0.2.0"})
    p = next(p for p in all_providers()
             if p.kernel_impl == "flashinfer_paged_decode")
    why = next(p.unavailability(env))
    assert "device" in why
    assert not p.available(env)


def test_cuda_libraries_decline_an_npu_box_that_lists_them():
    npu = Env("2.5.0", "npu", "Ascend910C", "Ascend-910C-A3", 1600.0,
              libs={p.import_name: "1.0" for p in all_providers()})
    leaked = [p.kernel_impl for p in all_providers()
              if p.lib.device == "cuda" and p.available(npu)]
    assert leaked == []


def test_npu_libraries_decline_a_cuda_box_that_lists_them():
    cuda = Env("2.5.0", "cuda", "H100", "H100-SXM", 3350.0, sm="sm_90",
               libs={p.import_name: "1.0" for p in all_providers()})
    leaked = [p.kernel_impl for p in all_providers()
              if p.lib.device == "npu" and p.available(cuda)]
    assert leaked == []


def test_gqa_pages_on_ascend_are_only_the_npu_backends():
    pages = served_kv_block_sizes("gqa_attention", gpu="Ascend-910C-A3")
    assert pages == (128,)
    assert 16 not in pages


def test_gqa_pages_on_cuda_keep_the_cuda_set():
    pages = set(served_kv_block_sizes("gqa_attention", gpu="H100-SXM"))
    assert {16, 32, 64} <= pages


def test_mla_pages_on_cuda_do_not_shrink():
    pages = served_kv_block_sizes("mla_attention", gpu="H100-SXM")
    assert pages == (32, 64, 128)


def test_mla_pages_on_ascend_do_not_take_cuda_pages():
    """No NPU backend serves `mla_attention` yet. The unsigned union used
    to hand it FlashInfer's 16 and FlashMLA's 64, which is how an NPU
    sweep grew cells at pages this family never allocates."""
    pages = served_kv_block_sizes("mla_attention", gpu="Ascend-910C-A3")
    assert pages == ()
    assert {16, 32, 64}.isdisjoint(pages)


def test_read_and_write_sides_share_the_ascend_kernel_page():
    """The binding (read) and the provider constant (write) are one fact.

    Delete `device=` on flashinfer and `test_flashinfer_declares_cuda`
    goes red first; set either side to 640 and this one does. 640 is the
    hybrid scheduler alignment, not `PAGE_AXIS`.
    """
    from backend.inference.framework_bindings import get_binding
    from kernel_bench.providers.ascend import ASCEND_PAGE
    assert get_binding("vllm_ascend").kv_block_size == ASCEND_PAGE == 128
    for sig in ("gqa_attention", "mla_attention"):
        pages = served_kv_block_sizes(sig, gpu="Ascend-910C-A3")
        assert {640, 2176, 4352}.isdisjoint(pages), (sig, pages)
