"""How each measured library gets INSTALLED — one record per library.

These facts used to live on every kernel, so a library with 12 kernels stated
its dependency 12 times and the copies drifted: torch declared `torch>=2.4`
eight times and `torch>=2.5` once; vllm declared `vllm>=0.6.0` and `vllm>=0.11`;
deepgemm gave three different build hints and flashmla two. `envs/*.txt` is
generated from those fields by sorting the set and taking the first, so the
generated env got the LOOSEST constraint any kernel happened to state — an
environment that satisfies some of its kernels and not the rest, which is not
an environment. The hint was worse than arbitrary: it was whichever kernel
registered last.

Resolved to the STRICTEST declared bound, because an env has to satisfy every
kernel in it:

  * torch   `>=2.5` — `torch_sdpa` passes `enable_gqa=`, which arrived in 2.5.
              The eight kernels saying `>=2.4` are not wrong about themselves,
              they are just not the binding constraint.
  * vllm, sglang  a fresh env installs `==` the release their framework
              binding was audited at (`audited_at`), so a new measurement is
              of the release the binding describes. That is a default, not a
              gate: `accepts` keeps the floor the kernels run on
              (`vllm>=0.10.2`, `sglang>=0.5.5.post3`), so a box with another
              release installed still measures, and the row records the
              version it found. `sglang-kernel` (the compiled half;
              `sgl-kernel` before 0.5.11) is a second package whose version
              sglang pins.

...and to the LIBRARY-level build hint, because a hint that describes one
kernel's requirement ("the sparse kernel needs Hopper", "MegaMoE, SM100/fp4")
is telling the reader of a whole env about a kernel they may not be running.
The machine-checkable half of a per-kernel qualification is `Provider.min_sm`,
which is enforced rather than described; the rest belongs next to that kernel.

`pip` is a tuple, so a library that genuinely needs several packages says so
instead of its kernels appearing to disagree. sglang is the case that forced
it: `sglang` (the framework) and `sgl-kernel` (the compiled half) are
separately versioned, and its kernels gate on whichever of the two they
actually import.

A source build carries no pip spec at all — the whole point is that pip cannot
resolve it — and says how to get it in `build_hint`.
"""
from __future__ import annotations

from backend.inference.framework_bindings import get_binding

from . import Library, register_library


def _audited_release(framework: str) -> str:
    """The release the framework's binding was audited at, as a pip version."""
    return get_binding(framework).audited_at.release


for _lib in (
    Library("torch", pip=("torch>=2.5",)),
    Library("vllm", pip=(f"vllm=={_audited_release('vllm')}",),
            accepts=("vllm>=0.10.2",), device="cuda"),
    Library("flashinfer", pip=("flashinfer-python>=0.2.0",),
            device="cuda"),
    # Two packages, not two opinions about one.
    Library("sglang", pip=(f"sglang=={_audited_release('sglang')}",
                           "sglang-kernel"),
            accepts=("sglang>=0.5.5.post3", "sgl-kernel"),
            device="cuda"),
    Library("humming", pip=("humming-kernels",), device="cuda"),
    # marlin ships inside vllm; the corpus keeps it as its own `framework`
    # because a marlin row is a different measurement from a vllm-native one.
    # It installs the SAME release as the `vllm` library and for the same
    # reason: a fresh env measures the release the vLLM binding was audited at.
    # `vllm>=0.20` was the whole declaration once, and a floor is not a
    # decision — the day a newer vLLM publishes, the next resolve installs it
    # and the box measures marlin at a release no binding describes, with the
    # row saying so only after the fact. The floor survives as `accepts`, so a
    # box that already has another release still measures.
    Library("marlin", pip=(f"vllm=={_audited_release('vllm')}",),
            accepts=("vllm>=0.20",), device="cuda"),
    # PyPI's flash-linear-attention ships `fla/layers` and `fla/models` and
    # nothing else — wheel and sdist, every version from 0.3.2 to 0.5.2 — so
    # `fla/ops`, which every one of these kernels imports, is in no published
    # artifact. What ships still satisfies `import fla` as a namespace package,
    # so availability said yes and every cell failed one at a time: 311 in a
    # single pass.
    #
    # The TAG rather than a commit, because the corpus records
    # `framework_version` from `__version__`, and a main commit reports "0.5.2"
    # too — rows measured from an arbitrary tree would be indistinguishable
    # from rows measured at the release. `v0.5.2` reports 0.5.2 and IS 0.5.2.
    Library("fla", pip=("flash-linear-attention @ git+https://github.com/"
                        "fla-org/flash-linear-attention.git@v0.5.2",),
            device="cuda"),
    # The vLLM plugin for Huawei Ascend NPUs. `torch_npu` is the import
    # gate rather than `vllm_ascend` itself: every kernel registered under
    # this library calls a `torch_npu.npu_*` entry point, and that package
    # is what the plugin's own wheels depend on. Neither is installable from
    # PyPI against a CANN toolchain the way a CUDA wheel is, so the box
    # brings its own (the `quay.io/ascend/vllm-ascend` image), which is the
    # same arrangement `fa3` has: declared as a source build so a fresh env
    # does not pretend it can resolve one.
    # The AscendC delta-rule kernels the plugin calls (`chunk_kda_fwd`,
    # `recurrent_kda`). A DISTRIBUTION whose name is not its module's: the
    # wheel is `flash-linear-attention-npu-a3` and the package sets no
    # `__version__`, so every availability probe read it as absent while the
    # import worked — the `kda` cells were planned and then skipped as
    # "fla_npu not importable" on a box that runs them.
    Library("fla_npu", pip=("flash-linear-attention-npu-a3",),
            dist_name="flash-linear-attention-npu-a3", source_build=True,
            device="npu"),
    Library("vllm_ascend", source_build=True,
            build_hint="run inside the vllm-ascend image for the release the "
                       "binding is audited at (quay.io/ascend/vllm-ascend"
                       ":v0.30.0-a3); needs CANN + torch_npu, which do not "
                       "resolve from PyPI",
            device="npu"),
    Library("fa2", pip=("flash-attn>=2.6,<3",), no_build_isolation=True,
            device="cuda"),
    Library("fa3", source_build=True,
            build_hint="build hopper/ from github.com/Dao-AILab/flash-attention "
                       "(needs sm90)",
            device="cuda"),
    Library("fa4", source_build=True,
            build_hint=(
                "FA4 via sglang jit_kernel (sglang.jit_kernel.flash_attention_v4) "
                "+ FlashAttention 4's CUTE frontend, sm90+. Its flash_attn build "
                "CONFLICTS with the flash_attn 2.x that fa2 measures, so the two "
                "cannot share one env."),
            device="cuda"),
    Library("flashmla", source_build=True,
            build_hint="pip install from github.com/deepseek-ai/FlashMLA "
                       "(no PyPI release)",
            device="cuda"),
    Library("deepgemm", source_build=True,
            build_hint="build from github.com/deepseek-ai/DeepGEMM "
                       "(no PyPI release)",
            device="cuda"),
    Library("deepinfer", source_build=True,
            build_hint=(
                "build DeepInfer at the commit framework_bindings/"
                "deepinfer.yaml audits, from github.com/deepseek-ai/"
                "DeepInfer (Python 3.12, CUDA 13, SM100; build.sh also "
                "builds its vendored DeepGEMM / FlashMLA / DeepSelect "
                "submodules and installs tile-kernels)"),
            device="cuda"),
    Library("cutlass_mla", source_build=True,
            build_hint="sgl_kernel cutlass_mla_decode — SM100 (Blackwell) only",
            device="cuda"),
    Library("flash_kda", source_build=True,
            build_hint="pip install from github.com/MoonshotAI/FlashKDA "
                       "(CUTLASS, SM90+, CUDA 12.9+, K=V=128 only)",
            device="cuda"),
    Library("mamba_ssm", source_build=True,
            build_hint="pip install mamba-ssm (CUDA build), or use vLLM/sglang "
                       "vendored kernel",
            device="cuda"),
):
    register_library(_lib)
