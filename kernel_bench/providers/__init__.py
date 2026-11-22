"""Provider registry + discovery.

A `Provider` is one library's implementation of an op-signature. Many
providers implement the same signature (that's the whole point — we
measure the cross-library envelope). Providers self-register on import,
guarded so a missing library simply doesn't register.

The driver asks `applicable(inv, env)` for the providers that (a) target
the invocation's signature, (b) are importable in this env, and (c) pass
their capability predicate for these dims/dtype/gpu. Only those run.

Changing a provider's `build` / `convert_inputs` / `wrap_output` changes what
its rows measured, and no row column can say so: purge that `kernel_impl`'s
rows (every outcome) in the same merge — `kernel_bench/ARCHITECTURE.md`,
"What a sweep records when it does not measure". A kernel that cannot run
yet is declared with `disabled_reason`, never left to raise in `build`: a
`NotImplementedError` from `build` is recorded as that kernel's `error` on
every cell it is offered.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..env import Env
from ..invocation import Invocation


@dataclass
class LatencyStats:
    p50_us: float
    p95_us: float
    mean_us: float
    samples: int
    #: Which clock produced these numbers, one of
    #: `benchmarks.TIMING_METHODS` — the row's `timing_method`.
    timing_method: str


class Launch:
    """One host-side call, callable and self-describing.

    The point is that the description is RENDERED FROM THE ARGUMENTS THE CALL
    IS MADE WITH, so it cannot drift from what ran. A provider that writes
    the signature out by hand beside its call has two copies of one fact, and
    the case that motivated it says which way that drifts: `moe_permute`'s
    cell stated a dispatch that reads token-ordered rows and writes an
    expanded expert-ordered buffer, while the timed call was `x[idx]` with
    both operands the same length — a 1.5x difference in traffic that nothing
    on the row could show (fixed 10-24: the cell names both sides and the
    call is `index_select(x, 0, idx, out=)`). A rendered call shows such a
    gap at a glance.

    Rendered as `module.fn(<arg>, …, kw=<arg>)` with a tensor as
    `<dtype>[<shape>]` (plus `:strided` when it is not contiguous, because
    for these kernels the layout is half the call), a tuple recursively, and
    anything else by `repr` when that is short and single-line (a dtype, a
    ScalarType, a device — these are dispatch keys) and by its type name
    otherwise. Truncated, since this goes on every row.
    """
    __slots__ = ("args", "fn", "kwargs")

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)

    @staticmethod
    def _render(value, depth: int = 0) -> str:
        if depth > 2:
            return "…"
        try:
            shape = getattr(value, "shape", None)
        except Exception:        # a `shape` property that raises is not ours to fix
            shape = None
        if shape is not None and hasattr(value, "dtype"):
            dtype = str(value.dtype).replace("torch.", "")
            # A dim that is not an int (SymInt, unbacked) renders as itself
            # rather than sinking the row: this is provenance, and the row's
            # own `shape_json` is the number a consumer computes with.
            dims = ",".join(str(d) for d in shape)
            contiguous = getattr(value, "is_contiguous", None)
            strided = ""
            try:
                if contiguous is not None and not contiguous():
                    strided = ":strided"
            except (TypeError, RuntimeError, NotImplementedError):
                strided = ""
            return f"{dtype}[{dims}]{strided}"
        if isinstance(value, (tuple, list)):
            inner = ",".join(Launch._render(v, depth + 1) for v in value[:8])
            more = ",…" if len(value) > 8 else ""
            return f"({inner}{more})" if isinstance(value, tuple) else f"[{inner}{more}]"
        # `repr` for anything that reprs short and on one line: a `dtype=`,
        # a `ScalarType`, a `torch.device` or an Enum IS the dispatch key
        # here, and folding it to its type name renders two different calls
        # identically — `dtype=dtype` for both bf16 and fp16.
        try:
            text = repr(value)
        except Exception:
            return type(value).__name__
        return (text if len(text) <= 40 and "\n" not in text
                else type(value).__name__)

    @staticmethod
    def _schema_names(fn) -> Optional[List[str]]:
        """Parameter names from a torch op's own schema, when it has one.

        `torch.ops.aten.*` overloads carry `_schema.arguments` — the
        authoritative names, published by the dispatcher itself — and have
        neither an introspectable signature nor a documented first line. A
        provider that calls a dispatcher op directly would otherwise render
        positionally for want of asking the one thing that knows.
        """
        schema = getattr(fn, "_schema", None)
        arguments = getattr(schema, "arguments", None)
        if not arguments:
            return None
        out = [getattr(a, "name", "") for a in arguments]
        return out if all(n.isidentifier() for n in out) else None

    @staticmethod
    def _doc_names(fn) -> Optional[List[str]]:
        """Parameter names from a C function's documented signature line.

        `inspect.signature` raises for most of the library calls measured
        here — `torch.index_add`, `torch.mul`, `torch.index_select` are all
        C functions with no `__text_signature__` — and CPython's convention
        for those is the first line of `__doc__`:
        `index_add(input: Tensor, dim: int, index: Tensor, source: Tensor,
        *, alpha=1) -> Tensor`. Parsed with a regex and nothing else: this
        text comes from an installed library and is never executed.

        Positional names only: everything from the first `*` or `/` marker
        on is keyword or unnameable, and `$self` is how a text signature
        spells the receiver.
        """
        import re
        doc = (getattr(fn, "__doc__", "") or "").lstrip()
        match = re.match(r"[\w.]+\(([^)]*)\)", doc)
        if not match:
            return None
        out: List[str] = []
        for part in match.group(1).split(","):
            part = part.strip().lstrip("$")
            if not part or part.startswith(("*", "/")):
                break
            name = part.split(":")[0].split("=")[0].strip()
            if not name.isidentifier():
                break
            out.append(name)
        return out or None

    def _names(self) -> Optional[List[str]]:
        """The callee's formal parameter names for the positional arguments,
        or None when it has no introspectable signature.

        Names are the difference between reading a row and guessing at it:
        `deep_gemm.fp8_gemm_nt((e4m3[…], f32[…]), (e4m3[…], f32[…]), bf16[…])`
        says three operands; with names it says which is `a`, which is `b`
        and which is the output `d`. Taken from the callee itself
        (`inspect.signature`), never from a list beside the call — a second
        list is a second thing to keep in step. Most torch builtins have no
        signature at all and are rendered positionally.
        """
        import inspect
        try:
            params = list(inspect.signature(self.fn).parameters.values())
        except Exception:
            params = []
        out: List[str] = []
        for p in params:
            # `*args` has no name per argument: repeating one name across
            # several values renders `fn(args=…, args=…)`, which is not a
            # call anyone could write. Everything from here on stays
            # positional. Same for POSITIONAL_ONLY, which cannot legally be
            # passed by keyword at all — `torch.ops.*` overloads, triton
            # JITFunctions and any `__call__(self, *a, **kw)` wrapper land
            # in one of these two.
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.POSITIONAL_ONLY):
                break
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                out.append(p.name)
        # No names from introspection covers two cases, not one: a C
        # function, where `signature` RAISES, and a wrapper whose signature
        # is `(*args, **kwargs)`, where it succeeds and says nothing — a
        # `torch.ops.aten.*` overload is the second. Both have their names
        # in their documented first line, so the fallback is keyed on having
        # no names rather than on how that happened.
        if not out:
            return self._schema_names(self.fn) or self._doc_names(self.fn)
        return out[:len(self.args)]

    def describe(self) -> str:
        # Root module + the function's own name, so the text reads like the
        # call a human would write: `torch.mul`, `deep_gemm.fp8_gemm_nt`. A
        # dunder keeps its class, since `torch.__getitem__` would name
        # nothing — `torch.TensorBase.__getitem__` is the gather it is.
        name = getattr(self.fn, "__name__", "") or type(self.fn).__name__
        qual = getattr(self.fn, "__qualname__", "") or name
        if name.startswith("__") and "." in qual:
            name = qual
        root = (getattr(self.fn, "__module__", "") or "").split(".")[0]
        names = self._names()
        parts = []
        for i, a in enumerate(self.args):
            rendered = self._render(a)
            parts.append(f"{names[i]}={rendered}"
                         if names and i < len(names) else rendered)
        parts += [f"{k}={self._render(v)}" for k, v in self.kwargs.items()]
        head = f"{root}.{name}" if root else name
        out = f"{head}({', '.join(parts)})"
        return out if len(out) <= 600 else out[:597] + "…)"


@dataclass
class BenchCase:
    """What a provider's `build` returns: the timed kernel + how to bring its
    output back to the signature's CANONICAL layout for the cross-check.

    `run` may be a `Launch` instead of a bare closure, and then the row
    records the call it rendered (`launch_signature`) — what was actually
    invoked, with its operands' shapes and dtypes, beside the cell's own.

    `run` is the zero-arg closure the driver times. `to_canonical(out)` maps
    the kernel's output (in the provider's native layout) to the canonical
    layout the signature's `reference` produces, so `allclose` is meaningful.
    `to_canonical=None` → this provider opts out of the correctness gate
    (still measured, just not verified).
    Kernel PARAMETERS that change what was measured don't live here — they are
    resolved before the build, by `Provider.impl_params`, so coverage can be
    asked about the right implementation and an unsupported configuration can
    decline before any input is allocated.
    """
    run: Callable[[], Any]
    to_canonical: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class Library:
    """A third-party library, and everything about INSTALLING it.

    Separate from the kernels it provides because these are facts about the
    library, and duplicating them per kernel let them drift: torch carries
    three different `pip` specs across its 12 kernels, vllm two across 14,
    sglang two that do not even name the same package. `envs/*.txt` is
    generated from that field, so which constraint an env got depended on
    registration order.

    NOT one-to-one with the CSV `framework` column, which is a corpus identity
    rather than an install target. Today `marlin` is provided by vllm,
    `cutlass_mla` by sgl_kernel, `fa4` by sglang's jit_kernel, and `sglang`
    itself covers two independently-versioned packages — `sglang` (the Python
    framework) and `sgl_kernel` (the compiled one), which is why "is sglang
    installed" was never a well-posed question.

    `pip` is a TUPLE, not one spec. A library can genuinely need more than one
    package — `sglang` needs both `sglang` (the framework) and `sgl-kernel`
    (the compiled half) — and expressing that as one string per kernel is what
    made those two look like a disagreement to reconcile rather than two
    requirements to install.

    `import_name` is NOT here: it belongs to the kernel. Two kernels of one
    library legitimately gate on different modules — sglang's rope kernels need
    `sgl_kernel` while its conv kernels need `sglang`, and fa4 needs
    `flash_attn.cute` rather than the `sglang` that merely wraps it. What is
    per-library is how the thing gets INSTALLED; what is per-kernel is which
    module has to import for that kernel to run.
    """
    name: str
    pip: Tuple[str, ...] = ()
    #: The versions a kernel of this library may run on, when wider than what
    #: a fresh env installs (`pip`). The env installs one release; the gate
    #: still accepts any release the library's kernels run on, and the row
    #: records the version that was installed. Empty: `pip` is the gate too.
    accepts: Tuple[str, ...] = ()
    source_build: bool = False
    build_hint: str = ""
    # The package builds from source and its setup.py imports torch at
    # build time (flash-attn). uv's default build isolation hides the
    # env's torch from the build, so it fails before compiling; this
    # declares that the build must see the target env.
    no_build_isolation: bool = False

    def spec_for(self, import_name: str) -> str:
        """The pip spec that constrains the module `import_name`, or "".

        A spec constrains a PACKAGE; a kernel gates on a MODULE. For a
        single-package library the two are the same requirement seen twice, so
        the one spec applies. For a multi-package library only the spec naming
        that module can judge it — `sgl-kernel` says nothing about the version
        of `sglang` a kernel imports — and the rest are real requirements of
        the library that are simply silent here.
        """
        specs = self.accepts or self.pip
        if not specs:
            return ""
        if len(specs) == 1:
            return specs[0]
        from packaging.requirements import Requirement
        root = import_name.split(".")[0].replace("_", "-").lower()
        for one in specs:
            if Requirement(one).name.replace("_", "-").lower() == root:
                return one
        return ""


#: The float dtypes a provider may declare when its launch was audited on
#: exactly this dispatch set. This is an audited capability, not a property of
#: every dtype that happens to store values unquantized. At the audited pin,
#: vLLM's stable floating dispatch names Float, Half, and BFloat16
#: (`VLLM_STABLE_DISPATCH_FLOATING_TYPES`; csrc/dispatch_utils.h and
#: csrc/pos_encoding_kernels.cu). Adding an unquantized dtype to the corpus
#: must not add it here: audit each kernel first, then widen this list as its
#: own decision.
TORCH_FLOAT: Tuple[str, ...] = ("bf16", "fp16", "fp32")


def _not_served(field: str, value: str, served: Tuple[Any, ...],
                *, empty: str = "none") -> str:
    """A refusal that names both sides of the mismatch.

    The field prefix stays machine-stable (`tools` and plan reports group on
    it), while the tail says what the cell asked for and what the declaration
    admits — a value alone cannot distinguish an unsupported token from a
    supported field with the wrong value.
    """
    what = (", ".join("plain" if token is None else str(token)
                      for token in served)
            if served else empty)
    return f"{field}:{value or '-'} is not served (serves {what})"


@dataclass(frozen=True)
class KernelSignature:
    """What a kernel claims it can serve — declared once, so the gate can be
    DERIVED from it instead of restated as a lambda.

    Two copies of one fact drift, and the drift is silent: a gate WIDER than
    the build measures a cell under a quantization the kernel never ran and
    records it as if it had. That happened four times in one day — NVFP4 folded
    with MXFP4 (different block size and scale encoding), `startswith("fp8")`
    swallowing e5m2 (a 128× different scale), humming accepting bf16 (its
    kernel asserts ≤8 bits), and GPTQ's uniform int4 grid used for E2M1's
    non-uniform one.

    Also the capability record the API publishes, which makes the claim a
    promise to consumers rather than an internal assertion.

    CLOSED-WORLD. The declaration lists what the kernel serves; whatever it
    does not list, it does not serve. A kernel never declares what it
    CANNOT do — a list of exclusions is open-ended, and every new dtype or
    variant axis the corpus grows would otherwise reach every kernel that
    forgot to exclude it (FA4 was handed fp32 because it declared no dtype;
    FA2, FA3, FlashInfer and torch SDPA each had to say "no sinks"). What a
    cell must match is derived from the signature, never listed here:

      * every dtype slot the signature names as material
        (`OpSignature.dtype_slots`) must be declared, and a cell whose token
        the declaration does not list is refused — registration rejects a
        kernel that leaves one empty, since it would serve no cell at all;
      * every variant axis the signature names (`OpSignature.variant_dims`)
        that a cell carries must be one the kernel declares
        (`variant_dims`, or `requires_dims`); the axes of the default
        geometry need no mention;
      * a cache format slot (`output_format`, `kv_format`) and the
        activation's scale granularity serve only the plain cell unless a
        format or granularity is named.

    `phases` stays open (`()` = either phase): a launch does not know which
    phase its shapes came from, so a phase list is a statement about the
    serving path a kernel stands for, not about what it can compute.
    """
    signature: str
    weight: Tuple[str, ...] = ()
    activation: Tuple[str, ...] = ()
    # The dtype the launch WRITES. Undeclared, the kernel writes what it
    # computes in — the cell's activation dtype — and a cell whose output
    # is another (a norm or an activation feeding an FP8 GEMM writes e4m3;
    # an indexer writes fp32 logits from bf16) is not this kernel's: the
    # launch that writes bf16 there is a different launch from the one the
    # cell describes. A kernel that writes another dtype says which. An
    # activation cast reads one format and writes another
    # (`quantise_activation`), so a kernel that writes e4m3 must decline
    # the fp4 cell it would otherwise be handed — a gate wider than the
    # build is the failure this whole declaration exists to prevent.
    output: Tuple[str, ...] = ()
    # The general form of the three fields above — one `dtype_env` slot and
    # the tokens this kernel runs at it — for a slot that has no field of
    # its own. `kv` is the case that forced it: vLLM's q-norm/rope/insert
    # dispatch picks its overload by the dtype of the cache it inserts K
    # into beside the query it outputs. (A launch whose OUTPUT is the
    # cached tensor — sglang's index-K store — states that format in
    # `output`, where the cell keys it.) Those are permanent
    # cache-format facts of the launch, so a cell keyed at another cache
    # dtype must be declined through the DECLARATION — where smoke's
    # synthesized cell, the published capability record and the stale-row
    # audit all read it — not only through `impl_id`, which just the bench
    # driver resolves. The slot's value is read the way the invocation
    # reads it (`Invocation.dtype_for`): a slot the cell does not carry is
    # the activation's dtype.
    dtype_slots: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Writes the launch makes BESIDE the primary output, as
    # (tensor, dtype): sglang's indexer-Q launch returns the FP8 query and
    # also writes the head weights rescaled fp32, and the one-slot `output`
    # cannot state both. Not a gate — the invocation has no slot for the
    # fact, so nothing here refuses a cell — but the capability record the
    # API publishes must not under-describe what the measured launch
    # writes.
    secondary_outputs: Tuple[Tuple[str, str], ...] = ()
    # Which activation elements share one dynamic scale
    # (`ActivationFormat.granularity`): the activation-side twin of
    # `weight_granularity` below, and part of a cast's identity for the same
    # reason — a per-token kernel and a per-128-group one are two kernels
    # between one pair of dtypes, and `scaled_fp8_quant` really does write
    # only the first. () = the cell names no activation scale: a kernel
    # that writes the plain activation does not write its scales.
    activation_granularity: Tuple[str, ...] = ()
    # The cached records the launch writes as its output, by cache-format
    # name (`backend/arch/cache_formats/`; the cell's `output_format`,
    # `LeafAccess.output_format`): the `output` token is the record's
    # element dtype and cannot say how its scales are organised. DeepSeek-
    # V4's FlashMLA record and an FP8 index key with one F32 scale are both
    # `fp8_e4m3`; an MXFP4 and an NVFP4 index key are both `fp4_e2m1`. A
    # kernel that writes one record declines a cell keyed at another, and a
    # cell that names no format (an activation, a plain cache) — unless it
    # also declares `benchmark_identity.PLAIN`, the cell that states no
    # format: `(PLAIN,)` is a kernel that writes only a plain tensor,
    # `(PLAIN, name)` one that writes either. () is `(PLAIN,)`: a kernel
    # that names no record writes none.
    output_format: Tuple[Optional[str], ...] = ()
    # The same for the cache the cell's `kv` slot names (`kv_format`,
    # `CachedTensor.format_name`): the record the launch reads, or writes
    # beside its output. vLLM's DeepSeek-V4 q/KV insert writes the FlashMLA
    # record and FlashMLA's sparse decode reads it; a per-tensor-scaled FP8
    # cache is `kv=fp8_e4m3` as well, and another kernel on both sides.
    # `PLAIN` admits the plain cache, as above, and () is `(PLAIN,)`.
    kv_format: Tuple[Optional[str], ...] = ()
    phases: Tuple[str, ...] = ()          # () = either phase (see above)
    divisible: Tuple[Tuple[str, int], ...] = ()      # (dim, modulus)
    enum: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()   # (dim, allowed values)
    # `enum` for an extent the cell names by different dims in different
    # geometries: (contract tensor, axis index, allowed extents), the extent
    # read off the tensor's layout for THIS cell (`OpSignature.layouts_for`,
    # `TensorLayout.extents`). V's head width is the case: it is `d_v_head`
    # when V is narrower than Q and `d_head` otherwise, and the signature
    # already declares which (`variant_inputs`). A kernel whose limit is on
    # V's width (FlashInfer dispatches its state merge on head_dim_vo) needs
    # that width, and restating the variant rule here would be a second copy
    # of it; reading the layout is the one copy.
    operand_enum: Tuple[Tuple[str, int, Tuple[int, ...]], ...] = ()
    # The signature's variant axes (`OpSignature.variant_dims`: a sliding
    # window, attention sinks, a V narrower than Q, MLA's per-head K/V) this
    # build implements when a cell carries one. Closed: an axis the
    # signature makes optional and this list does not name is one the build
    # does not implement, so a cell carrying it is declined before inputs
    # are allocated — official FA2/FA3 exports have no sink argument and
    # would otherwise reach the Python wrapper and raise TypeError. The
    # default geometry's axes need no mention.
    variant_dims: Tuple[str, ...] = ()
    # Variant axes the cell MUST carry: the build implements only the
    # geometry they select (non-absorbed MLA's separate KV head count; the
    # vendored FA3 that stands for vLLM's sink path). A required axis is a
    # served one, so it is not listed in `variant_dims` as well. Distinct
    # from `divisible`/`enum`, which constrain a dim's value: this one says
    # the dim must be THERE, and reading an absent dim as a default is how a
    # kernel ends up measured against a shape the model does not have.
    requires_dims: Tuple[str, ...] = ()
    # Axis-pair divisibility. GQA is the motivating case: a kernel may allow
    # fewer query heads than KV heads neither at all nor as a per-axis bound;
    # the contract is `n_heads % n_kv_heads == 0`.
    requires_divisible_dims: Tuple[Tuple[str, str], ...] = ()
    #: The paged-KV pages this kernel can serve — the backend's own
    #: `get_supported_kernel_block_sizes()`. Declared here beside everything
    #: else the kernel claims, so the pages a SWEEP measures at are derived
    #: from what some backend can actually run (`served_kv_block_sizes`)
    #: instead of being listed a second time where the sweep is written. A
    #: page outside every backend's set cannot run in production, and
    #: measuring one is measuring fiction — `page_size=1` did exactly that
    #: for 2867 rows.
    #:
    #: Three states, and the empty one is a DECLARATION rather than a gap:
    #:   * a tuple of pages — this backend reads the cache through a page
    #:     table and these are the pages it runs;
    #:   * `()` — it reads the cache as one contiguous buffer and pages
    #:     nothing (`torch_sdpa` and the other dense references);
    #:   * `None` — nobody said, which `register` REFUSES for any kernel
    #:     whose signature reads a cache at all. The page is part of the
    #:     measurement key, so an unstated one is a cell keyed under a
    #:     paging layout that never happened.
    kv_block_sizes: Optional[Tuple[int, ...]] = None
    # Runtime capabilities this registered build does not implement. Unlike the
    # fields above, these are not all present on an Invocation today (paging is
    # a serving-backend fact, not a model shape); publishing them here keeps the
    # limitation next to the kernel that owns it instead of hiding it in a
    # build hint. Tokens are stable snake_case so API consumers can branch on
    # them; the UI may render a human label.
    unsupported_features: Tuple[str, ...] = ()
    # Inclusive upper bounds, e.g. FlashAttention's head-dimension ceiling.
    # A ceiling is not expressible as a modulus or an enumeration — the set of
    # legal values is open below it — and without one every oversized cell is
    # attempted and refused by the library itself: 2,236 failed builds in a
    # single pass across fa2 and fa3.
    max_dims: Tuple[Tuple[str, int], ...] = ()
    # Joint ceilings: the PRODUCT of these dims must not exceed the bound.
    # Some limits are not per-axis at all — a kernel indexing its state with
    # int32 fails when the element count crosses 2**31 no matter how the
    # extents are distributed, so no combination of per-dim ceilings describes
    # it. Bisected on hardware: sglang's causal_conv1d_update faults at exactly
    # one step past B * C * W = 2**31, at C = 1024, 2048, 4096 and 8192 alike.
    max_product: Tuple[Tuple[Tuple[str, ...], int], ...] = ()
    # The same shape of ceiling, for a different fact: the kernel RUNS the
    # cell and the harness declines to sample it. A torch floor written as a
    # Python loop over experts is the case — above a few thousand routed rows
    # its wall clock is host dispatch rather than device time, at 60x the
    # sampling cost of a plain GEMM cell, for a number no deployment pays.
    # Kept apart from `max_product` because the two say opposite things about
    # the rows ALREADY measured past them: a capability ceiling means those
    # rows timed a launch the kernel cannot run (they are wrong, see
    # `tools.unserved_rows`), a budget ceiling means they are honest
    # measurements of an implementation nobody chose to keep paying for.
    samples_up_to: Tuple[Tuple[Tuple[str, ...], int], ...] = ()
    # Which weight scale layouts the kernel reads (`TensorFormat.
    # weight_granularity`: `per-tile:128x128`, `per-group:32`, ...). A cell
    # carries the checkpoint's; a kernel that quantizes differently is a
    # different measurement of the same dtypes — cutlass per-channel FP8 and
    # DeepGEMM's 128x128 blocks shared one cell and the envelope took the
    # faster without saying the checkpoint can only run one of them. A kernel
    # that declares `weight_group` instead has it derived (`Provider`).
    weight_granularity: Tuple[str, ...] = ()
    # Kernel parameters the launch needs the invocation to carry (
    # `Invocation.kernel_params`, filled from what the MODEL publishes —
    # `OpSignature.config_params`). A kernel whose scalar is missing does
    # not run: vLLM says so itself ("SITU requires activation_situ_beta
    # from FusedMoEConfig"), and picking a value instead measures a
    # function the model does not compute while the reference, picking the
    # same one, confirms it. Declared here rather than raised in `build`
    # so the cell declines before any input is allocated and `report`
    # counts it as the gap it is.
    requires_params: Tuple[str, ...] = ()
    #: This KERNEL is handed the recurrent state a state-carrying cell
    #: carries (`Invocation.carries_state`) and writes it back — the build
    #: allocates the state, passes it to the launch, and the launch updates
    #: it. Per kernel, not per signature: under one signature a framework's
    #: scan takes the state while a torch floor starts from zeros and keeps
    #: nothing. A cell that carries a state bills its read and write-back
    #: (`traffic._settle_cache_reads` / `_settle_state_writes`), so only a
    #: kernel declaring this is that cell; any other refuses it
    #: ("carries_state") and the cell is a counted gap. A kernel declaring
    #: it still serves a cell with no state (a compressor at ratio 1 keeps
    #: no ring): it then takes none.
    takes_state: bool = False

    def refusals(self, inv: "Invocation"):
        """Which CLAUSE of this declaration each failure comes from, lazily.

        A gate that answers only yes/no cannot say why a kernel serves
        nothing, and "why" is the whole question when a kernel has never
        run: a `max_dims` ceiling the corpus always crosses means the card
        is wrong for it, while an `activation` list that misses the one
        dtype the corpus uses is a declaration bug to fix. Counting the
        clauses over real planned cells separates those two, and
        `tools/provider_gaps.py` does exactly that.

        A GENERATOR, and `accepts` reads only its first item, so the gate
        still short-circuits and there is one implementation of the gate
        rather than a predicate plus a diagnostic that drift apart.

        Tokens are `field` or `field:detail`, stable enough to group on.
        """
        if inv.signature != self.signature:
            yield "signature"
            return                      # the rest is about another contract
        if inv.carries_state and not self.takes_state:
            # The cell bills a recurrent state's read and write-back; a
            # kernel that is not handed the state never moves those bytes.
            yield "carries_state"
        env = inv.dtype_env
        for slot, value, admitted in self._dtype_gates(env):
            if value not in admitted:
                yield _not_served(slot, value, admitted)
        granularity = env.get("activation_granularity", "")
        if granularity not in (self.activation_granularity or ("",)):
            yield _not_served(
                "activation_granularity", granularity,
                self.activation_granularity,
                empty="no activation scale")
        for slot, names in self.formats().items():
            if env.get(slot) not in names:
                yield _not_served(slot, env.get(slot, ""), names)
        for name in self.unserved_variants(inv.dims):
            yield f"variant_dims:{name} is present but unserved"
        if self.phases and inv.phase not in self.phases:
            yield _not_served("phases", inv.phase, self.phases)
        for name, mod in self.divisible:
            v = int(inv.dims.get(name, 0) or 0)
            if v <= 0:
                yield (f"divisible:{name}={v} is not a positive "
                       f"multiple of {mod}")
            elif v % mod:
                yield f"divisible:{name}={v} is not divisible by {mod}"
        for name, allowed in self.enum:
            v = int(inv.dims.get(name, 0) or 0)
            if v not in allowed:
                yield (f"enum:{name}={v} is not one of "
                       f"{', '.join(map(str, sorted(allowed)))}")
        if self.operand_enum:
            layouts = {lay.name: lay for lay in _signature_layouts(
                self.signature, inv.dims)}
            for tensor, axis, allowed in self.operand_enum:
                try:
                    extent = layouts[tensor].extents(inv.dims)[axis]
                except KeyError:            # the cell lacks a dim of the axis
                    extent = None
                if extent not in allowed:
                    yield _not_served(
                        f"operand_enum:{tensor}[{axis}]", str(extent),
                        tuple(str(x) for x in allowed))
        for name in self.requires_dims:
            if int(inv.dims.get(name, 0) or 0) <= 0:
                yield f"requires_dims:{name}>0, got {inv.dims.get(name, 0)}"
        for numerator, denominator in self.requires_divisible_dims:
            num = int(inv.dims.get(numerator, 0) or 0)
            den = int(inv.dims.get(denominator, 0) or 0)
            if num <= 0 or den <= 0:
                yield (f"requires_divisible_dims:{numerator}={num} and "
                       f"{denominator}={den} must both be positive")
            elif num % den:
                yield (f"requires_divisible_dims:{numerator}={num} is not "
                       f"divisible by {denominator}={den}")
        for name in self.requires_params:
            if inv.kernel_params.get(name) is None:
                yield f"requires_params:{name}"
        for name, hi in self.max_dims:
            v = int(inv.dims.get(name, 0) or 0)
            if v > hi:
                yield f"max_dims:{name}={v} exceeds {hi}"
        for field_name, bounds in (("max_product", self.max_product),
                                   ("samples_up_to", self.samples_up_to)):
            for names, hi in bounds:
                prod = 1
                for name in names:
                    prod *= int(inv.dims.get(name, 0) or 0)
                if prod > hi:
                    yield (f"{field_name}:{'*'.join(names)}={prod} "
                           f"exceeds {hi}")

    def formats(self) -> Dict[str, Tuple[Optional[str], ...]]:
        """The cache formats served per format slot of the cell
        (`benchmark_identity.FORMAT_SLOTS`): `output_format`, `kv_format`.
        A slot that names none serves the plain tensor only (`PLAIN`)."""
        from backend.inference.benchmark_identity import FORMAT_SLOTS, PLAIN
        return {slot: getattr(self, slot) or (PLAIN,) for slot in FORMAT_SLOTS}

    def _material_slots(self) -> Tuple[str, ...]:
        """The dtype slots the signature says drive the kernel
        (`OpSignature.dtype_slots`) — the ones a declaration must speak for."""
        from backend.inference.op_signatures import get_signature
        sig = get_signature(self.signature)
        return sig.dtype_slots if sig is not None else ()

    def writes(self, activation: str) -> Tuple[str, ...]:
        """The dtypes the launch writes as its output, for a cell whose
        activation is `activation`: the declared `output`, or — undeclared —
        the activation itself, which is what a kernel that computes in the
        cell's dtype writes."""
        return self.output or (activation,)

    def admits(self, dtype_slot: str) -> Tuple[str, ...]:
        """The tokens the declaration lists for dtype slot `dtype_slot`
        (() = none listed; which of those is a refusal is `_dtype_gates`)."""
        if dtype_slot in ("weight", "activation", "output"):
            return getattr(self, dtype_slot)
        return dict(self.dtype_slots).get(dtype_slot, ())

    def _dtype_gates(self, env: Mapping[str, str]):
        """`(slot, cell value, admitted tokens)` for every dtype slot this
        declaration gates: the signature's material slots, closed — an
        undeclared one admits nothing — plus any other slot it chose to
        declare, and the output, which defaults to the activation's dtype.

        Values are read as the invocation reads them for a slot with no
        field of its own (`Invocation.dtype_for`: absent is the
        activation's); `weight` and `activation` raw, as they always were."""
        material = self._material_slots()
        activation = env.get("activation", "")
        for slot in ("weight", "activation"):
            declared = getattr(self, slot)
            if declared or slot in material:
                yield slot, env.get(slot, ""), declared
        written = (self.output if "output" in material
                   else self.writes(activation))
        yield "output", env.get("output") or activation, written
        declared_slots = dict(self.dtype_slots)
        extra = tuple(s for s in material
                      if s not in ("weight", "activation", "output")
                      and s not in declared_slots)
        for slot in (*declared_slots, *extra):
            yield (slot, env.get(slot) or activation,
                   declared_slots.get(slot, ()))

    def served_variants(self) -> Tuple[str, ...]:
        """The signature's variant axes this kernel serves when present:
        those it allows and those it requires."""
        return self.variant_dims + self.requires_dims

    def unserved_variants(self, dims: Mapping[str, Any]) -> Tuple[str, ...]:
        """The variant axes (`OpSignature.variant_dims`) a cell of `dims`
        carries — as a positive extent, the reading `requires_dims` takes of
        presence — that this kernel does not declare it serves."""
        from backend.inference.op_signatures import get_signature
        sig = get_signature(self.signature)
        if sig is None:
            return ()
        served = set(self.served_variants())
        return tuple(name for name in sig.variant_dims
                     if name not in served
                     and int(dims.get(name, 0) or 0) > 0)

    def accepts(self, inv: "Invocation") -> bool:
        return next(self.refusals(inv), None) is None

    def describe(self) -> Dict[str, Any]:
        """JSON-safe capability record — what the API publishes."""
        record = {
            "signature": self.signature,
            "weight_dtypes": list(self.weight),
            "activation_dtypes": list(self.activation),
            "output_dtypes": list(self.output),
            "activation_granularities": list(self.activation_granularity),
            # Formats as served: a slot that names none serves the plain
            # tensor, published as null (`PLAIN`).
            "output_formats": list(self.formats()["output_format"]),
            "kv_formats": list(self.formats()["kv_format"]),
            # The signature's optional axes this kernel serves; any other a
            # cell carries is declined.
            "variant_dims": list(self.served_variants()),
            "weight_granularities": list(self.weight_granularity),
            "required_kernel_params": list(self.requires_params),
            "takes_state": self.takes_state,
            "phases": list(self.phases),
            "unsupported_features": list(self.unsupported_features),
            "shape_constraints": (
                [f"{n} % {m} == 0" for n, m in self.divisible]
                + [f"{n} in {list(v)}" for n, v in self.enum]
                + [f"{t}.shape[{a}] in {list(v)}"
                   for t, a, v in self.operand_enum]
                + [f"{n} present" for n in self.requires_dims]
                + [f"{n} % {d} == 0"
                   for n, d in self.requires_divisible_dims]
                + [f"{n} <= {h}" for n, h in self.max_dims]
                + [f"{' * '.join(ns)} <= {h}" for ns, h in self.max_product]
                + [f"{' * '.join(ns)} <= {h} (sampling budget)"
                   for ns, h in self.samples_up_to]),
        }
        for slot, tokens in self.dtype_slots:
            record[f"{slot}_dtypes"] = list(tokens)
        if self.secondary_outputs:
            record["secondary_outputs"] = [
                {"tensor": name, "dtype": dtype}
                for name, dtype in self.secondary_outputs]
        return record


def _signature_layouts(signature: str, dims: Mapping[str, Any]):
    """The contract tensors of `signature` in the geometry `dims` selects —
    what `KernelSignature.operand_enum` reads an extent off. Registration
    has already refused an unknown signature or tensor."""
    from backend.inference.op_signatures import get_signature
    return get_signature(signature).layouts_for(dims)


_LIBRARIES: Dict[str, Library] = {}


def register_library(lib: Library) -> None:
    if lib.name in _LIBRARIES:
        raise ValueError(f"library {lib.name!r} registered twice")
    _LIBRARIES[lib.name] = lib


def get_library(name: str) -> Library:
    """The install facts for a library. Raises rather than inventing an empty
    record: a kernel whose library is undeclared would otherwise generate an
    env file that installs nothing and looks complete."""
    try:
        return _LIBRARIES[name]
    except KeyError:
        raise KeyError(
            f"no Library registered for {name!r} — declare it in "
            f"kernel_bench/providers/_libraries.py") from None


def all_libraries() -> List[Library]:
    discover()
    return [_LIBRARIES[k] for k in sorted(_LIBRARIES)]


@dataclass
class Provider:
    """One (library, kernel variant) implementation of a signature.

    `build(inv, env, inputs)` receives the measurement tensors already in
    this kernel's launch layout when `launch_inputs` is declared — a unique
    permute/pack from the signature's canonical tensors, applied by the
    driver. What remains in `build` is the library's call convention
    (varlen cu_seqlens, paged block tables, kwargs), not the axis order.
    Returns a `BenchCase` — the timed kernel plus an output canonicalizer.
    All providers of a signature start from the SAME canonical inputs, so
    their outputs (and latencies) are directly comparable against the one
    reference.
    """
    # Names a `Library` record — how the thing is INSTALLED lives there, once,
    # and `register()` rejects a name no library declared. What is per-kernel is
    # `import_name`: the module that has to import for THIS kernel to run, which
    # is not one per library (sglang's rope kernels gate on `sgl_kernel`, its
    # conv kernels on `sglang`, and fa4 on `flash_attn.cute` rather than the
    # `sglang` that merely wraps it).
    library: str                 # "fa3" | "flashinfer" | "flashmla" | ...
    signature: str               # op-signature name it implements
    kernel_impl: str             # variant tag → goes in the CSV identity
    import_name: str             # module whose presence gates availability
    build: Callable[[Invocation, Env, Dict[str, Any]], "BenchCase"]
    # The declaration that will replace `applies_to`. Both may be present
    # during migration; the lambda stays authoritative until they are shown
    # equivalent over real planned invocations, so declaring one cannot change
    # which cells get measured.
    serves: Optional["KernelSignature"] = None
    # The gate a kernel has not yet migrated off. Exactly one of this and
    # `serves` is given: two ways to say what a kernel serves is two places to
    # be wrong, which is the drift this refactor exists to remove.
    applies_to: Optional[Callable[[Invocation, Env], bool]] = None
    # The fusion boundary (fused_op) this provider realizes — its SPAN over the
    # canonical op graph is that fused_op's `naive_pattern` (fusion_plans, the
    # single span vocabulary). "" ⇒ a degenerate 1-node span (the provider
    # covers exactly its own `signature`'s op). Set it when a kernel absorbs
    # more than one canonical op (e.g. MegaMoE = the whole expert block). The
    # per-kernel-impl graph (Stage 3 cover) reads this to know what each impl
    # covers vs what must be stitched with torch.
    fused_op: str = ""
    # Resolves the deployment kernel parameters this implementation will run
    # with, as a suffix on `kernel_impl` — PURELY from the invocation, before any
    # input is allocated. Two reasons it can't wait for `build`:
    #   * coverage is asked before build, so without it a corrected parameter
    #     looks like the old measurement and gets skipped as already covered;
    #   * a parameter this backend cannot serve should decline BEFORE the inputs
    #     for a shape it won't measure are allocated on the GPU.
    # Raises (NotImplementedError) to decline. "" ⇒ no resolved parameters.
    impl_params: Optional[Callable[[Invocation], str]] = None
    # Minimum compute capability as an integer, e.g. 100 for a Blackwell-only
    # kernel (`sm_100`). An AVAILABILITY fact like `import_name`, not a shape
    # claim, so it lives here rather than in `serves`: the kernel is absent on
    # this card in the same sense the library would be if it were not installed.
    # Left in a `build_hint` string it constrains nothing — cutlass_mla_decode
    # documented "SM100 (Blackwell) only" there and was still attempted, and
    # raised, 279-347 times per MLA model on an H100, allocating a paged latent
    # cache each time before sgl_kernel refused.
    min_sm: Optional[int] = None
    # The releases of the package `import_name` gates on that still CARRY this
    # kernel, as a PEP 440 version specifier ("<0.4.5"), when narrower than
    # the releases its library accepts. The same kind of fact as `min_sm`, for
    # the same reason: a kernel a release removed is absent from an env that
    # installs that release, and the import gate cannot see it because the
    # package still imports — sglang-kernel 0.4.5 retired the AOT
    # `moe_fused_gate`, and `sgl_kernel` imported while 1,030 cells failed one
    # at a time on `from sgl_kernel import moe_fused_gate`. Per kernel because
    # the library's other kernels are unaffected. "" = no narrower bound.
    version_specifier: str = ""
    # The dtype slots this build actually COMPUTES in, when they are not the
    # cell's: `{"weight": "bf16", "activation": "bf16"}` on a kernel that
    # dequantizes a quantised checkpoint at load and multiplies in bf16. Goes
    # on the row as `measured_dtype_env_json`, so a consumer can tell a
    # precision fallback from a kernel that ran the cell's own arithmetic —
    # the two have very different latencies and very different weight bytes,
    # and before this column the only hint was a word in `kernel_impl`.
    #
    # Declared per provider rather than decided per cell, because it is a
    # property of the BUILD: a dequantising path dequantizes whatever the
    # checkpoint stored. None (the common case) means the kernel computes at
    # the cell's precision and the row's column stays empty.
    computes_at: Optional[Mapping[str, str]] = None
    #: The same fact when one build resolves it from the cell — a GEMM floor
    #: promotes bf16 and fp16 operands to fp32, or runs beside an fp32 router
    #: weight at fp32. `emit.build_row` reads this once, both for the row's
    #: measured-dtype provenance and for the compute peak it is judged against.
    computes_at_for: Optional[Callable[[Invocation], Mapping[str, str]]] = None
    # Elements along the reduction axis that share one weight scale, for a
    # kernel that quantizes the weight itself. 0 = one scale for the whole
    # tensor; None = the kernel makes no claim, and its comparison is left as
    # it was. NOT derivable from the weight dtype: marlin packs NVFP4 in blocks
    # of 16 while humming groups the same element encoding by 128, and the two
    # groupings put the weight on DIFFERENT grids. Declared so the harness can
    # place the canonical weight on this kernel's grid before either the kernel
    # or the reference sees it — see kernel_bench/quant.py.
    weight_group: Optional[int] = None
    # How this kernel STORES that scale — also part of its grid. marlin's int4
    # path does `scales.to(dt)`, so on a bf16 cell the scale is bf16, and the
    # weight lands somewhere an exact fp32 scale does not. Modelling the element
    # levels alone left NVFP4 at 0.19 relative error, still an order of
    # magnitude over the band; modelling the scale too takes it to exactly 0.
    # See `quant.SCALE_ENCODINGS`.
    weight_scale: str = "exact"
    # Why this kernel is registered but serves nothing. A kernel that measures
    # WRONG is disabled here rather than deleted: an empty gate leaves its cells
    # showing as a GAP in the coverage report, while deleting the registration
    # makes them look like cells nobody ever wanted. The reason travels with it,
    # into the published capability record.
    disabled_reason: str = ""
    # Why no cell of THIS corpus reaches a kernel whose declaration is right.
    # The three reasons a registered kernel can have no rows are different
    # facts and want different actions: `disabled_reason` says the kernel
    # must not be measured, `min_sm` says the card is wrong for it, and this
    # says the kernel is correct, enabled and runnable on this card, and no
    # checkpoint we serve asks for it — marlin's weight-only FP8 against a
    # corpus whose FP8 models quantise the activation too, the fp16 twin of
    # a Mamba-2 gated norm every corpus model ships in bf16. Undeclared, the
    # audit cannot tell that from a declaration bug: both read as "no rows,
    # no reason declared", and a `serves` nobody can honour is
    # indistinguishable there — and in the published capability record —
    # from a measurement.
    #
    # ONE SENTENCE, naming the corpus fact a reader can check; never a list
    # of model names. Which kernels serve nothing is already DERIVABLE
    # (`tools/provider_gaps.py` joins every planned cell against `accepts`),
    # so the only thing that cannot be derived is the author's assertion
    # that serving nothing here is EXPECTED. It is declared beside the claim
    # so the capability record the API publishes is qualified in the same
    # place it is made. A kernel that serves nothing because our walk states
    # the wrong dtype, or because no plan produces its cell, does NOT get
    # one: that gap is ours to fix, and it must keep reading as a gap.
    unreached_reason: str = ""
    # How this kernel wants each named canonical tensor laid out at launch:
    # `(name, grouping)` where `grouping` is a permutation of the contract
    # tensor's physical dims (nested tuples pack). Axis names come from the
    # signature — this is not a second layout. Empty = canonical is already
    # what the library takes. Concat, paging, and varlen metadata stay in
    # `build`.
    launch_inputs: Tuple[Any, ...] = ()
    # Canonical [B,L,D] inputs stored as [pages,page,1,record_bytes] uint8.
    # Each pair names the input and its dtype_env cache-format slot. The
    # launch gate derives exact byte extents from that format and page axis.
    paged_byte_cache_inputs: Tuple[Tuple[str, str], ...] = ()
    # Name of a `launch_inputs` tensor whose grouping to invert on the
    # kernel output. Last-dim width follows the tensor. None = `build`
    # still owns the output mapping.
    launch_output: Optional[str] = None
    #: A provider-specific oracle for a launch that deliberately deviates
    #: from the signature's reference function. Empty uses the signature's
    #: one oracle. A non-empty oracle must name the deviation in its docstring
    #: and canonicalize the launch into the same basis it computes; it is a
    #: correctness contract, not a way to loosen the shared band.
    reference: Optional[Callable[[Dict[str, Any]], Any]] = None

    @property
    def lib(self) -> Library:
        """The install facts, which belong to the library rather than to this
        kernel. Read through, never copied: a per-kernel `pip` is how torch
        came to state three different constraints and the generated env to
        pick whichever sorted first."""
        return get_library(self.library)

    @property
    def pip(self) -> Tuple[str, ...]:
        return self.lib.pip

    @property
    def source_build(self) -> bool:
        """Not pip-resolvable — FA3-hopper, FlashMLA, DeepGEMM are custom
        source builds (GPU-arch-specific, subdirectory builds). Provisioned
        OUT OF BAND and only TARGETED here: excluded from envs/*.txt and never
        uv-resolved; `run-all` points at a caller-supplied interpreter."""
        return self.lib.source_build

    @property
    def build_hint(self) -> str:
        return self.lib.build_hint

    def refusals(self, inv: Invocation, env: Env):
        """Why this kernel declines this cell, clause by clause — the same
        gate `accepts` takes, read for its reasons instead of its verdict.

        A generator, consumed for its first item by `accepts`, so the two
        cannot disagree. A kernel still on a hand-written `applies_to` can
        only say that the lambda said no, which is one more reason to
        migrate it.
        """
        if self.disabled_reason:
            yield "disabled"
            return                 # a disabled kernel's scope is not the point
        cell = inv.weight_granularity
        if cell is not None and cell not in self.weight_granularities:
            yield _not_served("weight_granularity", cell,
                              self.weight_granularities)
        if self.serves is not None:
            yield from self.serves.refusals(inv)
        elif not self.applies_to(inv, env):
            yield "applies_to"

    def accepts(self, inv: Invocation, env: Env) -> bool:
        """The gate in force: the declaration when there is one, otherwise the
        lambda. `register` guarantees exactly one exists, so this cannot be
        both or neither."""
        return next(self.refusals(inv, env), None) is None

    @property
    def weight_granularities(self) -> Tuple[str, ...]:
        """The weight scale layouts this kernel reads: declared on `serves`,
        or what its `weight_group` already says (0 = one scale per tensor,
        g = one per g elements of the reduction axis). Empty = no claim, and
        a cell whose weight carries scales is then not this kernel's."""
        if self.serves is not None and self.serves.weight_granularity:
            return self.serves.weight_granularity
        if self.weight_group is None:
            return ()
        return ("per-tensor",) if self.weight_group == 0 else (
            f"per-group:{self.weight_group}",)

    def capabilities(self):
        """What this kernel publishes about itself, or None while it is still
        on a hand-written gate — an undeclared scope is reported as unknown
        rather than guessed at."""
        if self.serves is None:
            return None
        out = self.serves.describe()
        if self.disabled_reason:
            out["disabled_reason"] = self.disabled_reason
        if self.unreached_reason:
            # The claim and its qualification travel together: a consumer
            # reading "serves weight-only FP8" has no other way to learn
            # that no checkpoint of this corpus hands it such a cell, and a
            # claim read without that reads as coverage.
            out["unreached_reason"] = self.unreached_reason
        if self.computes_at:
            # Published, because "serves an fp8 cell" and "multiplies in fp8"
            # are different promises and a consumer reading the capability
            # record has no other way to tell them apart.
            out["computes_at"] = dict(self.computes_at)
        return out

    def actual_computes_at(self, inv: Invocation) -> Mapping[str, str]:
        """The precision this build computes `inv` at, resolved or static."""
        if self.computes_at_for is not None:
            return self.computes_at_for(inv)
        return self.computes_at or {}

    def impl_id(self, inv: Invocation) -> str:
        """`kernel_impl` as it will be recorded for this invocation — the
        registered name plus whatever parameters resolve to. This is the
        implementation's identity, and it is what coverage must ask about."""
        if self.impl_params is None:
            return self.kernel_impl
        params = self.impl_params(inv) or ""
        # `@` so the name stays readable and splittable where it is shown and
        # grouped — concatenated, `sglang_dilated_conv1d_update` + `dilation=3`
        # read as `…updatedilation=3`. It was NOT free to settle: rows had
        # already been published without it (38,995 on the mirrors as of
        # 2026-10-30, `flashinfer_paged_verify_b64` and
        # `flashmla_decode_b64`), so one implementation has two recorded
        # spellings and nothing may read the name by splitting on `@`.
        # `catalog_kernel` classifies a recorded name against the registry
        # and is the only thing that may; three by-implementation tools cut
        # at the character instead and each reported a kernel measured under
        # the older spelling as never run.
        return f"{self.kernel_impl}@{params}" if params else self.kernel_impl

    def convert_inputs(self, inv: Invocation, inputs: Dict[str, Any]
                       ) -> Dict[str, Any]:
        """Canonical tensors → this kernel's launch layout (permute/pack)."""
        if not self.launch_inputs:
            return inputs
        from backend.inference.op_signatures import (
            get_signature, layout_from_grouping)
        sig = get_signature(inv.signature)
        if sig is None:
            raise ValueError(
                f"{self.library}:{self.kernel_impl} declares launch_inputs "
                f"but signature {inv.signature!r} is unknown")
        src_by_name = {lay.name: lay for lay in sig.layouts_for(inv.dims)}
        out = dict(inputs)
        for name, grouping in self.launch_inputs:
            if name not in out:
                raise ValueError(
                    f"{self.library}:{self.kernel_impl} launch tensor "
                    f"{name!r} is not in the canonical inputs")
            src = src_by_name.get(name)
            if src is None:
                raise ValueError(
                    f"{self.library}:{self.kernel_impl} launch tensor "
                    f"{name!r} is not in this invocation's layout")
            dst = layout_from_grouping(src, grouping)
            if dst is None:
                raise ValueError(
                    f"{self.library}:{self.kernel_impl}: grouping {grouping} "
                    f"is not a permute/pack of canonical {name} {src.axes}")
            out[name] = apply_layout(out[name], src, dst, inv.dims)
        return out

    def wrap_output(self, inv: Invocation,
                    to_canonical: Optional[Callable[[Any], Any]]
                    ) -> Optional[Callable[[Any], Any]]:
        """Invert the named input's launch grouping onto the kernel output."""
        if self.launch_output is None:
            return to_canonical
        from backend.inference.op_signatures import (
            get_signature, layout_from_grouping)
        sig = get_signature(inv.signature)
        if sig is None:
            raise ValueError(
                f"{self.library}:{self.kernel_impl} declares launch_output "
                f"but signature {inv.signature!r} is unknown")
        src_by_name = {lay.name: lay for lay in sig.layouts_for(inv.dims)}
        name = self.launch_output
        src = src_by_name.get(name)
        grouping = dict(self.launch_inputs).get(name)
        if src is None or grouping is None:
            raise ValueError(
                f"{self.library}:{self.kernel_impl} launch_output {name!r} "
                f"is not a launched canonical tensor")
        launch_lay = layout_from_grouping(src, grouping)
        if launch_lay is None:
            raise ValueError(
                f"{self.library}:{self.kernel_impl}: grouping {grouping} "
                f"is not a permute/pack of canonical {name} {src.axes}")
        inner = to_canonical if to_canonical is not None else (lambda x: x)

        def _wrapped(out):
            return apply_layout(inner(out), launch_lay, src, inv.dims)
        return _wrapped

    def unavailability(self, env: Env):
        """Why this kernel cannot run in `env` at all, clause by clause —
        `available` read for its reasons, as `refusals` is for `accepts`."""
        if not env.has(self.import_name):
            yield f"{self.import_name} not importable"
            return
        if not self._version_ok(env):
            scope = str(self._version_spec())
            from packaging.requirements import Requirement
            from packaging.specifiers import SpecifierSet
            one = self.lib.spec_for(self.import_name)
            library = (Requirement(one).specifier
                       if one else SpecifierSet())
            kernel = SpecifierSet(self.version_specifier)
            if library and kernel:
                # The intersection can read as a duplicated pin (`>=2.5,>=2.9`)
                # when the second constraint is this kernel's narrower floor.
                # Say whose it is: one is the library every kernel shares,
                # the other is this kernel's own narrower requirement.
                scope += (f" (library floor {library}; "
                          f"this kernel's floor {kernel})")
            yield (f"{self.import_name} {env.version_of(self.import_name)} is "
                   f"outside {scope}")
        if not self._sm_ok(env):
            yield f"{env.sm} is below sm_{self.min_sm}"

    def available(self, env: Env) -> bool:
        return next(self.unavailability(env), None) is None

    def declines(self, inv: Invocation, env: Env):
        """Why this kernel will not measure `inv` in `env`, lazily: another
        library's env, then `unavailability`, then `refusals` — the one gate
        a run takes (`applicable`), and the one a plan asks ahead of it of
        the env a box DECLARES (`box.declared_envs`). A gate that raises
        propagates; the caller says so."""
        if env.library is not None and self.library != env.library:
            yield f"measured in the {self.library} env"
            return
        why = next(self.unavailability(env), None)
        if why is not None:
            yield why
            return
        yield from self.refusals(inv, env)

    def _sm_ok(self, env: Env) -> bool:
        """False only when this card is provably below the kernel's floor.

        An unknown or unparseable `sm` allows, matching `_version_ok`: the
        harness declines what it can prove it cannot run, and does not invent
        a reason to skip otherwise.
        """
        if self.min_sm is None or not env.sm:
            return True
        digits = "".join(c for c in str(env.sm) if c.isdigit())
        if not digits:
            return True
        return int(digits) >= int(self.min_sm)

    def _version_spec(self):
        """The releases this kernel may run on: its library's pip spec for
        the gated module, narrowed by its own `version_specifier`. Text that
        does not parse raises rather than reading as "no constraint" — that
        is a declaration being wrong."""
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        one = self.lib.spec_for(self.import_name)
        spec = Requirement(one).specifier if one else SpecifierSet()
        return spec & SpecifierSet(self.version_specifier)

    def _version_ok(self, env: Env) -> bool:
        """True unless the installed version is PROVEN outside the spec.

        An unknown installed version allows — importability already passed,
        and inventing a reason to skip is the failure the sm gate also
        refuses."""
        spec = self._version_spec()
        if not spec:
            return True
        installed = env.version_of(self.import_name)
        if not installed or installed == "unknown":
            return True
        from packaging.version import InvalidVersion, Version
        try:
            version = Version(installed)
        except InvalidVersion:
            return True                        # unknown, same as absent
        return spec.contains(version, prereleases=True)

    def version(self, env: Env) -> Optional[str]:
        return env.version_of(self.import_name)


_PROVIDERS: List[Provider] = []
_DISCOVERED = False


def apply_layout(t: Any, src: Any, dst: Any, dims: Dict[str, int]) -> Any:
    """Permute/pack `t` from `src` layout to `dst`. Unique or raise.

    Extents come from the tensor; `dims` only splits a packed physical dim.
    A no-op when the two layouts name the same axes in the same order.
    """
    from backend.inference.op_signatures import (
        layout_conversion, packed_shape, unpacked_shape)
    perm = layout_conversion(src, dst)
    if perm is None:
        raise ValueError(
            f"no unique permute/pack from {src.axes} to {dst.axes}")
    if src.axes == dst.axes:
        return t
    physical = tuple(int(x) for x in t.shape)
    unpacked = unpacked_shape(src.axes, physical, dims)
    x = t.reshape(unpacked)
    if perm != tuple(range(len(perm))):
        x = x.permute(*perm)
        unpacked = tuple(unpacked[i] for i in perm)
    return x.reshape(packed_shape(dst.axes, unpacked)).contiguous()


def _canonical_layouts_named(sig: Any, name: str) -> List[Any]:
    return [lay for lay in (*sig.inputs, *sig.variant_inputs) if lay.name == name]


def _check_launch_layouts(p: Provider) -> None:
    """A declared grouping must be a permute/pack of the contract tensor.

    Checked here so a kernel that cannot convert fails at registration, not
    on the first cell it would have measured.
    """
    if not p.launch_inputs and p.launch_output is None:
        return
    from backend.inference.op_signatures import (
        get_signature, layout_from_grouping)
    sig = get_signature(p.signature)
    if sig is None:
        raise ValueError(
            f"{p.library}:{p.kernel_impl} declares a launch layout but "
            f"signature {p.signature!r} is unknown")
    for name, grouping in p.launch_inputs:
        srcs = _canonical_layouts_named(sig, name)
        if not srcs:
            raise ValueError(
                f"{p.library}:{p.kernel_impl} launch tensor {name!r} "
                f"is not a contract input of {p.signature!r}")
        for src in srcs:
            if layout_from_grouping(src, grouping) is None:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: grouping {grouping} "
                    f"is not a permute/pack of canonical {name} {src.axes}")
    if p.launch_output is None:
        return
    launched = {name for name, _g in p.launch_inputs}
    if p.launch_output not in launched:
        raise ValueError(
            f"{p.library}:{p.kernel_impl} launch_output "
            f"{p.launch_output!r} is not in launch_inputs")


def register(p: Provider) -> None:
    if (p.serves is None) == (p.applies_to is None):
        raise ValueError(
            f"{p.library}:{p.kernel_impl} must declare exactly one of "
            f"`serves` or `applies_to` — two ways to state a kernel's scope "
            f"is two places for them to disagree")
    # Raises for a library nobody declared. This is the link that makes the
    # split hold: `library` is no longer a free-text label a typo can invent,
    # it names a record, and a kernel of an undeclared library cannot reach the
    # registry at all rather than reaching it with an env that installs nothing.
    get_library(p.library)
    p._version_spec()            # InvalidSpecifier: the declaration is wrong
    _check_launch_layouts(p)
    _check_paged_declaration(p)
    _check_computes_at(p)
    _check_dtype_slots(p)
    _check_formats(p)
    _check_operand_enum(p)
    _check_closed_world(p)
    _PROVIDERS.append(p)


def _check_closed_world(p: Provider) -> None:
    """The declaration speaks for every slot and axis the signature leaves
    open, so `refusals` never has to read silence as a claim.

    * every material dtype slot (`OpSignature.dtype_slots`) lists at least
      one token — an empty one would serve no cell, which is a disablement
      (`disabled_reason`), not a declaration;
    * a kernel that computes its activation in another dtype
      (`computes_at`) says what it writes, since the default — the cell's
      activation — is then not what the launch writes;
    * `variant_dims` and `requires_dims` name the signature's own variant
      axes, each once: an axis of the default geometry needs no mention,
      and one named in both is stated twice.
    """
    if p.serves is None:
        return
    from backend.inference.op_signatures import get_signature
    sig = get_signature(p.serves.signature)
    if sig is None:
        raise ValueError(f"{p.library}:{p.kernel_impl} serves unknown "
                         f"signature {p.serves.signature!r}")
    for slot in sig.dtype_slots:
        if not p.serves.admits(slot):
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: {sig.name!r} reads dtype slot "
                f"{slot!r} and the declaration lists no token for it. A "
                f"declaration is closed — list the dtypes the build handles "
                f"(`weight` / `activation` / `output`, or `dtype_slots` for "
                f"any other slot).")
    if (p.computes_at and "activation" in p.computes_at
            and not p.serves.output):
        raise ValueError(
            f"{p.library}:{p.kernel_impl} computes at activation "
            f"{p.computes_at['activation']!r}; declare the `output` it "
            f"writes — the default is the cell's activation")
    named = p.serves.served_variants()
    if len(set(named)) != len(named):
        raise ValueError(
            f"{p.library}:{p.kernel_impl}: a variant axis is named twice "
            f"across variant_dims {p.serves.variant_dims} and requires_dims "
            f"{p.serves.requires_dims}; a required axis is already served")
    stray = [n for n in named if n not in sig.variant_dims]
    if stray:
        raise ValueError(
            f"{p.library}:{p.kernel_impl}: {stray} are not variant axes of "
            f"{sig.name!r} (its variant_dims: {list(sig.variant_dims)}); an "
            f"axis of the default geometry needs no declaration")


def _check_formats(p: Provider) -> None:
    """A declared cache record (`output_format`, `kv_format`) names a
    registry cache format, once, whose element dtype the declaration admits
    in the dtype slot it stands beside (`output`, the `kv` dtype slot).

    The cell's format slots are validated against the same registry
    (`benchmarks._check_dtype_env_json`), so a name outside it is a gate no
    cell can pass — checked here, where a typo still says which provider
    to fix, as `dtype_slots` tokens are checked against the corpus's.
    `benchmark_identity.PLAIN` (the cell that states no format) is the one
    entry that names no format: it admits the plain tensor, whatever its
    dtype."""
    if p.serves is None:
        return
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    from backend.inference.benchmark_identity import FORMAT_SLOTS, PLAIN
    for slot, names in p.serves.formats().items():
        if len(set(names)) != len(names):
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: {slot} names a format "
                f"twice: {names}")
        admitted = p.serves.admits(FORMAT_SLOTS[slot])
        for name in names:
            if name is PLAIN:
                continue
            fmt = CACHE_FORMATS.get(name) if isinstance(name, str) else None
            if fmt is None:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: {slot} {name!r} is not "
                    f"a cache format; named formats are "
                    f"{sorted(CACHE_FORMATS)} (backend/arch/cache_formats/)")
            token = corpus_token(fmt.dtype)
            if admitted and token not in admitted:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: {slot} {name!r} stores "
                    f"{token} elements, which its `{FORMAT_SLOTS[slot]}` "
                    f"{admitted} refuses")


def _check_operand_enum(p: Provider) -> None:
    """An `operand_enum` names a contract tensor in EVERY geometry of the
    signature and an axis it has, with a non-empty set of extents — checked
    at registration, so a typo says which provider to fix instead of
    raising out of the gate on the first cell."""
    if p.serves is None or not p.serves.operand_enum:
        return
    from backend.inference.op_signatures import get_signature
    sig = get_signature(p.serves.signature)
    if sig is None:
        raise ValueError(
            f"{p.library}:{p.kernel_impl}: operand_enum on unknown "
            f"signature {p.serves.signature!r}")
    geometries = [sig.inputs] + ([sig.variant_inputs]
                                 if sig.variant_inputs else [])
    for tensor, axis, allowed in p.serves.operand_enum:
        if not allowed:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: operand_enum[{tensor!r}] is "
                f"empty; drop the entry instead of refusing every cell")
        for layouts in geometries:
            by_name = {lay.name: lay for lay in layouts}
            if tensor not in by_name:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: operand_enum names "
                    f"{tensor!r}, not a contract tensor of "
                    f"{p.serves.signature!r} in every geometry")
            rank = len(by_name[tensor].axes)
            if not -rank <= axis < rank:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: operand_enum axis {axis} "
                    f"is outside {tensor!r}'s rank {rank}")


def _check_dtype_slots(p: Provider) -> None:
    """A dtype-slot constraint names a real slot once, in corpus tokens.

    `weight`, `activation` and `output` keep their own fields: stating one
    of them here as well would be two channels for one gate, which is the
    drift `KernelSignature` exists to remove. Checked at registration, where
    a typo still says which provider to fix.
    """
    if p.serves is None:
        return
    from backend.arch.dtypes import CORPUS_TOKENS
    from backend.inference.benchmarks import DTYPE_SLOTS
    # The slots `KernelSignature` gives fields of their own: its fields that
    # are dtype slots, read off the dataclass rather than listed again.
    first_class = DTYPE_SLOTS & {f.name for f in dataclasses.fields(KernelSignature)}
    seen = set()
    for slot, tokens in p.serves.dtype_slots:
        if slot in first_class or slot not in DTYPE_SLOTS:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: dtype_slots names {slot!r}; "
                f"the slot must be one of {sorted(DTYPE_SLOTS)} and not a "
                f"first-class field ({sorted(first_class)} have their own)")
        if slot in seen:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: dtype_slots constrains "
                f"{slot!r} twice — one gate per slot")
        seen.add(slot)
        if not tokens:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: dtype_slots[{slot!r}] is "
                f"empty; drop the entry instead of refusing every cell")
        for token in tokens:
            if token not in CORPUS_TOKENS:
                raise ValueError(
                    f"{p.library}:{p.kernel_impl}: dtype_slots[{slot!r}] "
                    f"holds {token!r}, not a corpus dtype token")
    for name, dtype in p.serves.secondary_outputs:
        if not name:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: secondary_outputs names an "
                f"empty tensor")
        if dtype not in CORPUS_TOKENS:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: secondary_outputs[{name!r}]="
                f"{dtype!r} is not a corpus dtype token")


def _check_computes_at(p: Provider) -> None:
    """A declared fallback precision uses the corpus's own dtype vocabulary.

    The value lands in a row's `measured_dtype_env_json`, which the schema
    validates against `DTYPE_SLOTS` / `CORPUS_TOKENS`; catching a typo at
    registration says which provider to fix, where the schema can only say
    the row is bad after the GPU time was spent.
    """
    if p.computes_at is None:
        return
    from backend.arch.dtypes import CORPUS_TOKENS
    from backend.inference.benchmarks import DTYPE_SLOTS
    if p.computes_at is not None and p.computes_at_for is not None:
        raise ValueError(
            f"{p.library}:{p.kernel_impl} declares both static and per-cell "
            "`computes_at`; two ways to state one precision is two places "
            "to disagree")
    if not p.computes_at:
        raise ValueError(
            f"{p.library}:{p.kernel_impl} declares an empty `computes_at`; "
            f"leave it None when the kernel computes at the cell's precision")
    for slot, token in p.computes_at.items():
        if slot not in DTYPE_SLOTS:
            raise ValueError(
                f"{p.library}:{p.kernel_impl} computes_at has unknown slot "
                f"{slot!r}; allowed: {sorted(DTYPE_SLOTS)}")
        if token not in CORPUS_TOKENS:
            raise ValueError(
                f"{p.library}:{p.kernel_impl} computes_at[{slot!r}]="
                f"{token!r} is not a corpus dtype token")


def _check_paged_declaration(p: Provider) -> None:
    """A kernel that reads a paged cache states the pages it can serve.

    Whether it reads one is the signature's own declaration — an operand
    with `role="kv"` (`OpSignature.reads_paged_cache`) — so this is the
    complement of that predicate and not a second list of attention names.
    Checked at registration, because the alternative is a cell measured at a
    page the backend cannot run: the page is part of the measurement key, so
    such a row is keyed under a paging layout that never happened.

    The gate used to be a test over the kernels that HAD declared, which is
    the one shape of check that cannot catch an omission.
    """
    if p.serves is None:
        return
    from backend.inference.op_signatures import get_signature
    sig = get_signature(p.serves.signature)
    if sig is None or not sig.reads_paged_cache:
        if p.serves.kv_block_sizes:
            raise ValueError(
                f"{p.library}:{p.kernel_impl} declares kv_block_sizes "
                f"{list(p.serves.kv_block_sizes)} but {p.serves.signature!r} "
                f"reads no cache (no `role=\"kv\"` operand), so there is no "
                f"page for a cell of it to be keyed by")
        return
    if p.serves.kv_block_sizes is None:
        raise ValueError(
            f"{p.library}:{p.kernel_impl} serves {p.serves.signature!r}, "
            f"which reads a KV cache, and says nothing about paging. "
            f"Declare `kv_block_sizes`: the pages this backend runs (its own "
            f"`get_supported_kernel_block_sizes()`), or `()` if it reads the "
            f"cache as one contiguous buffer. The page is part of the "
            f"measurement key, so an unstated one is a cell keyed under a "
            f"paging layout that never happened.")
    for page in p.serves.kv_block_sizes:
        if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
            raise ValueError(
                f"{p.library}:{p.kernel_impl}: kv_block_sizes holds "
                f"{page!r}; a page is a positive number of tokens")


def discover() -> None:
    """Import provider submodules so they self-register. Idempotent.
    Import-guarded per module: a submodule that needs a missing library
    still imports (it just registers providers whose `available()` is
    False in this env)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    from . import attention   # noqa: F401  (self-registers on import)
    from . import gemm         # noqa: F401
    from . import sglang_mla   # noqa: F401
    from . import linear_ssm   # noqa: F401
    from . import norm_act     # noqa: F401
    from . import rope         # noqa: F401  (rope + fused_qk_norm_rope)
    from . import moe_gate     # noqa: F401
    from . import mhc          # noqa: F401  (the hyper-connection seam)
    from . import hc           # noqa: F401  (the other hyper-connection boundaries)
    from . import indexer      # noqa: F401
    from . import conv         # noqa: F401  (conv2d torch baseline)
    from . import quantise     # noqa: F401  (the activation cast, both ways)
    from . import compress_store  # noqa: F401  (the KV compressors' spans)
    from . import deepinfer  # noqa: F401  (DeepSeek's V4.1 runtime stack)
    _DISCOVERED = True


def served_kv_block_sizes(signature: Optional[str] = None) -> Tuple[int, ...]:
    """Every paged-KV page some registered backend can serve, ascending —
    of `signature`'s backends when one is named. A page only another
    family's backend serves is no page this kernel runs at: vLLM forces an
    MLA cache to FlashMLA's 64 (`platforms/cuda.py`), and the union keyed
    DeepSeek's MLA decode at FlashInfer's 16 too.

    The pages a sweep measures at, derived from the declarations rather than
    listed beside the sweep: a page no backend can run is a page no
    deployment can run, and a measurement of it is a row nothing will ever
    look up. Each paged provider declares its own set in
    `KernelSignature.kv_block_sizes`, mirrored from that backend's
    `get_supported_kernel_block_sizes()`.

    Over `all_providers()`, which DISCOVERS: the union has to be of what is
    registered, not of whichever modules the caller happened to import. The
    sweep imported only `attention`, so DSA's indexer page reached the union
    by the accident of another import pulling `indexer` in, and a page
    declared by a module added later would have been missed with nothing
    failing.
    """
    pages = {page for provider in all_providers()
             if signature is None or provider.signature == signature
             for page in ((provider.serves.kv_block_sizes or ())
                          if provider.serves else ())}
    return tuple(sorted(pages))


def all_providers() -> List[Provider]:
    discover()
    return list(_PROVIDERS)


def catalog_kernel(kernel_impl: str) -> Optional[str]:
    """The registered kernel a recorded `kernel_impl` names, or None.

    A row records `Provider.impl_id` — the registered name plus whatever
    parameters resolved (`_b64`, `_c128`) — so it belongs to the longest
    registered name it starts with (`torch_gelu_tanh`, not `torch_gelu`).
    Registered names are unique across libraries, so the kernel alone
    decides; which framework dispatched it does not. The one classifier:
    stored on every row, read by the Kernel Bench filter and coverage."""
    for name in _registered_longest_first():
        if kernel_impl.startswith(name):
            return name
    return None


def kernel_library(kernel: str) -> str:
    """The library a registered kernel belongs to."""
    return _library_by_kernel()[kernel]


def _library_by_kernel() -> Dict[str, str]:
    return {p.kernel_impl: p.library for p in all_providers()}


def _registered_longest_first() -> Tuple[str, ...]:
    discover()
    return _longest_first(len(_PROVIDERS))


@lru_cache(maxsize=4)
def _longest_first(n_registered: int) -> Tuple[str, ...]:
    # Keyed by the registry's size: a provider registered later is seen.
    return tuple(sorted(_library_by_kernel(), key=lambda n: (-len(n), n)))


def dependency_manifest() -> Dict[str, Dict[str, Any]]:
    """`library -> {import_names, pip, providers, source_build, build_hint}`.

    Two halves from two owners, joined here and nowhere else: the install facts
    come from the `Library` record, the import gates from the kernels that
    registered under it. Only libraries with at least one kernel appear — an
    env for a library nothing measures is an env nobody should build."""
    discover()
    out: Dict[str, Dict[str, Any]] = {}
    for p in _PROVIDERS:
        lib = get_library(p.library)
        m = out.setdefault(p.library, {"import_names": set(),
                                       "pip": tuple(lib.pip),
                                       "providers": [],
                                       "source_build": lib.source_build,
                                       "build_hint": lib.build_hint,
                                       "no_build_isolation": None})
        if lib.no_build_isolation and m["no_build_isolation"] is None:
            # The PACKAGE whose build must see the env (its setup.py imports
            # torch) — the first pip spec's name.
            from packaging.requirements import Requirement
            m["no_build_isolation"] = Requirement(lib.pip[0]).name
        m["import_names"].add(p.import_name)
        m["providers"].append(p.kernel_impl)
    return out


def probe_import_names() -> set:
    """Every module name to probe for presence/version — union across
    providers plus torch (the base). Consumed by `env.detect_env`."""
    names = {p.import_name for p in all_providers()}
    names.add("torch")
    return names


def render_env_files() -> Dict[str, str]:
    """Generate the `envs/*.txt` pip requirement files from the provider
    manifest — one per library stack, torch's own included: torch is the base
    every env installs, and ALSO a library whose providers (the floors) are
    measured in one env, its own, like any other library's. The files are
    DERIVED, so the `Library` record stays the single source."""
    m = dependency_manifest()
    torch_pip = "\n".join(get_library("torch").pip)
    # The harness's own runtime deps, in EVERY env: `kernel_bench run`
    # imports `backend.inference.benchmarks` (row validation on persist),
    # which reads the dim registry through `arch` — pydantic for the
    # schema, yaml for the registry file, and huggingface_hub, which `arch`
    # imports for config metadata (`dim_resolution` → `hf.metadata`), and
    # httpx, which `--skip-existing` imports to ask Kernel Bench
    # (`backend.measure.lookup`). The source-build envs have nothing else to
    # bring either in, so without them every one of them failed at import.
    # Minimal, but not zero.
    harness_pip = ("pydantic", "pyyaml", "huggingface-hub", "httpx")
    files: Dict[str, str] = {}
    for lib, info in sorted(m.items()):
        # torch's env names its own pip spec once, not twice.
        pip = [ln for ln in info["pip"] if ln not in get_library("torch").pip]
        if info["source_build"]:
            # Still emit a file. Skipping these left a fresh environment with
            # NOTHING for them — not a wrong instruction, an absent one, which
            # is why `fla` sat at 311 failed builds against a dependency the
            # repo never described. A library pip cannot resolve is a
            # provisioning gap, and a gap has to be written down to be closed.
            lines = [
                f"# Provider env: {lib} — SOURCE BUILD, provisioned OUT OF BAND.",
                "# GENERATED from the provider registry "
                "(`python -m kernel_bench envs`). Edit the provider, not this file.",
                f"# Import gate (what availability checks, and what the build "
                f"actually needs): {', '.join(sorted(info['import_names']))}",
                f"# Providers: {', '.join(sorted(info['providers']))}",
                "#",
                *[f"# {ln}" for ln in (info["build_hint"] or
                                       "NO BUILD HINT — this library cannot be "
                                       "provisioned from this repo.").split(". ")],
                "#",
                "# The lines below are what pip CAN install. If that is only "
                "torch, this",
                "# library is not reproducible from the repo and someone has to "
                "build it.",
                torch_pip,
                *harness_pip,
                *pip,
            ]
            files[f"{lib}.txt"] = "\n".join(lines) + "\n"
            continue
        lines = [
            f"# Provider env: {lib} — GENERATED from the library registry "
            f"(`python -m kernel_bench envs`). Edit the provider, not this file.",
            "# Measure just this env (uv resolves+caches the deps, no venv):",
            f"#   PYTHONPATH=. uv run --no-project --with-requirements "
            f"kernel_bench/envs/{lib}.txt \\",
            f"#       -- python -m kernel_bench run --jobs jobs.jsonl "
            f"-o results/{lib}.csv --library {lib} --skip-existing",
            "# Or every provider env in one command:  "
            "python -m kernel_bench run-all --jobs jobs.jsonl",
            f"# Providers: {', '.join(sorted(info['providers']))}",
            torch_pip,
            *harness_pip,
            *pip,
        ]
        files[f"{lib}.txt"] = "\n".join(lines) + "\n"
    return files


def applicable(inv: Invocation, env: Env) -> List[Provider]:
    """Providers eligible to measure this invocation in this env."""
    discover()
    out = []
    for p in _PROVIDERS:
        if p.signature != inv.signature:
            continue
        try:
            if next(p.declines(inv, env), None) is not None:
                continue
        except Exception as e:
            # A gate that RAISES is a broken provider, not an inapplicable
            # one, and swallowing the two together hides the first as the
            # second: when `accepts` did not exist yet, every migrated kernel
            # vanished from measurement silently. Still non-fatal — one bad
            # provider must not sink a sweep — but it says so.
            import sys
            print(f"provider {p.library}:{p.kernel_impl} gate raised "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        out.append(p)
    return out


def reads_the_key(inv: Invocation) -> Optional[str]:
    """`None` when SOME registered kernel's declaration reads this cell's
    dtype environment, else the slot values nothing reads.

    Two different things look alike in a plan's gap list and only one of
    them is the corpus's fault:

    * a kernel accepts the cell and this BOX cannot run it — DeepGEMM's
      FP8xFP4 grouped GEMM wants `sm_100` and the box is an H100. The
      cell is a real gap: measuring it needs the card, and it stays in
      the coverage denominator until somebody has one.
    * NO kernel anywhere reads the cell at all. Then the key describes a
      launch nothing in the corpus performs, and that is a cost-model
      error rather than a missing measurement — it is what pricing an
      FP8 activation against an MXFP4 weight on a card with no FP4 MMA
      looked like, 4,094 cells of a launch no framework runs there.

    Asked of the DECLARATIONS alone (`KernelSignature.accepts`), so the
    answer does not move with which libraries a box installed. A kernel
    still gated by a hand-written `applies_to` cannot answer a question
    about declarations, so it withholds the claim rather than supporting
    it: this says "unreadable" only when every kernel of the signature
    has declared its scope and none of them covers this cell.
    """
    discover()
    mine = [p for p in _PROVIDERS if p.signature == inv.signature]
    if not mine or any(p.serves is None for p in mine):
        return None
    if any(p.serves.accepts(inv) for p in mine):
        return None
    slots = {slot: value for slot, value in sorted(inv.dtype_env.items())
             if value}
    return ", ".join(f"{slot}={value}" for slot, value in slots.items())


def candidates(inv: Invocation, envs: Mapping[str, Env]
               ) -> Tuple[List[Provider], Dict[str, str]]:
    """The providers that would measure `inv` on a box whose library envs
    are `envs` (library -> the env declared for it, `box.declared_envs`),
    and why each other provider of its signature would not.

    `applicable`'s gate, asked per provider of the env its own library owns,
    so a plan decides from the declarations what the run will decide again
    from the real env. Also the run's resolved parameters: a kernel whose
    `impl_params` declines the cell is skipped by the driver before any
    input exists, and is no candidate here."""
    discover()
    chosen: List[Provider] = []
    why: Dict[str, str] = {}
    for p in _PROVIDERS:
        if p.signature != inv.signature:
            continue
        env = envs.get(p.library)
        if env is None:
            why[p.kernel_impl] = f"no {p.library} env on this box"
            continue
        try:
            reasons = list(p.declines(inv, env))
        except Exception as e:      # as `applicable` treats it: not run
            reasons = [f"gate raised {type(e).__name__}: {e}"]
        if not reasons:
            try:
                p.impl_id(inv)
            except NotImplementedError as e:    # the driver's decline; any
                reasons.append(f"impl_params: {e}")  # other errors are bugs
        if not reasons:
            chosen.append(p)
        else:
            why[p.kernel_impl] = "; ".join(reasons)
    return chosen, why


# Last, so every name above exists: the library records self-register, and
# `register()` can therefore reject a kernel naming one that does not.
from . import _libraries    # noqa: F401
