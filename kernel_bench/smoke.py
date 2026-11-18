"""Layer 3 of the declaration guarantee: run what every kernel claims it serves.

Layers 1 and 2 (`register`) are structural — a declaration can be well-formed,
uniquely named, and still false. `serves` says "this kernel handles these
(weight, activation) combinations at these shapes"; nothing so far makes that
statement meet the kernel. This module does: for every declared combination it
synthesizes the smallest legal cell, builds the kernel, runs it once, and
compares against the signature's fp32 reference.

That is the layer that catches what the others cannot. Every one of these was a
declaration wider than its build, and none of them announced itself:

  * humming accepting bf16 — its kernel asserts <=8 bits, with an empty message
  * marlin fp4 with a mis-named scale attribute — `Invalid thread config`
  * e5m2 packed with an e4m3 scale — runs fine, numbers 128x off
  * GPTQ's uniform int4 grid used for E2M1's non-uniform one — plausibly lossy

The last two are the reason "did not crash" is not the criterion. A wrong
NUMBER is the only unambiguous evidence, so it is the only thing this module
asserts on. An exception during build is REPORTED but not judged: on a given
box it may equally mean "the declaration is wrong" or "this kernel wants
Hopper", and guessing between those would either hide real lies or fail CI on
correct code. What the caller gets is the full table plus the two facts that
are unambiguous — nothing declared produced wrong numbers, and nothing declared
went untried.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from .env import Env
from .invocation import Invocation
from .kernel_symbol import describe as describe_kernel
from .providers import Provider, all_providers
from . import quant as _quant
from .driver import _act_torch_dtype
from .refs import get_ref, input_generator


def _smoke_cell(p, dims, dtype_env, phase, kernel_params) -> Invocation:
    """The hand-stated cell this smoke check runs, key included.

    A measured cell takes its key from the plan (`Invocation.physical_key`,
    which the driver refuses to assemble) because a row has to be stored
    under the key an estimate asks with. A smoke cell is stated here whole —
    its dims, its dtype env, its phase — goes in no corpus, and is asked
    only whether the kernel runs and agrees with the oracle; so its key is
    stated here too, and the inputs it draws are that cell's own
    (`Invocation.input_seed`), fixed across reruns like a measurement's.
    """
    canonical = {"sort_keys": True, "separators": (",", ":")}
    return Invocation(
        signature=p.signature, op=p.signature, dims=dims,
        dtype_env=dtype_env, phase=phase, tp=1,
        kernel_params=kernel_params,
        physical_key=(p.signature, json.dumps(dims, **canonical),
                      json.dumps(dtype_env, **canonical), phase))
from .signatures import get_signature


# The smallest value that is a legal, representative extent for each axis.
# Deliberately explicit and exhaustive rather than a computed default: a dim
# absent from this table makes the cell UNSYNTHESIZABLE and is reported as
# such. Inventing a plausible size for an axis nobody declared is how a kernel
# ends up measured against a shape that does not exist — the DSA indexer was
# benched at d_head = seq_len for 2,430 rows exactly that way.
MINIMAL_DIMS: Dict[str, int] = {
    "B": 1,
    "s_query": 64,
    "s_ctx": 128,
    "s_kv": 128,
    "n_heads": 8,
    "n_kv_heads": 2,
    "d_head": 64,
    "d_model": 256,
    "d_ffn": 512,
    # The two widths a q / latent-KV norm pair normalises over. Unequal, so a
    # kernel that swapped the rows (or sized both by one) does not pass; and
    # as DeepSeek-V4.1's a-norm pair has them, the q-LoRA row a whole number
    # of MXFP8 groups at 128 and the latent KV row the model's 512-wide head.
    "d_q_lora": 128,
    "d_kv_lora": 512,
    "n_vocab": 256,
    "n_experts": 8,
    "n_experts_active": 2,
    "n_topk_keys": 32,
    # The group a fused collective reduces over. 2 is the smallest that is a
    # collective at all: at 1 the all-reduce is the identity.
    "n_comm_ranks": 2,
    # A bare collective's payload, as one flat buffer (NCCL's `count`), and
    # the spacing of its group on the mesh. 1 is the adjacent-rank case every
    # tensor-parallel collective has; a wider stride is an outer axis
    # composed over an inner one.
    "n_comm_elems": 256,
    "n_comm_stride": 1,
    # A routed exchange's fan-out (`routed_all_to_all`): how many
    # destinations a token's top-k reaches, which is how many copies of it
    # the dispatch sends. `n_experts_active` above is the top-k width; the
    # two below are the destinations it draws from and the ones that hold
    # weights. Unequal, so a cell that confused the denominator with the
    # weighted count does not pass: 8 of the 10 destinations are experts,
    # the other 2 a zero-compute reserve.
    "n_route_targets": 10,
    "n_route_experts": 8,
    "n_indexer_heads": 4,
    "d_indexer_head": 64,
    "s_sparse_block": 32,
    "n_lin_k_heads": 4,
    "n_lin_v_heads": 4,
    "d_lin_k_head": 128,
    "d_lin_v_head": 128,
    "n_ssm_heads": 4,
    "d_ssm_head": 64,
    "d_ssm_state": 128,
    # The chunk the Mamba-2 scan tiles the sequence into. 32 divides the
    # table's `s_query` of 64, so the minimal cell is two whole tiles —
    # a chunk that does not divide the query block exercises the kernel's
    # ragged-tail path, which is not what a smoke cell is for.
    "ssm_chunk_size": 32,
    # The paged-KV page, an axis of every signature that reads a paged cache
    # (`op_signatures.PAGE_AXIS`). 64 is a page every paged backend but
    # CUTLASS MLA serves (FlashInfer, FlashInfer MLA, FlashMLA); a paged
    # backend that serves no divisor of it declines the cell, as in a
    # measurement, while a contiguous floor (`kv_block_sizes=()`) reads no
    # page and still runs it.
    "kv_block_size": 64,
    "d_conv_channels": 256,
    "s_conv_kernel": 4,
    # The causal conv's tap spacing. 1 is every mixer conv and the only
    # spacing the framework kernels take; the dilated form is the torch
    # floor's alone.
    "dilation": 1,
    # conv2d (Step-3.7 ViT downsamplers). s_conv_grid is a perfect square
    # (12²=144) — the flattened H·W patch grid the reference reshapes to
    # [C_in, H, W], so the built tensor's length equals the declared extent.
    "C_in": 8,
    "C_out": 6,
    "kernel_size": 3,
    "stride": 2,
    "s_conv_grid": 144,
    # Total routed rows across all experts in a grouped MoE cell. The base
    # value is E×s_expert_rows at the table's own extents; the repair below
    # re-derives it after any per-kernel constraint moves n_experts or
    # s_expert_rows, because the default token counts (s_expert_rows per
    # expert) must sum to exactly this.
    "n_tokens": 512,
    # Padded rows one expert of a routed cell holds — the M of the
    # [n_experts, M, d] buffer, named apart from the dispatch's token side
    # (`s_query`) so moe_permute states both.
    "s_expert_rows": 64,
    # Axes `make_inputs` reads that the signature does not list as required.
    # That gap is real (a signature can under-declare what its inputs need) and
    # is what the discovery loop below walks; the extents still have to be
    # declared here, so a newly-needed axis surfaces as `no_cell` naming itself
    # rather than as a plausible guess.
    "d_reduce": 256,
    "d_out": 256,
    "d_rope": 64,
    "d_v_head": 64,
    # Three-axis MRoPE at the minimal rotary width: [11, 11, 10] owns all 32
    # pairs of a 64-channel rotation; smoke takes the interleaved lane layout,
    # the variant whose flag is present rather than absent.
    "mrope_section_t": 11,
    "mrope_section_h": 11,
    "mrope_section_w": 10,
    "mrope_interleaved": 1,
    # MiniMax-M3's index-Q head count beside its one index-K head.
    "n_index_heads": 4,
    # The KV compressors' group size and pooled-slot count (the
    # compress/store spans): the ratio is the window a pool closes over, and
    # 64 query tokens is 32 slots of it.
    "compress_ratio": 2,
    "s_slots": 32,
    # DeepSeek-V4's overlap-compressor window (its compress_ratio is this
    # same 4); the overlap span names the axis as the spec does.
    "Cape": 4,
    "n_ssm_groups": 1,
    "n_expert_groups": 1,
    "n_groups_active": 1,
    # Variants on gqa_attention: stripped from the default (full-attn) cell,
    # bound only when a sliding-window / sink / asymmetric-V job carries them.
    "s_window": 32,
    "n_attn_sinks": 8,
    # DSv4 Hyper-Connection: d_hc_mix = (2 + n_hidden_copies) * n_hidden_copies.
    "n_hidden_copies": 4,
    "d_hc_mix": 24,
    # Kimi-K3 AttnRes stack depth (ceil(n_layers / block_size)).
    "Sar": 4,
}

# Relations a per-axis table cannot express. Each is a real constraint of the
# op, not a convenience: a cell violating one is not a smaller cell, it is an
# illegal one, and the kernel would be within its rights to produce anything.
_REPAIRS: Tuple[Tuple[str, Any], ...] = (
    # GQA: query heads are distributed over KV heads, so the ratio must divide.
    ("n_heads", lambda d: d["n_heads"] - d["n_heads"] % d["n_kv_heads"]
     if d.get("n_kv_heads") else d.get("n_heads")),
    # The cache is at least as long as the queries that read it.
    ("s_ctx", lambda d: max(d.get("s_ctx", 0), d.get("s_query", 0))),
    ("s_kv", lambda d: max(d.get("s_kv", 0), d.get("s_query", 0))),
    # Routing picks a subset, and sparse attention selects from what exists.
    ("n_experts_active", lambda d: min(d.get("n_experts_active", 0),
                                       d.get("n_experts", 1 << 30))),
    ("n_topk_keys", lambda d: min(d.get("n_topk_keys", 0),
                                  d.get("s_kv", d.get("s_ctx", 1 << 30)))),
    # The token total follows the expert geometry: a kernel whose declaration
    # moved s_expert_rows (tile alignment) or n_experts needs n_tokens moved
    # with it, or the default counts (s_expert_rows per expert) no longer sum
    # to it.
    ("n_tokens", lambda d: (d["n_experts"] * d["s_expert_rows"]
                            if "n_experts" in d and "s_expert_rows" in d
                            else d.get("n_tokens", 0))),
)


@dataclass
class SmokeResult:
    """One declared (weight, activation) combination, met."""
    kernel_impl: str
    library: str
    signature: str
    weight: str
    activation: str
    phase: str
    outcome: str
    detail: str = ""
    dims: Dict[str, int] = field(default_factory=dict)
    # What the device ran (`<digest>:<name>`), or "" when it was not observed.
    # Layer 4's evidence — see `kernel_bench/kernel_symbol.py`.
    kernel_symbol: str = ""

    #: Every outcome a smoke check records, worst first — which is also the
    #: order the report prints them in, so the report has no list of its own:
    #:
    #:   "wrong"       numbers disagree — the declaration is FALSE
    #:   "no_cell"     no legal minimal shape could be built — never passed
    #:   "error"       build or run raised — reported, not judged (module
    #:                 docstring)
    #:   "declined"    build raised NotImplementedError — an explicit not-here
    #:   "unchecked"   ran, no reference to compare against
    #:   "ok"          numbers agree
    #:   "unavailable" the library is not installed in this env
    #:
    #: Closed: a result naming anything else is refused on construction, so a
    #: new outcome is declared here — where the two judgements below and the
    #: report read it — rather than appearing as a string the report drops.
    OUTCOMES: ClassVar[Tuple[str, ...]] = (
        "wrong", "no_cell", "error", "declined", "unchecked", "ok",
        "unavailable")

    def __post_init__(self) -> None:
        if self.outcome not in self.OUTCOMES:
            raise ValueError(
                f"{self.kernel_impl}: unknown smoke outcome "
                f"{self.outcome!r} — declare it in SmokeResult.OUTCOMES "
                f"(known: {', '.join(self.OUTCOMES)})")

    @property
    def false_declaration(self) -> bool:
        """Whether this result says the declaration is false. A wrong NUMBER
        is the only unambiguous evidence of that, and deliberately the only
        thing layer 3 asserts on (module docstring): an `error` is reported,
        not judged, and `no_cell` is `untried`."""
        return self.outcome == "wrong"

    @property
    def untried(self) -> bool:
        """Whether the check never happened — which must never read as a
        pass."""
        return self.outcome == "no_cell"


def minimal_dims(provider: Provider, phase: str) -> Tuple[Optional[Dict[str, int]], str]:
    """The smallest legal cell this kernel says it serves, or (None, why).

    Built from the signature's cell axes plus every axis the kernel's own
    declaration constrains — a kernel that pins `d_lin_k_head` to 128 needs that
    axis present even if the signature would not have asked for it.
    """
    sig = get_signature(provider.signature)
    if sig is None:
        return None, f"no registered signature {provider.signature!r}"
    ks = provider.serves
    # `variant_dims` deliberately excluded: binding an axis whose presence
    # selects a geometry picks one for every kernel of the signature, and the
    # kernels that serve the other then get inputs they cannot read.
    # The kernel's own declaration may select one (a kernel that requires
    # the variant's axes serves only that geometry), and then the cell is
    # that geometry's (`cell_dims_for`).
    constrained = ([] if ks is None else
                   [n for n, _ in ks.divisible] + [n for n, _ in ks.enum]
                   + list(ks.requires_dims))
    wanted = list(sig.cell_dims_for(set(constrained))) + constrained

    dims: Dict[str, int] = {}
    unknown = []
    for name in wanted:
        if name in dims:
            continue
        if name not in MINIMAL_DIMS:
            unknown.append(name)
            continue
        dims[name] = MINIMAL_DIMS[name]
    if unknown:
        return None, ("no minimal extent declared for "
                      + ", ".join(sorted(set(unknown))))

    if ks is not None:
        # The kernel's own constraints win over the table: an enum picks its
        # smallest allowed value, a modulus rounds up to the next legal one.
        for name, allowed in ks.enum:
            if allowed:
                dims[name] = min(allowed)
        for name, mod in ks.divisible:
            cur = dims.get(name, mod)
            dims[name] = int(math.ceil(cur / mod) * mod) if mod > 0 else cur
        # A ceiling the table crosses is met under it: the cap's own axes
        # shrink (last first, the earlier ones set by the constraints above)
        # until the product fits, down to the 1 the harness can always run.
        for name, hi in ks.max_dims:
            if name in dims:
                dims[name] = min(dims[name], hi)
        for names, cap in ks.max_product:
            axes = [n for n in names if n in dims]
            while axes and math.prod(dims[n] for n in axes) > cap:
                for n in reversed(axes):
                    others = math.prod(dims[m] for m in axes if m != n)
                    if dims[n] > 1:
                        dims[n] = max(1, cap // others)
                        break
                    if others > cap:
                        continue            # this axis is already at 1
                else:
                    return None, (f"max_product {'*'.join(names)} <= {cap} "
                                  f"fits no minimal cell")

    for name, fix in _REPAIRS:
        if name in dims:
            try:
                v = fix(dims)
            except Exception:
                continue
            if v:
                dims[name] = int(v)

    if any(v <= 0 for v in dims.values()):
        return None, f"repairs produced a non-positive extent: {dims}"
    return dims, ""


def _combinations(provider: Provider) -> List[Tuple[str, str, str]]:
    """Every (weight, activation, phase) the kernel declares.

    A slot the signature does not make material may be undeclared (a norm's
    weight, an embedding's activation): it is met with one representative
    value. `phases` () is either phase, met at prefill.
    """
    ks = provider.serves
    weights = list(ks.weight) or [""]
    acts = list(ks.activation) or ["bf16"]
    phases = list(ks.phases) or ["prefill"]
    return [(w, a, p) for w in weights for a in acts for p in phases]


def cell_dtype_env(p: Provider, weight: str, activation: str) -> Dict[str, str]:
    """The dtype environment of the cell a provider declares for one
    (weight, activation) — what smoke builds it at, and what a check that
    runs the signature's oracle builds it at: every slot the declaration
    constrains, the activation's dtype everywhere it does not.
    """
    dtype_env = {"activation": activation, "output": activation,
                 "kv": activation, "metadata": "fp32",
                 "weight": weight or activation}
    # The slots whose value is not the activation's. `output` is the
    # activation's for every kernel that writes what it read, and a cast is
    # the one that does not (`quantise_activation`: wide in, narrow out, or
    # the reverse) — so it is read off the declaration, the same way
    # `_combinations` reads the weight and the activation. Same for which
    # elements share an activation scale: a cast declares it, and a cell
    # without it is one no provider can read a group size from.
    if p.serves.output:
        dtype_env["output"] = p.serves.output[0]
    if p.serves.activation_granularity:
        dtype_env["activation_granularity"] = p.serves.activation_granularity[0]
    # Every other slot the declaration constrains, the same way: a kernel
    # that serves only an FP8 cache is met with its FP8 cell, not the
    # `kv = activation` default above — building the default anyway is how
    # smoke reported ok for a cache format the launch declines.
    # A slot that admits the activation's dtype is met at it: the floors
    # that read their cache at the cell's width declare every width, and
    # the first of them beside another activation is a cell no deployment
    # has (and no oracle accepts).
    for slot, tokens in p.serves.dtype_slots:
        dtype_env[slot] = activation if activation in tokens else tokens[0]
    # The cached record a launch writes or reads, and its element dtype
    # with it: the two slots name one tensor (`format_refusal`).
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    from backend.inference.benchmark_identity import FORMAT_SLOTS, PLAIN
    for slot, names in p.serves.formats().items():
        # A declaration that admits the plain tensor is met with it: the
        # slot left out, the dtype slot as above.
        if names and PLAIN not in names:
            dtype_env[slot] = names[0]
            dtype_env[FORMAT_SLOTS[slot]] = corpus_token(
                CACHE_FORMATS[names[0]].dtype)
    return dtype_env


def smoke_check(env: Env, *, only_library: str = "",
                device: str = "cuda") -> List[SmokeResult]:
    """Meet every declaration. One result per declared combination."""
    out: List[SmokeResult] = []
    for p in all_providers():
        if p.serves is None:
            continue                       # still on a lambda; nothing declared
        if only_library and p.library != only_library:
            continue
        for weight, activation, phase in _combinations(p):
            r = _one(p, env, weight, activation, phase, device)
            out.append(r)
    return out


def _one(p: Provider, env: Env, weight: str, activation: str, phase: str,
         device: str) -> SmokeResult:
    # Set once a runnable case exists, so the outcomes that HAVE a device to
    # observe can carry the evidence and the rest plainly cannot.
    observed = {"case": None}

    def result(outcome: str, detail: str = "", dims=None) -> SmokeResult:
        symbol = ""
        case = observed["case"]
        if case is not None and outcome in ("ok", "unchecked", "wrong"):
            # After the verdict, never before: `observe` calls the kernel twice
            # more, and many of these are in-place — a second invocation feeds
            # fused_add_rmsnorm its own output. The verdict is already decided
            # here, so the extra calls cannot move it.
            try:
                symbol = describe_kernel(case.run)
            except Exception:
                symbol = ""
        return SmokeResult(p.kernel_impl, p.library, p.signature, weight,
                           activation, phase, outcome, detail, dims or {},
                           symbol)

    why = next(p.unavailability(env), None)
    if why is not None:
        return result("unavailable", why)

    dims, why = minimal_dims(p, phase)
    if dims is None:
        return result("no_cell", why)

    dtype_env = cell_dtype_env(p, weight, activation)
    # The scalars the kernel takes beside its operands. A MEASUREMENT reads
    # them off the model (`OpSignature.config_params` → `kernel_params`) and
    # declines the cell without them, because a corpus row at a bound no
    # model publishes is a row about nothing. A smoke cell is not a
    # measurement and goes in no corpus: it asks whether the kernel runs and
    # agrees with the oracle, and both sides read the same value from the
    # same place, so any legal one does. 1.0 is legal for every scalar
    # declared so far (a clamp bound, SiTU's betas) and exercises the
    # branches a larger one would skip.
    kernel_params = {name: 1.0 for name in (p.serves.requires_params
                                            if p.serves is not None else ())}
    inv = _smoke_cell(p, dims, dtype_env, phase, kernel_params)

    # The declaration must accept its own minimal cell. If it does not, the
    # cell is wrong (a repair or a table entry), not the kernel — report it
    # rather than silently testing nothing.
    if not p.serves.accepts(inv):
        return result("no_cell", "the declaration rejects its own minimal cell",
                      dims)

    # A gate the declaration cannot express still declines before build:
    # `impl_id` resolves the launch's parameterisation purely from the
    # invocation, pre-allocation, and raises for a cell it will not run
    # (the driver's own pre-build channel, driver.py). Without this the
    # refusal surface is one smoke never reads, and a kernel whose build
    # allocates what the launch would decline reports ok for it.
    try:
        p.impl_id(inv)
    except NotImplementedError as e:
        return result("declined", str(e)[:120], dims)

    spec = get_ref(p.signature)
    if spec is None:
        return result("no_cell", f"no refs entry for {p.signature!r}", dims)

    # `make_inputs` may read axes the signature does not list as required. Walk
    # that gap by adding exactly the axis it names and retrying, rather than
    # pre-seeding every dim in the table: an axis this cell does not genuinely
    # need must stay absent, since `requires_dims` gates on presence and a
    # helpfully-supplied axis would silently widen what the cell tests.
    inputs = None
    for _ in range(len(MINIMAL_DIMS) + 1):
        try:
            inputs = spec.make_inputs(inv, device,
                                      input_generator(inv, device))
            break
        except KeyError as e:
            name = e.args[0] if e.args else ""
            if not isinstance(name, str) or name in dims:
                return result("error", f"inputs: KeyError: {name!r}", dims)
            if name not in MINIMAL_DIMS:
                return result("no_cell",
                              f"`make_inputs` needs {name!r}, which has no "
                              f"declared minimal extent", dims)
            dims[name] = MINIMAL_DIMS[name]
            inv = _smoke_cell(p, dims, dtype_env, phase, kernel_params)
        except Exception as e:
            return result("error", f"inputs: {type(e).__name__}: {e}", dims)
    if inputs is None:
        return result("no_cell", "input construction did not converge", dims)

    # Same grid the driver puts the weight on. Without this, smoke exercises a
    # path the sweep never takes: it built from the raw canonical weight while
    # the driver rounds first, so a kernel that quantizes its own weight was
    # reported against a reference the sweep does not use. That made marlin
    # look like six false declarations when what it showed was the un-rounded
    # comparison the grid exists to replace.
    eff = None
    try:
        eff = _quant.effective_weights(
            inputs, dtype_env.get("weight", ""), p.weight_group,
            scale_encoding=p.weight_scale,
            act_dtype=_act_torch_dtype(inv))
    except Exception as e:
        return result("no_cell", f"grid: {type(e).__name__}: {e}", dims)
    if eff is not None:
        inputs = eff

    # The reference is taken BEFORE the kernel runs, matching the driver. Many
    # of these kernels are in-place (fused_add_rmsnorm updates the residual,
    # rope rotates q/k in the buffer); computing the reference afterwards would
    # compare the kernel against its own output and report a correct kernel as
    # a false declaration. This harness did exactly that on its first run.
    ref_out = None
    if spec.reference is not None:
        try:
            ref_out = spec.reference(inputs)
        except Exception as e:
            return result("error", f"reference: {type(e).__name__}: {e}", dims)

    try:
        case = p.build(inv, env, inputs)
    except NotImplementedError as e:
        return result("declined", str(e)[:120], dims)
    except Exception as e:
        return result("error", f"build: {type(e).__name__}: {e}", dims)
    observed["case"] = case

    # Exactly ONE run. Many of these kernels are in-place, so a second
    # invocation feeds them their own output: fused_add_rmsnorm re-adds the
    # residual and lands at cosine 0.60 against a reference it actually agrees
    # with. `_cross_check` does the run when there is a reference, so running
    # here as well is the second call — this harness reported a correct kernel
    # as a false declaration for exactly that reason.
    if ref_out is None or case.to_canonical is None:
        try:
            case.run()
        except NotImplementedError as e:
            return result("declined", str(e)[:120], dims)
        except Exception as e:
            return result("error", f"run: {type(e).__name__}: {e}", dims)
        return result("unchecked", "no reference to compare against", dims)

    try:
        # `_xcheck_dtype`, not `inv.dtype`: which band a cell reads is one
        # question with one answer (the least precise operand the kernel
        # reads, not the activation alone), and asking it a second way here
        # is what made smoke call FlashMLA's fp8 sparse decode a false
        # declaration — `wrong` is the only outcome that means that — for
        # sitting outside a bf16 band it never claimed.
        from .driver import _cross_check, _xcheck_basis
        # Five values, not three: `typical` and `worst` are the two the
        # verdict is MADE on (schema 13 put them on the row for the same
        # reason). Unpacking three silently turned every checked provider in
        # this harness into `run: ValueError: too many values to unpack` —
        # 75 of them — which reads as a provider fault and is not one.
        x_dtype, x_records = _xcheck_basis(inv, inputs)
        verdict, rel, cos, _typical, _worst = _cross_check(
            case, ref_out, x_dtype, records=x_records)
    except NotImplementedError as e:
        return result("declined", str(e)[:120], dims)
    except Exception as e:
        return result("error", f"run: {type(e).__name__}: {e}", dims)

    if verdict is not None:
        return result("wrong", f"rel_err={verdict:.3g} cosine={cos:.5f}", dims)
    return result("ok", f"rel_err={rel:.3g} cosine={cos:.5f}", dims)


def same_kernel_twice(results: List[SmokeResult]) -> List[Tuple[str, List[str]]]:
    """Kernels of one signature that launched the SAME sequence on the SAME cell.

    Two `kernel_impl`s whose device work is byte-identical are one kernel under
    two names — one measurement recorded twice, and a cross-library comparison
    that is really a library compared with itself. This is the check layer 4
    can make with no declaration at all: it needs no expected symbol, only the
    observation that two of them agree.

    Grouped by the CELL, not just the signature — a kernel's own `divisible` /
    `enum` constraints can move its minimal dims, and two symbols observed at
    different shapes say nothing about each other.

    Reported, not asserted. Vendoring is real (a framework may ship another
    library's kernel), so a collision is a fact worth seeing rather than a
    verdict; layer 3 keeps its single assertion on wrong numbers.
    """
    groups: Dict[tuple, Dict[str, List[str]]] = {}
    for r in results:
        if not r.kernel_symbol:
            continue
        cell = (r.signature, r.phase, r.weight, r.activation,
                tuple(sorted(r.dims.items())))
        groups.setdefault(cell, {}).setdefault(r.kernel_symbol, []).append(
            r.kernel_impl)
    out: List[Tuple[str, List[str]]] = []
    for cell, by_symbol in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for symbol, impls in by_symbol.items():
            if len(impls) > 1:
                out.append((f"{cell[0]}/{cell[1]} {symbol}", sorted(impls)))
    return out


def format_report(results: List[SmokeResult]) -> str:
    lines: List[str] = []
    by_outcome: Dict[str, int] = {}
    for r in results:
        by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
    for outcome in SmokeResult.OUTCOMES:
        rows = [r for r in results if r.outcome == outcome]
        if not rows:
            continue
        lines.append(f"\n{outcome}  ({len(rows)})")
        for r in sorted(rows, key=lambda x: (x.library, x.kernel_impl)):
            slot = f"{r.weight or '-'}/{r.activation}"
            lines.append(f"  {r.kernel_impl:<34} {slot:<20} {r.phase:<8} {r.detail}")
    dupes = same_kernel_twice(results)
    if dupes:
        # Its own block, above the counts: a collision buried in a tally reads
        # as two healthy measurements, which is exactly what it is not.
        lines.append(f"\nsame kernel under two names  ({len(dupes)})")
        for where, impls in dupes:
            lines.append(f"  {', '.join(impls)}")
            lines.append(f"      {where}")
    observed = sum(1 for r in results if r.kernel_symbol)
    runnable = sum(1 for r in results
                   if r.outcome in ("ok", "unchecked", "wrong"))
    lines.append("\n" + "  ".join(f"{k}={v}" for k, v in sorted(by_outcome.items())))
    lines.append(f"kernel symbol observed for {observed}/{runnable} that ran")
    return "\n".join(lines)
