"""Per-tensor packing / quant-scheme consistency.

The byte anchor (`model_storage_bytes`) compares the *aggregate*
`Σ(element_count × dtype_width)` against the checkpoint. That product is
packing-agnostic: a spec can mis-model a quantized tensor's packing so that a
2× error in the logical element count is cancelled by a ½× error in the dtype
width, leaving the byte sum — and `valid` — untouched. `total_params` would
show the 2× blow-up, but it is a non-gating diagnostic (and its HF side depends
on `packed_unpack_factor` guessing every tensor's scheme, fragile for vendors
who split methods across config fields, e.g. DSv4's `expert_dtype`).

This check decomposes the byte product back into its two factors and anchors
each independently against the safetensors header — the unambiguous physical
truth on disk:

  1. **Stored element count.** For every matched tensor, the spec's stored
     element count (`storage_bytes / dtype_bytes(storage_dtype)`) must equal the
     header's `∏ shape`. A wrong logical shape under a correct scheme (e.g.
     declaring the *packed* dims as logical) makes the stored count diverge.

  2. **Stored dtype — for quantized weights and their metadata siblings only.**
     The spec's `storage_dtype` must match the header's dtype. The scale
     sibling's dtype is the scheme fingerprint: a UE8M0 group scale IS microscaled
     FP4. A spec that mis-assigns a packing=1 int8 scheme to genuinely-FP4 data
     keeps the weight container (I8) but cannot fake the E8M0 scale — the
     mismatch fires here. Given the scheme is thus pinned and each scheme fixes
     its `packing_factor`, the logical count (→ `total_params`, → FLOPs) is
     forced.

Full-precision tensors (bf16/f16/f32 defaults) get the size check only — their
dtype identity is a family-spec concern (the `llama` spec serves both fp16
Llama-2 and bf16 Llama-3) orthogonal to packing, and is 2-byte-neutral anyway.
"""
from __future__ import annotations

from ._finding import Finding

from backend.arch.dtypes import UnknownDtype, canonical
from backend.arch.quant_schemes import QUANT_SCHEMES
from backend.arch.eval.formats import Formats

from typing import Any, Dict, List, Optional


# The only storage-dtype alias observed across the snapshot battery: the
# UE8M0 microscale is spelled `F8_E8M0FNU` by the materialize side (the precise
# OCP name) and `F8_E8M0` by the safetensors header. Same 1-byte exponent-only
# scale. Everything else must match on the nose.


def _canon_dtype(dtype: str) -> str:
    """The canonical spelling from `arch.dtypes` (`F8_E8M0FNU` and the
    header's `F8_E8M0` are one row); a spelling the table does not know
    stays as written and fails the equality it is compared in."""
    key = str(dtype).upper().strip()
    try:
        return canonical(key)
    except UnknownDtype:
        return key


def run_packing_consistency_check(analysis, header) -> List[Finding]:
    """Anchor each tensor's stored (count, dtype) against the safetensors
    header (`hf.header.SafetensorsHeader`). Returns a list of findings
    (empty = clean). Tensors the spec emits whose name selects no header
    tensor are left to the existing unmatched-tensor reporting — this check
    only speaks about tensors it can pair to physical ground truth.
    """
    from backend.arch.dtypes import UnknownDtype, dtype_bytes

    grouped: Dict[str, list] = {}
    for row in analysis.iter_checkpoint():
        grouped.setdefault(row.name, []).append(row)

    errors: List[str] = []
    for name, tensors in grouped.items():
        selection = header.select(name)
        if not selection:
            continue  # unmatched — handled by the unmatched-tensor path

        # (1) Stored element-count identity — every matched name: the
        # header's elements over every tensor the name selects, whatever
        # shapes they come in (a per-layer head-count override,
        # Step-3.7-Flash's q_proj, is two groups).
        header_total_elems = int(selection.elements)
        spec_total_elems = 0
        comparable = True
        for row in tensors:
            try:
                width = dtype_bytes(row.storage_dtype or "")
            except UnknownDtype:
                comparable = False      # no width, no element identity
                break
            spec_total_elems += (
                round(row.storage_bytes / width) * int(row.count)
            )
        if comparable and spec_total_elems != header_total_elems:
            names = ", ".join(sorted({row.name for row in tensors}))
            stored = "; ".join(f"shape {list(g.shape)} x {n}"
                               for g, n in selection.parts)
            errors.append(Finding(kind="packing_element_count_mismatch", detail=
                f"Stored element-count mismatch for {name!r}: "
                f"spec stores {spec_total_elems:,} elems across "
                f"{names}, checkpoint header has {header_total_elems:,} "
                f"({stored}). "
                f"A packing/logical-shape error hides in the byte sum but "
                f"diverges here."))

        # (2) Stored-dtype identity — quantized weights + their metadata
        # siblings (scales / zero-points / g_idx), where dtype encodes the
        # quant scheme. Full-precision defaults are size-checked only. A
        # name the header stores in several dtypes (GLM-5.3's BF16 and
        # FP8 `o_proj` layers) admits each of them.
        header_dtypes = {_canon_dtype(d) for d in selection.dtypes}
        for row in tensors:
            is_quant_tensor = (row.method is not None) or bool(row.is_metadata)
            if (is_quant_tensor
                    and _canon_dtype(row.storage_dtype or "")
                    not in header_dtypes):
                kind = "scale/metadata" if row.is_metadata else "quantized weight"
                errors.append(Finding(kind="packing_dtype_mismatch", detail=
                    f"Stored-dtype mismatch for {name!r} ({kind}): "
                    f"spec stores {row.storage_dtype}, checkpoint header is "
                    f"{'/'.join(selection.dtypes)}. The scale-sibling dtype is the quant-scheme "
                    f"fingerprint (e.g. UE8M0 => microscaled FP4); a mismatch means "
                    f"the spec assigned the wrong scheme, so its logical param count "
                    f"and FLOPs are wrong even though the byte sum matches."))

    return errors


def run_quant_override_check(analysis) -> List[Finding]:
    """Every `quant_context.overrides` row must decide at least one stored
    tensor's format.

    A row that selects nothing — its pattern names no tensor the spec stores,
    or an earlier row always wins first — changes no byte, so the byte anchor
    above cannot see it. It still reads like a live rule, and the dtype it
    claims is not the one the model gets. DeepSeek-V4's `\\.ffn\\.weight$`
    router row was one: the router is `ffn.gate.weight`, and it took BF16 from
    `per_role.router` all along.

    And every row that names a quant `method:` must name it for tensors that
    method's scheme describes. A vector or a buffer has no `(out, in)` axes
    for a block or a group to tile; it is stored at the default dtype with no
    metadata, so the row's method decides nothing for it — the dtype the row
    claims is not the one the model gets.
    """
    program = analysis.program
    policy = Formats.of(program).saved
    if not policy.overrides:
        return []
    rows = [(row, program.by_uid(row.uid)) for row in analysis.tensors()
            if not row.is_metadata]
    stored = [(row.name, program.stack_of(node), node.enclosing_components,
               program.stored_layers(row.uid))
              for row, node in rows]
    findings: List[Finding] = []
    undescribed: Dict[int, List[str]] = {}
    for index, name in policy.undescribed_overrides(
            (*site[:3], row.rep.saved)
            for site, (row, _) in zip(stored, rows, strict=True)
            if row.rep is not None):
        undescribed.setdefault(index, []).append(name)
    for index, names in sorted(undescribed.items()):
        entry = policy.overrides[index]
        findings.append(Finding(kind="quant_override_undescribed", detail=
            f"quant_context.overrides[{index}] ({entry.selector_text()}) "
            f"names method {entry.method!r} for {sorted(set(names))}, which "
            f"that scheme does not describe: a quant scheme describes tensors "
            f"stored with two axes (out, in) — a linear's matrix, a per-head "
            f"bank, a lookup table's (rows, width) — and these are not. Store "
            f"them with a row that names only their `dtype:` (no `method:`), "
            f"or narrow this row so it selects the weight tensors it means."))
    for index in policy.unused_overrides(stored):
        entry = policy.overrides[index]
        findings.append(Finding(kind="quant_override_unused", detail=
            f"quant_context.overrides[{index}] ({entry.selector_text()}) "
            f"decides the format "
            f"of no stored tensor: nothing the spec stores matches it, or an "
            f"earlier override always wins first. Point it at the checkpoint "
            f"name it means, or delete it."))
    return findings


def _stack_kind(program, node) -> str:
    """The stack a precision choice names: `backbone` or `aux:<kind>`."""
    stack = program.stack_of(node)
    for declared in program.stacks:
        if declared.name == stack:
            return (f"aux:{declared.aux_kind}" if declared.is_aux
                    else "backbone")
    return stack


def run_quant_role_check(*analyses) -> List[Finding]:
    """Could a finer ROLE have expressed this precision split?

    A deployment chooses weight precision by role and stack (`backbone` /
    `aux:<kind>`), never by tensor name, so a role is the finest thing a
    precision choice can address — and a role addresses a NODE, every copy
    of it, at every layer. So the question this asks is whether the split
    the checkpoint shows is one a role could have made. It could exactly
    when two NODES of one role are each stored at one precision and those
    precisions disagree: that is a BF16 `o_proj` beside FP8 `q/k/v`, or BF16
    shared experts beside INT4 routed ones, and the part stored apart takes
    a finer role that refines the broader one (`roles.yaml`), declared by `role:` on
    its node, with a `quant_context.per_role` entry of its own. Pinning it
    by tensor name instead is refused here.

    A split no role could have made is a split of ONE node across its OWN
    layers: Step-5 stores its routed experts FP8 in layers 3-87 and BF16 in
    88-90, one node, same role, stack, component and graph throughout. A
    role cannot say "this node, at these layers" — it is a property of the
    node at every layer it runs — so no assignment of roles reproduces that
    split. It is declared instead by a layer-scoped
    `quant_context.overrides` row (`OverrideSelector.layers`), it arrives
    here as several `Analysis.tensors()` rows of one node, and it passes.

    So the test is whether the precisions line up with the NODES. Where
    every node of a role holds one precision and two of them disagree, a
    finer role is exactly the declaration that was missing. GLM-5.3-Flash's
    `o_proj` is that case even though its split follows the layer schedule:
    the BF16 one is a node of the linear-attention component and the FP8 one
    a node of the full-attention component, each uniform, and roles
    (`attention_linear` / `attention_kv_up` / `attention_indexer`) are how
    that checkpoint says it.

    Asked in every phase form of the graph the analyses walk (prefill, and
    decode where a component swaps in its decode twin).

    The role is the one the leaf carries (a slot such as `attention_swa` is
    addressable by itself). The tensors are the ones a quant scheme describes
    (`StoredTensor.matrix_dims`: a matrix, a per-head bank, a lookup table's
    rows); a role's gains, biases and buffers are outside every role rule by
    design."""
    # (role, stack) → node → dtype → the evidence for it. Per NODE, because
    # a node is what a role addresses.
    groups: Dict[tuple, Dict[str, Dict[str, Dict[str, set]]]] = {}
    for analysis in analyses:
        program = analysis.program
        for row in analysis.tensors():
            if (row.is_metadata or row.rep is None
                    or row.tensor.matrix_dims is None):
                continue
            node = program.by_uid(row.uid)
            if not node.role:
                continue
            saved = row.rep.saved
            by_node = groups.setdefault(
                (node.role, _stack_kind(program, node)), {})
            entry = by_node.setdefault(node.display_path, {}).setdefault(
                saved.logical_dtype,
                {"nodes": set(), "tensors": set(), "rules": set()})
            entry["nodes"].add(node.display_path)
            entry["tensors"].add(row.name)
            entry["rules"].add(saved.selected_by)
    findings: List[Finding] = []
    for (role, stack), by_node in sorted(groups.items(), key=str):
        # Only the nodes stored at ONE precision can be told apart by a
        # role. A node holding several is split across its own layers, which
        # no role expresses and a layer-scoped row declares.
        by_dtype: Dict[str, Dict[str, set]] = {}
        for dtypes in by_node.values():
            if len(dtypes) > 1:
                continue
            for dtype, evidence in dtypes.items():
                held = by_dtype.setdefault(
                    dtype, {"nodes": set(), "tensors": set(), "rules": set()})
                for key, values in evidence.items():
                    held[key].update(values)
        if len(by_dtype) < 2:
            continue
        split = "; ".join(
            f"{dtype}: nodes {sorted(entry['nodes'])[:6]} (tensors such as "
            f"{sorted(entry['tensors'])[0]!r}, by "
            f"{', '.join(sorted(entry['rules']))})"
            for dtype, entry in sorted(by_dtype.items()))
        findings.append(Finding(kind="quant_role_mixed_precision", detail=(
            f"Role {role!r} on the {stack} stack holds weight matrices the "
            f"checkpoint stores at {len(by_dtype)} precisions — {split}. "
            f"Precision is chosen by role, so each role holds one stored "
            f"precision, and each of these nodes holds one: a finer role "
            f"is what tells them apart. Give the nodes stored apart a finer "
            f"role from roles.yaml that refines {role!r} (for example "
            f"`attention_output` for an output projection, `ffn_moe_shared` "
            f"for shared experts) with `role:` on the node, state its dtype "
            f"as a `quant_context.per_role` entry under that role, and drop "
            f"the override row that pinned those tensors. One node stored "
            f"two ways across its OWN layers is the other case and is "
            f"declared rather than refused: scope the override row to those "
            f"layers (`select: {{layers: ..., tensors_re: ...}}`)."),
            extra={"role": role, "stack": stack,
                   "dtypes": sorted(by_dtype)}))
    return findings


def _format_signature(row) -> tuple:
    """The stored format of one row, independent of the row's own extents.

    A decode twin may hold a stored tensor whole (the same `wkv`) or as a
    part of it (absorbed MLA's `q_absorbed_nope` / `v_absorbed`, the two
    halves of `kv_b_proj`, declared with `module: kv_b_proj`). A part has
    other extents and so other scale extents, and is still the same
    format: the same scheme at the same granularity, the same slot type
    and packing, the same sibling names and types."""
    saved = row.rep.saved if row.rep is not None else None
    metadata = tuple(sorted((meta.name, meta.dtype) for meta in row.metadata))
    packing = saved.packing_factor if saved is not None else 1
    method = saved.method if saved is not None else None
    granularity = saved.weight_granularity if saved is not None else None
    return (row.storage_dtype, row.logical_dtype, packing, method,
            granularity, metadata)


def run_phase_format_check(prefill, decode) -> List[Finding]:
    """A checkpoint tensor resolves to one stored format whichever form of
    the graph reads it.

    A component may name a decode twin (`decode_variant:`, an absorbed MLA,
    a one-token compressor) that holds the same checkpoint tensors under its
    own nodes. Those nodes resolve their format on their own — by their own
    role — so a twin node missing the role its prefill node carries reads
    the same file tensor at another precision, and the prefill-only gates
    cannot see it. The tensor is the stored name on its stack, whatever
    extents the node holding it reads it at (a twin node may hold a part of
    it); the format is `_format_signature`."""
    def formats(analysis) -> Dict[tuple, Dict[tuple, set]]:
        program = analysis.program
        out: Dict[tuple, Dict[tuple, set]] = {}
        for row in analysis.tensors():
            if row.is_metadata or row.is_buffer:
                continue
            node = program.by_uid(row.uid)
            key = (_stack_kind(program, node), row.name)
            out.setdefault(key, {}).setdefault(
                _format_signature(row), set()).add(
                    f"{node.display_path} (role {node.role})")
        return out

    before, after = formats(prefill), formats(decode)
    findings: List[Finding] = []
    for key in sorted(set(before) & set(after)):
        if set(before[key]) == set(after[key]):
            continue
        stack, name = key

        def describe(found):
            return "; ".join(
                f"{sig[1]} stored {sig[0]} by {sorted(nodes)[:3]}"
                for sig, nodes in sorted(found.items(), key=str))
        findings.append(Finding(kind="quant_phase_format_mismatch", detail=(
            f"Checkpoint tensor {name!r} (on the {stack} stack) resolves to one "
            f"format in prefill — {describe(before[key])} — and another in "
            f"decode — {describe(after[key])}. The file stores it once. Give "
            f"the decode-form node the role its prefill node carries (`role:` "
            f"on the node in the `decode_variant` component), so both resolve "
            f"through the same `quant_context.per_role` entry."),
            extra={"tensor": name, "stack": stack}))
    return findings


def run_quant_method_check(quant_context: Any,
                           unknown_method: Optional[str]) -> List[Finding]:
    """A checkpoint whose `quant_method` no registered scheme is named for
    (`quant_method_unknown` in its metadata) is described by the spec: its
    `quant_context` rows name, by `method:`, the registered schemes whose
    tensors the checkpoint stores (or `none` where a tensor is stored as is).

    A spec that names no method treats every tensor as `default_dtype`,
    which is a guess about a checkpoint whose method nobody has read. Once a
    method is named, the storage, coverage and packing gates check the
    description against the header like any other."""
    if not unknown_method:
        return []
    rows = [] if quant_context is None else [
        *quant_context.per_role.values(), *quant_context.overrides]
    if any(row.method for row in rows):
        return []
    return [Finding(kind="quant_method_unknown", detail=(
        f"The checkpoint's quant_method {unknown_method!r} is not a "
        f"registered quant scheme, and quant_context names no method for its "
        f"tensors. Describe how they are stored: a quant_context.per_role or "
        f"overrides row per group of tensors, whose `method:` is the "
        f"registered scheme those tensors and their metadata siblings follow "
        f"({', '.join(sorted(QUANT_SCHEMES))}), or `none` for tensors stored "
        f"unquantized. The storage, coverage and packing gates then check the "
        f"description against the checkpoint header."),
        extra={"quant_method": unknown_method})]
