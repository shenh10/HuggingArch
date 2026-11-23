"""Pure response-builder functions for FastAPI endpoints.

These take the raw analyzer/spec results and shape them into the
front-end-facing dicts (block_graph, architecture_view, model_config).
Kept separate from `api FastAPI app` so HTTP routing stays focused on
request handling and these data transformations stay easy to test in
isolation.
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any, Dict, List, Optional

from ._consts import (
    OP_LABEL as _OP_LABEL,
    STYLE_BY_OP as _STYLE_BY_OP,
    STYLE_BY_OP as _STYLE,
    STYLE_BY_ROLE as _ROLE_STYLE,
)
from backend.arch.dim_resolution import references_config
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.projection import project_model
from backend.arch.eval.traffic import derive
from backend.arch.ops import ShapeKind, is_out_in_matrix, owns_lookup_table
from backend.arch.spec_tree import iter_blocks, iter_tree
from ._badges import collect_badges
from ._layout import layout_assignments_in_order as _layout_assignments_in_order
from ._memory import compute_memory, stored_weights
from ._node_build import NodeBuilder, apply_param_absorption, strip_transport_keys
from .kv_marks import KVMarks
from ._shapes import (
    build_edges,
    build_report_index,
    infer_input_shape,
    fmt_shape_workload,
)
from ._quant import (
    resolve_precision as _resolve_precision,
)
from ._quant_match import attach_dtypes as attach_dtypes_via_materialize


def build_block_graph_data(resolved_spec, batch_size=1, seq_len=1, *,
                           precision: Any = None,
                           workload_symbolic: bool = False):
    """Build structured block graph data with recursive component expansion and computed shapes.

    Every node that writes or reads the KV cache carries ``kv`` and every
    cache edge ``kind: "cache"`` (`KVMarks`). Every width — edges, cached
    tensors, weights — is the walk's under `precision` (a precision profile,
    as shipped when None).

    `workload_symbolic` draws the graph for the model rather than for one
    batch and length: shape labels keep their workload axes symbolic
    (`[B,s_ctx,4096]`) and the KV marks carry no per-request or per-step
    figure for a growing cache. The Inspector asks for it."""
    # Walk the full model DAG once. The eval program threads each layer's
    # output into the next layer's input alias, so block
    # templates see their real upstream shape (e.g. dsv4_block sees
    # the `[B, S, Hc, H]` from `hc_expand`'s broadcast, not the empty
    # `inputs: [x]` placeholder). Indexing first occurrence per
    # template suffices: every assignment of the same template name
    # produces an identical BlockResult shape-wise.
    model_result = project_model(
        resolved_spec, batch=batch_size, input_tokens=seq_len,
        checkpoint=True, precision=precision,
    )
    # The graph drawn is the one that walk evaluated: a component the phase
    # runs in another form (`transforms.phase_form` — a sparse attention
    # prefills absorbed) is drawn in that form, so each node carries its own
    # numbers and they are the estimate's.
    walked = model_result.analysis.program.resolved

    # Templates the walk actually instantiated — not every name written
    # in spec.model. A group with `repeat: 0` (optional MTP absent from
    # this checkpoint) still names its template in YAML; iter_blocks
    # skips it, so rendering must too, or it raises "eval omitted a
    # registered template" for a stack that does not exist.
    # Same template name → same I/O (copied onto BlockMeta at enumerate).
    registered_templates = set()
    io_by_template: Dict[str, tuple] = {}
    meta_by_template: Dict[str, Any] = {}
    for meta in iter_blocks(walked):
        if not meta.template_name:
            continue
        registered_templates.add(meta.template_name)
        io_by_template.setdefault(meta.template_name, (meta.inputs, meta.outputs))
        meta_by_template.setdefault(meta.template_name, meta)

    result = {}
    unmatched_nodes: List[dict] = []

    materialize_index = model_result.analysis.tensor_index()
    stored = stored_weights(model_result.analysis)
    traffic = derive(model_result.analysis)
    kv_marks = KVMarks.of_analysis(
        model_result.analysis, workload_symbolic=workload_symbolic)
    template_to_br = {}
    for tmpl_name, _count, br in model_result.block_results:
        template_to_br.setdefault(tmpl_name, br)

    # Flat aggregation_id-keyed report index across every block report.
    # NodeBuilder consumes it directly while walking SpecIRNodes — no
    # separate shape_map walk to keep in sync. aggregation_id is globally
    # unique (template-prefixed for block-internal nodes), so cross-block
    # collisions can't happen even with one shared dict.
    report_index: Dict[str, Any] = {}
    for _tmpl_name, _count, br in model_result.block_results:
        build_report_index(br.nodes, report_index)
    node_builder = NodeBuilder(
        walked, _STYLE, _ROLE_STYLE, report_index=report_index,
        workload_symbolic=workload_symbolic,
    )

    for tmpl_name in walked.spec.all_block_names():
        if tmpl_name not in registered_templates:
            continue  # spurious template not referenced by any model layer
        meta = meta_by_template[tmpl_name]
        inputs, outputs = io_by_template[tmpl_name]

        br = template_to_br.get(tmpl_name)
        if br is None:
            raise ValueError(
                f"Eval analysis omitted registered template {tmpl_name!r}; "
                "rendering requires every model-referenced block to be part "
                "of the analyzed Program"
            )

        nodes_data = node_builder.build_graph(tmpl_name)

        # Attach weight dtype via the materialize-driven index. spec_tree
        # emits canonical leaf ids, materialize maps them through the
        # scheme registry, and `_quant_match` keys the result by leaf
        # aggregation_id (the same id `_node_build` assigned to renderer
        # nodes). This covers every leaf the spec describes — per_role +
        # overrides + default_dtype — with the SEMANTIC dtype label
        # (INT4 / FP4 / F8_E4M3 / BF16) instead of raw storage dtype.
        #
        # The index is flat (`{aggregation_id: mt}`): the template prefix
        # natively disambiguates same-display_path leaves across hybrid
        # templates (MTP block's `hc_attn_fn` vs main backbone's), so no
        # per-template sub-dict is needed.
        attach_dtypes_via_materialize(nodes_data, materialize_index)

        # Surface any leaf with stored params that didn't get a dtype —
        # an invariant breach (materialize coverage should be 100% on the
        # supported_models set). When triggered, it's a signal that
        # spec_tree or the scheme registry missed a leaf.
        _collect_unmatched_leaves(nodes_data, unmatched_nodes, tmpl_name)

        # Step 3: bytes. Weight bytes per node come from the weight ledger;
        # activation width from the deployment's dtype policy.
        compute_memory(nodes_data, model_result.analysis, stored, traffic)

        # Step 4: Fold lone-consumer parameter_vector / parameter_tensor
        # siblings into their consumer's `absorbed_params`. Runs after
        # dtype + memory attach so absorbed entries carry weight_dtype /
        # weight_bytes; runs BEFORE build_edges so the edges that used
        # to draw from the folded params disappear naturally.
        nodes_data = apply_param_absorption(nodes_data)

        # Presentation only: no kernel runs here, so no bytes. Kept
        # out of `compute_memory`, which prices program nodes.
        # Same `_block_stream_roles` the architecture card uses (I.4
        # leftover: stop `component_stream_roles` on the YAML graph).
        _, named = _block_stream_roles(meta, model_result.analysis.program)
        stream_reads = [s for s in named if s not in inputs]
        for inp_name in list(inputs) + stream_reads:
            is_stream = inp_name in stream_reads
            n_in = {
                "id": inp_name,
                "op": "stream_in" if is_stream else "input",
                "label": f"stream: {inp_name}" if is_stream else "input",
                "style": "io",
                "inputs": [],
                "is_source": True,
                "weight_bytes": 0,
                "out_bytes": 0,
                "access_bytes": 0,
                "arithmetic_intensity": None,
            }
            found = infer_input_shape(inp_name, br.nodes)
            if found and found[0]:
                shape, axes = found
                n_in["tensor_out"] = (
                    fmt_shape_workload(shape, axes, node_builder.name_to_value)
                    if workload_symbolic
                    else "[" + ",".join(str(d) for d in shape) + "]")
            nodes_data.insert(0, n_in)

        # Where the block publishes each inner stream it writes: an exit
        # box per stream, fed by the node that makes it — the counterpart of
        # the `stream_in` boxes above, so a stream named on the overview
        # can be found in the block.
        template = _block_stream_template(meta, model_result.analysis.program)
        if template is not None:
            writers = model_result.analysis.program.stream_writers(template)
            for name, nodes in writers.items():
                nodes_data.append({
                    "id": f"{name}>",
                    "op": "stream_out",
                    "label": f"stream: {name}",
                    "style": "io",
                    # From the block-level box the writer sits in (an edge
                    # joins siblings); the writer itself says so when its
                    # box is opened (`stream_writes`).
                    "inputs": [_mark_stream_writer(nodes_data, node, name)
                               for node in nodes],
                    "is_source": False,
                    "weight_bytes": 0,
                    "out_bytes": 0,
                    "access_bytes": 0,
                    "arithmetic_intensity": None,
                })

        kv_marks.stamp(nodes_data)
        edges_data = build_edges(nodes_data)
        strip_transport_keys(nodes_data)

        result[tmpl_name] = {
            "inputs": list(inputs),
            "outputs": list(outputs),
            "reads": list(meta.stream_reads),
            "nodes": nodes_data,
            "edges": edges_data,
            # The wrapper this block calls and its bindings
            # (`BlockMeta.wrapper_op` / `wrapper_params`): the template the
            # loader expanded inline shorthand into, or the one a
            # hand-written adapter (bailing_*, qwen3_next_*) binds. The
            # frontend renders a readable label from them ("attn + ffn" for
            # pre_norm topologies). A block that is neither (an inline
            # component like longcat_block) shows its own name, no params.
            "template": meta.wrapper_op or tmpl_name,
            "params": dict(meta.wrapper_params),
        }
    return result, unmatched_nodes


def _mark_stream_writer(nodes_data, node, name: str) -> str:
    """Mark the drawn box of IR `node` as writing stream `name`
    (`stream_writes`), and return the id of the top-level box it sits in.
    A writer drawn nowhere is a graph that cannot say where the stream is
    made: refused."""
    chain = []
    at = node
    while at is not None:
        chain.append(at.aggregation_id)
        at = at.parent
    by_id = {}

    def index(boxes):
        for box in boxes:
            by_id[box["id"]] = box
            index(box.get("children") or ())

    index(nodes_data)
    drawn = [box_id for box_id in chain if box_id in by_id]
    top = {box["id"] for box in nodes_data}
    if not drawn or drawn[-1] not in top:
        raise ValueError(
            f"stream {name!r} is written by {node.display_path}, which the "
            "block graph draws in no top-level box")
    writes = by_id[drawn[0]].setdefault("stream_writes", [])
    if name not in writes:
        writes.append(name)
    return drawn[-1]


def _collect_unmatched_leaves(nodes, out, template_name):
    """Walk a built block-graph tree; append `{template, node_id, params}`
    entries for every leaf with stored params that didn't get a
    `weight_dtype` stamp from the materialize index. Empty under normal
    operation (spec_tree + materialize cover the supported models 100%);
    populated only when a new spec / scheme combination breaks coverage."""
    for n in nodes:
        is_leaf = not n.get("children")
        params = n.get("params") or 0
        if is_leaf and params > 0 and not n.get("weight_dtype"):
            out.append({
                "template": template_name,
                "node_id": n.get("id"),
                "params": params,
            })
        if n.get("children"):
            _collect_unmatched_leaves(n["children"], out, template_name)


def _block_stream_template(meta, program):
    if meta is None or program is None:
        return None
    return next((t for t in program.blocks
                 if t.name == meta.template_name), None)


def _block_stream_roles(meta, program=None):
    """Inner names this invocation's stack declares, plus IR stream refs."""
    if meta is None:
        return [], []
    template = _block_stream_template(meta, program)
    if template is not None:
        produces, reads = program.component_stream_roles(template)
        return list(produces), list(reads)
    produces = list(meta.reads.inner)
    reads = list(meta.reads.each)
    return produces, reads


def _annotate_stream_roles(cards, program):
    """Attach `stream_produces` / `stream_reads` to every card (recursing
    into cycle_blocks, group layers and a kind card's variants) so the
    frontend can draw the stream;
    `stream_sources`: per inner stream the card's layers read, the
    `reads_from` they declare and the layer each reads
    (`Program.declared_stream_reads`); and `stream_windows`: the most layers
    between a reader and the layer that produced what it reads
    (`Program.stream_distances`)."""
    windows = program.stream_distances()
    meta_by_tmpl = {}
    for template in program.blocks:
        meta_by_tmpl.setdefault(template.name, template.meta)
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        tmpl = card.get("template")
        if tmpl:
            meta = meta_by_tmpl.get(tmpl)
            prod, reads = _block_stream_roles(meta, program)
            template = _block_stream_template(meta, program)
            sources = ({} if template is None
                       else program.declared_stream_reads(
                           program.block_index(template)))
            if prod:
                card["stream_produces"] = prod
            if reads:
                card["stream_reads"] = reads
            if sources:
                card["stream_sources"] = {
                    name: {"declared": form,
                           "reads": [[layer, source] for layer, source in rows]}
                    for name, (form, rows) in sources.items()}
            hits = {s: windows[s] for s in (prod + reads) if s in windows}
            if hits:
                card["stream_windows"] = hits
        for key in ("cycle_blocks", "layers", "variants"):
            if isinstance(card.get(key), list):
                _annotate_stream_roles(card[key], program)


def build_architecture_view(summary, layer_groups, resolved_spec, format_params, config=None):
    """Build the interactive architecture view data for the frontend."""

    attn_info = summary.get("attention_info", {})
    ffn_info = summary.get("ffn_info", {})
    attn_type = attn_info.get("attention_type") or None
    H = summary.get("hidden_size", 0)
    L = summary.get("num_layers", 0)

    badges, _attn_labels = collect_badges(
        resolved_spec, summary, config, fallback_attn_type=attn_type,
    )

    # The main model (no MTP, no encoder): the headline every page states.
    total = summary.get("main_total_params", 0)

    # Subtitle
    active = summary.get("main_active_params", total)
    subtitle_parts = [f"{format_params(total)} total"]
    if active < total:
        subtitle_parts.append(f"{format_params(active)} active")
    subtitle = " · ".join(subtitle_parts)

    program = Program.of(resolved_spec) if resolved_spec else None
    by_tmpl = {}
    if program is not None:
        for template in program.blocks:
            by_tmpl.setdefault(template.name, template)

    def _block_label(tmpl, count, block_groups_count):
        """Build label and detail for a block group.

        The label is the invocation's short name (`BlockMeta.short_name`),
        the one every page shows the block by.
        """
        invocation = by_tmpl[tmpl]
        call = invocation.attention_call()
        # KDA keeps a GatedDeltaNet-shaped cache, but the architecture card
        # names the user-facing variant (KDA), not the cache family. A block
        # with no attention (a Mamba or MoE sub-layer) names none.
        at = (call.attn_variant or call.kv_label) if call is not None else None
        is_moe = invocation.ffn_type() == "moe"

        detail_parts = [f"×{count} layers"]
        if at: detail_parts.append(at)
        if is_moe:
            detail_parts.append(
                f"Top-{ffn_info['top_k']}/{ffn_info['num_experts']}")

        return invocation.meta.short_name, is_moe, " · ".join(detail_parts)

    # Walk the tree spine — groups become extra sections, everything else is main path
    layers = []
    extra_sections = []

    def _render_group_sublayer(sub, group_is_singleton: bool = False):
        """Convert a group sub-layer (a spine entry) to a list of display
        dicts.

        Returns a list to support sub-layers whose `assignments:` clause
        expands into multiple cards (per-template-run cards or a
        compressed periodic cycle container) — same layout as the main
        transformer stack uses for top-level `assignments:`. Single-card
        cases return a 1-element list so the caller can extend uniformly.

        `group_is_singleton` says the group declares no indexed stack
        address (`checkpoint.prefix`): the sub is a one-shot wrapper around
        an entire tower (vision/audio), not a transformer block stack. For those we suppress attn/ffn detection
        — generic expert-bank inference can misclassify the
        wrapper's internal ViT resblocks (47 ≥ 8), and the resulting
        "Top-K/E" / attention-tag label would be misleading.
        """
        path = tuple(sub.layer_path)
        if sub.kind == "assignments":
            # Heterogeneous sub-stack (e.g. MiMo V2.5 vision tower: 4
            # full-attention blocks at indices 0/9/18/27 + 24 sink blocks
            # at the rest). Reuse the same layout pass the main
            # transformer stack uses so periodic patterns collapse into
            # cycle containers and contiguous same-template runs collapse
            # into single cards — without this branch, group sub-layers
            # with `assignments:` would fall through to the fallback and
            # render as a single empty `type=op` placeholder.
            return _with_stream_maps(program, metas_at.get(path, []),
                                     _layout_assignments_in_order(
                                         metas_at.get(path, []), _block_label))
        if sub.kind == "block" and path in metas_at:
            tmpl = metas_at[path][0].template_name
            invocation = by_tmpl[tmpl]
            # Singleton wrapper (vision tower / audio encoder): the
            # sub is one-shot, not a transformer block stack. Suppress
            # attn/ffn detection — expert-bank inference
            # fallback misfires on the wrapper's internal ViT
            # resblocks and renders a spurious "Top-K/E" label.
            if group_is_singleton:
                return [{
                    "name": invocation.meta.short_name,
                    "type": "dense_block",
                    "detail": "",
                    "expandable": True,
                    "template": tmpl,
                }]
            call = invocation.attention_call()
            at = call.kv_label if call is not None else None
            is_moe = invocation.ffn_type() == "moe"
            # Count from the sub-layer's own repeat when it declares one.
            count = sub.repeat if sub.repeat is not None else 1
            detail_parts = []
            if count > 1:
                detail_parts.append(f"×{count} layers")
            if at:
                detail_parts.append(at)
            if is_moe:
                detail_parts.append(
                    f"Top-{ffn_info['top_k']}/{ffn_info['num_experts']}")
            return [{
                "name": invocation.meta.short_name,
                "type": "moe_block" if is_moe else "dense_block",
                "detail": " · ".join(detail_parts),
                "expandable": True,
                "template": tmpl,
            }]
        if sub.kind == "op":
            node = _root_node(nodes_at, sub)
            if _is_view(node.declaration):
                passthrough.add(path)
                return []
            return [_op_card(node, summary)]
        return [{"name": sub.name, "type": "op"}]

    # The cards each spine entry is drawn as, by `layer_path`: a group's
    # entries and the group itself are its section. A view op is no card
    # (`passthrough`): a read of it is a read of what it reads.
    cards_at: Dict[tuple, List[str]] = {}
    passthrough: set = set()
    # Inner-stream reads between two cards of one stack, `(from, to,
    # stream)`, once the cards have their ids.
    across: List[tuple] = []
    # Per `(layer_path, name)` an assignments stack binds to a formal past
    # its carried one: the cards whose templates read that formal.
    formal_readers: Dict[tuple, List[str]] = {}

    def _place(path: tuple, cards: List[dict]) -> None:
        """Give the cards a spine entry produced their ids and record
        them as that entry's."""
        base = "L" + ".".join(str(i) for i in path)
        ids = []
        for k, card in enumerate(cards):
            card["id"] = base if len(cards) == 1 else f"{base}#{k}"
            ids.append(card["id"])
        if ids:
            cards_at[path] = ids

    if resolved_spec:
        tree = iter_tree(resolved_spec)
        # The model-level op each spine entry is (`SpecIRNode.layer_path`),
        # and the block invocations each stack entry expanded into
        # (`BlockMeta.layer_path`): the cards read these, not YAML.
        nodes_at = {tuple(n.layer_path): n for n in tree.top_level
                    if n.layer_path}
        metas_at: Dict[tuple, list] = {}
        for meta, _roots in tree.blocks:
            metas_at.setdefault(tuple(meta.layer_path), []).append(meta)
        for entry in tree.spine:
            path = tuple(entry.layer_path)
            start_idx = len(layers)
            # Group → extra section with sub-layers (auxiliary modules like MTP).
            # Main transformer stacks should use `template:` + `assignments:`
            # (or an inline block with template), never `group: true`.
            if entry.kind == "group":
                repeat = entry.repeat if entry.repeat is not None else 1
                if repeat == 0:
                    continue
                sub_layers: list = []
                # A group whose stacks the checkpoint does not index
                # (`BlockMeta.indexed`) is a one-shot tower.
                group_is_singleton = not any(
                    meta.indexed for sub_entry in entry.children
                    for meta in metas_at.get(tuple(sub_entry.layer_path), ()))
                for sub_entry in entry.children:
                    sub_layers.extend(
                        _render_group_sublayer(sub_entry, group_is_singleton=group_is_singleton) or []
                    )
                # The section is one node of the view's graph: whatever
                # reads the group or a layer in it reads the section, and
                # what it reads (the group's own `inputs:` — the MTP head's
                # backbone tap — or its layers') enters it.
                section = {
                    "name": _section_name(program, entry.name),
                    "repeat": repeat,
                    "type": "extra_module",
                    "sub_layers": sub_layers,
                }
                _place(path, [section])
                extra_sections.append(section)
                for sub_entry in entry.children:
                    if sub_entry.kind in ("assignments", "block"):
                        formal_readers.update(_formal_readers(
                            program, tuple(sub_entry.layer_path),
                            sub_entry.inputs,
                            lambda reading, at=section["id"]:
                                [at] if reading else []))
                # The group's own `inputs:` hands its stack what its layers
                # read: the section reads a name past the carried one when
                # one of its layers does.
                for position, name in enumerate(entry.inputs):
                    if position == 0:
                        continue
                    consumed = any(
                        formal_readers.get((tuple(sub.layer_path), name),
                                           [section["id"]])
                        for sub in entry.children if name in sub.inputs)
                    formal_readers[(path, name)] = (
                        [section["id"]] if consumed else [])
            elif entry.kind == "assignments":
                # Render assignments in actual layer-index order. Compress
                # contiguous runs of one block kind (`family_name`) into one
                # card, its templates the card's variants, and collapse
                # arithmetic-progression markers (e.g. layers 4, 8, 12, …)
                # into a single "cycle" card so the diagram reflects the
                # real interleaved layout.
                cards = _layout_assignments_in_order(
                    metas_at.get(path, []), _block_label,
                )
                layers.extend(_with_stream_maps(
                    program, metas_at.get(path, []), cards))
            elif entry.kind == "block":
                if path not in metas_at:
                    continue
                template = metas_at[path][0].template_name
                count = entry.repeat if entry.repeat is not None else 1
                label, is_moe, detail = _block_label(template, count, 1)
                layers.append({"name": label, "type": "moe_block" if is_moe else "dense_block",
                               "detail": detail, "expandable": True,
                               "template": template})
            else:
                node = _root_node(nodes_at, entry)
                if owns_lookup_table(node.declaration) and entry.repeat is None:
                    # Spec-driven: use spec's `name` (e.g. `embed_tokens`,
                    # `language_model.embed_tokens`); op embedded in detail
                    # alongside the dim breakdown.
                    layers.append({"name": node.name, "type": "embedding",
                                   "detail": f"embedding · d = {H:,} · vocab = {_published_vocab(summary):,}",
                                   "expandable": False})
                elif owns_lookup_table(node.declaration):
                    # Repeated bank of identical embeddings (e.g. MiMo-V2.5
                    # `speech_embeddings`, the schema-prompt's `mtp_embed`
                    # idiom). Render ONE collapsed card with `×N` count,
                    # bank's dims may differ from model-wide V/H (the speech
                    # bank binds `n_speech_vocab` / `d_audio_model`) — read
                    # off the layer's own dims, whose keys are the
                    # embedding op's declared dims (`n_vocab` / `d_model`).
                    # A multimodal model may have no global V/H at all, in
                    # which case the layer's own binding is the only answer.
                    # The vocab is the bank's own fact: a bank that binds no
                    # `n_vocab` is refused, not drawn with the model's (or 0).
                    Vlocal = _own_vocab(node)
                    # Its width likewise: the node's own `d_model` where it
                    # binds one, else the model's published width.
                    Hlocal = node.dims.get("d_model", H) if node.dims else H
                    layers.append({
                        "name": node.name,
                        "type": "embedding",
                        "detail": f"×{entry.repeat} · d = {Hlocal:,} · vocab = {Vlocal:,}",
                        "expandable": False,
                    })
                elif _STYLE_BY_OP.get(node.op) == "norm":
                    # Use spec's own `name` (e.g. `final_norm`, `model.norm`)
                    # as the label; the op (`rmsnorm`/`layernorm`) is the
                    # subtitle. Mirrors how block cards show the bound
                    # component as a subtitle under the spec node name —
                    # keeps the architecture overview spec-faithful.
                    layers.append({"name": node.name, "type": "norm",
                                   "detail": _OP_LABEL[node.op],
                                   "expandable": False})
                elif node.boundary_role == "output_head":
                    # The declared output head (`boundary_role`), whatever
                    # the checkpoint calls it — GLM-4's is `output_layer`.
                    # Its vocab is the model's published one
                    # (`Headline.vocab`, carried as `summary.vocab_size`),
                    # the same number the embedding card shows.
                    layers.append({"name": node.name, "type": "lm_head",
                                   "detail": f"vocab = {_published_vocab(summary):,}",
                                   "expandable": False})
                elif _is_view(node.declaration):
                    passthrough.add(path)
                elif not _is_operand(node.declaration):
                    # Every other model-level op leaf — Kimi-K3's output AttnRes
                    # combine, V4-Pro's hyper-connection broadcast and head
                    # component — is a card of its own kind, the same card a
                    # group sub-layer gets. Without this arm a leaf that is
                    # none of the named kinds vanished from the view (and once
                    # rendered as a second `lm_head`). A zero-input leaf (a
                    # parameter tensor, a rotary buffer) is an operand of some
                    # other card, not a stage of the model.
                    layers.append({**_op_card(node, summary), "expandable": False})
            if entry.kind != "group":
                _place(path, layers[start_idx:])
            if entry.kind == "assignments":
                across.extend(_reads_across(program, metas_at.get(path, []),
                                            layers[start_idx:]))
            if entry.kind in ("assignments", "block"):
                cards = layers[start_idx:]
                formal_readers.update(_formal_readers(
                    program, path, entry.inputs,
                    lambda reading, cards=cards: [
                        card["id"] for card in cards
                        if _card_templates(card) & reading]))
        edges, model_inputs = _view_edges(program, cards_at, passthrough,
                                          across, formal_readers)
    else:
        edges, model_inputs = None, []
        layers.append({"name": "Token Embedding", "type": "embedding",
                        "detail": f"d = {H:,} · vocab = {_published_vocab(summary):,}",
                        "expandable": False})
        t_groups = [g for g in layer_groups if g.get("layer_type") == "transformer_block"]
        if t_groups:
            layers.append({"name": "Transformer Block", "type": "dense_block",
                "detail": f"×{len(t_groups)} layers · {attn_type}",
                "expandable": True,
            })
        layers.append({"name": "Norm", "type": "norm", "expandable": False})
        layers.append({"name": "LM Head", "type": "lm_head",
                       "detail": f"vocab = {_published_vocab(summary):,}",
                       "expandable": False})

    # Summary table
    summary_table = {"Type": "Dense", "Layers": str(L), "Attention": attn_type or "—"}
    if ffn_info.get("ffn_type") == "moe":
        summary_table["Type"] = "MoE"
        E = ffn_info.get("num_experts", 0)
        K = ffn_info.get("top_k", 0)
        dense_count = summary.get("num_dense_ffn_layers", 0)
        moe_count = summary.get("num_moe_layers", 0)
        # "xD + yM" splits the layers in two: it is shown only when the
        # published counts are such a split. A layer that runs a dense FFN
        # beside its experts (LongCat) is in both, so the line is just L.
        summary_table["Layers"] = (f"{dense_count}D + {moe_count}M"
                                   if dense_count and dense_count + moe_count == L
                                   else str(L))
        summary_table["Experts"] = f"{K}/{E}"

    # Annotate cards with inter-layer stream roles so the inspector can draw
    # the inner stream (DSA full→shared top-k, etc.). Topology only.
    if resolved_spec is not None:
        _annotate_stream_roles(layers, Program.of(resolved_spec))
        for section in extra_sections:
            _annotate_stream_roles(section["sub_layers"], program)

    return {
        "model_name": model_name(summary, resolved_spec),
        "subtitle": subtitle,
        "badges": badges,
        "model_inputs": model_inputs,
        "layers": layers,
        "extra_sections": extra_sections,
        # Who reads whom, between the cards above (by `id`): the view's
        # topology. None when there is no spec to read it off, and the
        # cards stand in order.
        "edges": edges,
        "summary_table": summary_table,
    }


def _with_stream_maps(program, metas, cards):
    """`cards` (one stack's), each that spans more than one layer carrying
    its `stream_map` when its layers read inner streams: per
    stream, per layer of the card, whether it writes it and whose write it
    reads (`Program.inner_stream_layers`) — the layer-to-layer reads inside
    the card, which no edge of the view draws."""
    if not metas:
        return cards
    rows = program.inner_stream_layers(tuple(metas[0].iteration_owner))
    for card in cards:
        span = set(card.get("layers") or ())
        if len(span) < 2:
            continue
        found = []
        for name, by_layer in rows.items():
            cells = [{"layer": layer, "writes": writes,
                      **({} if source is None else {"from": source})}
                     for layer, (writes, source) in by_layer.items()
                     if layer in span]
            # A stream every layer only writes for itself relates no two
            # layers: nothing to map.
            if any("from" in cell for cell in cells):
                found.append({"name": name, "cells": cells})
        if found:
            card["stream_map"] = found
    return cards


def _reads_across(program, metas, cards):
    """`(from card, to card, stream)` per inner-stream read of one stack
    whose reader and writer layers stand on different cards
    (`Program.inner_stream_layers`): DeepSeek-V4.1's layers 15-19 read the
    compressed KV of layer 14, its own Engram card."""
    if not metas:
        return []
    card_at = {layer: card["id"] for card in cards
               for layer in card.get("layers") or ()}
    out = []
    rows = program.inner_stream_layers(tuple(metas[0].iteration_owner))
    for name, by_layer in rows.items():
        for layer, (_writes, source) in by_layer.items():
            if isinstance(source, str) or source is None:
                continue        # a seed, or no read
            src, dst = card_at.get(source), card_at.get(layer)
            if src is not None and dst is not None and src != dst:
                out.append((src, dst, name))
    return out


def _card_templates(card) -> set:
    """The templates a card runs: its own, its variants', its cycle
    positions'."""
    out = {card["template"]} if card.get("template") else set()
    for key in ("variants", "cycle_blocks"):
        for inner in card.get(key) or ():
            out |= _card_templates(inner)
    return out


def _formal_readers(program, path, inputs, cards_of):
    """Per name the stack at `path` binds past its first formal (the
    carried stream, read by the stack's first card), the boxes that read
    the formal it binds: `cards_of` maps the templates that read it
    (`BlockTemplate.reads_formal`) to boxes — the stack's own cards, or the
    section a group's stack is drawn inside, none when no template reads
    it. DeepSeek-V4.1's `input_ids` reaches only its Engram layers."""
    templates = [t for t in program.blocks
                 if tuple(t.meta.layer_path) == path]
    out = {}
    for position, name in enumerate(inputs):
        if position == 0:
            continue
        reading = {t.meta.template_name for t in templates
                   if position < len(t.input_names())
                   and t.reads_formal(t.input_names()[position])}
        out[(path, name)] = cards_of(reading)
    return out


def _view_edges(program, cards_at, passthrough, across=(),
                formal_readers=None):
    """The architecture view's edges and the model inputs they start from.

    Every read between model layers (`Program.layer_reads`) becomes an edge
    from the card that produces what is read — the LAST card of a layer
    drawn as several, its exit — to the FIRST card of the reader, its
    entry; the cards of one layer chain in order (the stack's carried
    residual), and a layer on one reads an inner stream a layer on another
    wrote (`across`). What a stack binds past its carried formal goes to
    the cards that read that formal (`formal_readers`), none if none do. A read of a view op (`passthrough`) is a read of what that op
    reads. A read inside one section is the section's own business, and a
    read of a stream is an edge of kind `stream` labelled with it."""
    def card_of(path, last):
        for depth in range(len(path), 0, -1):
            ids = cards_at.get(path[:depth])
            if ids:
                return ids[-1] if last else ids[0]
        return None

    reads_of: Dict[tuple, list] = {}
    for read in program.layer_reads:
        reads_of.setdefault(read.reader, []).append(read)

    def origins(read, seen=()):
        """`(card id, stream)` per producer `read` reaches, through views."""
        if not read.sources:
            return [(f"in:{read.name}", read.stream)]
        out = []
        for source in read.sources:
            if source in passthrough and source not in seen:
                for inner in reads_of.get(source, ()):
                    out.extend((at, read.stream or stream) for at, stream
                               in origins(inner, (*seen, source)))
                continue
            at = card_of(source, last=True)
            if at is not None:
                out.append((at, read.stream))
        return out

    edges: Dict[tuple, dict] = {}

    def add(src, dst, stream):
        if src == dst:
            return
        kind = "data" if stream is None else "stream"
        edge = edges.setdefault((src, dst, kind),
                                {"from": src, "to": dst, "kind": kind})
        if stream is not None and stream not in edge.setdefault("streams", []):
            edge["streams"].append(stream)

    for ids in cards_at.values():
        for a, b in pairwise(ids):
            add(a, b, None)
    for src, dst, stream in across:
        add(src, dst, stream)
    for reader, reads in reads_of.items():
        if reader in passthrough:
            continue
        entry = card_of(reader, last=False)
        if entry is None:
            continue
        for read in reads:
            dsts = (formal_readers or {}).get((reader, read.name), [entry])
            for src, stream in origins(read):
                for dst in dsts:
                    add(src, dst, stream)
    out = list(edges.values())
    named = sorted({e["from"][3:] for e in out if e["from"].startswith("in:")},
                   key=list(program.model_inputs).index)
    return out, [{"id": f"in:{name}", "name": name} for name in named]


def _is_view(decl) -> bool:
    """A view op (`OpSpec.is_view_op`): it lays its input out again and
    computes nothing, so the overview draws no card for it — the block
    graph absorbs the same ops into their neighbours."""
    return decl is not None and bool(decl.is_view_op)


def _model_class(resolved_spec) -> Optional[str]:
    """The checkpoint's model class (`Program.architectures`), or None."""
    if resolved_spec is None:
        return None
    return next(iter(Program.of(resolved_spec).architectures), None)


def _section_name(program, name: str) -> str:
    """An auxiliary stack's heading: its name, and what it is when the name
    does not say (`video (vision encoder, runs visual's weights)`)."""
    stack = (next((st for st in program.stacks if st.name == name), None)
             if program is not None else None)
    if stack is None:
        return name
    what = []
    kind = stack.aux_kind if stack.aux_kind not in (None, name) else None
    if stack.is_encoder:
        what.append(f"{kind} encoder" if kind else "encoder")
    elif kind:
        what.append(kind)
    if stack.reruns:
        what.append(f"runs {stack.reruns}'s weights")
    return f"{name} ({', '.join(what)})" if what else name


def model_name(summary: Dict[str, Any], resolved_spec) -> str:
    """The diagram's title: the config's `model_type` (`DEEPSEEK_V3`), else
    the checkpoint's model class as it is spelled, else the spec's name."""
    model_type = summary.get("model_type") or ""
    if model_type and model_type.lower() != "unknown":
        return model_type.upper()
    return (_model_class(resolved_spec)
            or (resolved_spec.spec.name if resolved_spec else None)
            or "Model")


def _op_card(node, summary) -> Dict[str, Any]:
    """The card for an op leaf that is none of the named kinds (embedding /
    norm / output head / block): the spec's name with the op as subtitle, a
    `[out, in]` weight's in → out beside it, a lookup table's vocabulary."""
    op_name = _OP_LABEL.get(node.op, node.op)
    decl = node.declaration
    detail = ""
    if owns_lookup_table(decl):
        detail = f"vocab = {_published_vocab(summary):,}"
    elif is_out_in_matrix(decl) and node.dims_raw:
        in_v, out_v = node.dims.get("in"), node.dims.get("out")
        if isinstance(in_v, int) and isinstance(out_v, int):
            detail = f"{in_v:,} → {out_v:,}"
        else:
            detail = (f"{node.dims_raw.get('in', '?')} → "
                      f"{node.dims_raw.get('out', '?')}")
    return {"name": f"{node.name} ({op_name})", "type": "op", "detail": detail}


def _is_operand(decl) -> bool:
    """A zero-input declaration — a parameter tensor, a rotary buffer: a DAG
    source the block graph draws no incoming edge for and folds into its
    consumer. Not a stage of the model, so not a card of the overview."""
    return decl is not None and decl.shape_kind == ShapeKind.SOURCE


def _root_node(nodes_at, entry):
    """The IR root node a model-level op entry is (`SpecIRNode.layer_path`),
    which carries its op, declaration and resolved dims."""
    node = nodes_at.get(tuple(entry.layer_path))
    if node is None:
        raise ValueError(
            f"model-level leaf {entry.name!r} has no IR root node — the tree "
            "and the spec's model list disagree; refusing to render it as "
            "undeclared")
    return node


def _published_vocab(summary) -> int:
    """The model's published vocab (`Headline.vocab`, carried as
    `summary.vocab_size`) for a card that states it.

    A declared embedding or output head always has one; a summary without a
    positive vocab is a malformed payload, refused here rather than drawn as
    a `vocab = 0` that reads like a measurement."""
    vocab = summary.get("vocab_size")
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab <= 0:
        raise ValueError(
            f"architecture view: a card states the model's vocab, but the "
            f"summary publishes vocab_size={vocab!r}")
    return vocab


def _own_vocab(node) -> int:
    """A repeated lookup-table bank's own vocab (`n_vocab` on its node):
    the bank may bind a vocab of its own, so the model's is no answer for
    it, and a bank that binds none is malformed."""
    vocab = (node.dims or {}).get("n_vocab")
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab <= 0:
        raise ValueError(
            f"{node.display_path}: a repeated embedding bank states its "
            f"vocab, but its node binds n_vocab={vocab!r}")
    return vocab


def _embeddings_tied(resolved_spec) -> Optional[bool]:
    """Whether the input embedding's table is also a projection's weight
    (`Program.embeddings_tied`) — not whether any weight is shared (an MTP
    head reading the LM head's is), nor whether HF JSON has the key."""
    if resolved_spec is None:
        return None
    return Program.of(resolved_spec).embeddings_tied


def build_model_config(summary, resolved_spec):
    """Build a structured model_config from the spec and the analysis'
    summary. No raw config: every number is the one resolution's (the
    program's binding, the spec's `view:` values), every label the spec's
    own text."""
    attn_info = summary.get("attention_info", {})
    ffn_info = summary.get("ffn_info", {})

    # Dimensions — Binding.this integers, not the loader table.
    dims = {}
    view = {}
    program = Program.of(resolved_spec) if resolved_spec else None
    if resolved_spec:
        for name, raw in program.base_binding.flat().items():
            try:
                dims[name] = int(value_of(raw))
            except (TypeError, ValueError):
                continue
        # Display facts that are not dims, bound by the spec's `view:` block
        # (ArchSpec.VIEW_KEYS) and resolved by dim_resolution in native types.
        view = dict(resolved_spec.view_values)

    def _extract_config_dims():
        """The dims the spec binds from the config: each symbol whose
        authored binding names a config field, with that binding as the
        spec writes it (`config.sliding_window ?? "128"`) and the value the
        one resolution bound — not the config re-read, nor a `??` chain
        re-walked to guess which alternative it took. A symbol the
        resolution left without an integer value has no row."""
        if not resolved_spec:
            return []
        return [{"field": str(binding), "symbol": symbol,
                 "value": dims[symbol]}
                for symbol, binding in resolved_spec.spec.dims.items()
                if references_config(binding) and symbol in dims]

    # Base info from spec-normalized summary
    # The main model (no MTP, no encoder), as every headline states it.
    total_params = summary.get("main_total_params", 0)
    active_params = summary.get("main_active_params", total_params)
    result = {
        "model_type": summary.get("model_type", "unknown"),
        # The checkpoint's model class: what the page names a model whose
        # config states no `model_type`.
        "model_class": _model_class(resolved_spec),
        "total_params": total_params,
        "total_active_params": active_params if active_params != total_params else None,
        "precision": _resolve_precision(summary),
        "tied_weight": _embeddings_tied(resolved_spec),
        "max_position_embeddings": dims.get("s_max"),
        "dims": {k: v for k, v in dims.items()
                 if k not in ("B", "S", "s_ctx") and isinstance(v, int)},
        "config_dims": _extract_config_dims(),
    }

    # Attention — each family's geometry as the analysis states it
    # (`Analysis.attention_families`, carried in the summary): the dims its
    # attention call reads, each named by the spec's config binding of it
    # (`config_dims`; None where a layer assignment sets it, or the spec
    # binds it from no config field). The model-level head dims no call
    # reads (an MLA model's `num_key_value_heads`) are listed apart, not
    # given to a family.
    attn = {
        "type": attn_info.get("attention_type") or None,
    }
    families = attn_info.get("families")
    if program is not None and families is not None:
        config_field = {row["symbol"]: row["field"]
                        for row in result["config_dims"]}

        def _fact(symbol, value, rebound=()):
            return {"symbol": symbol, "value": value,
                    "config": (None if symbol in rebound
                               else config_field.get(symbol))}

        attn["families"] = [
            {"name": family["name"], "layers": family["layers"],
             "dims": [_fact(d["symbol"], d["value"], family["rebound"])
                      for d in family["dims"]]}
            for family in families]
        read = {d["symbol"] for family in families for d in family["dims"]}
        headline = program.headline
        attn["model_dims"] = [
            _fact(symbol, value) for symbol, value in (
                ("n_heads", headline.heads),
                ("n_kv_heads", headline.kv_heads),
                ("d_head", headline.head_dim))
            if value is not None and symbol not in read]
    # RoPE theta from the spec's view. Some hybrid attention models (e.g. Step-3.5
    # Flash) store a per-layer list — condense to distinct values.
    def _fmt_magnitude(v):
        if not isinstance(v, (int, float)):
            return str(v)
        if float(v).is_integer():
            v = int(v)
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:g}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:g}k"
        return str(v)
    if "rope_theta" in view:
        rt = view["rope_theta"]
        if isinstance(rt, (list, tuple)):
            uniq = sorted({x for x in rt if isinstance(x, (int, float))}, reverse=True)
            attn["rope_theta"] = " / ".join(_fmt_magnitude(v) for v in uniq) + f" (per-layer, {len(rt)} layers)"
        else:
            attn["rope_theta"] = _fmt_magnitude(rt)
    rope_scaling = view.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        attn["rope_scaling"] = rope_scaling.get("type") or rope_scaling.get("rope_type")
    elif rope_scaling:
        attn["rope_scaling"] = str(rope_scaling)
    # TODO(audit C7): window / SWA KV heads read dims by name; read an eval-published window fact once there is one.
    if dims.get("s_window"):
        attn["sliding_window"] = dims["s_window"]
    # SWA-variant head widths (hybrid models such as MiMo-V2) — view keys;
    # the KV-head count is the dim `n_kv_heads_swa`.
    if "swa.num_heads" in view:
        attn["swa"] = {
            "num_heads": view["swa.num_heads"],
            "num_kv_heads": dims.get("n_kv_heads_swa"),
            "head_dim": view.get("swa.head_dim"),
            "v_head_dim": view.get("swa.v_head_dim"),
            "rope_theta": (_fmt_magnitude(view["swa.rope_theta"])
                           if "swa.rope_theta" in view else None),
        }
    result["attention"] = attn

    # FFN — from spec-normalized info. A type the analyzer did not state
    # (None, or no summary) is not modeled, shown as null, never as dense.
    ffn = {
        "type": ffn_info.get("ffn_type"),
        "intermediate_size": ffn_info.get("intermediate_size"),
        "activation": view.get("hidden_act"),
    }
    if ffn_info.get("ffn_type") == "moe":
        ffn["num_experts"] = ffn_info.get("num_experts")
        ffn["top_k"] = ffn_info.get("top_k")
        if ffn_info.get("num_shared_experts"):
            ffn["shared_experts"] = ffn_info["num_shared_experts"]
        # The analyzer states the width for every MoE model (None when its
        # banks bind different ones); a summary without it is refused.
        width = ffn_info["expert_intermediate_size"]
        if width:
            ffn["moe_intermediate_size"] = width
    if program is not None:
        # The backbone layers that run a dense FFN, from the declared layer
        # groups (`Headline.dense_layers`); with none, the dense width
        # (`intermediate_size`) does not apply.
        ffn["dense_layers"] = list(program.headline.dense_layers)
        mtp = sum(template.stored_count
                  for template in program.stack_blocks("mtp"))
        if mtp:
            ffn["mtp_layers"] = mtp
    result["ffn"] = ffn

    # Norm — type from the spec's model layers, epsilon from the view
    norm_op = None
    if resolved_spec:
        norm_op = next((node.op for node in iter_tree(resolved_spec).top_level
                        if len(node.layer_path) == 1
                        and _STYLE_BY_OP.get(node.op) == "norm"), None)
    result["norm"] = {
        "type": _OP_LABEL.get(norm_op, norm_op or "unknown"),
        "eps": view.get("norm.eps"),
    }

    # Block types — from BlockMeta kinds (supports heterogeneous)
    if resolved_spec:
        program = Program.of(resolved_spec)
        # One row per template: the groups a `read: all` stack splits one
        # declaration into (Kimi-K3's `block_mla_moe` by history length)
        # are one kind of block, and their layers sum.
        block_types: Dict[str, Dict[str, Any]] = {}
        for template in program.stack_blocks():
            row = block_types.get(template.name)
            if row is not None:
                row["count"] += template.count
                continue
            call = template.attention_call()
            display = (call.attn_variant or call.kv_label) if call is not None else None
            block_types[template.name] = {
                "template": template.name,
                # The short name every page shows the block by.
                "label": program.block_label(template),
                "count": template.count,
                # Canonical display label (attn_variant or kv_label) so DSA
                # layers don't read as plain MLA; None for a block with no
                # attention, as `ffn_type` is for one with no FFN.
                "attention_type": display,
                "ffn_type": template.ffn_type(),
            }
        block_types = list(block_types.values())
        result["block_types"] = block_types

    return result
