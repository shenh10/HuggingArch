"""Readable dump of a spec's RESOLVED structure — substrate for the
inspector's source⟺structure equivalence reflection.

The component graphs an author writes are scattered across yaml files and
reference each other by name; this consolidates them into one view: the
per-layer block assembly, then each component's nodes with op + dataflow
(`<- inputs`) + role + KV attributes. The agent reads this next to the
modeling source and JUDGES whether they're equivalent — the tool only
PRESENTS the structure, it does not decide equivalence (that judgment is
lossy to mechanize; the source is the ground truth and reading it is the
agent's job).

CLI: `python -m backend.arch.structure <model_id> [--spec NAME]`.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from backend.custom_id import is_custom_id
from backend.arch.spec_tree import iter_blocks

if TYPE_CHECKING:
    from backend.arch.eval.program import BlockTemplate


@dataclass(frozen=True)
class BlockClassification:
    """Display/accounting traits of one invocation, from IR role and eval.cache."""

    #: `kv_label` of the attention component the block reaches, or None when
    #: the block has no attention (a Mamba block). Never inferred from head
    #: counts: the only blocks that ever hit that inference were ones with
    #: no attention at all, and they came out labelled MHA.
    attention_type: Optional[str]
    attention_display: Optional[str]
    #: `moe` / `dense` from the ffn component's role, or None when no
    #: reachable component carries one. An inline expert bank the author
    #: did not label shows as unlabelled rather than being guessed from a
    #: repeat count; the spec is where that fact belongs.
    ffn_type: Optional[str]


def classify_block(template: "BlockTemplate") -> BlockClassification:
    """Kinds of this invocation: its attention call's cache identity and its
    FFN type."""
    call = template.attention_call()
    ffn_type = template.ffn_type()
    return BlockClassification(
        attention_type=call.kv_label if call is not None else None,
        attention_display=(
            (call.attn_variant or call.kv_label) if call is not None else None),
        ffn_type=ffn_type,
    )


def layer_configs(resolved) -> List[Dict[str, Any]]:
    """The backbone's layer groups (`Program.stack_blocks`) with public
    structural metadata."""
    from backend.arch.eval.program import Program
    program = Program.of(resolved)
    configs: List[Dict[str, Any]] = []
    for template in program.stack_blocks():
        classification = classify_block(template)
        item: Dict[str, Any] = {
            "count": template.count,
            "template": template.name,
            "attn_type": classification.attention_type,
            "attn_display": classification.attention_display,
            "ffn_type": classification.ffn_type,
        }
        configs.append(item)
    return configs


def iter_reachable_components(resolved, seeds: Optional[Sequence[str]] = None):
    """Yield (component_name, BlockGraph) for every component reachable from
    the spec's model — following BOTH `op:` references AND `params:`
    component bindings (the `template + params: {attn, ffn}` sugar, where the
    real attention/FFN component is named in `params`, not `op`). Deduped by
    name. Shared structural walk reused by structure rendering + the
    type-contract field checks; the walk itself is `ArchSpec.reachable_graphs`,
    which the load-time stream scope check asks too.

    `seeds` narrows the walk to what those templates reach, for a caller
    asking about one assignment rather than the whole model — a per-layer
    binding (`assignments[].dims`) is only in force for the components its
    own assignment reaches, and the axiom checker asks exactly that.
    """
    return resolved.spec.reachable_graphs(seeds)


def render_structure(resolved) -> str:
    """Render the resolved spec structure as readable text."""
    spec = resolved.spec
    out: List[str] = [f"=== {spec.name} — resolved structure ==="]

    # Per-layer block assembly (template → count), one row per template
    # the model instantiates (`iter_blocks`).
    metas = iter_blocks(resolved)
    if metas:
        out.append("\nBlock templates (per-layer):")
        seen: set = set()
        for meta in metas:
            # A looped stack's later pass and its gate are rows of their own.
            key = (meta.template_name, meta.pass_, meta.loop_gate)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"  {meta.template_name}"
                       + ("  (loop gate)" if meta.loop_gate
                          else f"  (pass {meta.pass_})" if meta.pass_ > 1
                          else "")
                       + f"  ×{meta.count} layers")

    out.append("\nComponents (op + dataflow + role + KV):")
    for name, bg in iter_reachable_components(resolved):
        role = bg.role
        attrs = []
        for f in ("kv_label", "attn_variant"):
            v = getattr(bg, f, None)
            if v:
                attrs.append(f"{f}={v}")
        head = f"\n[{name}]  role={role}"
        if attrs:
            head += "  " + " ".join(attrs)
        out.append(head)
        for nname, gn in bg.graph.items():
            ins = ", ".join(gn.inputs) if gn.inputs else "—"
            dims = gn.dims or {}
            dimstr = ""
            if dims:
                dimstr = " {" + ", ".join(f"{k}={v}" for k, v in dims.items()) + "}"
            rtag = f"  role={gn.role}" if gn.role else ""
            out.append(f"    {nname}: {gn.op}{dimstr}{rtag}  <- {ins}")
    return "\n".join(out)


def _resolve(model_id: str, spec_name: Optional[str], overlay_dir: Optional[str] = None):
    """Resolve a model_id to a ResolvedSpec via the same loader path as
    validate (`hf.metadata.fetch_config`).

    `overlay_dir` (a per-run overlay root) layers `<dir>/models` +
    `<dir>/components/drafts` onto the shared tree so an in-progress private
    spec resolves during the agent's own iteration loop."""
    from backend.arch.loader import SpecLoader, custom_spec_file, overlay_kwargs
    from backend.arch.snapshots import default_overlay_dir
    from backend.hf.metadata import fetch_config

    # The agent's in-loop `structure` runs without --overlay-dir; find its
    # per-run overlay via the env the run injected (explicit arg still wins).
    overlay_dir = default_overlay_dir(overlay_dir)
    config = fetch_config(model_id, overlay_dir)
    if config is None:
        raise ValueError(f"Cannot fetch config for {model_id}")

    loader = SpecLoader(**overlay_kwargs(overlay_dir))
    if spec_name:
        return loader.resolve_by_name(spec_name, config, model_id=model_id)
    if is_custom_id(model_id):
        spec_path = custom_spec_file(model_id, overlay_dir)
        if spec_path:
            spec = loader.load_model_spec(spec_path)
            return loader.resolve(spec, config, model_id=model_id) if spec else None
        return None
    return loader.match(config, model_id=model_id)


def main():
    parser = argparse.ArgumentParser(
        description="Dump a spec's resolved structure for equivalence reflection")
    parser.add_argument("model_id", help="HuggingFace model ID (or custom:<name>)")
    parser.add_argument("--spec", help="Spec name (auto-match if not given)")
    parser.add_argument("--overlay-dir", dest="overlay_dir", default=None,
                        help="Per-run overlay root (layers <dir>/models + "
                             "<dir>/components/drafts on the shared tree)")
    args = parser.parse_args()

    resolved = _resolve(args.model_id, args.spec, overlay_dir=args.overlay_dir)
    if resolved is None:
        print(f"No spec resolved for {args.model_id}", file=sys.stderr)
        sys.exit(1)
    print(render_structure(resolved))


if __name__ == "__main__":
    main()
