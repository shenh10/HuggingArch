"""Bound runtime plans for explicitly declared SGLang sparse DCP graphs.

The spec owns node identities. The layout owns execution placement; persistent
allocations and communication consume the same binding rather than model names.
"""
from backend.arch.runtime_profile import RuntimeProfile
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BoundSparseDcpPlan:
    component: Any
    nodes: Mapping[str, Any]
    decode: bool
    degree: int
    replicate_q: bool = True
    preproject_v: bool = True

    def role(self, node):
        return next((role for role, value in self.nodes.items()
                     if value.uid == node.uid), None)


@dataclass(frozen=True)
class SparseDcpPlans:
    plans: tuple = ()
    # Absorbed weight copies exist even without decode context parallelism.
    storage_plans: tuple = ()

    @classmethod
    def of(cls, program, parallel, phase, runtime_profile=RuntimeProfile.DECLARED):
        runtime_profile = RuntimeProfile(runtime_profile)
        if runtime_profile == RuntimeProfile.ARCHITECTURE:
            return cls()
        if program is None or parallel is None:
            return cls()
        plans = []
        storage_plans = []
        for component in program.sparse_dcp_components:
            spec = component.sparse_dcp
            # This first implementation follows the audited full attention-TP
            # group. Smaller/orthogonal groups require a different replica plan.
            if parallel.dcp > 1 and (parallel.dcp != 8 or parallel.tp_attention != 8 or parallel.cp != 1):
                raise NotImplementedError('SGLang sparse DCP reference profile requires dcp=tp_attention=8 and cp=1')
            nodes = {}
            for role, path in spec.model_dump().items():
                if role == 'backend':
                    continue
                node = component
                for segment in path.split('.'):
                    found = [child for child in node.children if child.name == segment]
                    if len(found) != 1:
                        raise ValueError(f'{component.display_path}: sparse DCP {role} path {path!r} is not unique')
                    node = found[0]
                expected = {
                    'query_projection': 'linear',
                    'absorbed_query': 'per_head_linear',
                    'stored_kv_projection': 'parameter_tensor',
                    'value_projection': 'per_head_linear',
                    'score': 'dsa_attention_score',
                    'softmax': 'dsa_softmax',
                    'apply': 'dsa_attention_apply',
                    'indexer_logits': 'indexer_logits',
                    'indexer_topk': 'dsa_topk_mask',
                }[role]
                if node.op != expected:
                    raise ValueError(
                        f'{component.display_path}: sparse DCP {role} '
                        f'requires {expected}, got {node.op}')
                nodes[role] = node
            enabled = parallel.dcp > 1
            plan = BoundSparseDcpPlan(
                component, nodes, enabled and phase == 'decode', parallel.dcp,
                replicate_q=enabled, preproject_v=enabled)
            storage_plans.append(plan)
            if enabled:
                plans.append(plan)
        return cls(tuple(plans), tuple(storage_plans))

    def owns_cache_node(self, node):
        """Whether a cache endpoint belongs to a declared striped DCP component."""
        owners = {plan.component.uid for plan in self.plans}
        while node is not None:
            if node.uid in owners:
                return True
            node = node.parent
        return False

    def for_node(self, node):
        return next((plan for plan in self.plans if plan.role(node) is not None), None)

    def role(self, node):
        plan = self.for_node(node)
        return plan.role(node) if plan else None

    def runtime_axis(self, node, slot, original, replicated):
        plan = self.for_node(node)
        if plan is None or not plan.decode:
            return original
        role = plan.role(node)
        if ((role == 'query_projection' and slot == 'out') or
                (role in ('absorbed_query', 'value_projection') and slot == 'n_heads')):
            return replicated()
        return original


def transition(rule, node, states, shapes):
    """Physical sparse kernels over striped cache ownership.

    The logical selection remains the global top-k. Its physical score matrix
    contains only the selected positions owned by the busiest cache rank; this
    is workload geometry, never a static cut of the selected index list.
    """
    from dataclasses import replace
    from backend.arch.eval.dim import Dim, ceil_div, equal, expr_of, minimum, ratio, value_of
    from backend.arch.eval.transition import InputRequirements, KernelGeometry
    from .edge import EdgeState
    from .layout import sharded
    from .value import ONLINE_ATTENTION
    from . import attention, linear

    plan = rule.sparse_dcp.for_node(node)
    if plan is None or not plan.decode:
        return None
    role = plan.role(node)
    if role not in ('indexer_logits', 'indexer_topk', 'score', 'softmax', 'apply', 'value_projection'):
        return None
    degree = plan.degree

    def extent(original, value):
        return Dim(expr_of(original), value) if isinstance(original, Dim) else value

    def axes(shape, state):
        return rule.attention.kernel_local_axes(node, shape, state, 'sparse DCP input', rule.project_local_extent)

    def selected(original):
        from .sparse_dcp_contract import SparseDcpWorkload
        context = rule.env[plan.nodes['indexer_logits'].uid]['s_kv']
        workload = SparseDcpWorkload(
            rows=1, context=context, topk=value_of(original), world=degree,
            selection_assumption='busiest_rank',
            busiest_rank_factor=rule.dcp_selected_load_factor)
        return extent(original, value_of(workload.valid_per_query))

    def geometry(inputs, output):
        from .sparse_dcp_contract import sparse_dcp_query_pairs
        context = value_of(rule.env[plan.nodes['indexer_logits'].uid]['s_kv'])
        width = int(value_of(rule.env[node.uid]['n_verify']))
        if not equal(width, 1):
            from backend.arch.eval.dim import require_equal
            from backend.arch.eval.transforms import Workload
            require_equal(Workload.symbols()['s_verify'], width,
                          lambda at: ValueError('sparse DCP verification width requires a new walk'))
        topk = value_of(rule.env[node.uid]['n_topk_keys'])
        selected_pairs, index_pairs = sparse_dcp_query_pairs(
            context, width, topk, degree, value_of(rule.dcp_selected_load_factor))
        if role in ('score', 'softmax', 'apply', 'indexer_logits', 'indexer_topk'):
            local_pairs = index_pairs if role.startswith('indexer_') else selected_pairs
            global_pairs = value_of(rule.env[node.uid]['s_pairs'])
            share = ratio(value_of(local_pairs), global_pairs, empty=0)
        else:
            share = None
        return KernelGeometry(inputs, output, pair_share=share)

    if role == 'indexer_logits':
        required, produced = attention.indexer_logits(rule, node, states, shapes)
        local = [list(axes(shape, state)) for shape, state in zip(shapes, required, strict=True)]
        local[1][1] = extent(local[1][1], ceil_div(value_of(local[1][1]), degree))
        out = (local[0][0], local[0][2], local[1][1])
        return InputRequirements(tuple(required), tuple(required), geometry(tuple(tuple(x) for x in local), out)), produced
    if role == 'indexer_topk':
        required, produced = attention.dsa_topk_mask(rule, node, states, shapes)
        local = list(axes(shapes[0], required[0]))
        local[-1] = extent(local[-1], ceil_div(value_of(local[-1]), degree))
        k = rule.env[node.uid]['k']
        out = (*local[:-1], extent(minimum(shapes[0][-1], k), value_of(k)))
        return InputRequirements(tuple(required), tuple(required), geometry((tuple(local),), out)), produced
    if role == 'score':
        query, _key, _mask = states
        q, k, indices = [list(axes(shape, state)) for shape, state in zip(shapes, states, strict=True)]
        k[1] = q[1]  # view of the one shared cached latent, over all query heads
        k[2] = selected(k[2])
        out = (q[0], q[1], q[2], k[2])
        provenance = rule.value_provenance(node, query=0, key=1)
        produced = EdgeState((query.layout[0], query.layout[1], query.layout[2], rule.replicated()),
                             None, query.workload, provenance)
        return InputRequirements(tuple(states), tuple(states), geometry((tuple(q), tuple(k), tuple(indices)), out)), produced
    if role == 'softmax':
        state = states[0]
        local = list(axes(shapes[0], state))
        local[-1] = selected(local[-1])
        return InputRequirements(tuple(states), tuple(states), geometry((tuple(local),), tuple(local))), state
    if role == 'apply':
        scores, value = states
        s, v = [list(axes(shape, state)) for shape, state in zip(shapes, states, strict=True)]
        v[1] = s[1]
        s[-1] = selected(s[-1])
        v[2] = selected(v[2])
        out = (s[0], s[1], s[2], v[3])
        from .transfer_table import CONTRACT, transfer
        pending = rule.cell(node, 'sparse owned-position contraction',
                            transfer(CONTRACT, (sharded(rule.cache_family),), reducer=ONLINE_ATTENTION)).pending_reduction
        pending = pending.with_provenance(node.uid, score=plan.nodes['score'].uid,
                                          query=plan.nodes['score'].inputs_obj[0].src.uid)
        produced = EdgeState((scores.layout[0], scores.layout[1], scores.layout[2], value.layout[3]), pending, scores.workload)
        return InputRequirements(tuple(states), tuple(states), geometry((tuple(s), tuple(v)), out)), produced
    if role == 'value_projection':
        required, produced = linear.per_head_linear(rule, node, states, shapes)
        local = axes(shapes[0], states[0])
        width = rule.env[node.uid]['out']
        out = (*local[:-1], width)
        head = rule.attention.declared_query_head(int(value_of(rule.env[node.uid]['n_heads'])))
        produced = replace(produced, layout=(produced.layout[0], head, *produced.layout[2:]),
                           value=states[0].value)
        return InputRequirements(tuple(required), tuple(required), KernelGeometry((local,), out)), produced
