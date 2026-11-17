"""Runtime weight allocations joined to the persistent artifact ledger.

An alias allocation charges only storage beyond its existing source artifact.
Execution still consumes that source's geometry, so storage accounting cannot
silently change FLOPs or weight traffic.
"""
from dataclasses import dataclass, replace
from math import prod

from backend.arch.eval.dim import value_of
from backend.arch.eval.storage import StorageKind
from .persistent import ArtifactKey, PhysicalArtifact, STORED, RUNTIME_ALLOCATION


@dataclass(frozen=True)
class RuntimeWeightAllocation:
    source: ArtifactKey
    name: str
    global_shape: tuple[int, ...]
    runtime_shape: tuple[int, ...]
    aliases_source: bool = False


def join_runtime_allocations(artifacts, allocations):
    """Append typed resident-only buffers, preserving checkpoint artifacts."""
    sources = {row.key: row for row in artifacts}
    names, aliases = set(), set()
    out = list(artifacts)
    for index, allocation in enumerate(allocations):
        source = sources[allocation.source]
        identity = (allocation.source, allocation.name)
        if identity in names:
            raise ValueError(f'duplicate runtime allocation {identity}')
        names.add(identity)
        if source.surface != STORED or source.resident_storage_dtype not in ('BF16', 'F16') or source.resident_packing_factor != 1:
            raise NotImplementedError('runtime weight copies require an unpacked BF16/F16 source')
        local = 2 * prod(allocation.runtime_shape)
        whole = 2 * prod(allocation.global_shape)
        if allocation.aliases_source:
            if allocation.source in aliases:
                raise ValueError('multiple runtime allocations alias the same source')
            aliases.add(allocation.source)
            local -= source.resident_bytes_per_instance
            whole -= source.resident_bytes_unsharded
        if min(local, whole) < 0:
            raise ValueError('runtime allocation cannot contain its aliased source')
        out.append(replace(
            source, surface=RUNTIME_ALLOCATION, metadata_index=index,
            artifact_name=allocation.name, kind=StorageKind.BUFFER,
            global_shape=allocation.global_shape, runtime_shape=allocation.runtime_shape,
            runtime_layout=None, metadata_geometry=None,
            saved_storage_dtype=None, saved_logical_dtype=None, saved_num_bits=None,
            saved_packing_factor=None, saved_selected_by=None, saved_physical_slot_shape=None,
            resident_physical_slot_shape=allocation.runtime_shape,
            saved_bytes_per_instance=0, saved_bytes_per_rank=0,
            resident_bytes_per_instance=local,
            resident_bytes_per_rank=local * source.runtime_instances,
            resident_bytes_unsharded=whole, runtime_allocation=allocation,
        ))
    return tuple(out)


def sparse_dcp_allocations(analysis, artifacts: tuple[PhysicalArtifact, ...]):
    """SGLang's separate local absorbed, Qrep and CUDA preproject-V buffers.

    Source: sglang 1b2870d33de7b5f31844fd74e97eea6d7896c53a,
    model_runner._prepare_replicated_q_proj, Kimi post_load_weights and
    forward_mla._prepare_dcp_v_weights. The last V gather is a distinct
    allocation even when w_vc_qrep already exists.
    """
    allocations = []
    for plan in analysis.rule.sparse_dcp.storage_plans:
        def stored(role):
            rows = [row for row in artifacts if row.surface == STORED and
                    row.node_uid == plan.nodes[role].uid]
            if len(rows) != 1:
                raise NotImplementedError('sparse DCP runtime copies require one unsplit weight per projection')
            return rows[0]

        q = stored('query_projection')
        kv = stored('stored_kv_projection')
        if plan.preproject_v and kv.resident_storage_dtype != 'BF16':
            raise NotImplementedError('SGLang sparse DCP reference preprojection requires BF16 dense weights; cache precision is independent')
        def dims(role):
            env = analysis.node_dims.values[plan.nodes[role].uid]
            return tuple(int(value_of(env[key])) for key in ('n_heads', 'in', 'out'))
        k_shape, v_shape = dims('absorbed_query'), dims('value_projection')
        heads = k_shape[0]
        if v_shape[0] != heads or prod(k_shape) + prod(v_shape) != prod(kv.global_shape):
            raise ValueError('absorbed K/V geometry does not cover the stored KV projection')
        numerator = heads * prod(kv.runtime_shape)
        denominator = prod(kv.global_shape)
        local_heads, remainder = divmod(numerator, denominator)
        if remainder or ((plan.replicate_q or plan.preproject_v) and local_heads * plan.degree != heads):
            raise ValueError('sparse DCP requires the full attention TP head group')

        def add(source, name, global_shape, runtime_shape, alias=False):
            allocations.append(RuntimeWeightAllocation(source.key, name, global_shape, runtime_shape, alias))

        for name, shape in (('w_kc', k_shape), ('w_vc', v_shape)):
            add(kv, name, shape, (local_heads, *shape[1:]))
        if plan.replicate_q:
            add(q, 'q_b_proj_qrep_weight', (q.global_shape[0] * plan.degree, *q.global_shape[1:]),
                q.global_shape, True)
            for name, shape in (('w_kc_qrep', k_shape), ('w_vc_qrep', v_shape)):
                add(kv, name, (heads * plan.degree, *shape[1:]), shape)
        if plan.preproject_v:
            add(kv, '_dcp_v_weights', (heads * plan.degree, *v_shape[1:]), v_shape)
    return tuple(allocations)


def runtime_read_aliases(analysis, artifacts):
    """Project read ownership from actual allocated operands, not TP factors."""
    from fractions import Fraction
    aliases = {uid: dict(sources) for uid, sources in (analysis.program.read_aliases or {}).items()}
    rows = {(row.node_uid, row.artifact_name): row for row in artifacts
            if row.surface == RUNTIME_ALLOCATION}
    sources = {row.key: row for row in artifacts}
    for plan in analysis.rule.sparse_dcp.plans:
        if not plan.decode:
            continue
        for role, name in (('absorbed_query', 'w_kc_qrep'), ('value_projection', '_dcp_v_weights')):
            if role == 'absorbed_query' and not plan.replicate_q:
                name = 'w_kc'
            if role == 'value_projection' and not plan.preproject_v:
                name = 'w_vc_qrep' if plan.replicate_q else 'w_vc'
            uid = plan.nodes['stored_kv_projection'].uid
            allocation = rows[uid, name]
            source = sources[allocation.runtime_allocation.source]
            aliases[plan.nodes[role].uid] = {
                uid: Fraction(prod(allocation.runtime_shape), prod(source.runtime_shape))}
    return aliases


def sparse_dcp_cache_reads(analysis, layer):
    """Invocation read laws; persistent context capacity keeps its own laws."""
    from backend.arch.eval import cache
    from backend.arch.eval.dim import ceil_div, require_equal
    from backend.arch.eval.layer import Layer
    from .sparse_dcp_contract import SparseDcpWorkload, sparse_dcp_query_pairs
    values = dict(layer.values)
    for plan in analysis.rule.sparse_dcp.plans:
        entry = values[plan.component.uid]
        if not plan.decode:
            # SGLang prefill gathers full index/MLA cache rows before its
            # local-head kernels. Persistent storage remains striped; only
            # this invocation's read surface is whole-context.
            values[plan.component.uid] = replace(
                entry, tensors=tuple(replace(tensor, positions_over=1)
                                     for tensor in entry.tensors))
            continue
        topk = value_of(analysis.node_dims.values[plan.nodes['indexer_topk'].uid]['k'])
        selection = SparseDcpWorkload(
            rows=1, context=cache.CACHE_LEN, topk=topk, world=plan.degree,
            selection_assumption='busiest_rank',
            busiest_rank_factor=value_of(analysis.rule.dcp_selected_load_factor))
        tensors = []
        for tensor in entry.tensors:
            if tensor.kind == 'index_key':
                read = ceil_div(cache.CACHE_LEN, plan.degree) * tensor.positions_over
                tensors.append(replace(tensor, runtime_read_positions=value_of(read)))
            elif tensor.kind in ('latent', 'rope_key'):
                read = selection.valid_per_query * tensor.positions_over
                context = cache.workload_dim(analysis, 's_kv')
                verify = analysis.node_dims.values[plan.nodes['score'].uid]['n_verify']
                width = int(value_of(verify))
                # Verification queries have different causal context lengths.
                # Specialize the finite sum to this workload's verify width;
                # the assumption prevents a compiled law being reused at a
                # different width without rebuilding the projection.
                require_equal(verify, width, lambda at: ValueError(
                    'sparse DCP cache reads require the analyzed verification width'))
                pairs, _ = sparse_dcp_query_pairs(
                    context, width, topk, plan.degree,
                    analysis.rule.dcp_selected_load_factor)
                pairs *= tensor.positions_over
                tensors.append(replace(tensor, runtime_read_positions=value_of(read), pairs=pairs))
            else:
                tensors.append(tensor)
        values[plan.component.uid] = replace(entry, tensors=tuple(tensors))
    return Layer(analysis.program, values)
