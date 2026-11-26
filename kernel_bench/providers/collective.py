"""Collectives over a group of cards, through torch.distributed's NCCL.

Each provider here is one NCCL collective as a serving framework issues it
on a card with no peer-to-peer access — which on a GeForce part is every
framework's path: vLLM's and sglang's custom all-reduce need P2P (or
NVLink), and without it both hand the tensor to NCCL. The measured number
is that launch.

A `build` here is ONE rank's launch: it asks torch.distributed which rank it
is, and its inputs are already that rank's (`RefSpec.per_rank`). The group
runs it on every rank (`comm.Group.case`), and the signature's oracle
(`refs`) computes the exchange from rank 0's copy. `kernel_bench/
ARCHITECTURE.md`, "How a collective is measured", says why it is shaped this
way.
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register


def _all_reduce(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch.distributed as dist
    buf = inputs["payload"].clone()

    def run():
        dist.all_reduce(buf)
        return buf
    return BenchCase(run=run, to_canonical=lambda out: out)


def _all_gather(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch.distributed as dist
    p = inputs["payload"].reshape(-1)
    rank, world = dist.get_rank(), dist.get_world_size()
    k = p.numel() // world
    send = p[rank * k:(rank + 1) * k].clone()
    out = torch.empty_like(p)

    def run():
        dist.all_gather_into_tensor(out, send)
        return out
    return BenchCase(run=run, to_canonical=lambda o: o)


def _reduce_scatter(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch.distributed as dist
    send = inputs["payload"].reshape(-1).clone()
    out = torch.empty(send.numel() // dist.get_world_size(), dtype=send.dtype,
                      device=send.device)

    def run():
        dist.reduce_scatter_tensor(out, send)
        return out
    return BenchCase(run=run, to_canonical=lambda o: o)


def _all_to_all(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch.distributed as dist
    send = inputs["payload"].reshape(-1).clone()
    out = torch.empty_like(send)

    def run():
        dist.all_to_all_single(out, send)
        return out
    return BenchCase(run=run, to_canonical=lambda o: o)


def _splits(inv: Invocation) -> str:
    """Every rank's share of the payload is equal: a payload that does not
    divide is not a launch any framework makes, declined before inputs."""
    n, ranks = inv.dims["n_comm_elems"], inv.dims["n_comm_ranks"]
    if n % ranks:
        raise NotImplementedError(f"{n} elements do not split over {ranks} ranks")
    return ""


# The all-reduce moves the whole buffer on every rank; the other three hand
# each rank an equal share of it.
for _signature, _build, _params in (("all_reduce", _all_reduce, None),
                                    ("all_gather", _all_gather, _splits),
                                    ("reduce_scatter", _reduce_scatter, _splits),
                                    ("all_to_all", _all_to_all, _splits)):
    register(Provider(
        library="torch", signature=_signature,
        kernel_impl=f"torch_nccl_{_signature}",
        import_name="torch",
        serves=KernelSignature(signature=_signature, activation=TORCH_FLOAT),
        build=_build,
        impl_params=_params,
    ))


def _allreduce_rmsnorm(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """all-reduce, then residual + RMSNorm: what a TP>1 layer seam runs where
    the fused all-reduce kernel is not available — NCCL's all-reduce and the
    norm as two launches, the unfused floor of the fused cell. Operands at
    the cell's dtype, as `torch_fused_add_rmsnorm` takes them."""
    import torch
    import torch.distributed as dist

    from ..refs import torch_dtype
    dt = torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).clone()
    residual, w = inputs["residual"].to(dt), inputs["w"].to(dt)
    eps = inputs["eps"]

    def run():
        dist.all_reduce(x)
        r = residual + x
        return r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * w
    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="allreduce_fused_rmsnorm",
    kernel_impl="torch_nccl_allreduce_rmsnorm",
    import_name="torch",
    serves=KernelSignature(signature="allreduce_fused_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_allreduce_rmsnorm,
))
