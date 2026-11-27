"""Prefill/decode disaggregation: the SOL first-order metrics.

Phase-C disaggregation splits serving into a prefill deployment and a decode
deployment. This module answers the two questions that split raises — what
the KV handoff costs, and how decode-heavy the pipeline is — from two
`estimate_capacity` results. It lives here, not in the API layer, because
both answers are speed-of-light semantics over estimate outputs.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

from backend.inference import gpu_specs
from backend.inference.comm_cost import COMM_REACH_SCALEUP, comm_reach
from backend.inference.gpu_specs import gbps_to_bytes_per_sec, inter_node_bw_gbps


class HandoffLink(NamedTuple):
    """The link a prefill -> decode KV handoff crosses: its name, its
    per-direction bandwidth in the sheet's GB/s, and -- where there is no
    bandwidth to divide by -- why not."""
    name: str
    bw_gbps: Optional[float]
    unavailable_reason: str = ""

    @classmethod
    def between(cls, prefill_result: Dict[str, Any],
                decode_result: Dict[str, Any]) -> "HandoffLink":
        """The link the two clusters' own placement puts between them.

        The two are separate machines, so the handoff always leaves the
        chassis. Whether it also leaves the SCALE-UP fabric is the question
        a collective group is asked (`comm_cost.comm_reach`), put to the
        ranks the two clusters occupy together: each rounded up to whole
        chassis, since a machine is not shared between them. They stay on
        scale-up only on a card whose domain is wider than its chassis and
        only while both fit inside one -- an Ascend supernode's 768 dies.
        On every card that declares no such domain the answer is the NIC,
        as it has always been: two 4-GPU clusters on 8-GPU NVLink nodes
        are 8 ranks, and still two machines.

        Clusters on different cards share no scale-up fabric."""
        decode_gpu = decode_result.get("gpu") or {}
        sku = decode_gpu.get("sku")
        spec = gpu_specs.get_gpu(sku) if sku else None
        sizes = (prefill_result.get("cluster_size"),
                 decode_result.get("cluster_size"))
        if (spec is not None and all(sizes)
                and (prefill_result.get("gpu") or {}).get("sku") == sku):
            host = gpu_specs.gpus_per_node(spec)
            occupied = sum(-(-int(n) // host) * host for n in sizes)
            if comm_reach(
                    occupied, ranks_per_host=host,
                    ranks_per_scaleup_domain=gpu_specs.ranks_per_scaleup_domain(
                        spec)) == COMM_REACH_SCALEUP:
                return cls(spec.scaleup_fabric or "", spec.scaleup_gbps or None,
                           "" if spec.scaleup_gbps else cls._unstated(
                               sku, "scaleup_gbps"))
        bw = inter_node_bw_gbps(decode_gpu.get("rdma_gbps"))
        return cls("RDMA", bw,
                   "" if bw else cls._unstated(sku, "rdma_gbps"))

    @staticmethod
    def _unstated(sku: Optional[str], column: str) -> str:
        return (f"{sku or 'The decode card'} states no bandwidth for the "
                f"link this handoff crosses (gpu_specs.csv column "
                f"`{column}`), so the transfer has no latency to report. "
                f"Pass `kv_transfer_bw_gbps` to state the link.")


def disagg_handoff(
    prefill_result: Dict[str, Any],
    decode_result: Dict[str, Any],
    *,
    kv_transfer_bw_gbps: Optional[float],
    avg_output_tokens: Optional[int],
) -> Dict[str, Any]:
    """Phase-C disaggregation metrics (SOL first-order):

    * KV handoff — the prefill deployment produces the prompt's KV and ships it
      to the decode deployment before decode starts. Latency ≈ per-rank KV
      bytes / per-rank link bandwidth (the transfer parallelizes across ranks,
      so the decode receiver's per-rank shard bounds the wall-clock). A
      stated `kv_transfer_bw_gbps` is the link. Otherwise it is the one the
      two clusters' placement puts between them (`HandoffLink.between`):
      the decode GPU's RDMA NIC, or the card's scale-up fabric where both
      clusters sit inside one scale-up domain wider than a chassis. A card
      that states no bandwidth for that link has no handoff latency to
      report, and says so (`unavailable_reason`) rather than reporting a
      link of 0 GB/s.
    * Rate matching — the per-request WALL-TIME ratio (decode time per request)
      / (prefill time per request). A first-order sizing signal for how
      decode-heavy the pipeline is; the true balanced decode:prefill replica
      ratio is this DIVIDED BY each deployment's request-level batch factor
      (decode serves many concurrent requests per step), so this is an upper
      bound on the replica ratio, not the ratio itself."""
    dc_kv = decode_result.get("kv_cache", {}) or {}
    per_req_kv = float(dc_kv.get("per_request_kv_bytes", 0.0) or 0.0)
    link = (HandoffLink("stated", float(kv_transfer_bw_gbps))
            if kv_transfer_bw_gbps and kv_transfer_bw_gbps > 0
            else HandoffLink.between(prefill_result, decode_result))
    bw_bytes = gbps_to_bytes_per_sec(link.bw_gbps)
    handoff_s = (per_req_kv / bw_bytes) if bw_bytes else None

    # A phase may decline to publish a step time — `t_sol_s is None` under
    # pipeline parallelism, where the rows are one stage's and the model's step
    # is every stage in sequence plus a bubble nobody modelled. `or 0.0` would
    # turn that refusal into `prefill_time_per_request_s: 0.0` and a rate match
    # of infinity; the disaggregation answer is simply absent instead, with the
    # phase's own reason carried alongside it.
    pf_ph = (prefill_result.get("phases", {}) or {}).get("prefill", {}) or {}
    dc_ph = (decode_result.get("phases", {}) or {}).get("decode", {}) or {}
    pf_raw, dc_raw = pf_ph.get("t_sol_s"), dc_ph.get("t_sol_s")
    unavailable = " · ".join(
        f"{name}: {r}" for name, r in
        (("prefill", pf_ph.get("time_unpublished_reason")),
         ("decode", dc_ph.get("time_unpublished_reason"))) if r)
    pf_t = float(pf_raw) if pf_raw is not None else None
    dc_t = float(dc_raw) if dc_raw is not None else None
    decode_time_per_req = (int(avg_output_tokens or 0) * dc_t
                           if dc_t is not None else None)
    ratio = (decode_time_per_req / pf_t
             if decode_time_per_req is not None and pf_t else None)
    return {
        "kv_transfer": {
            "per_request_kv_bytes": per_req_kv,
            # Which link, and at what bandwidth: null with the reason
            # beside it where the card states none, never 0 GB/s.
            "link": link.name,
            "link_bw_gbps": link.bw_gbps,
            "handoff_latency_s": handoff_s,
            "unavailable_reason": link.unavailable_reason,
        },
        "rate_match": {
            "prefill_time_per_request_s": pf_t,
            "decode_time_per_request_s": decode_time_per_req,
            # Per-request wall-time ratio — an UPPER bound on the balanced
            # decode:prefill replica ratio (batch parallelism lowers the true
            # ratio; see docstring). Named for what it is, not overclaimed.
            "decode_to_prefill_time_ratio": ratio,
            # Non-empty exactly when a phase above is null, and says which one
            # and why — so a missing rate match reads as withheld, not as zero.
            "unavailable_reason": unavailable,
        },
    }
