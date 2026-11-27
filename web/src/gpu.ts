/** One row of `/inference/gpus` (backend/inference/gpu_specs.csv). */
export type GpuItem = {
  sku?: string;
  name?: string;
  id?: string;
  arch?: string;
  bf16_tflops?: number;
  /** HBM capacity in GiB (2^30 bytes). */
  mem_gib?: number;
  /** HBM bandwidth, GB/s. */
  mem_bw_gbps?: number;
  /** Per-direction link bandwidths, GB/s. */
  scaleup_gbps?: number;
  /** The scale-up link's name (NVLink, HCCS, UB); set with `scaleup_gbps`. */
  scaleup_fabric?: string;
  pcie_gbps?: number;
  rdma_gbps?: number;
  rdma_hop_latency_us?: number;
  gpus_per_node?: number;
};

export function skuOf(g: GpuItem): string {
  return g.sku || g.name || g.id || "";
}

/** The intra-node link: the SKU's scale-up link when it has one, else PCIe. */
export function intraNodeLink(g: GpuItem): string {
  if (g.scaleup_gbps) return `${g.scaleup_fabric} ${g.scaleup_gbps} GB/s`;
  if (g.pcie_gbps) return `PCIe ${g.pcie_gbps} GB/s`;
  return "—";
}

function hbmBandwidth(gbps: number): string {
  return gbps >= 1000 ? `${+(gbps / 1000).toFixed(2)} TB/s` : `${gbps} GB/s`;
}

/** BF16 dense peak · HBM capacity · HBM bandwidth · intra-node link. */
export function gpuSpecSummary(g: GpuItem): string {
  return [
    g.bf16_tflops ? `BF16 ${g.bf16_tflops} TFLOPS` : null,
    g.mem_gib ? `${g.mem_gib} GiB` : null,
    g.mem_bw_gbps ? `HBM ${hbmBandwidth(g.mem_bw_gbps)}` : null,
    intraNodeLink(g),
  ].filter(Boolean).join(" · ");
}

/** A GPU picker's option text: the SKU followed by its headline specs. */
export function gpuOptionLabel(g: GpuItem): string {
  const sku = skuOf(g);
  const specs = gpuSpecSummary(g);
  return specs ? `${sku} — ${specs}` : sku;
}

/** A picker's options; a current SKU the list lacks is kept, as a bare SKU. */
export function gpuOptions(gpus: GpuItem[], current?: string): { value: string; label: string }[] {
  const opts = gpus.map((g) => ({ value: skuOf(g), label: gpuOptionLabel(g) }));
  return current && !opts.some((o) => o.value === current)
    ? [{ value: current, label: current }, ...opts]
    : opts;
}
