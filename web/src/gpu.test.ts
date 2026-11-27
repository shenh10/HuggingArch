// How a GPU row names itself and how a picker lists it. The identity rule
// (which field wins) is the one a refactor can quietly reverse: the SKU is
// what every request and cache key carries.
import { describe, expect, test } from "vitest";

import { gpuOptionLabel, gpuOptions, gpuSpecSummary, intraNodeLink, skuOf } from "./gpu";

describe("skuOf", () => {
  test("sku wins over name, name over id, and nothing is the empty string", () => {
    expect(skuOf({ sku: "H100-SXM", name: "H100", id: "h100" })).toBe("H100-SXM");
    expect(skuOf({ name: "H100", id: "h100" })).toBe("H100");
    expect(skuOf({ id: "h100" })).toBe("h100");
    expect(skuOf({})).toBe("");
  });
});

describe("intraNodeLink", () => {
  test("the scale-up link when the SKU has one, PCIe otherwise, a dash for neither", () => {
    expect(intraNodeLink({ scaleup_gbps: 450, scaleup_fabric: "NVLink", pcie_gbps: 64 })).toBe("NVLink 450 GB/s");
    expect(intraNodeLink({ scaleup_gbps: 196, scaleup_fabric: "HCCS", pcie_gbps: 64 })).toBe("HCCS 196 GB/s");
    expect(intraNodeLink({ pcie_gbps: 64 })).toBe("PCIe 64 GB/s");
    expect(intraNodeLink({})).toBe("—");
  });
});

describe("gpuSpecSummary", () => {
  test("HBM bandwidth reads in TB/s from 1000 GB/s up", () => {
    expect(gpuSpecSummary({ mem_bw_gbps: 3350 })).toBe("HBM 3.35 TB/s · —");
    expect(gpuSpecSummary({ mem_bw_gbps: 1000 })).toBe("HBM 1 TB/s · —");
    expect(gpuSpecSummary({ mem_bw_gbps: 999 })).toBe("HBM 999 GB/s · —");
  });

  test("a spec the row does not carry is left out, not printed empty", () => {
    expect(gpuSpecSummary({ bf16_tflops: 989, mem_gib: 80, mem_bw_gbps: 3350, scaleup_gbps: 450, scaleup_fabric: "NVLink" }))
      .toBe("BF16 989 TFLOPS · 80 GiB · HBM 3.35 TB/s · NVLink 450 GB/s");
    expect(gpuSpecSummary({ mem_gib: 80 })).toBe("80 GiB · —");
  });
});

describe("gpuOptions", () => {
  test("a picker lists each SKU with its specs", () => {
    expect(gpuOptionLabel({ sku: "H100-SXM", mem_gib: 80 })).toBe("H100-SXM — 80 GiB · —");
    expect(gpuOptions([{ sku: "H100-SXM", mem_gib: 80 }]))
      .toEqual([{ value: "H100-SXM", label: "H100-SXM — 80 GiB · —" }]);
  });

  test("a current SKU the list lacks is kept at the front, bare", () => {
    const gpus = [{ sku: "H100-SXM" }, { sku: "A100" }];
    expect(gpuOptions(gpus, "MI300X").map((o) => o.value)).toEqual(["MI300X", "H100-SXM", "A100"]);
    // And one the list does have is not listed twice.
    expect(gpuOptions(gpus, "A100").map((o) => o.value)).toEqual(["H100-SXM", "A100"]);
    expect(gpuOptions(gpus).map((o) => o.value)).toEqual(["H100-SXM", "A100"]);
  });
});
