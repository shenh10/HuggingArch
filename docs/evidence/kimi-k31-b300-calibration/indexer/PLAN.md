Independent paged indexer calibration, declared before measuring

GPU7 only; GPU0 and host CPUs may be running the parent pytest suite. The ordinary kernel_bench CUDA-event L2-drain protocol measures only DeepGEMM paged logits, including output drain as specified by the repository. Untimed quantization, cache packing, and scheduling metadata. This is not a CUPTI conversion and is not communication timing.

Training: canonical key-derived input seed, exact six local context lengths 256,257,288,2048,2049,2080, batch1/query1/32heads/128width/page64, TP8 DCP8 EP2 Kimi planner. Holdout: independently regenerate each identical physical shape with seed XOR2026111302, separate process and CSV. Only training rows enter a calibration overlay. This is independent input/run stability validation at measured shapes, NOT unseen-shape generalization or full-model heldout prediction. No global multiplier or interpolation.

Trace local contexts256/257 and2048/2049, serving-midpoint288 and2080 are intentionally all represented. Real kernel output must pass the normal FP8 reference cross-check and emitted rows must pass the normal CSV validator. Raw timing method/provenance preserved; profiling observed warm cache and distributed contention are different regimes.

Planner fallback before any timing: DCP8 graph raises comm6 anchor validation error (dcp-planner-failure.log). Generate the identical local kernel shape using TP8 DCP1 local context; this does not represent distributed execution. Match physical keys against repaired DCP8 planner before claiming overlay coverage.
