# Final Submission Evidence Package

This folder intentionally contains the small set of assets used by the final report, poster, technical summary, and presentation.
Raw CSVs in the parent `results/` directory remain the reproducibility trail.

## Storyline

1. **16-node AccelForge feedback loop.** Does a logical-memory network level change energy and latency? Yes: optical multicast reduces cluster energy by ~72% and latency by ~98%.
2. **Distance-aware link model.** Is pJ/bit constant with scale? No: electrical/chiplet links grow with distance; photonics is conversion dominated.
3. **TPU cube mapping and multicast.** Is the gain just bandwidth? No: shared-state multicast collapses N unicasts into one broadcast.
4. **Mapping vs distributed schedule.** Can the AccelForge mapping stay stable while compiler choices change? Yes: 0/24 inner signatures change, but 5/16 distributed choices change.
5. **Strong/weak scaling and OCS/OPS.** What improves with larger systems and switching semantics? Photonic multicast scales flatter; OPS mainly helps fine-grained phases.
6. **9216-chip Gemma/Gemini proxy.** Does the stack matter end to end? Inference improves far more than training because training becomes compute-bound.

## Files

- `final_assumptions.csv`: compact table of model constants and assumptions.
- `final_results.csv`: headline metrics used in the submission.
- `final_evidence_chain.csv`: one-row-per-experiment narrative map.
- `01_evidence_chain.png`: the full experiment spine.
- `02_distance_and_scaling.png`: distance physics plus strong/weak scaling.
- `03_mapping_and_schedule.png`: inner-mapping invariance vs. distributed-schedule changes.
- `04_multicast_and_chiplet.png`: multicast and advanced-packaging comparison.
- `05_superpod_end_to_end.png`: full-superpod speedups and bottleneck shares.
- `06_switching_and_limits.png`: switching-mode sensitivity and residual bottlenecks.
- `07_cluster_workload_heatmap.png`: 16-node energy/latency reduction heatmap.
- `08_technology_catalog.png`: per-link bandwidth and energy across the link technologies.
- `09_convergence_traces.png`: iterative network-aware feedback loop convergence.
- `10_node_count_scaling.png`: how the photonic advantage scales with node count.
- `node_scaling_sweep.csv`: numerical data for the node-scaling figure.

## Class Concept Translation

- **Einsum.** Each LLM stage is represented as a parameterized GEMM proxy.
- **Mapping and dataflow.** Loop order, tile shape, and tensor residency are chosen at the per-chip level.
- **Dataplacement.** The logical cluster-memory residency of each tensor determines whether it becomes remote traffic.
- **Memory hierarchy.** A logical cluster-memory level sits above per-chip HBM, unified buffer, register, and compute units.
- **Data movement.** Mapper action counts are converted to bits and priced by topology distance, contention, and multicast semantics.
- **Emerging technology.** Photonics is evaluated through energy, latency, topology, scaling, and compiler-visible scheduling.
