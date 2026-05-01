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
- `03_mapping_and_schedule.png`: why AccelForge mappings can stay stable while compiler schedules change.
- `04_multicast_and_chiplet.png`: multicast and advanced-packaging comparison.
- `05_superpod_end_to_end.png`: Gemma/Gemini full-superpod speedups and bottlenecks.
- `06_switching_and_limits.png`: OCS/OPS sensitivity and model scope.

## Class Concept Translation

- **Einsum:** each LLM stage is represented by a parameterized GEMM proxy.
- **Mapping/dataflow:** AccelForge chooses storage, temporal, spatial, and compute loop structure.
- **Dataplacement:** the logical-memory residency of a tensor determines whether it becomes remote traffic.
- **Memory hierarchy:** `LogicalClusterMemory` sits above per-chip HBM/UB/register/MAC.
- **Data movement:** mapper action counts become bits and are priced by topology distance, contention, and multicast semantics.
- **Emerging technology:** photonics is evaluated through energy, latency, topology, scaling, and compiler-visible scheduling.
