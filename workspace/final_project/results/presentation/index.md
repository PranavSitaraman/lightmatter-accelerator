# Presentation Package

This folder contains the condensed final-project deliverables for slides or a short report.

## Files

- `executive_summary.csv`: headline metrics and interpretations.
- `01_cluster_reductions.png`: 16-node optical-crossbar gains vs electrical mesh.
- `02_tpu_mapping_reductions.png`: TPU-cube mapping-aware gains after fixed-point re-optimization.
- `03_topology_diagram.png`: simplified topology comparison diagram.
- `04_compiler_strategy_choices.png`: XLA/Pallas-style dataflow choices enabled by photonic multicast.
- `05_scaling_laws.png`: 64-chip to 9216-chip scaling laws.
- `06_chiplet_baseline.png`: photonic comparison against the 3.5 Tbps / 6 pJ/bit chiplet baseline.
- `07_superpod_end_to_end_speedups.png`: full-superpod Gemini-class training/inference speedups.
- `08_superpod_stage_breakdown.png`: where latency remains after full-stack photonic optimization.
- `09_superpod_context_scaling.png`: long-context inference scaling for electrical/chiplet/photonic fabrics.
- `optimal_mapping_choices.csv`: condensed view of which communication mapping each interconnect makes optimal.

## Headline Metrics

- **16-node AccelForge cluster / Best energy reduction**: 72.21 % for `decode_mlp / shared_dram_multicast`. Optical crossbar vs electrical 4x4 mesh.
- **16-node AccelForge cluster / Best latency reduction**: 97.57 % for `kv_attention / shared_dram_multicast`. Network-aware fixed-point mapping result.
- **TPU cube pure network / Best energy reduction**: 99.91 % for `Photonic switched fabric (M1000) / shared_weight_multicast`. Large traffic-case model vs TPU v5p electrical torus.
- **TPU cube AccelForge mapping / Best energy reduction**: 77.70 % for `TPU cube M1000 switched fabric / kv_attention / shared_dram_multicast`. Includes fixed-point remapping after link-cost feedback.
- **TPU cube AccelForge mapping / Photonic cases with new mapping signature**: 0.00 of 24 for `photonic variants vs electrical torus`. Counts final mappings whose compact signature differs.
- **Distributed inference mapping / Architectures with changed optimal mapping**: 5.00 of 16 for `prefill_projection / Lightmatter Optical Crossbar / Lightmatter EVK100 -> shared_dram_multicast`. Topology/link changes can make multicast preferable to node-to-node exchange.
- **Compiler/dataflow paradigm / Best changed strategy**: 67.75 % for `kv_attention / M1000 optical switched fabric -> Remote pooled-HBM streaming`. Models XLA/Pallas-visible cache, sharding, and multicast choices.
- **Scaling law / Best multicast energy exponent**: 0.00 log-log slope for `M1000 optical switched fabric`. Lower is better when scaling from 64 chips to superpod scale.
- **Advanced chiplet comparison / Best energy reduction vs chiplet**: 99.85 % for `M1000 optical switched fabric / shared_state_multicast`. Compares against 3.5 Tbps, 6 pJ/bit advanced packaging.
- **Full superpod end-to-end / Best latency speedup**: 14.50 x for `Gemini 3 Deep Think long-context proxy / inference`. 9216-chip Ironwood/Gemini-class proxy with full photonic compiler stack.
- **Full superpod end-to-end / Best energy reduction**: 51.07 % for `Gemini 3 Deep Think long-context proxy / inference`. End-to-end compute + HBM + network energy, including compiler/dataflow effects.
- **Superpod ablation / Largest final-step increment**: 1.06 x vs previous for `Gemini 3 Deep Think long-context proxy / inference`. Incremental effect after topology, multicast, remote HBM, and MoE-locality ablations.

## Optimal Distributed Mapping Choices

- `decode_mlp` / `Electrical 4x4 Mesh / Electrical 800G/dir` chooses `node_to_node` (same as baseline).
- `decode_mlp` / `Lightmatter Optical Crossbar / Lightmatter EVK100` chooses `node_to_node` (same as baseline).
- `kv_attention` / `Electrical 4x4 Mesh / Electrical 800G/dir` chooses `node_to_node` (same as baseline).
- `kv_attention` / `Lightmatter Optical Crossbar / Lightmatter EVK100` chooses `node_to_node` (same as baseline).
- `prefill_projection` / `Electrical 4x4 Mesh / Electrical 800G/dir` chooses `node_to_node` (same as baseline).
- `prefill_projection` / `Lightmatter Optical Crossbar / Lightmatter EVK100` chooses `shared_dram_multicast` (changed).
- `kv_attention` / `TPU cube EVK100 photonic torus` chooses `node_to_node` (same as baseline).
- `kv_attention` / `TPU cube L200 switched fabric` chooses `node_to_node` (same as baseline).
- `kv_attention` / `TPU cube L200X switched fabric` chooses `node_to_node` (same as baseline).
- `kv_attention` / `TPU cube M1000 switched fabric` chooses `node_to_node` (same as baseline).
- `kv_attention` / `TPU cube electrical torus` chooses `node_to_node` (same as baseline).
- `prefill_projection` / `TPU cube EVK100 photonic torus` chooses `shared_dram_multicast` (changed).
- `prefill_projection` / `TPU cube L200 switched fabric` chooses `shared_dram_multicast` (changed).
- `prefill_projection` / `TPU cube L200X switched fabric` chooses `shared_dram_multicast` (changed).
- `prefill_projection` / `TPU cube M1000 switched fabric` chooses `shared_dram_multicast` (changed).
- `prefill_projection` / `TPU cube electrical torus` chooses `node_to_node` (same as baseline).
