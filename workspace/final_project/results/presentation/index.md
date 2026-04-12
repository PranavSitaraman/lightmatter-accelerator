# Presentation Package

This folder contains the condensed final-project deliverables for slides or a short report.

## Files

- `executive_summary.csv`: headline metrics and interpretations.
- `01_cluster_reductions.png`: 16-node optical-crossbar gains vs electrical mesh.
- `02_tpu_mapping_reductions.png`: TPU-cube mapping-aware gains after fixed-point re-optimization.
- `03_topology_diagram.png`: simplified topology comparison diagram.
- `optimal_mapping_choices.csv`: condensed view of which communication mapping each interconnect makes optimal.

## Headline Metrics

- **16-node AccelForge cluster / Best energy reduction**: 72.21 % for `decode_mlp / shared_dram_multicast`. Optical crossbar vs electrical 4x4 mesh.
- **16-node AccelForge cluster / Best latency reduction**: 97.57 % for `kv_attention / shared_dram_multicast`. Network-aware fixed-point mapping result.
- **TPU cube pure network / Best energy reduction**: 99.91 % for `Photonic switched fabric (M1000) / shared_weight_multicast`. Large traffic-case model vs TPU v5p electrical torus.
- **TPU cube AccelForge mapping / Best energy reduction**: 77.70 % for `TPU cube M1000 switched fabric / kv_attention / shared_dram_multicast`. Includes fixed-point remapping after link-cost feedback.
- **TPU cube AccelForge mapping / Photonic cases with new mapping signature**: 0.00 of 24 for `photonic variants vs electrical torus`. Counts final mappings whose compact signature differs.
- **Distributed inference mapping / Architectures with changed optimal mapping**: 5.00 of 16 for `prefill_projection / Lightmatter Optical Crossbar / Lightmatter EVK100 -> shared_dram_multicast`. Topology/link changes can make multicast preferable to node-to-node exchange.

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
