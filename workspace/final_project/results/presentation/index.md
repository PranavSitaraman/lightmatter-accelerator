# Presentation Evidence Notes

These notes summarize the evidence used in the final project presentation and report.

## Evidence Included

- Headline metrics: the most important energy, latency, scaling, and end-to-end results.
- Cluster study: 16-node electrical mesh versus optical-crossbar behavior.
- TPU cube study: mapping-aware ICI replacement for a 64-chip cube.
- Topology study: visual comparison of mesh, torus, photonic torus, switched optical fabric, and chiplet baselines.
- Compiler/dataflow study: XLA/GSPMD-style collectives versus remote HBM and photonic multicast cache-fill strategies.
- Scaling study: 64-chip to 9216-chip per-chip scaling laws.
- Chiplet comparison: calibration against the 3.5 Tbps / 6 pJ/bit advanced-packaging baseline.
- SuperPod study: 9216-chip Gemma/Gemini inference/training speedups, stage breakdowns, ablations, and context scaling.
- Unified end-to-end study: distance-parametrized pJ/bit, strong/weak scaling, and OCS-vs-OPS switching sensitivities.

## Headline Metrics

- **16-node AccelForge cluster / Best energy reduction**: 71.95 % for decode MLP / shared-DRAM multicast. Optical crossbar vs electrical 4x4 mesh.
- **16-node AccelForge cluster / Best latency reduction**: 97.57 % for decode MLP / shared-DRAM multicast. Network-aware fixed-point mapping result.
- **TPU cube pure network / Best energy reduction**: 99.93 % for Photonic torus (EVK100) / shared-weight multicast. Large traffic-case model vs TPU v5p electrical torus.
- **TPU cube AccelForge mapping / Best energy reduction**: 77.45 % for TPU cube EVK100 photonic torus / KV attention / shared-DRAM multicast. Includes fixed-point remapping after link-cost feedback.
- **TPU cube AccelForge mapping / Photonic cases with new mapping signature**: 0.00 of 24 for photonic variants vs electrical torus. Counts final mappings whose compact signature differs.
- **Distributed inference mapping / Architectures with changed optimal mapping**: 5.00 of 16 for prefill projection / Lightmatter Optical Crossbar / Lightmatter EVK100 to shared-DRAM multicast. Topology/link changes can make multicast preferable to node-to-node exchange.
- **Compiler/dataflow paradigm / Best changed strategy**: 73.86 % for KV attention / EVK100 photonic torus to Remote pooled-HBM streaming. Models XLA/Pallas-visible cache, sharding, and multicast choices.
- **Scaling law / Best multicast energy exponent**: 0.00 log-log slope for EVK100 photonic torus. Lower is better when scaling from 64 chips to superpod scale.
- **Advanced chiplet comparison / Best energy reduction vs chiplet**: 99.88 % for EVK100 photonic torus / shared-state multicast. Compares against 3.5 Tbps, 6 pJ/bit advanced packaging.
- **Full superpod end-to-end / Best latency speedup**: 42.43 x for Gemma 4 26B MoE proxy / inference. 9216-chip Ironwood Gemma/Gemini proxy with full photonic compiler stack.
- **Full superpod end-to-end / Best energy reduction**: 94.05 % for Gemma 4 26B MoE proxy / inference. End-to-end compute + HBM + network energy, including compiler/dataflow effects.
- **Superpod ablation / Largest final-step increment**: 1.07 x vs previous for Gemma 4 26B MoE proxy / inference. Incremental effect after topology, multicast, remote HBM, and MoE-locality ablations.

## Modeling Notes

- Topology is not modeled as a pure bandwidth problem: the network model includes hop count, contention, per-hop router latency, and multicast semantics.
- Link energy is no longer only a constant pJ/bit. Electrical links have low conversion cost and distance-dependent transmission energy; photonic links are conversion-dominated with almost distance-independent transmission.
- The AccelForge loop feeds network read/write energy and latency back into the logical memory level and reruns until convergence, so the mapper can react to in-flight data cost rather than seeing a free global memory.
- The full-superpod stage model converts raw transport time into exposed stall by multiplying raw network latency by the non-overlapped fraction; this is how optical IC affects both transport time and stall time.
- The analytic model does not include packet-level queueing or OCS reconfiguration timing.

## TPU-Speak Translation

- Node-to-node exchange: GSPMD collective-permute / sequence- or tensor-sharded activation exchange.
- Node-to-DRAM fetch: remote HBM / remote KV / remote optimizer-state fetch.
- Shared-DRAM multicast: broadcast cache fill / replicated hot-state refill.
- XLA/GSPMD collective: baseline sharding and collectives.
- Remote-HBM streaming: pooled-HBM streaming schedule.
- Photonic multicast cache: optical broadcast prefetch to many local caches.
- MoE dispatch / locality routing: expert-parallel token routing and placement.

## Optimal Distributed Mapping Choices

- decode MLP / Electrical 4x4 Mesh / Electrical 800G/dir chooses node-to-node exchange (same as baseline).
- decode MLP / Lightmatter Optical Crossbar / Lightmatter EVK100 chooses node-to-node exchange (same as baseline).
- KV attention / Electrical 4x4 Mesh / Electrical 800G/dir chooses node-to-node exchange (same as baseline).
- KV attention / Lightmatter Optical Crossbar / Lightmatter EVK100 chooses node-to-node exchange (same as baseline).
- prefill projection / Electrical 4x4 Mesh / Electrical 800G/dir chooses node-to-node exchange (same as baseline).
- prefill projection / Lightmatter Optical Crossbar / Lightmatter EVK100 chooses shared-DRAM multicast (changed).
- KV attention / TPU cube EVK100 photonic torus chooses node-to-node exchange (same as baseline).
- KV attention / TPU cube L200 switched fabric chooses node-to-node exchange (same as baseline).
- KV attention / TPU cube L200X switched fabric chooses node-to-node exchange (same as baseline).
- KV attention / TPU cube M1000 switched fabric chooses node-to-node exchange (same as baseline).
- KV attention / TPU cube electrical torus chooses node-to-node exchange (same as baseline).
- prefill projection / TPU cube EVK100 photonic torus chooses shared-DRAM multicast (changed).
- prefill projection / TPU cube L200 switched fabric chooses shared-DRAM multicast (changed).
- prefill projection / TPU cube L200X switched fabric chooses shared-DRAM multicast (changed).
- prefill projection / TPU cube M1000 switched fabric chooses shared-DRAM multicast (changed).
- prefill projection / TPU cube electrical torus chooses node-to-node exchange (same as baseline).
