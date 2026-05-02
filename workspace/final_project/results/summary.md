# Lightmatter Final Project — Summary

Summary of the AccelForge + photonic-interconnect architecture study.

## Executive Takeaways

- The raw network model and the AccelForge fixed-point mapper both show the same direction: optical links help most when traffic leaves a single local HBM bank.
- Re-running the mapper is explicitly measured: the study reports both fixed-point tile/residency sensitivity and the higher-level distributed mapping choice that becomes optimal under each interconnect.
- A separate compiler/dataflow strategy model shows the more important effect: photonic multicast can make software-managed cache fills and shared-state broadcast preferable to classic SPMD activation exchange.
- Scaling-law sweeps show that flat switched photonic fabrics keep hop count and contention nearly constant while torus and chiplet fabrics accumulate hop/contention cost as cubes grow toward superpod scale.
- The full-superpod study models public Gemma 4 open-model proxies plus Gemini-class stress proxies on Ironwood/TPU7x and reports end-to-end inference/training speedups by stage and by optimization ablation.
- The latest unified study replaces flat pJ/bit constants with conversion-plus-distance physics, and compares strong scaling, weak scaling, optical circuit switching, and optical packet switching in one end-to-end chain.
- The TPU cube study separates two design points: a conservative EVK100 photonic torus and higher-bandwidth switched-fabric options (L200/L200X/M1000).
- EVK100 public collateral includes both a 1.9 pJ/bit Passage-family value and a more conservative 3.2 pJ/bit EVK100 technical-spec value; this project uses 1.9 pJ/bit as the main optimistic/reference point and includes sensitivity hooks for alternatives.

## Technology Catalog

| Technology | BW (one-way) | Energy (pJ/bit) | Multicast |
|---|---|---|---|
| Electrical 800G | 800 Gbps | 10.0 | No |
| Advanced chiplet 3.5T | 3.5 Tbps | 6.0 | No |
| Lightmatter EVK50 | 0.8 Tbps | 4.6 | Yes |
| Lightmatter EVK100 | 3.2 Tbps | 1.9 | Yes |
| Lightmatter L200 | 16.0 Tbps | 4.5 | Yes |
| Lightmatter L200X | 32.0 Tbps | 4.8 | Yes |
| Lightmatter M1000 | 57.3 Tbps | 2.3 | Yes |

## Modeling Notes

- The topology model is **not** a pure bandwidth multiplier. Network energy and latency depend on topology-specific hop count, contention-driven bandwidth derating, per-hop router latency, and whether the fabric supports true multicast.
- In the AccelForge fixed-point loop, network read/write energy and latency are folded back into the logical cluster-memory level and the mapper is rerun until the logical-memory costs stabilize.
- In the full-superpod model, only exposed network time stalls execution: `T_stage = max(T_compute, T_HBM) + (1 - overlap) * T_network`. Optical configurations therefore help in two ways: they reduce raw network time and they enable more communication/computation overlap through better data placement and cache-fill strategies.
- The model is intentionally analytic rather than packet-level: it does not include VC arbitration, queueing transients, or explicit OCS reconfiguration timing.

## TPU-Speak Translation

| Study term | TPU/XLA analogy | Meaning in this project |
|---|---|---|
| Node-to-node exchange | GSPMD collective-permute / sequence- or tensor-sharded activation exchange | Activations/outputs stay sharded and are exchanged directly among chips. |
| Node-to-DRAM fetch | Remote HBM / remote KV / remote optimizer-state fetch | A chip reads model state from a remote memory tier instead of relying only on local residency. |
| Shared-DRAM multicast | Broadcast cache fill / replicated hot-state refill | One logical weight/KV/expert payload fans out to many chips; optics can charge only the unique payload. |
| XLA/GSPMD collective | Baseline XLA/GSPMD sharding plan | Classic sharded tensors plus collectives over the mesh/torus. |
| Remote-HBM streaming | Pooled-HBM streaming schedule | Compiler prefers remote reads over local replication when the network is cheap enough. |
| Photonic multicast cache | Optical broadcast prefetch | Compiler fills many local caches with one multicast transfer before compute. |
| MoE dispatch / locality routing | Expert-parallel token routing | Expert dispatch/combine traffic and expert placement choices for MoE models. |
| Gradient synchronization | Reduce-scatter / all-reduce | Training collective over active parameters. |

## Cluster Study Highlights (16-node system)

### node-to-DRAM fetch

- Best energy reduction: decode MLP → 67.69%
- Best latency reduction: decode MLP → 91.83%

### node-to-node exchange

- Best energy reduction: prefill projection → 9.30%
- Best latency reduction: decode MLP → 0.00%

### shared-DRAM multicast

- Best energy reduction: decode MLP → 71.95%
- Best latency reduction: decode MLP → 97.57%

## TPU Cube Highlights (64-node 4×4×4 cube)

### Photonic switched fabric (L200)

- Best energy reduction: shared-weight multicast → 99.82%
- Best latency reduction: shared-weight multicast → 99.98%

### Photonic switched fabric (L200X)

- Best energy reduction: shared-weight multicast → 99.81%
- Best latency reduction: shared-weight multicast → 99.99%

### Photonic switched fabric (M1000)

- Best energy reduction: shared-weight multicast → 99.91%
- Best latency reduction: shared-weight multicast → 99.99%

### Photonic torus (EVK100)

- Best energy reduction: shared-weight multicast → 99.93%
- Best latency reduction: shared-weight multicast → 99.61%

## TPU Cube Mapping Adaptation (AccelForge fixed point)

- Best mapping-aware TPU-cube energy reduction: TPU cube EVK100 photonic torus on KV attention / shared-DRAM multicast → 77.45%.
- Mapping signatures differed from the electrical baseline in 0 of 24 photonic mapping cases.
- Representative mapping interpretation: the local GEMM tile stays stable, but remote/shared state becomes cheaper at the logical-memory level; the larger placement decision is where the photonic advantage appears.

## Optimal Distributed Mapping Choices

Each row treats the traffic scenario as a distributed inference mapping choice: node-to-node activation exchange, node-to-DRAM remote fetch, or shared-DRAM multicast.

| Workload | Architecture | Optimal mapping | Changed vs baseline |
|---|---|---|---|
| decode MLP | Electrical 4x4 Mesh / Electrical 800G/dir | node-to-node exchange | False |
| decode MLP | Lightmatter Optical Crossbar / Lightmatter EVK100 | node-to-node exchange | False |
| KV attention | Electrical 4x4 Mesh / Electrical 800G/dir | node-to-node exchange | False |
| KV attention | Lightmatter Optical Crossbar / Lightmatter EVK100 | node-to-node exchange | False |
| prefill projection | Electrical 4x4 Mesh / Electrical 800G/dir | node-to-node exchange | False |
| prefill projection | Lightmatter Optical Crossbar / Lightmatter EVK100 | shared-DRAM multicast | True |
| KV attention | TPU cube EVK100 photonic torus | node-to-node exchange | False |
| KV attention | TPU cube L200 switched fabric | node-to-node exchange | False |
| KV attention | TPU cube L200X switched fabric | node-to-node exchange | False |
| KV attention | TPU cube M1000 switched fabric | node-to-node exchange | False |
| KV attention | TPU cube electrical torus | node-to-node exchange | False |
| prefill projection | TPU cube EVK100 photonic torus | shared-DRAM multicast | True |
| prefill projection | TPU cube L200 switched fabric | shared-DRAM multicast | True |
| prefill projection | TPU cube L200X switched fabric | shared-DRAM multicast | True |
| prefill projection | TPU cube M1000 switched fabric | shared-DRAM multicast | True |
| prefill projection | TPU cube electrical torus | node-to-node exchange | False |

## Compiler/Dataflow Strategy Choices

This layer models XLA/Pallas-visible choices above the inner AccelForge tile mapper.

| Workload | Architecture | Optimal strategy | Changed vs TPU electrical | Energy gain vs baseline |
|---|---|---|---|---|
| decode MLP | Advanced chiplet torus | XLA SPMD collective | False | 0.05% |
| decode MLP | EVK100 photonic torus | Remote pooled-HBM streaming | True | 73.57% |
| decode MLP | M1000 optical packet fabric | Remote pooled-HBM streaming | True | 60.75% |
| decode MLP | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 67.42% |
| decode MLP | TPU electrical torus | XLA SPMD collective | False | 0.00% |
| KV attention | Advanced chiplet torus | XLA SPMD collective | False | 1.30% |
| KV attention | EVK100 photonic torus | Remote pooled-HBM streaming | True | 73.86% |
| KV attention | M1000 optical packet fabric | Remote pooled-HBM streaming | True | 61.54% |
| KV attention | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 67.94% |
| KV attention | TPU electrical torus | XLA SPMD collective | False | 0.00% |
| prefill projection | Advanced chiplet torus | Remote pooled-HBM streaming | True | 20.41% |
| prefill projection | EVK100 photonic torus | Remote pooled-HBM streaming | True | 53.70% |
| prefill projection | M1000 optical packet fabric | Remote pooled-HBM streaming | True | 52.33% |
| prefill projection | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 53.04% |
| prefill projection | TPU electrical torus | XLA SPMD collective | False | 0.00% |

## Scaling-Law Exponents

Lower exponents mean better scaling as the system grows from one 64-chip cube toward 9216 chips.

| Architecture | Scenario | Metric | Exponent |
|---|---|---|---|
| Advanced chiplet torus | all_to_all_activation | energy per chip | 0.669 |
| Advanced chiplet torus | all_to_all_activation | latency per chip | 0.333 |
| Advanced chiplet torus | shared-state multicast | energy per chip | 1.627 |
| Advanced chiplet torus | shared-state multicast | latency per chip | 1.333 |
| EVK100 photonic torus | all_to_all_activation | energy per chip | 0.002 |
| EVK100 photonic torus | all_to_all_activation | latency per chip | 0.333 |
| EVK100 photonic torus | shared-state multicast | energy per chip | 0.002 |
| EVK100 photonic torus | shared-state multicast | latency per chip | 0.333 |
| M1000 optical switched fabric | all_to_all_activation | energy per chip | 0.005 |
| M1000 optical switched fabric | all_to_all_activation | latency per chip | 0.000 |
| M1000 optical switched fabric | shared-state multicast | energy per chip | 0.005 |
| M1000 optical switched fabric | shared-state multicast | latency per chip | 0.000 |
| TPU electrical torus | all_to_all_activation | energy per chip | 0.668 |
| TPU electrical torus | all_to_all_activation | latency per chip | 0.333 |
| TPU electrical torus | shared-state multicast | energy per chip | 1.627 |
| TPU electrical torus | shared-state multicast | latency per chip | 1.333 |

## Advanced Chiplet Baseline

The chiplet baseline uses the requested 3.5 Tbps one-way, 6 pJ/bit per-hop model.

| Scenario | Architecture | Energy reduction vs chiplet | Latency reduction vs chiplet |
|---|---|---|---|
| all_to_all_activation | EVK100 photonic torus | 89.76% | -9.38% |
| all_to_all_activation | M1000 optical switched fabric | 87.36% | 98.47% |
| remote weight refill | EVK100 photonic torus | 92.19% | -9.38% |
| remote weight refill | M1000 optical switched fabric | 90.36% | 98.47% |
| shared-state multicast | EVK100 photonic torus | 99.88% | 98.29% |
| shared-state multicast | M1000 optical switched fabric | 99.85% | 99.98% |

## Full-Superpod Gemma/Gemini End-to-End Study

Gemma 4 open-model proxies use public size/context facts; Gemini stress proxies remain explicit dense/MoE/long-context assumptions because Google has not published Gemini 3/3.1 parameter counts.

| Model | Phase | Configuration | Speedup vs electrical | Energy reduction | Network latency share | Throughput tokens/s |
|---|---|---|---|---|---|---|
| Gemini 3 Deep Think long-context proxy | inference | M1000 full photonic compiler stack | 14.50x | 83.80% | 0.08% | 1.17e+07 |
| Gemini 3 Deep Think long-context proxy | inference | M1000 switched optical topology only | 12.02x | 83.17% | 0.58% | 9.71e+06 |
| Gemini 3 Deep Think long-context proxy | inference | Advanced chiplet torus + XLA SPMD | 3.65x | 33.19% | 67.50% | 2.95e+06 |
| Gemini 3 Deep Think long-context proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 90.67% | 8.07e+05 |
| Gemini 3 Deep Think long-context proxy | training | M1000 full photonic compiler stack | 1.43x | 3.38% | 0.00% | 3.89e+06 |
| Gemini 3 Deep Think long-context proxy | training | M1000 switched optical topology only | 1.19x | 3.37% | 0.00% | 3.24e+06 |
| Gemini 3 Deep Think long-context proxy | training | Advanced chiplet torus + XLA SPMD | 1.10x | 1.34% | 1.23% | 2.97e+06 |
| Gemini 3 Deep Think long-context proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 5.52% | 2.71e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | M1000 full photonic compiler stack | 11.01x | 78.48% | 0.06% | 2.42e+07 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | M1000 switched optical topology only | 9.15x | 77.91% | 0.42% | 2.01e+07 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | Advanced chiplet torus + XLA SPMD | 3.38x | 31.08% | 60.41% | 7.41e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 87.71% | 2.2e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | M1000 full photonic compiler stack | 1.42x | 2.72% | 0.00% | 4.86e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | M1000 switched optical topology only | 1.18x | 2.72% | 0.00% | 4.05e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | Advanced chiplet torus + XLA SPMD | 1.09x | 1.08% | 0.99% | 3.73e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 4.47% | 3.43e+06 |
| Gemma 4 26B MoE proxy | inference | M1000 full photonic compiler stack | 42.43x | 94.05% | 0.26% | 2.19e+08 |
| Gemma 4 26B MoE proxy | inference | M1000 switched optical topology only | 34.07x | 93.35% | 1.80% | 1.76e+08 |
| Gemma 4 26B MoE proxy | inference | Advanced chiplet torus + XLA SPMD | 4.38x | 37.25% | 86.43% | 2.26e+07 |
| Gemma 4 26B MoE proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 96.76% | 5.15e+06 |
| Gemma 4 26B MoE proxy | training | M1000 full photonic compiler stack | 3.37x | 46.41% | 0.03% | 7.29e+08 |
| Gemma 4 26B MoE proxy | training | M1000 switched optical topology only | 2.80x | 46.35% | 0.09% | 6.07e+08 |
| Gemma 4 26B MoE proxy | training | Advanced chiplet torus + XLA SPMD | 1.98x | 18.50% | 24.12% | 4.28e+08 |
| Gemma 4 26B MoE proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 59.78% | 2.17e+08 |
| Gemma 4 31B dense proxy | inference | M1000 full photonic compiler stack | 27.88x | 91.18% | 0.16% | 1.06e+08 |
| Gemma 4 31B dense proxy | inference | M1000 switched optical topology only | 22.73x | 90.49% | 1.17% | 8.63e+07 |
| Gemma 4 31B dense proxy | inference | Advanced chiplet torus + XLA SPMD | 4.15x | 36.11% | 80.58% | 1.58e+07 |
| Gemma 4 31B dense proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 95.10% | 3.79e+06 |
| Gemma 4 31B dense proxy | training | M1000 full photonic compiler stack | 1.66x | 11.92% | 0.00% | 9.6e+07 |
| Gemma 4 31B dense proxy | training | M1000 switched optical topology only | 1.39x | 11.91% | 0.01% | 8e+07 |
| Gemma 4 31B dense proxy | training | Advanced chiplet torus + XLA SPMD | 1.23x | 4.75% | 4.65% | 7.08e+07 |
| Gemma 4 31B dense proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 18.58% | 5.77e+07 |

## Superpod Optimization Ablations

Rows show how much each step contributes relative to the electrical Ironwood baseline.

| Model | Phase | Technique | Speedup vs baseline | Increment vs previous | Network energy share |
|---|---|---|---|---|---|
| Gemini 3.1 Pro-class sparse MoE proxy | inference | baseline | 1.00x | 1.00x | 78.10% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | advanced chiplet | 3.38x | 3.38x | 68.22% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | photonic torus + multicast | 5.43x | 1.61x | 0.39% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | switched topology only | 9.15x | 1.68x | 0.87% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | optical packet switching | 9.44x | 1.03x | 0.39% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | multicast cache | 9.69x | 1.03x | 0.40% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | remote pooled HBM | 10.09x | 1.04x | 0.31% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | MoE locality routing | 10.36x | 1.03x | 0.29% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | full photonic compiler stack | 11.01x | 1.06x | 0.23% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | baseline | 1.00x | 1.00x | 2.72% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | advanced chiplet | 1.09x | 1.09x | 1.66% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | photonic torus + multicast | 1.16x | 1.07x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | switched topology only | 1.18x | 1.02x | 0.01% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | optical packet switching | 1.22x | 1.03x | 0.01% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | multicast cache | 1.25x | 1.03x | 0.01% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | remote pooled HBM | 1.30x | 1.04x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | MoE locality routing | 1.33x | 1.03x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | full photonic compiler stack | 1.42x | 1.06x | 0.00% |

## Distance-Parametrized Electrical vs Photonic Energy

Electrical/chiplet links are modeled as low-conversion, distance-dependent transmission energy. Photonic links are modeled as conversion-dominated with a tiny transmission term.

| Nodes | Scenario | Architecture | Avg distance | Effective pJ/bit | Conversion uJ/chip | Transmission uJ/chip |
|---|---|---|---|---|---|---|
| 64 | shared-state multicast | electrical torus | 20.0 mm | 39.40 | 109951.16 | 21550427.90 |
| 64 | shared-state multicast | advanced chiplet torus | 20.0 mm | 23.70 | 54975.58 | 12974237.21 |
| 64 | shared-state multicast | EVK100 photonic torus | 20.0 mm | 1.85 | 15891.38 | 8.59 |
| 64 | shared-state multicast | M1000 optical circuit switch | 250.0 mm | 2.29 | 19585.05 | 42.95 |
| 64 | shared-state multicast | M1000 optical packet switch | 250.0 mm | 2.76 | 23622.32 | 42.95 |
| 9216 | shared-state multicast | electrical torus | 471.7 mm | 924.80 | 15832967.44 | 73195615619.71 |
| 9216 | shared-state multicast | advanced chiplet torus | 471.7 mm | 556.75 | 7916483.72 | 44066748179.21 |
| 9216 | shared-state multicast | EVK100 photonic torus | 471.7 mm | 1.87 | 15891.38 | 202.61 |
| 9216 | shared-state multicast | M1000 optical circuit switch | 3000.0 mm | 2.34 | 19585.05 | 515.40 |
| 9216 | shared-state multicast | M1000 optical packet switch | 3000.0 mm | 2.81 | 23622.32 | 515.40 |

## Strong and Weak Scaling

Strong scaling keeps the modeled Gemma/Gemini workload fixed while increasing chip count. Weak scaling scales aggregate request/training tokens with chip count so problem size per resource stays approximately fixed.

| Model | Mode | Chips | Phase | Configuration | Speedup | Energy / work unit |
|---|---|---|---|---|---|---|
| Gemma 4 26B MoE proxy | strong | 64 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 1.37e+03 J |
| Gemma 4 26B MoE proxy | strong | 64 | inference | M1000 full photonic compiler stack | 9.16x | 788 J |
| Gemma 4 26B MoE proxy | strong | 64 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 1.02e+05 J |
| Gemma 4 26B MoE proxy | strong | 64 | training | M1000 full photonic compiler stack | 1.56x | 1.01e+05 J |
| Gemma 4 26B MoE proxy | strong | 9216 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 1.32e+04 J |
| Gemma 4 26B MoE proxy | strong | 9216 | inference | M1000 full photonic compiler stack | 42.43x | 788 J |
| Gemma 4 26B MoE proxy | strong | 9216 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 1.88e+05 J |
| Gemma 4 26B MoE proxy | strong | 9216 | training | M1000 full photonic compiler stack | 3.37x | 1.01e+05 J |
| Gemma 4 26B MoE proxy | weak | 64 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 1.37e+03 J |
| Gemma 4 26B MoE proxy | weak | 64 | inference | M1000 full photonic compiler stack | 9.16x | 788 J |
| Gemma 4 26B MoE proxy | weak | 64 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 1.02e+05 J |
| Gemma 4 26B MoE proxy | weak | 64 | training | M1000 full photonic compiler stack | 1.56x | 1.01e+05 J |
| Gemma 4 26B MoE proxy | weak | 9216 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 1.88e+06 J |
| Gemma 4 26B MoE proxy | weak | 9216 | inference | M1000 full photonic compiler stack | 43.98x | 1.07e+05 J |
| Gemma 4 26B MoE proxy | weak | 9216 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 1.88e+05 J |
| Gemma 4 26B MoE proxy | weak | 9216 | training | M1000 full photonic compiler stack | 3.36x | 1.01e+05 J |
| Gemma 4 31B dense proxy | strong | 64 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 2.4e+03 J |
| Gemma 4 31B dense proxy | strong | 64 | inference | M1000 full photonic compiler stack | 6.42x | 1.61e+03 J |
| Gemma 4 31B dense proxy | strong | 64 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 7.55e+05 J |
| Gemma 4 31B dense proxy | strong | 64 | training | M1000 full photonic compiler stack | 1.38x | 7.53e+05 J |
| Gemma 4 31B dense proxy | strong | 9216 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 1.82e+04 J |
| Gemma 4 31B dense proxy | strong | 9216 | inference | M1000 full photonic compiler stack | 27.88x | 1.61e+03 J |
| Gemma 4 31B dense proxy | strong | 9216 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 8.55e+05 J |
| Gemma 4 31B dense proxy | strong | 9216 | training | M1000 full photonic compiler stack | 1.66x | 7.53e+05 J |
| Gemma 4 31B dense proxy | weak | 64 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 2.4e+03 J |
| Gemma 4 31B dense proxy | weak | 64 | inference | M1000 full photonic compiler stack | 6.42x | 1.61e+03 J |
| Gemma 4 31B dense proxy | weak | 64 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 7.55e+05 J |
| Gemma 4 31B dense proxy | weak | 64 | training | M1000 full photonic compiler stack | 1.38x | 7.53e+05 J |
| Gemma 4 31B dense proxy | weak | 9216 | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 2.55e+06 J |
| Gemma 4 31B dense proxy | weak | 9216 | inference | M1000 full photonic compiler stack | 33.34x | 1.89e+05 J |
| Gemma 4 31B dense proxy | weak | 9216 | training | Ironwood electrical torus + XLA SPMD | 1.00x | 8.54e+05 J |
| Gemma 4 31B dense proxy | weak | 9216 | training | M1000 full photonic compiler stack | 1.66x | 7.53e+05 J |

## Optical Circuit Switching vs Optical Packet Switching

| Model | Phase | Switching family | Speedup vs OCS topology-only | Network latency share |
|---|---|---|---|---|
| Gemini 3.1 Pro-class sparse MoE proxy | inference | OCS full compiler stack | 1.20x | 0.06% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | OPS fine-grained | 1.03x | 0.11% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | OCS full compiler stack | 1.20x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | OPS fine-grained | 1.03x | 0.00% |
| Gemma 4 26B MoE proxy | inference | OCS full compiler stack | 1.25x | 0.26% |
| Gemma 4 26B MoE proxy | inference | OPS fine-grained | 1.06x | 0.50% |
| Gemma 4 26B MoE proxy | training | OCS full compiler stack | 1.20x | 0.03% |
| Gemma 4 26B MoE proxy | training | OPS fine-grained | 1.03x | 0.04% |

## Unified Thesis Evidence Chain

| Layer | Question | Evidence | Result | Interpretation |
|---|---|---|---|---|
| AccelForge logical memory | Does topology-aware logical memory change measured accelerator cost? | decode_mlp / shared_dram_multicast | 71.95 % energy reduction | Network cost is visible to the mapper through LogicalClusterMemory. |
| Distributed mapping | Does the best inter-chip communication pattern change? | cluster + TPU cube optimal mapping table | 5.00 of 16 cases | Photonic multicast makes shared-state/cache-fill schedules optimal in cases where electrical prefers exchange. |
| Compiler/dataflow | Does XLA/Pallas-visible policy change above the local tile mapping? | kv_attention / EVK100 photonic torus | 73.86 % energy gain | Best policy becomes Remote pooled-HBM streaming. |
| Distance physics | What happens when electrical paths get physically longer? | 9216-node shared-state multicast | 395.21 x electrical/optical pJ/bit | Electrical cost grows with path distance; optical cost is dominated by conversion and fanout. |
| End-to-end strong/weak scaling | Does the advantage survive when scaling system and workload? | Gemma 4 26B MoE proxy / inference / weak | 43.98 x latency speedup | Strong/weak scaling separates latency shrink from aggregate-throughput scaling. |
| Optical switching semantics | When might packet switching beat circuit switching? | Gemma 4 26B MoE proxy / inference | 1.06 x vs OCS topology-only | OPS sensitivity is strongest for fine-grained dynamic communication; OCS remains better for bulk multicast/cache-fill. |

## Sources

- **lightmatter_evk50** (2025-12-16): Passage EVK50 (SC25 demo): 800 Gbps per fiber, 16-lambda BiDi NRZ 50G/lambda. 4.6 pJ/bit total wall-plug (2.6 pJ/bit photonics + Guide + ~2 pJ/bit XSR SerDes). BER < 10^-9, >1 km reach, polarisation-insensitive, 25–105 °C range.
  https://lightmatter.co/resource/bobturnersc25/
- **lightmatter_evk100** (viewed 2026-04-12): Passage EVK100: 3.2 Tbps per fiber at 112G PAM4 (16 lambda). 1.9 pJ/bit reference value on the Passage product family page / ISES Korea 2025; the EVK100 technical-spec page also lists a conservative 3.2 pJ/bit point. Monolithic PIC: MRM modulators, TIAs, Tx drivers on single chip.
  https://lightmatter.co/products/passage/
- **lightmatter_l200** (2025 product page): Passage L200: 32 Tbps optical I/O per chip (56G NRZ, 3D CoW, 320 SerDes). L200X: 64 Tbps at 112G PAM4. Both <5 pJ/bit. Bandwidth density >1.5 Tbps/mm. Targeted 2026.
  https://lightmatter.co/products/l200/
- **lightmatter_m1000** (viewed 2026-04-12): Passage M1000 EVK: 114.6 Tbps total bi-directional bandwidth, 2.3 pJ/bit system-level efficiency including laser, 1024 SerDes lanes, 256 optical fibers, 4,000 mm² active interposer area, integrated solid-state OCS.
  https://lightmatter.co/products/passage-m1000-evk/
- **lightmatter_ocp** (2026-03-16): OCP CPO reference architecture initiative. Partners: Celestica, Corning, Dell, Flex, Foxconn, Hyve, Keysight, Qualcomm, Quanta. Standards: OIF, IEEE, OCP.
  https://lightmatter.co/press-release/lightmatter-announces-reference-architecture-initiative-with-industry-leaders-in-the-open-compute-project-for-co-packaged-optics/
- **lightmatter_record_1p6tbps** (2026-03): World-record 1.6 Tbps per fiber (Qualcomm 112G PAM4 SerDes + Passage L-Series). 8x more bandwidth per fiber than Near-Packaged Optics.
  https://lightmatter.co/press-release/lightmatter-achieves-record-1-6-tbps-per-fiber-to-accelerate-ai-optical-interconnect/
- **lightmatter_tco** (2025-08): >60 % TCO savings and 3–7x training speedup for 4.7T-param MoE (256 experts, 8 active) over 576 XPUs using L-Series vs electrical. Roadmap: pluggables 17–20 pJ/bit → 3D CPO as low as 3 pJ/bit.
  https://lightmatter.co/wp-content/uploads/2025/08/ISES-Korea-08-2025_Ritesh-Jain.pdf
- **Gemini 3** (2025-11-18): Google introduced Gemini 3 as its latest and most intelligent Gemini model family, with Gemini 3 Pro in preview across Gemini, AI Studio, Vertex AI, Search, and Antigravity. Public posts do not disclose parameter counts.
  https://blog.google/products-and-platforms/products/gemini/gemini-3/
- **Gemini 3.1 release notes** (viewed 2026-04-17): Gemini app release notes state that Gemini 3.1 Pro is rolling out globally with higher limits for Google AI Pro and Ultra plans. The notes identify advanced reasoning but do not publish architecture or parameter sizes.
  https://gemini.google/gb/release-notes/?hl=en-GB
- **Gemini 3 Flash** (2025-12-17): Gemini 3 Flash is described as a speed-optimized Gemini 3 model. Google reported more than 1T API tokens/day after Gemini 3 launch, 30% fewer thinking tokens than 2.5 Pro on typical traffic, and 3x faster behavior than 2.5 Pro based on external benchmarking cited in the post.
  https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/
- **Gemini 3 Deep Think** (2026-02-12): Google released an updated Gemini 3 Deep Think mode for scientific, research, and engineering tasks; it emphasizes longer, more expensive reasoning. Public posts do not disclose model size.
  https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/
- **gemma_4** (2026-04): Google released Gemma 4 open models in E2B, E4B, 26B MoE, and 31B dense sizes, with up to 256K context on larger models, Apache 2.0 licensing, and support for TPU-accelerated serving on Google Cloud. This provides the open-model anchor for the end-to-end proxy workloads.
  https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- **xla_spmd** (viewed 2026-04-17): PyTorch/XLA SPMD exposes mesh and partition specs; XLA transforms a single-device program into a partitioned program with collectives using sharding annotations. This motivates modeling compiler-visible dataflow choices, not just link swaps.
  https://docs.pytorch.org/xla/master/spmd.html
- **TPU7x / Ironwood** (last updated 2026-04-15): TPU7x/Ironwood is Google's latest Cloud TPU. A pod has 9216 chips, 2307 BF16 TFLOP/s/chip, 4614 FP8 TFLOP/s/chip, 192 GiB HBM/chip, 7380 GiB/s HBM bandwidth/chip, and 1200 GB/s bidirectional ICI/chip. Google states it is designed for dense and MoE models, pre-training, sampling, and decode-heavy inference.
  https://docs.cloud.google.com/tpu/docs/tpu7x
- **TPU v5p** (viewed 2026-04-17): TPU v5p: 8960 chips per pod; 459 TFLOPS/chip, 95 GiB HBM, 2575 GiB/s HBM BW, 1200 GB/s bidirectional ICI per chip, 3D torus; 4x4x4 and larger slices have full torus connectivity.
  https://cloud.google.com/tpu/docs/v5p
- **Ironwood blog_blog** (viewed 2026-04-17): Google describes Ironwood as XLA/software co-designed. A 64-chip cube uses dense 3D torus ICI; cubes scale to 9216 chips through an OCS fabric with dynamic optical reconfiguration and 88473.6 Tbps aggregate superpod connectivity.
  https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/
- **tpu_rack_hotchips** (2025-08-24): Ironwood (2025): 4614 TFLOPS/chip, 192 GB HBM, 7.4 TBps HBM BW. 4x4x4 torus building block; 6 cube faces × 16 ICI links each. 96 optical + 80 copper + 64 PCB ICI connections per rack. Rack power-cap SLO < 15 ms; burst TDP window = 100 ms.
  https://hc2025.hotchips.org/assets/program/tutorials/HC2025.T1DCRacks.S7.Pankaj.pdf
- **advanced_chiplet_baseline** (project modeling assumption): Advanced-packaging comparison point requested by the project prompt: 3.5 Tbps one-way link bandwidth and 6 pJ/bit per network hop.
  https://docs.cloud.google.com/tpu/docs/v5p
