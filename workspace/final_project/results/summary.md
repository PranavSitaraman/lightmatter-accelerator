# Lightmatter Final Project — Summary

Generated from the AccelForge + photonic-interconnect study pipeline.

## Executive Takeaways

- The raw network model and the AccelForge fixed-point mapper both show the same direction: optical links help most when traffic leaves a single local HBM bank.
- Re-running the mapper is explicitly measured: the pipeline reports both fixed-point tile/residency sensitivity and the higher-level distributed mapping choice that becomes optimal under each interconnect.
- A separate compiler/dataflow strategy model shows the more important effect: photonic multicast can make software-managed cache fills and shared-state broadcast preferable to classic SPMD activation exchange.
- Scaling-law sweeps show that flat switched photonic fabrics keep hop count and contention nearly constant while torus and chiplet fabrics accumulate hop/contention cost as cubes grow toward superpod scale.
- The full-superpod study models Gemini 3/3.1-class public proxy workloads on a 9216-chip Ironwood pod and reports end-to-end inference/training speedups by stage and by optimization ablation.
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

## Cluster Study Highlights (16-node system)

### node_to_dram

- Best energy reduction: `decode_mlp` → `67.91%`
- Best latency reduction: `decode_mlp` → `91.84%`

### node_to_node

- Best energy reduction: `prefill_projection` → `9.40%`
- Best latency reduction: `decode_mlp` → `0.00%`

### shared_dram_multicast

- Best energy reduction: `decode_mlp` → `72.21%`
- Best latency reduction: `kv_attention` → `97.57%`

## TPU Cube Highlights (64-node 4×4×4 cube)

### Photonic switched fabric (L200)

- Best energy reduction: `shared_weight_multicast` → `99.82%`
- Best latency reduction: `shared_weight_multicast` → `99.98%`

### Photonic switched fabric (L200X)

- Best energy reduction: `shared_weight_multicast` → `99.81%`
- Best latency reduction: `shared_weight_multicast` → `99.99%`

### Photonic switched fabric (M1000)

- Best energy reduction: `shared_weight_multicast` → `99.91%`
- Best latency reduction: `shared_weight_multicast` → `99.99%`

### Photonic torus (EVK100)

- Best energy reduction: `shared_weight_multicast` → `99.70%`
- Best latency reduction: `shared_weight_multicast` → `99.61%`

## TPU Cube Mapping Adaptation (AccelForge fixed point)

- Best mapping-aware TPU-cube energy reduction: `TPU cube M1000 switched fabric` on `kv_attention / shared_dram_multicast` → `77.70%`.
- Mapping signatures differed from the electrical baseline in `0` of `24` photonic mapping cases.
- Example final mapping signature: `node.NodeX=0.25; node.NodeY=0.25; node.NodeZ=0.25; pe.reuse_input=1; pe.reuse_output=1; logical.Weights::read=8192.0 Kib; logical_usage.Weights=1.606e-07; hbm_usage.Activations=1.255e-09; hbm_usage.Outputs=8.031e-08; action::HBM::Activations::read=1.0 Kib; action::HBM::Outputs::write=64.0 Kib; action::MAC::None::compute=1024.0 Kib; action::Register::Weights::read=8192.0 Kib; action::Register::Weights::write=8192.0 Kib; action::UnifiedBuffer::Activations::read=128.0 Kib; action::UnifiedBuffer::Activations::write=1.0 Kib; action::UnifiedBuffer::Outputs::read=128.0 Kib; action::UnifiedBuffer::Outputs::write=128.0 Kib`

## Optimal Distributed Mapping Choices

Here `scenario` is treated as the distributed inference mapping choice: node-to-node activation exchange, node-to-DRAM remote fetch, or shared-DRAM multicast.

| Workload | Architecture | Optimal mapping | Changed vs baseline |
|---|---|---|---|
| decode_mlp | Electrical 4x4 Mesh / Electrical 800G/dir | node_to_node | False |
| decode_mlp | Lightmatter Optical Crossbar / Lightmatter EVK100 | node_to_node | False |
| kv_attention | Electrical 4x4 Mesh / Electrical 800G/dir | node_to_node | False |
| kv_attention | Lightmatter Optical Crossbar / Lightmatter EVK100 | node_to_node | False |
| prefill_projection | Electrical 4x4 Mesh / Electrical 800G/dir | node_to_node | False |
| prefill_projection | Lightmatter Optical Crossbar / Lightmatter EVK100 | shared_dram_multicast | True |
| kv_attention | TPU cube EVK100 photonic torus | node_to_node | False |
| kv_attention | TPU cube L200 switched fabric | node_to_node | False |
| kv_attention | TPU cube L200X switched fabric | node_to_node | False |
| kv_attention | TPU cube M1000 switched fabric | node_to_node | False |
| kv_attention | TPU cube electrical torus | node_to_node | False |
| prefill_projection | TPU cube EVK100 photonic torus | shared_dram_multicast | True |
| prefill_projection | TPU cube L200 switched fabric | shared_dram_multicast | True |
| prefill_projection | TPU cube L200X switched fabric | shared_dram_multicast | True |
| prefill_projection | TPU cube M1000 switched fabric | shared_dram_multicast | True |
| prefill_projection | TPU cube electrical torus | node_to_node | False |

## Compiler/Dataflow Strategy Choices

This layer models XLA/Pallas-visible choices above the inner AccelForge tile mapper.

| Workload | Architecture | Optimal strategy | Changed vs TPU electrical | Energy gain vs baseline |
|---|---|---|---|---|
| decode_mlp | Advanced chiplet torus | XLA SPMD collective | False | 0.05% |
| decode_mlp | EVK100 photonic torus | XLA SPMD collective | False | 0.11% |
| decode_mlp | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 67.21% |
| decode_mlp | TPU electrical torus | XLA SPMD collective | False | 0.00% |
| kv_attention | Advanced chiplet torus | XLA SPMD collective | False | 1.32% |
| kv_attention | EVK100 photonic torus | XLA SPMD collective | False | 2.67% |
| kv_attention | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 67.75% |
| kv_attention | TPU electrical torus | XLA SPMD collective | False | 0.00% |
| prefill_projection | Advanced chiplet torus | Remote pooled-HBM streaming | True | 20.45% |
| prefill_projection | EVK100 photonic torus | Photonic multicast cache | True | 45.43% |
| prefill_projection | M1000 optical switched fabric | Remote pooled-HBM streaming | True | 53.31% |
| prefill_projection | TPU electrical torus | XLA SPMD collective | False | 0.00% |

## Scaling-Law Exponents

Lower exponents mean better scaling as the system grows from one 64-chip cube toward 9216 chips.

| Architecture | Scenario | Metric | Exponent |
|---|---|---|---|
| Advanced chiplet torus | all_to_all_activation | energy_uJ_per_chip | 0.336 |
| Advanced chiplet torus | all_to_all_activation | latency_us_per_chip | 0.333 |
| Advanced chiplet torus | shared_state_multicast | energy_uJ_per_chip | 1.295 |
| Advanced chiplet torus | shared_state_multicast | latency_us_per_chip | 1.333 |
| EVK100 photonic torus | all_to_all_activation | energy_uJ_per_chip | 0.336 |
| EVK100 photonic torus | all_to_all_activation | latency_us_per_chip | 0.333 |
| EVK100 photonic torus | shared_state_multicast | energy_uJ_per_chip | 0.295 |
| EVK100 photonic torus | shared_state_multicast | latency_us_per_chip | 0.333 |
| M1000 optical switched fabric | all_to_all_activation | energy_uJ_per_chip | 0.000 |
| M1000 optical switched fabric | all_to_all_activation | latency_us_per_chip | 0.000 |
| M1000 optical switched fabric | shared_state_multicast | energy_uJ_per_chip | 0.000 |
| M1000 optical switched fabric | shared_state_multicast | latency_us_per_chip | 0.000 |
| TPU electrical torus | all_to_all_activation | energy_uJ_per_chip | 0.336 |
| TPU electrical torus | all_to_all_activation | latency_us_per_chip | 0.333 |
| TPU electrical torus | shared_state_multicast | energy_uJ_per_chip | 1.295 |
| TPU electrical torus | shared_state_multicast | latency_us_per_chip | 1.333 |

## Advanced Chiplet Baseline

The chiplet baseline uses the requested 3.5 Tbps one-way, 6 pJ/bit per-hop model.

| Scenario | Architecture | Energy reduction vs chiplet | Latency reduction vs chiplet |
|---|---|---|---|
| all_to_all_activation | EVK100 photonic torus | 68.33% | -9.37% |
| all_to_all_activation | M1000 optical switched fabric | 87.42% | 98.47% |
| remote_weight_refill | EVK100 photonic torus | 68.33% | -9.37% |
| remote_weight_refill | M1000 optical switched fabric | 90.42% | 98.47% |
| shared_state_multicast | EVK100 photonic torus | 99.51% | 98.29% |
| shared_state_multicast | M1000 optical switched fabric | 99.85% | 99.98% |

## Full-Superpod Gemini-Class End-to-End Study

Google has not published Gemini 3/3.1 parameter counts, so this section reports explicit dense/MoE proxies rather than claiming confidential model internals.

| Model | Phase | Configuration | Speedup vs electrical | Energy reduction | Network latency share | Throughput tokens/s |
|---|---|---|---|---|---|---|
| Gemini 3 Deep Think long-context proxy | inference | M1000 full photonic compiler stack | 14.50x | 51.07% | 0.07% | 1.17e+07 |
| Gemini 3 Deep Think long-context proxy | inference | M1000 switched optical topology only | 12.02x | 49.18% | 0.58% | 9.71e+06 |
| Gemini 3 Deep Think long-context proxy | inference | Advanced chiplet torus + XLA SPMD | 3.65x | 19.92% | 67.50% | 2.95e+06 |
| Gemini 3 Deep Think long-context proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 90.67% | 8.07e+05 |
| Gemini 3 Deep Think long-context proxy | training | M1000 full photonic compiler stack | 1.43x | 0.67% | 0.00% | 3.89e+06 |
| Gemini 3 Deep Think long-context proxy | training | M1000 switched optical topology only | 1.19x | 0.67% | 0.00% | 3.24e+06 |
| Gemini 3 Deep Think long-context proxy | training | Advanced chiplet torus + XLA SPMD | 1.10x | 0.27% | 1.23% | 2.97e+06 |
| Gemini 3 Deep Think long-context proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 5.52% | 2.71e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | M1000 full photonic compiler stack | 11.02x | 45.27% | 0.05% | 2.42e+07 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | M1000 switched optical topology only | 9.15x | 43.82% | 0.40% | 2.01e+07 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | Advanced chiplet torus + XLA SPMD | 3.38x | 17.72% | 60.41% | 7.41e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 87.71% | 2.2e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | M1000 full photonic compiler stack | 1.42x | 0.54% | 0.00% | 4.86e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | M1000 switched optical topology only | 1.18x | 0.53% | 0.00% | 4.05e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | Advanced chiplet torus + XLA SPMD | 1.09x | 0.22% | 0.99% | 3.73e+06 |
| Gemini 3.1 Pro-class sparse MoE proxy | training | Ironwood electrical torus + XLA SPMD | 1.00x | 0.00% | 4.47% | 3.43e+06 |

## Superpod Optimization Ablations

Rows show how much each step contributes relative to the electrical Ironwood baseline.

| Model | Phase | Technique | Speedup vs baseline | Increment vs previous | Network energy share |
|---|---|---|---|---|---|
| Gemini 3.1 Pro-class sparse MoE proxy | inference | baseline | 1.00x | 1.00x | 44.30% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | advanced chiplet | 3.38x | 3.38x | 32.30% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | photonic torus + multicast | 5.43x | 1.61x | 6.68% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | switched topology only | 9.15x | 1.68x | 0.85% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | multicast cache | 9.70x | 1.06x | 0.39% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | remote pooled HBM | 10.09x | 1.04x | 0.30% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | MoE locality routing | 10.36x | 1.03x | 0.28% |
| Gemini 3.1 Pro-class sparse MoE proxy | inference | full photonic compiler stack | 11.02x | 1.06x | 0.22% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | baseline | 1.00x | 1.00x | 0.54% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | advanced chiplet | 1.09x | 1.09x | 0.33% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | photonic torus + multicast | 1.16x | 1.07x | 0.08% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | switched topology only | 1.18x | 1.02x | 0.01% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | multicast cache | 1.25x | 1.06x | 0.01% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | remote pooled HBM | 1.30x | 1.04x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | MoE locality routing | 1.33x | 1.03x | 0.00% |
| Gemini 3.1 Pro-class sparse MoE proxy | training | full photonic compiler stack | 1.42x | 1.06x | 0.00% |

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
- **gemini_3** (2025-11-18): Google introduced Gemini 3 as its latest and most intelligent Gemini model family, with Gemini 3 Pro in preview across Gemini, AI Studio, Vertex AI, Search, and Antigravity. Public posts do not disclose parameter counts.
  https://blog.google/products-and-platforms/products/gemini/gemini-3/
- **gemini_3_1_release_notes** (viewed 2026-04-17): Gemini app release notes state that Gemini 3.1 Pro is rolling out globally with higher limits for Google AI Pro and Ultra plans. The notes identify advanced reasoning but do not publish architecture or parameter sizes.
  https://gemini.google/gb/release-notes/?hl=en-GB
- **gemini_3_flash** (2025-12-17): Gemini 3 Flash is described as a speed-optimized Gemini 3 model. Google reported more than 1T API tokens/day after Gemini 3 launch, 30% fewer thinking tokens than 2.5 Pro on typical traffic, and 3x faster behavior than 2.5 Pro based on external benchmarking cited in the post.
  https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/
- **gemini_3_deep_think** (2026-02-12): Google released an updated Gemini 3 Deep Think mode for scientific, research, and engineering tasks; it emphasizes longer, more expensive reasoning. Public posts do not disclose model size.
  https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/
- **xla_spmd** (viewed 2026-04-17): PyTorch/XLA SPMD exposes mesh and partition specs; XLA transforms a single-device program into a partitioned program with collectives using sharding annotations. This motivates modeling compiler-visible dataflow choices, not just link swaps.
  https://docs.pytorch.org/xla/master/spmd.html
- **tpu7x** (last updated 2026-04-15): TPU7x/Ironwood is Google's latest Cloud TPU. A pod has 9216 chips, 2307 BF16 TFLOP/s/chip, 4614 FP8 TFLOP/s/chip, 192 GiB HBM/chip, 7380 GiB/s HBM bandwidth/chip, and 1200 GB/s bidirectional ICI/chip. Google states it is designed for dense and MoE models, pre-training, sampling, and decode-heavy inference.
  https://docs.cloud.google.com/tpu/docs/tpu7x
- **tpu_v5p** (viewed 2026-04-17): TPU v5p: 8960 chips per pod; 459 TFLOPS/chip, 95 GiB HBM, 2575 GiB/s HBM BW, 1200 GB/s bidirectional ICI per chip, 3D torus; 4x4x4 and larger slices have full torus connectivity.
  https://cloud.google.com/tpu/docs/v5p
- **ironwood_blog** (viewed 2026-04-17): Google describes Ironwood as XLA/software co-designed. A 64-chip cube uses dense 3D torus ICI; cubes scale to 9216 chips through an OCS fabric with dynamic optical reconfiguration and 88473.6 Tbps aggregate superpod connectivity.
  https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/
- **tpu_rack_hotchips** (2025-08-24): Ironwood (2025): 4614 TFLOPS/chip, 192 GB HBM, 7.4 TBps HBM BW. 4x4x4 torus building block; 6 cube faces × 16 ICI links each. 96 optical + 80 copper + 64 PCB ICI connections per rack. Rack power-cap SLO < 15 ms; burst TDP window = 100 ms.
  https://hc2025.hotchips.org/assets/program/tutorials/HC2025.T1DCRacks.S7.Pankaj.pdf
- **advanced_chiplet_baseline** (project modeling assumption): Advanced-packaging comparison point requested by the project prompt: 3.5 Tbps one-way link bandwidth and 6 pJ/bit per network hop.
  https://docs.cloud.google.com/tpu/docs/v5p
