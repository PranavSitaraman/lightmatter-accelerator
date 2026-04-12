# Lightmatter Final Project — Summary

Generated from the AccelForge + photonic-interconnect study pipeline.

## Executive Takeaways

- The raw network model and the AccelForge fixed-point mapper both show the same direction: optical links help most when traffic leaves a single local HBM bank.
- Re-running the mapper is explicitly measured: the pipeline reports both fixed-point tile/residency sensitivity and the higher-level distributed mapping choice that becomes optimal under each interconnect.
- The TPU cube study separates two design points: a conservative EVK100 photonic torus and higher-bandwidth switched-fabric options (L200/L200X/M1000).
- EVK100 public collateral includes both a 1.9 pJ/bit Passage-family value and a more conservative 3.2 pJ/bit EVK100 technical-spec value; this project uses 1.9 pJ/bit as the main optimistic/reference point and includes sensitivity hooks for alternatives.

## Technology Catalog

| Technology | BW (one-way) | Energy (pJ/bit) | Multicast |
|---|---|---|---|
| Electrical 800G | 800 Gbps | 10.0 | No |
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
- **tpu_v5p** (viewed 2026-04-12): TPU v5p: 459 TFLOPS/chip, 95 GB HBM, 2.8 TBps HBM BW, 1200 GB/s bidirectional ICI per chip, 3D torus, 4x4x4 full cube.  
  https://cloud.google.com/tpu/docs/v5p
- **tpu_rack_hotchips** (2025-08-24): Ironwood (2025): 4614 TFLOPS/chip, 192 GB HBM, 7.4 TBps HBM BW. 4x4x4 torus building block; 6 cube faces × 16 ICI links each. 96 optical + 80 copper + 64 PCB ICI connections per rack. Rack power-cap SLO < 15 ms; burst TDP window = 100 ms.  
  https://hc2025.hotchips.org/assets/program/tutorials/HC2025.T1DCRacks.S7.Pankaj.pdf
