# Lightmatter Final Project — Summary

Generated from the AccelForge + photonic-interconnect study pipeline.

## Technology Catalog

| Technology | BW (one-way) | Energy (pJ/bit) | Multicast |
|---|---|---|---|
| Electrical 800G | 800 Gbps | 10.0 | No |
| Lightmatter EVK50 | 0.8 Tbps | 4.6 | Yes |
| Lightmatter EVK100 | 3.2 Tbps | 1.9 | Yes |
| Lightmatter L200 | 16.0 Tbps | 4.5 | Yes |
| Lightmatter L200X | 32.0 Tbps | 4.8 | Yes |
| Lightmatter M1000 | 57.0 Tbps | 4.0 | Yes |

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

- Best energy reduction: `shared_weight_multicast` → `99.84%`
- Best latency reduction: `shared_weight_multicast` → `99.99%`

### Photonic torus (EVK100)

- Best energy reduction: `shared_weight_multicast` → `99.70%`
- Best latency reduction: `shared_weight_multicast` → `99.61%`

## Sources

- **lightmatter_evk50** (2025-12-16): Passage EVK50 (SC25 demo): 800 Gbps per fiber, 16-lambda BiDi NRZ 50G/lambda. 4.6 pJ/bit total wall-plug (2.6 pJ/bit photonics + Guide + ~2 pJ/bit XSR SerDes). BER < 10^-9, >1 km reach, polarisation-insensitive, 25–105 °C range.  
  https://lightmatter.co/resource/bobturnersc25/
- **lightmatter_evk100** (2025): Passage EVK100: 3.2 Tbps per fiber at 112G PAM4 (16 lambda). 1.9 pJ/bit (per ISES Korea 2025 efficiency table). Monolithic PIC: MRM modulators, TIAs, Tx drivers on single chip.  
  https://lightmatter.co/products/passage/
- **lightmatter_l200** (2025 product page): Passage L200: 32 Tbps optical I/O per chip (56G NRZ, 3D CoW, 320 SerDes). L200X: 64 Tbps at 112G PAM4. Both <5 pJ/bit. Bandwidth density >1.5 Tbps/mm. Targeted 2026.  
  https://lightmatter.co/products/l200/
- **lightmatter_m1000** (2025-05): Passage M1000: 114 Tbps total, 256 edge-attached fibers, 1024 SerDes lanes. 7735 mm² package, 34 chiplets, built-in solid-state OCS. Sampling summer 2025.  
  https://lightmatter.co/press-release/lightmatter-unveils-passage-m1000-photonic-superchip-worlds-fastest-ai-interconnect/
- **lightmatter_ocp** (2026-03-16): OCP CPO reference architecture initiative. Partners: Celestica, Corning, Dell, Flex, Foxconn, Hyve, Keysight, Qualcomm, Quanta. Standards: OIF, IEEE, OCP.  
  https://lightmatter.co/press-release/lightmatter-announces-reference-architecture-initiative-with-industry-leaders-in-the-open-compute-project-for-co-packaged-optics/
- **lightmatter_record_1p6tbps** (2026-03): World-record 1.6 Tbps per fiber (Qualcomm 112G PAM4 SerDes + Passage L-Series). 8x more bandwidth per fiber than Near-Packaged Optics.  
  https://lightmatter.co/press-release/lightmatter-achieves-record-1-6-tbps-per-fiber-to-accelerate-ai-optical-interconnect/
- **lightmatter_tco** (2025-08): >60 % TCO savings and 3–7x training speedup for 4.7T-param MoE (256 experts, 8 active) over 576 XPUs using L-Series vs electrical. Roadmap: pluggables 17–20 pJ/bit → 3D CPO as low as 3 pJ/bit.  
  https://lightmatter.co/wp-content/uploads/2025/08/ISES-Korea-08-2025_Ritesh-Jain.pdf
- **tpu_v5p** (viewed 2026-04-01): TPU v5p: 459 TFLOPS/chip, 95 GB HBM, 2.8 TBps HBM BW, 1200 GB/s bidirectional ICI per chip, 3D torus, 4x4x4 full cube.  
  https://cloud.google.com/tpu/docs/v5p
- **tpu_rack_hotchips** (2025-08-24): Ironwood (2025): 4614 TFLOPS/chip, 192 GB HBM, 7.4 TBps HBM BW. 4x4x4 torus building block; 6 cube faces × 16 ICI links each. 96 optical + 80 copper + 64 PCB ICI connections per rack. Rack power-cap SLO < 15 ms; burst TDP window = 100 ms.  
  https://hc2025.hotchips.org/assets/program/tutorials/HC2025.T1DCRacks.S7.Pankaj.pdf
