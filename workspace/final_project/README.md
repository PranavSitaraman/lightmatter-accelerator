# Final Project — Lightmatter Photonic Interconnects in AccelForge

**6.5930 Hardware Architecture for Deep Learning**

This project evaluates Lightmatter photonic interconnects as a replacement for electrical
ICI (specifically TPU v5p cube ICI) by integrating a topology-aware network model with
AccelForge's iterative hardware mapper.

## Deliverables

| File | Description |
|---|---|
| `4_lightmatter_interconnects.ipynb` | **Main executed notebook** — all 3 milestones + TPU cube ICI study + mapping-choice analysis + compiler/dataflow strategies + scaling laws + chiplet baseline + full-superpod Gemini-class study. 80 cells, fully executed, 0 errors. |
| `scripts/lightmatter_interconnects.py` | Core study engine: topology models, iterative solver, mapping signatures, TPU cube mapping study, compiler/dataflow study, scaling laws, full-superpod model, technology catalog, sweep functions, CLI. |
| `scripts/generate_lightmatter_notebook.py` | Programmatic notebook generator. |
| `tests/test_lightmatter_project.py` | 68-test suite covering all components. |
| `arch/tpu_lightmatter_cluster.yaml` | AccelForge architecture: 2D/3D logical cluster memory + node array + HBM + UB + MAC. |
| `workload/single_matmul.yaml` | Parameterized GEMM workload template (M, K, N, bits). |
| `results/presentation/` | Concise final-report package: executive summary, topology diagram, and nine condensed plots. |

## Technology Catalog

| Technology | BW (one-way) | Energy (pJ/bit) | Multicast | Source |
|---|---|---|---|---|
| Electrical 800G | 800 Gbps | 10.0 (assumed) | No | TPU v5p ICI inference |
| **EVK50** | 800 Gbps | 4.6 | Yes | SC25 demo, Dec 2025 |
| **EVK100** | 3.2 Tbps | **1.9** | Yes | Passage family / ISES Korea 2025 reference value; 3.2 pJ/bit tracked as conservative sensitivity context |
| **L200** | 16 Tbps | 4.5 | Yes | Product page (32 Tbps total) |
| **L200X** | 32 Tbps | 4.8 | Yes | Product page (64 Tbps total) |
| **M1000** | 57.3 Tbps | 2.3 | Yes + OCS | M1000 EVK product page (114.6 Tbps total bi-directional) |
| **Advanced Chiplet 3.5T** | 3.5 Tbps | 6.0 per hop | No | Prompt-specified advanced packaging baseline |

## Key Results

### Cluster Study — 16-node optical crossbar (EVK100) vs. electrical 4×4 mesh

| Scenario | Best energy reduction | Best latency reduction |
|---|---|---|
| `node_to_dram` | **68 %** (decode_mlp) | **92 %** (decode_mlp) |
| `shared_dram_multicast` | **72 %** (decode_mlp) | **98 %** (kv_attention) |
| `node_to_node` | 9 % (prefill_projection) | < 1 % |

### TPU Cube ICI Replacement — 64-chip 4×4×4 cube

| Variant | all_to_all | weight_refill | multicast |
|---|---|---|---|
| Photonic torus (EVK100) | 81 % energy ↓ | 81 % energy ↓ | **99.7 %** energy ↓ |
| Photonic switched (L200) | 85 % energy ↓ | 89 % energy ↓ | **99.8 %** energy ↓ |
| Photonic switched (M1000) | 92 % energy ↓ | 94 % energy ↓ | **99.9 %** energy ↓ |

### Mapping-Choice + Compiler/Dataflow Result

The fixed-point AccelForge tile/residency signatures are intentionally conservative in the tested TPU-cube cases, but the best **distributed inference mapping** and compiler-visible dataflow change when photonic multicast and high-radix optical switching become cheap. For `prefill_projection`, the electrical baseline prefers `node_to_node`, while EVK100/L200/L200X/M1000 photonic variants prefer `shared_dram_multicast`, yielding about 11 % mapping-aware energy gain on the 4×4×4 cube.

The added compiler/dataflow model captures an XLA/GSPMD-style choice among sharded collectives, remote pooled-HBM streaming, replicated HBM caching, and photonic multicast cache fills. It shows strategy changes that are not simple constant-factor link substitutions: M1000 optical switching prefers remote pooled-HBM streaming for `decode_mlp` and `kv_attention`, while EVK100 photonic torus prefers multicast cache fill for `prefill_projection`.

### Scaling + Stronger Baseline Result

Scaling studies now compare cube growth from 64 chips to superpod scale against a prompt-specified advanced chiplet baseline of 3.5 Tb/s and 6 pJ/bit/hop. The photonic switched-fabric model has flat per-chip scaling exponents for the modeled traffic cases, while torus/chiplet traffic grows with hop count. Against the advanced chiplet baseline, M1000 optical switching reaches about 87-90 % lower energy for unicast/refill traffic and about 99.85 % lower energy for multicast-shared state in the 64-chip cube case.

### Full Superpod Gemini-Class Result

The project now models public Gemini 3/3.1-class proxy workloads on a full 9216-chip Ironwood/TPU7x pod. Because Google does not publish Gemini 3/3.1 parameter counts, the study uses explicit dense, sparse MoE, and long-context proxies rather than claiming confidential model internals. The M1000 full photonic compiler stack reaches 14.5x modeled inference latency speedup for the long-context proxy and 1.43x modeled training-step speedup, showing the key distinction: inference remains network/cache/expert-routing sensitive, while training becomes compute dominated once collectives are reduced.

## Generated Outputs (results/)

```
cluster_study.csv          — 3 workloads × 3 scenarios × 2 topologies
cluster_improvements.csv   — optical vs. mesh reduction per case
extended_cluster_study.csv — all 6 workloads × 3 scenarios
node_scaling.csv           — improvement vs. node count (2–128)
tpu_cube_case_study.csv    — 5 variants × 3 traffic cases
tpu_improvements.csv       — % reductions vs. electrical torus baseline
tpu_mapping_study.csv      — AccelForge fixed-point 4×4×4 TPU-cube remapping study
optimal_mapping_choices.csv — best distributed mapping per architecture/workload
compiler_strategy_study.csv — XLA/GSPMD/Pallas-style compiler/dataflow strategy sweep
optimal_compiler_strategies.csv — best strategy per workload/architecture
scaling_law_study.csv      — 64-chip to superpod-scale energy/latency scaling
scaling_exponents.csv      — log-log per-chip scaling exponents
chiplet_comparison.csv     — photonic vs. advanced 3.5 Tb/s, 6 pJ/bit chiplet baseline
superpod_stage_study.csv   — stage-level Gemini-class inference/training superpod model
superpod_end_to_end_summary.csv — end-to-end training/inference totals and speedups
superpod_ablation_study.csv — topology/compiler/dataflow ablation ladder
superpod_context_scaling.csv — long-context inference scaling sensitivity
presentation/              — concise executive summary + 9 chart/diagram assets
sources.csv                — research bibliography
summary.md                 — auto-generated highlights report
*.png                      — all plots (tech_catalog, convergence, heatmap, ...)
```

## Workflow

```bash
# Run the full study (CSVs + summary)
make study

# Regenerate and execute the notebook
make notebook

# Run all 68 tests
make test

# Clean generated outputs
make clean-results
```

All targets run inside the AccelForge Docker container (`timeloopaccelergy/accelforge:latest-amd64`).

## Project Structure — Milestone Mapping

| Milestone | Where |
|---|---|
| M1: Network model + logical memory | `solve_iteratively`, `Topology` classes, `arch/tpu_lightmatter_cluster.yaml` |
| M2: Iterative mapper + convergence analysis | `solve_iteratively`, `compare_topologies`, notebook §Milestone 2 |
| M3: Proposed optical topology | `OpticalCrossbarTopology`, `cluster_improvement_frame`, notebook §Milestone 3 |
| TPU cube ICI study | `tpu_cube_case_study`, `tpu_improvement_frame`, `tpu_cube_mapping_study`, notebook §TPU Cube |
| Mapping-choice analysis | `optimal_distributed_mapping_frame`, `mapping_signature`, `mapping_features_frame`, notebook §TPU Cube Mapping Adaptation |
| Compiler/dataflow adaptation | `compiler_strategy_study`, `optimal_compiler_strategy_frame`, notebook §Compiler/Dataflow |
| Extended: sensitivity + scaling | `bandwidth_sensitivity_network`, `energy_per_bit_sensitivity_network`, `scaling_law_study`, notebook §Sensitivity and §Scaling Laws |
| Stronger advanced-packaging baseline | `ADVANCED_CHIPLET_3P5T`, `chiplet_comparison_frame`, notebook §Advanced Chiplet Baseline |
| Full-superpod Gemini-class study | `superpod_stage_study`, `superpod_end_to_end_summary`, `superpod_ablation_frame`, notebook §Full Superpod |
