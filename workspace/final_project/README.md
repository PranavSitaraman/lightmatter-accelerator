# Final Project — Lightmatter Photonic Interconnects in AccelForge

**6.5930 Hardware Architecture for Deep Learning**

This project evaluates Lightmatter photonic interconnects as a replacement for electrical
ICI (specifically TPU v5p cube ICI) by integrating a topology-aware network model with
AccelForge's iterative hardware mapper.

## Deliverables

| File | Description |
|---|---|
| `4_lightmatter_interconnects.ipynb` | **Main executed notebook** — all 3 milestones + TPU cube ICI study + sensitivity analysis. 55 cells, fully executed, 0 errors. |
| `scripts/lightmatter_interconnects.py` | Core study engine: topology models, iterative solver, technology catalog, sweep functions, CLI. |
| `scripts/generate_lightmatter_notebook.py` | Programmatic notebook generator. |
| `tests/test_lightmatter_project.py` | 52-test suite covering all components. |
| `arch/tpu_lightmatter_cluster.yaml` | AccelForge architecture: logical cluster memory + node array + HBM + UB + MAC. |
| `workload/single_matmul.yaml` | Parameterized GEMM workload template (M, K, N, bits). |

## Technology Catalog

| Technology | BW (one-way) | Energy (pJ/bit) | Multicast | Source |
|---|---|---|---|---|
| Electrical 800G | 800 Gbps | 10.0 (assumed) | No | TPU v5p ICI inference |
| **EVK50** | 800 Gbps | 4.6 | Yes | SC25 demo, Dec 2025 |
| **EVK100** | 3.2 Tbps | **1.9** | Yes | ISES Korea 2025 paper |
| **L200** | 16 Tbps | 4.5 | Yes | Product page (32 Tbps total) |
| **L200X** | 32 Tbps | 4.8 | Yes | Product page (64 Tbps total) |
| **M1000** | 57 Tbps | 4.0 | Yes + OCS | Press release (114 Tbps total) |

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
| Photonic switched (M1000) | 87 % energy ↓ | 90 % energy ↓ | **99.8 %** energy ↓ |

## Generated Outputs (results/)

```
cluster_study.csv          — 3 workloads × 3 scenarios × 2 topologies
cluster_improvements.csv   — optical vs. mesh reduction per case
extended_cluster_study.csv — all 6 workloads × 3 scenarios
node_scaling.csv           — improvement vs. node count (2–128)
tpu_cube_case_study.csv    — 5 variants × 3 traffic cases
tpu_improvements.csv       — % reductions vs. electrical torus baseline
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

# Run all 52 tests
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
| TPU cube ICI study | `tpu_cube_case_study`, `tpu_improvement_frame`, notebook §TPU Cube |
| Extended: sensitivity + scaling | `bandwidth_sensitivity_network`, `energy_per_bit_sensitivity_network`, notebook §Sensitivity |
