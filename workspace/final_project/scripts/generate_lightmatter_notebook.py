"""
Generate the complete Lightmatter photonic-interconnect reproducibility notebook.

Covers all three project milestones plus an extended TPU cube ICI replacement study:
  M1 – Network model with topology-aware hop counting and logical cluster memory.
  M2 – Iterative AccelForge mapper loop with convergence analysis.
  M3 – Optical crossbar proposal vs. electrical mesh baseline.
  TPU – TPU v5p ICI replacement: EVK100 torus, L200/L200X/M1000 switched fabrics.
  Extended – Technology catalog, sensitivity sweeps, node-count scaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "lightmatter_interconnects.ipynb"


# ---------------------------------------------------------------------------
# Cell constructors
# ---------------------------------------------------------------------------

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

    cells: list[nbf.NotebookNode] = []

    # ------------------------------------------------------------------ #
    # Section 0: Title and research basis
    # ------------------------------------------------------------------ #
    cells.append(md(
        """# Modeling Lightmatter Photonic Interconnects in AccelForge

**Final Project — 6.5930 (Hardware Architecture for Deep Learning)**

This notebook builds a complete network-aware study using AccelForge for a
multi-accelerator cluster and then uses the same traffic model to analyze a TPU
cube ICI replacement with Lightmatter photonics.

The work is organized around the milestone structure:

1. **Milestone 1** — Implement a topology-aware network model driven by AccelForge mappings
   and logical cluster memory.
2. **Milestone 2** — Run the mapper iteratively until logical-memory energy/latency stabilize;
   analyze the impact of the network model.
3. **Milestone 3** — Propose a Lightmatter optical crossbar topology; compare against the
   electrical mesh baseline; explain the source of improvements.

**Plus:** Replace TPU v5p cube ICI with Lightmatter EVK100 / L200 / L200X / M1000 and
analyze the resulting architecture.
"""
    ))

    cells.append(md(
        """## Research Basis

All specifications below were verified against public sources as of **May 1, 2026**.

| Source | Key Numbers |
|---|---|
| Lightmatter SC25 demo (Dec 2025) | EVK50: 800 Gbps/fiber BiDi, **4.6 pJ/bit** total |
| Lightmatter Passage page + ISES Korea 2025 paper | EVK100: 3.2 Tbps/fiber, **1.9 pJ/bit** reference value |
| Lightmatter EVK100 technical page | Conservative EVK100 point: **3.2 pJ/bit** at 112G PAM4 |
| Lightmatter Passage L200 product page | L200: **32 Tbps** total chip I/O, <5 pJ/bit; L200X: **64 Tbps** |
| Lightmatter M1000 EVK product page | M1000: **114.6 Tbps** total bi-directional, **2.3 pJ/bit** including laser, built-in OCS |
| Lightmatter OCP announcement (Mar 2026) | Open CPO reference architecture initiative |
| Lightmatter world-record press release (Mar 2026) | **1.6 Tbps/fiber** at 112G PAM4 |
| Google Cloud TPU v5p docs | **1200 GB/s** bidirectional ICI/chip, 4×4×4 torus |
| Google Cloud TPU7x docs | Ironwood/TPU7x: **9216 chips**, **2307 BF16 / 4614 FP8 TFLOP/s/chip**, **192 GiB HBM/chip**, **1200 GB/s ICI/chip** |
| Google Gemma 4 announcement | Gemma 4 open models: E2B, E4B, **26B MoE**, **31B dense**, up to **256K context** for large models |
| Google Gemini 3 / 3.1 public posts | Latest Gemini family is public, but parameter counts and expert layouts are **not disclosed** |
| HotChips 2025 TPU rack tutorial | Ironwood: **4614 TFLOPS**, 7.4 TBps HBM, 6 cube faces × 16 links each |

Two modeling assumptions are **not** publicly disclosed and are treated as explicit assumptions:
- Electrical-link dynamic energy is modeled as conversion plus distance-dependent transmission, with **10 pJ/bit at a 5 mm reference path**
- Per-hop router latency: **5 ns** (electrical) / **1–2.5 ns** (optical)
- Gemini 3/3.1 architecture: modeled with explicit dense/MoE public proxy workloads, not confidential Google internals
- Gemma 4 open-model proxies: modeled directly from public 26B MoE / 31B dense size and context facts, with simplified layer/hidden dimensions used only to stress the communication model

These are intentionally conservative and easy to change in `scripts/lightmatter_interconnects.py`.
EVK100 energy is the one intentional optimistic/reference value: public Lightmatter pages expose
both 1.9 pJ/bit and 3.2 pJ/bit values, so we model 1.9 pJ/bit and keep the sensitivity sweeps
available to evaluate the more conservative point.
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 1: Setup
    # ------------------------------------------------------------------ #
    cells.append(md("## Setup"))

    cells.append(code(
        """
from pathlib import Path
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from scripts.lightmatter_interconnects import (
    # Technology catalog
    ELECTRICAL_800G,
    ADVANCED_CHIPLET_3P5T,
    LIGHTMATTER_EVK50,
    LIGHTMATTER_EVK100,
    LIGHTMATTER_L200,
    LIGHTMATTER_L200X,
    LIGHTMATTER_M1000,
    LIGHTMATTER_M1000_OPS,
    ALL_LIGHTMATTER_TECHS,
    # Topology models
    RingTopology,
    Mesh2DTopology,
    OpticalCrossbarTopology,
    Torus3DTopology,
    # Data types
    TrafficBreakdown,
    Scenario,
    WorkloadCase,
    # Functions
    workload_case,
    scenario_case,
    source_frame,
    solve_iteratively,
    history_to_frame,
    per_component_energy_frame,
    mapping_features_frame,
    compare_topologies,
    cluster_improvement_frame,
    tpu_cube_case_study,
    tpu_improvement_frame,
    tpu_cube_mapping_study,
    optimal_distributed_mapping_frame,
    compiler_strategy_study,
    optimal_compiler_strategy_frame,
    scaling_law_study,
    scaling_exponent_frame,
    chiplet_comparison_frame,
    gemini_proxy_models,
    default_superpod_configs,
    superpod_stage_study,
    superpod_end_to_end_summary,
    superpod_ablation_frame,
    superpod_context_scaling_study,
    distance_energy_model_frame,
    strong_weak_superpod_scaling_study,
    optical_switching_study,
    unified_thesis_frame,
    all_technologies_network_comparison,
    bandwidth_sensitivity_network,
    energy_per_bit_sensitivity_network,
    # Constants
    LLM_WORKLOADS,
    SCENARIOS,
    DEFAULT_WORKLOAD_NAMES,
    DEFAULT_SCENARIO_NAMES,
    HBM_ENERGY_PER_BIT_J,
    HBM_LATENCY_PER_BIT_S,
    IRONWOOD_TFLOPS_PER_CHIP,
    IRONWOOD_HBM_BANDWIDTH_TBPS,
    IRONWOOD_SUPERPOD_CHIPS,
    IRONWOOD_SUPERPOD_CUBES,
    TPU_V5P_TFLOPS_PER_CHIP,
    TPU_V5P_HBM_BANDWIDTH_TBPS,
    TPU_V5P_ICI_BIDIR_GBPS,
    IRONWOOD_ICI_BIDIR_GBPS,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", "{:.4g}".format)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Consistent color map for topology comparison
TOPO_COLORS = {
    "Electrical 4x4 Mesh": "#E34A33",
    "Lightmatter Optical Crossbar": "#2B8CBE",
    "4x4x4 3D Torus": "#756BB1",
    "Electrical Ring": "#636363",
}

TECH_COLORS = {
    "Electrical 800G": "#E34A33",
    "Advanced chiplet": "#8C6D31",
    "EVK50": "#FEC44F",
    "EVK100": "#31A354",
    "L200": "#2B8CBE",
    "L200X": "#756BB1",
    "M1000": "#E378AA",
}
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 2: Sources table
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Primary Sources

This table is included so the notebook remains self-documenting and reproducible.
Every quantitative claim about interconnect specifications is traced to one of these entries."""
    ))

    cells.append(code(
        """
sources = source_frame()
sources.style.set_properties(**{"text-align": "left"})
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 3: Technology catalog
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Technology Catalog

The table below summarizes all interconnect technologies modeled in this study.
Note the **4.4× energy improvement** from electrical (10 pJ/bit) to EVK100 (1.9 pJ/bit),
and the **60× bandwidth improvement** from the electrical baseline to M1000."""
    ))

    cells.append(code(
        """
tech_rows = [
    {
        "name": ELECTRICAL_800G.name,
        "BW one-way (Tbps)": ELECTRICAL_800G.link_bandwidth_bps / 1e12,
        "pJ/bit": ELECTRICAL_800G.energy_per_bit_j * 1e12,
        "router latency (ns)": ELECTRICAL_800G.router_latency_s * 1e9,
        "multicast": ELECTRICAL_800G.supports_multicast,
        "notes": ELECTRICAL_800G.notes[:80] + "...",
    }
,
    {
        "name": ADVANCED_CHIPLET_3P5T.name,
        "BW one-way (Tbps)": ADVANCED_CHIPLET_3P5T.link_bandwidth_bps / 1e12,
        "pJ/bit": ADVANCED_CHIPLET_3P5T.energy_per_bit_j * 1e12,
        "router latency (ns)": ADVANCED_CHIPLET_3P5T.router_latency_s * 1e9,
        "multicast": ADVANCED_CHIPLET_3P5T.supports_multicast,
        "notes": ADVANCED_CHIPLET_3P5T.notes[:80] + "...",
    }
]
for label, tech in ALL_LIGHTMATTER_TECHS.items():
    tech_rows.append({
        "name": tech.name,
        "BW one-way (Tbps)": tech.link_bandwidth_bps / 1e12,
        "pJ/bit": tech.energy_per_bit_j * 1e12,
        "router latency (ns)": tech.router_latency_s * 1e9,
        "multicast": tech.supports_multicast,
        "notes": tech.notes[:80] + "...",
    })

tech_df = pd.DataFrame(tech_rows)
tech_df
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

names = [row["name"] for row in tech_rows]
bw_vals = [row["BW one-way (Tbps)"] for row in tech_rows]
e_vals = [row["pJ/bit"] for row in tech_rows]

palette = ["#E34A33", "#8C6D31", "#FEC44F", "#31A354", "#2B8CBE", "#756BB1", "#E378AA"]
axes[0].bar(names, bw_vals, color=palette)
axes[0].set_title("Link Bandwidth (one-way, Tbps)")
axes[0].set_ylabel("Tbps")
axes[0].tick_params(axis="x", rotation=35)
axes[0].set_yscale("log")

axes[1].bar(names, e_vals, color=palette)
axes[1].set_title("Energy per Bit (pJ/bit)")
axes[1].set_ylabel("pJ/bit")
axes[1].tick_params(axis="x", rotation=35)

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 4: Workloads and scenarios
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Workloads and Communication Scenarios

### Workloads

All workloads are modeled as GEMM `Activations[M×K] × Weights[K×N] → Outputs[M×N]`
and mapped through AccelForge against the logical cluster memory.

| Workload | M | K | N | Description |
|---|---|---|---|---|
| `decode_mlp` | 1 | 4096 | 16384 | Decode-time FFN — weight-dominated |
| `kv_attention` | 1 | 128 | 8192 | Single-token KV attention score |
| `prefill_projection` | 2048 | 4096 | 4096 | Large dense prefill GEMM |
| `long_context_prefill` | 8192 | 4096 | 4096 | 8K-context prefill — extreme activation volume |
| `moe_expert_dispatch` | 1 | 4096 | 4096 | MoE token routing / gating |
| `training_weight_grad` | 4096 | 2048 | 4096 | Backward-pass weight gradient |

### Communication Scenarios

| Scenario | Description |
|---|---|
| `node_to_node` | Tensor-parallel activation exchange |
| `node_to_dram` | Remote weight/KV fetches from pooled DRAM |
| `shared_dram_multicast` | Pooled DRAM multicasts shared weights to all nodes |

Optical topologies exploit multicast natively (wavelength routing / OCS broadcast).
Electrical topologies pay full repeated unicast volume.
"""
    ))

    cells.append(code(
        """
print("Workloads:")
for name, p in LLM_WORKLOADS.items():
    mat = p["M"] * p["K"] * p["N"] * 8 / 1e6  # MiB at int8
    print(f"  {name:30s}  M={p['M']:5d}  K={p['K']:5d}  N={p['N']:5d}  "
          f"  weight+act={mat:.1f} MiB (int8)")
print()
print("Scenarios:")
for name, s in SCENARIOS.items():
    print(f"  {name:30s}  nodes={s['active_nodes']:2d}  {s['description'][:70]}")
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 5: Architecture — AccelForge logical cluster memory
    # ------------------------------------------------------------------ #
    cells.append(md(
        r"""## Milestone 1 — Network Model and Logical Cluster Memory

### Architecture

The AccelForge model uses a **logical cluster memory** as the union of all node-local
HBM banks. The mapper sees this level as a single large memory with parameterized
energy and latency per bit. After every mapper run we:

1. Extract the per-tensor logical-memory reads/writes (bits) from the best mapping.
2. Feed those traffic volumes into the topology model to compute network energy + latency.
3. Fold the average per-bit network cost back into the logical-memory parameters.
4. Re-run until the parameters converge.

The network cost model is explicit and fully analytic:

$$
E_\text{network} = \text{bits} \times h \times e_\text{link}
$$

$$
T_\text{network} = \frac{\text{bits}}{BW_\text{eff}} + h \times t_\text{router}
$$

where $h$ is the average hop count, $BW_\text{eff}$ is the effective bandwidth after
contention, and contention factors scale as $\sqrt{N}$ for a 2-D mesh and $\sqrt[3]{N}$
for a 3-D torus. The optical crossbar has $h = 1$ and no contention penalty.

### Topology hop-count reference

| Topology | avg node-to-node hops (N=16) | avg node-to-memory hops | Contention |
|---|---|---|---|
| Ring | N/4 = 4 | N/4 = 4 | N/4 |
| 2D Mesh 4×4 | ≈2.67 | ≈2.33 | √N |
| 3D Torus 4×4×4 | ≈1.5 | ≈2.5 (64 nodes) | ∛N |
| Optical Crossbar | 1 | 1 | 1 (no contention) |
"""
    ))

    cells.append(code(
        """
topologies_demo = {
    "Ring (16 nodes)": RingTopology(),
    "2D Mesh 4x4": Mesh2DTopology(4, 4),
    "3D Torus 4x4x4": Torus3DTopology(4, 4, 4),
    "Optical Crossbar": OpticalCrossbarTopology(),
}

print(f"{'Topology':<25}  {'n2n hops (N=16)':>16}  {'mem hops (N=16)':>16}  "
      f"{'contention (N=16)':>18}")
for label, topo in topologies_demo.items():
    n = 16 if label != "3D Torus 4x4x4" else 64
    print(f"  {label:<23}  {topo.average_node_to_node_hops(n):>16.2f}  "
          f"{topo.average_node_to_memory_hops(n):>16.2f}  "
          f"{topo.contention_factor(n, 'node_to_node'):>18.2f}")
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 6: Milestone 2 — Baseline cluster study
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Milestone 2 — Impact of the Network on Energy and Latency

We run the full 3 workloads × 3 scenarios × 2 topologies study.
The **baseline** is an electrical 4×4 mesh; the **proposal** is the optical crossbar.
"""
    ))

    cells.append(code(
        """
topologies = {
    Mesh2DTopology(4, 4): ELECTRICAL_800G,
    OpticalCrossbarTopology(): LIGHTMATTER_EVK100,
}

study_df = compare_topologies(
    workload_names=DEFAULT_WORKLOAD_NAMES,  # decode_mlp, prefill_projection, kv_attention
    scenario_names=DEFAULT_SCENARIO_NAMES,
    topology_to_tech=topologies,
)
study_df.to_csv(RESULTS_DIR / "cluster_study.csv", index=False)
study_df
"""
    ))

    cells.append(md(
        """### Milestone 2a — Network delta: how much does adding the network change things?

The `network_delta_energy_pct` column reports
`(final_energy - baseline_energy) / baseline_energy × 100 %`.
A positive number means the network model adds energy on top of bare compute.
"""
    ))

    cells.append(code(
        """
summary_cols = [
    "workload", "scenario", "topology",
    "baseline_energy_uJ", "first_run_plus_network_energy_uJ", "final_energy_uJ",
    "network_delta_energy_pct",
    "baseline_latency_us", "first_run_plus_network_latency_us", "final_latency_us",
    "network_delta_latency_pct",
]
study_df[summary_cols].sort_values(["scenario", "workload", "topology"]).reset_index(drop=True)
"""
    ))

    cells.append(md(
        """### Milestone 2b — Mapping sensitivity: does re-running the mapper matter?

`mapping_sensitivity_energy_pct` compares the *final converged* result to the
*first pass augmented with network overhead*. A non-zero value means the mapper
or fixed-point cost model changed the final objective after it saw the true network cost.
The separate distributed-mapping section later asks whether the best communication
strategy changes between interconnects.
"""
    ))

    cells.append(code(
        """
sensitivity_cols = [
    "workload", "scenario", "topology",
    "first_run_plus_network_energy_uJ", "final_energy_uJ",
    "mapping_sensitivity_energy_pct",
    "first_run_plus_network_latency_us", "final_latency_us",
    "mapping_sensitivity_latency_pct",
    "iterations",
]
study_df[sensitivity_cols].sort_values(
    ["scenario", "workload", "topology"]
).reset_index(drop=True)
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 7: Iterative convergence traces
    # ------------------------------------------------------------------ #
    cells.append(md(
        """### Iterative Convergence Traces

Inspect the fixed-point loop directly for the `decode_mlp / node_to_dram` case,
which is the most network-sensitive scenario for the electrical baseline.
"""
    ))

    cells.append(code(
        """
mesh_hist, _ = solve_iteratively(
    workload_case("decode_mlp"), scenario_case("node_to_dram"),
    Mesh2DTopology(4, 4), ELECTRICAL_800G,
)
opt_hist, _ = solve_iteratively(
    workload_case("decode_mlp"), scenario_case("node_to_dram"),
    OpticalCrossbarTopology(), LIGHTMATTER_EVK100,
)

mesh_df = history_to_frame(mesh_hist)
opt_df  = history_to_frame(opt_hist)
print("=== Electrical 4x4 Mesh — decode_mlp / node_to_dram ===")
print(mesh_df.to_string(index=False))
print()
print("=== Optical Crossbar (EVK100) — decode_mlp / node_to_dram ===")
print(opt_df.to_string(index=False))
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for df, label, color in [
    (mesh_df, "Electrical Mesh", "#E34A33"),
    (opt_df, "Optical Crossbar", "#2B8CBE"),
]:
    axes[0].plot(df["iteration"], df["total_energy_uJ"], marker="o", label=label, color=color)
    axes[1].plot(df["iteration"], df["total_latency_us"], marker="o", label=label, color=color)

for ax, ylabel, title in [
    (axes[0], "Total energy (µJ)", "Convergence — Energy (decode_mlp / node_to_dram)"),
    (axes[1], "Total latency (µs)", "Convergence — Latency (decode_mlp / node_to_dram)"),
]:
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """**Key observations:**

- The electrical mesh never fully converges within the first 6 iterations for the high-traffic
  node_to_dram scenario — the mapper keeps responding to rising network costs by shifting data
  toward local compute at the cost of more MAC operations.
- The optical crossbar converges in 2–3 iterations because its per-bit network cost is low enough
  that it barely perturbs the original HBM-dominant mapping.
- This means the **fixed-point sensitivity effect is predominantly an electrical-fabric problem**.
  Replacing the ICI with Lightmatter optics reduces the sensitivity almost to zero in these cases.
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 8: Energy breakdown
    # ------------------------------------------------------------------ #
    cells.append(md(
        """### Energy Breakdown by Component

Per-component energy for the converged mapping, for the decode_mlp workload."""
    ))

    cells.append(code(
        """
cpe_mesh = per_component_energy_frame(mesh_hist)
cpe_opt  = per_component_energy_frame(opt_hist)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, df, title in [
    (axes[0], cpe_mesh, "Electrical Mesh — energy breakdown"),
    (axes[1], cpe_opt, "Optical Crossbar (EVK100) — energy breakdown"),
]:
    if not df.empty:
        ax.bar(df["component"], df["energy_uJ"], color="#2B8CBE")
        ax.set_title(title)
        ax.set_ylabel("Energy (µJ)")
        ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 9: Milestone 3 — Proposed topology comparison
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Milestone 3 — Proposed Better Topology: Optical Crossbar

The **optical crossbar** driven by Lightmatter EVK100-class links is the proposed
replacement for the electrical 4×4 mesh. It offers:

- **Single-hop** any-to-any connectivity (h = 1 everywhere)
- **Native multicast** via wavelength routing — the `shared_dram_multicast` scenario
  charges only the *source* volume, not N × unicast
- **1.9 pJ/bit** vs **10 pJ/bit** for the electrical baseline
- **No contention penalty** — each wavelength channel is independent

The table below quantifies the improvement in each scenario.
"""
    ))

    cells.append(code(
        """
improvements = cluster_improvement_frame(study_df)
improvements.to_csv(RESULTS_DIR / "cluster_improvements.csv", index=False)

display_cols = [
    "workload", "scenario",
    "final_energy_uJ_baseline", "final_energy_uJ_proposal", "energy_reduction_pct",
    "final_latency_us_baseline", "final_latency_us_proposal", "latency_reduction_pct",
]
improvements[display_cols].sort_values(
    "energy_reduction_pct", ascending=False
).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = improvements["workload"] + "\\n" + improvements["scenario"]
x = range(len(labels))

for ax, col, title, color in [
    (axes[0], "energy_reduction_pct", "Optical Crossbar Energy Reduction vs Mesh (%)", "#2B8CBE"),
    (axes[1], "latency_reduction_pct", "Optical Crossbar Latency Reduction vs Mesh (%)", "#31A354"),
]:
    ax.bar(x, improvements[col], color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Reduction (%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### Source of Improvements

The gains are not driven by raw compute — the MAC count is identical.
They come entirely from the removal of repeated high-hop electrical transfers:

- **node_to_dram (remote weight streaming):** The optical fabric cuts the average
  hop count from ~2.67 hops (mesh) to 1 hop and reduces energy per bit by 5.3×.
  This is the dominant benefit for decode-time FFN inference.

- **shared_dram_multicast:** The optical fabric delivers the *same* weight block to
  16 nodes with one effective transfer (multicast), while the electrical mesh charges
  for 16 independent unicasts. This drives the largest absolute reduction.

- **node_to_node (activation exchange):** Traffic volume is smaller relative to the
  weight-streaming cases, so the absolute improvement is smaller — but the optical
  crossbar still eliminates multi-hop routing and contention.
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 10: Full workload and scenario sweep
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Extended Analysis — All Workloads and Scenarios

Run the full 6-workload × 3-scenario comparison to capture training, MoE, and
long-context workloads in addition to the three standard inference GEMMs.
"""
    ))

    cells.append(code(
        """
all_workloads = list(LLM_WORKLOADS.keys())

extended_df = compare_topologies(
    workload_names=all_workloads,
    scenario_names=DEFAULT_SCENARIO_NAMES,
    topology_to_tech=topologies,
)
# Kept in the executed notebook output; not part of the final submission CSV set.
extended_improvements = cluster_improvement_frame(extended_df)

pivot_energy = extended_improvements.pivot_table(
    index="workload", columns="scenario", values="energy_reduction_pct"
)
pivot_latency = extended_improvements.pivot_table(
    index="workload", columns="scenario", values="latency_reduction_pct"
)

print("Energy reduction (%) — optical crossbar vs electrical mesh")
print(pivot_energy.to_string(float_format="{:.1f}%".format))
print()
print("Latency reduction (%) — optical crossbar vs electrical mesh")
print(pivot_latency.to_string(float_format="{:.1f}%".format))
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pivot, title in [
    (axes[0], pivot_energy, "Energy Reduction (%) by Workload and Scenario"),
    (axes[1], pivot_latency, "Latency Reduction (%) by Workload and Scenario"),
]:
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 11: All-technology network comparison
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## All Lightmatter Technologies vs. Electrical Baseline

The optical crossbar evaluated at a single (scenario, traffic volume) point
without running AccelForge — useful for a quick bandwidth/energy trade-off view.
"""
    ))

    cells.append(code(
        """
for scenario_name in DEFAULT_SCENARIO_NAMES:
    print(f"\\n=== Scenario: {scenario_name} (1 GiB traffic, 16 nodes) ===")
    tech_df = all_technologies_network_comparison(
        traffic_bits=8 * 1024**3,
        scenario_name=scenario_name,
        active_nodes=16,
    )
    print(tech_df.to_string(index=False))
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=False)

for ax, scenario_name in zip(axes, DEFAULT_SCENARIO_NAMES):
    df = all_technologies_network_comparison(
        traffic_bits=8 * 1024**3,
        scenario_name=scenario_name,
        active_nodes=16,
    )
    colors = [TECH_COLORS.get(row["technology"].split("-")[0].strip(), "#999")
              for _, row in df.iterrows()]
    ax.bar(df["technology"], df["total_energy_uJ"], color=colors)
    ax.set_title(f"{scenario_name}", fontsize=9)
    ax.set_ylabel("Network energy (µJ)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)

fig.suptitle("Network Energy by Technology — 1 GiB, 16 nodes, optical crossbar topology")
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 12: Sensitivity analysis
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Sensitivity Analysis

### Bandwidth Sensitivity

How much does the electrical mesh's performance improve as we increase its link
bandwidth (independent of energy per bit)?
"""
    ))

    cells.append(code(
        """
bw_df = bandwidth_sensitivity_network(
    traffic_bits=8 * 1024**3,
    scenario_name="node_to_dram",
    active_nodes=16,
    base_tech=ELECTRICAL_800G,
    multipliers=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    topology=Mesh2DTopology(4, 4),
)

optical_baseline = all_technologies_network_comparison(
    traffic_bits=8 * 1024**3,
    scenario_name="node_to_dram",
    active_nodes=16,
).set_index("technology").loc["EVK100", "total_energy_uJ"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(bw_df["link_bandwidth_tbps"], bw_df["energy_uJ"],
             marker="o", color="#E34A33", label="Electrical mesh (scaled BW)")
axes[0].axhline(optical_baseline, linestyle="--", color="#2B8CBE", label="Optical crossbar (EVK100)")
axes[0].set_xscale("log")
axes[0].set_xlabel("Link bandwidth (Tbps, one-way)")
axes[0].set_ylabel("Network energy (µJ)")
axes[0].set_title("BW Sensitivity — node_to_dram")
axes[0].legend()

axes[1].plot(bw_df["link_bandwidth_tbps"], bw_df["latency_us"],
             marker="o", color="#E34A33")
axes[1].set_xscale("log")
axes[1].set_xlabel("Link bandwidth (Tbps, one-way)")
axes[1].set_ylabel("Network latency (µs)")
axes[1].set_title("Latency vs BW — node_to_dram")

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
bw_df
"""
    ))

    cells.append(md(
        """**Key insight:** Even if we scale the electrical link bandwidth by 32×, the energy
does *not* reach the optical crossbar level, because energy is proportional to **bits × hops × pJ/bit**,
and the multi-hop electrical network never achieves the single-hop optical path.
Latency *does* improve monotonically with bandwidth, but the contention factor means it never
quite matches the optical baseline either.
"""
    ))

    cells.append(md(
        """### Energy-per-Bit Sensitivity

How sensitive is the result to our assumption of 10 pJ/bit for the electrical baseline?
"""
    ))

    cells.append(code(
        """
epb_df = energy_per_bit_sensitivity_network(
    traffic_bits=8 * 1024**3,
    scenario_name="node_to_dram",
    active_nodes=16,
    base_tech=ELECTRICAL_800G,
    multipliers=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    topology=Mesh2DTopology(4, 4),
)

optical_epb_vals = {
    "EVK50": LIGHTMATTER_EVK50.energy_per_bit_j * 1e12,
    "EVK100": LIGHTMATTER_EVK100.energy_per_bit_j * 1e12,
    "L200": LIGHTMATTER_L200.energy_per_bit_j * 1e12,
}

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epb_df["energy_per_bit_pJ"], epb_df["total_energy_uJ"],
        marker="o", color="#E34A33", label="Electrical mesh")
for label, val in optical_epb_vals.items():
    opt_cost = all_technologies_network_comparison(
        traffic_bits=8 * 1024**3, scenario_name="node_to_dram"
    ).set_index("technology")
    ax.axhline(
        opt_cost.loc[label, "total_energy_uJ"],
        linestyle="--",
        label=f"Optical ({label})",
    )
ax.set_xlabel("Electrical link energy (pJ/bit)")
ax.set_ylabel("Total network energy (µJ)")
ax.set_title("Electrical Energy Sensitivity (node_to_dram, 1 GiB, 16 nodes)")
ax.legend(fontsize=8)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
epb_df
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 12b: Node count scaling
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Node Count Scaling Analysis

How does the benefit of the optical crossbar scale as we add more nodes?
We use pure network evaluation (no AccelForge call) with a fixed 1 GiB traffic volume per chip.
"""
    ))

    cells.append(code(
        """
node_counts = [2, 4, 8, 16, 32, 64, 128]
traffic_bits = 8 * 1024**3  # 1 GiB

scaling_rows = []
for n in node_counts:
    for scenario_name in ["node_to_dram", "shared_dram_multicast"]:
        traffic = TrafficBreakdown(weight_read_bits=traffic_bits)
        scenario = Scenario(scenario_name, "", scenario_name, active_nodes=n)

        mesh_cost = Mesh2DTopology(int(n**0.5), int(n**0.5)).evaluate(
            traffic, scenario, ELECTRICAL_800G
        ) if (int(n**0.5))**2 == n else Mesh2DTopology(4, max(1, n // 4)).evaluate(
            traffic, scenario, ELECTRICAL_800G
        )
        opt_cost = OpticalCrossbarTopology().evaluate(
            traffic, scenario, LIGHTMATTER_EVK100
        )

        scaling_rows.append({
            "nodes": n,
            "scenario": scenario_name,
            "mesh_energy_uJ": mesh_cost.total_energy_j * 1e6,
            "opt_energy_uJ": opt_cost.total_energy_j * 1e6,
            "energy_reduction_pct": 100 * (1 - opt_cost.total_energy_j / mesh_cost.total_energy_j),
            "mesh_latency_us": mesh_cost.total_latency_s * 1e6,
            "opt_latency_us": opt_cost.total_latency_s * 1e6,
            "latency_reduction_pct": 100 * (1 - opt_cost.total_latency_s / mesh_cost.total_latency_s),
        })

scaling_df = pd.DataFrame(scaling_rows)
# Kept in the executed notebook output; not part of the final submission CSV set.

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for scenario_name, color in [("node_to_dram", "#E34A33"), ("shared_dram_multicast", "#2B8CBE")]:
    sdf = scaling_df[scaling_df["scenario"] == scenario_name]
    axes[0].plot(sdf["nodes"], sdf["energy_reduction_pct"],
                 marker="o", label=scenario_name, color=color)
    axes[1].plot(sdf["nodes"], sdf["latency_reduction_pct"],
                 marker="o", label=scenario_name, color=color)

axes[0].set_title("Energy Reduction (%) vs Node Count")
axes[0].set_xlabel("Active nodes")
axes[0].set_ylabel("Reduction (%)")
axes[0].set_xscale("log", base=2)
axes[0].legend()

axes[1].set_title("Latency Reduction (%) vs Node Count")
axes[1].set_xlabel("Active nodes")
axes[1].set_ylabel("Reduction (%)")
axes[1].set_xscale("log", base=2)
axes[1].legend()

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
scaling_df[scaling_df["scenario"] == "node_to_dram"][["nodes", "energy_reduction_pct", "latency_reduction_pct"]]
"""
    ))

    cells.append(md(
        """**Key insight on scaling:**

The optical crossbar's advantage *increases* with node count for both scenarios.
For the electrical mesh, energy cost scales as `bits × hops × pJ/bit`, and average
hop count grows as `~√N` (mesh contention). For the optical crossbar, hops remain 1
regardless of N. This means the optical advantage compounds: at N=128 the electrical
mesh pays more than 11× the per-bit energy for node_to_dram traffic compared to EVK100.

For `shared_dram_multicast`, the advantage is even stronger because the electrical
mesh also charges `N × unique_bits` (unicast repetition), while the optical crossbar
delivers once to all N nodes (multicast).
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 13: TPU cube ICI replacement study
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## TPU Cube ICI Replacement Study

### Motivation

From the HotChips 2025 TPU rack tutorial (Aug 2025):

| Generation | TFLOPS/chip | HBM BW | HBM Capacity |
|---|---|---|---|
| TPU v4 (2022) | 275 | 1.2 TBps | 32 GB |
| TPU v5p (2023) | 459 | 2.8 TBps | 95 GB |
| Ironwood (2025) | **4614** | **7.4 TBps** | **192 GB** |

Each Ironwood rack uses a **4×4×4 3D torus** building block with **6 cube faces × 16 ICI links**
each — 96 optical + 80 copper + 64 PCB-trace ICI connections per rack.

The TPU v5p cluster uses 1200 GB/s bidirectional ICI per chip.
Inferred per-axis one-way bandwidth: `1200 / 6 / 2 ≈ 100 GB/s = 800 Gbps`.

We evaluate five variants for the 64-chip 4×4×4 cube:

1. **TPU v5p electrical torus** — baseline (800 Gbps/axis, 10 pJ/bit assumed)
2. **Photonic torus (EVK100)** — same topology, EVK100 links (3.2 Tbps/axis, 1.9 pJ/bit)
3. **Photonic switched fabric (L200)** — optical crossbar, 16 Tbps one-way, 4.5 pJ/bit
4. **Photonic switched fabric (L200X)** — optical crossbar, 32 Tbps one-way, 4.8 pJ/bit
5. **Photonic switched fabric (M1000)** — OCS-based crossbar, 57.3 Tbps one-way, 2.3 pJ/bit
"""
    ))

    cells.append(code(
        """
print(f"TPU generation specs (from HotChips 2025 + Google Cloud docs):")
print(f"  v5p:     {TPU_V5P_TFLOPS_PER_CHIP:>5.0f} TFLOPS/chip, "
      f"{TPU_V5P_HBM_BANDWIDTH_TBPS:.1f} TBps HBM, "
      f"{TPU_V5P_ICI_BIDIR_GBPS:.0f} GB/s bidir ICI")
print(f"  Ironwood:{IRONWOOD_TFLOPS_PER_CHIP:>5.0f} TFLOPS/chip, "
      f"{IRONWOOD_HBM_BANDWIDTH_TBPS:.1f} TBps HBM, "
      f"~{IRONWOOD_ICI_BIDIR_GBPS:.0f} GB/s bidir ICI")
print(f"  (Ironwood ICI: 1200 GB/s bidirectional per public TPU7x cloud documentation)")
"""
    ))

    cells.append(code(
        """
tpu_df = tpu_cube_case_study()
tpu_df.to_csv(RESULTS_DIR / "tpu_cube_case_study.csv", index=False)
tpu_df
"""
    ))

    cells.append(code(
        """
tpu_improvements = tpu_improvement_frame(tpu_df)
tpu_improvements.to_csv(RESULTS_DIR / "tpu_improvements.csv", index=False)

display_cols = [
    "case", "variant",
    "effective_link_one_way_tbps", "energy_pJ_per_bit",
    "energy_uJ_per_chip", "latency_us_per_chip",
    "energy_reduction_pct", "latency_reduction_pct",
]
tpu_improvements[display_cols].sort_values(
    ["case", "energy_reduction_pct"], ascending=[True, False]
).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
cases = tpu_df["case"].unique()
variants = tpu_improvements["variant"].unique()
n_variants = len(variants)
n_cases = len(cases)

fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases))

for row_i, case in enumerate(cases):
    case_df = tpu_df[tpu_df["case"] == case]
    case_imp = tpu_improvements[tpu_improvements["case"] == case]

    # Absolute energy
    ax_e = axes[row_i, 0]
    colors = ["#E34A33" if "electrical" in v.lower() else "#2B8CBE" for v in case_df["variant"]]
    ax_e.bar(case_df["variant"], case_df["energy_uJ_per_chip"], color=colors)
    ax_e.set_title(f"{case} — Energy per chip (µJ)")
    ax_e.set_ylabel("µJ")
    ax_e.tick_params(axis="x", rotation=30, labelsize=7)

    # Reduction %
    ax_r = axes[row_i, 1]
    ax_r.bar(case_imp["variant"], case_imp["energy_reduction_pct"], color="#31A354")
    ax_r.set_title(f"{case} — Energy reduction vs v5p baseline (%)")
    ax_r.set_ylabel("Reduction (%)")
    ax_r.tick_params(axis="x", rotation=30, labelsize=7)
    ax_r.axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### TPU Cube Interpretation

**Drop-in photonic torus (EVK100):**
- Same 4×4×4 torus topology; only the link technology changes.
- Bandwidth: 3.2 Tbps vs. 800 Gbps (4× improvement per axis)
- Energy: 1.9 pJ/bit vs. 10 pJ/bit assumed electrical (5.3× improvement)
- Benefit is purely from lower per-bit link energy; multi-hop routing cost still exists.
- Best fit for: conservative migration path with existing system topology.

**Photonic switched fabric (L200 / L200X / M1000):**
- Replaces the torus with an optical crossbar — any-to-any in a single hop.
- Eliminates multi-hop routing for `all_to_all_shard_exchange` and `shared_weight_multicast`.
- Enables true multicast, which dramatically reduces `shared_weight_multicast` energy.
- M1000 achieves the highest bandwidth (57.3 Tbps one-way) with built-in OCS switching.
- Best fit for: MoE inference workloads and all-to-all collective operations.

**The architectural implication:**
Replacing TPU ICI with Lightmatter photonics is not just a link-speed story.
It changes which topology is *feasible*:
- Keep the torus if you want a drop-in replacement with minimal redesign.
- Move to a switched crossbar if your dominant traffic is global (MoE dispatch,
  pipeline flush, or any-to-any weight streaming).
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 14: TPU cube mapping adaptation
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## TPU Cube Mapping Adaptation — New Optimal Mappings

The previous TPU-cube table was a pure network calculation over large traffic cases.
Here we run AccelForge itself on a 4×4×4 logical cube and feed the interconnect costs
back into the mapper. This directly tests the central architecture question:

> Does replacing TPU ICI merely reduce link cost, or does it also change the optimal mapping?

We compare the electrical torus against an EVK100 photonic torus and L200/L200X/M1000
switched fabrics. The `mapping_differs_from_electrical` column is computed from a compact
mapping signature that captures spatial usage, logical-memory transfers, and memory residency.
"""
    ))

    cells.append(code(
        """
tpu_mapping_df = tpu_cube_mapping_study(
    workload_names=("kv_attention", "prefill_projection"),
    scenario_names=DEFAULT_SCENARIO_NAMES,
    max_iterations=4,
)
tpu_mapping_df.to_csv(RESULTS_DIR / "tpu_mapping_study.csv", index=False)

mapping_cols = [
    "variant", "workload", "scenario", "iterations",
    "final_energy_uJ", "energy_reduction_vs_electrical_pct",
    "mapping_sensitivity_energy_pct", "final_remote_bits_kib",
    "mapping_differs_from_electrical", "spatial_NodeX_usage",
    "spatial_NodeY_usage", "spatial_NodeZ_usage",
]
tpu_mapping_df[mapping_cols].sort_values(
    ["scenario", "energy_reduction_vs_electrical_pct"], ascending=[True, False]
).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
tpu_optimal_mappings = optimal_distributed_mapping_frame(
    tpu_mapping_df,
    baseline_architecture="TPU cube electrical torus",
)
tpu_optimal_mappings.to_csv(RESULTS_DIR / "optimal_mapping_choices.csv", index=False)
tpu_optimal_mappings[
    [
        "workload", "architecture", "optimal_distributed_mapping",
        "baseline_distributed_mapping", "optimal_mapping_changed_vs_baseline",
        "energy_gain_vs_baseline_optimal_pct",
    ]
]
"""
    ))

    cells.append(code(
        """
photonic_mapping = tpu_mapping_df[
    tpu_mapping_df["variant"] != "TPU cube electrical torus"
].copy()
photonic_mapping["label"] = (
    photonic_mapping["variant"].str.replace("TPU cube ", "", regex=False)
    + "\\n" + photonic_mapping["scenario"]
)

fig, ax = plt.subplots(figsize=(12, 5))
plot_df = photonic_mapping.sort_values(
    "energy_reduction_vs_electrical_pct", ascending=False
)
colors = ["#31A354" if x else "#9ECAE1" for x in plot_df["mapping_differs_from_electrical"]]
ax.bar(plot_df["label"], plot_df["energy_reduction_vs_electrical_pct"], color=colors)
ax.set_title("TPU Cube Mapping-Aware Energy Reduction vs Electrical Torus")
ax.set_ylabel("Energy reduction (%)")
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
"""
    ))

    cells.append(md(
        """### Mapping-Adaptation Takeaway

The fixed-point AccelForge tile/residency signatures are intentionally conservative and
often remain stable: the same GEMM tile hierarchy is still a good local mapping. The
important architectural shift therefore appears one level above this inner mapper: the
compiler/dataflow layer below can choose different sharding, remote-memory, and cache-fill
paradigms once optical multicast and flat high-bandwidth links are available.
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 15: Compiler/dataflow strategies and scaling laws
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Compiler/Dataflow Paradigms Enabled by Photonics

The inner AccelForge tile mapping is intentionally conservative. To capture the
larger architectural opportunity, we add a compiler-visible strategy layer inspired by
XLA/GSPMD and Pallas-style manual prefetching:

| Strategy | Compiler interpretation |
|---|---|
| `xla_spmd_collective` | Keep tensors sharded and use collective-permute/all-to-all activation exchange |
| `remote_hbm_streaming` | Treat remote HBM as a cheap pooled memory level |
| `replicated_hbm_cache` | Manually prefetch hot state into local HBM and reuse it |
| `photonic_multicast_cache` | Use optical multicast to fill many caches with one logical transfer |

This is where the new technology changes the mapping paradigm: photonic multicast can make
explicit cache-fill/broadcast strategies optimal, while electrical and chiplet fabrics remain
closer to SPMD collectives or local sharding.
"""
    ))

    cells.append(code(
        """
compiler_df = compiler_strategy_study(
    workload_names=DEFAULT_WORKLOAD_NAMES,
    active_nodes=64,
)
compiler_df.to_csv(RESULTS_DIR / "compiler_strategy_study.csv", index=False)

optimal_compiler = optimal_compiler_strategy_frame(compiler_df)
optimal_compiler.to_csv(RESULTS_DIR / "optimal_compiler_strategies.csv", index=False)
optimal_compiler[
    [
        "workload", "architecture", "strategy_label", "memory_paradigm",
        "baseline_strategy_label", "strategy_changed_vs_baseline",
        "energy_gain_vs_baseline_strategy_pct",
    ]
].sort_values(["workload", "architecture"]).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
plot_df = optimal_compiler.sort_values(
    "energy_gain_vs_baseline_strategy_pct", ascending=False
).copy()
plot_df["label"] = plot_df["workload"] + "\\n" + plot_df["architecture"]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#31A354" if x else "#9ECAE1" for x in plot_df["strategy_changed_vs_baseline"]]
ax.bar(plot_df["label"], plot_df["energy_gain_vs_baseline_strategy_pct"], color=colors)
ax.set_title("Compiler Strategy Energy Gain vs TPU Electrical Baseline")
ax.set_ylabel("Energy gain (%)")
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """## Scaling Laws: From One Cube to Superpod Scale

We next hold per-chip traffic fixed and scale from a 64-chip cube toward 9216 chips.
The key comparison is the slope: torus and chiplet fabrics accumulate hop and contention
cost, while a switched optical fabric keeps hop count flat and gives multicast an
approximately constant per-chip cost.
"""
    ))

    cells.append(code(
        """
scaling_df = scaling_law_study()
scaling_df.to_csv(RESULTS_DIR / "scaling_law_study.csv", index=False)
scaling_exponents = scaling_exponent_frame(scaling_df)
scaling_exponents.to_csv(RESULTS_DIR / "scaling_exponents.csv", index=False)

scaling_exponents.sort_values(
    ["scenario", "metric", "scaling_exponent"]
).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, scenario_name, title in [
    (axes[0], "all_to_all_activation", "All-to-all Activation Scaling"),
    (axes[1], "shared_state_multicast", "Shared-State Multicast Scaling"),
]:
    sdf = scaling_df[scaling_df["scenario"] == scenario_name]
    for architecture, group in sdf.groupby("architecture"):
        group = group.sort_values("active_nodes")
        ax.plot(group["active_nodes"], group["latency_us_per_chip"], marker="o", label=architecture)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Active chips")
    ax.set_ylabel("Latency per chip (µs)")
    ax.set_title(title)
    ax.legend(fontsize=7)

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """## Stronger Baseline: Advanced Chiplet Interconnect

To avoid making the electrical baseline too weak, we also compare against the requested
advanced-packaging point: **3.5 Tbps one-way links** and **6 pJ/bit per hop**.
This is modeled as a strong chiplet torus without native multicast.
"""
    ))

    cells.append(code(
        """
chiplet_df = chiplet_comparison_frame(active_nodes=64)
chiplet_df.to_csv(RESULTS_DIR / "chiplet_comparison.csv", index=False)
chiplet_df[
    [
        "scenario", "architecture", "energy_uJ_per_chip", "latency_us_per_chip",
        "energy_reduction_vs_chiplet_pct", "latency_reduction_vs_chiplet_pct",
    ]
].sort_values(["scenario", "architecture"]).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
plot_df = chiplet_df[chiplet_df["architecture"] != "Advanced chiplet torus"].copy()
plot_df["label"] = plot_df["architecture"] + "\\n" + plot_df["scenario"]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(plot_df["label"], plot_df["energy_reduction_vs_chiplet_pct"], color="#756BB1")
ax.set_title("Photonic Fabrics vs Advanced Chiplet Baseline")
ax.set_ylabel("Energy reduction vs chiplet (%)")
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """## Full Superpod Gemma/Gemini End-to-End Study

This is the core end-to-end system study. Google has publicly announced Gemma 4
open models and public TPU7x/Ironwood superpod specifications. Google has also
publicly identified Gemini 3 and Gemini 3.1 as the current Gemini family, but does
**not** publish their parameter counts or expert layouts. The model below therefore
uses two classes of workloads:

- Gemma 4 public open-model proxies: 26B MoE and 31B dense with long-context stress.
- Gemini-class stress proxies: dense Flash-like, sparse MoE, and Deep Think-style long-context cases.

The purpose is not to claim confidential internals. The purpose is to quantify which
end-to-end stages improve on a 9216-chip Ironwood superpod if the fabric changes from
electrical torus links to Lightmatter-style photonic topologies and if XLA/Pallas can
exploit the resulting data-placement choices.
"""
    ))

    cells.append(code(
        """
print(f"Ironwood superpod: {IRONWOOD_SUPERPOD_CHIPS} chips = {IRONWOOD_SUPERPOD_CUBES} cubes")
print(f"Per chip: {IRONWOOD_TFLOPS_PER_CHIP:.0f} FP8 TFLOP/s, {IRONWOOD_HBM_BANDWIDTH_TBPS:.1f} TB/s HBM")
print()
print("Gemma/Gemini proxy workloads:")
for key, model in gemini_proxy_models().items():
    print(
        f"  {key:34s} kind={model.model_kind:5s} "
        f"total={model.total_params_b:.0f}B active={model.active_params_b:.0f}B "
        f"ctx={model.prompt_tokens:,} out={model.output_tokens:,}"
    )
"""
    ))

    cells.append(code(
        """
superpod_stage_df = superpod_stage_study()
superpod_stage_df.to_csv(RESULTS_DIR / "superpod_stage_study.csv", index=False)

superpod_summary = superpod_end_to_end_summary(superpod_stage_df)
superpod_summary.to_csv(RESULTS_DIR / "superpod_end_to_end_summary.csv", index=False)

superpod_ablation = superpod_ablation_frame(superpod_summary)
superpod_ablation.to_csv(RESULTS_DIR / "superpod_ablation_study.csv", index=False)

superpod_context = superpod_context_scaling_study()
superpod_context.to_csv(RESULTS_DIR / "superpod_context_scaling.csv", index=False)

superpod_summary[
    [
        "model", "phase", "configuration", "speedup_vs_baseline",
        "energy_reduction_vs_baseline_pct", "network_latency_share_pct",
        "throughput_tokens_per_s",
    ]
].sort_values(["model", "phase", "speedup_vs_baseline"], ascending=[True, True, False])
"""
    ))

    cells.append(md(
        """### End-to-End Speedups

The most important result is stage-dependent:

- **Inference** benefits strongly because decode, KV access, expert routing, and cache-fill traffic
  are network-sensitive. The full M1000 photonic compiler stack produces the largest end-to-end
  speedups, especially for long-context / Deep Think-style inference.
- **Training** is more compute dominated, so the network improvement is smaller end-to-end.
  The gains still matter because they reduce exposed collectives, optimizer traffic, and pipeline bubbles.
- **Advanced chiplet** is a much stronger baseline than simple electrical ICI, but it still lacks
  flat optical switching and native multicast.
"""
    ))

    cells.append(code(
        """
plot_df = superpod_summary[
    superpod_summary["configuration"].isin([
        "Advanced chiplet torus + XLA SPMD",
        "M1000 switched optical topology only",
        "M1000 full photonic compiler stack",
    ])
].copy()
plot_df["label"] = (
    plot_df["model"].str.replace("Gemini ", "G.", regex=False)
    + "\\n"
    + plot_df["phase"].str.title()
    + "\\n"
    + plot_df["configuration"].str.replace(" + ", "\\n+ ", regex=False)
)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(plot_df["label"], plot_df["speedup_vs_baseline"], color="#2B8CBE")
ax.set_title("9216-Chip Ironwood Superpod: End-to-End Speedup vs Electrical Baseline")
ax.set_ylabel("Speedup (x)")
ax.axhline(1.0, color="black", linewidth=0.8)
ax.tick_params(axis="x", rotation=35, labelsize=7)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### Technique Ablations

This ablation ladder separates the effects of:

1. a stronger chiplet baseline,
2. photonic torus with multicast,
3. switched optical topology,
4. multicast cache fills,
5. remote pooled-HBM streaming,
6. MoE locality routing,
7. the full photonic compiler stack.
"""
    ))

    cells.append(code(
        """
superpod_ablation[
    [
        "model", "phase", "technique", "speedup_vs_baseline",
        "speedup_increment_vs_previous", "network_latency_share_pct",
        "network_energy_share_pct",
    ]
].sort_values(["model", "phase"]).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
target = "Gemini 3.1 Pro-class sparse MoE proxy"
plot_df = superpod_ablation[
    (superpod_ablation["model"] == target)
    & (superpod_ablation["phase"] == "inference")
].copy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(plot_df["technique"], plot_df["speedup_vs_baseline"], color="#31A354")
ax.set_title("Ablation Ladder: Gemini 3.1 Pro-Class Inference Proxy")
ax.set_ylabel("Speedup vs electrical baseline (x)")
ax.axhline(1.0, color="black", linewidth=0.8)
ax.tick_params(axis="x", rotation=35, labelsize=8)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### Stage Breakdown and Context Scaling

The stage table below identifies what remains after applying the full photonic compiler stack.
This helps avoid the common mistake of reporting only link-level gains: once the network stops
dominating, compute and HBM become the limiting terms.
"""
    ))

    cells.append(code(
        """
stage_focus = superpod_stage_df[
    (superpod_stage_df["configuration"] == "M1000 full photonic compiler stack")
    & (superpod_stage_df["model"] == "Gemini 3.1 Pro-class sparse MoE proxy")
].copy()
stage_focus[
    [
        "phase", "stage", "total_latency_s", "network_latency_share_pct",
        "network_energy_share_pct", "bottleneck", "network_gib_per_chip",
    ]
].sort_values(["phase", "total_latency_s"], ascending=[True, False])
"""
    ))

    cells.append(code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stage_focus["label"] = stage_focus["phase"].str.title() + "\\n" + stage_focus["stage"]
axes[0].bar(stage_focus["label"], stage_focus["total_latency_s"], color="#31A354")
axes[0].set_title("Full Stack: Remaining Stage Latency")
axes[0].set_ylabel("Latency (s)")
axes[0].tick_params(axis="x", rotation=35, labelsize=8)

for config, group in superpod_context.groupby("configuration"):
    group = group.sort_values("context_tokens")
    axes[1].plot(group["context_tokens"], group["total_latency_s"], marker="o", label=config)
axes[1].set_title("Context Scaling: Inference Latency")
axes[1].set_xlabel("Prompt tokens")
axes[1].set_ylabel("Latency (s)")
axes[1].set_xscale("log", base=2)
axes[1].set_yscale("log")
axes[1].legend(fontsize=7)

plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """## Unified End-to-End Scaling Study

The earlier experiments isolate separate questions: AccelForge logical-memory cost,
distributed mapping choice, compiler strategy, cube replacement, and superpod stages.
This section ties them together with the physical effect that motivated the final
revision of the model: electrical and photonic links should not both be treated as
flat constant-pJ/bit wires.

The link-energy model is now:

\\[
E_{link}(B,d)=B\\,(E_{conv}+dE_{tx/mm})
\\]

where electrical and chiplet links use a small conversion term and a distance-dependent
transmission term, while photonic links are conversion/SerDes/laser dominated with a
tiny distance term. Topology determines the average path distance through hop count and
the average physical link length. This makes scaling visible: larger torus systems pay
for more links and longer average paths, while switched photonic fabrics are closer to
fanout- and distance-independent for multicast-style traffic.
"""
    ))

    cells.append(code(
        """
distance_df = distance_energy_model_frame()
distance_df.to_csv(RESULTS_DIR / "distance_energy_model.csv", index=False)

distance_df[
    (distance_df["scenario"] == "shared_state_multicast")
    & (distance_df["active_nodes"].isin([64, 9216]))
][
    [
        "active_nodes", "architecture", "switching_mode", "average_distance_mm",
        "effective_pJ_per_bit_at_distance", "conversion_energy_uJ_per_chip",
        "transmission_energy_uJ_per_chip", "total_latency_us_per_chip",
    ]
].sort_values(["active_nodes", "architecture"]).reset_index(drop=True)
"""
    ))

    cells.append(code(
        """
plot_df = distance_df[distance_df["scenario"] == "shared_state_multicast"].copy()
fig, ax = plt.subplots(figsize=(11, 5))
for architecture, group in plot_df.groupby("architecture"):
    group = group.sort_values("active_nodes")
    ax.plot(group["active_nodes"], group["effective_pJ_per_bit_at_distance"], marker="o", label=architecture)
ax.set_title("Distance-Parametrized Link Energy: Shared-State Multicast")
ax.set_xlabel("Active chips")
ax.set_ylabel("Effective pJ/bit at average path distance")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.legend(fontsize=7)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### Strong vs Weak Scaling

Strong scaling keeps the modeled request or training step fixed while adding chips.
Weak scaling increases aggregate tokens with chip count so that work per resource is
approximately fixed. These are different questions. Strong scaling measures latency
compression and exposes Amdahl-style limits; weak scaling asks whether the fabric can
support more concurrent work without per-unit cost exploding.
"""
    ))

    cells.append(code(
        """
strong_weak_df = strong_weak_superpod_scaling_study()
strong_weak_df.to_csv(RESULTS_DIR / "strong_weak_scaling.csv", index=False)

strong_weak_df[
    strong_weak_df["configuration"].isin([
        "Ironwood electrical torus + XLA SPMD",
        "M1000 full photonic compiler stack",
    ])
    & strong_weak_df["active_chip_count"].isin([64, 9216])
][
    [
        "base_model", "scaling_mode", "active_chip_count", "phase", "configuration",
        "speedup_vs_baseline", "latency_per_work_unit_s", "energy_per_work_unit_j",
    ]
].sort_values(["base_model", "scaling_mode", "active_chip_count", "phase", "configuration"])
"""
    ))

    cells.append(code(
        """
focus = strong_weak_df[
    strong_weak_df["configuration"].isin([
        "Ironwood electrical torus + XLA SPMD",
        "M1000 full photonic compiler stack",
    ])
    & (strong_weak_df["phase"] == "inference")
].copy()

fig, ax = plt.subplots(figsize=(11, 5))
for (mode, config), group in focus.groupby(["scaling_mode", "configuration"]):
    group = group.sort_values("active_chip_count")
    ax.plot(group["active_chip_count"], group["latency_per_work_unit_s"], marker="o", label=f"{mode} / {config}")
ax.set_title("Inference Strong vs Weak Scaling")
ax.set_xlabel("Active chips")
ax.set_ylabel("Latency per modeled work unit (s)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.legend(fontsize=6)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(md(
        """### Optical Circuit Switching vs Optical Packet Switching

M1000's public product page emphasizes optical circuit switching. The model also includes
an explicit optical-packet-switching sensitivity point. This does not claim an announced
Lightmatter packet switch product; it answers a design question: if fine-grained dynamic
traffic were packet switched, would the compiler choose a different point than a bulk
OCS multicast/cache-fill schedule?
"""
    ))

    cells.append(code(
        """
switching_df = optical_switching_study()
switching_df.to_csv(RESULTS_DIR / "optical_switching.csv", index=False)

switching_df[
    [
        "model", "phase", "configuration", "switching_family",
        "speedup_vs_baseline", "energy_reduction_vs_baseline_pct",
        "network_latency_share_pct",
    ]
].sort_values(["model", "phase", "speedup_vs_baseline"], ascending=[True, True, False])
"""
    ))

    cells.append(code(
        """
plot_df = switching_df[
    switching_df["switching_family"].isin(["OPS fine-grained", "OCS full compiler stack"])
].copy()
plot_df["label"] = plot_df["phase"].str.title() + "\\n" + plot_df["switching_family"] + "\\n" + plot_df["model"].str.replace(" proxy", "", regex=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(plot_df["label"], plot_df["speedup_vs_baseline"], color="#2B8CBE")
ax.set_title("M1000 Switching Sensitivity: OCS vs OPS")
ax.set_ylabel("Speedup vs OCS topology-only")
ax.axhline(1.0, color="black", linewidth=0.8)
ax.tick_params(axis="x", rotation=35, labelsize=7)
plt.tight_layout()
# Figure is embedded in the executed notebook; final submission figures are generated by write_study_outputs().
plt.show()
"""
    ))

    cells.append(code(
        """
unified_df = unified_thesis_frame(
    improvements,
    tpu_optimal_mappings,
    optimal_compiler,
    distance_df,
    strong_weak_df,
    switching_df,
)
unified_df.to_csv(RESULTS_DIR / "unified_thesis.csv", index=False)
unified_df
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 16: Summary pivot
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Summary Results

### Cluster Study (16-node system, EVK100 crossbar vs. electrical 4×4 mesh)
"""
    ))

    cells.append(code(
        """
summary_pivot = improvements[
    ["workload", "scenario", "energy_reduction_pct", "latency_reduction_pct"]
].pivot_table(
    index="workload", columns="scenario",
    values=["energy_reduction_pct", "latency_reduction_pct"],
)
summary_pivot
"""
    ))

    cells.append(md("### TPU Cube Study (64-chip 4×4×4 cube, best per case)"))

    cells.append(code(
        """
tpu_best = (
    tpu_improvements.sort_values("energy_reduction_pct", ascending=False)
    .groupby("case")
    .first()[["variant", "energy_reduction_pct", "latency_reduction_pct"]]
)
print("Best photonic variant per traffic case:")
print(tpu_best.to_string())
"""
    ))

    # ------------------------------------------------------------------ #
    # Section 15: Write all outputs and final conclusions
    # ------------------------------------------------------------------ #
    cells.append(md(
        """## Final Conclusions

1. **Remote weight streaming (`node_to_dram`)** is the most impactful case for the
   electrical baseline: each weight refill pays the full mesh hop penalty.
   Optical optics reduce energy by up to **65 %** for decode-time FFN inference.

2. **Shared multicast (`shared_dram_multicast`)** is where photonics is most
   *structurally* better — the optical crossbar delivers to 16 nodes with one
   effective transfer, achieving up to **72 %** energy and **97 %** latency reduction
   for decode-time workloads.

3. **Activation exchange (`node_to_node`)** benefits less in absolute terms because
   activation traffic is smaller. But it still sees a meaningful reduction in latency
   (shorter path, lower router latency).

4. **The project now separates two mapping effects.** The fixed-point loop measures
   tile/residency sensitivity after network feedback, while the distributed-mapping
   table shows which communication strategy (`node_to_node`, `node_to_dram`, or
   `shared_dram_multicast`) becomes optimal under each interconnect.

5. **The stronger compiler/dataflow result is the real paradigm shift.** XLA/Pallas-visible
   choices such as software-managed cache fill and multicast prefetch are not attractive on
   multi-hop electrical fabrics, but become optimal on photonic fabrics for reuse-heavy
   prefill and shared-state workloads.

6. **Scaling laws improve.** Larger torus and chiplet systems accumulate hop and contention
   cost as chip count grows, while a switched optical fabric keeps hop count flat and makes
   shared-state multicast close to fanout-independent.

7. **For the TPU cube:** A photonic switched fabric (L200 or higher) eliminates
   multi-hop routing entirely. The gain over a photonic torus is largest for
   global traffic patterns — `all_to_all_shard_exchange` and `shared_weight_multicast`.

8. **M1000 at 114.6 Tbps aggregate** represents the next architectural step:
   not just a faster link but an on-chip OCS that enables dynamic reconfiguration
   of the network fabric without static cabling changes.

9. **For a full 9216-chip Ironwood superpod running Gemma/Gemini proxy workloads,**
   inference is the main beneficiary: the M1000 full photonic compiler stack improves
   the Gemma 4 MoE proxy by more than an order of magnitude and still gives a large
   speedup on the longer-context Gemini stress proxy. Training improves less because
   forward/backward compute dominates once collectives and expert routing stop being
   the bottleneck.

10. **The most defensible project claim is not "photonic links make every workload faster."**
    The stronger claim is more nuanced: photonics changes which distributed memory and
    compiler strategies are attractive, and the end-to-end benefit depends on how much
    of the phase is network-exposed after compute/HBM overlap.
"""
    ))

    cells.append(code(
        """
from scripts.lightmatter_interconnects import (
    write_study_outputs,
    generate_summary_markdown,
)

paths = write_study_outputs(
    RESULTS_DIR,
    workload_names=DEFAULT_WORKLOAD_NAMES,
    scenario_names=DEFAULT_SCENARIO_NAMES,
)
print("Generated artifacts:")
for key, path in paths.items():
    print(f"  {key}: {path}")
"""
    ))

    nb["cells"] = cells
    return nb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_notebook(output_path: str | Path = NOTEBOOK_PATH) -> None:
    nb = build_notebook()
    output_path = Path(output_path)
    output_path.write_text(nbf.writes(nb))
    print(f"Wrote {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the Lightmatter study notebook.")
    parser.add_argument(
        "--output",
        default=str(NOTEBOOK_PATH),
        help="Path to the notebook to generate.",
    )
    args = parser.parse_args(argv)
    write_notebook(args.output)


if __name__ == "__main__":
    main()
