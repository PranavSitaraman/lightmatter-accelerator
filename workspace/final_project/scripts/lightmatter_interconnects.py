"""
Modeling Lightmatter photonic interconnects with AccelForge.

Covers all three project milestones:
  M1 – Topology-aware network model driven by AccelForge logical-memory mappings.
  M2 – Iterative mapper loop that augments logical-memory costs until they stabilize.
  M3 – Optical crossbar vs. electrical mesh comparison with full source attribution.

Plus: TPU cube ICI replacement study using EVK100, L200, L200X, and M1000 class links.

Public sources were checked / dated April 12, 2026.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

import accelforge as af


THIS_DIR = Path(__file__).resolve().parent
LAB_DIR = THIS_DIR.parent
ARCH_DIR = LAB_DIR / "arch"
WORKLOAD_DIR = LAB_DIR / "workload"

ARCH_PATH = ARCH_DIR / "tpu_lightmatter_cluster.yaml"
WORKLOAD_PATH = WORKLOAD_DIR / "single_matmul.yaml"

# ---------------------------------------------------------------------------
# Physical constants derived from public TPU v5p documentation
# ---------------------------------------------------------------------------

# HBM energy from Google TPU v5p white-paper table: 7.03 pJ/bit
HBM_ENERGY_PER_BIT_J: float = 7.03e-12

# HBM latency derived from 1200 GB/s bidirectional ICI bandwidth in TPU v5p docs.
# 1200 GB/s = 9600 Gb/s -> 1 bit takes 1/(8 * 1200e9) s.
HBM_LATENCY_PER_BIT_S: float = 1 / (8 * 1200e9)

# ---------------------------------------------------------------------------
# TPU generation reference specs (HotChips 2025, Google Cloud docs)
# ---------------------------------------------------------------------------

# TPU v4 (2022)
TPU_V4_TFLOPS_PER_CHIP = 275.0
TPU_V4_HBM_GB = 32
TPU_V4_HBM_BANDWIDTH_TBPS = 1.2
TPU_V4_ICI_BIDIR_GBPS = 600.0   # 3D torus; ~100 GB/s per axis per direction

# TPU v5p (2023)
TPU_V5P_TFLOPS_PER_CHIP = 459.0
TPU_V5P_HBM_GB = 95
TPU_V5P_HBM_BANDWIDTH_TBPS = 2.8
TPU_V5P_ICI_BIDIR_GBPS = 1200.0  # public: 1200 GB/s per chip

# Ironwood / TPU7x (2025/2026) – Google Cloud TPU7x docs + HotChips 2025
IRONWOOD_BF16_TFLOPS_PER_CHIP = 2307.0
IRONWOOD_FP8_TFLOPS_PER_CHIP = 4614.0
IRONWOOD_TFLOPS_PER_CHIP = IRONWOOD_FP8_TFLOPS_PER_CHIP
IRONWOOD_HBM_GB = 192
IRONWOOD_HBM_BANDWIDTH_TBPS = 7.4
IRONWOOD_SUPERPOD_CHIPS = 9216
IRONWOOD_CUBE_CHIPS = 64
IRONWOOD_SUPERPOD_CUBES = IRONWOOD_SUPERPOD_CHIPS // IRONWOOD_CUBE_CHIPS
IRONWOOD_ICI_BIDIR_GBPS = 1200.0  # public TPU7x table: GB/s per chip


# ---------------------------------------------------------------------------
# Research bibliography
# ---------------------------------------------------------------------------

RESEARCH_NOTES: dict[str, dict[str, str]] = {
    "lightmatter_evk50": {
        "date": "2025-12-16",
        "url": "https://lightmatter.co/resource/bobturnersc25/",
        "summary": (
            "Passage EVK50 (SC25 demo): 800 Gbps per fiber, 16-lambda BiDi NRZ 50G/lambda. "
            "4.6 pJ/bit total wall-plug (2.6 pJ/bit photonics + Guide + ~2 pJ/bit XSR SerDes). "
            "BER < 10^-9, >1 km reach, polarisation-insensitive, 25–105 °C range."
        ),
    },
    "lightmatter_evk100": {
        "date": "viewed 2026-04-12",
        "url": "https://lightmatter.co/products/passage/",
        "summary": (
            "Passage EVK100: 3.2 Tbps per fiber at 112G PAM4 (16 lambda). "
            "1.9 pJ/bit reference value on the Passage product family page / ISES Korea 2025; "
            "the EVK100 technical-spec page also lists a conservative 3.2 pJ/bit point. "
            "Monolithic PIC: MRM modulators, TIAs, Tx drivers on single chip."
        ),
    },
    "lightmatter_l200": {
        "date": "2025 product page",
        "url": "https://lightmatter.co/products/l200/",
        "summary": (
            "Passage L200: 32 Tbps optical I/O per chip (56G NRZ, 3D CoW, 320 SerDes). "
            "L200X: 64 Tbps at 112G PAM4. Both <5 pJ/bit. "
            "Bandwidth density >1.5 Tbps/mm. Targeted 2026."
        ),
    },
    "lightmatter_m1000": {
        "date": "viewed 2026-04-12",
        "url": "https://lightmatter.co/products/passage-m1000-evk/",
        "summary": (
            "Passage M1000 EVK: 114.6 Tbps total bi-directional bandwidth, "
            "2.3 pJ/bit system-level efficiency including laser, 1024 SerDes lanes, "
            "256 optical fibers, 4,000 mm² active interposer area, integrated solid-state OCS."
        ),
    },
    "lightmatter_ocp": {
        "date": "2026-03-16",
        "url": (
            "https://lightmatter.co/press-release/lightmatter-announces-reference-architecture"
            "-initiative-with-industry-leaders-in-the-open-compute-project-for-co-packaged-optics/"
        ),
        "summary": (
            "OCP CPO reference architecture initiative. Partners: Celestica, Corning, Dell, "
            "Flex, Foxconn, Hyve, Keysight, Qualcomm, Quanta. Standards: OIF, IEEE, OCP."
        ),
    },
    "lightmatter_record_1p6tbps": {
        "date": "2026-03",
        "url": "https://lightmatter.co/press-release/lightmatter-achieves-record-1-6-tbps-per-fiber-to-accelerate-ai-optical-interconnect/",
        "summary": (
            "World-record 1.6 Tbps per fiber (Qualcomm 112G PAM4 SerDes + Passage L-Series). "
            "8x more bandwidth per fiber than Near-Packaged Optics."
        ),
    },
    "lightmatter_tco": {
        "date": "2025-08",
        "url": "https://lightmatter.co/wp-content/uploads/2025/08/ISES-Korea-08-2025_Ritesh-Jain.pdf",
        "summary": (
            ">60 % TCO savings and 3–7x training speedup for 4.7T-param MoE "
            "(256 experts, 8 active) over 576 XPUs using L-Series vs electrical. "
            "Roadmap: pluggables 17–20 pJ/bit → 3D CPO as low as 3 pJ/bit."
        ),
    },
    "gemini_3": {
        "date": "2025-11-18",
        "url": "https://blog.google/products-and-platforms/products/gemini/gemini-3/",
        "summary": (
            "Google introduced Gemini 3 as its latest and most intelligent Gemini model "
            "family, with Gemini 3 Pro in preview across Gemini, AI Studio, Vertex AI, "
            "Search, and Antigravity. Public posts do not disclose parameter counts."
        ),
    },
    "gemini_3_1_release_notes": {
        "date": "viewed 2026-04-17",
        "url": "https://gemini.google/gb/release-notes/?hl=en-GB",
        "summary": (
            "Gemini app release notes state that Gemini 3.1 Pro is rolling out globally "
            "with higher limits for Google AI Pro and Ultra plans. The notes identify "
            "advanced reasoning but do not publish architecture or parameter sizes."
        ),
    },
    "gemini_3_flash": {
        "date": "2025-12-17",
        "url": "https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/",
        "summary": (
            "Gemini 3 Flash is described as a speed-optimized Gemini 3 model. Google "
            "reported more than 1T API tokens/day after Gemini 3 launch, 30% fewer "
            "thinking tokens than 2.5 Pro on typical traffic, and 3x faster behavior "
            "than 2.5 Pro based on external benchmarking cited in the post."
        ),
    },
    "gemini_3_deep_think": {
        "date": "2026-02-12",
        "url": "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/",
        "summary": (
            "Google released an updated Gemini 3 Deep Think mode for scientific, "
            "research, and engineering tasks; it emphasizes longer, more expensive "
            "reasoning. Public posts do not disclose model size."
        ),
    },
    "xla_spmd": {
        "date": "viewed 2026-04-17",
        "url": "https://docs.pytorch.org/xla/master/spmd.html",
        "summary": (
            "PyTorch/XLA SPMD exposes mesh and partition specs; XLA transforms a single-device "
            "program into a partitioned program with collectives using sharding annotations. "
            "This motivates modeling compiler-visible dataflow choices, not just link swaps."
        ),
    },
    "tpu7x": {
        "date": "last updated 2026-04-15",
        "url": "https://docs.cloud.google.com/tpu/docs/tpu7x",
        "summary": (
            "TPU7x/Ironwood is Google's latest Cloud TPU. A pod has 9216 chips, "
            "2307 BF16 TFLOP/s/chip, 4614 FP8 TFLOP/s/chip, 192 GiB HBM/chip, "
            "7380 GiB/s HBM bandwidth/chip, and 1200 GB/s bidirectional ICI/chip. "
            "Google states it is designed for dense and MoE models, pre-training, "
            "sampling, and decode-heavy inference."
        ),
    },
    "tpu_v5p": {
        "date": "viewed 2026-04-17",
        "url": "https://cloud.google.com/tpu/docs/v5p",
        "summary": (
            "TPU v5p: 8960 chips per pod; 459 TFLOPS/chip, 95 GiB HBM, "
            "2575 GiB/s HBM BW, 1200 GB/s bidirectional ICI per chip, 3D torus; "
            "4x4x4 and larger slices have full torus connectivity."
        ),
    },
    "ironwood_blog": {
        "date": "viewed 2026-04-17",
        "url": "https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/",
        "summary": (
            "Google describes Ironwood as XLA/software co-designed. A 64-chip cube uses dense "
            "3D torus ICI; cubes scale to 9216 chips through an OCS fabric with dynamic optical "
            "reconfiguration and 88473.6 Tbps aggregate superpod connectivity."
        ),
    },
    "tpu_rack_hotchips": {
        "date": "2025-08-24",
        "url": "https://hc2025.hotchips.org/assets/program/tutorials/HC2025.T1DCRacks.S7.Pankaj.pdf",
        "summary": (
            "Ironwood (2025): 4614 TFLOPS/chip, 192 GB HBM, 7.4 TBps HBM BW. "
            "4x4x4 torus building block; 6 cube faces × 16 ICI links each. "
            "96 optical + 80 copper + 64 PCB ICI connections per rack. "
            "Rack power-cap SLO < 15 ms; burst TDP window = 100 ms."
        ),
    },
    "advanced_chiplet_baseline": {
        "date": "project modeling assumption",
        "url": "https://docs.cloud.google.com/tpu/docs/v5p",
        "summary": (
            "Advanced-packaging comparison point requested by the project prompt: "
            "3.5 Tbps one-way link bandwidth and 6 pJ/bit per network hop."
        ),
    },
}

# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

LLM_WORKLOADS: dict[str, dict] = {
    # --- Inference decode ---
    "decode_mlp": {
        "M": 1,
        "K": 4096,
        "N": 16384,
        "description": (
            "Decode-time FFN/MLP projection (batch=1). "
            "Large weight matrix dominates remote memory traffic."
        ),
    },
    "kv_attention": {
        "M": 1,
        "K": 128,
        "N": 8192,
        "description": (
            "Single-token KV attention score projection. "
            "KV-cache read volume is the main network pressure."
        ),
    },
    # --- Inference prefill ---
    "prefill_projection": {
        "M": 2048,
        "K": 4096,
        "N": 4096,
        "description": "Large dense prefill GEMM (seq=2048). Heavy activation-exchange traffic.",
    },
    "long_context_prefill": {
        "M": 8192,
        "K": 4096,
        "N": 4096,
        "description": (
            "Long-context prefill GEMM (seq=8192). "
            "Stresses both activation exchange and weight-streaming traffic."
        ),
    },
    # --- MoE ---
    "moe_expert_dispatch": {
        "M": 1,
        "K": 4096,
        "N": 4096,
        "description": (
            "MoE expert dispatch: single-token query against expert gating matrix. "
            "Representative of token-routing overhead in sparse MoE models."
        ),
    },
    # --- Training ---
    "training_weight_grad": {
        "M": 4096,
        "K": 2048,
        "N": 4096,
        "description": (
            "Training weight-gradient accumulation GEMM. "
            "Representative of backward-pass data-parallel gradient traffic."
        ),
    },
}

# Default subset used for the main cluster study (keeps runtime manageable)
DEFAULT_WORKLOAD_NAMES: tuple[str, ...] = (
    "decode_mlp",
    "prefill_projection",
    "kv_attention",
)

# ---------------------------------------------------------------------------
# Communication scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "node_to_node": {
        "description": (
            "Tensor-parallel activation exchange where inputs and outputs traverse the fabric."
        ),
        "active_nodes": 16,
    },
    "node_to_dram": {
        "description": (
            "Remote weight or KV fetches from pooled memory attached through the fabric."
        ),
        "active_nodes": 16,
    },
    "shared_dram_multicast": {
        "description": (
            "Shared DRAM multicasts a common weight block or expert state to many nodes."
        ),
        "active_nodes": 16,
    },
}

DEFAULT_SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIOS)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadCase:
    name: str
    M: int
    K: int
    N: int
    description: str
    bits_per_value: int = 8


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    comm_pattern: str
    active_nodes: int = 16


@dataclass(frozen=True)
class InterconnectTech:
    name: str
    link_bandwidth_bps: float
    energy_per_bit_j: float
    router_latency_s: float
    supports_multicast: bool = False
    notes: str = ""


@dataclass(frozen=True)
class CompilerStrategy:
    name: str
    description: str
    requires_multicast: bool = False
    cache_reuse_scale: float = 1.0
    compiler_hint: str = ""


@dataclass(frozen=True)
class TrafficBreakdown:
    input_read_bits: float = 0.0
    weight_read_bits: float = 0.0
    output_write_bits: float = 0.0

    @property
    def total_bits(self) -> float:
        return self.input_read_bits + self.weight_read_bits + self.output_write_bits


@dataclass
class NetworkCost:
    read_energy_per_bit_j: float = 0.0
    write_energy_per_bit_j: float = 0.0
    read_latency_per_bit_s: float = 0.0
    write_latency_per_bit_s: float = 0.0
    total_energy_j: float = 0.0
    total_latency_s: float = 0.0
    detail: dict[str, float] | None = None


@dataclass
class IterationRecord:
    iteration: int
    total_energy_j: float
    total_latency_s: float
    logical_read_energy_j: float
    logical_write_energy_j: float
    logical_read_latency_s: float
    logical_write_latency_s: float
    traffic: TrafficBreakdown
    network: NetworkCost
    per_component_energy: dict[str, float]
    mapping_signature: str
    mapping_features: dict[str, float]


@dataclass(frozen=True)
class GeminiProxyModel:
    """
    Public, auditable proxy for a Gemini-class LLM.

    Google does not disclose Gemini 3/3.1 parameter counts or exact expert
    structure, so the end-to-end study uses explicit dense/MoE proxy sizes and
    reports all assumptions in the generated artifacts.
    """

    name: str
    model_kind: str
    total_params_b: float
    active_params_b: float
    layers: int
    hidden_size: int
    moe_experts: int = 1
    active_experts: int = 1
    prompt_tokens: int = 32_768
    output_tokens: int = 512
    train_tokens_per_step: int = 16_777_216
    disclosure_note: str = ""


@dataclass(frozen=True)
class SuperpodConfig:
    """Hardware + compiler/dataflow configuration for the 9216-chip study."""

    name: str
    topology: Topology
    tech: InterconnectTech
    active_chips: int = IRONWOOD_SUPERPOD_CHIPS
    fp8_tflops_per_chip: float = IRONWOOD_FP8_TFLOPS_PER_CHIP
    bf16_tflops_per_chip: float = IRONWOOD_BF16_TFLOPS_PER_CHIP
    hbm_gib_per_chip: int = IRONWOOD_HBM_GB
    hbm_bandwidth_gibps_per_chip: float = 7380.0
    compiler_strategy: str = "xla_spmd_collective"
    topology_family: str = "torus"
    compute_efficiency: float = 0.62
    network_overlap: float = 0.35
    tensor_parallel_traffic_scale: float = 1.0
    remote_weight_traffic_scale: float = 0.25
    gradient_traffic_scale: float = 1.0
    moe_traffic_scale: float = 1.0
    kv_traffic_scale: float = 1.0
    cache_reuse_steps: float = 32.0
    decode_token_factor: float = 1.0
    description: str = ""


# ---------------------------------------------------------------------------
# Topology models
# ---------------------------------------------------------------------------

class Topology:
    """Base topology model. Subclasses implement hop-count and contention logic."""

    def __init__(self, name: str):
        self.name = name

    def average_node_to_node_hops(self, active_nodes: int) -> float:
        raise NotImplementedError

    def average_node_to_memory_hops(self, active_nodes: int) -> float:
        raise NotImplementedError

    def multicast_hops(self, active_nodes: int) -> float:
        return self.average_node_to_memory_hops(active_nodes)

    def contention_factor(self, active_nodes: int, traffic_type: str) -> float:
        return 1.0

    def effective_bandwidth(
        self, tech: InterconnectTech, active_nodes: int, traffic_type: str
    ) -> float:
        return tech.link_bandwidth_bps / max(1.0, self.contention_factor(active_nodes, traffic_type))

    def _transfer_energy_latency(
        self,
        bits: float,
        hops: float,
        tech: InterconnectTech,
        active_nodes: int,
        traffic_type: str,
    ) -> tuple[float, float]:
        if bits <= 0:
            return 0.0, 0.0
        bw = self.effective_bandwidth(tech, active_nodes, traffic_type)
        energy = bits * hops * tech.energy_per_bit_j
        latency = bits / bw + hops * tech.router_latency_s
        return energy, latency

    def evaluate(
        self, traffic: TrafficBreakdown, scenario: Scenario, tech: InterconnectTech
    ) -> NetworkCost:
        """Evaluate network cost for the given traffic and technology."""
        active_nodes = max(1, scenario.active_nodes)
        read_energy = 0.0
        write_energy = 0.0
        read_latency = 0.0
        write_latency = 0.0
        detail: dict[str, float] = {}

        if scenario.name == "node_to_node":
            hops = self.average_node_to_node_hops(active_nodes)
            in_e, in_l = self._transfer_energy_latency(
                traffic.input_read_bits, hops, tech, active_nodes, "node_to_node"
            )
            out_e, out_l = self._transfer_energy_latency(
                traffic.output_write_bits, hops, tech, active_nodes, "node_to_node"
            )
            read_energy += in_e
            read_latency += in_l
            write_energy += out_e
            write_latency += out_l
            detail.update({
                "input_bits": traffic.input_read_bits,
                "output_bits": traffic.output_write_bits,
                "hops": hops,
            })

        elif scenario.name == "node_to_dram":
            hops = self.average_node_to_memory_hops(active_nodes)
            wt_e, wt_l = self._transfer_energy_latency(
                traffic.weight_read_bits, hops, tech, active_nodes, "node_to_dram"
            )
            read_energy += wt_e
            read_latency += wt_l
            detail.update({"weight_bits": traffic.weight_read_bits, "hops": hops})

        elif scenario.name == "shared_dram_multicast":
            repeated_bits = traffic.weight_read_bits
            unique_bits = repeated_bits / active_nodes if active_nodes else repeated_bits
            hops = self.multicast_hops(active_nodes)
            if tech.supports_multicast:
                eff_bits = unique_bits
            else:
                eff_bits = repeated_bits
            wt_e, wt_l = self._transfer_energy_latency(
                eff_bits, hops, tech, active_nodes, "shared_dram_multicast"
            )
            read_energy += wt_e
            read_latency += wt_l
            detail.update({
                "weight_bits": repeated_bits,
                "effective_bits": eff_bits,
                "multicast_fanout": active_nodes,
                "hops": hops,
            })

        read_bits = traffic.input_read_bits + traffic.weight_read_bits
        write_bits = traffic.output_write_bits
        return NetworkCost(
            read_energy_per_bit_j=read_energy / read_bits if read_bits else 0.0,
            write_energy_per_bit_j=write_energy / write_bits if write_bits else 0.0,
            read_latency_per_bit_s=read_latency / read_bits if read_bits else 0.0,
            write_latency_per_bit_s=write_latency / write_bits if write_bits else 0.0,
            total_energy_j=read_energy + write_energy,
            total_latency_s=read_latency + write_latency,
            detail=detail,
        )


class RingTopology(Topology):
    """1-D bidirectional ring. Average hop = N/4; contention scales with ring size."""

    def __init__(self):
        super().__init__("Electrical Ring")

    def average_node_to_node_hops(self, active_nodes: int) -> float:
        n = max(2, active_nodes)
        total = 0
        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cw = abs(i - j)
                total += min(cw, n - cw)
                count += 1
        return total / count

    def average_node_to_memory_hops(self, active_nodes: int) -> float:
        n = max(2, active_nodes)
        total = 0
        for i in range(n):
            cw = abs(i - 0)
            total += min(cw, n - cw)
        return total / n

    def contention_factor(self, active_nodes: int, traffic_type: str) -> float:
        return max(1.0, active_nodes / 4.0)


class Mesh2DTopology(Topology):
    """2-D rectangular mesh. Hop = Manhattan distance; contention ∝ √N."""

    def __init__(self, x: int, y: int):
        super().__init__(f"Electrical {x}x{y} Mesh")
        self.x = x
        self.y = y

    def _coords(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.x) for j in range(self.y)]

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def average_node_to_node_hops(self, active_nodes: int) -> float:
        coords = self._coords()[:active_nodes]
        total = sum(
            self._manhattan(s, d)
            for s in coords
            for d in coords
            if s != d
        )
        count = len(coords) * (len(coords) - 1)
        return total / count if count else 1.0

    def average_node_to_memory_hops(self, active_nodes: int) -> float:
        coords = self._coords()[:active_nodes]
        memory = (self.x // 2, self.y // 2)
        total = sum(self._manhattan(c, memory) + 1 for c in coords)
        return total / len(coords)

    def contention_factor(self, active_nodes: int, traffic_type: str) -> float:
        return max(1.0, math.sqrt(active_nodes))


class OpticalCrossbarTopology(Topology):
    """
    Lightmatter-inspired optical crossbar / switched fabric.

    All hops = 1 (any-to-any, single-hop). Multicast is native (wavelength routing
    or OCS broadcast). No contention factor: the optical fabric scales bandwidth
    linearly with more wavelengths rather than sharing a fixed pool.
    """

    def __init__(self):
        super().__init__("Lightmatter Optical Crossbar")

    def average_node_to_node_hops(self, active_nodes: int) -> float:
        return 1.0

    def average_node_to_memory_hops(self, active_nodes: int) -> float:
        return 1.0

    def multicast_hops(self, active_nodes: int) -> float:
        return 1.0

    def contention_factor(self, active_nodes: int, traffic_type: str) -> float:
        # Optical fabric: each wavelength channel is independent — no contention penalty
        return 1.0


class Torus3DTopology(Topology):
    """
    3-D torus (wrapping in all three dimensions). Matches TPU v5p cube topology.
    Contention factor scales as ∛N.
    """

    def __init__(self, x: int, y: int, z: int):
        super().__init__(f"{x}x{y}x{z} 3D Torus")
        self.x = x
        self.y = y
        self.z = z

    def _coords(self) -> list[tuple[int, int, int]]:
        return [
            (i, j, k)
            for i in range(self.x)
            for j in range(self.y)
            for k in range(self.z)
        ]

    def _dist(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        dx = min(abs(a[0] - b[0]), self.x - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.y - abs(a[1] - b[1]))
        dz = min(abs(a[2] - b[2]), self.z - abs(a[2] - b[2]))
        return dx + dy + dz

    def average_node_to_node_hops(self, active_nodes: int) -> float:
        coords = self._coords()[:active_nodes]
        if len(coords) > 512:
            return (self.x + self.y + self.z) / 4.0
        total = sum(
            self._dist(s, d)
            for s in coords
            for d in coords
            if s != d
        )
        count = len(coords) * (len(coords) - 1)
        return total / count if count else 1.0

    def average_node_to_memory_hops(self, active_nodes: int) -> float:
        coords = self._coords()[:active_nodes]
        if len(coords) > 512:
            return (self.x + self.y + self.z) / 4.0 + 1.0
        memory = (0, 0, 0)
        total = sum(self._dist(c, memory) + 1 for c in coords)
        return total / len(coords)

    def contention_factor(self, active_nodes: int, traffic_type: str) -> float:
        return max(1.0, active_nodes ** (1.0 / 3.0))


# ---------------------------------------------------------------------------
# Interconnect technology catalog
# ---------------------------------------------------------------------------

ELECTRICAL_800G = InterconnectTech(
    name="Electrical 800G/dir",
    link_bandwidth_bps=800e9,
    energy_per_bit_j=10e-12,
    router_latency_s=5e-9,
    supports_multicast=False,
    notes=(
        "800 Gb/s one-way inferred from TPU v5p's 1200 GB/s bidirectional ICI split over "
        "six nearest-neighbour torus links. Energy 10 pJ/bit and router latency 5 ns are "
        "modeling assumptions for a high-speed electrical fabric, not public Google disclosures."
    ),
)

ADVANCED_CHIPLET_3P5T = InterconnectTech(
    name="Advanced chiplet 3.5T",
    link_bandwidth_bps=3.5e12,
    energy_per_bit_j=6.0e-12,
    router_latency_s=2.0e-9,
    supports_multicast=False,
    notes=(
        "Project-requested advanced-packaging baseline: 3.5 Tbps one-way links, "
        "6 pJ/bit per hop. Modeled as high-quality electrical/chiplet fabric without "
        "native optical multicast."
    ),
)

LIGHTMATTER_EVK50 = InterconnectTech(
    name="Lightmatter EVK50",
    link_bandwidth_bps=800e9,       # 16-lambda BiDi, 50 Gbps/lambda -> 800 Gbps/fiber
    energy_per_bit_j=4.6e-12,       # 2.6 pJ/bit photonics + Guide + ~2 pJ/bit XSR SerDes
    router_latency_s=1.0e-9,        # optical: no physical packet-switching overhead
    supports_multicast=True,
    notes=(
        "SC25 demo (Dec 2025): 800 Gbps/fiber BiDi, 16-lambda NRZ 50G/lane, "
        "4.6 pJ/bit total wall-plug (2.6 pJ/bit photonics + ~2 pJ/bit off-the-shelf XSR SerDes). "
        "BER < 10^-9 (typically 10^-12), >1 km SMF reach, 25–105 °C."
    ),
)

LIGHTMATTER_EVK100 = InterconnectTech(
    name="Lightmatter EVK100",
    link_bandwidth_bps=3200e9,      # 3.2 Tbps/fiber at 112G PAM4, 16 lambda
    energy_per_bit_j=1.9e-12,       # ISES Korea 2025 efficiency table: 1.9 pJ/bit
    router_latency_s=1.5e-9,
    supports_multicast=True,
    notes=(
        "Passage EVK100: 3.2 Tbps/fiber, 1.9 pJ/bit reference value at 112G PAM4 "
        "(Passage family page / ISES Korea 2025). The EVK100 technical-spec page also lists "
        "a conservative 3.2 pJ/bit point, so the sensitivity sweeps should be used for that case. "
        "Monolithic PIC integrates MRM modulators, TIAs, Tx drivers on a single chip."
    ),
)

LIGHTMATTER_L200 = InterconnectTech(
    name="Lightmatter L200",
    link_bandwidth_bps=16e12,       # 32 Tbps total chip I/O / 2 directions = 16 Tbps one-way
    energy_per_bit_j=4.5e-12,       # <5 pJ/bit (conservative: 4.5 pJ/bit)
    router_latency_s=2.0e-9,
    supports_multicast=True,
    notes=(
        "Passage L200: 32 Tbps optical I/O per chip (56G NRZ, 3D chip-on-wafer, 320 SerDes). "
        "Modeled as 16 Tbps one-way. <5 pJ/bit; bandwidth density >1.5 Tbps/mm."
    ),
)

LIGHTMATTER_L200X = InterconnectTech(
    name="Lightmatter L200X",
    link_bandwidth_bps=32e12,       # 64 Tbps total chip I/O / 2 = 32 Tbps one-way
    energy_per_bit_j=4.8e-12,       # <5 pJ/bit at 112G PAM4 (slightly higher modulation cost)
    router_latency_s=2.0e-9,
    supports_multicast=True,
    notes=(
        "Passage L200X: 64 Tbps optical I/O per chip at 112G PAM4, same 3D CoW form factor as L200. "
        "Modeled as 32 Tbps one-way. <5 pJ/bit."
    ),
)

LIGHTMATTER_M1000 = InterconnectTech(
    name="Lightmatter M1000",
    link_bandwidth_bps=57.3e12,     # 114.6 Tbps aggregate / 2 directions = 57.3 Tbps one-way
    energy_per_bit_j=2.3e-12,       # System-level efficiency including laser
    router_latency_s=2.5e-9,        # slightly higher: includes built-in OCS switching
    supports_multicast=True,
    notes=(
        "Passage M1000 EVK: 114.6 Tbps total bi-directional bandwidth, 2.3 pJ/bit "
        "system-level efficiency including laser, 256 fibers, 1024 SerDes lanes, "
        "4,000 mm² active interposer, built-in solid-state OCS. Modeled as 57.3 Tbps one-way."
    ),
)

# Convenience dict of all Lightmatter technologies in bandwidth order
ALL_LIGHTMATTER_TECHS: dict[str, InterconnectTech] = {
    "EVK50": LIGHTMATTER_EVK50,
    "EVK100": LIGHTMATTER_EVK100,
    "L200": LIGHTMATTER_L200,
    "L200X": LIGHTMATTER_L200X,
    "M1000": LIGHTMATTER_M1000,
}


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def workload_case(name: str) -> WorkloadCase:
    params = LLM_WORKLOADS[name]
    return WorkloadCase(name=name, **params)


def scenario_case(name: str, active_nodes: int | None = None) -> Scenario:
    params = dict(SCENARIOS[name])
    if active_nodes is not None:
        params["active_nodes"] = active_nodes
    return Scenario(name=name, comm_pattern=name, **params)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": k, "date": v["date"], "summary": v["summary"], "url": v["url"]}
            for k, v in RESEARCH_NOTES.items()
        ]
    )


# ---------------------------------------------------------------------------
# Compiler/dataflow strategy model
# ---------------------------------------------------------------------------

COMPILER_STRATEGIES: dict[str, CompilerStrategy] = {
    "xla_spmd_collective": CompilerStrategy(
        name="XLA SPMD collective",
        description=(
            "Compiler keeps tensors sharded across the mesh and uses all-to-all / "
            "collective-permute style activation exchange."
        ),
        compiler_hint="mark_sharding / partition-spec favors tensor-parallel exchange",
    ),
    "remote_hbm_streaming": CompilerStrategy(
        name="Remote pooled-HBM streaming",
        description=(
            "Compiler treats remote memory as a cheap logical level and streams weights "
            "or KV blocks from pooled HBM instead of forcing local residency."
        ),
        compiler_hint="prefer remote read schedule and avoid replication",
    ),
    "replicated_hbm_cache": CompilerStrategy(
        name="Replicated HBM cache",
        description=(
            "Manual/Pallas-style prefetch replicates hot state into local HBM and "
            "amortizes the fill over several decode/prefill steps."
        ),
        cache_reuse_scale=1.0,
        compiler_hint="manual cache fill + local reuse",
    ),
    "photonic_multicast_cache": CompilerStrategy(
        name="Photonic multicast cache",
        description=(
            "Compiler uses optical multicast to fill many local caches with one logical "
            "transfer, enabling a different data-placement paradigm."
        ),
        requires_multicast=True,
        cache_reuse_scale=2.0,
        compiler_hint="broadcast/prefetch shared state before compute",
    ),
}

WORKLOAD_REUSE_STEPS: dict[str, int] = {
    "decode_mlp": 128,
    "kv_attention": 32,
    "prefill_projection": 8,
    "long_context_prefill": 4,
    "moe_expert_dispatch": 64,
    "training_weight_grad": 2,
}


def _workload_tensor_bits(workload: WorkloadCase) -> TrafficBreakdown:
    return TrafficBreakdown(
        input_read_bits=workload.M * workload.K * workload.bits_per_value,
        weight_read_bits=workload.K * workload.N * workload.bits_per_value,
        output_write_bits=workload.M * workload.N * workload.bits_per_value,
    )


def _workload_compute_cost(workload: WorkloadCase) -> tuple[float, float]:
    macs = workload.M * workload.K * workload.N
    energy_j = macs * 0.084e-12
    latency_s = macs / (TPU_V5P_TFLOPS_PER_CHIP * 1e12)
    return energy_j, latency_s


def evaluate_compiler_strategy(
    workload: WorkloadCase,
    strategy_name: str,
    topology: Topology,
    tech: InterconnectTech,
    active_nodes: int = 64,
    hbm_gib_per_node: int = 95,
) -> dict[str, float | str | bool]:
    """
    Evaluate a compiler-visible dataflow strategy.

    This is intentionally one layer above AccelForge's inner tile mapper. It models
    the choices an XLA/SPMD/Pallas-style compiler would expose: keep tensors sharded
    and exchange activations, stream from pooled HBM, or explicitly prefetch/cache
    model state. The goal is to capture new paradigms enabled by optical multicast
    and flat high-bandwidth fabrics.
    """
    if strategy_name not in COMPILER_STRATEGIES:
        raise ValueError(f"Unknown compiler strategy: {strategy_name}")

    strategy = COMPILER_STRATEGIES[strategy_name]
    tensor_bits = _workload_tensor_bits(workload)
    compute_energy_j, compute_latency_s = _workload_compute_cost(workload)
    reuse_steps = max(1.0, WORKLOAD_REUSE_STEPS.get(workload.name, 8) * strategy.cache_reuse_scale)

    cache_capacity_bits = hbm_gib_per_node * 1024**3 * 8 * 0.35
    reusable_state_bits = tensor_bits.weight_read_bits
    cache_feasible = reusable_state_bits <= cache_capacity_bits

    infeasible_reason = ""
    if strategy.requires_multicast and not tech.supports_multicast:
        infeasible_reason = "requires native multicast"

    if strategy_name == "xla_spmd_collective":
        traffic = TrafficBreakdown(
            input_read_bits=tensor_bits.input_read_bits,
            output_write_bits=tensor_bits.output_write_bits,
        )
        network_scenario = scenario_case("node_to_node", active_nodes=active_nodes)
        local_hbm_bits = (
            tensor_bits.input_read_bits + tensor_bits.output_write_bits + tensor_bits.weight_read_bits
        )
        memory_paradigm = "sharded tensors + collectives"
    elif strategy_name == "remote_hbm_streaming":
        traffic = TrafficBreakdown(weight_read_bits=tensor_bits.weight_read_bits)
        network_scenario = scenario_case("node_to_dram", active_nodes=active_nodes)
        local_hbm_bits = tensor_bits.input_read_bits + tensor_bits.output_write_bits
        memory_paradigm = "pooled remote memory"
    elif strategy_name == "replicated_hbm_cache":
        traffic = TrafficBreakdown(
            weight_read_bits=tensor_bits.weight_read_bits * active_nodes / reuse_steps
        )
        network_scenario = scenario_case("shared_dram_multicast", active_nodes=active_nodes)
        local_hbm_bits = (
            tensor_bits.input_read_bits + tensor_bits.output_write_bits + tensor_bits.weight_read_bits
        )
        memory_paradigm = "software-managed replicated cache"
    elif strategy_name == "photonic_multicast_cache":
        traffic = TrafficBreakdown(
            weight_read_bits=tensor_bits.weight_read_bits * active_nodes / reuse_steps
        )
        network_scenario = scenario_case("shared_dram_multicast", active_nodes=active_nodes)
        local_hbm_bits = (
            tensor_bits.input_read_bits + tensor_bits.output_write_bits + tensor_bits.weight_read_bits
        )
        memory_paradigm = "optical multicast cache fill"
    else:
        raise AssertionError(f"Unhandled strategy: {strategy_name}")

    network = topology.evaluate(traffic, network_scenario, tech)
    local_hbm_energy_j = local_hbm_bits * HBM_ENERGY_PER_BIT_J
    local_hbm_latency_s = local_hbm_bits * HBM_LATENCY_PER_BIT_S
    cache_pressure = reusable_state_bits / max(cache_capacity_bits, 1.0)
    if not cache_feasible and "cache" in strategy_name:
        cache_spill_penalty_j = (cache_pressure - 1.0) * reusable_state_bits * HBM_ENERGY_PER_BIT_J
        cache_spill_penalty_s = (cache_pressure - 1.0) * reusable_state_bits * HBM_LATENCY_PER_BIT_S
    else:
        cache_spill_penalty_j = 0.0
        cache_spill_penalty_s = 0.0

    total_energy_j = compute_energy_j + local_hbm_energy_j + network.total_energy_j + cache_spill_penalty_j
    total_latency_s = compute_latency_s + max(local_hbm_latency_s, network.total_latency_s) + cache_spill_penalty_s

    if infeasible_reason:
        total_energy_j = float("inf")
        total_latency_s = float("inf")

    return {
        "strategy": strategy_name,
        "strategy_label": strategy.name,
        "compiler_hint": strategy.compiler_hint,
        "memory_paradigm": memory_paradigm,
        "requires_multicast": strategy.requires_multicast,
        "feasible": not bool(infeasible_reason),
        "infeasible_reason": infeasible_reason,
        "active_nodes": active_nodes,
        "reuse_steps": reuse_steps,
        "cache_pressure": cache_pressure,
        "cache_feasible": cache_feasible,
        "network_energy_uJ": network.total_energy_j * 1e6,
        "network_latency_us": network.total_latency_s * 1e6,
        "local_hbm_energy_uJ": local_hbm_energy_j * 1e6,
        "compute_energy_uJ": compute_energy_j * 1e6,
        "total_energy_uJ": total_energy_j * 1e6,
        "total_latency_us": total_latency_s * 1e6,
        "traffic_kib": traffic.total_bits / 1024.0,
    }


def default_compiler_architectures(
    active_nodes: int = 64,
) -> list[tuple[str, Topology, InterconnectTech]]:
    shape = torus_shape_for_nodes(active_nodes)
    return [
        ("TPU electrical torus", Torus3DTopology(*shape), ELECTRICAL_800G),
        ("Advanced chiplet torus", Torus3DTopology(*shape), ADVANCED_CHIPLET_3P5T),
        ("EVK100 photonic torus", Torus3DTopology(*shape), LIGHTMATTER_EVK100),
        ("M1000 optical switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_M1000),
    ]


def compiler_strategy_study(
    workload_names: Iterable[str] = DEFAULT_WORKLOAD_NAMES,
    active_nodes: int = 64,
    architectures: Iterable[tuple[str, Topology, InterconnectTech]] | None = None,
    strategy_names: Iterable[str] = tuple(COMPILER_STRATEGIES),
) -> pd.DataFrame:
    rows = []
    architectures = list(architectures or default_compiler_architectures(active_nodes))
    for workload_name in workload_names:
        workload = workload_case(workload_name)
        for architecture, topology, tech in architectures:
            for strategy_name in strategy_names:
                result = evaluate_compiler_strategy(
                    workload,
                    strategy_name,
                    topology,
                    tech,
                    active_nodes=active_nodes,
                )
                result.update({
                    "workload": workload.name,
                    "architecture": architecture,
                    "topology": topology.name,
                    "technology": tech.name,
                })
                rows.append(result)
    return pd.DataFrame(rows)


def optimal_compiler_strategy_frame(
    strategy_df: pd.DataFrame,
    baseline_architecture: str = "TPU electrical torus",
) -> pd.DataFrame:
    if strategy_df.empty:
        return pd.DataFrame()
    feasible = strategy_df[strategy_df["feasible"]].copy()
    idx = feasible.groupby(["workload", "architecture"])["total_energy_uJ"].idxmin()
    best = feasible.loc[idx].copy().sort_values(["workload", "architecture"]).reset_index(drop=True)
    baseline = best[best["architecture"] == baseline_architecture][
        ["workload", "strategy", "strategy_label", "total_energy_uJ", "total_latency_us"]
    ].rename(
        columns={
            "strategy": "baseline_strategy",
            "strategy_label": "baseline_strategy_label",
            "total_energy_uJ": "baseline_total_energy_uJ",
            "total_latency_us": "baseline_total_latency_us",
        }
    )
    best = best.merge(baseline, on="workload", how="left")
    best["strategy_changed_vs_baseline"] = best["strategy"] != best["baseline_strategy"]
    best["energy_gain_vs_baseline_strategy_pct"] = 100 * (
        1 - best["total_energy_uJ"] / best["baseline_total_energy_uJ"]
    )
    best["latency_gain_vs_baseline_strategy_pct"] = 100 * (
        1 - best["total_latency_us"] / best["baseline_total_latency_us"]
    )
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# AccelForge integration
# ---------------------------------------------------------------------------

def _build_spec(
    workload: WorkloadCase,
    scenario: Scenario,
    logical_read_energy_j: float = HBM_ENERGY_PER_BIT_J,
    logical_write_energy_j: float = HBM_ENERGY_PER_BIT_J,
    logical_read_latency_s: float = HBM_LATENCY_PER_BIT_S,
    logical_write_latency_s: float = HBM_LATENCY_PER_BIT_S,
    node_x: int = 4,
    node_y: int = 4,
    node_z: int = 1,
    pe_x: int = 64,
    pe_y: int = 64,
    hbm_gib: int = 8,
) -> af.Spec:
    jinja = {
        "COMM_PATTERN": scenario.comm_pattern,
        "LOGICAL_READ_ENERGY": logical_read_energy_j,
        "LOGICAL_WRITE_ENERGY": logical_write_energy_j,
        "LOGICAL_READ_LATENCY": logical_read_latency_s,
        "LOGICAL_WRITE_LATENCY": logical_write_latency_s,
        "NODES_X": node_x,
        "NODES_Y": node_y,
        "NODES_Z": node_z,
        "PE_X": pe_x,
        "PE_Y": pe_y,
        "HBM_GIB": hbm_gib,
        "M": workload.M,
        "K": workload.K,
        "N": workload.N,
        "BITS_PER_VALUE": workload.bits_per_value,
    }
    spec = af.Spec.from_yaml(ARCH_PATH, WORKLOAD_PATH, jinja_parse_data=jinja)
    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    spec.mapper.max_pmapping_templates_per_einsum = 1
    return spec


def _best_mapping(mappings: af.Mapping) -> af.Mapping:
    return mappings[0] if len(mappings) else mappings


def _extract_action_bits(
    best_mapping: af.Mapping, component: str
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    row = best_mapping.data.iloc[0]
    for col, value in row.items():
        parts = col.split("<SEP>")
        if len(parts) != 5 or parts[1] != "action":
            continue
        _, _, found_component, tensor, action = parts
        if found_component != component:
            continue
        result[(tensor, action)] = result.get((tensor, action), 0.0) + float(value)
    return result


def extract_traffic(best_mapping: af.Mapping) -> TrafficBreakdown:
    """Parse AccelForge mapping output to recover logical-memory traffic in bits."""
    actions = _extract_action_bits(best_mapping, "LogicalClusterMemory")

    def _sum(keys: Iterable[tuple[str, str]]) -> float:
        return sum(actions.get(k, 0.0) for k in keys)

    return TrafficBreakdown(
        input_read_bits=_sum({("input", "read"), ("Activations", "read")}),
        weight_read_bits=_sum({("weight", "read"), ("Weights", "read")}),
        output_write_bits=_sum({("output", "write"), ("Outputs", "write")}),
    )


def mapping_feature_dict(best_mapping: af.Mapping) -> dict[str, float]:
    """
    Return a compact numeric fingerprint of the best mapping.

    AccelForge exposes a very rich `Total<SEP>mapping` object that is too verbose for
    reports. This helper keeps only the fields that are useful for explaining why a
    mapping changed: spatial fanout usage, selected memory residency, and action bits.
    """
    if best_mapping is None or best_mapping.data.empty:
        return {}

    row = best_mapping.data.iloc[0]
    features: dict[str, float] = {}
    for col, value in row.items():
        parts = col.split("<SEP>")
        if len(parts) != 5:
            continue
        einsum, kind, level, component, detail = parts
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == 0:
            continue
        if kind == "usage" and level == "spatial":
            features[f"spatial::{component}::{detail}"] = numeric
        elif kind == "usage" and level == "memory":
            features[f"memory::{component}::{detail}"] = numeric
        elif kind == "action":
            features[f"action::{level}::{component}::{detail}"] = numeric
    return features


def mapping_signature(best_mapping: af.Mapping, max_terms: int = 18) -> str:
    """
    Human-readable mapping fingerprint for tables and presentation summaries.

    The signature is intentionally lossy: it keeps the high-signal spatial and
    memory/action choices while avoiding AccelForge's very long object repr.
    """
    features = mapping_feature_dict(best_mapping)
    if not features:
        return "empty"

    priority = {
        "spatial::NodeArray::NodeX": 0,
        "spatial::NodeArray::NodeY": 1,
        "spatial::NodeArray::NodeZ": 2,
        "spatial::ProcessingElement::reuse_input": 3,
        "spatial::ProcessingElement::reuse_output": 4,
    }

    def _sort_key(item: tuple[str, float]) -> tuple[float, str]:
        key, _ = item
        if key in priority:
            return (priority[key], key)
        if key.startswith("action::LogicalClusterMemory"):
            return (10, key)
        if key.startswith("memory::LogicalClusterMemory"):
            return (20, key)
        if key.startswith("memory::HBM"):
            return (30, key)
        return (100, key)

    terms = []
    for key, value in sorted(features.items(), key=_sort_key)[:max_terms]:
        short_key = (
            key.replace("spatial::NodeArray::", "node.")
            .replace("spatial::ProcessingElement::", "pe.")
            .replace("action::LogicalClusterMemory::", "logical.")
            .replace("memory::LogicalClusterMemory::", "logical_usage.")
            .replace("memory::HBM::", "hbm_usage.")
            .replace("memory::UnifiedBuffer::", "ub_usage.")
        )
        if abs(value) >= 1024:
            value_text = f"{value / 1024:.1f} Kib"
        else:
            value_text = f"{value:.4g}"
        terms.append(f"{short_key}={value_text}")
    return "; ".join(terms)


def mapping_features_frame(best_mapping: af.Mapping) -> pd.DataFrame:
    """Return the mapping fingerprint as a tidy DataFrame."""
    rows = []
    for key, value in mapping_feature_dict(best_mapping).items():
        parts = key.split("::")
        rows.append({
            "kind": parts[0],
            "component": parts[1] if len(parts) > 1 else "",
            "detail": "::".join(parts[2:]),
            "value": value,
        })
    return pd.DataFrame(rows).sort_values(["kind", "component", "detail"]).reset_index(drop=True)


def run_one_iteration(
    workload: WorkloadCase,
    scenario: Scenario,
    logical_read_energy_j: float,
    logical_write_energy_j: float,
    logical_read_latency_s: float,
    logical_write_latency_s: float,
    node_shape: tuple[int, int, int] = (4, 4, 1),
    pe_shape: tuple[int, int] = (64, 64),
    hbm_gib: int = 8,
) -> tuple[af.Mapping, TrafficBreakdown]:
    spec = _build_spec(
        workload,
        scenario,
        logical_read_energy_j=logical_read_energy_j,
        logical_write_energy_j=logical_write_energy_j,
        logical_read_latency_s=logical_read_latency_s,
        logical_write_latency_s=logical_write_latency_s,
        node_x=node_shape[0],
        node_y=node_shape[1],
        node_z=node_shape[2],
        pe_x=pe_shape[0],
        pe_y=pe_shape[1],
        hbm_gib=hbm_gib,
    )
    mappings = spec.map_workload_to_arch(
        print_progress=False,
        print_number_of_pmappings=False,
    )
    best = _best_mapping(mappings)
    return best, extract_traffic(best)


# ---------------------------------------------------------------------------
# Iterative fixed-point solver (Milestone 2)
# ---------------------------------------------------------------------------

def solve_iteratively(
    workload: WorkloadCase,
    scenario: Scenario,
    topology: Topology,
    tech: InterconnectTech,
    max_iterations: int = 6,
    convergence_tol: float = 1e-4,
    node_shape: tuple[int, int, int] = (4, 4, 1),
    pe_shape: tuple[int, int] = (64, 64),
    hbm_gib: int = 8,
) -> tuple[list[IterationRecord], af.Mapping]:
    """
    Run AccelForge mapper repeatedly, folding network costs back into the
    logical-memory energy/latency until convergence or max_iterations.

    Returns the full iteration history and the final best mapping.
    """
    logical_read_energy_j = HBM_ENERGY_PER_BIT_J
    logical_write_energy_j = HBM_ENERGY_PER_BIT_J
    logical_read_latency_s = HBM_LATENCY_PER_BIT_S
    logical_write_latency_s = HBM_LATENCY_PER_BIT_S

    history: list[IterationRecord] = []
    final_mapping = None

    for iteration in range(max_iterations):
        mapping, traffic = run_one_iteration(
            workload,
            scenario,
            logical_read_energy_j,
            logical_write_energy_j,
            logical_read_latency_s,
            logical_write_latency_s,
            node_shape=node_shape,
            pe_shape=pe_shape,
            hbm_gib=hbm_gib,
        )
        network = topology.evaluate(traffic, scenario, tech)
        total_energy = float(mapping.energy())
        total_latency = float(mapping.latency())
        features = mapping_feature_dict(mapping)

        history.append(
            IterationRecord(
                iteration=iteration,
                total_energy_j=total_energy,
                total_latency_s=total_latency,
                logical_read_energy_j=logical_read_energy_j,
                logical_write_energy_j=logical_write_energy_j,
                logical_read_latency_s=logical_read_latency_s,
                logical_write_latency_s=logical_write_latency_s,
                traffic=traffic,
                network=network,
                per_component_energy={
                    k: float(v)
                    for k, v in mapping.energy(per_component=True).items()
                },
                mapping_signature=mapping_signature(mapping),
                mapping_features=features,
            )
        )
        final_mapping = mapping

        new_read_energy = HBM_ENERGY_PER_BIT_J + network.read_energy_per_bit_j
        new_write_energy = HBM_ENERGY_PER_BIT_J + network.write_energy_per_bit_j
        new_read_latency = HBM_LATENCY_PER_BIT_S + network.read_latency_per_bit_s
        new_write_latency = HBM_LATENCY_PER_BIT_S + network.write_latency_per_bit_s

        deltas = [
            abs(new_read_energy - logical_read_energy_j) / max(logical_read_energy_j, 1e-30),
            abs(new_write_energy - logical_write_energy_j) / max(logical_write_energy_j, 1e-30),
            abs(new_read_latency - logical_read_latency_s) / max(logical_read_latency_s, 1e-30),
            abs(new_write_latency - logical_write_latency_s) / max(logical_write_latency_s, 1e-30),
        ]

        logical_read_energy_j = new_read_energy
        logical_write_energy_j = new_write_energy
        logical_read_latency_s = new_read_latency
        logical_write_latency_s = new_write_latency

        if max(deltas) < convergence_tol:
            break

    return history, final_mapping


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def history_to_frame(history: Iterable[IterationRecord]) -> pd.DataFrame:
    """Convert an iteration history into a tidy DataFrame."""
    rows = []
    for r in history:
        rows.append({
            "iteration": r.iteration,
            "total_energy_uJ": r.total_energy_j * 1e6,
            "total_latency_us": r.total_latency_s * 1e6,
            "logical_read_pJ_per_bit": r.logical_read_energy_j * 1e12,
            "logical_write_pJ_per_bit": r.logical_write_energy_j * 1e12,
            "logical_read_ps_per_bit": r.logical_read_latency_s * 1e12,
            "logical_write_ps_per_bit": r.logical_write_latency_s * 1e12,
            "remote_input_kibits": r.traffic.input_read_bits / 1024.0,
            "remote_weight_kibits": r.traffic.weight_read_bits / 1024.0,
            "remote_output_kibits": r.traffic.output_write_bits / 1024.0,
            "network_energy_uJ": r.network.total_energy_j * 1e6,
            "network_latency_us": r.network.total_latency_s * 1e6,
            "mapping_signature": r.mapping_signature,
            "spatial_NodeX_usage": r.mapping_features.get("spatial::NodeArray::NodeX", 0.0),
            "spatial_NodeY_usage": r.mapping_features.get("spatial::NodeArray::NodeY", 0.0),
            "spatial_NodeZ_usage": r.mapping_features.get("spatial::NodeArray::NodeZ", 0.0),
        })
    return pd.DataFrame(rows)


def per_component_energy_frame(history: list[IterationRecord]) -> pd.DataFrame:
    """Return per-component energy for the final (converged) iteration."""
    if not history:
        return pd.DataFrame()
    final = history[-1]
    rows = [
        {"component": k, "energy_uJ": v * 1e6}
        for k, v in final.per_component_energy.items()
        if v > 0
    ]
    return pd.DataFrame(rows).sort_values("energy_uJ", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main comparison studies (Milestones 2 & 3)
# ---------------------------------------------------------------------------

def compare_topologies(
    workload_names: Iterable[str],
    scenario_names: Iterable[str],
    topology_to_tech: dict[Topology, InterconnectTech],
    max_iterations: int = 6,
    node_shape: tuple[int, int, int] = (4, 4, 1),
    pe_shape: tuple[int, int] = (64, 64),
    hbm_gib: int = 8,
) -> pd.DataFrame:
    """
    Run the full iterative study for every (workload, scenario, topology) combination.

    Returns a tidy DataFrame with baseline, first-run+network, and final converged
    energy and latency columns, plus mapping-sensitivity statistics.
    """
    rows = []
    for workload_name in workload_names:
        workload = workload_case(workload_name)
        for scenario_name in scenario_names:
            scenario = scenario_case(scenario_name)
            for topology, tech in topology_to_tech.items():
                history, final_mapping = solve_iteratively(
                    workload,
                    scenario,
                    topology,
                    tech,
                    max_iterations=max_iterations,
                    node_shape=node_shape,
                    pe_shape=pe_shape,
                    hbm_gib=hbm_gib,
                )
                baseline_energy = history[0].total_energy_j
                baseline_latency = history[0].total_latency_s
                final_energy = history[-1].total_energy_j
                final_latency = history[-1].total_latency_s
                first_plus_net_energy = baseline_energy + history[0].network.total_energy_j
                first_plus_net_latency = baseline_latency + history[0].network.total_latency_s
                first_traffic_bits = history[0].traffic.total_bits
                final_traffic_bits = history[-1].traffic.total_bits
                rows.append({
                    "workload": workload.name,
                    "scenario": scenario.name,
                    "topology": topology.name,
                    "technology": tech.name,
                    "iterations": len(history),
                    "baseline_energy_uJ": baseline_energy * 1e6,
                    "baseline_latency_us": baseline_latency * 1e6,
                    "first_run_plus_network_energy_uJ": first_plus_net_energy * 1e6,
                    "first_run_plus_network_latency_us": first_plus_net_latency * 1e6,
                    "final_energy_uJ": final_energy * 1e6,
                    "final_latency_us": final_latency * 1e6,
                    "network_delta_energy_pct": 100 * (final_energy / baseline_energy - 1.0),
                    "network_delta_latency_pct": 100 * (final_latency / baseline_latency - 1.0),
                    "mapping_sensitivity_energy_pct": (
                        100 * (final_energy / first_plus_net_energy - 1.0)
                        if first_plus_net_energy > 0 else 0.0
                    ),
                    "mapping_sensitivity_latency_pct": (
                        100 * (final_latency / first_plus_net_latency - 1.0)
                        if first_plus_net_latency > 0 else 0.0
                    ),
                    "remote_bits_kib": history[-1].traffic.total_bits / 1024.0,
                    "first_remote_bits_kib": first_traffic_bits / 1024.0,
                    "remote_bits_change_pct": (
                        100 * (final_traffic_bits / first_traffic_bits - 1.0)
                        if first_traffic_bits > 0 else 0.0
                    ),
                    "mapping_changed": history[0].mapping_signature != history[-1].mapping_signature,
                    "first_mapping_signature": history[0].mapping_signature,
                    "final_mapping_signature": history[-1].mapping_signature,
                    "spatial_NodeX_usage": history[-1].mapping_features.get(
                        "spatial::NodeArray::NodeX", 0.0
                    ),
                    "spatial_NodeY_usage": history[-1].mapping_features.get(
                        "spatial::NodeArray::NodeY", 0.0
                    ),
                    "spatial_NodeZ_usage": history[-1].mapping_features.get(
                        "spatial::NodeArray::NodeZ", 0.0
                    ),
                    "mac_energy_uJ": (
                        final_mapping.energy(per_component=True).get("MAC", 0.0) * 1e6
                    ),
                    "logical_memory_read_pJ_per_bit": (
                        history[-1].logical_read_energy_j * 1e12
                    ),
                })
    return pd.DataFrame(rows)


def default_topology_suite() -> dict[Topology, InterconnectTech]:
    return {
        Mesh2DTopology(4, 4): ELECTRICAL_800G,
        OpticalCrossbarTopology(): LIGHTMATTER_EVK100,
    }


def cluster_improvement_frame(
    cluster_df: pd.DataFrame,
    baseline_topology: str = "Electrical 4x4 Mesh",
    proposal_topology: str = "Lightmatter Optical Crossbar",
) -> pd.DataFrame:
    baseline = cluster_df[cluster_df["topology"] == baseline_topology].copy()
    proposal = cluster_df[cluster_df["topology"] == proposal_topology].copy()
    merged = baseline.merge(proposal, on=["workload", "scenario"], suffixes=("_baseline", "_proposal"))
    merged["energy_reduction_pct"] = 100 * (
        1 - merged["final_energy_uJ_proposal"] / merged["final_energy_uJ_baseline"]
    )
    merged["latency_reduction_pct"] = 100 * (
        1 - merged["final_latency_us_proposal"] / merged["final_latency_us_baseline"]
    )
    return merged.sort_values(["scenario", "workload"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# All-technology comparison (pure network evaluation, no AccelForge)
# ---------------------------------------------------------------------------

def all_technologies_network_comparison(
    traffic_bits: float = 8 * 1024**3,   # 1 GiB default
    scenario_name: str = "shared_dram_multicast",
    active_nodes: int = 16,
    topology: Topology | None = None,
) -> pd.DataFrame:
    """
    Pure network cost comparison for all catalog technologies on a single
    (scenario, traffic volume) point. Does not call AccelForge.

    Useful for a quick technology sweep independent of mapping.
    """
    if topology is None:
        topology = OpticalCrossbarTopology()
    scenario = scenario_case(scenario_name, active_nodes=active_nodes)
    traffic = TrafficBreakdown(
        input_read_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
        weight_read_bits=traffic_bits,
        output_write_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
    )

    techs = {
        "Electrical 800G": ELECTRICAL_800G,
        "Advanced Chiplet 3.5T": ADVANCED_CHIPLET_3P5T,
        **ALL_LIGHTMATTER_TECHS,
    }
    rows = []
    for label, tech in techs.items():
        cost = topology.evaluate(traffic, scenario, tech)
        rows.append({
            "technology": label,
            "link_bandwidth_tbps": tech.link_bandwidth_bps / 1e12,
            "energy_per_bit_pJ": tech.energy_per_bit_j * 1e12,
            "supports_multicast": tech.supports_multicast,
            "total_energy_uJ": cost.total_energy_j * 1e6,
            "total_latency_us": cost.total_latency_s * 1e6,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity sweeps (Milestone 3 / extended analysis)
# ---------------------------------------------------------------------------

def sweep_node_counts(
    workload_name: str,
    scenario_name: str,
    topology: Topology,
    tech: InterconnectTech,
    node_counts: Iterable[int] = (4, 8, 16, 32, 64),
    max_iterations: int = 4,
) -> pd.DataFrame:
    """
    Evaluate converged network energy/latency for varying numbers of active nodes.
    Runs solve_iteratively for each node count; each call requires AccelForge.
    """
    rows = []
    workload = workload_case(workload_name)
    for n in node_counts:
        scenario = scenario_case(scenario_name, active_nodes=n)
        history, _ = solve_iteratively(
            workload, scenario, topology, tech, max_iterations=max_iterations
        )
        final = history[-1]
        rows.append({
            "active_nodes": n,
            "final_energy_uJ": final.total_energy_j * 1e6,
            "final_latency_us": final.total_latency_s * 1e6,
            "network_energy_uJ": final.network.total_energy_j * 1e6,
            "network_latency_us": final.network.total_latency_s * 1e6,
            "iterations_to_converge": len(history),
        })
    return pd.DataFrame(rows)


def bandwidth_sensitivity_network(
    traffic_bits: float = 8 * 1024**3,
    scenario_name: str = "node_to_dram",
    active_nodes: int = 16,
    base_tech: InterconnectTech = ELECTRICAL_800G,
    multipliers: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    topology: Topology | None = None,
) -> pd.DataFrame:
    """
    Pure-network bandwidth sensitivity sweep. Scales link_bandwidth_bps by each
    multiplier and reports energy + latency. No AccelForge call.
    """
    if topology is None:
        topology = Mesh2DTopology(4, 4)
    scenario = scenario_case(scenario_name, active_nodes=active_nodes)
    traffic = TrafficBreakdown(
        input_read_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
        weight_read_bits=traffic_bits,
        output_write_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
    )
    rows = []
    for mult in multipliers:
        tech = InterconnectTech(
            name=f"{base_tech.name} x{mult:.2g}BW",
            link_bandwidth_bps=base_tech.link_bandwidth_bps * mult,
            energy_per_bit_j=base_tech.energy_per_bit_j,
            router_latency_s=base_tech.router_latency_s,
            supports_multicast=base_tech.supports_multicast,
        )
        cost = topology.evaluate(traffic, scenario, tech)
        rows.append({
            "bw_multiplier": mult,
            "link_bandwidth_tbps": tech.link_bandwidth_bps / 1e12,
            "energy_uJ": cost.total_energy_j * 1e6,
            "latency_us": cost.total_latency_s * 1e6,
        })
    return pd.DataFrame(rows)


def energy_per_bit_sensitivity_network(
    traffic_bits: float = 8 * 1024**3,
    scenario_name: str = "node_to_dram",
    active_nodes: int = 16,
    base_tech: InterconnectTech = ELECTRICAL_800G,
    multipliers: Iterable[float] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    topology: Topology | None = None,
) -> pd.DataFrame:
    """
    Pure-network energy-per-bit sensitivity sweep.
    """
    if topology is None:
        topology = Mesh2DTopology(4, 4)
    scenario = scenario_case(scenario_name, active_nodes=active_nodes)
    traffic = TrafficBreakdown(
        input_read_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
        weight_read_bits=traffic_bits,
        output_write_bits=traffic_bits if scenario_name == "node_to_node" else 0.0,
    )
    rows = []
    for mult in multipliers:
        tech = InterconnectTech(
            name=f"{base_tech.name} x{mult:.2g}E",
            link_bandwidth_bps=base_tech.link_bandwidth_bps,
            energy_per_bit_j=base_tech.energy_per_bit_j * mult,
            router_latency_s=base_tech.router_latency_s,
            supports_multicast=base_tech.supports_multicast,
        )
        cost = topology.evaluate(traffic, scenario, tech)
        rows.append({
            "energy_multiplier": mult,
            "energy_per_bit_pJ": tech.energy_per_bit_j * 1e12,
            "total_energy_uJ": cost.total_energy_j * 1e6,
            "total_latency_us": cost.total_latency_s * 1e6,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaling-law and advanced-packaging studies
# ---------------------------------------------------------------------------

def torus_shape_for_nodes(active_nodes: int) -> tuple[int, int, int]:
    known = {
        64: (4, 4, 4),
        128: (4, 4, 8),
        256: (4, 8, 8),
        512: (8, 8, 8),
        1024: (8, 8, 16),
        2048: (8, 16, 16),
        4096: (16, 16, 16),
        6144: (16, 16, 24),
        9216: (12, 24, 32),
    }
    if active_nodes in known:
        return known[active_nodes]
    x = max(1, round(active_nodes ** (1 / 3)))
    y = x
    z = max(1, math.ceil(active_nodes / (x * y)))
    return (x, y, z)


def scaling_law_study(
    node_counts: Iterable[int] = (64, 128, 256, 512, 1024, 2048, 4096, 9216),
    traffic_gib_per_chip: float = 1.0,
) -> pd.DataFrame:
    """
    Compare scaling laws for torus, chiplet, and optical fabrics.

    Rows are per-chip network costs for a fixed per-chip traffic volume. The
    important result is the slope: torus fabrics accumulate hop/contention cost
    as the system grows, while switched optical fabrics keep hop count and
    contention flat and make multicast nearly independent of fanout.
    """
    bits = traffic_gib_per_chip * 1024**3 * 8
    rows = []
    for nodes in node_counts:
        shape = torus_shape_for_nodes(nodes)
        fabrics = [
            ("TPU electrical torus", Torus3DTopology(*shape), ELECTRICAL_800G),
            ("Advanced chiplet torus", Torus3DTopology(*shape), ADVANCED_CHIPLET_3P5T),
            ("EVK100 photonic torus", Torus3DTopology(*shape), LIGHTMATTER_EVK100),
            ("M1000 optical switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_M1000),
        ]
        for architecture, topology, tech in fabrics:
            for scenario_name, traffic in (
                (
                    "all_to_all_activation",
                    TrafficBreakdown(input_read_bits=bits, output_write_bits=bits),
                ),
                (
                    "shared_state_multicast",
                    TrafficBreakdown(weight_read_bits=bits * nodes),
                ),
            ):
                scenario = scenario_case(
                    "node_to_node" if scenario_name == "all_to_all_activation"
                    else "shared_dram_multicast",
                    active_nodes=nodes,
                )
                cost = topology.evaluate(traffic, scenario, tech)
                rows.append({
                    "active_nodes": nodes,
                    "torus_shape": "x".join(str(v) for v in shape),
                    "architecture": architecture,
                    "topology": topology.name,
                    "technology": tech.name,
                    "scenario": scenario_name,
                    "traffic_gib_per_chip": traffic_gib_per_chip,
                    "average_hops": (
                        topology.average_node_to_node_hops(nodes)
                        if scenario.name == "node_to_node"
                        else topology.multicast_hops(nodes)
                    ),
                    "contention_factor": topology.contention_factor(nodes, scenario.name),
                    "energy_uJ_per_chip": cost.total_energy_j * 1e6,
                    "latency_us_per_chip": cost.total_latency_s * 1e6,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    baseline = df[df["active_nodes"] == min(node_counts)][
        ["architecture", "scenario", "energy_uJ_per_chip", "latency_us_per_chip"]
    ].rename(
        columns={
            "energy_uJ_per_chip": "energy_uJ_at_min_nodes",
            "latency_us_per_chip": "latency_us_at_min_nodes",
        }
    )
    df = df.merge(baseline, on=["architecture", "scenario"], how="left")
    df["energy_scale_vs_min_nodes"] = df["energy_uJ_per_chip"] / df["energy_uJ_at_min_nodes"]
    df["latency_scale_vs_min_nodes"] = df["latency_us_per_chip"] / df["latency_us_at_min_nodes"]
    return df


def scaling_exponent_frame(scaling_df: pd.DataFrame) -> pd.DataFrame:
    """Fit log-log scaling exponents for each architecture/scenario pair."""
    rows = []
    for (architecture, scenario), group in scaling_df.groupby(["architecture", "scenario"]):
        group = group.sort_values("active_nodes")
        xs = [math.log(float(v)) for v in group["active_nodes"]]
        for metric in ("energy_uJ_per_chip", "latency_us_per_chip"):
            ys = [math.log(max(float(v), 1e-30)) for v in group[metric]]
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            denom = sum((x - x_mean) ** 2 for x in xs)
            slope = (
                sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
                if denom else 0.0
            )
            rows.append({
                "architecture": architecture,
                "scenario": scenario,
                "metric": metric,
                "scaling_exponent": slope,
            })
    return pd.DataFrame(rows)


def chiplet_comparison_frame(
    active_nodes: int = 64,
    traffic_gib_per_chip: float = 1.0,
) -> pd.DataFrame:
    """Compare photonic fabrics against a stronger advanced-chiplet baseline."""
    shape = torus_shape_for_nodes(active_nodes)
    bits = traffic_gib_per_chip * 1024**3 * 8
    variants = [
        ("Advanced chiplet torus", Torus3DTopology(*shape), ADVANCED_CHIPLET_3P5T),
        ("EVK100 photonic torus", Torus3DTopology(*shape), LIGHTMATTER_EVK100),
        ("M1000 optical switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_M1000),
    ]
    rows = []
    for scenario_name, traffic, scenario_key in (
        (
            "all_to_all_activation",
            TrafficBreakdown(input_read_bits=bits, output_write_bits=bits),
            "node_to_node",
        ),
        (
            "remote_weight_refill",
            TrafficBreakdown(weight_read_bits=bits),
            "node_to_dram",
        ),
        (
            "shared_state_multicast",
            TrafficBreakdown(weight_read_bits=bits * active_nodes),
            "shared_dram_multicast",
        ),
    ):
        scenario = scenario_case(scenario_key, active_nodes=active_nodes)
        for architecture, topology, tech in variants:
            cost = topology.evaluate(traffic, scenario, tech)
            rows.append({
                "active_nodes": active_nodes,
                "scenario": scenario_name,
                "architecture": architecture,
                "topology": topology.name,
                "technology": tech.name,
                "energy_uJ_per_chip": cost.total_energy_j * 1e6,
                "latency_us_per_chip": cost.total_latency_s * 1e6,
            })
    df = pd.DataFrame(rows)
    baseline = df[df["architecture"] == "Advanced chiplet torus"][
        ["scenario", "energy_uJ_per_chip", "latency_us_per_chip"]
    ].rename(
        columns={
            "energy_uJ_per_chip": "chiplet_energy_uJ_per_chip",
            "latency_us_per_chip": "chiplet_latency_us_per_chip",
        }
    )
    df = df.merge(baseline, on="scenario", how="left")
    df["energy_reduction_vs_chiplet_pct"] = 100 * (
        1 - df["energy_uJ_per_chip"] / df["chiplet_energy_uJ_per_chip"]
    )
    df["latency_reduction_vs_chiplet_pct"] = 100 * (
        1 - df["latency_us_per_chip"] / df["chiplet_latency_us_per_chip"]
    )
    return df.sort_values(["scenario", "architecture"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full-superpod Gemini-class end-to-end model
# ---------------------------------------------------------------------------

FP8_FLOP_ENERGY_J = 0.035e-12
BF16_FLOP_ENERGY_J = 0.060e-12
INFERENCE_PARAM_BYTES = 1.0
TRAINING_PARAM_BYTES = 2.0
ACTIVATION_BYTES = 2.0
OPTIMIZER_BYTES_PER_PARAM = 8.0


def gemini_proxy_models() -> dict[str, GeminiProxyModel]:
    """
    Public proxy suite for the latest Gemini-class workload.

    Gemini 3/3.1 parameter counts are not public. The proxies below are not
    claims about Google's confidential architecture; they are explicit system
    stress points for dense, MoE, and Deep-Think/long-reasoning regimes.
    """
    return {
        "gemini3_flash_proxy": GeminiProxyModel(
            name="Gemini 3 Flash-class efficient dense proxy",
            model_kind="dense",
            total_params_b=120.0,
            active_params_b=120.0,
            layers=96,
            hidden_size=12_288,
            prompt_tokens=16_384,
            output_tokens=512,
            train_tokens_per_step=16_777_216,
            disclosure_note=(
                "Proxy for the speed-optimized Gemini 3 Flash family; Google publishes "
                "speed/token-efficiency claims but not parameter counts."
            ),
        ),
        "gemini3_1_pro_moe_proxy": GeminiProxyModel(
            name="Gemini 3.1 Pro-class sparse MoE proxy",
            model_kind="moe",
            total_params_b=4_700.0,
            active_params_b=600.0,
            layers=128,
            hidden_size=16_384,
            moe_experts=256,
            active_experts=8,
            prompt_tokens=32_768,
            output_tokens=768,
            train_tokens_per_step=16_777_216,
            disclosure_note=(
                "Uses the public Lightmatter 4.7T-parameter, 256-expert, 8-active MoE "
                "scenario as a Gemini-class frontier proxy; it is not a disclosed "
                "Gemini 3.1 architecture."
            ),
        ),
        "gemini3_deepthink_long_context_proxy": GeminiProxyModel(
            name="Gemini 3 Deep Think long-context proxy",
            model_kind="moe",
            total_params_b=4_700.0,
            active_params_b=750.0,
            layers=160,
            hidden_size=18_432,
            moe_experts=256,
            active_experts=12,
            prompt_tokens=131_072,
            output_tokens=2_048,
            train_tokens_per_step=8_388_608,
            disclosure_note=(
                "Proxy for longer-horizon reasoning workloads where prefill, KV traffic, "
                "and expert routing dominate more of the system budget."
            ),
        ),
    }


def default_superpod_configs(
    active_chips: int = IRONWOOD_SUPERPOD_CHIPS,
) -> list[SuperpodConfig]:
    shape = torus_shape_for_nodes(active_chips)
    return [
        SuperpodConfig(
            name="Ironwood electrical torus + XLA SPMD",
            topology=Torus3DTopology(*shape),
            tech=ELECTRICAL_800G,
            active_chips=active_chips,
            compiler_strategy="xla_spmd_collective",
            topology_family="torus",
            compute_efficiency=0.62,
            network_overlap=0.35,
            description="Public TPU7x/Ironwood torus baseline with conservative XLA SPMD collectives.",
        ),
        SuperpodConfig(
            name="Advanced chiplet torus + XLA SPMD",
            topology=Torus3DTopology(*shape),
            tech=ADVANCED_CHIPLET_3P5T,
            active_chips=active_chips,
            compiler_strategy="xla_spmd_collective",
            topology_family="torus",
            compute_efficiency=0.65,
            network_overlap=0.42,
            description="Stronger electrical/advanced-packaging baseline requested in the prompt.",
        ),
        SuperpodConfig(
            name="EVK100 photonic torus + multicast cache",
            topology=Torus3DTopology(*shape),
            tech=LIGHTMATTER_EVK100,
            active_chips=active_chips,
            compiler_strategy="photonic_multicast_cache",
            topology_family="torus",
            compute_efficiency=0.69,
            network_overlap=0.58,
            tensor_parallel_traffic_scale=0.85,
            remote_weight_traffic_scale=0.20,
            gradient_traffic_scale=0.85,
            moe_traffic_scale=0.62,
            kv_traffic_scale=0.55,
            cache_reuse_steps=64.0,
            description="Keeps TPU cube/torus structure but exposes cheap optical multicast to the compiler.",
        ),
        SuperpodConfig(
            name="M1000 switched optical topology only",
            topology=OpticalCrossbarTopology(),
            tech=LIGHTMATTER_M1000,
            active_chips=active_chips,
            compiler_strategy="xla_spmd_collective",
            topology_family="switched",
            compute_efficiency=0.70,
            network_overlap=0.48,
            description="Ablation: replace the fabric only, without compiler/data-placement changes.",
        ),
        SuperpodConfig(
            name="M1000 + multicast cache",
            topology=OpticalCrossbarTopology(),
            tech=LIGHTMATTER_M1000,
            active_chips=active_chips,
            compiler_strategy="photonic_multicast_cache",
            topology_family="switched",
            compute_efficiency=0.74,
            network_overlap=0.64,
            tensor_parallel_traffic_scale=0.75,
            remote_weight_traffic_scale=0.12,
            gradient_traffic_scale=0.80,
            moe_traffic_scale=0.52,
            kv_traffic_scale=0.45,
            cache_reuse_steps=128.0,
            description="Ablation: switched optics plus multicast cache fills for shared weights/state.",
        ),
        SuperpodConfig(
            name="M1000 + remote pooled HBM",
            topology=OpticalCrossbarTopology(),
            tech=LIGHTMATTER_M1000,
            active_chips=active_chips,
            compiler_strategy="remote_hbm_streaming",
            topology_family="switched",
            compute_efficiency=0.77,
            network_overlap=0.70,
            tensor_parallel_traffic_scale=0.72,
            remote_weight_traffic_scale=0.10,
            gradient_traffic_scale=0.76,
            moe_traffic_scale=0.44,
            kv_traffic_scale=0.34,
            cache_reuse_steps=160.0,
            description="Ablation: compiler treats remote HBM as a cheap logical memory tier.",
        ),
        SuperpodConfig(
            name="M1000 + MoE locality routing",
            topology=OpticalCrossbarTopology(),
            tech=LIGHTMATTER_M1000,
            active_chips=active_chips,
            compiler_strategy="moe_locality_routing",
            topology_family="switched",
            compute_efficiency=0.79,
            network_overlap=0.72,
            tensor_parallel_traffic_scale=0.70,
            remote_weight_traffic_scale=0.10,
            gradient_traffic_scale=0.74,
            moe_traffic_scale=0.22,
            kv_traffic_scale=0.32,
            cache_reuse_steps=160.0,
            description="Ablation: expert placement/routing uses the flat optical fabric to reduce all-to-all pressure.",
        ),
        SuperpodConfig(
            name="M1000 full photonic compiler stack",
            topology=OpticalCrossbarTopology(),
            tech=LIGHTMATTER_M1000,
            active_chips=active_chips,
            compiler_strategy="full_photonic_stack",
            topology_family="switched",
            compute_efficiency=0.84,
            network_overlap=0.78,
            tensor_parallel_traffic_scale=0.62,
            remote_weight_traffic_scale=0.06,
            gradient_traffic_scale=0.55,
            moe_traffic_scale=0.18,
            kv_traffic_scale=0.25,
            cache_reuse_steps=256.0,
            description=(
                "Combined strategy: switched optics, multicast cache fills, pooled HBM, "
                "MoE locality, and communication/computation overlap."
            ),
        ),
    ]


def _model_param_bits(model: GeminiProxyModel, active_only: bool, bytes_per_param: float) -> float:
    params_b = model.active_params_b if active_only else model.total_params_b
    return params_b * 1e9 * bytes_per_param * 8


def _attention_flops(model: GeminiProxyModel, query_tokens: float, key_tokens: float) -> float:
    return 4.0 * model.layers * model.hidden_size * query_tokens * key_tokens


def _network_for_stage(
    config: SuperpodConfig,
    scenario_name: str,
    bits_per_chip: float,
    multicast_fanout: int | None = None,
) -> NetworkCost:
    active = (
        max(1, min(config.active_chips, int(multicast_fanout)))
        if scenario_name == "shared_dram_multicast" and multicast_fanout is not None
        else config.active_chips
    )
    if scenario_name == "shared_dram_multicast":
        traffic = TrafficBreakdown(weight_read_bits=bits_per_chip * active)
    elif scenario_name == "node_to_dram":
        traffic = TrafficBreakdown(weight_read_bits=bits_per_chip)
    else:
        traffic = TrafficBreakdown(
            input_read_bits=bits_per_chip / 2.0,
            output_write_bits=bits_per_chip / 2.0,
        )
    scenario = scenario_case(scenario_name, active_nodes=active)
    return config.topology.evaluate(traffic, scenario, config.tech)


def _stage_row(
    model: GeminiProxyModel,
    config: SuperpodConfig,
    phase: str,
    stage: str,
    stage_description: str,
    compute_flops: float,
    hbm_bits_per_chip: float,
    network_bits_per_chip: float,
    scenario_name: str,
    precision: str,
    traffic_role: str,
    multicast_fanout: int | None = None,
) -> dict[str, float | str | bool]:
    active = config.active_chips
    tflops = (
        config.fp8_tflops_per_chip
        if precision == "fp8"
        else config.bf16_tflops_per_chip
    )
    flop_energy = FP8_FLOP_ENERGY_J if precision == "fp8" else BF16_FLOP_ENERGY_J
    peak_flops = active * tflops * 1e12 * config.compute_efficiency
    compute_latency_s = compute_flops / peak_flops if peak_flops else 0.0
    compute_energy_j = compute_flops * flop_energy

    hbm_energy_j = hbm_bits_per_chip * active * HBM_ENERGY_PER_BIT_J
    hbm_bandwidth_bps = config.hbm_bandwidth_gibps_per_chip * 1024**3 * 8
    hbm_latency_s = hbm_bits_per_chip / hbm_bandwidth_bps if hbm_bandwidth_bps else 0.0

    net = _network_for_stage(config, scenario_name, network_bits_per_chip, multicast_fanout)
    network_energy_j = net.total_energy_j * active
    exposed_network_latency_s = net.total_latency_s * (1.0 - config.network_overlap)
    total_latency_s = max(compute_latency_s, hbm_latency_s) + exposed_network_latency_s
    total_energy_j = compute_energy_j + hbm_energy_j + network_energy_j
    bottleneck = "compute" if compute_latency_s >= hbm_latency_s else "hbm"
    if exposed_network_latency_s > max(compute_latency_s, hbm_latency_s):
        bottleneck = "network"

    return {
        "model": model.name,
        "model_kind": model.model_kind,
        "phase": phase,
        "stage": stage,
        "stage_description": stage_description,
        "configuration": config.name,
        "compiler_strategy": config.compiler_strategy,
        "topology": config.topology.name,
        "technology": config.tech.name,
        "topology_family": config.topology_family,
        "active_chips": active,
        "superpod_cubes": active / IRONWOOD_CUBE_CHIPS,
        "precision": precision,
        "traffic_role": traffic_role,
        "scenario": scenario_name,
        "supports_multicast": config.tech.supports_multicast,
        "compute_flops": compute_flops,
        "hbm_gib_per_chip": hbm_bits_per_chip / 8 / 1024**3,
        "network_gib_per_chip": network_bits_per_chip / 8 / 1024**3,
        "compute_latency_s": compute_latency_s,
        "hbm_latency_s": hbm_latency_s,
        "raw_network_latency_s": net.total_latency_s,
        "exposed_network_latency_s": exposed_network_latency_s,
        "total_latency_s": total_latency_s,
        "compute_energy_j": compute_energy_j,
        "hbm_energy_j": hbm_energy_j,
        "network_energy_j": network_energy_j,
        "total_energy_j": total_energy_j,
        "network_latency_share_pct": 100 * exposed_network_latency_s / total_latency_s if total_latency_s else 0.0,
        "network_energy_share_pct": 100 * network_energy_j / total_energy_j if total_energy_j else 0.0,
        "bottleneck": bottleneck,
        "average_hops": net.detail.get("hops", 0.0) if net.detail else 0.0,
        "contention_factor": config.topology.contention_factor(active, scenario_name),
        "compute_efficiency_pct": config.compute_efficiency * 100,
        "network_overlap_pct": config.network_overlap * 100,
        "configuration_description": config.description,
        "model_disclosure_note": model.disclosure_note,
        "prompt_tokens": model.prompt_tokens,
        "output_tokens": model.output_tokens,
        "train_tokens_per_step": model.train_tokens_per_step,
    }


def evaluate_superpod_stages(
    model: GeminiProxyModel,
    config: SuperpodConfig,
) -> list[dict[str, float | str | bool]]:
    active = config.active_chips
    inference_param_bits_active = _model_param_bits(model, active_only=True, bytes_per_param=INFERENCE_PARAM_BYTES)
    training_param_bits_active = _model_param_bits(model, active_only=True, bytes_per_param=TRAINING_PARAM_BYTES)
    training_param_bits_total = _model_param_bits(model, active_only=False, bytes_per_param=TRAINING_PARAM_BYTES)

    prompt = float(model.prompt_tokens)
    output = float(model.output_tokens)
    train_tokens = float(model.train_tokens_per_step)
    hidden = float(model.hidden_size)
    layers = float(model.layers)
    moe_factor = max(1.0, float(model.active_experts))

    prefill_compute = (
        2.0 * model.active_params_b * 1e9 * prompt
        + _attention_flops(model, prompt, prompt)
    )
    prefill_hbm_bits = (
        inference_param_bits_active / active
        + prompt * hidden * layers * ACTIVATION_BYTES * 8 / active
    )
    prefill_network_bits = (
        prompt * hidden * ACTIVATION_BYTES * 8 / active
        * math.log2(max(active, 2))
        * config.tensor_parallel_traffic_scale
    )

    decode_compute = (
        2.0 * model.active_params_b * 1e9 * output
        + _attention_flops(model, output, prompt + output / 2.0)
    )
    kv_bits = (
        (prompt + output / 2.0)
        * output
        * hidden
        * ACTIVATION_BYTES
        * 8
        / active
        * config.kv_traffic_scale
    )
    decode_weight_bits = (
        inference_param_bits_active
        / active
        / max(1.0, config.cache_reuse_steps)
        * config.remote_weight_traffic_scale
    )
    decode_hbm_bits = (
        inference_param_bits_active / active / max(1.0, min(config.cache_reuse_steps, 64.0))
        + kv_bits
    )
    moe_dispatch_bits = (
        output
        * hidden
        * ACTIVATION_BYTES
        * 8
        * moe_factor
        / active
        * config.moe_traffic_scale
        if model.model_kind == "moe"
        else 0.0
    )

    training_compute = (
        6.0 * model.active_params_b * 1e9 * train_tokens
        + 3.0 * _attention_flops(model, train_tokens / 4096.0, 4096.0)
    )
    training_hbm_bits = (
        training_param_bits_active * 3.0 / active
        + train_tokens * hidden * layers * ACTIVATION_BYTES * 8 / active
    )
    activation_exchange_bits = (
        train_tokens
        * hidden
        * ACTIVATION_BYTES
        * 8
        / active
        * math.log2(max(active, 2))
        * config.tensor_parallel_traffic_scale
    )
    gradient_bits = (
        training_param_bits_active
        / active
        * 2.0
        * config.gradient_traffic_scale
    )
    training_moe_bits = (
        train_tokens
        * hidden
        * ACTIVATION_BYTES
        * 8
        * moe_factor
        / active
        * config.moe_traffic_scale
        if model.model_kind == "moe"
        else 0.0
    )
    optimizer_bits = (
        model.total_params_b
        * 1e9
        * OPTIMIZER_BYTES_PER_PARAM
        * 8
        / active
        * 0.02
        * config.gradient_traffic_scale
    )

    return [
        _stage_row(
            model, config, "inference", "prefill",
            "Prompt prefill: dense/MoE active-parameter compute plus tensor-parallel activation exchange.",
            prefill_compute, prefill_hbm_bits, prefill_network_bits,
            "node_to_node", "fp8", "activation_exchange",
        ),
        _stage_row(
            model, config, "inference", "decode_kv_attention",
            "Decode attention: KV-cache reads and sequence-length-sensitive attention traffic.",
            decode_compute * 0.20, decode_hbm_bits * 0.35, kv_bits,
            "node_to_dram", "fp8", "kv_cache_fetch",
        ),
        _stage_row(
            model, config, "inference", "decode_weight_stream",
            "Decode FFN/expert state: remote pooled-HBM or multicast cache refill of hot weights.",
            decode_compute * 0.80, decode_hbm_bits * 0.65, decode_weight_bits,
            "shared_dram_multicast", "fp8", "weight_or_expert_refill",
            multicast_fanout=IRONWOOD_CUBE_CHIPS,
        ),
        _stage_row(
            model, config, "inference", "moe_dispatch",
            "Sparse expert dispatch and combine. Dense proxies have zero traffic for this stage.",
            0.03 * decode_compute if model.model_kind == "moe" else 0.0,
            moe_dispatch_bits * 0.25,
            moe_dispatch_bits,
            "node_to_node", "fp8", "moe_all_to_all",
        ),
        _stage_row(
            model, config, "training", "forward_backward_compute",
            "Training forward/backward/weight-gradient compute for one global superpod step.",
            training_compute, training_hbm_bits, activation_exchange_bits,
            "node_to_node", "bf16", "activation_exchange",
        ),
        _stage_row(
            model, config, "training", "gradient_sync",
            "Data-parallel gradient reduce-scatter/all-reduce over active parameters.",
            0.0, training_param_bits_active / active, gradient_bits,
            "node_to_node", "bf16", "gradient_collective",
        ),
        _stage_row(
            model, config, "training", "moe_dispatch",
            "Training-time expert token routing and combine traffic.",
            0.02 * training_compute if model.model_kind == "moe" else 0.0,
            training_moe_bits * 0.25,
            training_moe_bits,
            "node_to_node", "bf16", "moe_all_to_all",
        ),
        _stage_row(
            model, config, "training", "optimizer_state",
            "Optimizer-state sharding/checkpoint traffic amortized per step.",
            0.0, training_param_bits_total / active * 0.02, optimizer_bits,
            "node_to_dram", "bf16", "optimizer_state",
        ),
    ]


def superpod_stage_study(
    model_names: Iterable[str] = ("gemini3_1_pro_moe_proxy", "gemini3_deepthink_long_context_proxy"),
    configs: Iterable[SuperpodConfig] | None = None,
    active_chips: int = IRONWOOD_SUPERPOD_CHIPS,
) -> pd.DataFrame:
    models = gemini_proxy_models()
    configs = list(configs or default_superpod_configs(active_chips))
    rows: list[dict[str, float | str | bool]] = []
    for model_name in model_names:
        model = models[model_name]
        for config in configs:
            rows.extend(evaluate_superpod_stages(model, config))
    return pd.DataFrame(rows)


def superpod_end_to_end_summary(
    stage_df: pd.DataFrame,
    baseline_configuration: str = "Ironwood electrical torus + XLA SPMD",
) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()
    grouped = (
        stage_df.groupby(["model", "model_kind", "phase", "configuration", "compiler_strategy", "technology"])
        .agg(
            total_latency_s=("total_latency_s", "sum"),
            total_energy_j=("total_energy_j", "sum"),
            compute_energy_j=("compute_energy_j", "sum"),
            hbm_energy_j=("hbm_energy_j", "sum"),
            network_energy_j=("network_energy_j", "sum"),
            exposed_network_latency_s=("exposed_network_latency_s", "sum"),
            network_gib_per_chip=("network_gib_per_chip", "sum"),
            active_chips=("active_chips", "first"),
            prompt_tokens=("prompt_tokens", "first"),
            output_tokens=("output_tokens", "first"),
            train_tokens_per_step=("train_tokens_per_step", "first"),
        )
        .reset_index()
    )
    grouped["network_energy_share_pct"] = 100 * grouped["network_energy_j"] / grouped["total_energy_j"]
    grouped["network_latency_share_pct"] = 100 * grouped["exposed_network_latency_s"] / grouped["total_latency_s"]
    grouped["tokens_per_step_or_request"] = grouped.apply(
        lambda row: (
            row["train_tokens_per_step"]
            if row["phase"] == "training"
            else row["prompt_tokens"] + row["output_tokens"]
        ),
        axis=1,
    )
    grouped["throughput_tokens_per_s"] = grouped["tokens_per_step_or_request"] / grouped["total_latency_s"]
    baseline = grouped[grouped["configuration"] == baseline_configuration][
        ["model", "phase", "total_latency_s", "total_energy_j", "throughput_tokens_per_s"]
    ].rename(
        columns={
            "total_latency_s": "baseline_total_latency_s",
            "total_energy_j": "baseline_total_energy_j",
            "throughput_tokens_per_s": "baseline_throughput_tokens_per_s",
        }
    )
    grouped = grouped.merge(baseline, on=["model", "phase"], how="left")
    grouped["speedup_vs_baseline"] = grouped["baseline_total_latency_s"] / grouped["total_latency_s"]
    grouped["energy_reduction_vs_baseline_pct"] = 100 * (
        1 - grouped["total_energy_j"] / grouped["baseline_total_energy_j"]
    )
    grouped["throughput_gain_vs_baseline_pct"] = 100 * (
        grouped["throughput_tokens_per_s"] / grouped["baseline_throughput_tokens_per_s"] - 1
    )
    return grouped.sort_values(["model", "phase", "configuration"]).reset_index(drop=True)


def superpod_ablation_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the end-to-end summary into concise per-technique ablation rows."""
    if summary_df.empty:
        return pd.DataFrame()
    technique_labels = {
        "Ironwood electrical torus + XLA SPMD": "baseline",
        "Advanced chiplet torus + XLA SPMD": "advanced chiplet",
        "EVK100 photonic torus + multicast cache": "photonic torus + multicast",
        "M1000 switched optical topology only": "switched topology only",
        "M1000 + multicast cache": "multicast cache",
        "M1000 + remote pooled HBM": "remote pooled HBM",
        "M1000 + MoE locality routing": "MoE locality routing",
        "M1000 full photonic compiler stack": "full photonic compiler stack",
    }
    df = summary_df.copy()
    df["technique"] = df["configuration"].map(technique_labels).fillna(df["configuration"])
    df["speedup_increment_vs_previous"] = 1.0
    order = list(technique_labels)
    rank = {name: i for i, name in enumerate(order)}
    df["configuration_rank"] = df["configuration"].map(rank).fillna(999).astype(int)
    rows = []
    for (model, phase), group in df.sort_values("configuration_rank").groupby(["model", "phase"]):
        prev_speedup = None
        for row in group.itertuples(index=False):
            speedup = float(row.speedup_vs_baseline)
            increment = speedup / prev_speedup if prev_speedup else 1.0
            prev_speedup = speedup
            item = row._asdict()
            item["speedup_increment_vs_previous"] = increment
            rows.append(item)
    return pd.DataFrame(rows).drop(columns=["configuration_rank"], errors="ignore")


def superpod_context_scaling_study(
    model_name: str = "gemini3_1_pro_moe_proxy",
    context_tokens: Iterable[int] = (4_096, 16_384, 32_768, 131_072, 1_048_576),
    output_tokens: int = 512,
    active_chips: int = IRONWOOD_SUPERPOD_CHIPS,
) -> pd.DataFrame:
    """Inference-only context-length scaling for baseline vs full photonic stack."""
    base_model = gemini_proxy_models()[model_name]
    configs = [
        c for c in default_superpod_configs(active_chips)
        if c.name in {
            "Ironwood electrical torus + XLA SPMD",
            "Advanced chiplet torus + XLA SPMD",
            "M1000 switched optical topology only",
            "M1000 full photonic compiler stack",
        }
    ]
    rows = []
    for ctx in context_tokens:
        model = GeminiProxyModel(
            name=f"{base_model.name} @ {ctx} ctx",
            model_kind=base_model.model_kind,
            total_params_b=base_model.total_params_b,
            active_params_b=base_model.active_params_b,
            layers=base_model.layers,
            hidden_size=base_model.hidden_size,
            moe_experts=base_model.moe_experts,
            active_experts=base_model.active_experts,
            prompt_tokens=ctx,
            output_tokens=output_tokens,
            train_tokens_per_step=base_model.train_tokens_per_step,
            disclosure_note=base_model.disclosure_note,
        )
        stage_df = pd.DataFrame(
            row for config in configs for row in evaluate_superpod_stages(model, config)
        )
        summary = superpod_end_to_end_summary(stage_df)
        summary = summary[summary["phase"] == "inference"].copy()
        summary["context_tokens"] = ctx
        summary["base_model_key"] = model_name
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows).sort_values(["context_tokens", "configuration"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# TPU Cube ICI Replacement Study
# ---------------------------------------------------------------------------

def tpu_cube_case_study() -> pd.DataFrame:
    """
    Model the energy and latency impact of replacing TPU v5p ICI with
    Lightmatter photonic interconnects across three representative traffic cases.

    Traffic volumes are derived from the public TPU v5p 4×4×4 cube:
      - all_to_all_shard_exchange : large activation shard exchange during pipeline flush
      - remote_weight_refill      : weight streaming from pooled memory (Mixture-of-Experts)
      - shared_weight_multicast   : common expert state delivered to many chips
    """
    # GiB per chip assuming 64-chip cube
    CHIPS = 64
    GIB_TO_BITS = 1024**3 * 8

    traffic_cases = [
        ("all_to_all_shard_exchange",  64 * GIB_TO_BITS / CHIPS),   # 64 GiB ÷ 64 chips
        ("remote_weight_refill",       16 * GIB_TO_BITS / CHIPS),   # 16 GiB ÷ 64 chips
        ("shared_weight_multicast",     8 * GIB_TO_BITS / CHIPS),   #  8 GiB ÷ 64 chips
    ]

    # Topology objects
    electrical_torus = Torus3DTopology(4, 4, 4)
    optical_torus = Torus3DTopology(4, 4, 4)    # same topology, different link tech
    optical_crossbar = OpticalCrossbarTopology()

    # Per-axis one-way bandwidth inferred from public ICI specs
    tpu_v5p_axis_one_way_bps = 800e9            # 1200 GB/s bidir / 6 links / 2 dirs
    photonic_torus_evk100_bps = 3200e9          # EVK100: 3.2 Tbps one-way
    l200_one_way_bps = 16e12                    # L200: 32 Tbps total / 2 dirs
    l200x_one_way_bps = 32e12                   # L200X: 64 Tbps total / 2 dirs
    m1000_one_way_bps = 57.3e12                 # M1000: 114.6 Tbps total / 2 dirs

    variants: list[tuple[str, Topology, InterconnectTech]] = [
        (
            "TPU v5p electrical torus",
            electrical_torus,
            InterconnectTech(
                name="TPU v5p ICI electrical",
                link_bandwidth_bps=tpu_v5p_axis_one_way_bps,
                energy_per_bit_j=10e-12,
                router_latency_s=5e-9,
                supports_multicast=False,
                notes=(
                    "800 Gb/s one-way per axis inferred from public 1200 GB/s bidirectional ICI "
                    "split across six nearest-neighbour torus links. Energy 10 pJ/bit assumed."
                ),
            ),
        ),
        (
            "Photonic torus (EVK100)",
            optical_torus,
            InterconnectTech(
                name="Photonic torus EVK100",
                link_bandwidth_bps=photonic_torus_evk100_bps,
                energy_per_bit_j=1.9e-12,
                router_latency_s=1.5e-9,
                supports_multicast=True,
                notes="4×4×4 torus topology preserved; each axis link upgraded to EVK100-class.",
            ),
        ),
        (
            "Photonic switched fabric (L200)",
            optical_crossbar,
            InterconnectTech(
                name="Photonic switched L200",
                link_bandwidth_bps=l200_one_way_bps,
                energy_per_bit_j=4.5e-12,
                router_latency_s=2.0e-9,
                supports_multicast=True,
                notes="Optical crossbar via L200; 16 Tbps one-way per chip from 32 Tbps total I/O.",
            ),
        ),
        (
            "Photonic switched fabric (L200X)",
            optical_crossbar,
            InterconnectTech(
                name="Photonic switched L200X",
                link_bandwidth_bps=l200x_one_way_bps,
                energy_per_bit_j=4.8e-12,
                router_latency_s=2.0e-9,
                supports_multicast=True,
                notes="Optical crossbar via L200X; 32 Tbps one-way per chip from 64 Tbps total I/O.",
            ),
        ),
        (
            "Photonic switched fabric (M1000)",
            optical_crossbar,
            InterconnectTech(
                name="Photonic switched M1000",
                link_bandwidth_bps=m1000_one_way_bps,
                energy_per_bit_j=2.3e-12,
                router_latency_s=2.5e-9,
                supports_multicast=True,
                notes="M1000 OCS-based crossbar; 57.3 Tbps one-way from 114.6 Tbps aggregate.",
            ),
        ),
    ]

    rows = []
    for traffic_name, bits_per_chip in traffic_cases:
        if traffic_name == "all_to_all_shard_exchange":
            traffic = TrafficBreakdown(
                input_read_bits=bits_per_chip,
                output_write_bits=bits_per_chip,
            )
            scenario_name = "node_to_node"
        elif traffic_name == "remote_weight_refill":
            traffic = TrafficBreakdown(weight_read_bits=bits_per_chip)
            scenario_name = "node_to_dram"
        else:  # shared_weight_multicast
            traffic = TrafficBreakdown(weight_read_bits=bits_per_chip)
            scenario_name = "shared_dram_multicast"

        scenario = Scenario(
            name=scenario_name,
            description=SCENARIOS[scenario_name]["description"],
            comm_pattern=scenario_name,
            active_nodes=CHIPS,
        )
        for variant_name, topology, tech in variants:
            net = topology.evaluate(traffic, scenario, tech)
            rows.append({
                "case": traffic_name,
                "variant": variant_name,
                "topology": topology.name,
                "technology": tech.name,
                "effective_link_one_way_tbps": tech.link_bandwidth_bps / 1e12,
                "energy_pJ_per_bit": tech.energy_per_bit_j * 1e12,
                "energy_uJ_per_chip": net.total_energy_j * 1e6,
                "latency_us_per_chip": net.total_latency_s * 1e6,
                "notes": tech.notes,
            })
    return pd.DataFrame(rows)


def tpu_improvement_frame(
    tpu_df: pd.DataFrame,
    baseline_variant: str = "TPU v5p electrical torus",
) -> pd.DataFrame:
    baseline = tpu_df[tpu_df["variant"] == baseline_variant].copy()
    others = tpu_df[tpu_df["variant"] != baseline_variant].copy()
    merged = others.merge(baseline, on="case", suffixes=("", "_baseline"))
    merged["energy_reduction_pct"] = 100 * (
        1 - merged["energy_uJ_per_chip"] / merged["energy_uJ_per_chip_baseline"]
    )
    merged["latency_reduction_pct"] = 100 * (
        1 - merged["latency_us_per_chip"] / merged["latency_us_per_chip_baseline"]
    )
    return merged.sort_values(["case", "variant"]).reset_index(drop=True)


def tpu_cube_mapping_study(
    workload_names: Iterable[str] = ("kv_attention",),
    scenario_names: Iterable[str] = DEFAULT_SCENARIO_NAMES,
    max_iterations: int = 4,
    node_shape: tuple[int, int, int] = (4, 4, 4),
    hbm_gib: int = 95,
) -> pd.DataFrame:
    """
    AccelForge mapping-adaptation study for a TPU-v5p-style 4x4x4 cube.

    Unlike `tpu_cube_case_study`, which is a pure network calculation over large
    traffic cases, this function runs the iterative AccelForge mapper. It answers
    the project-specific question: when the cube interconnect changes, how much
    does the fixed-point mapping objective move, and does the compact
    tile/residency signature change?
    """
    active_nodes = math.prod(node_shape)
    variants: list[tuple[str, Topology, InterconnectTech]] = [
        ("TPU cube electrical torus", Torus3DTopology(*node_shape), ELECTRICAL_800G),
        ("TPU cube EVK100 photonic torus", Torus3DTopology(*node_shape), LIGHTMATTER_EVK100),
        ("TPU cube L200 switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_L200),
        ("TPU cube L200X switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_L200X),
        ("TPU cube M1000 switched fabric", OpticalCrossbarTopology(), LIGHTMATTER_M1000),
    ]

    rows = []
    for workload_name in workload_names:
        workload = workload_case(workload_name)
        for scenario_name in scenario_names:
            scenario = scenario_case(scenario_name, active_nodes=active_nodes)
            for variant, topology, tech in variants:
                history, _ = solve_iteratively(
                    workload,
                    scenario,
                    topology,
                    tech,
                    max_iterations=max_iterations,
                    node_shape=node_shape,
                    hbm_gib=hbm_gib,
                )
                first = history[0]
                final = history[-1]
                first_plus_net_energy = first.total_energy_j + first.network.total_energy_j
                first_plus_net_latency = first.total_latency_s + first.network.total_latency_s
                rows.append({
                    "variant": variant,
                    "workload": workload.name,
                    "scenario": scenario.name,
                    "active_nodes": active_nodes,
                    "node_shape": "x".join(str(v) for v in node_shape),
                    "topology": topology.name,
                    "technology": tech.name,
                    "iterations": len(history),
                    "baseline_energy_uJ": first.total_energy_j * 1e6,
                    "baseline_latency_us": first.total_latency_s * 1e6,
                    "first_run_plus_network_energy_uJ": first_plus_net_energy * 1e6,
                    "first_run_plus_network_latency_us": first_plus_net_latency * 1e6,
                    "final_energy_uJ": final.total_energy_j * 1e6,
                    "final_latency_us": final.total_latency_s * 1e6,
                    "network_energy_uJ": final.network.total_energy_j * 1e6,
                    "network_latency_us": final.network.total_latency_s * 1e6,
                    "network_delta_energy_pct": 100 * (
                        final.total_energy_j / first.total_energy_j - 1.0
                    ),
                    "mapping_sensitivity_energy_pct": (
                        100 * (final.total_energy_j / first_plus_net_energy - 1.0)
                        if first_plus_net_energy > 0 else 0.0
                    ),
                    "mapping_sensitivity_latency_pct": (
                        100 * (final.total_latency_s / first_plus_net_latency - 1.0)
                        if first_plus_net_latency > 0 else 0.0
                    ),
                    "first_remote_bits_kib": first.traffic.total_bits / 1024.0,
                    "final_remote_bits_kib": final.traffic.total_bits / 1024.0,
                    "remote_bits_change_pct": (
                        100 * (final.traffic.total_bits / first.traffic.total_bits - 1.0)
                        if first.traffic.total_bits > 0 else 0.0
                    ),
                    "mapping_changed_during_convergence": (
                        first.mapping_signature != final.mapping_signature
                    ),
                    "first_mapping_signature": first.mapping_signature,
                    "final_mapping_signature": final.mapping_signature,
                    "spatial_NodeX_usage": final.mapping_features.get(
                        "spatial::NodeArray::NodeX", 0.0
                    ),
                    "spatial_NodeY_usage": final.mapping_features.get(
                        "spatial::NodeArray::NodeY", 0.0
                    ),
                    "spatial_NodeZ_usage": final.mapping_features.get(
                        "spatial::NodeArray::NodeZ", 0.0
                    ),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    baseline = df[df["variant"] == "TPU cube electrical torus"][
        ["workload", "scenario", "final_energy_uJ", "final_latency_us", "final_mapping_signature"]
    ].rename(
        columns={
            "final_energy_uJ": "electrical_final_energy_uJ",
            "final_latency_us": "electrical_final_latency_us",
            "final_mapping_signature": "electrical_mapping_signature",
        }
    )
    df = df.merge(baseline, on=["workload", "scenario"], how="left")
    df["energy_reduction_vs_electrical_pct"] = 100 * (
        1 - df["final_energy_uJ"] / df["electrical_final_energy_uJ"]
    )
    df["latency_reduction_vs_electrical_pct"] = 100 * (
        1 - df["final_latency_us"] / df["electrical_final_latency_us"]
    )
    df["mapping_differs_from_electrical"] = (
        df["final_mapping_signature"] != df["electrical_mapping_signature"]
    )
    return df.sort_values(["scenario", "workload", "variant"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary report generator
# ---------------------------------------------------------------------------

def generate_summary_markdown(
    cluster_df: pd.DataFrame,
    cluster_improvement_df: pd.DataFrame,
    tpu_df: pd.DataFrame,
    tpu_improvement_df: pd.DataFrame,
    tpu_mapping_df: pd.DataFrame | None = None,
    optimal_mapping_df: pd.DataFrame | None = None,
    compiler_strategy_df: pd.DataFrame | None = None,
    optimal_compiler_df: pd.DataFrame | None = None,
    scaling_exponents_df: pd.DataFrame | None = None,
    chiplet_df: pd.DataFrame | None = None,
    superpod_summary_df: pd.DataFrame | None = None,
    superpod_ablation_df: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = [
        "# Lightmatter Final Project — Summary",
        "",
        "Generated from the AccelForge + photonic-interconnect study pipeline.",
        "",
        "## Executive Takeaways",
        "",
        "- The raw network model and the AccelForge fixed-point mapper both show the same direction: optical links help most when traffic leaves a single local HBM bank.",
        "- Re-running the mapper is explicitly measured: the pipeline reports both fixed-point tile/residency sensitivity and the higher-level distributed mapping choice that becomes optimal under each interconnect.",
        "- A separate compiler/dataflow strategy model shows the more important effect: photonic multicast can make software-managed cache fills and shared-state broadcast preferable to classic SPMD activation exchange.",
        "- Scaling-law sweeps show that flat switched photonic fabrics keep hop count and contention nearly constant while torus and chiplet fabrics accumulate hop/contention cost as cubes grow toward superpod scale.",
        "- The full-superpod study models Gemini 3/3.1-class public proxy workloads on a 9216-chip Ironwood pod and reports end-to-end inference/training speedups by stage and by optimization ablation.",
        "- The TPU cube study separates two design points: a conservative EVK100 photonic torus and higher-bandwidth switched-fabric options (L200/L200X/M1000).",
        "- EVK100 public collateral includes both a 1.9 pJ/bit Passage-family value and a more conservative 3.2 pJ/bit EVK100 technical-spec value; this project uses 1.9 pJ/bit as the main optimistic/reference point and includes sensitivity hooks for alternatives.",
        "",
        "## Technology Catalog",
        "",
        "| Technology | BW (one-way) | Energy (pJ/bit) | Multicast |",
        "|---|---|---|---|",
        f"| Electrical 800G | {ELECTRICAL_800G.link_bandwidth_bps/1e9:.0f} Gbps | "
        f"{ELECTRICAL_800G.energy_per_bit_j*1e12:.1f} | No |",
        f"| {ADVANCED_CHIPLET_3P5T.name} | {ADVANCED_CHIPLET_3P5T.link_bandwidth_bps/1e12:.1f} Tbps | "
        f"{ADVANCED_CHIPLET_3P5T.energy_per_bit_j*1e12:.1f} | No |",
    ]
    for label, tech in ALL_LIGHTMATTER_TECHS.items():
        lines.append(
            f"| {tech.name} | {tech.link_bandwidth_bps/1e12:.1f} Tbps | "
            f"{tech.energy_per_bit_j*1e12:.1f} | Yes |"
        )

    lines.extend(["", "## Cluster Study Highlights (16-node system)", ""])
    for scenario in cluster_improvement_df["scenario"].unique():
        sdf = cluster_improvement_df[cluster_improvement_df["scenario"] == scenario]
        be = sdf.loc[sdf["energy_reduction_pct"].idxmax()]
        bl = sdf.loc[sdf["latency_reduction_pct"].idxmax()]
        lines.extend([
            f"### {scenario}",
            "",
            f"- Best energy reduction: `{be['workload']}` → `{be['energy_reduction_pct']:.2f}%`",
            f"- Best latency reduction: `{bl['workload']}` → `{bl['latency_reduction_pct']:.2f}%`",
            "",
        ])

    lines.extend(["## TPU Cube Highlights (64-node 4×4×4 cube)", ""])
    for variant in tpu_improvement_df["variant"].unique():
        vdf = tpu_improvement_df[tpu_improvement_df["variant"] == variant]
        be = vdf.loc[vdf["energy_reduction_pct"].idxmax()]
        bl = vdf.loc[vdf["latency_reduction_pct"].idxmax()]
        lines.extend([
            f"### {variant}",
            "",
            f"- Best energy reduction: `{be['case']}` → `{be['energy_reduction_pct']:.2f}%`",
            f"- Best latency reduction: `{bl['case']}` → `{bl['latency_reduction_pct']:.2f}%`",
            "",
        ])

    if tpu_mapping_df is not None and not tpu_mapping_df.empty:
        lines.extend(["## TPU Cube Mapping Adaptation (AccelForge fixed point)", ""])
        mapping_cases = tpu_mapping_df[
            tpu_mapping_df["variant"] != "TPU cube electrical torus"
        ].copy()
        changed = mapping_cases[mapping_cases["mapping_differs_from_electrical"]]
        best_energy = mapping_cases.loc[
            mapping_cases["energy_reduction_vs_electrical_pct"].idxmax()
        ]
        lines.extend([
            f"- Best mapping-aware TPU-cube energy reduction: `{best_energy['variant']}` on "
            f"`{best_energy['workload']} / {best_energy['scenario']}` → "
            f"`{best_energy['energy_reduction_vs_electrical_pct']:.2f}%`.",
            f"- Mapping signatures differed from the electrical baseline in "
            f"`{len(changed)}` of `{len(mapping_cases)}` photonic mapping cases.",
            f"- Example final mapping signature: `{best_energy['final_mapping_signature']}`",
            "",
        ])

    if optimal_mapping_df is not None and not optimal_mapping_df.empty:
        lines.extend([
            "## Optimal Distributed Mapping Choices",
            "",
            "Here `scenario` is treated as the distributed inference mapping choice: node-to-node activation exchange, node-to-DRAM remote fetch, or shared-DRAM multicast.",
            "",
            "| Workload | Architecture | Optimal mapping | Changed vs baseline |",
            "|---|---|---|---|",
        ])
        for row in optimal_mapping_df.itertuples(index=False):
            changed = getattr(row, "optimal_mapping_changed_vs_baseline", False)
            lines.append(
                f"| {row.workload} | {row.architecture} | "
                f"{row.optimal_distributed_mapping} | {bool(changed)} |"
            )
        lines.append("")

    if optimal_compiler_df is not None and not optimal_compiler_df.empty:
        lines.extend([
            "## Compiler/Dataflow Strategy Choices",
            "",
            "This layer models XLA/Pallas-visible choices above the inner AccelForge tile mapper.",
            "",
            "| Workload | Architecture | Optimal strategy | Changed vs TPU electrical | Energy gain vs baseline |",
            "|---|---|---|---|---|",
        ])
        for row in optimal_compiler_df.itertuples(index=False):
            lines.append(
                f"| {row.workload} | {row.architecture} | {row.strategy_label} | "
                f"{bool(row.strategy_changed_vs_baseline)} | "
                f"{row.energy_gain_vs_baseline_strategy_pct:.2f}% |"
            )
        lines.append("")

    if scaling_exponents_df is not None and not scaling_exponents_df.empty:
        lines.extend([
            "## Scaling-Law Exponents",
            "",
            "Lower exponents mean better scaling as the system grows from one 64-chip cube toward 9216 chips.",
            "",
            "| Architecture | Scenario | Metric | Exponent |",
            "|---|---|---|---|",
        ])
        for row in scaling_exponents_df.itertuples(index=False):
            lines.append(
                f"| {row.architecture} | {row.scenario} | {row.metric} | "
                f"{row.scaling_exponent:.3f} |"
            )
        lines.append("")

    if chiplet_df is not None and not chiplet_df.empty:
        lines.extend([
            "## Advanced Chiplet Baseline",
            "",
            "The chiplet baseline uses the requested 3.5 Tbps one-way, 6 pJ/bit per-hop model.",
            "",
            "| Scenario | Architecture | Energy reduction vs chiplet | Latency reduction vs chiplet |",
            "|---|---|---|---|",
        ])
        non_baseline = chiplet_df[chiplet_df["architecture"] != "Advanced chiplet torus"]
        for row in non_baseline.itertuples(index=False):
            lines.append(
                f"| {row.scenario} | {row.architecture} | "
                f"{row.energy_reduction_vs_chiplet_pct:.2f}% | "
                f"{row.latency_reduction_vs_chiplet_pct:.2f}% |"
            )
        lines.append("")

    if superpod_summary_df is not None and not superpod_summary_df.empty:
        lines.extend([
            "## Full-Superpod Gemini-Class End-to-End Study",
            "",
            "Google has not published Gemini 3/3.1 parameter counts, so this section reports explicit dense/MoE proxies rather than claiming confidential model internals.",
            "",
            "| Model | Phase | Configuration | Speedup vs electrical | Energy reduction | Network latency share | Throughput tokens/s |",
            "|---|---|---|---|---|---|---|",
        ])
        plot_df = superpod_summary_df.sort_values(
            ["model", "phase", "speedup_vs_baseline"],
            ascending=[True, True, False],
        )
        for row in plot_df.itertuples(index=False):
            if row.configuration not in (
                "Ironwood electrical torus + XLA SPMD",
                "Advanced chiplet torus + XLA SPMD",
                "M1000 switched optical topology only",
                "M1000 full photonic compiler stack",
            ):
                continue
            lines.append(
                f"| {row.model} | {row.phase} | {row.configuration} | "
                f"{row.speedup_vs_baseline:.2f}x | "
                f"{row.energy_reduction_vs_baseline_pct:.2f}% | "
                f"{row.network_latency_share_pct:.2f}% | "
                f"{row.throughput_tokens_per_s:.3g} |"
            )
        lines.append("")

    if superpod_ablation_df is not None and not superpod_ablation_df.empty:
        lines.extend([
            "## Superpod Optimization Ablations",
            "",
            "Rows show how much each step contributes relative to the electrical Ironwood baseline.",
            "",
            "| Model | Phase | Technique | Speedup vs baseline | Increment vs previous | Network energy share |",
            "|---|---|---|---|---|---|",
        ])
        best_model = "Gemini 3.1 Pro-class sparse MoE proxy"
        ablation = superpod_ablation_df[
            superpod_ablation_df["model"].eq(best_model)
        ].copy()
        for row in ablation.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.phase} | {row.technique} | "
                f"{row.speedup_vs_baseline:.2f}x | "
                f"{row.speedup_increment_vs_previous:.2f}x | "
                f"{row.network_energy_share_pct:.2f}% |"
            )
        lines.append("")

    lines.extend(["## Sources", ""])
    for _, row in source_frame().iterrows():
        lines.append(f"- **{row['source']}** ({row['date']}): {row['summary']}\n  {row['url']}")

    return "\n".join(lines) + "\n"


def executive_summary_frame(
    cluster_improvement_df: pd.DataFrame,
    tpu_improvement_df: pd.DataFrame,
    tpu_mapping_df: pd.DataFrame,
    optimal_mapping_df: pd.DataFrame | None = None,
    optimal_compiler_df: pd.DataFrame | None = None,
    scaling_exponents_df: pd.DataFrame | None = None,
    chiplet_df: pd.DataFrame | None = None,
    superpod_summary_df: pd.DataFrame | None = None,
    superpod_ablation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Condense the study into a few presentation-ready headline metrics."""
    rows: list[dict[str, str | float]] = []
    if not cluster_improvement_df.empty:
        best_cluster_energy = cluster_improvement_df.loc[
            cluster_improvement_df["energy_reduction_pct"].idxmax()
        ]
        best_cluster_latency = cluster_improvement_df.loc[
            cluster_improvement_df["latency_reduction_pct"].idxmax()
        ]
        rows.extend([
            {
                "category": "16-node AccelForge cluster",
                "metric": "Best energy reduction",
                "case": f"{best_cluster_energy['workload']} / {best_cluster_energy['scenario']}",
                "value": best_cluster_energy["energy_reduction_pct"],
                "unit": "%",
                "interpretation": "Optical crossbar vs electrical 4x4 mesh.",
            },
            {
                "category": "16-node AccelForge cluster",
                "metric": "Best latency reduction",
                "case": f"{best_cluster_latency['workload']} / {best_cluster_latency['scenario']}",
                "value": best_cluster_latency["latency_reduction_pct"],
                "unit": "%",
                "interpretation": "Network-aware fixed-point mapping result.",
            },
        ])

    if not tpu_improvement_df.empty:
        best_tpu_network = tpu_improvement_df.loc[
            tpu_improvement_df["energy_reduction_pct"].idxmax()
        ]
        rows.append({
            "category": "TPU cube pure network",
            "metric": "Best energy reduction",
            "case": f"{best_tpu_network['variant']} / {best_tpu_network['case']}",
            "value": best_tpu_network["energy_reduction_pct"],
            "unit": "%",
            "interpretation": "Large traffic-case model vs TPU v5p electrical torus.",
        })

    if not tpu_mapping_df.empty:
        photonic = tpu_mapping_df[tpu_mapping_df["variant"] != "TPU cube electrical torus"]
        if not photonic.empty:
            best_tpu_mapping = photonic.loc[
                photonic["energy_reduction_vs_electrical_pct"].idxmax()
            ]
            rows.extend([
                {
                    "category": "TPU cube AccelForge mapping",
                    "metric": "Best energy reduction",
                    "case": (
                        f"{best_tpu_mapping['variant']} / "
                        f"{best_tpu_mapping['workload']} / {best_tpu_mapping['scenario']}"
                    ),
                    "value": best_tpu_mapping["energy_reduction_vs_electrical_pct"],
                    "unit": "%",
                    "interpretation": "Includes fixed-point remapping after link-cost feedback.",
                },
                {
                    "category": "TPU cube AccelForge mapping",
                    "metric": "Photonic cases with new mapping signature",
                    "case": "photonic variants vs electrical torus",
                    "value": float(photonic["mapping_differs_from_electrical"].sum()),
                    "unit": f"of {len(photonic)}",
                    "interpretation": "Counts final mappings whose compact signature differs.",
                },
            ])

    if optimal_mapping_df is not None and not optimal_mapping_df.empty:
        changed = (
            optimal_mapping_df[
                optimal_mapping_df["optimal_mapping_changed_vs_baseline"].astype(bool)
            ]
            if "optimal_mapping_changed_vs_baseline" in optimal_mapping_df.columns
            else pd.DataFrame()
        )
        if not changed.empty:
            example = changed.iloc[0]
            rows.append({
                "category": "Distributed inference mapping",
                "metric": "Architectures with changed optimal mapping",
                "case": (
                    f"{example['workload']} / {example['architecture']} -> "
                    f"{example['optimal_distributed_mapping']}"
                ),
                "value": float(len(changed)),
                "unit": f"of {len(optimal_mapping_df)}",
                "interpretation": "Topology/link changes can make multicast preferable to node-to-node exchange.",
            })

    if optimal_compiler_df is not None and not optimal_compiler_df.empty:
        changed = optimal_compiler_df[optimal_compiler_df["strategy_changed_vs_baseline"].astype(bool)]
        if not changed.empty:
            best_changed = changed.loc[changed["energy_gain_vs_baseline_strategy_pct"].idxmax()]
            rows.append({
                "category": "Compiler/dataflow paradigm",
                "metric": "Best changed strategy",
                "case": (
                    f"{best_changed['workload']} / {best_changed['architecture']} -> "
                    f"{best_changed['strategy_label']}"
                ),
                "value": best_changed["energy_gain_vs_baseline_strategy_pct"],
                "unit": "%",
                "interpretation": "Models XLA/Pallas-visible cache, sharding, and multicast choices.",
            })

    if scaling_exponents_df is not None and not scaling_exponents_df.empty:
        multicast_energy = scaling_exponents_df[
            (scaling_exponents_df["scenario"] == "shared_state_multicast")
            & (scaling_exponents_df["metric"] == "energy_uJ_per_chip")
        ]
        if not multicast_energy.empty:
            best_scale = multicast_energy.loc[multicast_energy["scaling_exponent"].idxmin()]
            rows.append({
                "category": "Scaling law",
                "metric": "Best multicast energy exponent",
                "case": best_scale["architecture"],
                "value": best_scale["scaling_exponent"],
                "unit": "log-log slope",
                "interpretation": "Lower is better when scaling from 64 chips to superpod scale.",
            })

    if chiplet_df is not None and not chiplet_df.empty:
        photonic = chiplet_df[chiplet_df["architecture"] != "Advanced chiplet torus"]
        if not photonic.empty:
            best_chiplet = photonic.loc[photonic["energy_reduction_vs_chiplet_pct"].idxmax()]
            rows.append({
                "category": "Advanced chiplet comparison",
                "metric": "Best energy reduction vs chiplet",
                "case": f"{best_chiplet['architecture']} / {best_chiplet['scenario']}",
                "value": best_chiplet["energy_reduction_vs_chiplet_pct"],
                "unit": "%",
                "interpretation": "Compares against 3.5 Tbps, 6 pJ/bit advanced packaging.",
            })

    if superpod_summary_df is not None and not superpod_summary_df.empty:
        photonic = superpod_summary_df[
            superpod_summary_df["configuration"].eq("M1000 full photonic compiler stack")
        ]
        if not photonic.empty:
            best_speedup = photonic.loc[photonic["speedup_vs_baseline"].idxmax()]
            rows.append({
                "category": "Full superpod end-to-end",
                "metric": "Best latency speedup",
                "case": f"{best_speedup['model']} / {best_speedup['phase']}",
                "value": best_speedup["speedup_vs_baseline"],
                "unit": "x",
                "interpretation": "9216-chip Ironwood/Gemini-class proxy with full photonic compiler stack.",
            })
            best_energy = photonic.loc[photonic["energy_reduction_vs_baseline_pct"].idxmax()]
            rows.append({
                "category": "Full superpod end-to-end",
                "metric": "Best energy reduction",
                "case": f"{best_energy['model']} / {best_energy['phase']}",
                "value": best_energy["energy_reduction_vs_baseline_pct"],
                "unit": "%",
                "interpretation": "End-to-end compute + HBM + network energy, including compiler/dataflow effects.",
            })

    if superpod_ablation_df is not None and not superpod_ablation_df.empty:
        changed = superpod_ablation_df[
            superpod_ablation_df["technique"].eq("full photonic compiler stack")
        ]
        if not changed.empty:
            best = changed.loc[changed["speedup_increment_vs_previous"].idxmax()]
            rows.append({
                "category": "Superpod ablation",
                "metric": "Largest final-step increment",
                "case": f"{best['model']} / {best['phase']}",
                "value": best["speedup_increment_vs_previous"],
                "unit": "x vs previous",
                "interpretation": "Incremental effect after topology, multicast, remote HBM, and MoE-locality ablations.",
            })

    return pd.DataFrame(rows)


def optimal_distributed_mapping_frame(
    study_df: pd.DataFrame,
    objective: str = "energy",
    baseline_architecture: str | None = None,
) -> pd.DataFrame:
    """
    Choose the best distributed communication mapping for each architecture.

    In this project the `scenario` column doubles as a distributed inference
    mapping choice: activation exchange (`node_to_node`), remote memory fetch
    (`node_to_dram`), or shared-memory multicast (`shared_dram_multicast`).
    This helper asks which choice becomes optimal after the interconnect changes.
    """
    if study_df.empty:
        return pd.DataFrame()
    metric_col = "final_energy_uJ" if objective == "energy" else "final_latency_us"
    if metric_col not in study_df.columns:
        raise ValueError(f"Missing objective column: {metric_col}")

    df = study_df.copy()
    if "variant" in df.columns:
        df["architecture"] = df["variant"]
    else:
        df["architecture"] = df["topology"] + " / " + df["technology"]

    group_cols = ["workload", "architecture"]
    idx = df.groupby(group_cols)[metric_col].idxmin()
    best = df.loc[idx].copy().sort_values(group_cols).reset_index(drop=True)
    best = best.rename(columns={"scenario": "optimal_distributed_mapping"})

    if baseline_architecture is not None:
        baseline = best[best["architecture"] == baseline_architecture][
            ["workload", "optimal_distributed_mapping", metric_col]
        ].rename(
            columns={
                "optimal_distributed_mapping": "baseline_distributed_mapping",
                metric_col: f"baseline_{metric_col}",
            }
        )
        best = best.merge(baseline, on="workload", how="left")
        best["optimal_mapping_changed_vs_baseline"] = (
            best["optimal_distributed_mapping"] != best["baseline_distributed_mapping"]
        )
        best[f"{objective}_gain_vs_baseline_optimal_pct"] = 100 * (
            1 - best[metric_col] / best[f"baseline_{metric_col}"]
        )

    keep_cols = [
        "workload",
        "architecture",
        "optimal_distributed_mapping",
        metric_col,
    ]
    for optional in (
        "baseline_distributed_mapping",
        f"baseline_{metric_col}",
        "optimal_mapping_changed_vs_baseline",
        f"{objective}_gain_vs_baseline_optimal_pct",
        "final_mapping_signature",
    ):
        if optional in best.columns:
            keep_cols.append(optional)
    return best[keep_cols].reset_index(drop=True)


def write_presentation_assets(
    output_dir: str | Path,
    cluster_improvement_df: pd.DataFrame,
    tpu_improvement_df: pd.DataFrame,
    tpu_mapping_df: pd.DataFrame,
    optimal_mapping_df: pd.DataFrame | None = None,
    optimal_compiler_df: pd.DataFrame | None = None,
    scaling_df: pd.DataFrame | None = None,
    scaling_exponents_df: pd.DataFrame | None = None,
    chiplet_df: pd.DataFrame | None = None,
    superpod_stage_df: pd.DataFrame | None = None,
    superpod_summary_df: pd.DataFrame | None = None,
    superpod_ablation_df: pd.DataFrame | None = None,
    superpod_context_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Write concise, presentation-ready deliverables.

    The raw CSVs remain available for auditability; this folder is intentionally
    smaller and easier to drop into slides or a final report.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    presentation_dir = output_dir / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)

    summary_df = executive_summary_frame(
        cluster_improvement_df,
        tpu_improvement_df,
        tpu_mapping_df,
        optimal_mapping_df,
        optimal_compiler_df,
        scaling_exponents_df,
        chiplet_df,
        superpod_summary_df,
        superpod_ablation_df,
    )
    paths: dict[str, Path] = {
        "presentation_index": presentation_dir / "index.md",
        "executive_summary": presentation_dir / "executive_summary.csv",
        "cluster_reductions_plot": presentation_dir / "01_cluster_reductions.png",
        "tpu_mapping_plot": presentation_dir / "02_tpu_mapping_reductions.png",
        "topology_diagram": presentation_dir / "03_topology_diagram.png",
        "compiler_strategy_plot": presentation_dir / "04_compiler_strategy_choices.png",
        "scaling_law_plot": presentation_dir / "05_scaling_laws.png",
        "chiplet_plot": presentation_dir / "06_chiplet_baseline.png",
        "superpod_speedup_plot": presentation_dir / "07_superpod_end_to_end_speedups.png",
        "superpod_stage_plot": presentation_dir / "08_superpod_stage_breakdown.png",
        "superpod_context_plot": presentation_dir / "09_superpod_context_scaling.png",
    }
    summary_df.to_csv(paths["executive_summary"], index=False)
    if optimal_mapping_df is not None and not optimal_mapping_df.empty:
        paths["presentation_optimal_mapping_choices"] = (
            presentation_dir / "optimal_mapping_choices.csv"
        )
        optimal_mapping_df.to_csv(paths["presentation_optimal_mapping_choices"], index=False)

    lines = [
        "# Presentation Package",
        "",
        "This folder contains the condensed final-project deliverables for slides or a short report.",
        "",
        "## Files",
        "",
        "- `executive_summary.csv`: headline metrics and interpretations.",
        "- `01_cluster_reductions.png`: 16-node optical-crossbar gains vs electrical mesh.",
        "- `02_tpu_mapping_reductions.png`: TPU-cube mapping-aware gains after fixed-point re-optimization.",
        "- `03_topology_diagram.png`: simplified topology comparison diagram.",
        "- `04_compiler_strategy_choices.png`: XLA/Pallas-style dataflow choices enabled by photonic multicast.",
        "- `05_scaling_laws.png`: 64-chip to 9216-chip scaling laws.",
        "- `06_chiplet_baseline.png`: photonic comparison against the 3.5 Tbps / 6 pJ/bit chiplet baseline.",
        "- `07_superpod_end_to_end_speedups.png`: full-superpod Gemini-class training/inference speedups.",
        "- `08_superpod_stage_breakdown.png`: where latency remains after full-stack photonic optimization.",
        "- `09_superpod_context_scaling.png`: long-context inference scaling for electrical/chiplet/photonic fabrics.",
        "- `optimal_mapping_choices.csv`: condensed view of which communication mapping each interconnect makes optimal.",
        "",
        "## Headline Metrics",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- **{row.category} / {row.metric}**: {row.value:.2f} {row.unit} "
            f"for `{row.case}`. {row.interpretation}"
        )
    if optimal_mapping_df is not None and not optimal_mapping_df.empty:
        lines.extend(["", "## Optimal Distributed Mapping Choices", ""])
        for row in optimal_mapping_df.itertuples(index=False):
            changed = getattr(row, "optimal_mapping_changed_vs_baseline", False)
            changed_text = "changed" if changed else "same as baseline"
            lines.append(
                f"- `{row.workload}` / `{row.architecture}` chooses "
                f"`{row.optimal_distributed_mapping}` ({changed_text})."
            )
    paths["presentation_index"].write_text("\n".join(lines) + "\n")

    if not cluster_improvement_df.empty:
        plot_df = cluster_improvement_df.sort_values(
            "energy_reduction_pct", ascending=False
        ).head(8)
        labels = plot_df["workload"] + "\n" + plot_df["scenario"]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(labels, plot_df["energy_reduction_pct"], color="#2B8CBE")
        ax.set_title("Optical Crossbar vs Electrical Mesh")
        ax.set_ylabel("Energy reduction (%)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(paths["cluster_reductions_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not tpu_mapping_df.empty:
        photonic = tpu_mapping_df[tpu_mapping_df["variant"] != "TPU cube electrical torus"].copy()
        if not photonic.empty:
            photonic["label"] = (
                photonic["variant"].str.replace("TPU cube ", "", regex=False)
                + "\n"
                + photonic["scenario"]
            )
            plot_df = photonic.sort_values(
                "energy_reduction_vs_electrical_pct", ascending=False
            ).head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [
                "#31A354" if changed else "#9ECAE1"
                for changed in plot_df["mapping_differs_from_electrical"]
            ]
            ax.bar(plot_df["label"], plot_df["energy_reduction_vs_electrical_pct"], color=colors)
            ax.set_title("TPU Cube: Mapping-Aware Photonic Gains")
            ax.set_ylabel("Energy reduction vs electrical torus (%)")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.axhline(0, color="black", linewidth=0.8)
            fig.tight_layout()
            fig.savefig(paths["tpu_mapping_plot"], dpi=180, bbox_inches="tight")
            plt.close(fig)

    if optimal_compiler_df is not None and not optimal_compiler_df.empty:
        changed = optimal_compiler_df.copy()
        changed["label"] = changed["workload"] + "\n" + changed["architecture"]
        plot_df = changed.sort_values("energy_gain_vs_baseline_strategy_pct", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = [
            "#31A354" if value else "#9ECAE1"
            for value in plot_df["strategy_changed_vs_baseline"]
        ]
        ax.bar(plot_df["label"], plot_df["energy_gain_vs_baseline_strategy_pct"], color=colors)
        ax.set_title("Compiler/Dataflow Strategy Gains vs TPU Electrical Baseline")
        ax.set_ylabel("Energy gain (%)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(paths["compiler_strategy_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    if scaling_df is not None and not scaling_df.empty:
        plot_df = scaling_df[scaling_df["scenario"] == "shared_state_multicast"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for architecture, group in plot_df.groupby("architecture"):
            group = group.sort_values("active_nodes")
            ax.plot(
                group["active_nodes"],
                group["latency_us_per_chip"],
                marker="o",
                label=architecture,
            )
        ax.set_title("Scaling Law: Shared-State Broadcast Latency")
        ax.set_xlabel("Active chips")
        ax.set_ylabel("Latency per chip (us)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(paths["scaling_law_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    if chiplet_df is not None and not chiplet_df.empty:
        plot_df = chiplet_df[chiplet_df["architecture"] != "Advanced chiplet torus"].copy()
        plot_df["label"] = plot_df["architecture"] + "\n" + plot_df["scenario"]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(plot_df["label"], plot_df["energy_reduction_vs_chiplet_pct"], color="#756BB1")
        ax.set_title("Photonic Fabrics vs Advanced Chiplet Baseline")
        ax.set_ylabel("Energy reduction vs chiplet (%)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(paths["chiplet_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    if superpod_summary_df is not None and not superpod_summary_df.empty:
        keep = superpod_summary_df[
            superpod_summary_df["configuration"].isin([
                "Advanced chiplet torus + XLA SPMD",
                "M1000 switched optical topology only",
                "M1000 full photonic compiler stack",
            ])
        ].copy()
        keep["label"] = (
            keep["phase"].str.title()
            + "\n"
            + keep["configuration"].str.replace(" + ", "\n+ ", regex=False)
        )
        plot_df = keep.sort_values("speedup_vs_baseline", ascending=False).head(16)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.bar(plot_df["label"], plot_df["speedup_vs_baseline"], color="#2B8CBE")
        ax.set_title("9216-Chip Superpod: End-to-End Speedup vs Electrical Ironwood")
        ax.set_ylabel("Speedup (x)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(1.0, color="black", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(paths["superpod_speedup_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    if superpod_stage_df is not None and not superpod_stage_df.empty:
        model_name = "Gemini 3.1 Pro-class sparse MoE proxy"
        plot_df = superpod_stage_df[
            superpod_stage_df["model"].eq(model_name)
            & superpod_stage_df["configuration"].eq("M1000 full photonic compiler stack")
        ].copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            plot_df["label"] = plot_df["phase"].str.title() + "\n" + plot_df["stage"]
            ax.bar(plot_df["label"], plot_df["total_latency_s"], color="#31A354")
            ax.set_title("Full Photonic Stack: Remaining Stage Latency")
            ax.set_ylabel("Stage latency (s)")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            fig.tight_layout()
            fig.savefig(paths["superpod_stage_plot"], dpi=180, bbox_inches="tight")
            plt.close(fig)

    if superpod_context_df is not None and not superpod_context_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for config, group in superpod_context_df.groupby("configuration"):
            group = group.sort_values("context_tokens")
            ax.plot(
                group["context_tokens"],
                group["total_latency_s"],
                marker="o",
                label=config,
            )
        ax.set_title("Gemini-Class Inference Context Scaling")
        ax.set_xlabel("Prompt context tokens")
        ax.set_ylabel("End-to-end inference latency (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(paths["superpod_context_plot"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    boxes = [
        (0.08, 0.62, "Electrical 4x4 mesh\nmulti-hop, no multicast\n10 pJ/bit assumed"),
        (0.40, 0.62, "EVK100 photonic torus\nsame cube topology\n1.9 pJ/bit reference"),
        (0.72, 0.62, "L200/L200X/M1000\nswitched optical fabric\nsingle-hop + multicast"),
        (0.40, 0.18, "AccelForge fixed point\nmapping -> traffic -> network cost -> remap"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7FBFF", edgecolor="#2B8CBE"),
            fontsize=10,
        )
    arrowprops = dict(arrowstyle="->", color="#636363", linewidth=1.5)
    ax.annotate("", xy=(0.31, 0.62), xytext=(0.18, 0.62), arrowprops=arrowprops)
    ax.annotate("", xy=(0.63, 0.62), xytext=(0.49, 0.62), arrowprops=arrowprops)
    ax.annotate("", xy=(0.40, 0.32), xytext=(0.40, 0.52), arrowprops=arrowprops)
    ax.set_title("Topology Path: Baseline to Photonic Mapping Study", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(paths["topology_diagram"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_study_outputs(
    output_dir: str | Path,
    workload_names: Iterable[str] = DEFAULT_WORKLOAD_NAMES,
    scenario_names: Iterable[str] = DEFAULT_SCENARIO_NAMES,
    topology_to_tech: dict[Topology, InterconnectTech] | None = None,
    tpu_mapping_workload_names: Iterable[str] = ("kv_attention", "prefill_projection"),
    tpu_mapping_scenario_names: Iterable[str] = DEFAULT_SCENARIO_NAMES,
    tpu_mapping_max_iterations: int = 4,
    include_presentation_assets: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_to_tech = topology_to_tech or default_topology_suite()
    cluster_df = compare_topologies(workload_names, scenario_names, topology_to_tech)
    cluster_improvements = cluster_improvement_frame(cluster_df)
    tpu_df = tpu_cube_case_study()
    tpu_improvements = tpu_improvement_frame(tpu_df)
    tpu_mapping_df = tpu_cube_mapping_study(
        workload_names=tpu_mapping_workload_names,
        scenario_names=tpu_mapping_scenario_names,
        max_iterations=tpu_mapping_max_iterations,
    )
    compiler_strategy_df = compiler_strategy_study(
        workload_names=workload_names,
        active_nodes=64,
    )
    optimal_compiler_df = optimal_compiler_strategy_frame(compiler_strategy_df)
    scaling_df = scaling_law_study()
    scaling_exponents_df = scaling_exponent_frame(scaling_df)
    chiplet_df = chiplet_comparison_frame(active_nodes=64)
    superpod_stage_df = superpod_stage_study()
    superpod_summary_df = superpod_end_to_end_summary(superpod_stage_df)
    superpod_ablation_df = superpod_ablation_frame(superpod_summary_df)
    superpod_context_df = superpod_context_scaling_study()
    cluster_optimal = optimal_distributed_mapping_frame(
        cluster_df,
        baseline_architecture=f"{Mesh2DTopology(4, 4).name} / {ELECTRICAL_800G.name}",
    )
    if not cluster_optimal.empty:
        cluster_optimal.insert(0, "study", "16-node cluster")
    tpu_optimal = optimal_distributed_mapping_frame(
        tpu_mapping_df,
        baseline_architecture="TPU cube electrical torus",
    )
    if not tpu_optimal.empty:
        tpu_optimal.insert(0, "study", "TPU cube mapping")
    optimal_mapping_df = pd.concat(
        [cluster_optimal, tpu_optimal], ignore_index=True
    )
    sources_df = source_frame()
    summary_md = generate_summary_markdown(
        cluster_df,
        cluster_improvements,
        tpu_df,
        tpu_improvements,
        tpu_mapping_df,
        optimal_mapping_df,
        compiler_strategy_df,
        optimal_compiler_df,
        scaling_exponents_df,
        chiplet_df,
        superpod_summary_df,
        superpod_ablation_df,
    )

    paths: dict[str, Path] = {
        "cluster_study": output_dir / "cluster_study.csv",
        "cluster_improvements": output_dir / "cluster_improvements.csv",
        "tpu_cube_case_study": output_dir / "tpu_cube_case_study.csv",
        "tpu_improvements": output_dir / "tpu_improvements.csv",
        "tpu_mapping_study": output_dir / "tpu_mapping_study.csv",
        "optimal_mapping_choices": output_dir / "optimal_mapping_choices.csv",
        "compiler_strategy_study": output_dir / "compiler_strategy_study.csv",
        "optimal_compiler_strategies": output_dir / "optimal_compiler_strategies.csv",
        "scaling_law_study": output_dir / "scaling_law_study.csv",
        "scaling_exponents": output_dir / "scaling_exponents.csv",
        "chiplet_comparison": output_dir / "chiplet_comparison.csv",
        "superpod_stage_study": output_dir / "superpod_stage_study.csv",
        "superpod_end_to_end_summary": output_dir / "superpod_end_to_end_summary.csv",
        "superpod_ablation_study": output_dir / "superpod_ablation_study.csv",
        "superpod_context_scaling": output_dir / "superpod_context_scaling.csv",
        "sources": output_dir / "sources.csv",
        "summary": output_dir / "summary.md",
    }
    cluster_df.to_csv(paths["cluster_study"], index=False)
    cluster_improvements.to_csv(paths["cluster_improvements"], index=False)
    tpu_df.to_csv(paths["tpu_cube_case_study"], index=False)
    tpu_improvements.to_csv(paths["tpu_improvements"], index=False)
    tpu_mapping_df.to_csv(paths["tpu_mapping_study"], index=False)
    optimal_mapping_df.to_csv(paths["optimal_mapping_choices"], index=False)
    compiler_strategy_df.to_csv(paths["compiler_strategy_study"], index=False)
    optimal_compiler_df.to_csv(paths["optimal_compiler_strategies"], index=False)
    scaling_df.to_csv(paths["scaling_law_study"], index=False)
    scaling_exponents_df.to_csv(paths["scaling_exponents"], index=False)
    chiplet_df.to_csv(paths["chiplet_comparison"], index=False)
    superpod_stage_df.to_csv(paths["superpod_stage_study"], index=False)
    superpod_summary_df.to_csv(paths["superpod_end_to_end_summary"], index=False)
    superpod_ablation_df.to_csv(paths["superpod_ablation_study"], index=False)
    superpod_context_df.to_csv(paths["superpod_context_scaling"], index=False)
    sources_df.to_csv(paths["sources"], index=False)
    paths["summary"].write_text(summary_md)
    if include_presentation_assets:
        paths.update(
            write_presentation_assets(
                output_dir,
                cluster_improvements,
                tpu_improvements,
                tpu_mapping_df,
                optimal_mapping_df,
                optimal_compiler_df,
                scaling_df,
                scaling_exponents_df,
                chiplet_df,
                superpod_stage_df,
                superpod_summary_df,
                superpod_ablation_df,
                superpod_context_df,
            )
        )
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_arg(value: str | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return fallback
    return tuple(x.strip() for x in value.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Lightmatter photonic interconnect study."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "cluster", "tpu", "tpu-mapping", "sources", "tech"),
        help="Artifact set to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LAB_DIR / "results"),
        help="Directory for generated CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOAD_NAMES),
        help="Comma-separated workload names for the cluster study.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIO_NAMES),
        help="Comma-separated scenario names for the cluster study.",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.command == "cluster":
        paths = write_study_outputs(
            output_dir,
            workload_names=_csv_arg(args.workloads, DEFAULT_WORKLOAD_NAMES),
            scenario_names=_csv_arg(args.scenarios, DEFAULT_SCENARIO_NAMES),
        )
        for key in ("cluster_study", "cluster_improvements", "summary"):
            print(f"{key}: {paths[key]}")
        return 0

    if args.command == "tpu":
        output_dir.mkdir(parents=True, exist_ok=True)
        tpu_df = tpu_cube_case_study()
        tpu_improvements = tpu_improvement_frame(tpu_df)
        tpu_mapping_df = tpu_cube_mapping_study()
        tpu_path = output_dir / "tpu_cube_case_study.csv"
        imp_path = output_dir / "tpu_improvements.csv"
        mapping_path = output_dir / "tpu_mapping_study.csv"
        tpu_df.to_csv(tpu_path, index=False)
        tpu_improvements.to_csv(imp_path, index=False)
        tpu_mapping_df.to_csv(mapping_path, index=False)
        print(f"tpu_cube_case_study: {tpu_path}")
        print(f"tpu_improvements: {imp_path}")
        print(f"tpu_mapping_study: {mapping_path}")
        return 0

    if args.command == "tpu-mapping":
        output_dir.mkdir(parents=True, exist_ok=True)
        tpu_mapping_df = tpu_cube_mapping_study(
            workload_names=_csv_arg(args.workloads, ("kv_attention",)),
            scenario_names=_csv_arg(args.scenarios, DEFAULT_SCENARIO_NAMES),
        )
        mapping_path = output_dir / "tpu_mapping_study.csv"
        tpu_mapping_df.to_csv(mapping_path, index=False)
        print(f"tpu_mapping_study: {mapping_path}")
        return 0

    if args.command == "sources":
        output_dir.mkdir(parents=True, exist_ok=True)
        sources_path = output_dir / "sources.csv"
        summary_path = output_dir / "summary.md"
        source_frame().to_csv(sources_path, index=False)
        summary_path.write_text(
            "\n".join(
                ["# Sources", ""]
                + [
                    f"- {row.source} ({row.date}): {row.url}"
                    for row in source_frame().itertuples(index=False)
                ]
            )
            + "\n"
        )
        print(f"sources: {sources_path}")
        print(f"summary: {summary_path}")
        return 0

    if args.command == "tech":
        output_dir.mkdir(parents=True, exist_ok=True)
        tech_df = all_technologies_network_comparison()
        tech_path = output_dir / "tech_comparison.csv"
        tech_df.to_csv(tech_path, index=False)
        print(f"tech_comparison: {tech_path}")
        return 0

    paths = write_study_outputs(
        output_dir,
        workload_names=_csv_arg(args.workloads, DEFAULT_WORKLOAD_NAMES),
        scenario_names=_csv_arg(args.scenarios, DEFAULT_SCENARIO_NAMES),
    )
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
