"""
Modeling Lightmatter photonic interconnects with AccelForge.

Covers all three project milestones:
  M1 – Topology-aware network model driven by AccelForge logical-memory mappings.
  M2 – Iterative mapper loop that augments logical-memory costs until they stabilize.
  M3 – Optical crossbar vs. electrical mesh comparison with full source attribution.

Plus: TPU cube ICI replacement study using EVK100, L200, L200X, and M1000 class links.

Public sources were checked / dated April 1, 2026.
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

# Ironwood (2025) – HotChips 2025 tutorial slide data
IRONWOOD_TFLOPS_PER_CHIP = 4614.0
IRONWOOD_HBM_GB = 192
IRONWOOD_HBM_BANDWIDTH_TBPS = 7.4
# ICI bandwidth not directly disclosed; inferred from HBM BW growth (2.64x v5p).
# TPU team slide: "1.5x–10x scaling across all critical vectors (FLOPs, HBM BW, ICI BW)".
# Conservative mid-point: 2x v5p ICI → 2400 GB/s bidirectional per chip.
IRONWOOD_ICI_BIDIR_GBPS_ESTIMATED = 2400.0  # estimated, not public


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
        "date": "2025",
        "url": "https://lightmatter.co/products/passage/",
        "summary": (
            "Passage EVK100: 3.2 Tbps per fiber at 112G PAM4 (16 lambda). "
            "1.9 pJ/bit (per ISES Korea 2025 efficiency table). "
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
        "date": "2025-05",
        "url": "https://lightmatter.co/press-release/lightmatter-unveils-passage-m1000-photonic-superchip-worlds-fastest-ai-interconnect/",
        "summary": (
            "Passage M1000: 114 Tbps total, 256 edge-attached fibers, 1024 SerDes lanes. "
            "7735 mm² package, 34 chiplets, built-in solid-state OCS. "
            "Sampling summer 2025."
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
    "tpu_v5p": {
        "date": "viewed 2026-04-01",
        "url": "https://cloud.google.com/tpu/docs/v5p",
        "summary": (
            "TPU v5p: 459 TFLOPS/chip, 95 GB HBM, 2.8 TBps HBM BW, "
            "1200 GB/s bidirectional ICI per chip, 3D torus, 4x4x4 full cube."
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
        "Passage EVK100: 3.2 Tbps/fiber, 1.9 pJ/bit at 112G PAM4 (ISES Korea 2025 paper). "
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
    link_bandwidth_bps=57e12,       # 114 Tbps aggregate / 2 directions = 57 Tbps one-way
    energy_per_bit_j=4.0e-12,       # <5 pJ/bit; better efficiency at scale
    router_latency_s=2.5e-9,        # slightly higher: includes built-in OCS switching
    supports_multicast=True,
    notes=(
        "Passage M1000: 114 Tbps total (256 fibers, 1024 SerDes), 7735 mm² package, 34 chiplets, "
        "built-in solid-state OCS. Sampling summer 2025. Modeled as 57 Tbps one-way."
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
    pe_x: int = 64,
    pe_y: int = 64,
) -> af.Spec:
    jinja = {
        "COMM_PATTERN": scenario.comm_pattern,
        "LOGICAL_READ_ENERGY": logical_read_energy_j,
        "LOGICAL_WRITE_ENERGY": logical_write_energy_j,
        "LOGICAL_READ_LATENCY": logical_read_latency_s,
        "LOGICAL_WRITE_LATENCY": logical_write_latency_s,
        "NODES_X": node_x,
        "NODES_Y": node_y,
        "PE_X": pe_x,
        "PE_Y": pe_y,
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


def run_one_iteration(
    workload: WorkloadCase,
    scenario: Scenario,
    logical_read_energy_j: float,
    logical_write_energy_j: float,
    logical_read_latency_s: float,
    logical_write_latency_s: float,
) -> tuple[af.Mapping, TrafficBreakdown]:
    spec = _build_spec(
        workload,
        scenario,
        logical_read_energy_j=logical_read_energy_j,
        logical_write_energy_j=logical_write_energy_j,
        logical_read_latency_s=logical_read_latency_s,
        logical_write_latency_s=logical_write_latency_s,
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
        )
        network = topology.evaluate(traffic, scenario, tech)
        total_energy = float(mapping.energy())
        total_latency = float(mapping.latency())

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
                    workload, scenario, topology, tech, max_iterations=max_iterations
                )
                baseline_energy = history[0].total_energy_j
                baseline_latency = history[0].total_latency_s
                final_energy = history[-1].total_energy_j
                final_latency = history[-1].total_latency_s
                first_plus_net_energy = baseline_energy + history[0].network.total_energy_j
                first_plus_net_latency = baseline_latency + history[0].network.total_latency_s
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

    techs = {"Electrical 800G": ELECTRICAL_800G, **ALL_LIGHTMATTER_TECHS}
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
    m1000_one_way_bps = 57e12                   # M1000: 114 Tbps total / 2 dirs

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
                energy_per_bit_j=4.0e-12,
                router_latency_s=2.5e-9,
                supports_multicast=True,
                notes="M1000 OCS-based crossbar; 57 Tbps one-way from 114 Tbps aggregate.",
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


# ---------------------------------------------------------------------------
# Summary report generator
# ---------------------------------------------------------------------------

def generate_summary_markdown(
    cluster_df: pd.DataFrame,
    cluster_improvement_df: pd.DataFrame,
    tpu_df: pd.DataFrame,
    tpu_improvement_df: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Lightmatter Final Project — Summary",
        "",
        "Generated from the AccelForge + photonic-interconnect study pipeline.",
        "",
        "## Technology Catalog",
        "",
        "| Technology | BW (one-way) | Energy (pJ/bit) | Multicast |",
        "|---|---|---|---|",
        f"| Electrical 800G | {ELECTRICAL_800G.link_bandwidth_bps/1e9:.0f} Gbps | "
        f"{ELECTRICAL_800G.energy_per_bit_j*1e12:.1f} | No |",
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

    lines.extend(["## Sources", ""])
    for _, row in source_frame().iterrows():
        lines.append(f"- **{row['source']}** ({row['date']}): {row['summary']}  \n  {row['url']}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_study_outputs(
    output_dir: str | Path,
    workload_names: Iterable[str] = DEFAULT_WORKLOAD_NAMES,
    scenario_names: Iterable[str] = DEFAULT_SCENARIO_NAMES,
    topology_to_tech: dict[Topology, InterconnectTech] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_to_tech = topology_to_tech or default_topology_suite()
    cluster_df = compare_topologies(workload_names, scenario_names, topology_to_tech)
    cluster_improvements = cluster_improvement_frame(cluster_df)
    tpu_df = tpu_cube_case_study()
    tpu_improvements = tpu_improvement_frame(tpu_df)
    sources_df = source_frame()
    summary_md = generate_summary_markdown(
        cluster_df, cluster_improvements, tpu_df, tpu_improvements
    )

    paths: dict[str, Path] = {
        "cluster_study": output_dir / "cluster_study.csv",
        "cluster_improvements": output_dir / "cluster_improvements.csv",
        "tpu_cube_case_study": output_dir / "tpu_cube_case_study.csv",
        "tpu_improvements": output_dir / "tpu_improvements.csv",
        "sources": output_dir / "sources.csv",
        "summary": output_dir / "summary.md",
    }
    cluster_df.to_csv(paths["cluster_study"], index=False)
    cluster_improvements.to_csv(paths["cluster_improvements"], index=False)
    tpu_df.to_csv(paths["tpu_cube_case_study"], index=False)
    tpu_improvements.to_csv(paths["tpu_improvements"], index=False)
    sources_df.to_csv(paths["sources"], index=False)
    paths["summary"].write_text(summary_md)
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
        choices=("all", "cluster", "tpu", "sources", "tech"),
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
        tpu_path = output_dir / "tpu_cube_case_study.csv"
        imp_path = output_dir / "tpu_improvements.csv"
        tpu_df.to_csv(tpu_path, index=False)
        tpu_improvements.to_csv(imp_path, index=False)
        print(f"tpu_cube_case_study: {tpu_path}")
        print(f"tpu_improvements: {imp_path}")
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
