"""
Compare Passage EVK100 (photonic) vs. Electrical Mesh using AccelForge's
native !Network component.

Two architecture models:
  - Photonic:   Network(hops=0) + PassageTransceiver(1.9 pJ/bit flat)
  - Electrical: Network(hops=10 pJ/bit, distance-dependent)

Supports 2D (--nodes-z 1) and 3D (--nodes-z 4) topologies.

Usage:
    python run_passage_network.py
    python run_passage_network.py --M 2048 --K 4096 --N 4096
    python run_passage_network.py --nodes-z 4          # 3D cube
    python run_passage_network.py --photonic-only       # skip electrical
    python run_passage_network.py --electrical-only     # skip photonic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import accelforge as af

THIS_DIR = Path(__file__).resolve().parent
LAB_DIR = THIS_DIR.parent
ARCH_DIR = LAB_DIR / "arch"
WORKLOAD_DIR = LAB_DIR.parent / "workload"

# Architecture files
PHOTONIC_ARCH = ARCH_DIR / "passage_evk100_flat.yaml"
ELECTRICAL_ARCH = ARCH_DIR / "electrical_mesh_baseline.yaml"
WORKLOAD_PATH = WORKLOAD_DIR / "single_matmul.yaml"


def build_spec(
    arch_path: Path,
    M: int = 1,
    K: int = 4096,
    N: int = 16384,
    bits_per_value: int = 8,
    nodes_x: int = 4,
    nodes_y: int = 4,
    nodes_z: int = 1,
    pe_x: int = 64,
    pe_y: int = 64,
) -> af.Spec:
    """Build an AccelForge Spec from an arch YAML + matmul workload."""
    jinja = {
        "NODES_X": nodes_x,
        "NODES_Y": nodes_y,
        "NODES_Z": nodes_z,
        "PE_X": pe_x,
        "PE_Y": pe_y,
        "M": M,
        "K": K,
        "N": N,
        "BITS_PER_VALUE": bits_per_value,
    }
    spec = af.Spec.from_yaml(arch_path, WORKLOAD_PATH, jinja_parse_data=jinja)
    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    return spec


def run_and_print(arch_path: Path, label: str, verbose: bool = True, **kwargs) -> None:
    """Build spec, run mapper, print results."""
    spec = build_spec(arch_path, **kwargs)

    mappings = spec.map_workload_to_arch(
        print_progress=verbose,
        print_number_of_pmappings=verbose,
    )
    if len(mappings) == 0:
        print(f"  [{label}] No valid mappings found!")
        return None, None

    best = mappings[0]
    energy_j = float(best.energy())
    latency_s = float(best.latency())

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Total energy:  {energy_j * 1e6:.4f} uJ")
    print(f"  Total latency: {latency_s * 1e6:.4f} us")

    per_comp = best.energy(per_component=True)
    print(f"\n  Per-component energy breakdown:")
    for comp, e in sorted(per_comp.items(), key=lambda x: -float(x[1])):
        e_val = float(e)
        if e_val > 0:
            pct = 100 * e_val / energy_j if energy_j > 0 else 0
            print(f"    {comp:30s}  {e_val*1e6:10.4f} uJ  ({pct:5.1f}%)")
    print()
    return energy_j, latency_s


def main():
    parser = argparse.ArgumentParser(
        description="Compare Passage EVK100 vs. Electrical Mesh in AccelForge"
    )
    parser.add_argument("--M", type=int, default=1, help="M dimension (default: 1)")
    parser.add_argument("--K", type=int, default=4096, help="K dimension (default: 4096)")
    parser.add_argument("--N", type=int, default=16384, help="N dimension (default: 16384)")
    parser.add_argument("--bits", type=int, default=8, help="Bits per value (default: 8)")
    parser.add_argument("--nodes-x", type=int, default=4, help="Nodes in X (default: 4)")
    parser.add_argument("--nodes-y", type=int, default=4, help="Nodes in Y (default: 4)")
    parser.add_argument("--nodes-z", type=int, default=1, help="Nodes in Z (default: 1, set to 4 for 3D)")
    parser.add_argument("--pe-x", type=int, default=64, help="PE array X (default: 64)")
    parser.add_argument("--pe-y", type=int, default=64, help="PE array Y (default: 64)")
    parser.add_argument("--quiet", action="store_true", help="Suppress mapper progress")
    parser.add_argument("--photonic-only", action="store_true", help="Only run photonic")
    parser.add_argument("--electrical-only", action="store_true", help="Only run electrical")
    args = parser.parse_args()

    common = dict(
        M=args.M, K=args.K, N=args.N,
        bits_per_value=args.bits,
        nodes_x=args.nodes_x, nodes_y=args.nodes_y, nodes_z=args.nodes_z,
        pe_x=args.pe_x, pe_y=args.pe_y,
        verbose=not args.quiet,
    )
    total_nodes = args.nodes_x * args.nodes_y * args.nodes_z
    topo = f"{args.nodes_x}x{args.nodes_y}" + (f"x{args.nodes_z}" if args.nodes_z > 1 else "")

    print(f"Topology: {topo} ({total_nodes} nodes)")
    print(f"Workload: Matmul M={args.M}, K={args.K}, N={args.N}, {args.bits}-bit")
    print(f"PE array: {args.pe_x}x{args.pe_y} per node\n")

    e_ph, l_ph = None, None
    e_el, l_el = None, None

    if not args.electrical_only:
        e_ph, l_ph = run_and_print(
            PHOTONIC_ARCH,
            f"Photonic EVK100 (flat cost) -- {topo}, {total_nodes} nodes",
            **common,
        )

    if not args.photonic_only:
        e_el, l_el = run_and_print(
            ELECTRICAL_ARCH,
            f"Electrical Mesh (per-hop cost) -- {topo}, {total_nodes} nodes",
            **common,
        )

    # Summary comparison
    if e_ph and e_el:
        print("=" * 65)
        print("  COMPARISON SUMMARY")
        print("=" * 65)
        print(f"  Photonic:   {e_ph*1e6:12.4f} uJ  /  {l_ph*1e6:12.4f} us")
        print(f"  Electrical: {e_el*1e6:12.4f} uJ  /  {l_el*1e6:12.4f} us")
        print(f"  Energy reduction:  {100*(1 - e_ph/e_el):.1f}%")
        print(f"  Latency reduction: {100*(1 - l_ph/l_el):.1f}%")
        print()


if __name__ == "__main__":
    main()
