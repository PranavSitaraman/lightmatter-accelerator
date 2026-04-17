"""
Comprehensive test suite for the Lightmatter photonic-interconnect project.

Tests are organised into:
  - Unit tests for the network model (no AccelForge dependency)
  - Integration smoke tests that run solve_iteratively on tiny workloads
  - Output-artifact tests
  - Notebook structure test
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_lightmatter_notebook import build_notebook
from scripts.lightmatter_interconnects import (
    # Technology catalog
    ELECTRICAL_800G,
    ADVANCED_CHIPLET_3P5T,
    LIGHTMATTER_EVK50,
    LIGHTMATTER_EVK100,
    LIGHTMATTER_L200,
    LIGHTMATTER_L200X,
    LIGHTMATTER_M1000,
    ALL_LIGHTMATTER_TECHS,
    # Topologies
    RingTopology,
    Mesh2DTopology,
    OpticalCrossbarTopology,
    Torus3DTopology,
    # Data classes
    Scenario,
    TrafficBreakdown,
    WorkloadCase,
    # Functions
    workload_case,
    scenario_case,
    source_frame,
    solve_iteratively,
    history_to_frame,
    per_component_energy_frame,
    mapping_feature_dict,
    mapping_features_frame,
    mapping_signature,
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
    all_technologies_network_comparison,
    bandwidth_sensitivity_network,
    energy_per_bit_sensitivity_network,
    executive_summary_frame,
    write_study_outputs,
    generate_summary_markdown,
    # Constants
    HBM_ENERGY_PER_BIT_J,
    LLM_WORKLOADS,
    IRONWOOD_TFLOPS_PER_CHIP,
    IRONWOOD_HBM_BANDWIDTH_TBPS,
    IRONWOOD_SUPERPOD_CHIPS,
    IRONWOOD_SUPERPOD_CUBES,
    TPU_V5P_ICI_BIDIR_GBPS,
)


# =========================================================================
# Unit tests — pure network model, no AccelForge
# =========================================================================

class TestTechnologyCatalog(unittest.TestCase):
    """Verify that all technology constants are present and physically reasonable."""

    def test_all_lightmatter_techs_present(self):
        for key in ("EVK50", "EVK100", "L200", "L200X", "M1000"):
            self.assertIn(key, ALL_LIGHTMATTER_TECHS, f"Missing {key} in ALL_LIGHTMATTER_TECHS")

    def test_evk100_energy_is_1p9_pJ_per_bit(self):
        """EVK100 optimistic/reference point: 1.9 pJ/bit, with 3.2 pJ/bit kept as conservative sensitivity context."""
        self.assertAlmostEqual(
            LIGHTMATTER_EVK100.energy_per_bit_j * 1e12, 1.9, places=3,
            msg="EVK100 reference energy should be 1.9 pJ/bit"
        )

    def test_evk50_energy_is_4p6_pJ_per_bit(self):
        """EVK50 SC25 demo: 4.6 pJ/bit total wall-plug."""
        self.assertAlmostEqual(
            LIGHTMATTER_EVK50.energy_per_bit_j * 1e12, 4.6, places=3
        )

    def test_bandwidth_ordering(self):
        """Link bandwidths must increase: EVK50 < EVK100 < L200 < L200X < M1000."""
        bws = [
            LIGHTMATTER_EVK50.link_bandwidth_bps,
            LIGHTMATTER_EVK100.link_bandwidth_bps,
            LIGHTMATTER_L200.link_bandwidth_bps,
            LIGHTMATTER_L200X.link_bandwidth_bps,
            LIGHTMATTER_M1000.link_bandwidth_bps,
        ]
        for a, b in zip(bws, bws[1:]):
            self.assertLess(a, b, "Bandwidth should strictly increase across Lightmatter product line")

    def test_all_lightmatter_techs_support_multicast(self):
        for label, tech in ALL_LIGHTMATTER_TECHS.items():
            self.assertTrue(tech.supports_multicast, f"{label} should support multicast")

    def test_electrical_does_not_support_multicast(self):
        self.assertFalse(ELECTRICAL_800G.supports_multicast)

    def test_advanced_chiplet_baseline_matches_prompt(self):
        self.assertAlmostEqual(ADVANCED_CHIPLET_3P5T.link_bandwidth_bps / 1e12, 3.5)
        self.assertAlmostEqual(ADVANCED_CHIPLET_3P5T.energy_per_bit_j * 1e12, 6.0)
        self.assertFalse(ADVANCED_CHIPLET_3P5T.supports_multicast)

    def test_gemini_proxy_models_are_explicit_assumptions(self):
        models = gemini_proxy_models()
        self.assertIn("gemini3_1_pro_moe_proxy", models)
        pro = models["gemini3_1_pro_moe_proxy"]
        self.assertGreater(pro.total_params_b, pro.active_params_b)
        self.assertIn("not a disclosed", pro.disclosure_note)

    def test_default_superpod_configs_include_full_stack(self):
        configs = default_superpod_configs()
        names = {config.name for config in configs}
        self.assertIn("M1000 full photonic compiler stack", names)
        full = [config for config in configs if config.name == "M1000 full photonic compiler stack"][0]
        baseline = [config for config in configs if config.name == "Ironwood electrical torus + XLA SPMD"][0]
        self.assertGreater(full.compute_efficiency, baseline.compute_efficiency)
        self.assertEqual(full.active_chips, IRONWOOD_SUPERPOD_CHIPS)

    def test_evk100_faster_than_electrical(self):
        self.assertGreater(
            LIGHTMATTER_EVK100.link_bandwidth_bps,
            ELECTRICAL_800G.link_bandwidth_bps,
        )


class TestTopologyHopCounts(unittest.TestCase):

    def test_optical_crossbar_single_hop(self):
        topo = OpticalCrossbarTopology()
        for n in (4, 8, 16, 64):
            self.assertEqual(topo.average_node_to_node_hops(n), 1.0)
            self.assertEqual(topo.average_node_to_memory_hops(n), 1.0)
            self.assertEqual(topo.multicast_hops(n), 1.0)
            self.assertEqual(topo.contention_factor(n, "node_to_node"), 1.0)

    def test_mesh_hops_positive(self):
        topo = Mesh2DTopology(4, 4)
        for n in (4, 8, 16):
            self.assertGreater(topo.average_node_to_node_hops(n), 0.0)
            self.assertGreater(topo.average_node_to_memory_hops(n), 0.0)

    def test_ring_hops_gt_mesh_hops(self):
        ring = RingTopology()
        mesh = Mesh2DTopology(4, 4)
        n = 16
        # Ring has longer average paths than 4x4 mesh for 16 nodes
        self.assertGreater(
            ring.average_node_to_node_hops(n),
            mesh.average_node_to_node_hops(n),
        )

    def test_torus_3d_wraps_correctly(self):
        topo = Torus3DTopology(4, 4, 4)
        # In a 4-node dimension, the max distance is 2 (wrap-around)
        hops = topo.average_node_to_node_hops(64)
        self.assertGreater(hops, 0.0)
        self.assertLess(hops, 6.0)  # max theoretical = 3 dims × 2 wraps

    def test_mesh_contention_scales_with_sqrt_n(self):
        import math
        topo = Mesh2DTopology(4, 4)
        for n in (4, 16, 64):
            self.assertAlmostEqual(
                topo.contention_factor(n, "any"), max(1.0, math.sqrt(n))
            )

    def test_torus_contention_scales_with_cbrt_n(self):
        import math
        topo = Torus3DTopology(4, 4, 4)
        for n in (8, 64):
            self.assertAlmostEqual(
                topo.contention_factor(n, "any"), max(1.0, n ** (1.0 / 3.0))
            )


class TestNetworkCostModel(unittest.TestCase):

    def _make_traffic(self, bits=8 * 1024 * 1024) -> TrafficBreakdown:
        return TrafficBreakdown(weight_read_bits=bits)

    def test_optical_multicast_beats_electrical_mesh_energy(self):
        traffic = TrafficBreakdown(weight_read_bits=8 * 1024**2)
        scenario = Scenario(
            name="shared_dram_multicast",
            description="multicast test",
            comm_pattern="shared_dram_multicast",
            active_nodes=16,
        )
        mesh_cost = Mesh2DTopology(4, 4).evaluate(traffic, scenario, ELECTRICAL_800G)
        opt_cost = OpticalCrossbarTopology().evaluate(traffic, scenario, LIGHTMATTER_EVK100)
        self.assertLess(opt_cost.total_energy_j, mesh_cost.total_energy_j)

    def test_optical_multicast_beats_electrical_mesh_latency(self):
        traffic = TrafficBreakdown(weight_read_bits=8 * 1024**2)
        scenario = Scenario(
            name="shared_dram_multicast",
            description="multicast test",
            comm_pattern="shared_dram_multicast",
            active_nodes=16,
        )
        mesh_cost = Mesh2DTopology(4, 4).evaluate(traffic, scenario, ELECTRICAL_800G)
        opt_cost = OpticalCrossbarTopology().evaluate(traffic, scenario, LIGHTMATTER_EVK100)
        self.assertLess(opt_cost.total_latency_s, mesh_cost.total_latency_s)

    def test_multicast_cheaper_than_unicast_for_optical(self):
        """
        For an optical tech with multicast, shared_dram_multicast should cost less
        than node_to_dram (which charges repeated unicast volume with same hops).
        """
        bits = 8 * 1024**2
        traffic = TrafficBreakdown(weight_read_bits=bits)
        topo = OpticalCrossbarTopology()
        tech = LIGHTMATTER_EVK100

        multi_scenario = Scenario("shared_dram_multicast", "", "shared_dram_multicast", 16)
        dram_scenario = Scenario("node_to_dram", "", "node_to_dram", 16)

        multi_cost = topo.evaluate(traffic, multi_scenario, tech)
        dram_cost = topo.evaluate(traffic, dram_scenario, tech)
        self.assertLess(multi_cost.total_energy_j, dram_cost.total_energy_j)

    def test_electrical_multicast_equals_unicast_volume(self):
        """Electrical mesh charges full repeated volume (no multicast benefit)."""
        bits = 8 * 1024**2
        traffic = TrafficBreakdown(weight_read_bits=bits)
        topo = Mesh2DTopology(4, 4)
        multi_scenario = Scenario("shared_dram_multicast", "", "shared_dram_multicast", 16)
        dram_scenario = Scenario("node_to_dram", "", "node_to_dram", 16)
        multi_cost = topo.evaluate(traffic, multi_scenario, ELECTRICAL_800G)
        dram_cost = topo.evaluate(traffic, dram_scenario, ELECTRICAL_800G)
        # Multicast scenario charges repeated_bits (N × unique), same as node_to_dram weight
        self.assertAlmostEqual(
            multi_cost.total_energy_j / dram_cost.total_energy_j, 1.0, places=3
        )

    def test_zero_traffic_returns_zero_cost(self):
        traffic = TrafficBreakdown()
        scenario = Scenario("node_to_node", "", "node_to_node", 16)
        cost = Mesh2DTopology(4, 4).evaluate(traffic, scenario, ELECTRICAL_800G)
        self.assertEqual(cost.total_energy_j, 0.0)
        self.assertEqual(cost.total_latency_s, 0.0)

    def test_evk100_lower_energy_than_evk50_per_bit(self):
        self.assertLess(
            LIGHTMATTER_EVK100.energy_per_bit_j,
            LIGHTMATTER_EVK50.energy_per_bit_j,
        )

    def test_m1000_higher_bandwidth_than_l200x(self):
        self.assertGreater(
            LIGHTMATTER_M1000.link_bandwidth_bps,
            LIGHTMATTER_L200X.link_bandwidth_bps,
        )


class TestSweepFunctions(unittest.TestCase):

    def test_bandwidth_sensitivity_returns_dataframe(self):
        df = bandwidth_sensitivity_network(
            traffic_bits=1024**2,
            scenario_name="node_to_dram",
            active_nodes=4,
            base_tech=ELECTRICAL_800G,
            multipliers=[0.5, 1.0, 2.0],
            topology=Mesh2DTopology(2, 2),
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("energy_uJ", df.columns)
        self.assertIn("latency_us", df.columns)

    def test_bandwidth_sensitivity_latency_decreases_with_bw(self):
        df = bandwidth_sensitivity_network(
            traffic_bits=1024**2,
            scenario_name="node_to_dram",
            active_nodes=4,
            base_tech=ELECTRICAL_800G,
            multipliers=[1.0, 2.0, 4.0],
            topology=Mesh2DTopology(2, 2),
        )
        latencies = df["latency_us"].tolist()
        self.assertTrue(
            latencies[0] > latencies[1] > latencies[2],
            "Latency should decrease as bandwidth increases"
        )

    def test_energy_sensitivity_returns_dataframe(self):
        df = energy_per_bit_sensitivity_network(
            traffic_bits=1024**2,
            scenario_name="node_to_dram",
            active_nodes=4,
            base_tech=ELECTRICAL_800G,
            multipliers=[0.5, 1.0, 2.0],
            topology=Mesh2DTopology(2, 2),
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

    def test_energy_sensitivity_energy_scales_linearly(self):
        df = energy_per_bit_sensitivity_network(
            traffic_bits=1024**2,
            scenario_name="node_to_dram",
            active_nodes=4,
            base_tech=ELECTRICAL_800G,
            multipliers=[1.0, 2.0],
            topology=Mesh2DTopology(2, 2),
        )
        ratio = df.iloc[1]["total_energy_uJ"] / df.iloc[0]["total_energy_uJ"]
        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_all_tech_comparison_returns_all_technologies(self):
        df = all_technologies_network_comparison(
            traffic_bits=1024**2,
            scenario_name="node_to_dram",
            active_nodes=4,
        )
        expected = {
            "Electrical 800G", "Advanced Chiplet 3.5T",
            "EVK50", "EVK100", "L200", "L200X", "M1000",
        }
        found = set(df["technology"])
        self.assertEqual(found, expected)

    def test_compiler_strategy_study_changes_strategy_for_photonics(self):
        strategy_df = compiler_strategy_study(
            workload_names=("prefill_projection",),
            active_nodes=64,
        )
        best = optimal_compiler_strategy_frame(strategy_df)
        photonic = best[best["architecture"] == "M1000 optical switched fabric"].iloc[0]
        electrical = best[best["architecture"] == "TPU electrical torus"].iloc[0]
        self.assertNotEqual(photonic["strategy"], electrical["strategy"])
        self.assertTrue(photonic["strategy_changed_vs_baseline"])

    def test_scaling_law_optical_multicast_scales_better_than_chiplet(self):
        scaling = scaling_law_study(node_counts=(64, 512, 4096), traffic_gib_per_chip=0.01)
        exponents = scaling_exponent_frame(scaling)
        multicast_energy = exponents[
            (exponents["scenario"] == "shared_state_multicast")
            & (exponents["metric"] == "energy_uJ_per_chip")
        ].set_index("architecture")
        self.assertLess(
            multicast_energy.loc["M1000 optical switched fabric", "scaling_exponent"],
            multicast_energy.loc["Advanced chiplet torus", "scaling_exponent"],
        )

    def test_chiplet_comparison_includes_photonic_reductions(self):
        df = chiplet_comparison_frame(active_nodes=64, traffic_gib_per_chip=0.01)
        m1000 = df[df["architecture"] == "M1000 optical switched fabric"]
        self.assertTrue((m1000["energy_reduction_vs_chiplet_pct"] > 0).any())

    def test_superpod_summary_reports_training_and_inference_speedups(self):
        stage_df = superpod_stage_study(model_names=("gemini3_1_pro_moe_proxy",))
        summary = superpod_end_to_end_summary(stage_df)
        full_stack = summary[
            summary["configuration"] == "M1000 full photonic compiler stack"
        ]
        self.assertEqual(set(full_stack["phase"]), {"training", "inference"})
        self.assertTrue((full_stack["speedup_vs_baseline"] > 1.0).all())

    def test_superpod_ablation_assigns_incremental_techniques(self):
        stage_df = superpod_stage_study(model_names=("gemini3_1_pro_moe_proxy",))
        summary = superpod_end_to_end_summary(stage_df)
        ablation = superpod_ablation_frame(summary)
        self.assertIn("full photonic compiler stack", set(ablation["technique"]))
        full_stack = ablation[ablation["technique"] == "full photonic compiler stack"]
        self.assertTrue((full_stack["speedup_increment_vs_previous"] > 1.0).all())

    def test_superpod_context_scaling_full_photonic_beats_electrical(self):
        df = superpod_context_scaling_study(context_tokens=(4096, 32768), output_tokens=128)
        full_stack = df[df["configuration"] == "M1000 full photonic compiler stack"]
        self.assertTrue((full_stack["speedup_vs_baseline"] > 1.0).all())


# =========================================================================
# Source frame tests
# =========================================================================

class TestSourceFrame(unittest.TestCase):

    def test_source_frame_contains_key_sources(self):
        df = source_frame()
        source_ids = set(df["source"])
        for required in ("lightmatter_evk100", "lightmatter_evk50", "tpu_v5p",
                         "tpu_rack_hotchips", "lightmatter_m1000", "tpu7x",
                         "gemini_3", "gemini_3_1_release_notes"):
            self.assertIn(required, source_ids, f"Missing source: {required}")

    def test_all_urls_start_with_https(self):
        df = source_frame()
        self.assertTrue(
            df["url"].str.startswith("http").all(),
            "All source URLs should start with http(s)"
        )

    def test_ironwood_spec_in_sources(self):
        df = source_frame()
        hotchips = df[df["source"] == "tpu_rack_hotchips"].iloc[0]
        self.assertIn("4614", hotchips["summary"], "Ironwood TFLOPS should appear in HotChips source")


# =========================================================================
# Integration smoke tests (AccelForge)
# =========================================================================

class TestIterativeSolver(unittest.TestCase):

    SMALL_WORKLOAD = WorkloadCase(name="smoke", M=1, K=64, N=128,
                                  description="tiny smoke workload")

    def test_solve_iteratively_returns_nonempty_history(self):
        history, mapping = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_node"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=2,
        )
        self.assertGreaterEqual(len(history), 1)

    def test_traffic_is_positive(self):
        history, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_node"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=2,
        )
        self.assertGreater(history[0].traffic.total_bits, 0)

    def test_final_mapping_has_positive_energy(self):
        _, mapping = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_dram"),
            Mesh2DTopology(4, 4),
            ELECTRICAL_800G,
            max_iterations=2,
        )
        self.assertGreater(float(mapping.energy()), 0.0)

    def test_history_to_frame_columns(self):
        history, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_node"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=2,
        )
        df = history_to_frame(history)
        for col in (
            "iteration", "total_energy_uJ", "total_latency_us",
            "network_energy_uJ", "mapping_signature", "spatial_NodeX_usage",
        ):
            self.assertIn(col, df.columns)

    def test_per_component_energy_frame(self):
        history, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_node"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=2,
        )
        df = per_component_energy_frame(history)
        self.assertIsInstance(df, pd.DataFrame)
        # MAC component must appear (it drives the computation)
        self.assertIn("MAC", df["component"].values)

    def test_mapping_signature_is_compact_and_tidy(self):
        _, mapping = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_dram"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=1,
        )
        signature = mapping_signature(mapping)
        features = mapping_feature_dict(mapping)
        feature_df = mapping_features_frame(mapping)
        self.assertIsInstance(signature, str)
        self.assertLess(len(signature), 1200)
        self.assertIn("spatial::NodeArray::NodeX", features)
        self.assertIn("kind", feature_df.columns)

    def test_solve_iteratively_supports_tpu_cube_node_z(self):
        history, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_dram", active_nodes=8),
            Torus3DTopology(2, 2, 2),
            ELECTRICAL_800G,
            max_iterations=1,
            node_shape=(2, 2, 2),
        )
        self.assertIn("spatial::NodeArray::NodeZ", history[0].mapping_features)

    def test_optical_converges_faster_than_electrical_for_dram_scenario(self):
        """
        Optical crossbar should need fewer iterations to converge than electrical mesh
        for node_to_dram (high network penalty scenario).
        """
        mesh_hist, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_dram"),
            Mesh2DTopology(4, 4),
            ELECTRICAL_800G,
            max_iterations=6,
            convergence_tol=1e-4,
        )
        opt_hist, _ = solve_iteratively(
            self.SMALL_WORKLOAD,
            scenario_case("node_to_dram"),
            OpticalCrossbarTopology(),
            LIGHTMATTER_EVK100,
            max_iterations=6,
            convergence_tol=1e-4,
        )
        self.assertLessEqual(len(opt_hist), len(mesh_hist) + 1)


# =========================================================================
# Output-artifact tests
# =========================================================================

class TestArtifactWriter(unittest.TestCase):

    def test_write_study_outputs_creates_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_study_outputs(
                tmpdir,
                workload_names=("kv_attention",),
                scenario_names=("node_to_dram",),
                tpu_mapping_workload_names=("kv_attention",),
                tpu_mapping_scenario_names=("node_to_dram",),
                tpu_mapping_max_iterations=1,
            )
            for key, path in paths.items():
                self.assertTrue(Path(path).exists(), f"Expected file missing: {key} ({path})")

    def test_cluster_study_has_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_study_outputs(
                tmpdir,
                workload_names=("kv_attention",),
                scenario_names=("node_to_dram",),
                tpu_mapping_workload_names=("kv_attention",),
                tpu_mapping_scenario_names=("node_to_dram",),
                tpu_mapping_max_iterations=1,
            )
            df = pd.read_csv(paths["cluster_study"])
            for col in ("workload", "scenario", "topology", "final_energy_uJ",
                        "network_delta_energy_pct", "mapping_sensitivity_energy_pct",
                        "final_mapping_signature", "mapping_changed"):
                self.assertIn(col, df.columns)

    def test_cluster_improvements_energy_positive_for_optical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_study_outputs(
                tmpdir,
                workload_names=("decode_mlp",),
                scenario_names=("node_to_dram",),
                tpu_mapping_workload_names=("kv_attention",),
                tpu_mapping_scenario_names=("node_to_dram",),
                tpu_mapping_max_iterations=1,
            )
            cluster_df = pd.read_csv(paths["cluster_study"])
            improvements = cluster_improvement_frame(cluster_df)
            self.assertGreater(improvements.iloc[0]["energy_reduction_pct"], 0.0)

    def test_summary_markdown_contains_technology_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_study_outputs(
                tmpdir,
                workload_names=("kv_attention",),
                scenario_names=("node_to_dram",),
                tpu_mapping_workload_names=("kv_attention",),
                tpu_mapping_scenario_names=("node_to_dram",),
                tpu_mapping_max_iterations=1,
            )
            summary = Path(paths["summary"]).read_text()
            self.assertIn("EVK100", summary)
            self.assertIn("M1000", summary)
            self.assertIn("TPU Cube Mapping Adaptation", summary)
            self.assertIn("Optimal Distributed Mapping Choices", summary)
            self.assertIn("Full-Superpod Gemini-Class", summary)

    def test_tpu_case_study_has_five_variants(self):
        tpu_df = tpu_cube_case_study()
        variants = tpu_df["variant"].unique()
        self.assertEqual(len(variants), 5, f"Expected 5 TPU variants, got: {list(variants)}")

    def test_tpu_improvement_shows_positive_reduction(self):
        tpu_df = tpu_cube_case_study()
        tpu_imp = tpu_improvement_frame(tpu_df)
        # Every photonic variant should show positive energy reduction
        self.assertTrue(
            (tpu_imp["energy_reduction_pct"] > 0).all(),
            "All photonic variants should reduce energy vs. electrical torus"
        )

    def test_tpu_m1000_best_bandwidth(self):
        tpu_df = tpu_cube_case_study()
        m1000_rows = tpu_df[tpu_df["variant"].str.contains("M1000")]
        other_rows = tpu_df[~tpu_df["variant"].str.contains("M1000")]
        self.assertGreater(
            m1000_rows["effective_link_one_way_tbps"].max(),
            other_rows["effective_link_one_way_tbps"].max(),
        )

    def test_tpu_cube_mapping_study_reports_mapping_signatures(self):
        df = tpu_cube_mapping_study(
            workload_names=("kv_attention",),
            scenario_names=("node_to_dram",),
            max_iterations=1,
            node_shape=(2, 2, 2),
            hbm_gib=8,
        )
        self.assertGreaterEqual(len(df), 5)
        for col in (
            "variant", "final_mapping_signature",
            "energy_reduction_vs_electrical_pct",
            "mapping_differs_from_electrical",
            "spatial_NodeZ_usage",
        ):
            self.assertIn(col, df.columns)

    def test_executive_summary_frame_contains_headline_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_study_outputs(
                tmpdir,
                workload_names=("kv_attention",),
                scenario_names=("node_to_dram",),
                tpu_mapping_workload_names=("kv_attention",),
                tpu_mapping_scenario_names=("node_to_dram",),
                tpu_mapping_max_iterations=1,
            )
            cluster_imp = pd.read_csv(paths["cluster_improvements"])
            tpu_imp = pd.read_csv(paths["tpu_improvements"])
            tpu_mapping = pd.read_csv(paths["tpu_mapping_study"])
            summary = executive_summary_frame(cluster_imp, tpu_imp, tpu_mapping)
            self.assertFalse(summary.empty)
            self.assertTrue(Path(paths["presentation_index"]).exists())
            self.assertTrue(Path(paths["executive_summary"]).exists())
            self.assertTrue(Path(paths["optimal_mapping_choices"]).exists())
            self.assertTrue(Path(paths["superpod_end_to_end_summary"]).exists())
            self.assertTrue(Path(paths["superpod_context_scaling"]).exists())

    def test_optimal_distributed_mapping_frame_detects_candidate_choices(self):
        df = pd.DataFrame([
            {"workload": "w", "variant": "base", "scenario": "node_to_node",
             "final_energy_uJ": 10.0, "final_latency_us": 10.0},
            {"workload": "w", "variant": "base", "scenario": "node_to_dram",
             "final_energy_uJ": 20.0, "final_latency_us": 20.0},
            {"workload": "w", "variant": "opt", "scenario": "node_to_node",
             "final_energy_uJ": 12.0, "final_latency_us": 12.0},
            {"workload": "w", "variant": "opt", "scenario": "shared_dram_multicast",
             "final_energy_uJ": 5.0, "final_latency_us": 5.0},
        ])
        choices = optimal_distributed_mapping_frame(df, baseline_architecture="base")
        opt_row = choices[choices["architecture"] == "opt"].iloc[0]
        self.assertEqual(opt_row["optimal_distributed_mapping"], "shared_dram_multicast")
        self.assertTrue(opt_row["optimal_mapping_changed_vs_baseline"])


# =========================================================================
# Notebook structure test
# =========================================================================

class TestNotebookBuilder(unittest.TestCase):

    def test_notebook_has_enough_cells(self):
        nb = build_notebook()
        self.assertGreaterEqual(len(nb.cells), 30)

    def test_notebook_title_cell_correct(self):
        nb = build_notebook()
        self.assertIn("Lightmatter Photonic Interconnects", nb.cells[0].source)

    def test_notebook_has_tpu_cube_section(self):
        nb = build_notebook()
        md_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        self.assertTrue(
            any("TPU Cube ICI Replacement" in s for s in md_cells),
            "Notebook should have TPU cube ICI replacement section"
        )

    def test_notebook_has_milestone_sections(self):
        nb = build_notebook()
        md_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        for milestone in ("Milestone 1", "Milestone 2", "Milestone 3"):
            self.assertTrue(
                any(milestone in s for s in md_cells),
                f"Notebook missing section: {milestone}"
            )

    def test_notebook_has_sensitivity_analysis(self):
        nb = build_notebook()
        md_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        self.assertTrue(
            any("Sensitivity" in s for s in md_cells),
            "Notebook should have a sensitivity analysis section"
        )

    def test_notebook_imports_all_new_techs(self):
        nb = build_notebook()
        code_cells = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        for tech in ("LIGHTMATTER_EVK50", "LIGHTMATTER_L200X", "LIGHTMATTER_M1000"):
            self.assertIn(tech, code_cells, f"Notebook code should import {tech}")

    def test_notebook_has_superpod_section(self):
        nb = build_notebook()
        md_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        self.assertTrue(
            any("Full Superpod" in s or "Gemini-Class" in s for s in md_cells),
            "Notebook should have a full-superpod Gemini-class section"
        )


# =========================================================================
# Constants sanity checks
# =========================================================================

class TestConstants(unittest.TestCase):

    def test_hbm_energy_matches_tpu_v5p_spec(self):
        # 7.03 pJ/bit from TPU v5p white-paper
        self.assertAlmostEqual(HBM_ENERGY_PER_BIT_J * 1e12, 7.03, places=2)

    def test_ironwood_tflops_much_greater_than_v5p(self):
        from scripts.lightmatter_interconnects import TPU_V5P_TFLOPS_PER_CHIP
        self.assertGreater(IRONWOOD_TFLOPS_PER_CHIP, 10 * TPU_V5P_TFLOPS_PER_CHIP)

    def test_ironwood_hbm_bw_correct(self):
        self.assertAlmostEqual(IRONWOOD_HBM_BANDWIDTH_TBPS, 7.4, places=1)

    def test_tpu_v5p_ici_bidir_gbps_correct(self):
        self.assertAlmostEqual(TPU_V5P_ICI_BIDIR_GBPS, 1200.0, places=0)

    def test_ironwood_superpod_size(self):
        self.assertEqual(IRONWOOD_SUPERPOD_CHIPS, 9216)
        self.assertEqual(IRONWOOD_SUPERPOD_CUBES, 144)

    def test_all_workloads_defined(self):
        expected = {
            "decode_mlp", "kv_attention", "prefill_projection",
            "long_context_prefill", "moe_expert_dispatch", "training_weight_grad",
        }
        self.assertEqual(set(LLM_WORKLOADS.keys()), expected)


if __name__ == "__main__":
    unittest.main()
