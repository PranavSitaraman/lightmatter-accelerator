# Superpod Workflow Summary

Phases: inference, training
Active chips: 9216
Models: gemini3_1_pro_moe_proxy, gemini3_deepthink_long_context_proxy

## Modeling Notes

- The topology model is not a pure bandwidth scalar: hop count, contention, router latency, and multicast support all affect transfer cost.
- Stage latency uses an exposed-stall model, `T_stage = max(T_compute, T_HBM) + (1 - overlap) * T_network`, so faster optical links can reduce both raw transport time and visible stalls.
- In TPU/XLA terms, `node_to_node` corresponds to collective activation exchange, `node_to_dram` corresponds to remote HBM or KV fetch, and `shared_dram_multicast` corresponds to broadcast cache fill of shared state.
- The model is analytic rather than packet-level and does not simulate queueing or OCS reconfiguration timing.

## Best Configurations

- Gemini 3 Deep Think long-context proxy / inference: M1000 full photonic compiler stack (14.50x latency speedup, 51.07% energy reduction).
- Gemini 3 Deep Think long-context proxy / training: M1000 full photonic compiler stack (1.43x latency speedup, 0.67% energy reduction).
- Gemini 3.1 Pro-class sparse MoE proxy / inference: M1000 full photonic compiler stack (11.02x latency speedup, 45.27% energy reduction).
- Gemini 3.1 Pro-class sparse MoE proxy / training: M1000 full photonic compiler stack (1.42x latency speedup, 0.54% energy reduction).

## Context Scaling

Long-context inference scaling is included because only inference has prompt/output-token context-length sweeps in this model.
