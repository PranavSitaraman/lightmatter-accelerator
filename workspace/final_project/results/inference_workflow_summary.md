# Inference Workflow Summary

Phases: inference
Active chips: 9216
Models: gemini3_1_pro_moe_proxy, gemini3_deepthink_long_context_proxy

## Best Configurations

- Gemini 3 Deep Think long-context proxy / inference: M1000 full photonic compiler stack (14.50x latency speedup, 51.07% energy reduction).
- Gemini 3.1 Pro-class sparse MoE proxy / inference: M1000 full photonic compiler stack (11.02x latency speedup, 45.27% energy reduction).

## Context Scaling

Long-context inference scaling is included because only inference has prompt/output-token context-length sweeps in this model.
