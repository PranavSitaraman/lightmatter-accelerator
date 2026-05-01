# Final Project

This project evaluates **Lightmatter photonic interconnects** as an emerging technology for accelerator clusters and TPU-like cube-scale systems using **AccelForge**.

The final deliverable lives in [workspace/final_project](/home/pranav/6.5930/final-project/workspace/final_project) and includes:

- an executed reproducibility notebook,
- reusable study code and a CLI,
- raw CSV/Markdown audit artifacts plus a six-figure final submission package,
- a compiler/dataflow strategy model that shows mapping changes beyond constant-factor link substitutions,
- distance-parametrized electrical/chiplet/photonic link-energy models,
- strong- and weak-scaling studies plus OCS/OPS optical switching sensitivity,
- superpod-scale scaling-law and advanced chiplet-baseline comparisons,
- a 9216-chip Ironwood Gemma/Gemini-class end-to-end training and inference model with staged ablations,
- automated unit/integration tests and phase-specific training/inference workflow outputs,
- a cleaned architecture/workload setup focused on the final project rather than the earlier lab template.

## Main Artifacts

### Submission deliverables

- Self-contained project report (IEEE conference format, based on the course template): [workspace/final_project/paper.tex](/home/pranav/6.5930/final-project/workspace/final_project/paper.tex) -> `paper.pdf`
- Technical summary sheet (1 page): [workspace/final_project/technical_summary.tex](/home/pranav/6.5930/final-project/workspace/final_project/technical_summary.tex) -> `technical_summary.pdf`
- Poster: [workspace/final_project/poster.tex](/home/pranav/6.5930/final-project/workspace/final_project/poster.tex) -> `poster.pdf`
- Presentation: [workspace/final_project/presentation.tex](/home/pranav/6.5930/final-project/workspace/final_project/presentation.tex) -> `presentation.pdf`

### Code, data, and figures

- Notebook: [workspace/final_project/lightmatter_interconnects.ipynb](/home/pranav/6.5930/final-project/workspace/final_project/lightmatter_interconnects.ipynb)
- Study engine: [workspace/final_project/scripts/lightmatter_interconnects.py](/home/pranav/6.5930/final-project/workspace/final_project/scripts/lightmatter_interconnects.py)
- Notebook generator: [workspace/final_project/scripts/generate_lightmatter_notebook.py](/home/pranav/6.5930/final-project/workspace/final_project/scripts/generate_lightmatter_notebook.py)
- Architecture: [workspace/final_project/arch/tpu_lightmatter_cluster.yaml](/home/pranav/6.5930/final-project/workspace/final_project/arch/tpu_lightmatter_cluster.yaml)
- Workload: [workspace/final_project/workload/single_matmul.yaml](/home/pranav/6.5930/final-project/workspace/final_project/workload/single_matmul.yaml)
- Canonical final submission package: [workspace/final_project/results/presentation](/home/pranav/6.5930/final-project/workspace/final_project/results/presentation)

The main notebook is regenerated from code and fully executed as part of validation. The final report, poster, summary, and deck use the curated six-figure package in `results/presentation`; the parent `results/` directory remains the raw reproducibility trail. The test suite covers the network model, AccelForge smoke paths, generated artifacts, notebook structure, distance physics, strong/weak scaling, OCS/OPS sensitivity, and superpod workflows.

## Added Studies

- **Compiler/dataflow adaptation:** compares XLA/GSPMD-style sharded collectives, remote pooled-HBM streaming, replicated HBM caching, and photonic multicast cache fills.
- **Scaling laws:** evaluates how per-chip energy and latency scale from one 64-chip cube toward TPU superpod scale.
- **Advanced chiplet baseline:** compares photonics against a stronger 3.5 Tb/s, 6 pJ/bit/hop chiplet interconnect rather than only against conventional electrical ICI.
- **Gemma/Gemini-class full-superpod model:** evaluates public Gemma 4 open-model proxies and Gemini-class stress proxies for dense, sparse MoE, and long-context inference/training on a 9216-chip Ironwood pod.
- **Distance physics:** separates conversion and distance-dependent transmission energy so larger electrical topologies pay for longer physical paths, while photonic links remain conversion dominated.
- **Strong/weak scaling and switching:** compares latency compression, throughput-style weak scaling, and OCS-vs-OPS optical semantics.

## Usage

From this directory:

```bash
make study
make inference
make training
make superpod
make notebook
make deliverables
make test
make validate
```

Study, notebook, and test commands run inside the AccelForge Docker image and write outputs under [workspace/final_project/results](/home/pranav/6.5930/final-project/workspace/final_project/results). `make deliverables` builds the report, presentation, poster, and technical summary PDFs from their LaTeX sources and removes LaTeX intermediates. `make inference`, `make training`, and `make superpod` regenerate only the end-to-end phase workflow artifacts without rerunning the slower AccelForge mapping sweeps.
