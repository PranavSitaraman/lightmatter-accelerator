# Final Project

This project evaluates **Lightmatter photonic interconnects** as an emerging technology for accelerator clusters and TPU-like cube-scale systems using **AccelForge**.

The final deliverable lives in [workspace/final_project](/home/pranav/6.5930/final-project/workspace/final_project) and includes:

- an executed exploratory notebook,
- reusable study code and a CLI,
- generated CSV, Markdown, and presentation-ready chart artifacts,
- automated unit/integration tests,
- a cleaned architecture/workload setup focused on the final project rather than the earlier lab template.

## Main Artifacts

- Notebook: [workspace/final_project/4_lightmatter_interconnects.ipynb](/home/pranav/6.5930/final-project/workspace/final_project/4_lightmatter_interconnects.ipynb)
- Study engine: [workspace/final_project/scripts/lightmatter_interconnects.py](/home/pranav/6.5930/final-project/workspace/final_project/scripts/lightmatter_interconnects.py)
- Notebook generator: [workspace/final_project/scripts/generate_lightmatter_notebook.py](/home/pranav/6.5930/final-project/workspace/final_project/scripts/generate_lightmatter_notebook.py)
- Architecture: [workspace/final_project/arch/tpu_lightmatter_cluster.yaml](/home/pranav/6.5930/final-project/workspace/final_project/arch/tpu_lightmatter_cluster.yaml)
- Workload: [workspace/final_project/workload/single_matmul.yaml](/home/pranav/6.5930/final-project/workspace/final_project/workload/single_matmul.yaml)
- Condensed presentation package: [workspace/final_project/results/presentation](/home/pranav/6.5930/final-project/workspace/final_project/results/presentation)

## Usage

From this directory:

```bash
make study
make notebook
make test
```

These commands run inside the AccelForge Docker image and write outputs under [workspace/final_project/results](/home/pranav/6.5930/final-project/workspace/final_project/results).
