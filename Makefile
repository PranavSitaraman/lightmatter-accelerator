DOCKER_IMAGE ?= timeloopaccelergy/accelforge:latest-amd64
PROJECT_DIR := /home/workspace/workspace/final_project
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

run-in-docker = docker run --rm --user "$(HOST_UID):$(HOST_GID)" -v "$(CURDIR):/home/workspace" -w $(PROJECT_DIR) $(DOCKER_IMAGE) bash -lc $(1)

.PHONY: all study notebook test superpod inference training validate clean-results

# Run tests, generate study outputs, then execute the notebook.
all: test study notebook

# Generate all CSV, Markdown, and presentation-ready outputs.
study:
	$(call run-in-docker,'python3 scripts/lightmatter_interconnects.py all --output-dir results')

# Regenerate and execute the notebook (40-minute timeout).
notebook:
	$(call run-in-docker,'python3 scripts/generate_lightmatter_notebook.py --output lightmatter_interconnects.ipynb && jupyter nbconvert --to notebook --execute --inplace lightmatter_interconnects.ipynb --ExecutePreprocessor.timeout=2400')

# Run the full 70-test suite.
test:
	$(call run-in-docker,'python3 -m unittest discover -s tests -v')

# Regenerate only the end-to-end 9216-chip training/inference workflow outputs.
superpod:
	$(call run-in-docker,'python3 scripts/lightmatter_interconnects.py superpod --output-dir results')

# Regenerate only inference-phase superpod outputs and context scaling.
inference:
	$(call run-in-docker,'python3 scripts/lightmatter_interconnects.py inference --output-dir results')

# Regenerate only training-phase superpod outputs.
training:
	$(call run-in-docker,'python3 scripts/lightmatter_interconnects.py training --output-dir results')

# Production validation path: tests plus full artifact regeneration.
validate: test study

# Remove generated CSV/PNG/Markdown outputs (notebook itself is kept).
clean-results:
	rm -rf workspace/final_project/results
