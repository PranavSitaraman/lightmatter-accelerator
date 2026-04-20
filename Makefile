DOCKER_IMAGE ?= timeloopaccelergy/accelforge:latest-amd64
PROJECT_DIR := /home/workspace/workspace/final_project
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

run-in-docker = docker run --rm --user "$(HOST_UID):$(HOST_GID)" -v "$(CURDIR):/home/workspace" -w $(PROJECT_DIR) $(DOCKER_IMAGE) bash -lc $(1)

.PHONY: all study notebook test clean-results

# Run tests, generate study outputs, then execute the notebook.
all: test study notebook

# Generate all CSV, Markdown, and presentation-ready outputs.
study:
	$(call run-in-docker,'python3 scripts/lightmatter_interconnects.py all --output-dir results')

# Regenerate and execute the notebook (40-minute timeout).
notebook:
	$(call run-in-docker,'python3 scripts/generate_lightmatter_notebook.py --output lightmatter_interconnects.ipynb && jupyter nbconvert --to notebook --execute --inplace lightmatter_interconnects.ipynb --ExecutePreprocessor.timeout=2400')

# Run the full 68-test suite.
test:
	$(call run-in-docker,'python3 -m unittest discover -s tests -v')

# Remove generated CSV/PNG/Markdown outputs (notebook itself is kept).
clean-results:
	rm -rf workspace/final_project/results
