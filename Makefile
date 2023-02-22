.PHONY: help preprocess postprocess
.DEFAULT_GOAL := help
SHELL = /bin/bash

NOW = $(shell date '+%Y%m%d-%H%M%S-%N')
GROUP := $(shell date '+%Y%m%d-%H%M')
HEAD_COMMIT = $(shell git rev-parse HEAD)

train: ## Run training.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus all \
		-v ~/.netrc:/home/jupyter/.netrc \
		-v $(shell pwd):/app -w /app \
		--shm-size=256g \
		ponkots-kaggle-gpu \
		python ./05-training.py  # +settings.run_fold=0

debug: ## Run training debug mode.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus '"device=1,2"' \
		-v $(shell pwd):/app -w /app \
		--shm-size=256g \
		ponkots-kaggle-gpu \
		python train.py settings.debug=True hydra.verbose=True +settings.run_fold=0

early-stop: ## Abort training gracefully.
	@touch abort-training.flag

push: clean ## Publish notebook.
	@rm -f ./notebook/inference.ipynb
	@python encode.py .
	@cd ./notebook && kaggle kernels push

push-model: ## Publish models.
	@cd ./dataset/training && \
		kaggle datasets version -m $(HEAD_COMMIT)-$(NOW) -r zip

clean: clean-build clean-pyc ## Remove all build and python artifacts.

clean-build: ## Remove build artifacts.
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove python artifacts.
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-training: ## Remove training artifacts.
	@rm -rf ./output ./multirun abort-training.flag

release-gpu: ## Release GPU memory.
	kill $(shell lsof -t /dev/nvidia*)

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
