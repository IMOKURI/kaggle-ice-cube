.PHONY: help notebook
.DEFAULT_GOAL := help
SHELL = /bin/bash

NOW = $(shell date '+%Y%m%d-%H%M%S-%N')
GROUP := $(shell date '+%Y%m%d-%H%M')
HEAD_COMMIT = $(shell git rev-parse HEAD)

# ponkots-kaggle-gpu 起動した notebook で graphnet をインストールして docker commit でイメージを保存しておく
up: ## Start jupyter notebook
	docker run -d --name notebook -u $(shell id -u):$(shell id -g) --gpus all \
		-v $(shell pwd):/home/jovyan/working -w /home/jovyan/working \
		-v /data/home/shared:/home/jovyan/input \
		-e http_proxy=http://web-proxy.jp.hpecorp.net:8080/ \
		-e https_proxy=http://web-proxy.jp.hpecorp.net:8080/ \
		-e no_proxy="localhost,127.0.0.1,127.0.1.1,sdf,16.171.32.147" \
		-e PYTHONUSERBASE=/home/$(shell whoami)/.local \
		-e XDG_RUNTIME_DIR=/home/$(shell whoami)/.local/share \
		--shm-size=2048g \
		-p 8888:8888 \
		ponkots-kaggle-gpu-ice-cube \
		jupyter notebook --no-browser --ip="0.0.0.0"

down: ## Stop jupyter notebook
	docker stop notebook
	docker rm notebook

create-db: ## Create DB
	docker run -d -u $(shell id -u):$(shell id -g) \
		-v $(shell pwd):/home/jovyan/working -w /home/jovyan/working \
		-v /data/home/shared:/home/jovyan/input \
		-e PYTHONUSERBASE=/home/$(shell whoami)/.local \
		-e XDG_RUNTIME_DIR=/home/$(shell whoami)/.local/share \
		--shm-size=2048g \
		ponkots-kaggle-gpu-ice-cube \
		python ./00-create-db.py

train: ## Run training
	rm -rf ./checkpoints
	rm -f ./nohup.out
	LD_LIBRARY_PATH="/home/sugiyama/miniconda3/envs/graphnet/lib" nohup  python ./05-training.py &

push: clean ## Publish notebook.
	@rm -f ./notebook/inference.ipynb
	@python encode.py .
	@cd ./notebook && kaggle kernels push

monitoring-lb: ## Monitor LB
	rm -f ./nohup.out
	nohup  python ./monitoring_lb.py &

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

release-gpu: ## Release GPU memory.
	kill $(shell lsof -t /dev/nvidia*)

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
