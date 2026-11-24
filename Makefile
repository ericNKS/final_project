.DEFAULT_GOAL := help

PYTHON ?= .venv-gpu/bin/python
COMPOSE ?= $(if $(shell command -v docker 2>/dev/null),docker compose,podman compose)
APP_UID ?= $(shell id -u)
APP_GID ?= $(shell id -g)
APP_USERNS ?= $(if $(filter podman,$(firstword $(COMPOSE))),keep-id,host)
export APP_UID APP_GID APP_USERNS

SCENARIO ?= all
BACKEND ?= gpu
HOST ?= ::1
PORT ?= 8000
SOURCE ?=
RUN ?=
RESULTS ?=
CUTOFF ?=
ARGS ?=
CMD ?= --help

SOURCE_ARG = $(if $(strip $(SOURCE)),--source "$(SOURCE)",)
RUN_ARG = $(if $(strip $(RUN)),--run "$(RUN)",)
RESULTS_ARG = $(if $(strip $(RESULTS)),--results "$(RESULTS)",)

.PHONY: help setup directories db migrate serve train evaluate verify report import-history test test-gpu build up down logs container

help:
	@printf '%s\n' \
	  'Local: make setup | serve | train | evaluate | verify | report | import-history' \
	  'Banco: make db | migrate' \
	  'Testes: make test | test-gpu' \
	  'Contêiner: make build | up | down | logs | container CMD="verify --scenario all"' \
	  'Opções: SCENARIO=10k|2m|all BACKEND=gpu SOURCE=arquivo RUN=pasta RESULTS=pasta ARGS="..."' \
	  'API: HOST=::1 PORT=8000. Importação: SCENARIO=10k CUTOFF=data SOURCE=arquivo.' \
	  'COMPOSE="docker compose" ou COMPOSE="podman compose"; detecção automática por padrão.'

setup:
	@test -x "$(PYTHON)" || python3 -m venv .venv-gpu
	"$(PYTHON)" -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env

directories:
	mkdir -p resultados data .cache

db:
	$(COMPOSE) up -d --wait --wait-timeout 60 db

migrate: db
	"$(PYTHON)" main.py migrate

serve: migrate
	"$(PYTHON)" main.py serve --host "$(HOST)" --port "$(PORT)" $(ARGS)

train:
	"$(PYTHON)" main.py train --scenario "$(SCENARIO)" --backend "$(BACKEND)" $(SOURCE_ARG) $(RESULTS_ARG) $(ARGS)

evaluate verify report:
	"$(PYTHON)" main.py $@ --scenario "$(SCENARIO)" $(RUN_ARG) $(RESULTS_ARG) $(ARGS)

import-history: migrate
	@test -n "$(CUTOFF)" || { printf '%s\n' 'Informe CUTOFF=data e SCENARIO=10k ou 2m'; exit 2; }
	"$(PYTHON)" main.py import-history --scenario "$(SCENARIO)" --cutoff "$(CUTOFF)" $(SOURCE_ARG) $(ARGS)

test:
	"$(PYTHON)" -m pytest -q $(ARGS)

test-gpu:
	RUN_GPU_TESTS=1 "$(PYTHON)" -m pytest tests/test_gpu_runtime.py -q $(ARGS)

build: directories
	$(COMPOSE) build app

up: build db
	$(COMPOSE) run --rm --no-deps app migrate
	$(COMPOSE) up -d app

container: build
	$(COMPOSE) run --rm --no-deps app $(CMD)

logs:
	$(COMPOSE) logs -f app

down:
	$(COMPOSE) down
