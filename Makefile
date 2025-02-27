PYTHON_VERSION := $(shell cat .python-version)

.PHONY: setup install run test cleanup_env

setup:
	@pyenv install --skip-existing $(PYTHON_VERSION)
	@poetry config virtualenvs.in-project true
	@poetry install

run:
	@poetry run python example_usage.py

test:
	@poetry run pytest


cleanup_env:
	@if poetry env info --path > /dev/null 2>&1; then \
		poetry env remove $$(basename $$(poetry env info --path)); \
	else \
		echo "No active Poetry environment found."; \
	fi