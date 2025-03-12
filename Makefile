PYTHON_VERSION := $(shell cat .python-version)
TARGET_FILE := ./tests/files/test_file.edf

.PHONY: setup install run test cleanup_env download_test_file

setup:
	@pyenv install --skip-existing $(PYTHON_VERSION)
	@poetry config virtualenvs.in-project true
	@poetry install --with dev

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

# Download test file target
download_test_file:
	@echo "Downloading and processing test file..."
	@wget -q "https://www.teuniz.net/edf_bdf_testfiles/test_generator.zip" -O test_generator.zip 
	@unzip -q test_generator.zip -d test_generator
	@rm test_generator.zip

	@mkdir -p ./tests/files
	@mv test_generator/test_generator.edf $(TARGET_FILE)
	@rm -rf test_generator test_generator.zip
	@echo "Done!"