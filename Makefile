TARGET_FILE := ./tests/files/test_file.edf

.PHONY: setup run test download_test_file

setup:
	@poetry install --with dev,test

test:
	@poetry run pytest

download_test_file:
	@echo "Downloading and processing test file..."
	@wget -q "https://www.teuniz.net/edf_bdf_testfiles/test_generator.zip" -O test_generator.zip 
	@unzip -q test_generator.zip -d test_generator
	@rm test_generator.zip

	@mkdir -p ./tests/files
	@mv test_


