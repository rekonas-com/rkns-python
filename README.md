# RKNS data format for ExG data
This package contains a Python implementation of the RKNS data format for ExG data.

## Development
1. `poetry install`
2. `poetry run python example_usage.py`

## Using as a dependency
1. `poetry build`
2. Add the path `<path_to_repository>/dist/rkns-0.1.0.tar.gz` to the toml file of your project.
Example: `rkns = { path = "<path_to_repository>/dist/rkns-0.1.0.tar.gz" }`