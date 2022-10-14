#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

black src tests
isort src tests
python -m flake8 src tests
validate-pyproject pyproject.toml
