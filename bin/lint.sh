#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0
uv run --group dev ruff check --fix
code=$?; test $code -eq 0 || exitCode=$code
uv run --group dev ruff format
code=$?; test $code -eq 0 || exitCode=$code
uv run --group dev validate-pyproject pyproject.toml
code=$?; test $code -eq 0 || exitCode=$code
exit $exitCode
