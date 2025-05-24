#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0
uv run ruff check
code=$?; test $code -eq 0 || exitCode=$code
uv run ruff format --check
code=$?; test $code -eq 0 || exitCode=$code
uv run validate-pyproject pyproject.toml
code=$?; test $code -eq 0 || exitCode=$code
exit $exitCode
