#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0
uv run ruff check --fix
code=$?; test $code -eq 0 || exitCode=$code
uv run ruff format
code=$?; test $code -eq 0 || exitCode=$code
exit $exitCode
