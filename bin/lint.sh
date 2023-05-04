#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0
black src tests
code=$?; test $code -eq 0 || exitCode=$code
isort src tests
code=$?; test $code -eq 0 || exitCode=$code
python -m flake8 src tests
code=$?; test $code -eq 0 || exitCode=$code
validate-pyproject pyproject.toml
code=$?; test $code -eq 0 || exitCode=$code
exit $exitCode
