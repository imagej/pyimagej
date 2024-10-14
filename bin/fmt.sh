#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

exitCode=0
ruff check --fix
code=$?; test $code -eq 0 || exitCode=$code
ruff format
code=$?; test $code -eq 0 || exitCode=$code
exit $exitCode
