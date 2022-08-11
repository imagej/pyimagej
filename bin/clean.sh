#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

find . -name __pycache__ -type d | while read d
  do rm -rf "$d"
done
rm -rf .pytest_cache build dist doc/rtd src/*.egg-info
