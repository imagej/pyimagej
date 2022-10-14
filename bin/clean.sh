#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

find . -name __pycache__ -type d | while read d
  do rm -rfv "$d"
done
rm -rfv .pytest_cache build dist doc/_build doc/rtd src/*.egg-info
