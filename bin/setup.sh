#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

mamba env create -f dev-environment.yml
