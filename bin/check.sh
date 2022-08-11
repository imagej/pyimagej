#!/bin/sh

case "$CONDA_PREFIX" in
  */pyimagej-dev)
    ;;
  *)
    echo "Please run 'make setup' and then 'mamba activate pyimagej-dev' first."
    exit 1
    ;;
esac
