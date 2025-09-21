#!/bin/sh

if ! command -v uv >/dev/null 2>&1; then
  echo "Please install uv (https://docs.astral.sh/uv/getting-started/installation/)."
  exit 1
fi
