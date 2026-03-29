#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [archive_path]" >&2
  exit 1
fi

RUN_DIR="$(realpath "$1")"
ARCHIVE_PATH="${2:-$RUN_DIR.tar.gz}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

PARENT_DIR="$(dirname "$RUN_DIR")"
BASE_NAME="$(basename "$RUN_DIR")"

tar -czf "$ARCHIVE_PATH" -C "$PARENT_DIR" "$BASE_NAME"
printf '[OK] archive=%s\n' "$ARCHIVE_PATH"
