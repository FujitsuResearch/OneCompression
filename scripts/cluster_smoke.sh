#!/usr/bin/env bash
# Smoke checks on the cluster login node.

set -euo pipefail

echo "=== cluster smoke ==="
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "date: $(date -Is)"

echo "=== cluster smoke passed ==="
