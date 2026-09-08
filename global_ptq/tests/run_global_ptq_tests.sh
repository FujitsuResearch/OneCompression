#!/bin/bash
# =============================================================================
# Test runner for onecomp_globalptq Global PTQ
#
# Follows onecomp development conventions (uv + pytest).
#
# Usage:
#   bash run_global_ptq_tests.sh [--unit] [--integration] [--multigpu]
#
# Without flags, runs all test phases.
#
# Prerequisites:
#   uv sync --extra dev        # install dev dependencies
#   uv pip install -e .        # install onecomp_globalptq in editable mode
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MULTIGPU_SCRIPT="$SCRIPT_DIR/test_global_ptq_distributed_multigpu.py"

RUN_UNIT=false
RUN_INTEGRATION=false
RUN_MULTIGPU=false
RUN_ALL=true

for arg in "$@"; do
    case "$arg" in
        --unit)        RUN_UNIT=true; RUN_ALL=false ;;
        --integration) RUN_INTEGRATION=true; RUN_ALL=false ;;
        --multigpu)    RUN_MULTIGPU=true; RUN_ALL=false ;;
        *)             echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if $RUN_ALL; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_MULTIGPU=true
fi

PASSED=0
FAILED=0

run_phase() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  Phase: $name"
    echo "================================================================"
    if "$@"; then
        echo "  => $name PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  => $name FAILED"
        FAILED=$((FAILED + 1))
    fi
}

# --- Phase 1: Unit tests (CPU, fast) ---
if $RUN_UNIT; then
    run_phase "unit" \
        uv run --project "$PROJECT_DIR" \
        pytest "$SCRIPT_DIR/test_global_ptq.py" -v -m "not slow" --tb=short
fi

# --- Phase 2: Integration tests (single GPU, slow) ---
if $RUN_INTEGRATION; then
    run_phase "integration" \
        uv run --project "$PROJECT_DIR" \
        pytest "$SCRIPT_DIR/test_global_ptq.py" -v -s --log-cli-level=INFO -m "slow" --tb=short
fi

# --- Phase 3: Multi-GPU DeepSpeed tests (torchrun) ---
if $RUN_MULTIGPU; then
    for test_name in deepspeed_zero2_gptq deepspeed_zero2_ntp deepspeed_zero2_via_runner deepspeed_zero2_intweight; do
        run_phase "multigpu:$test_name" \
            uv run --project "$PROJECT_DIR" \
            torchrun --nproc_per_node=2 "$MULTIGPU_SCRIPT" --test "$test_name"
    done
fi

# --- Summary ---
echo ""
echo "================================================================"
TOTAL=$((PASSED + FAILED))
echo "  Summary: $PASSED/$TOTAL passed"
if [ "$FAILED" -gt 0 ]; then
    echo "  $FAILED FAILED"
    exit 1
fi
echo "  All tests passed!"
echo "================================================================"
