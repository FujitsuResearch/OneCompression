#!/usr/bin/env bash
# GitLab CI cluster test orchestrator.
# Flow: git sync here → sbatch one GPU job → uv sync / pytest on compute.
# Invoked via SSH from .gitlab-ci.yml (.cluster_ssh); env vars come from printf exports.
set -euo pipefail

: "${ONECOMP_REPO:?ONECOMP_REPO is required}"
: "${CI_COMMIT_SHA:?CI_COMMIT_SHA is required}"
: "${CI_JOB_TOKEN:?CI_JOB_TOKEN is required}"
: "${CI_SERVER_HOST:?CI_SERVER_HOST is required}"
: "${CI_PROJECT_PATH:?CI_PROJECT_PATH is required}"
: "${CI_SLURM_PARTITION:?CI_SLURM_PARTITION is required}"
: "${CI_SLURM_MEM:?CI_SLURM_MEM is required}"
: "${CI_SLURM_CPUS:?CI_SLURM_CPUS is required}"
: "${CI_SLURM_TIME:?CI_SLURM_TIME is required}"
: "${CI_SLURM_GPUS:?CI_SLURM_GPUS is required}"
: "${CI_UV_VENV:?CI_UV_VENV is required}"
: "${CI_TORCH_EXTRA:?CI_TORCH_EXTRA is required}"

CI_SLURM_EXCLUDE="${CI_SLURM_EXCLUDE:-}"
SLURM_EXCLUDE_LINE=""
if [[ -n "${CI_SLURM_EXCLUDE}" ]]; then
  SLURM_EXCLUDE_LINE="#SBATCH --exclude=${CI_SLURM_EXCLUDE}"
fi

CLUSTER_MODE="${CLUSTER_MODE:-test}" # setup | test
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"
JOB_LABEL="${JOB_LABEL:-test}"
PYTEST_TARGET="${PYTEST_TARGET:-tests/}"
PYTEST_MARKERS="${PYTEST_MARKERS:-}"

echo "=== cluster test ==="
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "arch: $(uname -m)"
echo "repo: ${ONECOMP_REPO}"
echo "commit: ${CI_COMMIT_SHA}"
echo "mode: ${CLUSTER_MODE}"
echo "job label: ${JOB_LABEL}"
echo "skip uv sync: ${SKIP_UV_SYNC}"
echo "target: ${PYTEST_TARGET}"

cd "${ONECOMP_REPO}"
mkdir -p output error .cache

# Sync repo to the MR commit. Temporarily swap origin to CI_JOB_TOKEN auth; restore on exit.
# flock: parallel matrix jobs share this Lustre checkout — serialize fetch/checkout.
ORIGIN_URL="$(git remote get-url origin)"
git remote set-url origin "https://gitlab-ci-token:${CI_JOB_TOKEN}@${CI_SERVER_HOST}/${CI_PROJECT_PATH}.git"
trap 'git remote set-url origin "${ORIGIN_URL}"' EXIT

REF="${CI_COMMIT_REF_NAME:-}"
flock "${ONECOMP_REPO}/.cache/git-sync.lock" bash -c '
  set -euo pipefail
  cd "'"${ONECOMP_REPO}"'"
  if [[ -n "'"${REF}"'" ]]; then
    git fetch origin "'"${REF}"'"
  else
    git fetch origin
  fi
  git checkout "'"${CI_COMMIT_SHA}"'"
'

PYTEST_TARGET_Q=""
for target in ${PYTEST_TARGET}; do
  PYTEST_TARGET_Q+=" $(printf '%q' "${target}")"
done
PYTEST_TARGET_Q="${PYTEST_TARGET_Q# }"
PYTEST_M_ARGS=""
if [[ -n "${PYTEST_MARKERS}" ]]; then
  PYTEST_M_ARGS="-m $(printf '%q' "${PYTEST_MARKERS}")"
fi

# CI splits work: setup job runs uv sync once; test shards skip sync (SKIP_UV_SYNC=1).
if [[ "${CLUSTER_MODE}" == "setup" ]]; then
  RUN_PYTEST=0
  RUN_UV_SYNC=1
elif [[ "${SKIP_UV_SYNC}" == "1" ]]; then
  RUN_PYTEST=1
  RUN_UV_SYNC=0
else
  RUN_PYTEST=1
  RUN_UV_SYNC=1
fi

job_script="$(mktemp)"
cat >"${job_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=onecomp-${JOB_LABEL}
#SBATCH --partition=${CI_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CI_SLURM_CPUS}
#SBATCH --mem=${CI_SLURM_MEM}
#SBATCH --gres=gpu:${CI_SLURM_GPUS}
#SBATCH --time=${CI_SLURM_TIME}
#SBATCH --output=${ONECOMP_REPO}/output/tests-${JOB_LABEL}-%j.log
#SBATCH --error=${ONECOMP_REPO}/output/tests-${JOB_LABEL}-%j.log
${SLURM_EXCLUDE_LINE}

set -euo pipefail

# Write exit code to a file; login node reads this (sacct can lag behind job finish).
EXIT_FILE="${ONECOMP_REPO}/output/tests-${JOB_LABEL}-\${SLURM_JOB_ID}.exit"
_on_exit() {
  echo \$? > "\${EXIT_FILE}"
}
trap _on_exit EXIT

cd "${ONECOMP_REPO}"
mkdir -p output error

current_sha="\$(git rev-parse HEAD)"
if [[ "\${current_sha}" != "${CI_COMMIT_SHA}" ]]; then
  echo "ERROR: repo at \${current_sha}, expected ${CI_COMMIT_SHA} (login must checkout before sbatch)"
  exit 1
fi
echo "repo HEAD: \${current_sha}"

deactivate 2>/dev/null || true
unset VIRTUAL_ENV PYTHONHOME

# uv binary is arch-specific; login and GPU nodes need separate installs.
ARCH="\$(uname -m)"
UV_BIN_DIR="\${HOME}/.local/bin-\${ARCH}"
if [[ ! -x "\${UV_BIN_DIR}/uv" ]]; then
  echo "=== installing uv for \${ARCH} -> \${UV_BIN_DIR} ==="
  mkdir -p "\${UV_BIN_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="\${UV_BIN_DIR}" UV_NO_MODIFY_PATH=1 sh
fi
export PATH="\${UV_BIN_DIR}:\${PATH}"
command -v uv
uv --version

export UV_PROJECT_ENVIRONMENT="${CI_UV_VENV}"
export UV_CACHE_DIR="${ONECOMP_REPO}/.cache/uv"

if [[ "${RUN_UV_SYNC}" -eq 1 ]]; then
  echo "=== uv sync (${CI_TORCH_EXTRA}, dev, vllm, visualize) -> ${CI_UV_VENV} on \$(uname -m) ==="
  mkdir -p "${ONECOMP_REPO}/.cache"
  # flock: concurrent pipelines must not run uv sync into the same venv at once.
  flock "${ONECOMP_REPO}/.cache/uv-sync.lock" \
    uv sync --extra ${CI_TORCH_EXTRA} --extra dev --extra vllm --extra visualize --frozen
fi

echo "=== job info ==="
echo "host:        \$(hostname)"
echo "arch:        \$(uname -m)"
echo "job id:      \${SLURM_JOB_ID:-N/A}"
echo "node list:   \${SLURM_JOB_NODELIST:-N/A}"
echo "cpus:        \${SLURM_CPUS_PER_TASK:-N/A}"
uv run python - <<'PY'
import platform, torch
print(f"python arch: {platform.machine()}")
print(f"torch:       {torch.__version__}")
print(f"cuda avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0:    {torch.cuda.get_device_name(0)}")
PY
echo "================"

if [[ "${RUN_PYTEST}" -eq 1 ]]; then
  uv run pytest -v --maxfail=0 --durations=20 ${PYTEST_M_ARGS} ${PYTEST_TARGET_Q}
else
  echo "=== cluster setup passed ==="
fi
EOF
chmod +x "${job_script}"

job_id="$(sbatch --parsable "${job_script}")"
job_id="${job_id%%;*}"
echo "Submitted batch job ${job_id}"
echo "=== streaming SLURM log to GitLab CI (also saved on cluster) ==="

slurm_log="${ONECOMP_REPO}/output/tests-${JOB_LABEL}-${job_id}.log"
exit_file="${ONECOMP_REPO}/output/tests-${JOB_LABEL}-${job_id}.exit"
stream_pid=""

cleanup() {
  if [[ -n "${stream_pid}" ]]; then
    kill "${stream_pid}" 2>/dev/null || true
    wait "${stream_pid}" 2>/dev/null || true
    stream_pid=""
  fi
  rm -f "${job_script}"
}
trap cleanup EXIT

# Stream GPU job log to GitLab CI in real time (SSH stdout → Runner job log).
(
  while [[ ! -f "${slurm_log}" ]]; do
    sleep 1
  done
  tail -n 0 -F "${slurm_log}"
) &
stream_pid=$!

# Wait for SLURM job to leave the queue, then poll for .exit (written by trap on GPU node).
while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
  sleep 5
done

slurm_exit_code=""
for _ in $(seq 1 24); do
  if [[ -f "${exit_file}" ]]; then
    slurm_exit_code="$(tr -d '[:space:]' < "${exit_file}")"
    break
  fi
  sleep 5
done

sleep 2
cleanup
trap - EXIT

sacct -j "${job_id}" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true

if [[ -z "${slurm_exit_code}" ]]; then
  echo "ERROR: missing exit file: ${exit_file}"
  echo "=== log file on cluster: ${slurm_log} ==="
  exit 1
fi

echo "=== SLURM job ${job_id} finished: exit=${slurm_exit_code} (from ${exit_file}) ==="

if [[ "${slurm_exit_code}" != "0" ]]; then
  echo "=== log file on cluster: ${slurm_log} ==="
  if [[ -f "${slurm_log}" ]]; then
    echo "--- ${slurm_log} (last 80 lines) ---"
    tail -n 80 "${slurm_log}"
  fi
  exit "${slurm_exit_code}"
fi

if [[ "${CLUSTER_MODE}" == "setup" ]]; then
  echo "=== cluster setup passed ==="
else
  echo "=== cluster test passed (${JOB_LABEL}) ==="
fi
