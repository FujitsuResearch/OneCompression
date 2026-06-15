#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  python="${VIRTUAL_ENV}/bin/python"
else
  venv_dir="${UV_PROJECT_ENVIRONMENT:-.venv}"
  python="${repo_root}/${venv_dir}/bin/python"
fi

if [[ ! -x "$python" ]]; then
  echo "Project virtualenv not found at '${python}'." >&2
  echo "Run 'uv sync --extra dev' (or activate your venv) first." >&2
  exit 1
fi

exec "$python" "$@"
