#!/usr/bin/env bash
#
# download_mt_bench_data_en.sh — Download English MT-Bench data from FastChat upstream
#
# Usage:
#   bash onecomp/eval/scripts/download_mt_bench_data_en.sh [output_dir]
#
# Examples:
#   bash onecomp/eval/scripts/download_mt_bench_data_en.sh
#   bash onecomp/eval/scripts/download_mt_bench_data_en.sh /path/to/custom_dir
#   MT_BENCH_DATA_DIR=/path/to/custom_dir bash onecomp/eval/scripts/download_mt_bench_data_en.sh
#
# source:
#   - lm-sys/FastChat (main): question.jsonl, reference_answer/gpt-4.jsonl, judge_prompts.jsonl

set -euo pipefail

FASTCHAT_EN_BASE="https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench"
FASTCHAT_MAIN_BASE="https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve via onecomp/eval (always in repo); data/ is created below by mkdir -p.
DEFAULT_DATA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/data/mt_bench_en"
DATA_DIR="${1:-${MT_BENCH_DATA_DIR:-${DEFAULT_DATA_DIR}}}"

download() {
    local url="$1"
    local dest="$2"
    echo "  -> ${dest}"
    mkdir -p "$(dirname "${dest}")"
    curl -fsSL -o "${dest}" "${url}"
}

echo "English MT-Bench data -> ${DATA_DIR}"
mkdir -p "${DATA_DIR}/reference_answer"

echo "[1/3] question.jsonl (80 questions)"
download \
    "${FASTCHAT_EN_BASE}/question.jsonl" \
    "${DATA_DIR}/question.jsonl"

echo "[2/3] reference_answer/gpt-4.jsonl"
download \
    "${FASTCHAT_EN_BASE}/reference_answer/gpt-4.jsonl" \
    "${DATA_DIR}/reference_answer/gpt-4.jsonl"

echo "[3/3] judge_prompts.jsonl"
download \
    "${FASTCHAT_MAIN_BASE}/judge_prompts.jsonl" \
    "${DATA_DIR}/judge_prompts.jsonl"

for f in \
    "${DATA_DIR}/question.jsonl" \
    "${DATA_DIR}/judge_prompts.jsonl" \
    "${DATA_DIR}/reference_answer/gpt-4.jsonl"
do
    if [[ ! -s "${f}" ]]; then
        echo "ERROR: download failed or empty: ${f}" >&2
        exit 1
    fi
done

n_questions=$(grep -c . "${DATA_DIR}/question.jsonl" || true)
echo ""
echo "Done. (${n_questions} questions)"
echo ""
echo "Next steps:"
echo "  export OPENAI_API_KEY=\"sk-...\""
if [[ "${DATA_DIR}" == "${DEFAULT_DATA_DIR}" ]]; then
    echo "  onecomp-eval model.path=/path/to/model"
else
    echo "  onecomp-eval model.path=/path/to/model evals.mt_bench.data_dir=\"${DATA_DIR}\""
    echo "  # or export MT_BENCH_DATA_DIR=\"${DATA_DIR}\""
fi
echo ""
echo "Japanese MT-Bench: bash onecomp/eval/scripts/download_mt_bench_data_jp.sh"
