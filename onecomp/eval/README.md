# `onecomp.eval`

Hydra-based evaluation pipeline: one vLLM server, subprocess evaluators (`mt_bench`, `throughput`), aggregated `summary.json` / `summary.csv`.

**Documentation (single source of truth):**

- [Evaluation user guide](https://FujitsuResearch.github.io/OneCompression/user-guide/evaluation/) — install, CLI, MT-Bench data (EN/JP), judge API keys, answer post-processing, Python API
- [Evaluation API](https://FujitsuResearch.github.io/OneCompression/api/eval/) — `EvalConfig`, `TaskResult`, mkdocstrings reference

## Quick start

Requires a **source checkout** (MT-Bench data and download scripts are not on PyPI). See the top-level README [for developers (pip)](../../README.md#for-developers-pip).

```bash
git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression

pip install -e ".[eval,vllm,cu130]"

bash onecomp/eval/scripts/download_mt_bench_data_en.sh
export OPENAI_API_KEY="sk-..."
onecomp-eval model.path=/path/to/model
```

Repo-local paths: [`conf/eval_config.yaml`](./conf/eval_config.yaml) · [`schema.py`](./schema.py) · SLURM helper [`takane_scripts/sbatch_onecomp_eval.sh`](../../takane_scripts/sbatch_onecomp_eval.sh)

## Evaluators

| Name | Description |
|------|-------------|
| `mt_bench` | MT-Bench (default: English; generation + GPT-4 judge + radar chart) |
| `throughput` | Chat Completions streaming benchmark (TTFT / decode tok/s) |

For Japanese MT-Bench, data download, locale switching, and extending the harness, see the [user guide](https://FujitsuResearch.github.io/OneCompression/user-guide/evaluation/).
