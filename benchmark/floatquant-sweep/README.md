# FloatQuant Sweep Benchmark

This benchmark prepares the reviewer-critical FloatQuant grid:

- RTN default versus RTN static MSE sweep.
- GPTQ default versus static MSE, static WMSE, in-loop WMSE, and in-loop conditional Hessian sweep.
- Local, full-grid, and adaptive scale-candidate strategies for reviewer-facing candidate-set ablations.
- The same GPTQ modes with QEP enabled.
- NVFP4 and MXFP4 under the same calibration and evaluation settings.

Run a smoke-size grid:

```bash
python quant_benchmark.py model_path=/path/to/model \
  calibration.num_calibration_samples=8 calibration.max_length=128 \
  'evaluation.ppl_datasets=[{name:wikitext2_smoke,dataset_name:wikitext,dataset_config:wikitext-2-raw-v1,split:test,max_samples:8,max_length:512,stride:512}]'
```

Run the default paper-style grid:

```bash
python quant_benchmark.py model_path=/path/to/model output_dir=outputs/floatquant-qwen25-7b
```

Useful overrides:

```bash
# Disable QEP for faster static-vs-in-loop ablations.
python quant_benchmark.py model_path=/path/to/model 'qep.enabled=[false]'

# Run only NVFP4 and the conditional mode.
python quant_benchmark.py model_path=/path/to/model 'filters.formats=[nvfp4]' \
  'filters.modes=[gptq_inloop_conditional]'

# Repeat timing with warmups and CUDA synchronization.
python quant_benchmark.py model_path=/path/to/model 'filters.formats=[nvfp4]' \
  'filters.modes=[gptq_static_mse,gptq_inloop_wmse]' \
  timing.warmup_runs=1 timing.repeats=5 timing.randomize_order=true

# Broaden calibration robustness.
python quant_benchmark.py model_path=/path/to/model calibration.seed=1 \
  calibration.num_calibration_samples=128 calibration.max_length=2048
```

Results are written to `floatquant_sweep_results.json` and per-quantizer
`quantization_statistics_*.json` files under `output_dir`.

Mode names, paper tags, command-generator profiles, and collector order are
centralized in `experiment_specs.py`. Add new paper-facing methods there first,
then keep the entrypoint scripts as thin orchestration wrappers.

## Slurm Arrays

Generate one Hydra command per model/seed/format/mode/QEP tuple:

```bash
python generate_slurm_commands.py \
  --model /path/to/model \
  --profile smoke \
  --python /path/to/benchmark/python \
  --commands-file outputs/floatquant-sweep/commands.txt
```

Submit the command file as a Slurm array from a node where `sbatch` is available:

```bash
PARTITION=debug GPUS=1 CPUS_PER_TASK=8 MEM=64G TIME=04:00:00 \
  COMMAND_FILE=outputs/floatquant-sweep/commands.txt \
  ./submit_array.sh
```

The current login shell used during development did not expose Slurm binaries on
`PATH`. If `submit_array.sh` exits with `sbatch is not on PATH`, load the site Slurm
module or run the command from the actual Slurm login node. 

## Real-Kernel Runs

The real-kernel driver quantizes one mode, exports a vLLM-native
`compressed-tensors` checkpoint, evaluates prompt-logprob perplexity through
stock vLLM, and can remeasure decode throughput:

```bash
python generate_real_kernel_commands.py \
  --model qwen25_05b=/path/to/Qwen2.5-0.5B-Instruct \
  --profile native --include-w4a4 \
  --python /path/to/benchmark/python \
  --commands-file outputs/floatquant-real-kernel/commands.txt

cd /path/to/benchmark/floatquant-sweep
export VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_NVFP4_GEMM_BACKEND=cutlass
PARTITION=spark_1D GPUS=1 CPUS_PER_TASK=8 MEM=96G TIME=12:00:00 \
  COMMAND_FILE=outputs/floatquant-real-kernel/commands.txt \
  LOG_DIR=outputs/floatquant-real-kernel/slurm-logs \
  ./submit_array.sh
```

Each task writes a resumable record under `records/`.  Convert completed records
to the paper JSON schema with:

```bash
python collect_real_kernel_json.py \
  --records-dir outputs/floatquant-real-kernel/records \
  --out-dir outputs/floatquant-paper-json
```

For the reviewer-critical strongest-method checks on 7B and W4A4, generate
the full real-kernel grid with QEP and W4A4 companions for every NVFP4 mode:

```bash
python generate_real_kernel_commands.py \
  --model qwen25_7b=/path/to/Qwen2.5-7B \
  --profile full --include-qep --include-w4a4 --w4a4-modes all \
  --speed-batches 1,8,32 \
  --python /path/to/benchmark/python \
  --commands-file outputs/floatquant-real-kernel-full-7b/commands.txt
```

The collector preserves these records in
`real_kernel_full_qwen7b_results.json` and, when speed is measured,
`speed7b_full_method_results.json`.  The original `--include-w4a4` default
still emits only the same-weight RTN+sweep W4A4 comparison used by the native
table; `--w4a4-modes all` is the intended setting for full-method W4A4
coverage.

For downstream tasks, generate lm-eval commands over exported checkpoints:

```bash
python generate_downstream_commands.py \
  --checkpoint nvfp4_w4a4=outputs/floatquant-real-kernel/checkpoints/qwen25_7b/nvfp4_rtn_sweep_w4a4 \
  --backend vllm --tasks arc_challenge,hellaswag,winogrande,ifeval,gsm8k,mmlu_pro \
  --python /path/to/benchmark/python
```
