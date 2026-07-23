"""Quantize, export, and evaluate FloatQuant checkpoints with real vLLM kernels.

This driver is intentionally one-mode-per-process so it can be used as a
Slurm-array command.  It writes a JSON record even on failure,
making long campaigns resumable and auditable.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from experiment_specs import REAL_KERNEL_MODE_SPECS as MODE_SPECS

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner, setup_logger
from onecomp.quantizer.floatquant import FloatQuant
from onecomp.quantizer.floatquant.vllm_export import collect_input_global_scales


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # NFS can still surface mkdir races even with exist_ok=True.
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
    tmp.replace(path)


def _checkpoint_size(path_or_id: str) -> dict:
    if not os.path.isdir(path_or_id):
        return {}
    files = glob.glob(os.path.join(path_or_id, "*.safetensors"))
    if not files:
        files = glob.glob(os.path.join(path_or_id, "*.bin"))
    total = sum(os.path.getsize(path) for path in files)
    return {
        "weight_bytes": total,
        "weight_mb": round(total / 1e6, 3),
        "weight_gb": round(total / 1e9, 3),
    }


def _shutdown_vllm(llm) -> None:
    """Best-effort vLLM shutdown across vLLM versions."""
    for attr in ("shutdown",):
        method = getattr(llm, attr, None)
        if callable(method):
            method()
            break
    else:
        engine = getattr(llm, "llm_engine", None)
        method = getattr(engine, "shutdown", None)
        if callable(method):
            method()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_texts(
    dataset_name: str, dataset_config: str | None, split: str, limit: int
) -> list[str]:
    if dataset_name == "allenai/c4":
        dataset = load_dataset(dataset_name, data_files=dataset_config, split=split)
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    texts = []
    for item in dataset:
        text = item.get("text", "")
        if text and text.strip():
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _vllm_perplexity(
    checkpoint: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_samples: int | None,
    max_length: int,
    batch_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=checkpoint,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_length * 2,
        trust_remote_code=True,
    )
    try:
        if dataset_name == "allenai/c4":
            dataset = load_dataset(dataset_name, data_files=dataset_config, split=split)
        elif dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        text = "\n\n".join(t for t in dataset["text"] if t)
        tokenizer = llm.get_tokenizer()
        token_ids = tokenizer(text).input_ids
        windows = [token_ids[i : i + max_length] for i in range(0, len(token_ids), max_length)]
        windows = [window for window in windows if len(window) >= 2]
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        batch_size = max(1, batch_size)
        started = time.perf_counter()
        outputs = []
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            print(
                f"[ppl] scoring windows {offset + 1}-{offset + len(batch)} / {len(windows)}",
                flush=True,
            )
            outputs.extend(
                llm.generate(
                    [TokensPrompt(prompt_token_ids=window) for window in batch],
                    params,
                )
            )
        elapsed = time.perf_counter() - started
        nll_sum = 0.0
        n_tokens = 0
        for window, output in zip(windows, outputs):
            for idx, entry in enumerate(output.prompt_logprobs or []):
                if entry is None:
                    continue
                token_logprob = entry.get(window[idx])
                if token_logprob is not None:
                    nll_sum -= token_logprob.logprob
                    n_tokens += 1
        ppl = math.exp(nll_sum / max(n_tokens, 1))
        return {
            "wikitext2_ppl" if dataset_name == "wikitext" else "ppl": ppl,
            "nll": nll_sum / max(n_tokens, 1),
            "num_scored_tokens": n_tokens,
            "num_windows": len(windows),
            "elapsed_seconds": elapsed,
        }
    finally:
        _shutdown_vllm(llm)


def _vllm_speed(
    checkpoint: str,
    batches: list[int],
    output_tokens: int,
    repeats: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=checkpoint,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        trust_remote_code=True,
    )
    try:
        params = SamplingParams(temperature=0.0, max_tokens=output_tokens, ignore_eos=True)
        prompt = (
            "Post-training quantization compresses large language models while preserving "
            "quality under deployment constraints."
        )
        summary = {}
        for batch in batches:
            prompts = [prompt for _ in range(batch)]
            llm.generate(prompts, params)
            values = []
            for _ in range(repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = llm.generate(prompts, params)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = sum(len(output.outputs[0].token_ids) for output in outputs)
                values.append(generated / elapsed)
            values_sorted = sorted(values)
            median = values_sorted[len(values_sorted) // 2]
            summary[f"bs{batch}"] = median
            summary[f"bs{batch}_values"] = values
        return summary
    finally:
        _shutdown_vllm(llm)


def _make_quantizer(mode: str, spec: dict, args: argparse.Namespace) -> FloatQuant:
    suffix = "_w4a4" if args.w4a4 else ""
    return FloatQuant(
        fmt=spec["fmt"],
        use_hessian=bool(spec["use_hessian"]),
        scale_timing=str(spec["scale_timing"]),
        scale_objective=str(spec["scale_objective"]),
        scale_candidate_strategy=str(spec["candidate"]),
        block_size=None,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        num_layers=args.num_layers,
        calc_quant_error=True,
        name=f"FloatQuant_{mode}{suffix}",
    )


def _quantize_and_export(args: argparse.Namespace, spec: dict, record: dict) -> str:
    model_config = ModelConfig(path=args.model_path, device=args.model_device, dtype=args.dtype)
    calibration = CalibrationConfig(
        calibration_dataset=args.calibration_dataset,
        max_length=args.calibration_max_length,
        num_calibration_samples=args.calibration_samples,
        strategy=args.calibration_strategy,
        seed=args.seed,
        batch_size=None if args.qep else args.calibration_batch_size,
    )
    quantizer = _make_quantizer(args.mode, spec, args)
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration,
        qep=args.qep,
        qep_config=(
            QEPConfig(
                general=args.qep_general,
                percdamp=args.qep_percdamp,
                perccorr=args.qep_perccorr,
                device=args.qep_device,
                exclude_layer_keywords=args.qep_exclude,
            )
            if args.qep
            else None
        ),
    )

    started = time.perf_counter()
    runner.run()
    record["quantization_seconds"] = time.perf_counter() - started
    stats_suffix = _run_suffix(args)
    stats_path = (
        Path(args.output_dir) / "stats" / f"{args.model_tag}_{args.mode}{stats_suffix}.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save_quantization_statistics(str(stats_path), quantizer=quantizer)
    record["quantization_statistics"] = str(stats_path)

    input_scales = None
    if args.w4a4:
        if spec["fmt"] != "nvfp4":
            raise ValueError("--w4a4 is only valid for nvfp4 modes.")
        texts = _load_texts(
            args.activation_dataset,
            args.activation_dataset_config,
            args.activation_split,
            args.activation_samples,
        )
        model = model_config.load_model(device_map=args.model_device)
        tokenizer = model_config.load_tokenizer()
        input_scales = collect_input_global_scales(
            model,
            tokenizer,
            quantizer.results.keys(),
            texts,
            device=args.model_device,
            max_length=args.activation_max_length,
            percentile=args.activation_percentile,
            scale_multiplier=args.activation_scale_multiplier,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint = (
        Path(args.output_dir) / "checkpoints" / args.model_tag / f"{args.mode}{stats_suffix}"
    )
    runner.save_vllm_native_model(str(checkpoint), input_global_scales=input_scales)
    del runner
    del quantizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(checkpoint)


def _run_suffix(args: argparse.Namespace) -> str:
    suffix = ""
    if args.qep:
        suffix += "_qep"
    if args.w4a4:
        suffix += "_w4a4"
    return suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_SPECS))
    parser.add_argument("--output-dir", default="outputs/floatquant-real-kernel")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qep", action="store_true")
    parser.add_argument("--qep-general", action="store_true")
    parser.add_argument("--qep-percdamp", type=float, default=0.01)
    parser.add_argument("--qep-perccorr", type=float, default=0.5)
    parser.add_argument("--qep-device", default="cuda:0")
    parser.add_argument("--qep-exclude", nargs="*", default=["mlp.down_proj"])
    parser.add_argument("--w4a4", action="store_true")
    parser.add_argument("--calibration-dataset", default="wikitext2")
    parser.add_argument("--calibration-strategy", default="drop_rand")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-max-length", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--activation-dataset", default="wikitext")
    parser.add_argument("--activation-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--activation-split", default="train")
    parser.add_argument("--activation-samples", type=int, default=64)
    parser.add_argument("--activation-max-length", type=int, default=512)
    parser.add_argument("--activation-percentile", type=float, default=100.0)
    parser.add_argument("--activation-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--ppl-dataset", default="wikitext")
    parser.add_argument("--ppl-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--ppl-max-samples", type=int, default=None)
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=16)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--speed-batches", default="")
    parser.add_argument("--speed-output-tokens", type=int, default=256)
    parser.add_argument("--speed-repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--eval-checkpoint",
        default="",
        help="Evaluate an already exported checkpoint instead of quantizing/exporting first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logger()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "records" / f"{args.model_tag}_{args.mode}{_run_suffix(args)}.json"
    if args.skip_existing and record_path.exists():
        with record_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("status") == "OK":
            print(f"Skipping existing OK record: {record_path}", flush=True)
            return 0

    spec = MODE_SPECS[args.mode]
    existing_record = None
    if args.eval_checkpoint and record_path.exists():
        with record_path.open(encoding="utf-8") as f:
            existing_record = json.load(f)

    record = (
        dict(existing_record)
        if existing_record
        else {
            "status": "STARTED",
            "model_path": args.model_path,
            "model_tag": args.model_tag,
            "mode": args.mode,
            "paper_tag": spec["paper_tag"],
            "qep": bool(args.qep),
            "w4a4": bool(args.w4a4),
            "argv": sys.argv,
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    record.update(
        {
            "status": "STARTED",
            "model_path": args.model_path,
            "model_tag": args.model_tag,
            "mode": args.mode,
            "paper_tag": spec["paper_tag"],
            "qep": bool(args.qep),
            "w4a4": bool(args.w4a4),
            "argv": sys.argv,
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    record.pop("error", None)
    record.pop("error_type", None)
    _write_json(record_path, record)

    try:
        if args.eval_checkpoint:
            checkpoint = args.eval_checkpoint
        elif spec["fmt"] is None:
            checkpoint = args.model_path
        else:
            checkpoint = _quantize_and_export(args, spec, record)
        record["checkpoint"] = checkpoint
        record.update(_checkpoint_size(checkpoint))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not args.skip_ppl:
            record["ppl"] = _vllm_perplexity(
                checkpoint,
                args.ppl_dataset,
                args.ppl_dataset_config,
                args.ppl_split,
                args.ppl_max_samples,
                args.ppl_max_length,
                args.ppl_batch_size,
                args.gpu_memory_utilization,
                args.enforce_eager,
            )
        if args.speed_batches:
            batches = [int(item) for item in args.speed_batches.split(",") if item]
            record["speed"] = _vllm_speed(
                checkpoint,
                batches,
                args.speed_output_tokens,
                args.speed_repeats,
                args.gpu_memory_utilization,
                args.enforce_eager,
            )
        record["status"] = "OK"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        record["status"] = "FAILED"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        _write_json(record_path, record)
        raise
    finally:
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
