"""
Multi-GPU DeepSpeed integration test for onecomp_globalptq GlobalPTQDistributed.

This is a standalone script (NOT pytest) because DeepSpeed requires
launch via ``torchrun``.  Each process executes the full script.

Each invocation runs a single test case specified by ``--test``.
This avoids process-group conflicts between DeepSpeed sessions.

Test categories:

  Functional (require 2+ GPUs):
    uv run torchrun --nproc_per_node=2 <this_script> --test deepspeed_zero2_gptq
    uv run torchrun --nproc_per_node=2 <this_script> --test deepspeed_zero2_ntp
    uv run torchrun --nproc_per_node=2 <this_script> --test deepspeed_zero2_via_runner
    uv run torchrun --nproc_per_node=2 <this_script> --test deepspeed_zero2_intweight

  Production-sized model (8B, require 2+ GPUs, downloads from HF):
    uv run torchrun --nproc_per_node=2 <this_script> --test large_model_distributed

  Speed-up benchmark (outputs timing JSON):
    uv run torchrun --nproc_per_node=N <this_script> --test speedup_timed

  Single-vs-multi GPU consistency (two-phase):
    uv run torchrun --nproc_per_node=1 <this_script> --test consistency_baseline
    uv run torchrun --nproc_per_node=2 <this_script> --test consistency_verify

Exit code 0 = test passed, non-zero = failure.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import torch


MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DS_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "ds_zero2.json",
)

LARGE_MODEL_ID_DEFAULT = "meta-llama/Llama-3.1-8B"

CONSISTENCY_DIR = os.path.join(
    os.path.dirname(__file__), "logs", "consistency"
)


def log(msg):
    rank = int(os.environ.get("RANK", 0))
    print(f"[rank {rank}] {msg}", flush=True)


def log_training_loss(gptq_instance, assert_decrease=False, check_checkpoint_dir=None):
    """Extract and log train/eval loss from a GlobalPTQDistributed run.

    Args:
        gptq_instance: GlobalPTQDistributed instance after .run().
        assert_decrease: If True, assert that eval loss decreased over training.
        check_checkpoint_dir: If set, assert that checkpoint subdirs exist there.
    """
    train_loss = getattr(gptq_instance, "_last_train_loss", None)
    log_history = getattr(gptq_instance, "_last_log_history", [])

    if train_loss is not None:
        log(f"  train_loss = {train_loss:.6f}")

    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    if train_losses:
        log(f"  step losses = {[round(l, 6) for l in train_losses]}")
    if eval_losses:
        log(f"  eval losses = {[round(l, 6) for l in eval_losses]}")

    if assert_decrease:
        losses_for_check = eval_losses if len(eval_losses) >= 2 else train_losses
        if len(losses_for_check) >= 4:
            mid = len(losses_for_check) // 2
            first_half_avg = sum(losses_for_check[:mid]) / mid
            second_half_avg = sum(losses_for_check[mid:]) / (len(losses_for_check) - mid)
            log(f"  loss trend: first_half_avg={first_half_avg:.4f}, "
                f"second_half_avg={second_half_avg:.4f} "
                f"(delta={second_half_avg - first_half_avg:+.4f})")
            assert second_half_avg < first_half_avg, (
                f"Loss did not decrease: first_half_avg={first_half_avg:.6f} "
                f"> second_half_avg={second_half_avg:.6f}"
            )
            log("  loss decrease confirmed")

    if check_checkpoint_dir is not None:
        ckpt_dirs = sorted(
            d for d in os.listdir(check_checkpoint_dir)
            if d.startswith("checkpoint-")
        ) if os.path.isdir(check_checkpoint_dir) else []
        log(f"  checkpoints: {ckpt_dirs} in {check_checkpoint_dir}")
        assert len(ckpt_dirs) > 0, (
            f"No checkpoint-* dirs found in {check_checkpoint_dir}"
        )
        log("  checkpoint saving confirmed")


def quantize_model():
    """Quantize TinyLlama with GPTQ (deterministic, each rank does it)."""
    from onecomp import GPTQ, ModelConfig, Runner, CalibrationConfig, setup_logger
    setup_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    model_config = ModelConfig(model_id=MODEL_ID, device=device)
    quantizer = GPTQ(wbits=4, groupsize=128)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=8),
    )
    runner.run()

    model, _tokenizer = runner.create_quantized_model(
        pack_weights=False, use_gemlite=False,
    )
    return model, model_config


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_deepspeed_zero2_gptq():
    """GlobalPTQDistributed with DeepSpeed ZeRO-2 on multiple GPUs."""
    log("Quantizing model...")
    model, model_config = quantize_model()

    from onecomp_globalptq import GlobalPTQDistributed

    ds_config = os.path.abspath(DS_CONFIG)
    log(f"DeepSpeed config: {ds_config}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), "logs", "ckpt_zero2_gptq")

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=32,
        max_length=128,
        eval_interval=1,
        deepspeed_config=ds_config,
        output_dir=ckpt_dir,
        save_strategy="epoch",
    )

    log("Running GlobalPTQDistributed...")
    gptq.run(model, model_config)
    log_training_loss(gptq, assert_decrease=True, check_checkpoint_dir=ckpt_dir)

    devices = {str(p.device) for p in model.parameters()}
    assert devices == {"cpu"}, f"Model should be on CPU, got {devices}"
    assert not model.training, "Model should be in eval mode"

    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    gptq_count = sum(
        1 for _n, m in model.named_modules()
        if isinstance(m, GPTQLinear)
    )
    assert gptq_count > 0, "GPTQ layers should still exist"

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


def test_deepspeed_zero2_ntp():
    """DeepSpeed ZeRO-2 with combined KL + NTP loss."""
    log("Quantizing model...")
    model, model_config = quantize_model()

    from onecomp_globalptq import GlobalPTQDistributed

    ds_config = os.path.abspath(DS_CONFIG)

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=32,
        max_length=128,
        eval_interval=1,
        w_distill=1.0,
        w_ntp=0.5,
        deepspeed_config=ds_config,
    )

    log("Running GlobalPTQDistributed (KL + NTP)...")
    gptq.run(model, model_config)
    log_training_loss(gptq, assert_decrease=True)

    devices = {str(p.device) for p in model.parameters()}
    assert devices == {"cpu"}, f"Model should be on CPU, got {devices}"
    assert not model.training, "Model should be in eval mode"

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


def test_deepspeed_zero2_via_runner():
    """Runner with GlobalPTQDistributed + DeepSpeed end-to-end."""
    from onecomp import GPTQ, ModelConfig, Runner, CalibrationConfig, setup_logger
    from onecomp_globalptq import GlobalPTQDistributed
    setup_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    model_config = ModelConfig(model_id=MODEL_ID, device=device)
    quantizer = GPTQ(wbits=4, groupsize=128)

    ds_config = os.path.abspath(DS_CONFIG)

    post = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=32,
        max_length=128,
        eval_interval=1,
        deepspeed_config=ds_config,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=8),
        post_processes=[post],
    )
    runner.run()
    log_training_loss(post, assert_decrease=True)

    assert runner.quantized_model is not None, "quantized_model should be set"

    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    gptq_count = sum(
        1 for _n, m in runner.quantized_model.named_modules()
        if isinstance(m, GPTQLinear)
    )
    assert gptq_count > 0, "GPTQ layers should still be present"

    del runner
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


def test_deepspeed_zero2_intweight():
    """GlobalPTQDistributed with integer weight optimization."""
    log("Quantizing model...")
    model, model_config = quantize_model()

    from onecomp_globalptq import GlobalPTQDistributed

    ds_config = os.path.abspath(DS_CONFIG)

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=32,
        max_length=128,
        eval_interval=1,
        gptq_optimize_intweight=True,
        deepspeed_config=ds_config,
    )

    log("Running GlobalPTQDistributed (intweight optimization)...")
    gptq.run(model, model_config)
    log_training_loss(gptq, assert_decrease=True)

    devices = {str(p.device) for p in model.parameters()}
    assert devices == {"cpu"}, f"Model should be on CPU, got {devices}"

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


# ---------------------------------------------------------------------------
# Test: Production-sized model (8B) on multiple GPUs
# ---------------------------------------------------------------------------


def _quantize_large_model():
    """Quantize a production-sized model (8B) with GPTQ on each rank."""
    from onecomp import GPTQ, ModelConfig, Runner, CalibrationConfig, setup_logger
    setup_logger()

    large_model_id = os.environ.get("LARGE_MODEL_ID", LARGE_MODEL_ID_DEFAULT)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    log(f"Quantizing {large_model_id} on {device}...")
    model_config = ModelConfig(model_id=large_model_id, device=device)
    runner = Runner(
        model_config=model_config,
        quantizer=GPTQ(wbits=4, groupsize=128),
        calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=8),
    )
    runner.run()

    model, _tokenizer = runner.create_quantized_model(
        pack_weights=False, use_gemlite=False,
    )
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Quantized model: {param_count / 1e9:.2f}B parameters")
    return model, model_config


def test_large_model_distributed():
    """Verify GlobalPTQDistributed works on a production-sized model (8B).

    Quantizes Llama-3-8B (or model set via LARGE_MODEL_ID env var) with
    GPTQ on each rank, then runs KL distillation with DeepSpeed ZeRO-2.
    This confirms correct behaviour at realistic model scale, beyond TinyLlama.

    Env vars:
        LARGE_MODEL_ID — model path or HF ID (default: meta-llama/Llama-3.1-8B)
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ds_config = os.path.abspath(DS_CONFIG)

    model, model_config = _quantize_large_model()

    from onecomp_globalptq import GlobalPTQDistributed

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=16,
        max_length=128,
        eval_interval=1,
        use_gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        deepspeed_config=ds_config,
        bf16=True,
    )

    log(f"Running GlobalPTQDistributed on {world_size} GPUs...")
    start = time.time()
    gptq.run(model, model_config)
    elapsed = time.time() - start
    log(f"GlobalPTQ completed in {elapsed:.1f}s")
    log_training_loss(gptq, assert_decrease=True)

    devices = {str(p.device) for p in model.parameters()}
    assert devices == {"cpu"}, f"Model should be on CPU, got {devices}"
    assert not model.training, "Model should be in eval mode"

    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    gptq_count = sum(
        1 for _n, m in model.named_modules()
        if isinstance(m, GPTQLinear)
    )
    assert gptq_count > 0, "GPTQ layers should still exist"

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


def test_large_model_distributed_ntp():
    """Same as large_model_distributed but with KL + NTP combined loss.

    Verifies that the heavier combined loss (teacher logits + next-token
    prediction) works correctly at 8B scale with DeepSpeed ZeRO-2.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ds_config = os.path.abspath(DS_CONFIG)

    model, model_config = _quantize_large_model()

    from onecomp_globalptq import GlobalPTQDistributed

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=16,
        max_length=128,
        eval_interval=1,
        w_distill=1.0,
        w_ntp=0.5,
        use_gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        deepspeed_config=ds_config,
        bf16=True,
    )

    log(f"Running GlobalPTQDistributed (KL + NTP) on {world_size} GPUs...")
    start = time.time()
    gptq.run(model, model_config)
    elapsed = time.time() - start
    log(f"GlobalPTQ (KL + NTP) completed in {elapsed:.1f}s")
    log_training_loss(gptq, assert_decrease=True)

    devices = {str(p.device) for p in model.parameters()}
    assert devices == {"cpu"}, f"Model should be on CPU, got {devices}"
    assert not model.training, "Model should be in eval mode"

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


# ---------------------------------------------------------------------------
# Test: Speed-up with more GPUs (benchmark, outputs timing JSON)
# ---------------------------------------------------------------------------


def _param_checksum(model) -> str:
    """Compute a deterministic hash of all model parameters."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def test_speedup_timed():
    """Benchmark GlobalPTQDistributed wall-clock time across GPU counts.

    Outputs a JSON file to tests/logs/ that can be compared across runs
    with different --nproc_per_node values to verify speed-up.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.manual_seed(42)
    log("Quantizing model for speed-up benchmark...")
    model, model_config = quantize_model()

    from onecomp_globalptq import GlobalPTQDistributed

    ds_config = os.path.abspath(DS_CONFIG)

    gptq = GlobalPTQDistributed(
        epochs=5,
        gptq_lr=5e-6,
        num_calibration_samples=32,
        max_length=128,
        eval_interval=1,
        deepspeed_config=ds_config,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log("Starting timed run...")
    t_start = time.time()
    gptq.run(model, model_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t_start
    log_training_loss(gptq, assert_decrease=True)

    if local_rank == 0:
        os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
        result = {
            "test": "speedup_timed",
            "world_size": world_size,
            "elapsed_s": round(elapsed, 2),
            "model": MODEL_ID,
            "epochs": 5,
            "param_checksum": _param_checksum(model),
        }
        out_path = os.path.join(
            os.path.dirname(__file__), "logs",
            f"speedup_{world_size}gpu.json",
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"Timing: {elapsed:.2f}s with {world_size} GPUs -> {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("PASSED")


# ---------------------------------------------------------------------------
# Test: Single-GPU vs Multi-GPU consistency
# ---------------------------------------------------------------------------

_CONSISTENCY_COMMON_CONFIG = dict(
    epochs=5,
    gptq_lr=5e-6,
    num_calibration_samples=32,
    max_length=128,
    eval_interval=1,
)


def _run_consistency_training(world_size):
    """Run GlobalPTQDistributed and return (final loss proxy, param checksum)."""
    torch.manual_seed(42)
    model, model_config = quantize_model()

    from onecomp_globalptq import GlobalPTQDistributed

    ds_config = os.path.abspath(DS_CONFIG)

    effective_batch = 2
    grad_accum = max(1, effective_batch // world_size)

    gptq = GlobalPTQDistributed(
        **_CONSISTENCY_COMMON_CONFIG,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        deepspeed_config=ds_config if world_size > 1 else None,
    )

    gptq.run(model, model_config)
    log_training_loss(gptq)

    checksum = _param_checksum(model)

    from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    test_text = "The meaning of life is"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=32,
                       truncation=True).to(device)
    model.to(device).eval()
    with torch.no_grad():
        student_logits = model(**inputs).logits

    teacher = AutoModelForCausalLM.from_pretrained(
        model_config.model_id, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits

    eval_loss = compute_kl_loss(teacher_logits.float(), student_logits.float()).item()

    del model, teacher
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss, checksum


def test_consistency_baseline():
    """Phase 1: Run single-GPU baseline and save results for later comparison.

    This test is designed to be run with --nproc_per_node=1.
    """
    log("Running consistency baseline (single GPU)...")
    eval_loss, checksum = _run_consistency_training(world_size=1)

    os.makedirs(CONSISTENCY_DIR, exist_ok=True)
    result = {
        "eval_loss": eval_loss,
        "param_checksum": checksum,
        "world_size": 1,
    }
    out_path = os.path.join(CONSISTENCY_DIR, "baseline_1gpu.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Baseline saved: loss={eval_loss:.6f}, checksum={checksum} -> {out_path}")
    log("PASSED")


def test_consistency_verify():
    """Phase 2: Run multi-GPU training and compare against single-GPU baseline.

    This test is designed to be run with --nproc_per_node=2+ AFTER
    consistency_baseline has completed.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    log(f"Running consistency verification ({world_size} GPUs)...")
    eval_loss, checksum = _run_consistency_training(world_size=world_size)

    if local_rank != 0:
        log("PASSED (non-rank-0, skipping comparison)")
        return

    baseline_path = os.path.join(CONSISTENCY_DIR, "baseline_1gpu.json")
    if not os.path.exists(baseline_path):
        log(f"WARNING: Baseline not found at {baseline_path}. "
            "Run consistency_baseline first. Skipping comparison.")
        log("PASSED (no baseline to compare)")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    base_loss = baseline["eval_loss"]
    base_checksum = baseline["param_checksum"]

    log(f"  Baseline (1 GPU): loss={base_loss:.6f}, checksum={base_checksum}")
    log(f"  Current ({world_size} GPUs): loss={eval_loss:.6f}, checksum={checksum}")

    os.makedirs(CONSISTENCY_DIR, exist_ok=True)
    result = {
        "eval_loss": eval_loss,
        "param_checksum": checksum,
        "world_size": world_size,
        "baseline_loss": base_loss,
        "baseline_checksum": base_checksum,
        "loss_diff": abs(eval_loss - base_loss),
        "loss_ratio": eval_loss / base_loss if base_loss != 0 else float("inf"),
        "checksum_match": checksum == base_checksum,
    }
    out_path = os.path.join(CONSISTENCY_DIR, f"verify_{world_size}gpu.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Comparison saved -> {out_path}")

    loss_diff = abs(eval_loss - base_loss)
    rel_tolerance = 0.3
    rel_diff = loss_diff / max(abs(base_loss), 1e-6)
    assert rel_diff < rel_tolerance, (
        f"Loss diverged too much: single-GPU={base_loss:.6f}, "
        f"{world_size}-GPU={eval_loss:.6f}, "
        f"rel_diff={rel_diff:.4f} (tolerance={rel_tolerance})"
    )
    log(f"Loss difference: abs={loss_diff:.6f}, "
        f"rel={rel_diff:.4f} (within {rel_tolerance})")

    if checksum == base_checksum:
        log("Parameter checksums MATCH (bit-exact)")
    else:
        log("Parameter checksums differ (expected with different parallelism)")

    log("PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REQUIRES_MULTI_GPU = {
    "deepspeed_zero2_gptq",
    "deepspeed_zero2_ntp",
    "deepspeed_zero2_via_runner",
    "deepspeed_zero2_intweight",
    "large_model_distributed",
    "speedup_timed",
    "consistency_verify",
}

SINGLE_GPU_OK = {
    "consistency_baseline",
}

TEST_MAP = {
    "deepspeed_zero2_gptq": test_deepspeed_zero2_gptq,
    "deepspeed_zero2_ntp": test_deepspeed_zero2_ntp,
    "deepspeed_zero2_via_runner": test_deepspeed_zero2_via_runner,
    "deepspeed_zero2_intweight": test_deepspeed_zero2_intweight,
    "large_model_distributed": test_large_model_distributed,
    "large_model_distributed_ntp": test_large_model_distributed_ntp,
    "speedup_timed": test_speedup_timed,
    "consistency_baseline": test_consistency_baseline,
    "consistency_verify": test_consistency_verify,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", required=True, choices=list(TEST_MAP.keys()),
        help="Test case to run",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log(f"Running test '{args.test}' (world_size={world_size})")

    if args.test in REQUIRES_MULTI_GPU and world_size < 2:
        print(
            f"ERROR: Test '{args.test}' requires at least 2 GPUs. "
            "Launch with: torchrun --nproc_per_node=2 <this_script> --test <name>",
            file=sys.stderr,
        )
        sys.exit(1)

    test_fn = TEST_MAP[args.test]
    test_fn()


if __name__ == "__main__":
    main()
