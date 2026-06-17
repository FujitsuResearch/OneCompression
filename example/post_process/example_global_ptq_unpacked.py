"""
Example: quantization + Global PTQ (unpacked buffers)

Demonstrates the explicit unpacked GlobalPTQ workflow:
    1. Quantize TinyLlama with the selected quantizer
    2. Evaluate baseline PPL (original vs quantized-only)
    3. Build an unpacked quantized model with pack_weights=False
    4. Apply GlobalPTQ directly and evaluate the result

This is a research/debug counterpart to example_global_ptq.py.  The normal
Runner-managed path builds packed buffers by default and is the recommended
path for reusable saved checkpoints and vLLM-serving workflows.

Current cases that must use unpacked GPTQLinear buffers:
    - JointQ(bits=1, ...), because GPTQLinear bit packing does not support
      1-bit JointQ output.
    - GPTQ/RTN bit widths in {1, 5, 6, 7}, because the GPTQLinear packing
      helpers currently support only {2, 3, 4, 8}.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura

Usage:
    python example/post_process/example_global_ptq_unpacked.py
"""

import torch

from onecomp import (
    DBF,  # noqa: F401
    GPTQ,
    JointQ,  # noqa: F401
    RTN,  # noqa: F401
    CalibrationConfig,
    GlobalPTQ,
    ModelConfig,
    Runner,
    setup_logger,
)


def main():
    setup_logger()

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(model_id=model_id, device=device)

    # GlobalPTQ currently supports these quantized module families:
    #   GPTQLinear:         GPTQ, RTN, JointQ
    #   DoubleBinaryLinear: DBF
    quantizer = GPTQ(wbits=4, groupsize=128)
    # quantizer = RTN(wbits=4, groupsize=128)
    # quantizer = JointQ(bits=4, group_size=128)
    # quantizer = DBF(target_bits=1.5)
    #
    # Mandatory unpacked GPTQLinear-backed settings today:
    # quantizer = GPTQ(wbits=1, groupsize=128)
    # quantizer = RTN(wbits=1, groupsize=128)
    # quantizer = JointQ(bits=1, group_size=128)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
    )
    # NOTE: The calibration settings above are kept compact so the demo runs
    # fast and may be insufficient for real quantization.  For higher quality,
    # prefer the CalibrationConfig() defaults
    # (max_length=2048, num_calibration_samples=512).
    # For qep=False runs with large calibration data, also pass ``batch_size``
    # as a CalibrationConfig argument, e.g.
    #   CalibrationConfig(
    #       max_length=2048,
    #       num_calibration_samples=512,
    #       batch_size=128,
    #   )
    # so that Runner.quantize_with_calibration_chunked runs instead of a
    # single all-at-once forward pass.
    runner.run()

    original_ppl, _, baseline_ppl = runner.calculate_perplexity(
        original_model=True,
        quantized_model=True,
    )
    print(f"\nOriginal PPL:             {original_ppl:.4f}")
    print(f"Quantized baseline PPL:   {baseline_ppl:.4f}")

    global_ptq = GlobalPTQ(
        epochs=3,
        gptq_lr=1e-5,
        dbf_lr=5e-4,
        calibration_config=CalibrationConfig(
            num_calibration_samples=32,
            max_length=512,
        ),
        eval_interval=1,
        use_gradient_checkpointing=True,
    )

    # NOTE: Explicitly request unpacked qweight/qzeros buffers for
    # GPTQLinear-backed quantizers. This is not the normal Runner-managed
    # post-process path; use the packed example for standard saved checkpoints.
    model, _ = runner.create_quantized_model(pack_weights=False, use_gemlite=False)
    global_ptq.run(model, model_config)
    runner.quantized_model = model

    _, _, global_ptq_ppl = runner.calculate_perplexity(quantized_model=True)
    print(f"Quantized + Global PTQ PPL: {global_ptq_ppl:.4f}")
    print(f"PPL improvement:            {baseline_ppl - global_ptq_ppl:.4f}")


if __name__ == "__main__":
    main()
