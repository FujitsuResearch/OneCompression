"""
QEP reference data generation script.

Quantize TinyLlama with GPTQ+QEP and save the quantization errors as reference data.
Run once before refactoring to generate baseline values for testing.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    python tests/onecomp/fixtures/generate_qep_gptq_reference.py
"""

from pathlib import Path

from onecomp import ModelConfig, Runner
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ
from onecomp import setup_logger


def run_qep_gptq_quantization():
    """Run GPTQ+QEP quantization and return the results.

    Uses the same settings shared between tests and reference data generation.

    Returns:
        dict: Quantization results (quantizer.results).
    """
    # Set up logger
    setup_logger()

    # Use TinyLlama model (small model for faster testing)
    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device="cuda:0",
    )

    # GPTQ quantizer (only a few layers)
    quantizer = GPTQ(
        wbits=4,
        groupsize=128,
        sym=False,
        num_layers=7,  # Quantize only the first 7 layers (for speed)
        calc_quant_error=True,
    )

    # Create Runner with QEP enabled (generic implementation)
    runner = Runner(
        model_config=model_config,
        max_length=512,
        num_calibration_samples=128,
        calibration_strategy="drop_rand",
        calibration_seed=0,
        quantizer=quantizer,
        qep=True,
        qep_config=QEPConfig(general=True),
    )

    # Run quantization
    runner.run()

    return runner


def generate_qep_gptq_reference():
    """Generate GPTQ+QEP reference data."""
    runner = run_qep_gptq_quantization()

    # Save quantization statistics (saved in the same directory as this script)
    output_path = Path(__file__).parent / "qep_gptq_reference.json"
    runner.save_quantization_statistics(str(output_path))
    print(f"\nReference data saved to: {output_path}")

    # Generate and save cumulative error reference data
    cumulative_error_path = Path(__file__).parent / "qep_gptq_cumulative_error_reference.json"
    runner.analyze_cumulative_error(
        layer_keywords=["mlp.down_proj", "self_attn.o_proj"],
        json_path=str(cumulative_error_path),
    )
    print(f"Cumulative error reference saved to: {cumulative_error_path}")


if __name__ == "__main__":
    generate_qep_gptq_reference()
