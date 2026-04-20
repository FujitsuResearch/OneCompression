"""Save/load round-trip tests for rotation + quantization (Qwen3).

3 cases: GPTQ quantized, GPTQ dequantized, RTN dequantized.
Requires CUDA and model download.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami

Usage::

    pytest tests/onecomp/pre_process/test_save_load_pipeline_qwen3.py -v -s --log-cli-level=INFO
"""

import pytest
import torch

from onecomp.pre_process.prepare_rotated_model import prepare_rotated_model

from .conftest import E2E_CALIB, PROMPT, QWEN3_ID


def _cases():
    mid = QWEN3_ID
    return [
        pytest.param(mid, "gptq", "quantized", id="qwen3-gptq-save_quantized"),
        pytest.param(mid, "gptq", "dequantized", id="qwen3-gptq-save_dequantized"),
        pytest.param(mid, "rtn", "dequantized", id="qwen3-rtn-save_dequantized"),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSaveLoadPipelineQwen3:
    """Verify save/load round-trip with rotation + quantization (Qwen3)."""

    @pytest.mark.parametrize("model_id, quant_type, save_type", _cases())
    def test_save_load(self, model_id, quant_type, save_type, tmp_path):
        from onecomp import GPTQ, RTN, ModelConfig, Runner, load_quantized_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda:0"
        model_config = ModelConfig(model_id=model_id, device=device)

        rot_dir = str(tmp_path / "rotated")
        rot_kwargs = dict(
            rotation=True,
            scaling=False,
            enable_training=True,
            calibration_config=E2E_CALIB,
            wbits=4,
            sym=False,
            groupsize=-1,
            training_args_override=dict(
                output_dir=str(tmp_path / "train_output"),
                max_steps=2,
                per_device_train_batch_size=1,
            ),
        )
        rotated_config = prepare_rotated_model(
            model_config=model_config,
            save_directory=rot_dir,
            **rot_kwargs,
        )

        if quant_type == "gptq":
            quantizer = GPTQ(wbits=4, groupsize=128)
        else:
            quantizer = RTN(wbits=4, groupsize=-1)
        runner = Runner(
            model_config=rotated_config,
            quantizer=quantizer,
            calibration_config=E2E_CALIB,
        )
        runner.run()

        tokenizer_pre = rotated_config.load_tokenizer()
        inputs_pre = tokenizer_pre(PROMPT, return_tensors="pt")
        inputs_pre = {k: v.to(device) for k, v in inputs_pre.items()}

        if save_type == "quantized":
            model_before, _ = runner.create_quantized_model(use_gemlite=False)
            model_before.to(device)
        else:
            model_before = AutoModelForCausalLM.from_pretrained(
                rotated_config.path,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            runner.update_model_weights(model_before)
            model_before.to(device)
        model_before.eval()
        with torch.no_grad():
            logits_before = model_before(**inputs_pre).logits.float().cpu()
        del model_before

        save_dir = str(tmp_path / "saved")
        if save_type == "quantized":
            runner.save_quantized_model(save_dir)
        else:
            runner.save_dequantized_model(save_dir)
        del runner

        if save_type == "quantized":
            model, tokenizer = load_quantized_model(save_dir, device_map=device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                save_dir,
                torch_dtype=torch.float16,
                device_map=device,
            )
            tokenizer = AutoTokenizer.from_pretrained(save_dir)

        model.eval()
        inputs = tokenizer(PROMPT, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits_after = model(**inputs).logits.float().cpu()

        assert logits_after is not None
        assert logits_after.shape[0] == 1
        assert torch.all(torch.isfinite(logits_after))

        max_diff = (logits_before - logits_after).abs().max().item()
        assert (
            max_diff < 1e-3
        ), f"save/load round-trip: max logits diff {max_diff:.6f} exceeds 1e-3"
        del model
