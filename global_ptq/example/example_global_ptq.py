"""
Example: Global PTQ with OneComp

Copyright 2025-2026 Fujitsu Ltd.

"""

from onecomp import Runner, ModelConfig, GPTQ, CalibrationConfig, setup_logger
from onecomp_globalptq import GlobalPTQ

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

setup_logger()
model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
calibration_config = CalibrationConfig(max_length=512, num_calibration_samples=128)
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(
    model_config=model_config,
    calibration_config=calibration_config,
    quantizer=gptq,
    post_processes=[GlobalPTQ()],
)
runner.run()
