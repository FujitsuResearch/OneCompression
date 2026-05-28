# Legacy vLLM Environment Setup (v0.15.1)

Use the following commands to create and run the legacy vLLM environment:

```bash
python3.12 -m venv envs/vllm/0.15.1/.venv
source envs/vllm/0.15.1/.venv/bin/activation
python -m pip install --no-deps -r envs/vllm/0.15.1/requirements.txt
python example/vllm_inference/example_gptq_vllm_inference.py
```
