"""Runtime patches for vLLM when serving OneComp GPT-OSS mixed_gptq checkpoints."""

from vllm_plugins.patches.apply_all import apply_gpt_oss_vllm_patches

__all__ = ["apply_gpt_oss_vllm_patches"]
