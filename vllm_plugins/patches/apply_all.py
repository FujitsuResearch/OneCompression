"""
Apply all vLLM runtime patches required for GPT-OSS mixed_gptq serving.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import logging
import sys

from vllm_plugins.patches import gpt_oss_gptq_moe as gptq_moe_patch
from vllm_plugins.patches import gpt_oss_wna16_bias as wna16_bias_patch

logger = logging.getLogger(__name__)

# vLLM versions the source patches have actually been verified against. The
# patches anchor on specific source strings in vLLM's ``fused_moe.py`` /
# ``gpt_oss.py``; other versions may share the same anchors yet behave
# differently, so anything outside this set is applied on a best-effort basis.
VERIFIED_VLLM_VERSIONS = ("0.20.2",)

_PATCHES = (
    ("gptq_moe_experts", gptq_moe_patch),
    ("wna16_bias", wna16_bias_patch),
)


def _installed_vllm_version() -> str | None:
    try:
        import vllm

        return getattr(vllm, "__version__", None)
    except Exception:  # pragma: no cover - vllm import failures surface later
        return None


def _log_version_status() -> None:
    version = _installed_vllm_version()
    verified = ", ".join(VERIFIED_VLLM_VERSIONS)
    if version is None:
        logger.warning(
            "Could not determine installed vLLM version; patches verified against: %s",
            verified,
        )
    elif version in VERIFIED_VLLM_VERSIONS:
        logger.warning("Installed vLLM %s is a patch-verified version.", version)
    else:
        logger.warning(
            "Installed vLLM %s is NOT in the patch-verified set (%s). Anchors may "
            "match but produce unintended behavior; verify serving output.",
            version,
            verified,
        )


def apply_gpt_oss_vllm_patches(*, dry_run: bool = False) -> list[str]:
    """Apply GPT-OSS vLLM patches. Safe to call repeatedly (idempotent)."""
    _log_version_status()
    messages: list[str] = []
    for name, module in _PATCHES:
        try:
            msg = module.apply(dry_run=dry_run)
        except Exception as exc:
            raise RuntimeError(f"patch {name} failed: {exc}") from exc
        messages.append(msg)
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply vLLM runtime patches for OneComp GPT-OSS mixed_gptq checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate patch patterns without writing files.",
    )
    args = parser.parse_args(argv)
    for line in apply_gpt_oss_vllm_patches(dry_run=args.dry_run):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
