"""Model card generation for quantized models published to the Hub.

Builds a Markdown model card with Hugging Face YAML frontmatter
(license, base_model, tags) plus tables describing the quantization
recipe and evaluation results, ready to be uploaded as ``README.md``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import os
from logging import getLogger
from typing import Any, Dict, Optional

logger = getLogger(__name__)

_DEFAULT_TAGS = ("onecomp", "quantized", "gptq")


def generate_model_card(
    model_id: str,
    recipe: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
    *,
    repo_id: Optional[str] = None,
) -> str:
    """Generate a Markdown model card for a quantized model.

    Args:
        model_id (str): Base model ID (Hugging Face Hub ID or local
            path); written to ``base_model`` in the frontmatter.
        recipe (dict or None): Quantization recipe, e.g.
            ``{"method": "AutoBit + QEP", "wbits": 4, "groupsize": 128}``.
            An optional ``"license"`` key overrides the frontmatter
            license (default: ``"apache-2.0"``).
        results (dict or None): Evaluation results, e.g.
            ``{"perplexity (wikitext2)": 8.12, "accuracy (lambada)": 0.65}``.
        repo_id (str or None): Hub repository containing the quantized model.
            When provided, this ID is used in the Usage example. When omitted,
            the existing base-model-derived name is used for compatibility.

    Returns:
        str: The model card as a Markdown string with YAML frontmatter.

    Examples:
        >>> card = generate_model_card(
        ...     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     recipe={"method": "AutoBit + QEP", "wbits": 4},
        ...     results={"perplexity (wikitext2)": 8.12},
        ...     repo_id="your-name/tinyllama-onecomp",
        ... )
        >>> card.startswith("---")
        True
    """
    recipe = dict(recipe or {})
    results = dict(results or {})
    license_id = recipe.pop("license", "apache-2.0")
    model_name = model_id.rstrip("/").split("/")[-1]
    is_local_model = os.path.exists(os.path.expanduser(model_id))
    if is_local_model:
        base_model_reference = f"`{model_id}`"
    else:
        base_model_reference = f"[{model_id}](https://huggingface.co/{model_id})"
    usage_model_id = repo_id or model_name

    lines = [
        "---",
        f"license: {license_id}",
        f"base_model: {model_id}",
        "tags:",
    ]
    lines.extend(f"  - {tag}" for tag in _DEFAULT_TAGS)
    lines.extend(
        [
            "---",
            "",
            f"# {model_name} (OneComp quantized)",
            "",
            f"Quantized version of {base_model_reference} "
            "produced with [OneComp](https://github.com/FujitsuResearch/OneCompression), "
            "Fujitsu's LLM compression library.",
            "",
        ]
    )

    if recipe:
        lines.extend(["## Quantization Recipe", ""])
        lines.extend(_render_table("Setting", "Value", recipe))
        lines.append("")

    if results:
        lines.extend(["## Evaluation Results", ""])
        lines.extend(_render_table("Metric", "Value", results))
        lines.append("")

    lines.extend(
        [
            "## Usage",
            "",
            "```python",
        ]
    )

    if repo_id:
        lines.extend(
            [
                "from huggingface_hub import snapshot_download",
                "from onecomp import load_quantized_model",
                "",
                f'model_dir = snapshot_download(repo_id="{repo_id}")',
                "model, tokenizer = load_quantized_model(model_dir)",
            ]
        )
    else:
        lines.extend(
            [
                "from onecomp import load_quantized_model",
                "",
                f'model, tokenizer = load_quantized_model("{usage_model_id}")',
            ]
        )

    lines.extend(
        [
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _render_table(key_header: str, value_header: str, rows: Dict[str, Any]) -> list:
    """Render a two-column Markdown table from a dictionary."""
    lines = [f"| {key_header} | {value_header} |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    return lines
