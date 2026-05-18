"""

Copyright 2025-2026 Fujitsu Ltd.

"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch


logger = getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for layer-wise quantization checkpointing.

    Saves quantization results periodically during quantization so that
    interrupted runs can be resumed from the latest checkpoint rather than
    restarting from scratch.

    Checkpoints are saved at transformer-block boundaries.  For a Llama-7B
    model with 32 blocks and ``interval_blocks=1`` (the default), up to 32
    checkpoint files are written.  With ``interval_blocks=4``, checkpoints are
    written after blocks 3, 7, 11, … and at the very end.

    Attributes:
        checkpoint_dir (str):
            Directory where checkpoint files are stored.  Created
            automatically if it does not exist.
        interval_blocks (int):
            Save a checkpoint every *N* completed transformer blocks.
            Default is 1 (checkpoint after every block).
        resume (bool):
            When True (default), automatically load the latest checkpoint in
            ``checkpoint_dir`` and skip already-quantized layers.  Set to
            False to start fresh even when a checkpoint exists.

    Examples:
        Basic usage – checkpoint every block, auto-resume:

        >>> from onecomp import CheckpointConfig, Runner, ModelConfig
        >>> from onecomp.quantizer.gptq import GPTQ
        >>> runner = Runner(
        ...     model_config=ModelConfig(model_id="meta-llama/Llama-2-7b-hf"),
        ...     quantizer=GPTQ(wbits=4, groupsize=128),
        ...     checkpoint_config=CheckpointConfig(checkpoint_dir="./ckpt"),
        ... )
        >>> runner.run()

        Checkpoint every 4 blocks, no auto-resume:

        >>> runner = Runner(
        ...     model_config=ModelConfig(model_id="meta-llama/Llama-2-7b-hf"),
        ...     quantizer=GPTQ(wbits=4, groupsize=128),
        ...     checkpoint_config=CheckpointConfig(
        ...         checkpoint_dir="./ckpt",
        ...         interval_blocks=4,
        ...         resume=False,
        ...     ),
        ... )
        >>> runner.run()
    """

    checkpoint_dir: str
    interval_blocks: int = 1
    resume: bool = True


_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)layers?\.(\d+)\.")


class CheckpointManager:
    """Manages checkpoint save / load for layer-wise quantization.

    Intended for internal use by :class:`~onecomp.runner.Runner`.  Users
    configure checkpointing via :class:`CheckpointConfig`.

    The checkpoint directory contains:

    - ``checkpoint_<NNNN>.pt`` – ``quantizer.results`` (partial) saved at
      block boundaries.  Only the most recent file is kept; older ones are
      removed automatically.
    - ``checkpoint_meta.json`` – metadata for compatibility verification.

    Args:
        config (CheckpointConfig): User-facing configuration.
        model_id (str): Model identifier (used for compatibility checks).
        quantizer: The :class:`~onecomp.quantizer.Quantizer` instance.
    """

    META_FILE = "checkpoint_meta.json"

    def __init__(self, config: CheckpointConfig, model_id: str, quantizer):
        self.config = config
        self.model_id = model_id
        self.quantizer = quantizer

        self._dir = Path(config.checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._last_block: Optional[int] = None
        self._blocks_since_save: int = 0
        self._completed_layers: set = set()
        self._loaded_results: Optional[dict] = None

    # ------------------------------------------------------------------
    # Block index extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_block_index(layer_name: str) -> Optional[int]:
        """Return the transformer-block index embedded in *layer_name*, or None.

        Handles naming patterns used by Llama, Qwen, Gemma, etc.:
        - ``model.layers.5.self_attn.q_proj``  → 5
        - ``transformer.h.3.attn.c_attn``      → 3
        - ``lm_head``                           → None
        """
        m = _BLOCK_INDEX_RE.search(layer_name)
        return int(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # Callback wired into Quantizer.on_layer_quantized
    # ------------------------------------------------------------------

    def on_layer_quantized(self, layer_name: str, results: dict) -> None:
        """Called by :class:`~onecomp.quantizer.Quantizer` after each layer.

        Tracks block transitions and triggers a checkpoint save when the
        configured number of blocks has been completed.
        """
        self._completed_layers.add(layer_name)
        block_idx = self._extract_block_index(layer_name)

        if block_idx is None:
            return

        if self._last_block is None:
            self._last_block = block_idx
            return

        if block_idx != self._last_block:
            # A new block started → the previous block is complete.
            self._blocks_since_save += 1
            if self._blocks_since_save >= self.config.interval_blocks:
                self._save(results, self._last_block)
                self._blocks_since_save = 0
            self._last_block = block_idx

    def flush(self, results: dict) -> None:
        """Save a final checkpoint regardless of the interval setting.

        Called by the runner after quantization completes so that the last
        (possibly partial) block is always persisted.
        """
        last_block = self._last_block if self._last_block is not None else 0
        self._save(results, last_block, label="final")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _checkpoint_path(self, block_idx: int, label: str = "") -> Path:
        tag = label if label else f"{block_idx:04d}"
        return self._dir / f"checkpoint_{tag}.pt"

    def _save(self, results: dict, block_idx: int, label: str = "") -> None:
        """Persist *results* and update the meta file."""
        ckpt_path = self._checkpoint_path(block_idx, label)
        torch.save(results, ckpt_path)
        logger.info("Checkpoint saved: %s (%d layers)", ckpt_path, len(results))

        meta = {
            "onecomp_version": self._onecomp_version(),
            "model_id": self.model_id,
            "quantizer_class": type(self.quantizer).__name__,
            "quantizer_key_params": self._key_params(),
            "completed_layers": list(results.keys()),
            "last_completed_block": block_idx,
            "checkpoint_file": ckpt_path.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path = self._dir / self.META_FILE
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

        # Remove stale checkpoint files (keep only the latest + final)
        for old in self._dir.glob("checkpoint_*.pt"):
            if old != ckpt_path and not old.name.startswith("checkpoint_final"):
                old.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Load / verify
    # ------------------------------------------------------------------

    def load_latest(self) -> Optional[dict]:
        """Load the latest checkpoint results, or return None if none exist."""
        meta_path = self._dir / self.META_FILE
        if not meta_path.exists():
            return None

        with meta_path.open() as f:
            meta = json.load(f)

        ckpt_file = self._dir / meta["checkpoint_file"]
        if not ckpt_file.exists():
            logger.warning("Checkpoint file listed in meta not found: %s", ckpt_file)
            return None

        results = torch.load(ckpt_file, weights_only=False)
        self._completed_layers = set(results.keys())
        self._loaded_results = results
        logger.info(
            "Loaded checkpoint from %s (%d layers completed)",
            ckpt_file,
            len(results),
        )
        return results

    def load_meta(self) -> Optional[dict]:
        """Return the metadata dict from the latest checkpoint, or None."""
        meta_path = self._dir / self.META_FILE
        if not meta_path.exists():
            return None
        with meta_path.open() as f:
            return json.load(f)

    def verify_compatibility(self) -> None:
        """Raise :exc:`ValueError` if the checkpoint was made with a different model or quantizer.

        Only the ``onecomp_version`` mismatch produces a warning instead of an error,
        because minor-version changes are expected to be backward-compatible.

        Raises:
            ValueError: If ``model_id``, ``quantizer_class``, or
                ``quantizer_key_params`` do not match.
        """
        meta = self.load_meta()
        if meta is None:
            return

        from .__version__ import __version__  # pylint: disable=import-outside-toplevel

        if meta.get("onecomp_version") != __version__:
            logger.warning(
                "Checkpoint was created with onecomp %s; current version is %s. "
                "Compatibility is not guaranteed.",
                meta.get("onecomp_version"),
                __version__,
            )

        if meta.get("model_id") != self.model_id:
            raise ValueError(
                f"Checkpoint model_id mismatch: checkpoint has "
                f"'{meta.get('model_id')}', but current model is '{self.model_id}'. "
                f"Use a fresh checkpoint_dir or set CheckpointConfig(resume=False)."
            )

        ckpt_class = meta.get("quantizer_class")
        cur_class = type(self.quantizer).__name__
        if ckpt_class != cur_class:
            raise ValueError(
                f"Checkpoint quantizer_class mismatch: checkpoint has '{ckpt_class}', "
                f"but current quantizer is '{cur_class}'."
            )

        ckpt_params = meta.get("quantizer_key_params", {})
        cur_params = self._key_params()
        if ckpt_params != cur_params:
            raise ValueError(
                f"Checkpoint quantizer_key_params mismatch:\n"
                f"  checkpoint : {ckpt_params}\n"
                f"  current    : {cur_params}\n"
                f"Use a fresh checkpoint_dir or set CheckpointConfig(resume=False)."
            )

    def get_completed_layers(self) -> set:
        """Return the set of layer names already present in the checkpoint."""
        return set(self._completed_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _onecomp_version() -> str:
        from .__version__ import __version__  # pylint: disable=import-outside-toplevel

        return __version__

    def _key_params(self) -> dict:
        """Extract serialisable key parameters from the quantizer for compatibility checks."""
        try:
            params = self.quantizer.get_quant_config()
            # Keep only JSON-serialisable scalar values.
            return {
                k: v
                for k, v in params.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        except Exception:  # pylint: disable=broad-except
            return {}
