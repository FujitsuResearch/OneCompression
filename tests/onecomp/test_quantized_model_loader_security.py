"""Regression tests for the unsafe-deserialization hardening (CWE-502).

These tests lock in the security behaviour added in v1.2.1:

* ``QuantizedModelLoader.load_quantized_model_pt`` and
  ``Quantizer.load_results`` / ``ResultLoader`` load ``.pt`` files with
  ``torch.load(weights_only=False)``, which deserializes arbitrary Python
  objects via ``pickle`` and can execute code embedded in a malicious file.
* Both entry points now refuse to load unless the caller explicitly passes
  ``allow_unsafe_deserialization=True``.  The refusal happens *before*
  ``torch.load`` is ever invoked, so no attacker-controlled code can run.
* The safe ``weights_only=True`` path of ``load_results`` still works
  without any opt-in.

The tests patch ``torch.load`` so they never actually deserialize anything;
the key assertions are (a) ``ValueError`` is raised on the default path and
(b) ``torch.load`` is *not* called in that case.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from unittest.mock import MagicMock, patch

import pytest

from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.quantizer import ResultLoader


def _make_pt_dir(tmp_path):
    """Create a save directory containing a dummy (never-read) model.pt."""
    save_dir = tmp_path / "saved_model_lora"
    save_dir.mkdir()
    # Content is irrelevant: torch.load is patched / never reached.
    (save_dir / "model.pt").write_bytes(b"not a real checkpoint")
    return save_dir


class TestLoadQuantizedModelPtOptIn:
    def test_refuses_without_opt_in_and_does_not_call_torch_load(self, tmp_path):
        save_dir = _make_pt_dir(tmp_path)

        with (
            patch("onecomp.quantized_model_loader.torch.load") as mock_load,
            patch("onecomp.quantized_model_loader.AutoTokenizer.from_pretrained") as mock_tok,
        ):
            with pytest.raises(ValueError, match="allow_unsafe_deserialization"):
                QuantizedModelLoader.load_quantized_model_pt(str(save_dir))

        # The refusal must happen before any deserialization.
        mock_load.assert_not_called()
        mock_tok.assert_not_called()

    def test_loads_with_opt_in(self, tmp_path):
        save_dir = _make_pt_dir(tmp_path)
        sentinel_model = MagicMock(name="model")
        sentinel_tokenizer = MagicMock(name="tokenizer")

        with (
            patch(
                "onecomp.quantized_model_loader.torch.load",
                return_value=sentinel_model,
            ) as mock_load,
            patch(
                "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
                return_value=sentinel_tokenizer,
            ),
        ):
            model, tokenizer = QuantizedModelLoader.load_quantized_model_pt(
                str(save_dir),
                device_map="",  # skip accelerate / device placement
                allow_unsafe_deserialization=True,
            )

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer
        mock_load.assert_called_once()
        # weights_only=False is intentional here (custom module objects).
        assert mock_load.call_args.kwargs.get("weights_only") is False

    def test_opt_in_emits_warning(self, tmp_path, caplog):
        save_dir = _make_pt_dir(tmp_path)

        with (
            patch(
                "onecomp.quantized_model_loader.torch.load",
                return_value=MagicMock(),
            ),
            patch(
                "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
                return_value=MagicMock(),
            ),
        ):
            with caplog.at_level("WARNING", logger="onecomp.quantized_model_loader"):
                QuantizedModelLoader.load_quantized_model_pt(
                    str(save_dir),
                    device_map="",
                    allow_unsafe_deserialization=True,
                )

        assert any("weights_only=False" in r.message for r in caplog.records)

    def test_missing_model_pt_raises_file_not_found(self, tmp_path):
        save_dir = tmp_path / "empty_dir"
        save_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            QuantizedModelLoader.load_quantized_model_pt(
                str(save_dir), allow_unsafe_deserialization=True
            )


class TestLoadResultsOptIn:
    def test_refuses_unsafe_without_opt_in(self, tmp_path):
        loader = ResultLoader()
        path = str(tmp_path / "results.pt")

        with patch("onecomp.quantizer._quantizer.torch.load") as mock_load:
            with pytest.raises(ValueError, match="allow_unsafe_deserialization"):
                loader.load_results(path, weights_only=False)
        mock_load.assert_not_called()

    def test_unsafe_with_opt_in_loads(self, tmp_path):
        loader = ResultLoader()
        path = str(tmp_path / "results.pt")
        sentinel = {"layer": object()}

        with patch("onecomp.quantizer._quantizer.torch.load", return_value=sentinel) as mock_load:
            out = loader.load_results(path, weights_only=False, allow_unsafe_deserialization=True)

        assert out is sentinel
        assert loader.results is sentinel
        assert mock_load.call_args.kwargs.get("weights_only") is False

    def test_safe_path_needs_no_opt_in(self, tmp_path):
        loader = ResultLoader()
        path = str(tmp_path / "results.pt")
        sentinel = {"layer": object()}

        with patch("onecomp.quantizer._quantizer.torch.load", return_value=sentinel) as mock_load:
            out = loader.load_results(path, weights_only=True)

        assert out is sentinel
        assert mock_load.call_args.kwargs.get("weights_only") is True


class TestResultLoaderConstructor:
    def test_constructor_refuses_unsafe_without_opt_in(self, tmp_path):
        path = str(tmp_path / "results.pt")
        with patch("onecomp.quantizer._quantizer.torch.load") as mock_load:
            with pytest.raises(ValueError, match="allow_unsafe_deserialization"):
                ResultLoader(results_file=path, weights_only=False)
        mock_load.assert_not_called()

    def test_constructor_loads_with_opt_in(self, tmp_path):
        path = str(tmp_path / "results.pt")
        sentinel = {"layer": object()}
        with patch("onecomp.quantizer._quantizer.torch.load", return_value=sentinel):
            loader = ResultLoader(
                results_file=path,
                weights_only=False,
                allow_unsafe_deserialization=True,
            )
        assert loader.results is sentinel
