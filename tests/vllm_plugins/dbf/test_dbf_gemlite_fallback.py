"""Copyright 2025-2026 Fujitsu Ltd."""

import logging
import pytest
import torch
from unittest.mock import MagicMock, patch

try:
    import vllm_plugins.dbf.vllm_plugin as _plugin_module
    from vllm_plugins.dbf.vllm_plugin import (
        DBFLinearMethod,
        DbfConfig,
        _disable_gemlite_runtime,
    )

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

pytestmark = pytest.mark.skipif(not _HAS_VLLM, reason="vllm not available")

_IN_F = 16
_MID_F = 4
_OUT_F = 8
_BATCH = 2

_MOCK_GEMLITE_IMPORT = (MagicMock(), 128)


def _make_method():
    return DBFLinearMethod(DbfConfig(quantization_bits=[]))


def _make_layer(part_count=1, use_gemlite=True):
    bp1_size = (_MID_F * _IN_F + 7) // 8
    bp3_size = (_OUT_F * _MID_F + 7) // 8

    class _Layer:
        pass

    layer = _Layer()
    layer._dbf_meta = {
        "part_count": part_count,
        "in_features": _IN_F,
        "out_sizes": [_OUT_F] * part_count,
        "mid_sizes": [_MID_F] * part_count,
        "scaling2_offsets": [i * _MID_F for i in range(part_count)],
        "scaling4_offsets": [i * _OUT_F for i in range(part_count)],
        "bp1_offsets": [i * bp1_size for i in range(part_count)],
        "bp3_offsets": [i * bp3_size for i in range(part_count)],
        "bp1_sizes": [bp1_size] * part_count,
        "bp3_sizes": [bp3_size] * part_count,
    }
    layer._dbf_use_gemlite = use_gemlite

    fake_binary = MagicMock()
    layer._dbf_gemlite_parts = [(fake_binary, fake_binary)] * part_count

    layer.scaling0 = torch.ones(_IN_F, dtype=torch.float16)
    layer.scaling2 = torch.ones(_MID_F * part_count, dtype=torch.float16)
    layer.scaling4 = torch.ones(_OUT_F * part_count, dtype=torch.float16)
    layer.bp1 = torch.zeros(bp1_size * part_count, dtype=torch.uint8)
    layer.bp3 = torch.zeros(bp3_size * part_count, dtype=torch.uint8)

    return layer


def _parts_side_effect_raise_on_gemlite(_layer, _x, meta, use_gemlite, _group_size):
    """Substitute for _compute_parts: raises on use_gemlite=True, returns zeros on False."""
    if use_gemlite:
        raise RuntimeError("kernel error")
    return torch.zeros(_BATCH, _OUT_F * meta["part_count"])


@pytest.fixture(autouse=True)
def reset_gemlite_disabled_flag(monkeypatch):
    """Reset the process-wide GemLite disabled flag before and after each test."""
    monkeypatch.setattr(_plugin_module, "_GEMLITE_RUNTIME_DISABLED", False)


@pytest.fixture
def method():
    return _make_method()


@pytest.fixture
def layer():
    return _make_layer()


class TestGemliteFallback:
    """Verify the GemLite → naive fallback behaviour in DBFLinearMethod.apply()."""

    def test_fallback_calls_naive_on_gemlite_exception(self, method, layer):
        """On a GemLite exception, apply() falls back to naive; GemLite is tried exactly once and the output matches the naive reference."""
        x = torch.randn(_BATCH, _IN_F)
        naive_ref = torch.zeros(_BATCH, _OUT_F)
        gemlite_tries = 0

        def _gemlite_raises(*_args, **_kwargs):
            nonlocal gemlite_tries
            gemlite_tries += 1
            raise RuntimeError("kernel error")

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(method, "_apply_gemlite", side_effect=_gemlite_raises):
                with patch.object(method, "_apply_naive", return_value=naive_ref):
                    result = method.apply(layer, x, bias=None)

        assert gemlite_tries == 1
        assert torch.equal(result, naive_ref)

    def test_fallback_warning_contains_env_vars(self, method, layer, caplog):
        """The fallback warning must mention both env var names."""
        x = torch.randn(_BATCH, _IN_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method, "_compute_parts", side_effect=_parts_side_effect_raise_on_gemlite
            ):
                with caplog.at_level(logging.WARNING):
                    method.apply(layer, x, bias=None)

        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "ONECOMP_DBF_NAIVE_LINEAR" in combined
        assert "TRITON_CACHE_AUTOTUNING" in combined

    def test_fallback_sets_process_wide_disabled_flag(self, method, layer):
        """After a fallback the process-wide disabled flag must be True."""
        x = torch.randn(_BATCH, _IN_F)
        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is False

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method, "_compute_parts", side_effect=_parts_side_effect_raise_on_gemlite
            ):
                method.apply(layer, x, bias=None)

        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is True

    def test_fallback_sets_layer_use_gemlite_false(self, method, layer):
        x = torch.randn(_BATCH, _IN_F)
        assert layer._dbf_use_gemlite is True

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method, "_compute_parts", side_effect=_parts_side_effect_raise_on_gemlite
            ):
                method.apply(layer, x, bias=None)

        assert layer._dbf_use_gemlite is False

    def test_disable_gemlite_runtime_logs_once(self, caplog):
        """_disable_gemlite_runtime() emits exactly one WARNING across multiple calls and sets the flag to True."""
        exc = RuntimeError("error")
        with caplog.at_level(logging.WARNING):
            _disable_gemlite_runtime(exc)
            _disable_gemlite_runtime(exc)
            _disable_gemlite_runtime(exc)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        message = warning_records[0].getMessage()
        assert "ONECOMP_DBF_NAIVE_LINEAR" in message
        assert "TRITON_CACHE_AUTOTUNING" in message
        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is True

    def test_subsequent_call_skips_gemlite(self, method, layer):
        """A second apply() after a fallback skips GemLite entirely and goes straight to naive."""
        x = torch.randn(_BATCH, _IN_F)
        naive_ref = torch.zeros(_BATCH, _OUT_F)
        gemlite_tries = 0

        def _gemlite_raises(*_args, **_kwargs):
            nonlocal gemlite_tries
            gemlite_tries += 1
            raise RuntimeError("kernel error")

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(method, "_apply_gemlite", side_effect=_gemlite_raises):
                with patch.object(method, "_apply_naive", return_value=naive_ref):
                    method.apply(layer, x, bias=None)  # first call: fallback triggered
                    method.apply(layer, x, bias=None)  # second call: goes straight to naive via process-wide flag

        # GemLite was tried only on the first call; second call bypasses it via the process-wide flag
        assert gemlite_tries == 1

    def test_oom_is_not_caught(self, method, layer, caplog):
        """OOM is re-raised: GemLite is more memory-efficient than naive, so falling back on OOM would make things worse."""
        x = torch.randn(_BATCH, _IN_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method, "_apply_gemlite",
                side_effect=torch.cuda.OutOfMemoryError("OOM"),
            ):
                with caplog.at_level(logging.WARNING):
                    with pytest.raises(torch.cuda.OutOfMemoryError):
                        method.apply(layer, x, bias=None)

        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is False
        assert layer._dbf_use_gemlite is True  # OOM must not touch the per-layer flag
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0  # OOM must not emit a WARNING

    def test_naive_direct_when_use_gemlite_false(self, method):
        """A layer with _dbf_use_gemlite=False bypasses GemLite entirely and goes straight to naive (else branch)."""
        layer = _make_layer(use_gemlite=False)
        x = torch.randn(_BATCH, _IN_F)
        naive_ref = torch.zeros(_BATCH, _OUT_F)

        with patch.object(method, "_apply_gemlite") as mock_gemlite:
            with patch.object(method, "_apply_naive", return_value=naive_ref):
                result = method.apply(layer, x, bias=None)

        mock_gemlite.assert_not_called()
        assert torch.equal(result, naive_ref)
        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is False

    def test_no_fallback_when_gemlite_succeeds(self, method, layer):
        x = torch.randn(_BATCH, _IN_F)
        gemlite_result = torch.zeros(_BATCH, _OUT_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(method, "_compute_parts", return_value=gemlite_result) as mock_parts:
                result = method.apply(layer, x, bias=None)

        mock_parts.assert_called_once()
        assert mock_parts.call_args.args[3] is True  # called with use_gemlite=True
        assert torch.equal(result, gemlite_result)
        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is False
        assert layer._dbf_use_gemlite is True
