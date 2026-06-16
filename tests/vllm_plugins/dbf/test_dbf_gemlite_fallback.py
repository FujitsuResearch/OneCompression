"""Copyright 2025-2026 Fujitsu Ltd."""

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


@pytest.fixture
def warning_spy(monkeypatch):
    """Capture WARNING messages by patching the module logger directly.

    More robust than caplog: it does not depend on the vLLM logger keeping
    propagate=True, so the assertions survive a future logger reconfiguration.
    """
    records: list[str] = []
    real_logger = _plugin_module.logger

    class _SpyLogger:
        def warning(self, msg, *args, **kwargs):
            records.append(msg % args if args else msg)

        def __getattr__(self, name):
            return getattr(real_logger, name)

    monkeypatch.setattr(_plugin_module, "logger", _SpyLogger())
    return records


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

    def test_fallback_warning_contains_env_vars(self, method, layer, warning_spy):
        """The fallback warning must mention both env var names."""
        x = torch.randn(_BATCH, _IN_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method, "_compute_parts", side_effect=_parts_side_effect_raise_on_gemlite
            ):
                method.apply(layer, x, bias=None)

        combined = " ".join(warning_spy)
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

    def test_disable_gemlite_runtime_logs_once(self, warning_spy):
        """_disable_gemlite_runtime() emits exactly one WARNING across multiple calls and sets the flag to True."""
        exc = RuntimeError("error")
        _disable_gemlite_runtime(exc)
        _disable_gemlite_runtime(exc)
        _disable_gemlite_runtime(exc)

        assert len(warning_spy) == 1
        message = warning_spy[0]
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
                    method.apply(
                        layer, x, bias=None
                    )  # second call: goes straight to naive via process-wide flag

        # GemLite was tried only on the first call; second call bypasses it via the process-wide flag
        assert gemlite_tries == 1

    def test_oom_is_not_caught(self, method, layer, warning_spy):
        """OOM is re-raised: GemLite is more memory-efficient than naive, so falling back on OOM would make things worse."""
        x = torch.randn(_BATCH, _IN_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(
                method,
                "_apply_gemlite",
                side_effect=torch.cuda.OutOfMemoryError("OOM"),
            ):
                with pytest.raises(torch.cuda.OutOfMemoryError):
                    method.apply(layer, x, bias=None)

        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is False
        assert layer._dbf_use_gemlite is True  # OOM must not touch the per-layer flag
        assert len(warning_spy) == 0  # OOM must not emit a WARNING

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

    def test_fused_fallback_matches_real_naive(self, method):
        """fused (part_count>1) fallback runs the REAL naive path and concatenates parts correctly.

        Exercises torch.cat(outputs, dim=-1), the per-part offset slicing of
        scaling/bp tensors, and the real _apply_naive numerics (only _apply_gemlite
        is forced to fail). This is the production-critical path: the layer that
        crashes under vLLM is the fused qkv_proj.
        """
        part_count = 3
        layer = _make_layer(part_count=part_count)

        # All sign bits set -> unpack_sign_bits yields +1 everywhere, so the
        # binary weight matrices are ones(mid, in) and ones(out, mid).
        layer.bp1 = torch.full_like(layer.bp1, 0xFF)
        layer.bp3 = torch.full_like(layer.bp3, 0xFF)
        # Distinct, non-unit per-part scalings to exercise offset slicing and to
        # catch a scaling/offset bug (uniform 1.0 would hide it). scaling0 is 2D
        # (part_count, in_features) as fused layers are built, so this also covers
        # the `scaling0.ndim == 2` per-part-index branch in _compute_parts().
        layer.scaling0 = torch.stack(
            [
                torch.linspace(0.5, 1.5, _IN_F, dtype=torch.float16) * (1.0 + 0.1 * p)
                for p in range(part_count)
            ]
        )
        layer.scaling2 = torch.linspace(0.5, 1.5, _MID_F * part_count, dtype=torch.float16)
        layer.scaling4 = torch.linspace(0.5, 1.5, _OUT_F * part_count, dtype=torch.float16)

        x = torch.randn(_BATCH, _IN_F)

        with patch(
            "vllm_plugins.dbf.vllm_plugin._try_import_gemlite",
            return_value=_MOCK_GEMLITE_IMPORT,
        ):
            with patch.object(method, "_apply_gemlite", side_effect=RuntimeError("kernel error")):
                result = method.apply(layer, x, bias=None)

        # Independent reference: +1 weight matrices, computed per part then concatenated.
        w1 = torch.ones(_MID_F, _IN_F)
        w3 = torch.ones(_OUT_F, _MID_F)
        expected_parts = []
        for p in range(part_count):
            s0 = layer.scaling0[p].float()
            s2 = layer.scaling2[p * _MID_F : (p + 1) * _MID_F].float()
            s4 = layer.scaling4[p * _OUT_F : (p + 1) * _OUT_F].float()
            h = (x * s0) @ w1.t()
            h = h * s2
            o = (h @ w3.t()) * s4
            expected_parts.append(o)
        expected = torch.cat(expected_parts, dim=-1)

        assert result.shape == (_BATCH, _OUT_F * part_count)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
        # Fallback still flips both the process-wide and the per-layer flags.
        assert _plugin_module._GEMLITE_RUNTIME_DISABLED is True
        assert layer._dbf_use_gemlite is False

    def test_bias_is_added_to_output(self, method):
        """apply() adds bias to the computed output."""
        layer = _make_layer(use_gemlite=False)  # else branch: naive only, no GemLite import
        x = torch.randn(_BATCH, _IN_F)
        base = torch.randn(_BATCH, _OUT_F)
        bias = torch.randn(_OUT_F)

        with patch.object(method, "_apply_naive", return_value=base):
            result = method.apply(layer, x, bias=bias)

        assert torch.allclose(result, base + bias)
