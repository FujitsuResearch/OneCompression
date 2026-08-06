"""ROCm-only workaround for vLLM 0.24.x TritonW4A16LinearKernel.

Copyright 2025-2026 Fujitsu Ltd.

"""

from __future__ import annotations

_APPLIED_MARKER_AUTO_GPTQ = "_onecomp_vllm_0_24_0_rocm_applied_auto_gptq"
_APPLIED_MARKER_KERNEL = "_onecomp_vllm_0_24_0_rocm_applied_kernel"

# Set after the first inference-time fixup so logs stay readable (once per process).
_LOGGED_AUTO_GPTQ_QZEROS_FIXUP = False
_LOGGED_TRITON_QZEROS_PERMUTE = False

# vLLM only attaches handlers to the ``vllm`` logger tree; ``__name__`` logs are silently dropped.
# Use a child of ``vllm`` so INFO lines appear in engine output (including EngineCore subprocesses).
_VLLM_LOGGER_NAME = "vllm.onecomp_v0_24_0_rocm"


def _get_logger():
    from vllm.logger import init_logger

    return init_logger(_VLLM_LOGGER_NAME)


def _is_target_env() -> tuple[bool, str]:
    """Return ``(is_target, reason)`` for the current Python / vLLM env."""
    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        return False, f"vllm not importable ({exc})"
    try:
        from vllm.platforms import current_platform
    except ImportError as exc:
        return False, f"vllm.platforms not importable ({exc})"

    if not current_platform.is_rocm():
        return False, "not ROCm"

    # Accept "0.24.0", "0.24.0+rocm723", "0.24.1", "0.24.0rc3.dev3", etc.
    base_version = vllm.__version__.split("+", 1)[0]
    if not base_version.startswith("0.24."):
        return False, f"vllm base version {base_version!r} not in 0.24.x"

    return True, "vllm 0.24.x on ROCm"


def _unbias_v1_zeros(zp_packed):
    """Add 1 (mod 16) to every 4-bit nibble of a GPTQ-packed int32 tensor.

    Parameters
    ----------
    zp_packed
        ``[K // G, N // 8]`` int32 tensor.  GPTQ sequential packing: each
        int32 holds 8 consecutive N-values at bit offsets
        ``[0, 4, 8, ..., 28]``.

    Returns
    -------
    Same shape / dtype / device, with every nibble incremented by 1 mod 16.
    """
    import torch

    shifts = torch.arange(8, device=zp_packed.device, dtype=torch.int32) * 4
    nibbles = ((zp_packed.unsqueeze(-1) >> shifts) & 0xF) + 1
    nibbles &= 0xF
    return torch.sum(nibbles << shifts, dim=-1, dtype=torch.int32)


def _patch_auto_gptq_process_weights(logger) -> None:
    """Wrap ``AutoGPTQLinearMethod.process_weights_after_loading`` with a +1
    fixup on ``qzeros`` for the GPTQv1 convention.

    Rationale
    ---------
    GPTQv1 checkpoints store ``stored_zero = real_zero - 1``.  Marlin/Machete
    kernels re-add the ``-1`` bias internally, but ``TritonW4A16LinearKernel``
    consumes ``qzeros`` verbatim, so we need to add 1 (mod 16) to every
    packed nibble before the kernel sees the parameter.  This is scoped to
    the AutoGPTQ path (i.e. GPTQv1 checkpoints); other quantization methods
    that reuse ``TritonW4A16LinearKernel`` (e.g. compressed-tensors) do not
    have the ``-1`` convention and are left untouched.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        TritonW4A16LinearKernel,
    )
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQConfig,
        AutoGPTQLinearMethod,
    )
    from vllm.scalar_type import scalar_types

    # Additive: unblock asymmetric GPTQ checkpoints.  Stock vLLM 0.24 still
    # only registers (n_bits, True) in TYPE_MAP, which makes AutoGPTQConfig
    # .from_config raise ValueError for sym=False models.  TritonW4A16
    # already advertises uint4 (asymmetric) in SUPPORTED_QUANT_TYPES, so
    # this extension does not change kernel selection for sym=True.
    AutoGPTQConfig.TYPE_MAP.setdefault((4, False), scalar_types.uint4)
    AutoGPTQConfig.TYPE_MAP.setdefault((8, False), scalar_types.uint8)

    if getattr(
        AutoGPTQLinearMethod.process_weights_after_loading,
        _APPLIED_MARKER_AUTO_GPTQ,
        False,
    ):
        logger.debug("onecomp env-patch already installed on AutoGPTQLinearMethod; skip")
        return

    _orig = AutoGPTQLinearMethod.process_weights_after_loading

    def _patched(self, layer):
        global _LOGGED_AUTO_GPTQ_QZEROS_FIXUP
        # +1 fixup runs BEFORE the kernel's process_weights_after_loading so
        # that we operate on the AutoGPTQ-native layout [K//G, N//8] where
        # the packed axis is dim -1 (which is what ``_unbias_v1_zeros``
        # assumes).  The subsequent shape/permute juggling in the kernel
        # PWAL preserves per-nibble values.
        if isinstance(self.kernel, TritonW4A16LinearKernel):
            zp = getattr(layer, "qzeros", None)
            if zp is not None and getattr(zp, "data", None) is not None:
                zp.data = _unbias_v1_zeros(zp.data)
                if not _LOGGED_AUTO_GPTQ_QZEROS_FIXUP:
                    _LOGGED_AUTO_GPTQ_QZEROS_FIXUP = True
                    logger.info(
                        "onecomp env-patch active at inference: "
                        "AutoGPTQLinearMethod qzeros +1 fixup "
                        "(TritonW4A16LinearKernel, qzeros shape=%s)",
                        tuple(zp.data.shape),
                    )
        return _orig(self, layer)

    setattr(_patched, _APPLIED_MARKER_AUTO_GPTQ, True)
    AutoGPTQLinearMethod.process_weights_after_loading = _patched
    logger.info("onecomp env-patch: wrapped AutoGPTQLinearMethod.process_weights_after_loading")


def _patch_triton_w4a16_kernel_process_weights(logger) -> None:
    """Wrap ``TritonW4A16LinearKernel.process_weights_after_loading`` to fix
    a missing ``permute_param_layout_`` call on ``qzeros``.

    Rationale
    ---------
    In vLLM 0.24.x the stock ``TritonW4A16LinearKernel.process_weights_after_loading``
    normalizes ``qweight`` and ``scales`` via ``permute_param_layout_`` so
    that both compressed-tensors (``output_dim=0, packed_dim=0``) and
    AutoGPTQ (``input_dim=0, output_dim=1, packed_dim=1``) checkpoint
    layouts land in the same physical form before the ``.t().contiguous()``
    that the kernel expects.  The block for ``qzeros`` skips this step and
    calls ``.t()`` unconditionally::

        # Checkpoint: [N//8, K//G] int32 (N packed at dim 0, K//G at dim 1)
        # Kernel needs: [K//G, N//8] -- just transpose
        replace_parameter(
            layer, self.w_zp_name,
            torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
        )

    That works only for the compressed-tensors layout.  For an AutoGPTQ
    checkpoint the parameter is already ``[K//G, N//8]``, so the ``.t()``
    flips it to ``[N//8, K//G]`` and the downstream shape assertion in
    ``triton_w4a16_gemm`` fires (observed as
    ``AssertionError: qzeros shape mismatch: torch.Size([N//8, K//G])``).

    The one-liner upstream is missing is a ``permute_param_layout_(zp,
    input_dim=1, output_dim=0, packed_dim=0)`` call before the ``.t()``.
    Injecting it via a wrapper here fixes AutoGPTQ without regressing the
    compressed-tensors path (that path already satisfies the target
    layout, so the permute is a no-op).
    """
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        TritonW4A16LinearKernel,
    )
    from vllm.model_executor.parameter import permute_param_layout_

    if getattr(
        TritonW4A16LinearKernel.process_weights_after_loading,
        _APPLIED_MARKER_KERNEL,
        False,
    ):
        logger.debug("onecomp env-patch already installed on TritonW4A16LinearKernel; skip")
        return

    _orig = TritonW4A16LinearKernel.process_weights_after_loading

    def _patched(self, layer):
        global _LOGGED_TRITON_QZEROS_PERMUTE
        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Normalise both compressed-tensors and AutoGPTQ layouts to
                # [N//8, K//G] with (input_dim=1, output_dim=0, packed_dim=0)
                # so the stock code's subsequent zp.data.t().contiguous()
                # ends up at the kernel's expected [K//G, N//8].
                permute_param_layout_(zp, input_dim=1, output_dim=0, packed_dim=0)
                if not _LOGGED_TRITON_QZEROS_PERMUTE:
                    _LOGGED_TRITON_QZEROS_PERMUTE = True
                    logger.info(
                        "onecomp env-patch active at inference: "
                        "TritonW4A16LinearKernel qzeros permute_param_layout_ "
                        "(param=%s, shape=%s)",
                        self.w_zp_name,
                        tuple(zp.data.shape),
                    )
        return _orig(self, layer)

    setattr(_patched, _APPLIED_MARKER_KERNEL, True)
    TritonW4A16LinearKernel.process_weights_after_loading = _patched
    logger.info("onecomp env-patch: wrapped TritonW4A16LinearKernel.process_weights_after_loading")


def apply() -> None:
    """vLLM ``vllm.general_plugins`` entry point.

    No-op on every platform / version other than ROCm + vLLM 0.24.x.
    """
    logger = _get_logger()

    is_target, reason = _is_target_env()
    if not is_target:
        logger.debug("onecomp env-patch vllm_v0_24_0_rocm skipped: %s", reason)
        return

    import vllm

    logger.info(
        "onecomp env-patch: loading for %s (vllm %s)",
        reason,
        vllm.__version__,
    )
    _patch_auto_gptq_process_weights(logger)
    _patch_triton_w4a16_kernel_process_weights(logger)
    logger.info(
        "onecomp env-patch installed: "
        "AutoGPTQLinearMethod qzeros +1 fixup for TritonW4A16LinearKernel; "
        "TritonW4A16LinearKernel qzeros layout permute injected; "
        "AutoGPTQConfig.TYPE_MAP extended with asymmetric uint4/uint8.",
    )
