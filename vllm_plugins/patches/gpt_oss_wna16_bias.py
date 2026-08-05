"""Add per-expert bias support to vLLM's WNA16 (int4/int8) fused-MoE kernel.

Copyright 2025-2026 Fujitsu Ltd.

gpt-oss experts carry gate/up/down bias, but vLLM's grouped-weight WNA16 path
(``fused_moe_kernel_gptq_awq`` / ``invoke_fused_moe_wna16_triton_kernel``) has
no bias support and ``dispatch_fused_moe_kernel`` even asserts ``B_bias is
None``.  This source patch:

  * extends the ``fused_moe_kernel_gptq_awq`` Triton kernel with a
    ``b_bias_ptr`` (+ strides + ``HAS_BIAS`` constexpr) and adds the bias to
    the fp32 accumulator after dequantization and before the routing-weight
    multiply (matching the unquantized ``fused_moe_kernel`` ordering),
  * threads ``B_bias`` through ``invoke_fused_moe_wna16_triton_kernel``, and
  * removes the ``assert B_bias is None`` in ``dispatch_fused_moe_kernel`` and
    forces the Triton path (the compiled CUDA WNA16 kernel has no bias) when a
    bias tensor is present.

Only the ``fc1``/``fc2`` GEMMs of a bias-carrying WNA16 layer are affected;
callers that pass no bias keep ``HAS_BIAS=False`` and are unchanged.
"""

from __future__ import annotations

from vllm_plugins.patches._paths import vllm_file

MARKER = "# onecomp: gpt-oss expert bias (WNA16)"
TARGET_REL = ("model_executor", "layers", "fused_moe", "fused_moe.py")

# 1) Kernel signature: add b_bias_ptr after c_ptr.
SIG_PTR_OLD = """def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,"""
SIG_PTR_NEW = """def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,"""

# 2) Kernel signature: add bias strides after the zero-point strides.
SIG_STRIDE_OLD = """    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,"""
SIG_STRIDE_NEW = """    stride_bze,
    stride_bzk,
    stride_bzn,
    stride_bbe,
    stride_bbn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,"""

# 3) Kernel signature: add HAS_BIAS constexpr at the end.
SIG_FLAG_OLD = """    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):"""
SIG_FLAG_NEW = """    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):"""

# 4) Kernel body: add bias to the accumulator before the routing-weight mul.
BODY_OLD = """        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)"""
BODY_NEW = """        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # onecomp: gpt-oss expert bias (WNA16) -- added after dequant, before the
    # routing-weight multiply so it matches the unquantized fused_moe_kernel.
    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
        accumulator = accumulator + bias[None, :].to(tl.float32)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)"""

# 5) Wrapper signature: accept an optional B_bias tensor.
WRAP_SIG_OLD = """    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0"""
WRAP_SIG_NEW = """    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
    B_bias: torch.Tensor | None = None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0"""

# 6) Wrapper: pass B_bias as the kernel's 4th positional argument.
WRAP_CALL_OLD = """    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,"""
WRAP_CALL_NEW = """    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_bias,
        B_scale,
        B_zp,
        topk_weights,"""

# 7a) Wrapper: the CUDA-branch default config omits GROUP_SIZE_M, but the
# (forced) triton wna16 kernel requires it, so guarantee a safe default.
WRAP_CFG_OLD = """            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    fused_moe_kernel_gptq_awq[grid]("""
WRAP_CFG_NEW = """            block_size_m=config["BLOCK_SIZE_M"],
        )
    )
    # onecomp: the CUDA-branch default config omits GROUP_SIZE_M, but the
    # (forced) triton wna16 kernel requires it; ensure safe defaults here.
    config.setdefault("GROUP_SIZE_M", 1)
    config.setdefault("SPLIT_K", 1)

    fused_moe_kernel_gptq_awq[grid]("""

# 7b) Wrapper: pass bias strides + HAS_BIAS to the kernel.
WRAP_STRIDE_OLD = """        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )"""
WRAP_STRIDE_NEW = """        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        HAS_BIAS=B_bias is not None,
        **config,
    )"""

# 8) Dispatch: drop the bias assert and skip the CUDA kernel when bias present.
DISPATCH_GUARD_OLD = """    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8,
        )"""
DISPATCH_GUARD_NEW = """    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        # onecomp: the compiled WNA16 CUDA kernel has no bias support, so force
        # the (patched) Triton path whenever expert bias is present.
        use_moe_wna16_cuda = B_bias is None and should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.size(0),
            bit=4 if use_int4_w4a16 else 8,
        )"""

# 9) Dispatch: forward B_bias into the WNA16 triton wrapper.
DISPATCH_CALL_OLD = """        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )

    else:"""
DISPATCH_CALL_NEW = """        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
            B_bias,
        )

    else:"""

# 10) fused_experts_impl: the CUDA-branch default config uses
# BLOCK_SIZE_M=min(16, M), which can be non-power-of-2 (e.g. a 6-token prompt).
# The forced Triton wna16 kernel does tl.arange(0, BLOCK_SIZE_M) and requires a
# power-of-2 tile, so round it up here -- before token alignment -- whenever an
# expert bias is present, keeping the padding and kernel grid consistent.
IMPL_CFG_OLD = """    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty("""
IMPL_CFG_NEW = """    config = get_config_func(M)

    # onecomp: gpt-oss expert bias (WNA16) forces the Triton wna16 kernel, whose
    # tl.arange(0, BLOCK_SIZE_M) needs a power-of-2 tile; round up the (possibly
    # non-pow2) CUDA-branch default before token alignment.
    if w1_bias is not None:
        _bsm = config.get("BLOCK_SIZE_M", 16)
        if _bsm & (_bsm - 1) != 0:
            config = dict(config)
            config["BLOCK_SIZE_M"] = 1 << (_bsm - 1).bit_length()

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty("""

_REPLACEMENTS = (
    ("kernel bias pointer", SIG_PTR_OLD, SIG_PTR_NEW),
    ("kernel bias strides", SIG_STRIDE_OLD, SIG_STRIDE_NEW),
    ("kernel HAS_BIAS flag", SIG_FLAG_OLD, SIG_FLAG_NEW),
    ("kernel bias add", BODY_OLD, BODY_NEW),
    ("wrapper signature", WRAP_SIG_OLD, WRAP_SIG_NEW),
    ("wrapper kernel call", WRAP_CALL_OLD, WRAP_CALL_NEW),
    ("wrapper group_size default", WRAP_CFG_OLD, WRAP_CFG_NEW),
    ("wrapper bias strides", WRAP_STRIDE_OLD, WRAP_STRIDE_NEW),
    ("dispatch guard", DISPATCH_GUARD_OLD, DISPATCH_GUARD_NEW),
    ("dispatch call", DISPATCH_CALL_OLD, DISPATCH_CALL_NEW),
    ("impl block-size pow2", IMPL_CFG_OLD, IMPL_CFG_NEW),
)


def apply(*, dry_run: bool = False) -> str:
    target = vllm_file(*TARGET_REL)
    text = target.read_text()
    if MARKER in text:
        return f"already patched: {target}"
    for label, old, new in _REPLACEMENTS:
        if old not in text:
            raise RuntimeError(f"{label} block not found in {target}")
        if text.count(old) != 1:
            raise RuntimeError(
                f"{label} block is ambiguous ({text.count(old)} matches) in {target}"
            )
        text = text.replace(old, new, 1)
    if not dry_run:
        target.write_text(text)
    return f"patched: {target}"
