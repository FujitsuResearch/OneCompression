"""
Tests for KL loss computation used by GlobalPTQ.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    pytest tests/onecomp/post_process/test_global_ptq_losses.py -v
"""

import pytest
import torch


class TestComputeKlLoss:
    """Tests for compute_kl_loss."""

    def test_identical_logits_give_zero_loss(self):
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        logits = torch.randn(2, 10, 100)
        loss = compute_kl_loss(logits, logits.clone(), temperature=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_logits_give_positive_loss(self):
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        teacher = torch.randn(2, 10, 100)
        student = torch.randn(2, 10, 100)
        loss = compute_kl_loss(teacher, student, temperature=1.0)
        assert loss.item() > 0

    def test_temperature_scaling(self):
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        teacher = torch.randn(2, 10, 50)
        student = teacher + 0.1 * torch.randn_like(teacher)
        loss_t1 = compute_kl_loss(teacher, student, temperature=1.0)
        loss_t2 = compute_kl_loss(teacher, student, temperature=2.0)
        assert loss_t1.item() != loss_t2.item()

    def test_gradient_flows_through_student(self):
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        teacher = torch.randn(2, 5, 30)
        student = torch.randn(2, 5, 30, requires_grad=True)
        loss = compute_kl_loss(teacher, student, temperature=1.0)
        loss.backward()
        assert student.grad is not None

    def test_float16_matches_float32_reference(self):
        """KL loss computed from float16 logits should closely match float32 reference.

        Real training passes float16 logits (from quantised model output).
        If the loss function does not upcast internally, softmax/KL in
        float16 introduces significant numerical error for large vocabs.
        """
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        torch.manual_seed(42)
        teacher = torch.randn(1, 8, 32000)
        student = teacher + 0.5 * torch.randn_like(teacher)

        ref = compute_kl_loss(teacher, student, temperature=1.0).item()
        fp16 = compute_kl_loss(teacher.half(), student.half(), temperature=1.0).float().item()

        rel_err = abs(ref - fp16) / max(abs(ref), 1e-10)
        assert rel_err < 0.01, (
            f"float16 KL ({fp16:.6f}) deviates {rel_err*100:.1f}% from "
            f"float32 reference ({ref:.6f}); loss function should upcast to float32"
        )

    def test_no_nan_with_float16_inputs(self):
        """KL loss must not produce NaN/Inf when given float16 logits."""
        from onecomp.post_process._global_ptq.losses import compute_kl_loss

        torch.manual_seed(0)
        teacher = torch.randn(1, 4, 32000).half()
        student = torch.randn(1, 4, 32000).half()
        loss = compute_kl_loss(teacher, student, temperature=1.0)
        assert not torch.isnan(loss), "KL loss is NaN with float16 inputs"
        assert not torch.isinf(loss), "KL loss is Inf with float16 inputs"
