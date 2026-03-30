"""Stiefel-manifold SGD optimizer (SGDG) for rotation training.

Provides helper functions for Cayley retraction and QR retraction on the
Stiefel manifold, along with the ``SGDG`` optimizer that handles both
Stiefel (orthogonal) and Euclidean parameter groups in a single step.

Adapted from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

import random

import torch
from torch.optim.optimizer import Optimizer, required


def unit(v, dim: int = 1, eps: float = 1e-8):
    """Normalize *v* along *dim* and return the unit vector with its norm.

    Args:
        v: 2-D tensor.
        dim: Dimension along which to normalize.
        eps: Small constant added to the norm for numerical safety.

    Returns:
        tuple: ``(unit_vector, norm)`` where ``norm`` has shape broadcastable to ``v``.
    """
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = 1):
    """L2 norm of a 2-D tensor along *dim* with ``keepdim=True``.

    Args:
        v: 2-D tensor.
        dim: Dimension along which to compute the norm.

    Returns:
        torch.Tensor: L2 norms with the reduced dimension kept as size 1.
    """
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    """Compute the matrix 1-norm (max absolute column sum).

    Args:
        W: 2-D tensor.

    Returns:
        Scalar tensor: max over columns of the sum of absolute values.
    """
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def cayley_loop(X, W, tan_vec, t):
    """Cayley retraction via fixed-point iteration (5 steps).

    Args:
        X: Current point on the Stiefel manifold, shape ``(n, p)``.
        W: Skew-symmetric matrix driving the retraction, shape ``(n, n)``.
        tan_vec: Tangent vector, shape ``(n, p)``.
        t: Step size scalar.

    Returns:
        torch.Tensor: Retracted point, shape ``(p, n)`` (transposed).
    """
    [n, p] = X.size()
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


def qr_retraction(tan_vec):
    """Re-orthogonalise a matrix via QR decomposition with sign correction.

    Used to correct accumulated numerical drift of points on the Stiefel
    manifold back to exact orthogonality.

    Args:
        tan_vec: Matrix of shape ``(p, n)`` with ``p <= n``.
            Modified in-place (transposed internally).

    Returns:
        torch.Tensor: Orthonormal matrix of shape ``(p, n)``.
    """
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()

    return q


epsilon = 1e-8


class SGDG(Optimizer):
    """SGD on the Stiefel manifold (SGD-G) with optional Euclidean fallback.

    When ``stiefel=True``, parameters are treated as points on the Stiefel
    manifold and updated via Cayley retraction with momentum.  When
    ``stiefel=False``, standard SGD with optional weight decay, dampening,
    and Nesterov momentum is used.

    Args:
        params (iterable): Parameters to optimize or dicts defining groups.
        lr (float): Learning rate.
        momentum (float, optional): Momentum factor. Defaults to 0.
        stiefel (bool, optional): Use SGD-G for orthogonal updates.
            Defaults to False.
        weight_decay (float, optional): L2 penalty (Euclidean mode).
            Defaults to 0.
        dampening (float, optional): Dampening for momentum (Euclidean mode).
            Defaults to 0.
        nesterov (bool, optional): Nesterov momentum (Euclidean mode).
            Defaults to False.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        stiefel: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)
                    g = p.grad.data.view(p.size()[0], -1)
                    lr = group["lr"]
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.t().size(), device=p.device)

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + epsilon)
                    alpha = min(t, lr)
                    p_new = cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)
                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p = d_p.add(p.data, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    p.data.add_(d_p, alpha=-group["lr"])

        return loss
