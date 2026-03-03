import time
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import sys

def check_nbits(wr, nbits):
    (wr_vals, wr_counts) = torch.unique(wr, sorted=True, return_counts=True)
    assert (len(wr_vals) <= 2**nbits)
    return wr_counts


def hessian_loss(dw, H):
    return (dw @ H @ dw.T).trace()


def calc_entropy(wr_count):
    # empirical distribution of weights into bit patterns
    wr_dist = wr_counts / wr_counts.sum()
    print(wr_dist)
    # log(2) = 0.69... to convert from base e to bits
    print("avg bits per weight: %f" %
          (torch.special.entr(wr_dist) / 0.69314718056).sum().item())



def _allonce(x, w, unbiased=False):
    if unbiased:
        z = torch.floor(w - x + torch.rand(x.shape).to(x.device))
    else:
        z = torch.round(w - x)
    return w - z


def round_allbal(
    w,
    H,
    nbits,
    npasses,
    unbiased=False,
    calc_entropy=False,
):
    """
    w in [0,1]^{m,d}
    d: input_shape, m: output_shape
    """
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    wr = w
    s = torch.zeros(m, d).to(w.device)

    w_hat = wr.clone()
    # allonce   
    H = H / H.diag().max()

    prebern = None
    # if unbiased:
    #     prebern = 2 * (torch.rand(m) >= 0.5) - 1
    #     s.fill_(0.0)
    for ip in range(npasses):
        for i in range(d):
            Hs = s @ H[:, i]
            epsXTsj = _allonce(Hs / H[i,i], wr[:, i], unbiased=unbiased)
            wr[:, i] -= epsXTsj
            s[:, i] -= epsXTsj
        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if ((w_hat == wr).all()):
            sys.stderr.write(f"breaking after {ip+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)

    wr_counts = check_nbits(wr, nbits)
    # print(f"m:{m}, d:{d}")
    if calc_entropy:
        calc_entropy(wr_counts)
    return wr


def round_allbal_block(
    w,
    H,
    nbits,
    npasses,
    blocksize=128,
    unbiased=False,
    calc_entropy=False,
):
    """
    w in [0,1]^{m,d}
    d: input_shape, m: output_shape
    """
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    wr = w
    s = torch.zeros(m, d).to(w.device)

    w_hat = wr.clone()
    # allonce   
    H = H / H.diag().max()

    for ip in range(npasses):
        for i1 in range(0, d, blocksize):
            i2 = min(i1 + blocksize, d)
            count = i2 - i1
            W1 = wr[:, i1:i2].clone()
            S0 = s[:, :i1]
            S1 = s[:, i1:i2].clone()
            S2 = s[:, i2:]
            H0 = H[:i1, i1:i2]
            H1 = H[i1:i2, i1:i2]
            H2 = H[i2:, i1:i2]

            for i in range(count):
                Hs = S0 @ H0[:, i] + S1 @ H1[:, i] + S2 @ H2[:, i]
                epsXTsj = _allonce(Hs / H1[i,i], W1[:, i], unbiased=unbiased)
                W1[:, i] -= epsXTsj
                S1[:, i] -= epsXTsj

            wr[:, i1:i2] = W1
            s[:, i1:i2] = S1

        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if ((w_hat == wr).all()):
            sys.stderr.write(f"breaking after {ip+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)
        
    wr_counts = check_nbits(wr, nbits)
    # print(f"m:{m}, d:{d}")
    if calc_entropy:
        calc_entropy(wr_counts)
    return wr



def round_sorted_ldlqRG(
    w,
    H,
    nbits,
    n_greedy_passes=9,
    unbiased=False,
    pivot=None
):
    p = torch.argsort(torch.diag(H))
    Hp = H[p,:][:,p]
    wp = w[:,p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:,p] = round_ldl(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr


def round_ldl(
    w,
    H,
    nbits,
    n_greedy_passes=9,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    note: this has been updated with a (hopefully) more efficient LDL pass
    """
    assert (not unbiased) or (n_greedy_passes == 0), "greedy passes are incompatible with unbiased LDL rounding"
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1/torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in reversed(range(d)):
        w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w[:,i:] - w_hat[:,i:]) @ L[i:,i] + eta[:,i]), min=0, max=2**nbits-1)
    
    wr = w_hat.clone()
    s = w_hat - w
    H = H / H.diag().max()

    for igp in range(n_greedy_passes):
        for i in reversed(range(d)):
            Hs = s @ H[:, i]
            epsXTsj = wr[:, i] - torch.round(wr[:, i] - Hs / H[i,i])
            wr[:, i] -= epsXTsj
            s[:, i] -= epsXTsj
        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if ((w_hat == wr).all()):
            sys.stderr.write(f"breaking after {igp+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)
    
    wr_counts = check_nbits(wr, nbits)
    return wr


def round_sorted_ldlqRG_block(
    w,
    H,
    nbits,
    n_greedy_passes=9,
    unbiased=False,
    pivot=None
):
    p = torch.argsort(torch.diag(H))
    Hp = H[p,:][:,p]
    wp = w[:,p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:,p] = round_ldl_block(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr


def round_ldl_block(
    w,
    H,
    nbits,
    blocksize=128,
    n_greedy_passes=9,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    note: this has been updated with a (hopefully) more efficient LDL pass
    """
    assert (not unbiased) or (n_greedy_passes == 0), "greedy passes are incompatible with unbiased LDL rounding"
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1/torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i2 in range(d, 0, -blocksize):
        i1 = max(i2 - blocksize, 0)
        count = i2 - i1
        W1 = w[:, i1:i2]
        W2Hdiff = w[:, i2:] - w_hat[:, i2:]
        WHat1 = w_hat[:, i1:i2].clone()
        L1 = L[:, i1:i2]
        Eta1 = eta[:, i1:i2]

        for i in reversed(range(count)):
            WHat1[:,i] = torch.clamp(
                torch.floor(W1[:,i] + (W1 - WHat1) @ L1[i1:i2,i] + W2Hdiff @ L1[i2:,i] + Eta1[:,i]), 
                min=0, max=2**nbits-1)

        w_hat[:, i1:i2] = WHat1

    wr = w_hat.clone()
    s = w_hat - w
    H = H / H.diag().max()

    for igp in range(n_greedy_passes):
        for i2 in range(d, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            W1 = wr[:, i1:i2].clone()
            S0 = s[:, :i1]
            S1 = s[:, i1:i2].clone()
            S2 = s[:, i2:]
            H0 = H[:i1, i1:i2]
            H1 = H[i1:i2, i1:i2]
            H2 = H[i2:, i1:i2]

            for i in reversed(range(count)):
                Hs = S0 @ H0[:, i] + S1 @ H1[:, i] + S2 @ H2[:, i]
                epsXTsj = W1[:, i] - torch.round(W1[:, i] - Hs / H1[i,i])
                W1[:, i] -= epsXTsj
                S1[:, i] -= epsXTsj

            wr[:, i1:i2] = W1
            s[:, i1:i2] = S1

        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if ((w_hat == wr).all()):
            sys.stderr.write(f"breaking after {igp+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)
    
    wr_counts = check_nbits(wr, nbits)
    return wr

def round_sorted_ldl_admm(
    w,
    H,
    nbits,
    n_greedy_passes=9,
    unbiased=False,
    pivot=None
):
    p = torch.argsort(torch.diag(H))
    Hp = H[p,:][:,p]
    wp = w[:,p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:,p] = round_ldl_admm(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr

def ldlp_admm(H, rho = 0.1, niters = 100):
    n = H.shape[0]
    L = torch.linalg.cholesky(2 * H + rho * torch.eye(n,device=H.device))
    Linv = torch.linalg.inv(L)
    M = (torch.arange(n)[None,:] < torch.arange(n)[:,None]).to(H.device)
    MH = M * H
    X = torch.zeros(n,n,device=H.device)
    Z = torch.zeros(n,n,device=H.device)
    W = torch.zeros(n,n,device=H.device)
    for ii in range(100):
        X = (((rho * Z - rho * W - 2 * MH) @ Linv.T) * M) @ Linv
        C = torch.diag(1 / torch.max(torch.tensor(1.0,device=H.device),((X + W).T @ (X + W)).diag().sqrt()))
        Z = (X + W) @ C
        W = W + X - Z
        objective = ((Z + torch.eye(n,device=H.device)) @ H @ (Z + torch.eye(n,device=H.device)).T).trace()
        print(f"ldlp_admm iter {ii}: {objective}")
    return Z

def round_ldl_admm(
    w,
    H,
    nbits,
    n_greedy_passes=9,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    """
    # assert (not unbiased) or (n_greedy_passes == 0), "greedy passes are incompatible with unbiased LDL rounding"
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    H = H / H.diag().max()
    L = torch.linalg.inv(ldlp_admm(H) + torch.eye(d,device=H.device))
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = torch.floor(w + eta)
    for i in range(d):
        w_hat_next = torch.clamp(torch.floor(w_hat - (w_hat - w) @ L + eta), min=0, max=2**nbits - 1)
        if (w_hat_next == w_hat).all():
            w_hat = w_hat_next
            print(i)
            break
        w_hat = w_hat_next

    wr = w_hat
    Hn = H @ torch.diag(1/torch.diag(H))
    M = (torch.arange(d)[None,:] < torch.arange(d)[:,None]).to(Hn.device)
    HnM = Hn * M.to(Hn.device)
    objective = (((wr - w) @ H @ (wr - w).T).trace())
    print(f"objective before greedy passes: {objective}")

    for jj in range(n_greedy_passes):
        wr_target = w + (w - wr) @ (Hn * M.T)
        for ii in range(d):
            wr_prev = wr.clone()
            wr = torch.clamp(torch.round(wr_target + (w - wr) @ HnM), min=0, max=2**nbits - 1)
            if (wr == wr_prev).all():
                print(f"triangle-greedy finished after {ii+1} steps")
                break
        objective = (((wr - w) @ H @ (wr - w).T).trace())
        print(f"objective after {jj+1} greedy passes: {objective}")
        if (ii == 0):
            print(f"triangle-greedy finished after {jj+1} passes")
            break

    wr_counts = check_nbits(wr, nbits)
    return wr


def round_ldl_gptqequiv(
    w,
    H,
    nbits,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    """
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    H = torch.flip(H, [0,1])
    L = torch.linalg.cholesky(H)
    L = torch.flip(L,[0,1])
    L = L @ torch.diag(1/torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in range(d):
        #w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w - w_hat) @ L[:,i] + eta[:,i]), min=0, max=2**nbits-1)
        # optimized version
        w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w[:,:i+1] - w_hat[:,:i+1]) @ L[:i+1,i] + eta[:,i]), min=0, max=2**nbits-1)

    # reverse order
    # L = torch.linalg.cholesky(H)
    # L = L @ torch.diag(1/torch.diag(L))
    # if unbiased:
    #     eta = torch.rand(w.shape).to(w.device)
    # else:
    #     eta = 0.5 * torch.ones(w.shape).to(w.device)
    # w_hat = w.clone()
    # for i in reversed(range(d)):
    #     w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w[:,i:] - w_hat[:,i:]) @ L[i:,i] + eta[:,i]), min=0, max=2**nbits-1)

    wr = w_hat
    wr_counts = check_nbits(wr, nbits)
    return wr



def round_vecbal_Hsort(
    w,
    H,
    nbits,
    npasses,
    unbiased=False,
    qmethod='ldlq',
    lazy_batch=False
):
    """
    permute in order of diagonal of hessian, heuristic trick
    round with higher diag(H) first, corresponding vectors have larger magnitude
    then use smaller vectors to correct for larger vectors
    """
    # LDL has special Hsort function, opposite order
    if qmethod == 'ldlq':
        if lazy_batch is False:
            return round_ldl(w.float(),
                                    H,
                                    nbits=nbits,
                                    n_greedy_passes=npasses,
                                    unbiased=unbiased)
        else:
            return round_ldl_block(w.float(),
                                    H,
                                    nbits=nbits,
                                    n_greedy_passes=npasses,
                                    unbiased=unbiased)
    elif qmethod == 'ldlqRG':
        if lazy_batch is False:
            return round_sorted_ldlqRG(w.float(),
                                    H,
                                    nbits=nbits,
                                    n_greedy_passes=npasses,
                                    unbiased=unbiased)
        else:
            return round_sorted_ldlqRG_block(w.float(),
                                        H,
                                        nbits=nbits,
                                        n_greedy_passes=npasses,
                                        unbiased=unbiased)
    elif qmethod == 'ldlbal_admm':
        return round_sorted_ldl_admm(w,
                                   H,
                                   nbits=nbits,
                                   n_greedy_passes=npasses,
                                   unbiased=unbiased)
    elif qmethod == 'ldl_gptqequiv':
        return round_ldl_gptqequiv(w, H, nbits=nbits, unbiased=unbiased)
    else:
        Hdiag = H.diag()
        p = Hdiag.sort(descending=True).indices
        Hp = H[:, p][p, :]
        wp = w[:, p]
        if qmethod == 'allbal':
            if lazy_batch is False:
                wp_hat = round_allbal(wp,
                                    Hp,
                                    nbits=nbits,
                                    npasses=npasses,
                                    unbiased=unbiased)
            else:
                wp_hat = round_allbal_block(wp,
                                    Hp,
                                    nbits=nbits,
                                    npasses=npasses,
                                    unbiased=unbiased)
        # re-inverts order
        ip = torch.argsort(p)
        w_hat = wp_hat[:, ip]
        return w_hat


@torch.no_grad()
def quantize_weight_vecbal(w,
                            H,
                            nbits,
                            npasses,
                            scale, zero, maxq,
                            unbiased=False,
                            qfn='a',
                            qmethod='bitbal',
                            lazy_batch=False,
                            ):
    if (qfn == 'a') and (qmethod == 'ldl_gptqequiv'):
        wr = round_ldl_gptqequiv((w/scale) + zero, H, nbits=nbits)
        return scale * (wr - zero)
        # note: don't want to return wr.half() for comparison
    elif qfn == 'a':
        wr = torch.clamp((w/scale) + zero, 0, maxq)
        wr = round_vecbal_Hsort(
            wr, H, nbits, npasses, unbiased=unbiased, qmethod=qmethod, 
            lazy_batch=lazy_batch)
        wr = scale * (wr - zero)
        return wr.half()
    elif qfn == 'b':
        scale = 2.4 * w.square().mean().sqrt() + 1e-16
        wr = w / scale
        wr = torch.clamp(((wr+1)/2) * maxq, 0, maxq)
        wr = round_vecbal_Hsort(
            wr, H, nbits, npasses, unbiased=unbiased, qmethod=qmethod,
            lazy_batch=lazy_batch)
        wr = (wr / maxq) * 2 - 1
        wr = wr * scale
        return wr.half()
    else:
        return NotImplementedError()
