import math
import gc

import torch
import torch.nn as nn
import transformers

from quant import Quantizer, quantize


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import primefac
import scipy
import math
from vector_balance import quantize_weight_vecbal 
from quant_quip import QuantizerQFN

def butterfly_factors(n):
    pf = list(primefac.primefac(n))
    return (math.prod(pf[0::2]), math.prod(pf[1::2]))

def gen_rand_orthos(m,p):
    if (p != 2):
        return torch.tensor(scipy.stats.special_ortho_group.rvs(p, size=m)).to(torch.float32)
    X = torch.zeros(m,2,2)
    t = torch.rand(m) * (2 * math.pi) 
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    X[:,0,0] = cos_t
    X[:,1,1] = cos_t
    X[:,0,1] = sin_t
    X[:,1,0] = -sin_t
    return X

# generates a random orthogonal butterfly matrix of dimension n
def gen_rand_ortho_butterfly(n):
    return ([gen_rand_orthos(n//p, p) for p in butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

# generates a random orthogonal butterfly matrix of dimension n, without blocking
def gen_rand_ortho_butterfly_noblock(n):
    return ([gen_rand_orthos(1, p) for p in butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

# generates a random orthogonal butterfly matrix of dimension n, no permutation, but yes blocking
def gen_rand_ortho_butterfly_nopermute(n):
    return ([gen_rand_orthos(n//p, p) for p in butterfly_factors(n)], torch.arange(n), torch.arange(n))

# multiply by a random orthogonal butterfly matrix
def mul_ortho_butterfly(Bpp, x):
    (B, p_in, p_out) = Bpp
    assert((len(x.shape) == 1) or (len(x.shape) == 2))
    orig_dim = 2
    if (len(x.shape) == 1):
        (n,) = x.shape
        x = x.reshape(n,1)
        orig_dim = 1
    (n,q) = x.shape
    x = x[p_in,:]
    pfn = tuple(butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i+1):])
        x = x.reshape(mpfx, p, msfx, q).permute(0,2,1,3).reshape(mpfx * msfx, p, q)
        x = B[i] @ x
        x = x.reshape(mpfx, msfx, p, q).permute(0,2,1,3).reshape(n,q)
    x = x[p_out,:]
    if (orig_dim == 1):
        x = x.reshape(n)
    return x

# generates a random orthogonal butterfly matrix of dimension n
# and converts it to a dense matrix
def rand_ortho_butterfly(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly(n), torch.eye(n))

def rand_ortho_butterfly_noblock(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(n), torch.eye(n))

def rand_ortho_butterfly_nopermute(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))

class Helper:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        cols = self.layer.weight.data.shape[1]
        if isinstance(self.layer, transformers.Conv1D):
            cols = self.layer.weight.data.shape[0]
        self.H = torch.zeros((cols, cols), device=self.dev)
        self.H_delta = torch.zeros((cols, cols), device=self.dev)
        self.nsamples = 0
    
    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp_scaled = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp_scaled.matmul(inp_scaled.t())

    def add_batch_qep(self, inp, inp_true):
        delta = inp_true - inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            delta = delta.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                delta = delta.reshape((-1, delta.shape[-1]))
            inp = inp.t()
            delta = delta.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
            delta = unfold(delta)
            delta = delta.permute([1, 0, 2])
            delta = delta.flatten(1)
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.H_delta *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp_scaled = math.sqrt(2 / self.nsamples) * inp.float()
        delta_scaled = math.sqrt(2 / self.nsamples) * delta.float()
        self.H += inp_scaled.matmul(inp_scaled.t())
        self.H_delta += delta_scaled.matmul(inp_scaled.t())

    def free(self):
        del self.H
        del self.H_delta
        gc.collect()
        torch.cuda.empty_cache()

    def run_gptq(
        self, layer, blocksize=128, percdamp=.01, wbits=16, groupsize=-1, actorder=False
    ):
        quantizer = Quantizer()
        quantizer.configure(
            wbits, perchannel=True, sym=False, mse=False
        )

        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        H = self.H.clone()

        if not quantizer.ready():
            quantizer.find_params(W, weight=True)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        #Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, H.shape[0], blocksize):
            i2 = min(i1 + blocksize, H.shape[0])
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            #Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                #Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            #Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        if isinstance(layer, transformers.Conv1D):
            Q = Q.t()
        layer.weight.data = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype)

        del H, Hinv, W, Q
        gc.collect()
        torch.cuda.empty_cache()

    def run_weight_correct(
        self, layer, percdamp=.01, perccorr=.25
    ):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        H = self.H.clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = H

        # 重み補正
        W += (W @ self.H_delta @ Hinv) * perccorr

        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype)

        del H, Hinv, W
        gc.collect()
        torch.cuda.empty_cache()

    def run_quip(self, layer, percdamp=.01, wbits=16, multigpu=False,
                preproc_gptqH=True, preproc_rescale=True, preproc_proj=True, preproc_proj_extra=1,
                qmethod='ldlq', npasses=0, unbiased=False, lazy_batch=False):
        quantizer = QuantizerQFN()
        quantizer.configure(
            wbits, perchannel=True, sym=False, qfn='b', mse=False
        )

        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        H = self.H.clone()

        if multigpu:
            W = W.to("cuda:1")
            H = H.to("cuda:1")

        # preproc
        if preproc_rescale:
            H /= H.abs().max()
            diagH = torch.diag(H)
            diagW2 = torch.diag(W.T @ W)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            W *= scaleWH[None,:]
            H /= scaleWH[None,:]
            H /= scaleWH[:,None]
            scaleWH = scaleWH.cpu()
        if preproc_proj:
            if preproc_proj_extra == 0:
                U = rand_ortho_butterfly(W.shape[0]).to(torch.float32).to(W.device)
                V = rand_ortho_butterfly(W.shape[1]).to(torch.float32).to(W.device)
            elif preproc_proj_extra == 1:
                U = rand_ortho_butterfly_noblock(W.shape[0]).to(torch.float32).to(W.device)
                V = rand_ortho_butterfly_noblock(W.shape[1]).to(torch.float32).to(W.device)
            elif preproc_proj_extra == 2:
                U = rand_ortho_butterfly_nopermute(W.shape[0]).to(torch.float32).to(W.device)
                V = rand_ortho_butterfly_nopermute(W.shape[1]).to(torch.float32).to(W.device)
            H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(H.shape[0], device=W.device)
            W = U @ W @ V.T
            H = V @ H @ V.T
            U = U.cpu()
            V = V.cpu()
        # H modification from gptq
        if preproc_gptqH:
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[0], device=H.device)
            H[diag, diag] += damp
        
        # quantization
        W = quantize_weight_vecbal(
            w=W, H=H,
            nbits=wbits,
            npasses=npasses,
            scale=quantizer.scale,
            zero=quantizer.zero,
            maxq=quantizer.maxq,
            unbiased=unbiased,
            qfn=quantizer.qfn,
            qmethod=qmethod,
            lazy_batch=lazy_batch
        ).float()

        # postproc
        if preproc_proj:
            U = U.to(W.device)
            V = V.to(W.device)
            W = (U.T @ W @ V)
            H = (V.T @ H @ V)
        if preproc_rescale:
            scaleWH = scaleWH.to(W.device)
            W = W / scaleWH[None,:]
            H = H * scaleWH[:,None]
            H = H * scaleWH[None,:]
        
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype).to(layer.weight.data.device)

        del H, W, U, V, scaleWH
        gc.collect()
        torch.cuda.empty_cache()
