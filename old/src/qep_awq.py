import torch
import torch.nn as nn
import copy
import tqdm
import functools
from collections import defaultdict

from datautils import *
from gptq import *
from modelutils import *
from quant import *
from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.auto_clip import auto_clip_layer
from awq.utils.module import get_op_name, get_op_by_name, set_op_by_name
from awq.quantize.auto_scale import get_act_scale
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from awq.quantize.qmodule import ScaledActivation
from awq.quantize.auto_scale import scale_fc_fc, scale_gelu_fc, scale_ln_fcs

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

@torch.no_grad()
def auto_get_scale(module, prev_op, layers, inp, w_bit, q_config, module2inspect=None, kwargs={}):
    def w_quantize_func(p):
        return pseudo_quantize_tensor(
            p,
            n_bit=w_bit,
            **q_config,
        ).detach()
    
    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # module2inspect: if given, we will check the output diff of this module instead of layers
    if module2inspect is None:
        assert len(layers) == 1
        module2inspect = layers[0]

    scales = _search_module_scale(module2inspect, layers, inp, kwargs)
    scales = scales.detach().cpu()
    # prev_op_name, [layer_name], scale
    return (
        get_op_name(module, prev_op),
        tuple([get_op_name(module, m) for m in layers]),
        scales,
    )

def apply_scale_config(module, scale_config, input_feat_dict=None):
    prev_op_name, layer_names, scales = scale_config
    prev_op = get_op_by_name(module, prev_op_name)
    layers = [get_op_by_name(module, name) for name in layer_names]

    # prev_op.cuda()
    # for layer in layers:
    #     layer.cuda()
    scales.cuda()

    if isinstance(prev_op, nn.Linear):
        assert len(layers) == 1
        scale_fc_fc(prev_op, layers[0], scales)
    elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm)):
        scale_ln_fcs(prev_op, layers, scales)
    elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation, nn.SiLU)):
        new_module = ScaledActivation(prev_op, scales)
        set_op_by_name(module, prev_op_name, new_module)
        scale_gelu_fc(prev_op, layers[0], scales)
    else:
        raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

    # apply the scaling to input feat if given; prepare it for clipping
    if input_feat_dict is not None:
        for layer_name in layer_names:
            inp = input_feat_dict[layer_name]
            inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))

    # prev_op.cpu()
    # for layer in layers:
    #     layer.cpu()
    scales.cpu()

@torch.no_grad()
def run_awq_with_QEP(model, dev,
    args,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",):

    # AWQ Calibation Datasets
    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps_AWQ = []
    layer_kwargs_AWQ = {}

    # QEP Calibation Datasets
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': {}}

    # AWQ Data
    class CatcherAWQ(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps_AWQ.append(inp)
            layer_kwargs_AWQ.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = CatcherAWQ(layers[0])
    try:
       model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps_AWQ = inps_AWQ[0]

    # QEP Data
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'].update(kwargs)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    layer_kwargs = cache['layer_kwargs']
    inps_true = inps.clone()

    sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj']
    ]

    for i in tqdm.tqdm(range(len(layers)), desc=f'Running {args.method}'):
        # prepare for AWQ
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps_AWQ = inps_AWQ.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps_AWQ = layer(inps_AWQ, **layer_kwargs_AWQ)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        # prepare for QEP
        layer = layers[i].to(dev)
        layer_true = copy.deepcopy(layers[i]).to(dev)
        full = find_layers(layer)
        full_true = find_layers(layer_true)
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            subset_true = {n: full_true[n] for n in names}
            hook_data = {}
            hook_data_true = {}

            def make_quant_hook(name):
                def hook(module, inp, out):
                    hook_data[name] = inp[0].detach().clone()
                return hook

            def make_true_hook(name):
                def hook(module, inp, out):
                    hook_data_true[name] = inp[0].detach().clone()
                return hook

            handles = []
            for name, module in subset.items():
                handles.append(module.register_forward_hook(make_quant_hook(name)))
            for name, module in subset_true.items():
                handles.append(module.register_forward_hook(make_true_hook(name)))

            helper = Helper(subset[names[0]])
            # down_proj does not perform  correction.
            if names[0] != 'mlp.down_proj':
                for j in range(args.nsamples):
                    _ = layer(inps[j].unsqueeze(0), **layer_kwargs)
                    _ = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)
                    helper.add_batch_qep(hook_data[name], hook_data_true[name])
            for h in handles:
                h.remove()

            for name, module in subset.items():
                if name != 'mlp.down_proj':
                    helper.run_weight_correct(
                        module, percdamp=args.percdampqep, perccorr=args.perccorr
                    )
            helper.free()
                
            # scale
            if auto_scale:
                if names == ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj']:
                    scale_config = auto_get_scale(
                        module=layer,
                        prev_op=layer.input_layernorm,
                        layers=[
                            layer.self_attn.q_proj,
                            layer.self_attn.k_proj,
                            layer.self_attn.v_proj,
                        ],
                        inp=input_feat["self_attn.q_proj"],
                        w_bit=w_bit,
                        q_config=q_config,
                        module2inspect=layer.self_attn,
                        kwargs=layer_kwargs_AWQ,
                    )
                if names == ['self_attn.o_proj'] and layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
                    scale_config = auto_get_scale(
                        module=layer,
                        prev_op=layer.self_attn.v_proj,
                        layers=[layer.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                        w_bit=w_bit,
                        q_config=q_config,
                    )
                if names == ['mlp.up_proj', 'mlp.gate_proj']:
                    scale_config = auto_get_scale(
                        module=layer,
                        prev_op=layer.post_attention_layernorm,
                        layers=[layer.mlp.gate_proj, layer.mlp.up_proj],
                        inp=input_feat["mlp.gate_proj"],
                        w_bit=w_bit,
                        q_config=q_config,
                        module2inspect=layer.mlp,
                    )
                if names == ['mlp.down_proj']:
                    scale_config = auto_get_scale(
                        module=layer,
                        prev_op=layer.mlp.up_proj,
                        layers=[layer.mlp.down_proj],
                        inp=input_feat["mlp.down_proj"],
                        w_bit=w_bit,
                        q_config=q_config,
                    )
                apply_scale_config(layer, scale_config, input_feat_dict=input_feat)

            for name, module in subset.items():
                # clip
                if mse_range:
                    if not any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                        max_val = auto_clip_layer(
                            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
                        )
                        max_val = max_val.to(named_linears[name].weight.device).to(named_linears[name].weight.dtype)
                        org_shape = named_linears[name].weight.shape
                        named_linears[name].weight.data = named_linears[name].weight.data.reshape(*max_val.shape[:2], -1)
                        named_linears[name].weight.data = torch.clamp(named_linears[name].weight.data, -max_val, max_val)
                        named_linears[name].weight.data = named_linears[name].weight.data.reshape(org_shape)

                # RTN
                module.weight.data = pseudo_quantize_tensor(
                    module.weight.data, n_bit=args.wbits, q_group_size=args.groupsize
                )

        # update inps
        for j in range(args.nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            inps_true[j] = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)[0]
        
        layers[i] = layer.cpu()
        del layer
        del layer_true
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache