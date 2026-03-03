import time

import torch
import torch.nn as nn
import copy
from datasets import load_dataset
from transformers import AutoTokenizer 
import tqdm
from accelerate import infer_auto_device_map, dispatch_model
import re

from gptq import *
from modelutils import *
from quant import *
from resultutils import *
from awq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_tensor
from awq.quantize.pre_quant import run_awq
from qep_awq import run_awq_with_QEP
from zeroShot.utils import *
from zeroShot.main import get_result

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dev):
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
            # down_proj does not perform correction.
            if args.method == 'gptq' or args.method == 'quip' or names[0] != 'mlp.down_proj':
                for j in range(args.nsamples):
                    _ = layer(inps[j].unsqueeze(0), **layer_kwargs)
                    if not args.qep or names[0] == 'mlp.down_proj':
                        helper.add_batch(hook_data[name])
                    else:
                        _ = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)
                        helper.add_batch_qep(hook_data[name], hook_data_true[name])
            for h in handles:
                h.remove()

            # quantization
            for name, module in subset.items():
                if args.qep and name != 'mlp.down_proj':
                    helper.run_weight_correct(
                        module, percdamp=args.percdampqep, perccorr=args.perccorr
                    )
                if args.method == 'rtn':
                    module.weight.data = pseudo_quantize_tensor(
                        module.weight.data, n_bit=args.wbits, q_group_size=args.groupsize
                    )
                elif args.method == 'gptq':
                    helper.run_gptq(
                        module, percdamp=args.percdamp, wbits=args.wbits, groupsize=args.groupsize, actorder=args.act_order
                    )
                elif args.method == 'quip':
                    helper.run_quip(
                        module, percdamp=args.percdamp, wbits=args.wbits, multigpu=args.multigpu
                    )
                else:
                    raise NotImplementedError

            helper.free()

        # update inps
        for j in range(args.nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            inps_true[j] = layer_true(inps_true[j].unsqueeze(0), **layer_kwargs)[0]
        
        layers[i] = layer.cpu()
        del layer
        del layer_true
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

@torch.no_grad()
def llama_eval(model, testenc, dev):
    #print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': {}}

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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs']

    for i in range(len(layers)):
        #print(i)
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache

    return ppl.item()

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        'method', type=str, choices=['fp16', 'rtn', 'gptq', 'awq', 'quip'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--qep', action='store_true',
        help='Whether to use QEP.'
    )
    parser.add_argument(
        '--percdampqep', type=float, default=1.0,
        help='Percent of the average Hessian diagonal to use for dampening of QEP.'
    )
    parser.add_argument(
        '--perccorr', type=float, default=0.5,
        help='Percent of the weight correction.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--save-model', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load-model', type=str, default='',
        help='Load quantized checkpoint from this pass.'
    )

    # Evalution
    parser.add_argument(
        '--save-result', action='store_true',
        help='Whether to save result.'
    )
    parser.add_argument(
        "--ppl", default=None,
        choices=MultiChoice(['wikitext2', 'ptb-new', 'c4-new'])
    )
    parser.add_argument(
        "--tasks", default=None,
        choices=MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument(
        '--generate', type=str, default='',
        help='Write input for model generation. example:"Hello, how are you today?"'
    )
    parser.add_argument(
        '--multigpu', action='store_true',
        help='Whether to use multigpu.'
    )

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!

    model = get_llama(args.model)
    model.eval()

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    elif args.method != "fp16":
        print(f"method={args.method}", flush=True)
        if args.qep:
            print("Use QEP.", flush=True)
        print(args, flush=True)

        if args.method == 'rtn' or args.method == 'gptq' or args.method == 'quip':
            if args.method == 'rtn' and not args.qep:
                pseudo_quantize_model_weight(model, args.wbits, {"q_group_size":args.groupsize})
            else:
                if args.method == 'quip':
                    assert(args.groupsize == -1)
                llama_sequential(model, DEV)
        elif args.method == 'awq':
            enc = AutoTokenizer.from_pretrained(args.model, use_fast=False)
            if args.qep:
                run_awq_with_QEP(
                    model,
                    DEV,
                    args,
                    enc,
                    w_bit=args.wbits,
                    q_config={"q_group_size":args.groupsize},
                    n_samples=128,
                    seqlen=512,
                )
            else:
                run_awq(
                    model,
                    enc,
                    w_bit=args.wbits,
                    q_config={"q_group_size":args.groupsize},
                    n_samples=128,
                    seqlen=512,
                )
        else:
            raise NotImplementedError
    
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    
    # evalution
    results = {}
    if args.ppl is not None:
        datasets_names = args.ppl.split(",")
        model.seqlen = 2048
        for datasets_name in datasets_names:
            print(datasets_name, flush=True)
            dataloader, testloader = get_loaders(
                datasets_name, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            ppl = llama_eval(model, testloader, DEV)
            print(ppl, flush=True)
            results[f"ppl_{datasets_name}"] = ppl

    if args.tasks is not None:
        tasks_results = get_result(args, model)["results"]
        flattened_results = {f"{outer}_{inner}": value 
                        for outer, inner_dict in tasks_results.items() 
                        for inner, value in inner_dict.items()}
        flattened_results = dict(sorted(flattened_results.items()))
        results.update(flattened_results)
    
    if args.save_result:
        save_experiment_results(args, results)

    # Text Generation
    if args.generate:
        model = model.to(DEV)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        inputs = tokenizer(args.generate, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated text:\n", generated_text)
