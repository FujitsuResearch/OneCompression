"""Add a GPTQ per-expert MoE weight-loading path to vLLM's gpt_oss model.

By default vLLM's ``GptOssModel.load_weights`` only understands MXFP4 (openai /
amd-quark), quark (fp8/mxfp4), and unquantized fused bf16 experts.  OneComp's
``mixed_gptq`` checkpoints that keep experts GPTQ-quantized (produced with
``Runner(..., moe_quant_experts=True)``) store per-expert tensors::

    model.layers.{L}.mlp.experts.{i}.gate_proj.{qweight,qzeros,scales,g_idx}
    model.layers.{L}.mlp.experts.{i}.up_proj.{qweight,qzeros,scales,g_idx}
    model.layers.{L}.mlp.experts.{i}.down_proj.{qweight,qzeros,scales,g_idx}

which the FusedMoE layer (served by ``GPTQMarlinMoEMethod`` via the mixed_gptq
plugin) exposes as fused ``w13_*`` / ``w2_*`` params.  This patch injects a
``_load_weights_gptq_moe`` method (modelled on vLLM's generic MoE loader) and
routes ``mixed_gptq`` checkpoints with GPTQ experts to it.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from vllm_plugins.patches._paths import vllm_file

MARKER = "# onecomp: gpt-oss gptq moe experts"
TARGET_REL = ("model_executor", "models", "gpt_oss.py")

# Anchor: the GptOssModel.load_weights method (identified by its qkv-only
# stacked_params_mapping).  We inject the new helpers immediately before it.
ANCHOR = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]"""

INJECT = """    def _has_gptq_experts(self) -> bool:
        # onecomp: gpt-oss gptq moe experts
        return any(
            n.endswith("experts.w13_qweight") for n, _ in self.named_parameters()
        )

    def _load_weights_gptq_moe(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights,
        stacked_params_mapping,
    ) -> set[str]:
        # onecomp: gpt-oss gptq moe experts
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
            num_redundant_experts=0,
        )
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        for name, loaded_weight in weights:
            if "sinks" in name:
                param = params_dict[name]
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts are handled below via expert_params_mapping.
                if "mlp.experts" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(ignore_suffixes) and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                is_expert_weight = False
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped, self):
                        continue
                    if mapped.endswith(ignore_suffixes) and mapped not in params_dict:
                        continue
                    param = params_dict[mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(mapped)
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params

"""

DISPATCH_OLD = """        else:
            return self._load_weights_other(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )"""

DISPATCH_NEW = """        elif quant_method == "mixed_gptq" and self._has_gptq_experts():
            # onecomp: gpt-oss gptq moe experts
            return self._load_weights_gptq_moe(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
        else:
            return self._load_weights_other(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )"""


def apply(*, dry_run: bool = False) -> str:
    target = vllm_file(*TARGET_REL)
    text = target.read_text()
    if MARKER in text:
        return f"already patched: {target}"
    if ANCHOR not in text:
        raise RuntimeError(f"GptOssModel.load_weights anchor not found in {target}")
    if DISPATCH_OLD not in text:
        raise RuntimeError(f"load_weights dispatch block not found in {target}")
    text = text.replace(ANCHOR, INJECT + ANCHOR, 1)
    text = text.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
    if not dry_run:
        target.write_text(text)
    return f"patched: {target}"
