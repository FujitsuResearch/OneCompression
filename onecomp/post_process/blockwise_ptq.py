"""
Block-wise Post-Training Quantization process.

Ported from qep-dev/src/blockwise_quantization/run_blockwise_ptq.py
and adapted for onecomp-lab's PostQuantizationProcess interface.

Key differences from qep-dev:
  - Entry point: BlockWisePTQ.run(quantized_model, model_config)
    instead of run_blockwise_quantization(model, cfg, dev).
  - Config: dataclass fields instead of cfg.blockwise_ptq / cfg.method.
  - Method detection: auto-detected from layer types, not cfg.method.
  - Teacher model: model_config.load_model(device_map="cpu")
    instead of get_model(cfg.model).
  - Calibration: prepare_calibration_dataset() instead of get_loaders().

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc
from dataclasses import dataclass
from logging import getLogger

import torch
import torch.nn as nn

from ..calibration import CalibrationConfig
from ..model_config import ModelConfig
from ._base import PostQuantizationProcess


@dataclass
class BlockWisePTQ(PostQuantizationProcess):
    """Block-wise Post-Training Quantization

    After layer-wise PTQ (GPTQ / DBF / OneBit) quantises each linear layer
    independently, block-wise PTQ minimises intermediate-representation
    MSE against an FP16 teacher model at the Transformer-block
    granularity.

    Args:
        lr (float):
            Learning rate for block-wise optimisation (DBF / OneBit / generic).
            Default is 1e-4.
        epochs (int):
            Number of optimisation epochs per block.  Default is 10.
        cbq_enable (bool):
            Whether to enable Cross-Block Quantisation (Phase 2) after
            greedy block-wise distillation.  Default is False.
        gptq_lr (float):
            Learning rate for GPTQ scales/zeros optimisation.
            Default is 1e-3.
        gptq_optimize_intweight (bool):
            Whether to optimise integer weights via Smooth STE.
            Default is False.
        gptq_intweight_lr (float):
            Learning rate for integer weight optimisation.
            Default is 1e-4.
        grad_clip (float):
            Gradient clipping norm.  Default is 1.0.
        optimize_binary (bool):
            Whether to optimise binary matrices (DBF) / sign matrices (OneBit).
            Default is True.
        k_smooth (float):
            SmoothSign STE temperature for binary/sign optimisation.
            Default is 100.0.
        calibration_config (CalibrationConfig or None):
            Calibration data configuration.  When ``None`` (default),
            a :class:`CalibrationConfig` is created with
            ``num_calibration_samples=128``.
            See :class:`~onecomp.calibration.CalibrationConfig`.

    Examples:
        >>> from onecomp import Runner, ModelConfig, GPTQ, BlockWisePTQ
        >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[BlockWisePTQ(lr=1e-4, epochs=10, cbq_enable=True)],
        ... )
        >>> runner.run()

    """

    lr: float = 1e-4
    epochs: int = 10
    cbq_enable: bool = False

    # GPTQ-specific (qep-dev: cfg.blockwise_ptq.gptq_lr etc.)
    gptq_lr: float = 1e-3
    gptq_optimize_intweight: bool = False
    gptq_intweight_lr: float = 1e-4
    grad_clip: float = 1.0

    # DBF / OneBit specific (qep-dev: cfg.blockwise_ptq.optimize_binary etc.)
    optimize_binary: bool = True
    k_smooth: float = 100.0

    # CBQ (Phase 2) specific (qep-dev: cfg.blockwise_ptq.cbq_*)
    cbq_epochs: int = 0
    cbq_lr: float = 5e-5

    # Calibration (qep-dev: cfg.dataset / cfg.nsamples via get_loaders())
    calibration_config: CalibrationConfig = None

    def run(
        self,
        quantized_model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """Execute block-wise PTQ on the quantized model.

        Modifies *quantized_model* in-place.  Returns None
        (per PostQuantizationProcess interface).

        Args:
            quantized_model (nn.Module):
                Quantized model on CPU.
            model_config (ModelConfig):
                Model configuration.
        """
        logger = getLogger(__name__)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.calibration_config is None:
            self.calibration_config = CalibrationConfig(
                calibration_dataset="c4",
                max_length=2048,
                num_calibration_samples=128,
                strategy="drop_rand",
                seed=0,
            )
        # ------------------------------------------------------------------
        # 1. Calibration data
        # ------------------------------------------------------------------
        # onecomp: prepare_calibration_dataset()
        #   returns {"input_ids": (N, seq_len), "attention_mask": (N, seq_len)}
        # qep-dev: get_loaders() returns list of (input_ids, labels)
        from ..calibration import prepare_calibration_dataset

        tokenizer = model_config.load_tokenizer()
        calibration_inputs = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device="cpu",
            calibration_config=self.calibration_config,
            model=quantized_model,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # 2. Get transformer blocks
        # ------------------------------------------------------------------
        from ._blockwise.helpers import (
            get_transformer_layers,
            collect_layer_inputs,
            layer_kwargs_to_device,
            layer_forward_single,
            deep_clone_layer_kwargs,
            auto_detect_quantization_strategy,
        )

        layers = get_transformer_layers(quantized_model)
        total_layers = len(layers)
        logger.info("Found %d transformer blocks", total_layers)

        # ------------------------------------------------------------------
        # 3. Collect layer inputs via Catcher hook
        # ------------------------------------------------------------------
        logger.info("Collecting layer inputs...")
        inps, layer_kwargs = collect_layer_inputs(
            quantized_model,
            layers,
            calibration_inputs,
            dev,
        )

        # ------------------------------------------------------------------
        # 4. Clone inputs for CBQ (Phase 2)
        # ------------------------------------------------------------------
        if self.cbq_enable:
            inps_cbq_saved = [x.clone() for x in inps]
            layer_kwargs_saved = deep_clone_layer_kwargs(layer_kwargs)

        # ------------------------------------------------------------------
        # 5. Auto-detect quantisation method per block
        # ------------------------------------------------------------------
        # onecomp: isinstance(mod, GPTQLinear) etc.
        # qep-dev: hasattr(mod, 'is_gptq_quantized') / _is_dbf_sequential()
        quant_strategy = auto_detect_quantization_strategy(layers)
        n_quantized = sum(1 for s in quant_strategy if s is not None)
        logger.info(
            "Strategy: %d quantized, %d FP16 blocks",
            n_quantized,
            total_layers - n_quantized,
        )

        # ------------------------------------------------------------------
        # 6. Load FP16 teacher model
        # ------------------------------------------------------------------
        # onecomp: model_config.load_model(device_map="cpu")
        # qep-dev: get_model(cfg.model) -> loads to GPU, then .cpu()
        logger.info("Loading FP16 teacher model...")
        teacher_model = model_config.load_model(device_map="cpu")
        teacher_model.eval()
        teacher_layers = get_transformer_layers(teacher_model)

        # ------------------------------------------------------------------
        # 7. Phase 1: Greedy block-wise optimisation
        # ------------------------------------------------------------------
        improvements = []

        for i in range(total_layers):
            method = quant_strategy[i]

            if method is None:
                # FP16 block: forward only to propagate activations
                logger.info(
                    "[Block %d/%d] FP16 — skipping optimisation",
                    i + 1,
                    total_layers,
                )
                layer = layers[i].to(dev)
                layer.eval()
                kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)
                with torch.no_grad():
                    for j in range(len(inps)):
                        inps[j] = layer_forward_single(
                            layer,
                            inps[j],
                            kw_gpu,
                            dev,
                        )
                layers[i] = layer.cpu()
                del layer, kw_gpu
                gc.collect()
                torch.cuda.empty_cache()
                continue

            logger.info(
                "[Block %d/%d] %s — block-wise distillation...",
                i + 1,
                total_layers,
                method,
            )

            # Student and Teacher to GPU
            layer_student = layers[i].to(dev)
            layer_teacher = teacher_layers[i].to(dev)
            kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)

            # Compute teacher outputs (store on CPU)
            with torch.no_grad():
                target_outputs = []
                for j in range(len(inps)):
                    t_out = layer_forward_single(
                        layer_teacher,
                        inps[j],
                        kw_gpu,
                        dev,
                    )
                    target_outputs.append(t_out)

            # Teacher back to CPU (do NOT del from ModuleList!)
            teacher_layers[i] = layer_teacher.cpu()
            del layer_teacher
            gc.collect()
            torch.cuda.empty_cache()

            # Method-specific optimisation
            improvement = self._optimize_single_block(
                layer=layer_student,
                target_outputs=target_outputs,
                inps=inps,
                layer_kwargs=layer_kwargs,
                dev=dev,
                method=method,
            )
            improvements.append(improvement)
            logger.info("[Block %d] improvement: %.2f%%", i + 1, improvement)

            # Update inps with student outputs
            layer_student.eval()
            with torch.no_grad():
                for j in range(len(inps)):
                    inps[j] = layer_forward_single(
                        layer_student,
                        inps[j],
                        kw_gpu,
                        dev,
                    )

            layers[i] = layer_student.cpu()
            del layer_student, target_outputs, kw_gpu
            gc.collect()
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 8. Phase 2: CBQ (Cross-Block Quantisation, window K=2)
        # ------------------------------------------------------------------
        # Jointly optimises two adjacent quantised blocks to reduce
        # greedy error accumulation from Phase 1.
        if self.cbq_enable:
            logger.info("[CBQ] Cross-Block Quantization (window K=2)")
            cbq_epochs = self.cbq_epochs if self.cbq_epochs > 0 else self.epochs
            cbq_lr = self.cbq_lr
            cbq_grad_clip = self.grad_clip

            cbq_kw_cpu = {}
            for k, v in layer_kwargs_saved.items():
                if k == "use_cache":
                    cbq_kw_cpu[k] = False
                    continue
                cbq_kw_cpu[k] = v
            cbq_kw_gpu = layer_kwargs_to_device(cbq_kw_cpu, dev)

            inps_cbq = [x.clone() for x in inps_cbq_saved]
            inps_teacher = [x.clone() for x in inps_cbq_saved]

            cbq_improvements = []

            for i in range(total_layers - 1):
                method_i = quant_strategy[i]
                method_j = quant_strategy[i + 1]

                if method_i is None or method_j is None or method_i != method_j:
                    # Skip: at least one FP16, or different methods
                    layer_s = layers[i].to(dev)
                    layer_s.eval()
                    layer_t = teacher_layers[i].to(dev)
                    layer_t.eval()
                    with torch.no_grad():
                        for j in range(len(inps_cbq)):
                            inps_cbq[j] = layer_forward_single(
                                layer_s,
                                inps_cbq[j],
                                cbq_kw_gpu,
                                dev,
                            )
                            inps_teacher[j] = layer_forward_single(
                                layer_t,
                                inps_teacher[j],
                                cbq_kw_gpu,
                                dev,
                            )
                    layers[i] = layer_s.cpu()
                    teacher_layers[i] = layer_t.cpu()
                    del layer_s, layer_t
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                logger.info("[CBQ] Window (%d, %d) — %s", i, i + 1, method_i)
                layer_si = layers[i].to(dev)
                layer_sj = layers[i + 1].to(dev)
                teacher_layer_i = teacher_layers[i].to(dev)
                teacher_layer_j = teacher_layers[i + 1].to(dev)

                # Teacher outputs for the window
                with torch.no_grad():
                    target_outputs_j = []
                    for j in range(len(inps_teacher)):
                        inp_gpu = inps_teacher[j].unsqueeze(0).to(dev)
                        mid_raw = teacher_layer_i(inp_gpu, **cbq_kw_gpu)
                        mid = mid_raw[0] if isinstance(mid_raw, tuple) else mid_raw
                        if mid.dim() == 2:
                            mid = mid.unsqueeze(0)
                        out_raw = teacher_layer_j(mid, **cbq_kw_gpu)
                        out = out_raw[0] if isinstance(out_raw, tuple) else out_raw
                        if out.dim() == 3 and out.size(0) == 1:
                            out = out.squeeze(0)
                        target_outputs_j.append(out.cpu())
                        del inp_gpu, mid_raw, mid, out_raw, out

                # CBQ optimisation
                if method_i == "gptq":
                    from ._blockwise.gptq_cbq_optimizer import (
                        optimize_gptq_cross_block,
                    )

                    init_err, final_err = optimize_gptq_cross_block(
                        layer_i=layer_si,
                        layer_j=layer_sj,
                        inps=inps_cbq,
                        target_outputs_j=target_outputs_j,
                        layer_kwargs=cbq_kw_cpu,
                        lr=cbq_lr,
                        epochs=cbq_epochs,
                        dev=dev,
                        grad_clip=cbq_grad_clip,
                        optimize_intweight=self.gptq_optimize_intweight,
                        intweight_lr=self.gptq_intweight_lr,
                    )
                elif method_i == "dbf":
                    from ._blockwise.dbf_cbq_optimizer import (
                        optimize_dbf_cross_block,
                    )

                    init_err, final_err = optimize_dbf_cross_block(
                        layer_i=layer_si,
                        layer_j=layer_sj,
                        inps=inps_cbq,
                        target_outputs_j=target_outputs_j,
                        layer_kwargs=cbq_kw_cpu,
                        lr=cbq_lr,
                        epochs=cbq_epochs,
                        dev=dev,
                        optimize_binary=self.optimize_binary,
                        k_smooth=self.k_smooth,
                        grad_clip=cbq_grad_clip,
                    )
                elif method_i == "onebit":
                    from ._blockwise.onebit_cbq_optimizer import (
                        optimize_onebit_cross_block,
                    )

                    init_err, final_err = optimize_onebit_cross_block(
                        layer_i=layer_si,
                        layer_j=layer_sj,
                        inps=inps_cbq,
                        target_outputs_j=target_outputs_j,
                        layer_kwargs=cbq_kw_cpu,
                        lr=cbq_lr,
                        epochs=cbq_epochs,
                        dev=dev,
                        optimize_sign=self.optimize_binary,
                        k_smooth=self.k_smooth,
                        grad_clip=cbq_grad_clip,
                    )
                else:
                    init_err, final_err = 0.0, 0.0

                if init_err > 0:
                    cbq_improvements.append(
                        (1 - final_err / init_err) * 100,
                    )

                # Propagate activations
                layer_si.eval()
                teacher_layer_i.eval()
                with torch.no_grad():
                    for j in range(len(inps_cbq)):
                        inps_cbq[j] = layer_forward_single(
                            layer_si,
                            inps_cbq[j],
                            cbq_kw_gpu,
                            dev,
                        )
                        inps_teacher[j] = layer_forward_single(
                            teacher_layer_i,
                            inps_teacher[j],
                            cbq_kw_gpu,
                            dev,
                        )

                layers[i] = layer_si.cpu()
                layers[i + 1] = layer_sj.cpu()
                teacher_layers[i] = teacher_layer_i.cpu()
                teacher_layers[i + 1] = teacher_layer_j.cpu()
                del (
                    layer_si,
                    layer_sj,
                    teacher_layer_i,
                    teacher_layer_j,
                    target_outputs_j,
                )
                gc.collect()
                torch.cuda.empty_cache()

            del cbq_kw_gpu
            if cbq_improvements:
                cbq_avg = sum(cbq_improvements) / len(cbq_improvements)
                logger.info("[CBQ] Average improvement: %.2f%%", cbq_avg)

        # ------------------------------------------------------------------
        # 9. Cleanup
        # ------------------------------------------------------------------
        del teacher_model, teacher_layers
        gc.collect()
        torch.cuda.empty_cache()

        quantized_model.eval()
        for param in quantized_model.parameters():
            param.requires_grad = False

        if improvements:
            avg = sum(improvements) / len(improvements)
            logger.info(
                "Block-wise PTQ complete. Average improvement: %.2f%%",
                avg,
            )
        else:
            logger.info("Block-wise PTQ complete. No quantized blocks found.")

    # ------------------------------------------------------------------
    # Method-specific dispatch
    # ------------------------------------------------------------------
    #
    # qep-dev: optimize_single_block() dispatches via cfg.method
    #   and reads parameters from cfg.blockwise_ptq.*
    # onecomp: dispatches via auto-detected method string
    #   and reads parameters from self (dataclass fields)
    #

    def _optimize_single_block(
        self,
        layer,
        target_outputs,
        inps,
        layer_kwargs,
        dev,
        method,
    ):
        """Dispatch to method-specific block optimiser.

        Returns improvement percentage.
        """
        if method == "gptq":
            from ._blockwise.gptq_block_optimizer import optimize_gptq_block

            init_err, final_err = optimize_gptq_block(
                layer=layer,
                inps=inps,
                target_outputs=target_outputs,
                layer_kwargs=layer_kwargs,
                lr=self.gptq_lr,
                epochs=self.epochs,
                dev=dev,
                grad_clip=self.grad_clip,
                optimize_intweight=self.gptq_optimize_intweight,
                intweight_lr=self.gptq_intweight_lr,
            )
        elif method == "dbf":
            from ._blockwise.dbf_block_optimizer import optimize_dbf_block

            init_err, final_err = optimize_dbf_block(
                layer=layer,
                inps=inps,
                target_outputs=target_outputs,
                layer_kwargs=layer_kwargs,
                lr=self.lr,
                epochs=self.epochs,
                dev=dev,
                optimize_binary=self.optimize_binary,
                k_smooth=self.k_smooth,
            )
        elif method == "onebit":
            from ._blockwise.onebit_block_optimizer import optimize_onebit_block

            init_err, final_err = optimize_onebit_block(
                layer=layer,
                inps=inps,
                target_outputs=target_outputs,
                layer_kwargs=layer_kwargs,
                lr=self.lr,
                epochs=self.epochs,
                dev=dev,
                optimize_sign=self.optimize_binary,
                k_smooth=self.k_smooth,
            )
        else:
            from ._blockwise.generic_block_optimizer import optimize_generic_block

            init_err, final_err = optimize_generic_block(
                layer=layer,
                inps=inps,
                target_outputs=target_outputs,
                layer_kwargs=layer_kwargs,
                lr=self.lr,
                epochs=self.epochs,
                dev=dev,
            )

        if init_err > 0:
            return ((init_err - final_err) / init_err) * 100.0
        return 0.0
