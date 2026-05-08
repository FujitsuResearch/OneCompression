"""
Copyright 2025-2026 Fujitsu Ltd.

"""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ._lpcd_config import LPCDConfig
@dataclass
class ClosedFormSolverArgument:
    lpcd_config: LPCDConfig
    block_q: nn.Module
    block_f: nn.Module
    inps_q: torch.Tensor
    inps_f: torch.Tensor
    kwargs: dict
    device: str


class LpcdMetric(nn.Module, ABC):
    """
    Abstract base class for LPCD metrics.
    The subclass will be implemented for each module group (e.g., qk-module)
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.is_ready = {module: False for _, module in self.named_targets()}
        self.is_refined = False

    def is_refineable(self) -> bool:
        return all(self.is_ready.values()) and not self.is_refined
    
    # Implement named targets of the metric
    @abstractmethod
    def named_targets(self) -> list[tuple[str, nn.Module]]:
        pass
    
    # Implement metric computation here
    @abstractmethod
    def forward(self, block_inps: torch.Tensor, kwargs: dict) -> torch.Tensor:
        pass

    # (optional) Implement closed-form solvers for specific cases (e.g., llama-out, etc.)
    def closed_form_solvers(self) -> list[Callable[[ClosedFormSolverArgument], None] | None]:
        return [None] * len(self.named_targets())
    

class LpcdMetricGroup:
    """LpcdMetricGroup organizes LpcdMetrics for a block
    """

    def __init__(self, metrics: list[tuple[LpcdMetric, LpcdMetric]]):
        self.metrics = metrics
        self.module_to_metric = {}
        for metric_q, metric_f in metrics:
            for _, module in metric_q.named_targets():
                self.module_to_metric[module] = metric_q

    def mark_as_ready(self, module: nn.Module):
        if module in self.module_to_metric:
            self.module_to_metric[module].is_ready[module] = True

    def get_refineable_metrics(self) -> list[LpcdMetric]:
        return [(metric_q, metric_f) for metric_q, metric_f in self.metrics if metric_q.is_refineable()]



from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

def make_lpcd_metrics(
        lpcd_config: LPCDConfig, 
        block_q: nn.Module, 
        block_f: nn.Module
    ) -> LpcdMetricGroup:
    # Implement logic to select and return the appropriate LpcdMetric subclass
    # based on the type of *block* (e.g., LlamaDecoderLayer, etc.)
    if isinstance(block_q, LlamaDecoderLayer):
        # Llama
        from .arch._llama import make_llama_lpcd_metrics
        return make_llama_lpcd_metrics(lpcd_config, block_q, block_f)
    elif isinstance(block_q, Qwen3DecoderLayer):
        # Qwen3
        from .arch._qwen3 import make_qwen3_lpcd_metrics
        return make_qwen3_lpcd_metrics(lpcd_config, block_q, block_f)

    # Add more architecture-specific cases here
    else:
        cls_name = block_q.__class__.__name__
        raise NotImplementedError(f"No LPCD module group implemented for {cls_name}")
    