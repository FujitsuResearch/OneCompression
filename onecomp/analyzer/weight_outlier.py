"""Weight outlier analysis for quantization difficulty estimation.

This module provides tools to analyze weight outliers in neural network layers,
which is useful for identifying layers that may be difficult to quantize.

Typical Usage
-------------

1. Load a model and analyze weight outliers:

    >>> from onecomp.analyzer import analyze_weight_outliers, save_weight_distribution_plots
    >>> from onecomp import ModelConfig
    >>>
    >>> model = ModelConfig(model_id="meta-llama/Llama-3.2-1B").load_model()
    >>> analysis = analyze_weight_outliers(model)

2. Check the summary statistics:

    >>> print(f"Analyzed layers: {analysis.summary['num_layers_analyzed']}")
    >>> print(f"Overall outlier ratio: {analysis.summary['overall_outlier_ratio']:.2e}")

3. Extract layers that may be difficult to quantize:

    >>> difficult = analysis.difficult_layers(top_k=10)
    >>> for layer in difficult:
    ...     print(f"{layer.name}: outlier_ratio={layer.outlier_ratio:.2e}")

4. Visualize weight distributions for inspection:

    >>> paths = save_weight_distribution_plots(model, difficult, out_dir="plots")
    >>> print(f"Saved {len(paths)} histogram plots")

5. (Optional) Export results to JSON:

    >>> import json
    >>> with open("outlier_analysis.json", "w") as f:
    ...     json.dump(analysis.to_dict(), f, indent=2)

Available Outlier Detection Methods
-----------------------------------

- **mad** (default): Median Absolute Deviation. Robust to outliers.
  threshold = median(|w|) + k * MAD(|w|)

- **zscore**: Z-score based. Sensitive to outliers in the data.
  threshold = mean(|w|) + k * std(|w|)

- **iqr**: Interquartile Range. Similar robustness to MAD.
  threshold = Q3(|w|) + k * IQR(|w|)

- **percentile**: Direct percentile cutoff (e.g., 99.9th percentile).

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from logging import getLogger
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class LayerOutlierStats:
    """Outlier statistics of a layer weight.

    Attributes:
        name: Layer name (e.g., "model.layers.0.self_attn.q_proj").
        module_type: Module class name (e.g., "Linear").
        shape: Weight tensor shape.
        device: Device where weight is stored (e.g., "cuda:0", "cpu").
        dtype: Weight data type (e.g., "torch.float16").
        numel: Total number of elements in weight tensor.
        method: Outlier detection method used ("mad", "zscore", "percentile", "iqr").
        threshold: Computed outlier threshold.
        outlier_count: Number of elements exceeding threshold.
        outlier_ratio: Fraction of elements exceeding threshold (outlier_count / numel).
        has_many_outliers: True if outlier_ratio >= many_outliers_ratio.
        abs_max: Maximum absolute value in weight tensor.
        mean: Mean of |w| (from sample).
        std: Standard deviation of |w| (from sample).
        median: Median of |w| (from sample, only for MAD method).
        mad: Median absolute deviation (only for MAD method).
        p99: 99th percentile of |w|.
        p999: 99.9th percentile of |w|.

    Examples:
        >>> stat = analysis.layers["model.layers.0.self_attn.q_proj"]
        >>> print(f"Shape: {stat.shape}")
        >>> print(f"Outlier ratio: {stat.outlier_ratio:.2e}")
        >>> print(f"Threshold: {stat.threshold:.4f}")
        >>> if stat.has_many_outliers:
        ...     print("This layer has many outliers!")
    """

    name: str
    module_type: str
    shape: tuple[int, ...]
    device: str
    dtype: str
    numel: int

    method: str
    threshold: float
    outlier_count: int
    outlier_ratio: float
    has_many_outliers: bool

    abs_max: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mad: float | None = None
    p99: float | None = None
    p999: float | None = None


@dataclass(frozen=True)
class WeightOutlierAnalysis:
    """Outlier analysis result containing per-layer statistics and summary.

    Attributes:
        layers: Dictionary mapping layer names to LayerOutlierStats.
        summary: Summary statistics including total outlier count and ratio.

    Examples:
        Access per-layer stats:

        >>> analysis = analyze_weight_outliers(model)
        >>> for name, stat in analysis.layers.items():
        ...     print(f"{name}: ratio={stat.outlier_ratio:.2e}")

        Access summary:

        >>> print(f"Total outliers: {analysis.summary['total_outliers']}")
        >>> print(f"Overall ratio: {analysis.summary['overall_outlier_ratio']:.2e}")

        Export to JSON:

        >>> import json
        >>> with open("analysis.json", "w") as f:
        ...     json.dump(analysis.to_dict(), f, indent=2)
    """

    layers: dict[str, LayerOutlierStats]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "layers": {k: asdict(v) for k, v in self.layers.items()},
            "summary": self.summary,
        }

    def difficult_layers(
        self,
        *,
        min_outlier_ratio: float | None = None,
        min_outlier_count: int | None = None,
        top_k: int | None = None,
        sort_by: Literal[
            "outlier_ratio", "outlier_count", "abs_max"
        ] = "outlier_ratio",
    ) -> list[LayerOutlierStats]:
        """Extract layers that look difficult to quantize (many outliers).

        A layer is considered "difficult" if it exceeds the given thresholds.
        Results are sorted in descending order by the specified key.

        Args:
            min_outlier_ratio: Minimum outlier ratio to include a layer.
            min_outlier_count: Minimum outlier count to include a layer.
            top_k: Return only top-k layers after filtering and sorting.
            sort_by: Sort key. One of "outlier_ratio", "outlier_count", "abs_max".

        Returns:
            List of LayerOutlierStats sorted by the specified key in descending order.

        Examples:
            Get top 5 layers with highest outlier ratio:

            >>> difficult = analysis.difficult_layers(top_k=5)
            >>> for layer in difficult:
            ...     print(f"{layer.name}: ratio={layer.outlier_ratio:.2e}")

            Get layers with outlier ratio >= 1%:

            >>> difficult = analysis.difficult_layers(min_outlier_ratio=0.01)

            Get layers sorted by absolute max value:

            >>> difficult = analysis.difficult_layers(top_k=10, sort_by="abs_max")
        """
        items = list(self.layers.values())

        if min_outlier_ratio is not None:
            items = [x for x in items if x.outlier_ratio >= min_outlier_ratio]
        if min_outlier_count is not None:
            items = [x for x in items if x.outlier_count >= min_outlier_count]

        def _key(x: LayerOutlierStats):
            if sort_by == "outlier_count":
                return x.outlier_count
            if sort_by == "abs_max":
                # abs_max can be None for empty tensors
                return -float("inf") if x.abs_max is None else x.abs_max
            return x.outlier_ratio

        items.sort(key=_key, reverse=True)
        if top_k is not None:
            items = items[: max(int(top_k), 0)]
        return items


OutlierMethod = Literal["mad", "zscore", "percentile", "iqr"]
PlotScale = Literal["linear", "log"]


@dataclass
class WeightOutlierAnalyzer:
    """Analyze whether each layer has many outliers in its weight tensor.

    This class is intended to provide useful signals for later quantization strategy.

    Args:
        num_layers: Maximum number of layers to analyze. None means all layers.
        include_layer_names: List of layer names to include (exact match).
        exclude_layer_names: List of layer names to exclude (exact match).
        include_layer_keywords: Include layers whose name contains any of these keywords.
        exclude_layer_keywords: Exclude layers whose name contains any of these keywords.
        target_layer_types: Tuple of layer types to analyze (default: nn.Linear).
        method: Outlier detection method. One of "mad", "zscore", "percentile", "iqr".
        mad_k: Multiplier for MAD method. threshold = median + mad_k * MAD.
        zscore_k: Multiplier for z-score method. threshold = mean + zscore_k * std.
        iqr_k: Multiplier for IQR method. threshold = Q3 + iqr_k * IQR.
        percentile: Percentile for percentile method (e.g., 0.999 = 99.9th percentile).
        many_outliers_ratio: Threshold ratio to flag a layer as "has_many_outliers".
        sample_size: Sample size for computing statistics. None means use all elements.
        compute_on_cpu: If True, compute statistics on CPU to save GPU memory.
        chunk_size: Chunk size for streaming outlier count computation.

    Examples:
        Basic usage:

        >>> from onecomp import ModelConfig
        >>> from onecomp.analyzer import WeightOutlierAnalyzer
        >>> model = ModelConfig(model_id="meta-llama/Llama-3.2-1B").load_model()
        >>> analyzer = WeightOutlierAnalyzer()
        >>> analysis = analyzer.analyze(model)
        >>> print(f"Analyzed {analysis.summary['num_layers_analyzed']} layers")
        >>> print(f"Overall outlier ratio: {analysis.summary['overall_outlier_ratio']:.2e}")

        Using z-score method with custom threshold:

        >>> analyzer = WeightOutlierAnalyzer(method="zscore", zscore_k=5.0)
        >>> analysis = analyzer.analyze(model)

        Analyzing only attention layers:

        >>> analyzer = WeightOutlierAnalyzer(
        ...     include_layer_keywords=["q_proj", "k_proj", "v_proj", "o_proj"]
        ... )
        >>> analysis = analyzer.analyze(model)

        Analyzing first 10 layers only:

        >>> analyzer = WeightOutlierAnalyzer(num_layers=10)
        >>> analysis = analyzer.analyze(model)

    Notes:
        - For large layers, some statistics (median/MAD/quantile) can be computed on a sample
          for speed. The outlier count itself is computed on the full tensor using the
          threshold derived from the (possibly sampled) statistics.
    """

    # Layer selection parameters (mirrors onecomp.quantizer._quantizer.Quantizer)
    num_layers: int | None = None
    include_layer_names: list[str] | None = None
    exclude_layer_names: list[str] = field(default_factory=lambda: ["lm_head"])
    include_layer_keywords: list[str] | None = None
    exclude_layer_keywords: list[str] | None = None
    target_layer_types: tuple = field(default_factory=lambda: (nn.Linear,))

    # Outlier detection parameters
    method: OutlierMethod = "mad"
    mad_k: float = 6.0
    zscore_k: float = 6.0
    iqr_k: float = 3.0
    percentile: float = 0.999

    many_outliers_ratio: float = 1e-3
    sample_size: int | None = 200_000
    compute_on_cpu: bool = True
    chunk_size: int = (
        2_000_000  # number of elements per chunk for streaming stats
    )

    def __post_init__(self):
        self.logger = getLogger(__name__)

    def _should_include_layer(self, name: str, module: nn.Module) -> bool:
        # 1. Filter by layer type
        if not isinstance(module, self.target_layer_types):
            return False

        # 2. include_layer_names (exact match)
        if (
            self.include_layer_names is not None
            and name not in self.include_layer_names
        ):
            return False

        # 3. include_layer_keywords (contains any)
        if self.include_layer_keywords is not None:
            if not any(
                keyword in name for keyword in self.include_layer_keywords
            ):
                return False

        # 4. exclude_layer_names (exact match)
        if name in self.exclude_layer_names:
            return False

        # 5. exclude_layer_keywords (contains any)
        if self.exclude_layer_keywords is not None:
            if any(keyword in name for keyword in self.exclude_layer_keywords):
                return False

        return True

    def _sample_abs(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (flat, abs_sample_float32).

        Avoids materializing full |w| in float32, so it works on large models.
        """
        flat = w.detach().reshape(-1)
        numel = int(flat.numel())

        if numel == 0:
            return flat, torch.empty((0,), dtype=torch.float32)

        if self.sample_size is None or numel <= self.sample_size:
            sample = flat
        else:
            idx = torch.randint(
                0, numel, (self.sample_size,), device=flat.device
            )
            sample = flat[idx]

        # Compute stats on a small sample only
        abs_sample = sample.abs()
        if self.compute_on_cpu and abs_sample.device.type != "cpu":
            abs_sample = abs_sample.to("cpu")
        abs_sample = abs_sample.float()
        return flat, abs_sample

    def _count_outliers_and_absmax(
        self, flat: torch.Tensor, threshold: float
    ) -> tuple[int, float | None]:
        """Count outliers (|w| > threshold) and compute abs_max in a streaming manner."""
        numel = int(flat.numel())
        if numel == 0:
            return 0, None

        outliers = 0
        abs_max_val: float | None = None
        chunk = int(max(self.chunk_size, 1))

        for start in range(0, numel, chunk):
            part = flat[start : start + chunk]
            abs_part = part.abs()
            if self.compute_on_cpu and abs_part.device.type != "cpu":
                abs_part = abs_part.to("cpu")
            outliers += int((abs_part > threshold).sum().item())
            part_max = float(abs_part.max().item())
            if abs_max_val is None or part_max > abs_max_val:
                abs_max_val = part_max

        return outliers, abs_max_val

    def _calc_threshold(self, abs_sample: torch.Tensor) -> tuple[float, dict]:
        eps = 1e-12
        extra: dict = {}

        if abs_sample.numel() == 0:
            return float("inf"), extra

        if self.method == "percentile":
            q = float(self.percentile)
            q = min(max(q, 0.0), 1.0)
            thr = torch.quantile(abs_sample, q).item()
            extra["p99"] = torch.quantile(abs_sample, 0.99).item()
            extra["p999"] = torch.quantile(abs_sample, 0.999).item()
            return float(thr), extra

        if self.method == "zscore":
            mean = abs_sample.mean().item()
            std = abs_sample.std(unbiased=False).item()
            thr = mean + self.zscore_k * max(std, eps)
            extra["mean"] = float(mean)
            extra["std"] = float(std)
            return float(thr), extra

        if self.method == "iqr":
            q1 = torch.quantile(abs_sample, 0.25).item()
            q3 = torch.quantile(abs_sample, 0.75).item()
            iqr = max(q3 - q1, eps)
            thr = q3 + self.iqr_k * iqr
            extra["p99"] = torch.quantile(abs_sample, 0.99).item()
            extra["p999"] = torch.quantile(abs_sample, 0.999).item()
            return float(thr), extra

        # default: MAD (median absolute deviation), using abs-values (robust)
        median = torch.median(abs_sample).item()
        mad = torch.median((abs_sample - median).abs()).item()
        thr = median + self.mad_k * max(mad, eps)
        extra["median"] = float(median)
        extra["mad"] = float(mad)
        extra["p99"] = torch.quantile(abs_sample, 0.99).item()
        extra["p999"] = torch.quantile(abs_sample, 0.999).item()
        return float(thr), extra

    def analyze(self, model: nn.Module) -> WeightOutlierAnalysis:
        """Analyze the given model and return per-layer outlier stats."""

        layers: dict[str, LayerOutlierStats] = {}
        total_params = 0
        total_outliers = 0

        for name, module in model.named_modules():
            if not self._should_include_layer(name, module):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue

            weight: torch.Tensor = module.weight.data
            flat, abs_sample = self._sample_abs(weight)
            thr, extra = self._calc_threshold(abs_sample)

            outlier_count, abs_max_val = self._count_outliers_and_absmax(
                flat, thr
            )
            numel = int(flat.numel())
            outlier_ratio = float(outlier_count / max(numel, 1))
            has_many = outlier_ratio >= self.many_outliers_ratio

            total_params += numel
            total_outliers += outlier_count

            stat = LayerOutlierStats(
                name=name,
                module_type=module.__class__.__name__,
                shape=tuple(weight.shape),
                device=str(weight.device),
                dtype=str(weight.dtype),
                numel=numel,
                method=self.method,
                threshold=float(thr),
                outlier_count=outlier_count,
                outlier_ratio=outlier_ratio,
                has_many_outliers=has_many,
                abs_max=abs_max_val,
                mean=(
                    float(abs_sample.mean().item())
                    if abs_sample.numel() > 0
                    else None
                ),
                std=(
                    float(abs_sample.std(unbiased=False).item())
                    if abs_sample.numel() > 0
                    else None
                ),
                median=extra.get("median"),
                mad=extra.get("mad"),
                p99=extra.get("p99"),
                p999=extra.get("p999"),
            )
            layers[name] = stat

            if self.num_layers is not None and len(layers) >= self.num_layers:
                break

        summary = {
            "method": self.method,
            "many_outliers_ratio": self.many_outliers_ratio,
            "sample_size": self.sample_size,
            "compute_on_cpu": self.compute_on_cpu,
            "num_layers_analyzed": len(layers),
            "total_params_analyzed": total_params,
            "total_outliers": total_outliers,
            "overall_outlier_ratio": float(
                total_outliers / max(total_params, 1)
            ),
        }

        return WeightOutlierAnalysis(layers=layers, summary=summary)


def analyze_weight_outliers(
    model: nn.Module,
    *,
    # Layer selection (same as WeightOutlierAnalyzer)
    num_layers: int | None = None,
    include_layer_names: list[str] | None = None,
    exclude_layer_names: list[str] | None = None,
    include_layer_keywords: list[str] | None = None,
    exclude_layer_keywords: list[str] | None = None,
    target_layer_types: tuple | None = None,
    # Outlier detection
    method: OutlierMethod = "mad",
    mad_k: float = 6.0,
    zscore_k: float = 6.0,
    iqr_k: float = 3.0,
    percentile: float = 0.999,
    many_outliers_ratio: float = 1e-3,
    sample_size: int | None = 200_000,
    compute_on_cpu: bool = True,
) -> WeightOutlierAnalysis:
    """Convenience API to analyze weight outliers.

    This function provides a simple interface to analyze weight outliers without
    manually instantiating WeightOutlierAnalyzer. See WeightOutlierAnalyzer for
    detailed parameter descriptions.

    Args:
        model: PyTorch model to analyze.
        num_layers: Maximum number of layers to analyze.
        include_layer_names: List of layer names to include (exact match).
        exclude_layer_names: List of layer names to exclude (exact match).
        include_layer_keywords: Include layers containing any of these keywords.
        exclude_layer_keywords: Exclude layers containing any of these keywords.
        target_layer_types: Tuple of layer types to analyze.
        method: Outlier detection method ("mad", "zscore", "percentile", "iqr").
        mad_k: Multiplier for MAD method.
        zscore_k: Multiplier for z-score method.
        iqr_k: Multiplier for IQR method.
        percentile: Percentile for percentile method.
        many_outliers_ratio: Threshold to flag "has_many_outliers".
        sample_size: Sample size for statistics computation.
        compute_on_cpu: Compute statistics on CPU to save GPU memory.

    Returns:
        WeightOutlierAnalysis containing per-layer stats and summary.

    Examples:
        Basic usage:

        >>> from onecomp.analyzer import analyze_weight_outliers
        >>> from onecomp import ModelConfig
        >>> model = ModelConfig(model_id="meta-llama/Llama-3.2-1B").load_model()
        >>> analysis = analyze_weight_outliers(model)
        >>> print(analysis.summary)

        Find difficult layers and inspect:

        >>> analysis = analyze_weight_outliers(model)
        >>> difficult = analysis.difficult_layers(top_k=5)
        >>> for layer in difficult:
        ...     print(f"{layer.name}: outlier_ratio={layer.outlier_ratio:.2e}")

        Use z-score method:

        >>> analysis = analyze_weight_outliers(model, method="zscore", zscore_k=5.0)

        Analyze only MLP layers:

        >>> analysis = analyze_weight_outliers(
        ...     model,
        ...     include_layer_keywords=["mlp"],
        ... )

        Save results to JSON:

        >>> import json
        >>> with open("outlier_analysis.json", "w") as f:
        ...     json.dump(analysis.to_dict(), f, indent=2)
    """
    analyzer = WeightOutlierAnalyzer(
        num_layers=num_layers,
        include_layer_names=include_layer_names,
        exclude_layer_names=exclude_layer_names or ["lm_head"],
        include_layer_keywords=include_layer_keywords,
        exclude_layer_keywords=exclude_layer_keywords,
        target_layer_types=target_layer_types or (nn.Linear,),
        method=method,
        mad_k=mad_k,
        zscore_k=zscore_k,
        iqr_k=iqr_k,
        percentile=percentile,
        many_outliers_ratio=many_outliers_ratio,
        sample_size=sample_size,
        compute_on_cpu=compute_on_cpu,
    )
    return analyzer.analyze(model)


def save_weight_distribution_plots(
    model: nn.Module,
    layers: list[str] | list[LayerOutlierStats],
    *,
    out_dir: str = "weight_outlier_plots",
    sample_size: int = 200_000,
    bins: int = 200,
    scale: PlotScale = "log",
    include_signed: bool = True,
) -> list[str]:
    """Save weight distribution plots for specified layers.

    This is intended for quickly inspecting whether "difficult layers" indeed have heavy tails.
    Each plot shows a histogram of |w| (absolute weight values) and optionally w (signed values).
    The outlier threshold is shown as a vertical red dashed line when LayerOutlierStats is provided.

    Args:
        model: Target model.
        layers: List of layer names OR list of LayerOutlierStats objects.
            If LayerOutlierStats is provided, threshold lines and outlier info are shown.
        out_dir: Output directory to save PNG files.
        sample_size: Number of elements sampled from the weight tensor for plotting.
        bins: Number of histogram bins.
        scale: Y-axis scale. "log" (recommended) or "linear".
        include_signed: If True, plot both |w| and w histograms (2 subplots per layer).

    Returns:
        List of saved PNG file paths.

    Examples:
        Plot difficult layers identified by analyze_weight_outliers:

        >>> from onecomp.analyzer import analyze_weight_outliers, save_weight_distribution_plots
        >>> from onecomp import ModelConfig
        >>> model = ModelConfig(model_id="meta-llama/Llama-3.2-1B").load_model()
        >>> analysis = analyze_weight_outliers(model)
        >>> difficult = analysis.difficult_layers(top_k=10)
        >>> paths = save_weight_distribution_plots(model, difficult, out_dir="plots")
        >>> print(f"Saved {len(paths)} plots")

        Plot specific layers by name:

        >>> layer_names = [
        ...     "model.layers.0.self_attn.q_proj",
        ...     "model.layers.0.self_attn.k_proj",
        ... ]
        >>> paths = save_weight_distribution_plots(model, layer_names)

        Use linear scale for y-axis:

        >>> paths = save_weight_distribution_plots(
        ...     model, difficult, scale="linear"
        ... )

        Plot only |w| histogram (no signed weights):

        >>> paths = save_weight_distribution_plots(
        ...     model, difficult, include_signed=False
        ... )

    Notes:
        - Requires matplotlib. If not installed, raises ImportError with guidance.
        - Uses sampling for speed; do not interpret absolute tail probability too literally.
    """
    try:
        import os
        import re
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from e

    os.makedirs(out_dir, exist_ok=True)

    # Normalize input
    if len(layers) == 0:
        return []
    if isinstance(layers[0], LayerOutlierStats):  # type: ignore[index]
        stats_list = layers  # type: ignore[assignment]
        names = [s.name for s in stats_list]  # type: ignore[attr-defined]
        stats_by_name = {s.name: s for s in stats_list}  # type: ignore[attr-defined]
    else:
        names = layers  # type: ignore[assignment]
        stats_by_name = {}

    # Fast module lookup
    name_to_module = dict(model.named_modules())
    saved: list[str] = []

    def _safe(s: str) -> str:
        s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
        return s[:200]

    for name in names:
        module = name_to_module.get(name)
        if (
            module is None
            or not hasattr(module, "weight")
            or module.weight is None
        ):
            continue
        w = module.weight.detach().reshape(-1)
        numel = int(w.numel())
        if numel == 0:
            continue

        n = min(int(sample_size), numel)
        if n < numel:
            idx = torch.randint(0, numel, (n,), device=w.device)
            sample = w[idx]
        else:
            sample = w

        # Move to CPU for plotting
        sample = sample.float().to("cpu")
        abs_sample = sample.abs()

        stat = stats_by_name.get(name)
        thr = None if stat is None else stat.threshold
        out_ratio = None if stat is None else stat.outlier_ratio
        out_cnt = None if stat is None else stat.outlier_count

        # Plot
        rows = 2 if include_signed else 1
        fig, axes = plt.subplots(
            rows, 1, figsize=(8, 3.5 * rows), tight_layout=True
        )
        if rows == 1:
            axes = [axes]

        ax = axes[0]
        ax.hist(abs_sample.numpy(), bins=bins)
        ax.set_title(f"{name} |w| (sample n={n})")
        ax.set_xlabel("|w|")
        ax.set_ylabel("count")
        if scale == "log":
            ax.set_yscale("log")
        if thr is not None:
            ax.axvline(
                thr,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"threshold={thr:.4g}",
            )
            ax.legend(loc="best")

        if include_signed:
            ax2 = axes[1]
            ax2.hist(sample.numpy(), bins=bins)
            ax2.set_title(f"{name} w (sample n={n})")
            ax2.set_xlabel("w")
            ax2.set_ylabel("count")
            if scale == "log":
                ax2.set_yscale("log")

        if out_ratio is not None and out_cnt is not None:
            fig.suptitle(
                f"{name}  outlier_ratio={out_ratio:.3e}  outlier_count={out_cnt}",
                y=1.02,
                fontsize=10,
            )

        out_path = os.path.join(out_dir, f"{_safe(name)}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved
