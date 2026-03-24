"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Optional, Union

import time
import math
import torch
from torch.nn import Linear, Conv2d, Conv1d


@dataclass
class QuantizationResult:
    """Base class for quantization results.

    Each quantization method inherits from this class and adds
    method-specific parameters as fields.

    Attributes:
        dequantized_weight: Dequantized weights (FP16) - required.
        quantization_time: Time taken for quantization (seconds).
        output_squared_error: Output squared error (when calc_quant_error=True).
        mean_output_squared_error: Output mean squared error (when calc_quant_error=True).
        weight_squared_error: Weight squared error (when calc_quant_error=True).
        mean_weight_squared_error: Weight mean squared error (when calc_quant_error=True).
        relative_output_squared_error: Output relative squared error
            ||WX^T - ŴX^T||²_F / ||WX^T||²_F (when calc_quant_error=True).
        relative_weight_squared_error: Weight relative squared error
            ||W - Ŵ||²_F / ||W||²_F (when calc_quant_error=True).
    """

    dequantized_weight: torch.Tensor

    # Quantization metadata
    quantization_time: float = None

    # Quantization error info (set when calc_quant_error=True)
    output_squared_error: float = None
    mean_output_squared_error: float = None
    weight_squared_error: float = None
    mean_weight_squared_error: float = None
    relative_output_squared_error: float = None
    relative_weight_squared_error: float = None

    def compute_dequantized_weight(self, device: torch.device = None) -> torch.Tensor:
        """Compute and return the dequantized weight.

        Subclasses should override this to recompute dequantized weights
        from quantization parameters (scale, zero_point, assignment, etc.).

        Args:
            device (torch.device): Device to perform computation on.
                If None, computation is performed on the device where
                quantization parameters reside.

        Returns:
            torch.Tensor: Dequantized weight tensor.

        Raises:
            NotImplementedError: Not implemented in the base class.
        """
        raise NotImplementedError(
            "compute_dequantized_weight() is not implemented in the base class. "
            "Subclasses should override this method."
        )


@dataclass
class Quantizer(metaclass=ABCMeta):
    """Abstract base class for quantizers

    Attributes:
        num_layers (int): The number of layers to be quantized,
            if None, all layers will be quantized.
        calc_quant_error (bool): If True, calculate quantization error.
        exclude_layer_names (list[str]): List of layer names to exclude
            from quantization (exact match).
        include_layer_names (list[str]): List of layer names to include
            for quantization (exact match). If None, all layers are candidates.
        include_layer_keywords (list[str]): List of keywords to match
            layer names for quantization. If any keyword is contained in the
            layer name, the layer is included. If None, all layers are candidates.
        exclude_layer_keywords (list[str]): List of keywords to exclude
            layer names from quantization. If any keyword is contained in the
            layer name, the layer is excluded.
        target_layer_types (tuple): Tuple of layer types to quantize.
            Default is (Linear,).

    Layer Selection Priority:
        1. target_layer_types: Filter by layer type
        2. include_layer_names: If specified, only include exact matches
        3. include_layer_keywords: If specified, only include layers containing any keyword
        4. exclude_layer_names: Exclude exact matches
        5. exclude_layer_keywords: Exclude layers containing any keyword
        6. num_layers: Limit the maximum number of layers

    To create a new Quantizer:
    - Inherit from this class.
    - Must implement the following method: quantize_layer
    - Set flag_calibration to True if calibration data is needed
    - Set flag_hessian to True if the Hessian matrix is needed

    Examples:
        # Example 1: Exclude lm_head (default behavior)
        quantizer = GPTQ(exclude_layer_names=["lm_head"])

        # Example 2: Quantize only specific layers
        quantizer = GPTQ(
            include_layer_names=["model.layers.0.self_attn.q_proj"]
        )

        # Example 3: Quantize layers containing specific keywords
        quantizer = GPTQ(
            include_layer_keywords=["q_proj", "k_proj", "v_proj"]
        )

        # Example 4: Exclude layers containing specific keywords
        quantizer = GPTQ(
            exclude_layer_keywords=["down_proj", "gate_proj"]
        )

    """

    # Quantizer name (used as a key in results dicts, logs, etc.)
    # If None, automatically set to the class name in __post_init__.
    name: str = None

    # Parameters for the quantizer
    num_layers: int = None
    calc_quant_error: bool = False

    # Layer selection parameters
    include_layer_names: list[str] = None  # Layers to explicitly quantize (exact match)
    exclude_layer_names: list[str] = field(default_factory=lambda: ["lm_head"])
    include_layer_keywords: list[str] = None  # Quantize layers containing these keywords
    exclude_layer_keywords: list[str] = None  # Exclude layers containing these keywords
    target_layer_types: tuple = field(default_factory=lambda: (Linear,))  # Target layer types

    # Hessian / statistics precision
    # dtype for computing and storing statistics such as X^T X.
    # torch.float32 is sufficient for GPTQ, DBF, etc. torch.float64 is preferred for JointQ.
    hessian_dtype: torch.dtype = torch.float32

    # internal use
    module_to_name: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    flag_calibration: bool = False
    flag_hessian: bool = False
    flag_xtx: bool = False  # Whether X^T X is needed (e.g., JointQ)

    def __post_init__(self):
        """__post_init__ method"""

        if self.name is None:
            self.name = type(self).__name__

        self.logger = getLogger(__name__)

    def validate_params(self):
        """Validate quantizer parameters.

        Override in subclasses when parameter validation is required.

        """
        pass

    def quantize(
        self, module, input, output
    ):  # pylint: disable=redefined-builtin, unused-argument
        """Quantize the layer

        This method is called by the register_forward_hook method of the layer.

        """

        name = self.module_to_name[module]

        self.logger.info("Quantizing layer: %s", name)
        start_time = time.time()
        if self.flag_hessian:
            hessian = self.calculate_hessian(module, input)
        else:
            hessian = None
        result = self.quantize_layer(module, input, hessian=hessian)
        end_time = time.time()
        if hessian is not None:
            del hessian

        # Backward compatibility: convert Tensor to QuantizationResult
        if isinstance(result, torch.Tensor):
            result = QuantizationResult(dequantized_weight=result)
        result.quantization_time = end_time - start_time

        self.results[name] = result
        torch.cuda.empty_cache()

        if self.calc_quant_error:
            # Record quantization error
            self._record_quantization_error(module, name, input, result)

    def quantize_with_qep(
        self,
        module,
        quant_input_activation,
        original_input_activation=None,
        percdamp=0.01,
        perccorr=0.5,
        hessian=None,
        delta_hatX=None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Quantize the layer with QEP

        Args:
            module (torch.nn.Module): The layer module
            quant_input_activation (torch.Tensor): The input activations of the quantized layer
            original_input_activation (torch.Tensor): The input activations of the original layer
            hessian (torch.Tensor): The Hessian matrix
            delta_hatX (torch.Tensor): The cross-term matrix
        """

        name = self.module_to_name[module]

        start_time = time.time()

        # Calculate the Hessian matrix
        if hessian is None and self.flag_hessian:
            hessian = self.calculate_hessian(module, quant_input_activation)

        # Adjust the weights to be quantized
        if delta_hatX is not None or original_input_activation is not None:
            self.logger.info("Adjusting the weight of the layer: %s", name)
            self.adjust_weight(
                module,
                quant_input_activation,
                original_input_activation,
                original_hessian=hessian,
                original_delta_hatX=delta_hatX,
                percdamp=percdamp,
                perccorr=perccorr,
            )
            torch.cuda.empty_cache()

        self.logger.info("Quantizing layer: %s", name)
        result = self.quantize_layer(module, quant_input_activation, hessian=hessian)
        end_time = time.time()
        if hessian is not None:
            del hessian

        # Backward compatibility: convert Tensor to QuantizationResult
        if isinstance(result, torch.Tensor):
            result = QuantizationResult(dequantized_weight=result)
        result.quantization_time = end_time - start_time

        self.results[name] = result
        torch.cuda.empty_cache()

        if self.calc_quant_error:
            # Record quantization error
            self._record_quantization_error(
                module,
                name,
                quant_input_activation,
                result,
            )

    def _record_quantization_error(
        self, module, name, input, result
    ):  # pylint: disable=redefined-builtin
        """Record the quantization error for the layer

        Args:
            module (torch.nn.Module): The layer module
            name (str): The name of the layer
            input (tuple or torch.Tensor): The input to the layer
            result (QuantizationResult): The quantization result
        """
        dequantized_weight = result.dequantized_weight

        (
            result.output_squared_error,
            result.mean_output_squared_error,
            result.relative_output_squared_error,
        ) = self.calculate_output_quantization_error(module, input, dequantized_weight)

        (
            result.weight_squared_error,
            result.mean_weight_squared_error,
            result.relative_weight_squared_error,
        ) = self.calculate_weight_quantization_error(module, dequantized_weight)

        torch.cuda.empty_cache()

    def adjust_weight(
        self,
        module,
        quant_input_activation,
        original_input_activation,
        original_hessian=None,
        original_delta_hatX=None,
        percdamp=0.01,
        perccorr=0.5,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """Adjust the weight of the layer"""

        # Get the weight of the layer
        weight = module.weight.data.clone()
        if isinstance(module, Conv2d):
            weight = weight.flatten(1)
        elif isinstance(module, Conv1d):
            weight = weight.t()
        weight = weight.float()

        # Get the Hessian matrix
        if original_hessian is None:
            hessian = self.calculate_hessian(module, quant_input_activation)
        else:
            hessian = original_hessian.clone()

        # Calculate delta^T hat_X
        if original_delta_hatX is None:
            delta_hatX = self.calculate_delta_hatX(
                module, quant_input_activation, original_input_activation
            )
        else:
            delta_hatX = original_delta_hatX.clone()

        # Dead columns guard
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        # QEP correction
        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[diag, diag] += damp
        cholesky = torch.linalg.cholesky(hessian)
        rhs = weight @ delta_hatX
        delta_weight = torch.cholesky_solve(rhs.t(), cholesky).t()
        weight = weight + (perccorr * delta_weight)

        if isinstance(module, Conv1d):
            weight = weight.t()
        module.weight.data = weight.reshape(module.weight.shape).to(module.weight.data.dtype)

    @abstractmethod
    def quantize_layer(
        self, module, input=None, hessian=None
    ) -> Union[torch.Tensor, QuantizationResult]:  # pylint: disable=redefined-builtin
        """Quantize the layer

        Args:
            module (torch.nn.Module): The layer module
            input (tuple or torch.Tensor): The input to the layer
            hessian (torch.Tensor): The Hessian matrix

        Returns:
            Union[torch.Tensor, QuantizationResult]:
                Dequantized weight (torch.Tensor),
                or a QuantizationResult object.
        """
        raise NotImplementedError("quantize_layer method is not implemented")

    def _should_quantize_layer(self, name: str, module) -> bool:
        """Determine whether a layer should be quantized.

        Filters layers according to the following priority:

        1. Filter by layer type using target_layer_types
        2. If include_layer_names is specified, only include exact matches
        3. If include_layer_keywords is specified, only include layers containing any keyword
        4. Exclude layers that exactly match exclude_layer_names
        5. Exclude layers containing any keyword in exclude_layer_keywords

        Args:
            name (str): The name of the layer
            module: The layer module

        Returns:
            bool: True if the layer should be quantized, False otherwise
        """

        # 1. Check layer type
        if not isinstance(module, self.target_layer_types):
            return False

        # 2. If include_layer_names is specified, check if name is in the list
        if self.include_layer_names is not None:
            if name not in self.include_layer_names:
                return False

        # 3. If include_layer_keywords is specified, check if name contains any keyword
        if self.include_layer_keywords is not None:
            matched = any(keyword in name for keyword in self.include_layer_keywords)
            if not matched:
                return False

        # 4. Exclude if name exactly matches exclude_layer_names
        if name in self.exclude_layer_names:
            return False

        # 5. Exclude if name contains any keyword in exclude_layer_keywords
        if self.exclude_layer_keywords is not None:
            for keyword in self.exclude_layer_keywords:
                if keyword in name:
                    return False

        return True

    def setup(self, model):
        """Setup the quantizer with the model

        Args:
            model: The model to be quantized

        """

        self.validate_params()

        assert len(self.module_to_name) == 0

        for name, module in model.named_modules():
            # Check if the layer should be quantized
            if self._should_quantize_layer(name, module):
                self.module_to_name[module] = name

            # If the number of layers is specified,
            # stop the loop when the number of layers is reached
            if self.num_layers is not None and len(self.module_to_name) >= self.num_layers:
                break

        if self.num_layers is not None:
            assert len(self.module_to_name) == self.num_layers, (
                f"Expected {self.num_layers} layers to quantize, "
                f"but found {len(self.module_to_name)}"
            )

    def execute_post_processing(self):
        """Execute the post processing"""

        # Clear
        self.module_to_name = {}

    # ========================================
    # Save / inference layer (optional override)
    # ========================================

    def get_quant_config(self) -> dict:
        """Return quantization_config dict for saving (used by save_quantized_model).

        Returns the content stored in model.config.quantization_config.
        Override this in quantizers that support save_quantized_model.

        Returns:
            dict: Config dict including ``quant_method``

        Raises:
            NotImplementedError: If this quantizer does not support save_quantized_model
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support save_quantized_model. "
            "Override get_quant_config() and create_inference_layer() to enable saving."
        )

    def create_inference_layer(
        self,
        result: "QuantizationResult",
        linear_module: Linear,
        **kwargs,
    ) -> Linear:
        """Build an inference layer from one entry in quantizer.results
        (used by save_quantized_model).

        Override in quantizers that support save_quantized_model;
        call from_quantization_result on the method's inference layer class and return it.

        Args:
            result: One entry from quantizer.results[name] (a QuantizationResult subclass)
            linear_module: Original Linear layer (used to get bias / device)
            **kwargs: Method-specific options (e.g. pack_weights, use_gemlite)

        Returns:
            Linear: Quantized inference layer (nn.Module)

        Raises:
            NotImplementedError: If this quantizer does not support save_quantized_model
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support save_quantized_model. "
            "Override create_inference_layer() to enable saving."
        )

    def apply_results_to_model(self, model, **kwargs):
        """Replace Linear layers in model with quantized inference layers from self.results.

        Call load_results(filepath) before this, or ensure self.results is already populated.
        **kwargs are passed to create_inference_layer (e.g. pack_weights=True for GPTQ).

        Args:
            model (nn.Module): The model to modify (in place). Typically the base model
                loaded with from_pretrained() so that state_dict keys match.

        Returns:
            None (modifies model in place)

        Example:
            >>> quantizer.load_results("quantization_results.pt")
            >>> quantizer.apply_results_to_model(model, pack_weights=True)
        """
        for name in list(self.results):
            result = self.results[name]
            *parent_path, attr_name = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            linear_module = getattr(parent, attr_name)
            quantized_layer = self.create_inference_layer(
                result=result,
                linear_module=linear_module,
                **kwargs,
            )
            setattr(parent, attr_name, quantized_layer)
            self.logger.debug("Replaced %s with %s", name, quantized_layer.__class__.__name__)
        if self.results:
            first_name = next(iter(self.results))
            *parent_path, attr_name = first_name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            layer_class_name = getattr(parent, attr_name).__class__.__name__
            self.logger.info(
                "Replaced %d Linear layers with %s",
                len(self.results),
                layer_class_name,
            )

    # ========================================
    # Save config post-processing hook
    # ========================================

    def finalize_quant_config_for_save(
        self,
        quant_config: dict[str, Any],
        quantized_layer_names: list[str],
        num_hidden_layers: Optional[int] = None,
    ) -> dict[str, Any]:
        """Optional hook to augment quantization_config just before saving.

        Runner builds the common fields (e.g. modules_in_block_to_quantize) and then
        calls this hook so each quantizer can inject method-specific metadata needed
        by downstream consumers (e.g. vLLM plugins).

        Default implementation is a no-op.
        """
        _ = quantized_layer_names
        _ = num_hidden_layers
        return quant_config

    def save_results(self, filepath):
        """Save the quantization results to a file.

        Saves self.results (a dict mapping layer name -> QuantizationResult) to a file.

        Args:
            filepath (str): The path to save the results.
                The .pt extension is recommended.

        Example:
            >>> quantizer.save_results("quantization_results.pt")
        """
        torch.save(self.results, filepath)
        self.logger.info("Saved quantization results to %s", filepath)

    def load_results(self, filepath, weights_only=False):
        """Load the quantization results from a file into self.results.

        Loads saved quantization results and stores them in self.results.

        Args:
            filepath (str): The path to load the results from.
            weights_only (bool): If True, only load tensor weights (safer but limited).
                Default is False to support loading QuantizationResult objects.

        Returns:
            dict: A dict mapping layer name -> QuantizationResult (same reference as self.results).

        Example:
            >>> quantizer = JointQ()
            >>> quantizer.load_results("quantization_results.pt")
            >>> for layer_name, result in quantizer.results.items():
            ...     print(f"{layer_name}: {result.dequantized_weight.shape}")

        Note:
            Backward compatibility:
            - QuantizationResult subclasses (JointQResult, GPTQResult, etc.)
              require the same class definitions to be available when loading.
            - Loading files saved with older versions may fail if class
              definitions have changed.
        """
        self.results = torch.load(filepath, weights_only=weights_only)
        self.logger.info("Loaded quantization results from %s", filepath)
        return self.results

    def calculate_hessian(self, module, input):  # pylint: disable=redefined-builtin
        """Calculate the Hessian matrix for the layer.

        Reference: QEP-dev src/method/test_methods.py (2024/09/23)

        Notes:
        - Processes in batches for memory efficiency
        - Uses scaling for numerical stabilization (not plain X^T X)
        - Assuming tmp is constant across iterations and the loop runs n times:
          H = 2/(tmp * n) * (X^T X)

        Args:
            module (torch.nn.Module): The layer module
            input (tuple): The input to the layer
        """

        device = module.weight.device

        # Get input activations
        if isinstance(input, tuple):
            input_activations = input[0].detach()
        else:
            input_activations = input.detach()

        assert isinstance(module, Linear)  # TODO: Support other layer types
        assert len(input_activations.shape) == 3  # (batch_size, seq_len, hidden_size)

        hidden_size = input_activations.shape[-1]
        assert hidden_size == module.weight.shape[1]

        # Initialize Hessian
        hessian = torch.zeros((hidden_size, hidden_size), device=device)
        nsamples = 0

        # Process in batches (small batches for memory efficiency)
        batch_size = min(input_activations.shape[0], 8)

        for i in range(0, input_activations.shape[0], batch_size):
            batch_activations = input_activations[i : i + batch_size].to(device).float()
            assert batch_activations.shape[-1] == hidden_size
            batch_activations = batch_activations.reshape((-1, batch_activations.shape[-1]))

            # Transpose to column vectors
            inp = batch_activations.t()

            # Note: Without scaling, this would simply be:
            # hessian += inp @ inp.T

            tmp = batch_activations.shape[0]
            hessian *= nsamples / (nsamples + tmp)
            nsamples += tmp

            # Scaling
            inp_scaled = math.sqrt(2 / nsamples) * inp.float()
            hessian += inp_scaled.matmul(inp_scaled.t())

        return hessian

    def calculate_delta_hatX(
        self, module, quant_input_activation, original_input_activation
    ):  # pylint: disable=too-many-locals
        """Calculate delta^T hat_X for the layer.

        delta := original_input_activation - quant_input_activation
        hat_X := quant_input_activation

        Computed using the same method as `calculate_hessian`.

        """
        # TODO: Would it be more efficient to compute together with hessian?

        assert isinstance(module, Linear)  # TODO: Support other layer types
        assert len(quant_input_activation.shape) == 3
        assert len(original_input_activation.shape) == 3
        assert quant_input_activation.shape == original_input_activation.shape
        # (batch_size, seq_len, hidden_size)

        device = module.weight.device
        hidden_size = quant_input_activation.shape[-1]
        assert hidden_size == module.weight.shape[1]

        # Initialize delta^T hat_X
        delta_hatX = torch.zeros((hidden_size, hidden_size), device=device)
        nsamples = 0

        # Process in batches (small batches for memory efficiency)
        batch_size = min(quant_input_activation.shape[0], 8)

        for i in range(0, quant_input_activation.shape[0], batch_size):
            batch_quant_activations = quant_input_activation[i : i + batch_size].to(device).float()
            batch_original_activations = (
                original_input_activation[i : i + batch_size].to(device).float()
            )
            assert batch_quant_activations.shape[-1] == hidden_size
            assert batch_original_activations.shape[-1] == hidden_size
            assert batch_quant_activations.shape == batch_original_activations.shape
            batch_delta = batch_original_activations - batch_quant_activations
            batch_delta = batch_delta.reshape((-1, hidden_size))
            batch_delta = batch_delta.t()
            batch_quant_activations = batch_quant_activations.reshape((-1, hidden_size))
            batch_quant_activations = batch_quant_activations.t()

            tmp = batch_delta.shape[1]  # num_samples = batch_size * seq_len
            delta_hatX *= nsamples / (nsamples + tmp)
            nsamples += tmp

            # Scale and update delta^T hat_X
            batch_delta_scaled = math.sqrt(2 / nsamples) * batch_delta.float()
            batch_quant_activations_scaled = (
                math.sqrt(2 / nsamples) * batch_quant_activations.float()
            )
            delta_hatX += batch_delta_scaled.matmul(batch_quant_activations_scaled.t())

        return delta_hatX

    def calculate_weight_quantization_error(self, module, dequantized_weight):
        """Calculate the weight quantization error for the layer.

        Computes the difference between original and quantized weights.
        Specifically, computes the squared Frobenius norm:

            ||W - Ŵ||²_F

        Where:
            - W: Original weight (out_features, in_features)
            - Ŵ: Dequantized weight

        Args:
            module (torch.nn.Module): The layer module
            dequantized_weight (torch.Tensor): The dequantized weight

        Returns:
            tuple: (weight_squared_error, mean_weight_squared_error,
                    relative_weight_squared_error)
                - weight_squared_error: Value of ||W - Ŵ||²_F
                - mean_weight_squared_error: MSE (weight_squared_error / num_elements)
                - relative_weight_squared_error: ||W - Ŵ||²_F / ||W||²_F (relative error)
        """
        device = module.weight.data.device
        matrix_W = module.weight.data.detach().to(device)
        matrix_W_hat = dequantized_weight.to(device)

        # Compute ||W||²_F (denominator for relative error)
        weight_norm_squared = torch.sum(matrix_W.double().pow(2)).item()

        # Compute ||W - Ŵ||²_F
        weight_diff = (matrix_W - matrix_W_hat).double()
        weight_diff.pow_(2)
        num_elements = weight_diff.numel()
        weight_squared_error = torch.sum(weight_diff).item()

        # MSE = weight_squared_error / num_elements
        mean_weight_squared_error = weight_squared_error / num_elements

        # Relative error = ||W - Ŵ||²_F / ||W||²_F
        relative_weight_squared_error = (
            weight_squared_error / weight_norm_squared if weight_norm_squared > 0 else None
        )

        return weight_squared_error, mean_weight_squared_error, relative_weight_squared_error

    def calculate_output_quantization_error(
        self, module, input, dequantized_weight
    ):  # pylint: disable=redefined-builtin
        """Calculate the output quantization error for the layer.

        Computes the output difference caused by weight quantization.
        Specifically, computes the squared Frobenius norm for input X:

            ||WX^T - ŴX^T||²_F

        Where:
            - W: Original weight (out_features, in_features)
            - Ŵ: Dequantized weight
            - X: Input activations (batch_size * sequence_length, in_features)

        Args:
            module (torch.nn.Module): The layer module
            input (tuple or torch.Tensor): The input to the layer
            dequantized_weight (torch.Tensor): The dequantized weight

        Returns:
            tuple: (output_squared_error, mean_output_squared_error,
                    relative_output_squared_error)
                - output_squared_error: Value of ||WX^T - ŴX^T||²_F
                - mean_output_squared_error: MSE (output_squared_error / num_elements)
                - relative_output_squared_error: ||WX^T - ŴX^T||²_F / ||WX^T||²_F (relative error)
        """

        if input is None:
            return None, None, None

        # Get input activations
        if isinstance(input, tuple):
            input_activations = input[0].detach()
        else:
            input_activations = input.detach()

        assert len(input_activations.shape) == 3  # (batch_size, seq_len, hidden_size)

        device = module.weight.data.device
        matrix_W = module.weight.data.detach()
        dequantized_weight_device = dequantized_weight.to(device)

        # Flatten to (total_samples, in_features)
        flat_input = input_activations.reshape(-1, input_activations.shape[-1])
        total_samples = flat_input.shape[0]

        # Process in batches for memory efficiency
        batch_size = min(total_samples, 512)

        output_squared_error = 0.0
        original_output_norm_squared = 0.0
        num_elements = 0

        for i in range(0, total_samples, batch_size):
            # batch_X_T: (in_features, batch_size)
            batch_X_T = flat_input[i : i + batch_size].T.to(device)

            # batch_original_output: (out_features, batch_size)
            batch_original_output = matrix_W @ batch_X_T  # computed in float16

            # Accumulate ||WX^T||²_F (denominator for relative error)
            original_output_norm_squared += torch.sum(batch_original_output.double().pow(2)).item()

            # batch_diff: (out_features, batch_size)
            # Compute ||WX^T - ŴX^T||²_F
            batch_diff = batch_original_output
            batch_diff -= dequantized_weight_device @ batch_X_T  # computed in float16
            batch_diff = batch_diff.double()  # convert to double
            batch_diff.pow_(2)  # in-place square

            output_squared_error += torch.sum(batch_diff).item()
            num_elements += batch_diff.numel()

            del batch_diff, batch_X_T

        torch.cuda.empty_cache()

        # MSE = output_squared_error / (out_features * total_samples)
        mean_output_squared_error = output_squared_error / num_elements

        # Relative error = ||WX^T - ŴX^T||²_F / ||WX^T||²_F
        relative_output_squared_error = (
            output_squared_error / original_output_norm_squared
            if original_output_norm_squared > 0
            else None
        )

        return output_squared_error, mean_output_squared_error, relative_output_squared_error


@dataclass
class ResultLoader(Quantizer):
    """Loader for reading saved quantization results.

    Does not perform any quantization (`setup()` selects 0 target layers).
    Primary use case is loading pre-saved `results`.

    Example:
        >>> from onecomp.quantizer import ResultLoader
        >>> loader = ResultLoader(results_file="quantization_results.pt")
        >>> loader.results.keys()
    """

    # Optional: load precomputed results at initialization time
    results_file: str = None
    weights_only: bool = False

    # Ensure no layers are selected for quantization by default
    target_layer_types: tuple = field(default_factory=tuple)

    # Explicitly no calibration / no hessian by default
    flag_calibration: bool = False
    flag_hessian: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.results_file is not None:
            self.load_results(self.results_file, weights_only=self.weights_only)

    def setup(self, model):  # pylint: disable=unused-argument
        """Select no layers (no-op).

        Ensures module_to_name is empty, since Runner is expected to call `setup()`.
        """
        self.module_to_name = {}

    def quantize_layer(
        self,
        module,
        input=None,
        hessian=None,  # pylint: disable=redefined-builtin, unused-argument
    ) -> Union[torch.Tensor, QuantizationResult]:
        """Raise error if called.

        ResultLoader is not intended to perform quantization,
        so calling this method raises an error.
        """
        raise RuntimeError(
            "ResultLoader.quantize_layer() should not be called. "
            "This class is intended only for loading precomputed results."
        )
