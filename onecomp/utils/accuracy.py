"""

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

from logging import getLogger

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


def calculate_accuracy(
    model=None,
    tokenizer=None,
    model_config=None,
    tasks=None,
    batch_size=8,
    num_fewshot=0,
    display_results=True,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-branches
    """Calculate the accuracy of the model

    Args:
        model: The model to evaluate. If None, model_config must be provided.
        tokenizer: The tokenizer to use. If None, model_config must be provided.
        model_config: The model configuration. Used if model or tokenizer is None.
        tasks (list): The list of tasks to evaluate.
            Default: ["arc_easy", "arc_challenge", "piqa", "winogrande"]
        batch_size (int): The batch size for evaluation.
        num_fewshot (int): The number of few-shot examples.
        display_results (bool): Whether to display the results.

    Example:
        >>> from onecomp import ModelConfig, calculate_accuracy
        >>> model_config = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        >>> calculate_accuracy(model_config=model_config)
        >>>
        >>> # Or with model and tokenizer directly
        >>> model = model_config.load_model()
        >>> tokenizer = model_config.load_tokenizer()
        >>> calculate_accuracy(model=model, tokenizer=tokenizer)

    """

    eval_model = None

    # create a `model` and `tokenizer` object from the model config
    if model is None:
        if model_config is None:
            raise ValueError(
                "model_config must be provided if model is not provided"
            )
        if model_config.has_additional_data():
            model = model_config.load_model()
        else:
            # Use model_id or path directly with HFLM
            eval_model = HFLM(
                pretrained=model_config.get_model_id_or_path(),
                device=model_config.device,
                dtype=model_config.dtype,
                batch_size=batch_size,
            )
            model = None  # Signal that eval_model is already created

    if model is not None:
        if tokenizer is None:
            if model_config is None:
                raise ValueError(
                    "model_config must be provided if tokenizer is not provided"
                )
            tokenizer = model_config.load_tokenizer()
        eval_model = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size
        )

    # calculate the accuracy
    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "piqa", "winogrande"]

    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
    )

    if display_results:
        logger = getLogger(__name__)
        logger.info("=" * 50)
        for task, metrics in results["results"].items():
            logger.info("Task: %s", task)
            for metric_name, value in metrics.items():
                if not metric_name.startswith("_"):
                    if isinstance(value, (float, int)):
                        logger.info("  %s: %.4f", metric_name, value)
                    else:
                        logger.info("  %s: %s", metric_name, value)
        logger.info("=" * 50)

    return results["results"]
