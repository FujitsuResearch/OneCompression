"""Export Hugging Face Llama/Qwen2 checkpoints to GGUF v3.

Converts a Hugging Face model directory (``config.json`` +
``*.safetensors``) into a single GGUF file consumable by llama.cpp and
Ollama.  Both original FP16/BF16 checkpoints and OneComp save
directories with dequantized (FP16) weights are supported; packed
quantized checkpoints must be saved with dequantized weights first.

Llama-family models use the SentencePiece tokenizer
(``tokenizer.model``) when present, parsed with a minimal protobuf
reader so no ``sentencepiece`` dependency is required.  BPE checkpoints
that ship only ``tokenizer.json`` (Qwen2, Llama-3 style) use the BPE
vocabulary and merges extracted from that file.

Classes:
    GGUFExportConfig: Configuration for GGUF export.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import json
import os
import struct
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open

from .gguf_reader import GGUFReader
from .gguf_writer import GGUFValueType, GGUFWriter

logger = getLogger(__name__)

SUPPORTED_ARCHITECTURES = ("llama", "qwen2")

# GGUF token type codes (same values as the SentencePiece piece types).
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

_DIRECT_TENSOR_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

_LAYER_COMPONENT_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}

_DIRECT_TENSOR_MAP_REVERSE = {v: k for k, v in _DIRECT_TENSOR_MAP.items()}

_LAYER_COMPONENT_MAP_REVERSE = {v: k for k, v in _LAYER_COMPONENT_MAP.items()}

_SKIP_SUFFIXES = ("rotary_emb.inv_freq",)

_QUANTIZED_SUFFIXES = (".qweight", ".qzeros", ".g_idx", ".W_q", ".meta")

# Largest finite float16 value; BF16/FP32 weights outside this range are
# clamped so that the F16 conversion never produces +/-inf.
_F16_MAX = 65504.0


@dataclass
class GGUFExportConfig:
    """Configuration for GGUF export.

    Attributes:
        out_path: Output GGUF file path.
        dtype: Target dtype for weight matrices, ``"f16"`` or ``"f32"``
            (1-D tensors such as norms are always stored as F32).
        architecture: Model architecture (``"llama"`` or ``"qwen2"``).
            Auto-detected from ``config.json`` when ``None``.
        name: Value for the ``general.name`` metadata key.  Defaults to
            the model directory name when ``None``.
    """

    out_path: str
    dtype: str = "f16"
    architecture: Optional[str] = None
    name: Optional[str] = None


def map_tensor_name(hf_name: str) -> Optional[str]:
    """Map a Hugging Face tensor name to its GGUF counterpart.

    Args:
        hf_name (str): Tensor name from the Hugging Face checkpoint
            (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).

    Returns:
        str or None: The GGUF tensor name (e.g. ``"blk.0.attn_q.weight"``),
        or ``None`` for tensors that must be skipped (e.g. rotary
        embedding caches).

    Raises:
        ValueError: If the tensor name is not recognised.
    """
    if hf_name.endswith(_SKIP_SUFFIXES):
        return None
    if hf_name in _DIRECT_TENSOR_MAP:
        return _DIRECT_TENSOR_MAP[hf_name]
    if hf_name.startswith("model.layers."):
        rest = hf_name[len("model.layers.") :]
        layer, _, component = rest.partition(".")
        if layer.isdigit():
            base, _, kind = component.rpartition(".")
            if kind in ("weight", "bias") and base in _LAYER_COMPONENT_MAP:
                return f"blk.{layer}.{_LAYER_COMPONENT_MAP[base]}.{kind}"
    raise ValueError(f"cannot map tensor name to GGUF: {hf_name}")


def reverse_map_tensor_name(gguf_name: str) -> str:
    """Map a GGUF tensor name back to its Hugging Face counterpart.

    This is the inverse of :func:`map_tensor_name` for every name that
    function can produce.

    Args:
        gguf_name (str): GGUF tensor name (e.g. ``"blk.0.attn_q.weight"``).

    Returns:
        str: The Hugging Face tensor name
        (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).

    Raises:
        ValueError: If the tensor name is not recognised.
    """
    if gguf_name in _DIRECT_TENSOR_MAP_REVERSE:
        return _DIRECT_TENSOR_MAP_REVERSE[gguf_name]
    if gguf_name.startswith("blk."):
        rest = gguf_name[len("blk.") :]
        layer, _, component = rest.partition(".")
        if layer.isdigit():
            base, _, kind = component.rpartition(".")
            if kind in ("weight", "bias") and base in _LAYER_COMPONENT_MAP_REVERSE:
                return f"model.layers.{layer}.{_LAYER_COMPONENT_MAP_REVERSE[base]}.{kind}"
    raise ValueError(f"cannot map GGUF tensor name to Hugging Face: {gguf_name}")


def permute_rope_rows(tensor: torch.Tensor, n_head: int) -> torch.Tensor:
    """Reorder Q/K rows from the Hugging Face RoPE layout to llama.cpp's.

    llama.cpp applies interleaved ("NORM") RoPE for the ``llama``
    architecture, while Hugging Face checkpoints store Q/K for the
    half-split ("rotate_half") convention.  This is the same row
    permutation that llama.cpp's ``convert_hf_to_gguf.py`` applies to
    ``attn_q``/``attn_k`` weights and biases.

    Args:
        tensor (torch.Tensor): Q or K projection ``weight`` (2-D) or
            ``bias`` (1-D) whose first dimension is
            ``n_head * head_dim``.
        n_head (int): Number of attention heads for this projection
            (``num_attention_heads`` for Q, ``num_key_value_heads``
            for K).

    Returns:
        torch.Tensor: Tensor with permuted rows, same shape and dtype.
    """
    head_dim = tensor.shape[0] // n_head
    view = tensor.reshape(n_head, 2, head_dim // 2, *tensor.shape[1:])
    return view.swapaxes(1, 2).reshape(tensor.shape)


def unpermute_rope_rows(tensor: torch.Tensor, n_head: int) -> torch.Tensor:
    """Invert :func:`permute_rope_rows` (llama.cpp layout back to HF).

    Args:
        tensor (torch.Tensor): Permuted Q or K projection tensor.
        n_head (int): Number of attention heads for this projection.

    Returns:
        torch.Tensor: Tensor with the original Hugging Face row order.
    """
    head_dim = tensor.shape[0] // n_head
    view = tensor.reshape(n_head, head_dim // 2, 2, *tensor.shape[1:])
    return view.swapaxes(1, 2).reshape(tensor.shape)


def load_gguf_state_dict(gguf_path: str) -> Dict[str, torch.Tensor]:
    """Load a GGUF file back into a Hugging Face style state dict.

    Reads every tensor of an F16/F32 GGUF file produced by
    :func:`export_gguf` and maps the names back to the Hugging Face
    convention, so the result can be loaded into a ``transformers``
    model with ``model.load_state_dict(..., strict=False)`` (models
    with ``tie_word_embeddings`` have no ``lm_head.weight`` entry).

    Args:
        gguf_path (str): Path of the GGUF file.

    Returns:
        Dict[str, torch.Tensor]: Hugging Face named tensors with the
        dtypes stored in the file (F16 matrices, F32 norms).

    Examples:
        >>> from onecomp.export import load_gguf_state_dict
        >>> state_dict = load_gguf_state_dict("./tinyllama-f16.gguf")
        >>> state_dict["model.embed_tokens.weight"].shape
        torch.Size([32000, 2048])
    """
    reader = GGUFReader(gguf_path)
    arch = reader.metadata.get("general.architecture")
    n_head = int(reader.metadata.get(f"{arch}.attention.head_count", 0))
    n_head_kv = int(reader.metadata.get(f"{arch}.attention.head_count_kv", n_head))

    state_dict: Dict[str, torch.Tensor] = {}
    for info in reader.tensors:
        hf_name = reverse_map_tensor_name(info.name)
        tensor = torch.from_numpy(reader.read_tensor(info.name).copy())
        # llama GGUF files store Q/K in the interleaved-RoPE row order;
        # undo the permutation applied by export_gguf.
        if arch == "llama":
            if ".attn_q." in info.name:
                tensor = unpermute_rope_rows(tensor, n_head)
            elif ".attn_k." in info.name:
                tensor = unpermute_rope_rows(tensor, n_head_kv)
        state_dict[hf_name] = tensor
    return state_dict


def export_gguf(model_dir: str, config: GGUFExportConfig) -> str:
    """Convert a Hugging Face model directory to a GGUF v3 file.

    Args:
        model_dir (str): Directory containing ``config.json`` and
            safetensors weights (an original checkpoint or a OneComp
            save directory with dequantized FP16 weights).
        config (GGUFExportConfig): Export configuration.

    Returns:
        str: Path of the written GGUF file.

    Raises:
        ValueError: If the architecture is unsupported, the checkpoint
            contains packed quantized weights, or a tensor name cannot
            be mapped.

    Examples:
        >>> from onecomp.export import GGUFExportConfig, export_gguf
        >>> export_gguf(
        ...     "./TinyLlama-1.1B-Chat-v1.0",
        ...     GGUFExportConfig(out_path="./tinyllama-f16.gguf"),
        ... )
        './tinyllama-f16.gguf'
    """
    if config.dtype not in ("f16", "f32"):
        raise ValueError(f"unsupported target dtype: {config.dtype}")

    hf_config = _load_json(os.path.join(model_dir, "config.json"))
    architecture = config.architecture or hf_config.get("model_type")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"unsupported architecture {architecture!r}; "
            f"supported: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )

    name = config.name or os.path.basename(os.path.normpath(model_dir))
    writer = GGUFWriter()
    _add_model_metadata(writer, architecture, name, hf_config, config.dtype)
    _add_tokenizer_metadata(writer, architecture, model_dir, hf_config)

    tie_word_embeddings = bool(hf_config.get("tie_word_embeddings", False))
    weight_dtype = np.float16 if config.dtype == "f16" else np.float32
    head_count = int(hf_config["num_attention_heads"])
    head_count_kv = int(hf_config.get("num_key_value_heads", head_count))
    tensor_count = 0
    for hf_name, tensor in _iter_safetensors(model_dir):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            logger.debug("Skipping tensor %s", hf_name)
            continue
        if gguf_name == "output.weight" and tie_word_embeddings:
            # llama.cpp falls back to token_embd.weight automatically.
            logger.debug("Skipping tied output.weight (%s)", hf_name)
            continue
        if architecture == "llama":
            # llama.cpp applies interleaved RoPE for this architecture,
            # so Q/K rows must be permuted (qwen2 uses NEOX RoPE and is
            # stored unchanged).
            if ".attn_q." in gguf_name:
                tensor = permute_rope_rows(tensor, head_count)
            elif ".attn_k." in gguf_name:
                tensor = permute_rope_rows(tensor, head_count_kv)
        dtype = np.float32 if tensor.dim() == 1 else weight_dtype
        array = tensor.to(torch.float32)
        if dtype == np.float16:
            # BF16/FP32 can represent values outside the F16 range;
            # clamp so the narrowing cast never produces +/-inf.
            array = array.clamp_(-_F16_MAX, _F16_MAX).to(torch.float16)
        writer.add_tensor(gguf_name, array.numpy().astype(dtype, copy=False))
        tensor_count += 1

    out_dir = os.path.dirname(os.path.abspath(config.out_path))
    os.makedirs(out_dir, exist_ok=True)
    writer.write(config.out_path)
    logger.info(
        "Exported %s (%s) to %s: %d tensors, %.2f GB",
        name,
        architecture,
        config.out_path,
        tensor_count,
        os.path.getsize(config.out_path) / 1024**3,
    )
    return config.out_path


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _add_model_metadata(
    writer: GGUFWriter, arch: str, name: str, hf_config: Dict[str, Any], dtype: str = "f16"
):
    """Write ``general.*`` and ``<arch>.*`` metadata keys."""
    hidden_size = int(hf_config["hidden_size"])
    head_count = int(hf_config["num_attention_heads"])

    writer.add_metadata("general.architecture", arch)
    writer.add_metadata("general.name", name)
    # file_type: 0 = all F32, 1 = mostly F16.
    writer.add_metadata("general.file_type", 1 if dtype == "f16" else 0)
    writer.add_metadata("general.quantization_version", 2)
    writer.add_metadata(f"{arch}.context_length", int(hf_config["max_position_embeddings"]))
    writer.add_metadata(f"{arch}.embedding_length", hidden_size)
    writer.add_metadata(f"{arch}.block_count", int(hf_config["num_hidden_layers"]))
    writer.add_metadata(f"{arch}.feed_forward_length", int(hf_config["intermediate_size"]))
    writer.add_metadata(f"{arch}.attention.head_count", head_count)
    writer.add_metadata(
        f"{arch}.attention.head_count_kv",
        int(hf_config.get("num_key_value_heads", head_count)),
    )
    writer.add_metadata(
        f"{arch}.attention.layer_norm_rms_epsilon",
        float(hf_config.get("rms_norm_eps", 1e-5)),
    )
    writer.add_metadata(f"{arch}.rope.freq_base", float(hf_config.get("rope_theta", 10000.0)))
    writer.add_metadata(f"{arch}.vocab_size", int(hf_config["vocab_size"]))
    # Some architectures (e.g. Gemma) set an explicit head_dim that differs
    # from hidden_size // head_count; prefer it when present.
    head_dim = int(hf_config.get("head_dim") or hidden_size // head_count)
    writer.add_metadata(f"{arch}.rope.dimension_count", head_dim)

    rope_scaling = hf_config.get("rope_scaling") or {}
    scaling_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if scaling_type is not None:
        logger.warning(
            "config.json declares rope_scaling type %r, which this exporter "
            "does not translate to GGUF; long-context behaviour under "
            "llama.cpp will differ from the HF model",
            scaling_type,
        )


def _add_tokenizer_metadata(
    writer: GGUFWriter, arch: str, model_dir: str, hf_config: Dict[str, Any]
):
    """Write ``tokenizer.ggml.*`` metadata extracted from the model dir."""
    vocab_size = int(hf_config["vocab_size"])
    sp_path = os.path.join(model_dir, "tokenizer.model")

    if os.path.isfile(sp_path):
        tokens, scores, token_types = _parse_sentencepiece_model(sp_path)
        _merge_added_tokens(model_dir, tokens, scores, token_types)
        _check_vocab_fits(tokens, vocab_size)
        _pad_vocab(tokens, scores, token_types, vocab_size)
        writer.add_metadata("tokenizer.ggml.model", "llama")
        writer.add_metadata("tokenizer.ggml.tokens", tokens)
        writer.add_metadata("tokenizer.ggml.scores", scores, element_type=GGUFValueType.FLOAT32)
        writer.add_metadata(
            "tokenizer.ggml.token_type", token_types, element_type=GGUFValueType.INT32
        )
        if TOKEN_TYPE_UNKNOWN in token_types:
            writer.add_metadata(
                "tokenizer.ggml.unknown_token_id", token_types.index(TOKEN_TYPE_UNKNOWN)
            )
    elif os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
        # BPE checkpoints (Qwen2, Llama-3, ...) ship tokenizer.json only.
        tokens, token_types, merges = _load_bpe_tokenizer(
            os.path.join(model_dir, "tokenizer.json")
        )
        _check_vocab_fits(tokens, vocab_size)
        _pad_vocab(tokens, None, token_types, vocab_size)
        writer.add_metadata("tokenizer.ggml.model", "gpt2")
        writer.add_metadata("tokenizer.ggml.pre", _BPE_PRE_TYPES[arch])
        writer.add_metadata("tokenizer.ggml.tokens", tokens)
        writer.add_metadata(
            "tokenizer.ggml.token_type", token_types, element_type=GGUFValueType.INT32
        )
        writer.add_metadata("tokenizer.ggml.merges", merges, element_type=GGUFValueType.STRING)
    else:
        raise ValueError(f"no supported tokenizer found in {model_dir}")

    bos_token_id = hf_config.get("bos_token_id")
    eos_token_id = hf_config.get("eos_token_id")
    extra_eos_ids: List[int] = []
    if isinstance(eos_token_id, list):
        eos_token_id, extra_eos_ids = eos_token_id[0], [int(t) for t in eos_token_id[1:]]
    if bos_token_id is not None:
        writer.add_metadata("tokenizer.ggml.bos_token_id", int(bos_token_id))
    if eos_token_id is not None:
        writer.add_metadata("tokenizer.ggml.eos_token_id", int(eos_token_id))
    for token_id in extra_eos_ids:
        # llama.cpp stops on eos / eot / eom; map the additional stop ids
        # (e.g. Llama-3.1's <|eot_id|> / <|eom_id|>) onto the matching keys
        # so multi-EOS models still terminate turns correctly.
        text = tokens[token_id] if 0 <= token_id < len(tokens) else ""
        if "eot" in text:
            writer.add_metadata("tokenizer.ggml.eot_token_id", token_id)
        elif "eom" in text:
            writer.add_metadata("tokenizer.ggml.eom_token_id", token_id)
        else:
            logger.warning(
                "Extra EOS token id %d (%r) has no GGUF stop-token key; "
                "generation may not stop on it under llama.cpp",
                token_id,
                text,
            )

    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.isfile(tokenizer_config_path):
        tokenizer_config = _load_json(tokenizer_config_path)
        for key in ("add_bos_token", "add_eos_token"):
            value = tokenizer_config.get(key)
            if isinstance(value, bool):
                writer.add_metadata(f"tokenizer.ggml.{key}", value)
        chat_template = tokenizer_config.get("chat_template")
        if isinstance(chat_template, str):
            writer.add_metadata("tokenizer.chat_template", chat_template)


def _merge_added_tokens(
    model_dir: str,
    tokens: List[str],
    scores: List[float],
    token_types: List[int],
):
    """Append tokenizer added tokens beyond the SentencePiece vocabulary.

    Hugging Face checkpoints may extend the base SentencePiece vocabulary
    with added tokens (``added_tokens`` in ``tokenizer.json`` or
    ``added_tokens.json``).  Ids already covered by the SentencePiece
    model are left untouched; ids past the end are appended so they are
    not replaced by ``[PAD]`` placeholders.
    """
    added: List[Dict[str, Any]] = []
    tokenizer_json_path = os.path.join(model_dir, "tokenizer.json")
    added_tokens_path = os.path.join(model_dir, "added_tokens.json")
    if os.path.isfile(tokenizer_json_path):
        added = _load_json(tokenizer_json_path).get("added_tokens", [])
    elif os.path.isfile(added_tokens_path):
        added = [
            {"id": token_id, "content": content, "special": False}
            for content, token_id in _load_json(added_tokens_path).items()
        ]

    for entry in sorted(added, key=lambda item: item["id"]):
        token_id = int(entry["id"])
        if token_id < len(tokens):
            continue  # already present in the SentencePiece vocabulary
        while len(tokens) < token_id:
            tokens.append(f"[PAD{len(tokens)}]")
            scores.append(0.0)
            token_types.append(TOKEN_TYPE_UNUSED)
        tokens.append(entry["content"])
        scores.append(0.0)
        token_types.append(TOKEN_TYPE_CONTROL if entry.get("special") else TOKEN_TYPE_USER_DEFINED)
        logger.debug("Merged added token %d: %r", token_id, entry["content"])


# tokenizer.ggml.pre selects the llama.cpp pre-tokenizer regex; it must match
# the model family, not just the BPE format.
_BPE_PRE_TYPES = {
    "qwen2": "qwen2",
    "llama": "llama-bpe",
}


def _check_vocab_fits(tokens: List[str], vocab_size: int):
    """Reject tokenizers larger than the model embedding table.

    Token ids at or above ``config.json``'s ``vocab_size`` have no
    corresponding embedding row, so the exported GGUF would either crash or
    silently mis-tokenize under llama.cpp.
    """
    if len(tokens) > vocab_size:
        raise ValueError(
            f"tokenizer defines {len(tokens)} tokens but config.json declares "
            f"vocab_size={vocab_size}; ids >= {vocab_size} have no embedding row"
        )


def _pad_vocab(
    tokens: List[str],
    scores: Optional[List[float]],
    token_types: List[int],
    vocab_size: int,
):
    """Pad the token lists with placeholder entries up to ``vocab_size``."""
    for i in range(len(tokens), vocab_size):
        tokens.append(f"[PAD{i}]")
        token_types.append(TOKEN_TYPE_UNUSED)
        if scores is not None:
            scores.append(0.0)


def _iter_safetensors(model_dir: str) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` pairs from the safetensors checkpoint.

    Handles both single-file (``model.safetensors``) and sharded
    (``model.safetensors.index.json``) checkpoints.

    Raises:
        ValueError: If no safetensors weights are found, or the
            checkpoint contains packed quantized tensors.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        index = _load_json(index_path)
        shards = sorted(set(index["weight_map"].values()))
    elif os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        shards = ["model.safetensors"]
    else:
        raise ValueError(f"no safetensors weights found in {model_dir}")

    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            packed = [k for k in keys if k.endswith(_QUANTIZED_SUFFIXES)]
            if packed:
                raise ValueError(
                    "checkpoint contains packed quantized tensors "
                    f"(e.g. {packed[0]}); save the model with dequantized "
                    "FP16 weights first (Runner.save_dequantized_model)"
                )
            for key in keys:
                yield key, f.get_tensor(key)


def _parse_sentencepiece_model(path: str) -> Tuple[List[str], List[float], List[int]]:
    """Extract vocabulary from a SentencePiece ``tokenizer.model`` file.

    The file is a serialized ``sentencepiece.ModelProto``; only the
    repeated ``pieces`` field (field number 1) is decoded, using a
    minimal protobuf wire-format parser so the ``sentencepiece``
    package is not required.

    Returns:
        tuple: ``(tokens, scores, token_types)`` lists, index-aligned
        with the SentencePiece vocabulary ids.
    """
    with open(path, "rb") as f:
        data = f.read()

    tokens: List[str] = []
    scores: List[float] = []
    token_types: List[int] = []
    for field_number, wire_type, value in _iter_protobuf_fields(data):
        if field_number != 1 or wire_type != 2:
            continue
        piece, score, piece_type = b"", 0.0, TOKEN_TYPE_NORMAL
        for sub_number, sub_wire, sub_value in _iter_protobuf_fields(value):
            if sub_number == 1 and sub_wire == 2:
                piece = sub_value
            elif sub_number == 2 and sub_wire == 5:
                (score,) = struct.unpack("<f", sub_value)
            elif sub_number == 3 and sub_wire == 0:
                piece_type = sub_value
        tokens.append(piece.decode("utf-8", errors="replace"))
        scores.append(score)
        token_types.append(piece_type)
    if not tokens:
        raise ValueError(f"no SentencePiece vocabulary found in {path}")
    return tokens, scores, token_types


def _iter_protobuf_fields(data: bytes) -> Iterator[Tuple[int, int, Any]]:
    """Iterate over top-level protobuf fields in ``data``.

    Yields:
        tuple: ``(field_number, wire_type, value)`` where the value is
        an int for varints, and raw bytes for fixed-width and
        length-delimited fields.
    """
    pos = 0
    end = len(data)
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            value, pos = _read_varint(data, pos)
        elif wire_type == 1:  # fixed64
            value, pos = data[pos : pos + 8], pos + 8
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire_type == 5:  # fixed32
            value, pos = data[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        yield field_number, wire_type, value


def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode a protobuf varint starting at ``pos``."""
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _load_bpe_tokenizer(path: str) -> Tuple[List[str], List[int], List[str]]:
    """Extract a BPE vocabulary from a Hugging Face ``tokenizer.json``.

    Returns:
        tuple: ``(tokens, token_types, merges)`` where tokens are
        ordered by id and merges are ``"left right"`` strings.
    """
    tokenizer = _load_json(path)
    vocab: Dict[str, int] = tokenizer["model"]["vocab"]
    raw_merges = tokenizer["model"]["merges"]
    merges = [" ".join(m) if isinstance(m, list) else m for m in raw_merges]

    size = max(vocab.values()) + 1
    added_tokens = tokenizer.get("added_tokens", [])
    if added_tokens:
        size = max(size, max(t["id"] for t in added_tokens) + 1)

    tokens = [f"[PAD{i}]" for i in range(size)]
    token_types = [TOKEN_TYPE_UNUSED] * size
    for token, token_id in vocab.items():
        tokens[token_id] = token
        token_types[token_id] = TOKEN_TYPE_NORMAL
    for added in added_tokens:
        tokens[added["id"]] = added["content"]
        token_types[added["id"]] = (
            TOKEN_TYPE_CONTROL if added.get("special") else TOKEN_TYPE_USER_DEFINED
        )
    return tokens, token_types, merges
