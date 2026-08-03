"""GGUF v3 binary writer implemented with the standard library and numpy.

Writes GGUF files (magic ``GGUF``, version 3, little-endian) that can be
consumed by llama.cpp-based runtimes such as Ollama.  Only the FP32/FP16
tensor types are supported, which is sufficient for exporting dequantized
OneComp models.

Classes:
    GGUFValueType: Metadata value type codes defined by the GGUF spec.
    GGMLQuantType: Tensor data type codes defined by the GGUF spec.
    GGUFWriter: Incremental writer for GGUF v3 files.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import struct
from enum import IntEnum
from logging import getLogger
from typing import Any, List, Optional, Tuple

import numpy as np

logger = getLogger(__name__)

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    """Metadata value type codes defined by the GGUF specification."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantType(IntEnum):
    """Tensor data type codes defined by the GGUF specification."""

    F32 = 0
    F16 = 1


_SCALAR_PACK_FORMATS = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}

_NUMPY_TO_GGML = {
    np.dtype(np.float32): GGMLQuantType.F32,
    np.dtype(np.float16): GGMLQuantType.F16,
}


class GGUFWriter:
    """Incremental writer for GGUF v3 files.

    Metadata key/value pairs and tensors are collected in memory with
    ``add_metadata`` / ``add_tensor`` and serialized with ``write``.
    Tensor data is placed after the header, padded to ``alignment``
    bytes as required by the specification.

    Examples:
        >>> writer = GGUFWriter()
        >>> writer.add_metadata("general.architecture", "llama")
        >>> writer.add_tensor("token_embd.weight", np.zeros((8, 4), np.float16))
        >>> writer.write("model-f16.gguf")
    """

    def __init__(self, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        """
        Args:
            alignment (int): Tensor data alignment in bytes
                (default: 32, written as ``general.alignment``).
        """
        self.alignment = alignment
        self._metadata: List[Tuple[str, Any, GGUFValueType, Optional[GGUFValueType]]] = []
        self._metadata_keys: set = set()
        self._tensors: List[Tuple[str, np.ndarray, GGMLQuantType]] = []
        self._tensor_names: set = set()

    def add_metadata(
        self,
        key: str,
        value: Any,
        value_type: Optional[GGUFValueType] = None,
        element_type: Optional[GGUFValueType] = None,
    ):
        """Add a metadata key/value pair.

        Args:
            key (str): Metadata key (e.g. ``"general.architecture"``).
            value: Scalar (bool/int/float/str) or homogeneous list.
            value_type (GGUFValueType or None): Explicit value type.
                When ``None``, the type is inferred from ``value``
                (int -> UINT32/UINT64/INT64, float -> FLOAT32, etc.).
            element_type (GGUFValueType or None): Element type for
                ARRAY values (e.g. ``GGUFValueType.INT32`` for
                ``tokenizer.ggml.token_type``).  Inferred from the
                first element when ``None``.

        Raises:
            ValueError: If the key is duplicated or the type cannot
                be inferred.
        """
        if key in self._metadata_keys:
            raise ValueError(f"duplicate metadata key: {key}")
        if value_type is None:
            value_type = self._infer_value_type(value)
        if key == "general.alignment":
            # Keep the padding behaviour in sync with the metadata value.
            self.alignment = int(value)
        self._metadata.append((key, value, value_type, element_type))
        self._metadata_keys.add(key)

    def add_tensor(self, name: str, array: np.ndarray):
        """Add a tensor to be written to the data section.

        Args:
            name (str): GGUF tensor name (e.g. ``"blk.0.attn_q.weight"``).
            array (np.ndarray): Tensor data of dtype float32 or float16.
                The numpy shape is stored in reversed (ne) order as
                required by the specification.

        Raises:
            ValueError: If the name is duplicated or the dtype is
                unsupported.
        """
        if name in self._tensor_names:
            raise ValueError(f"duplicate tensor name: {name}")
        ggml_type = _NUMPY_TO_GGML.get(array.dtype)
        if ggml_type is None:
            raise ValueError(f"unsupported tensor dtype for GGUF export: {array.dtype} ({name})")
        self._tensors.append((name, np.ascontiguousarray(array), ggml_type))
        self._tensor_names.add(name)

    def write(self, path: str):
        """Serialize the collected metadata and tensors to ``path``.

        Args:
            path (str): Output file path.
        """
        metadata = list(self._metadata)
        if "general.alignment" not in self._metadata_keys:
            metadata.append(("general.alignment", self.alignment, GGUFValueType.UINT32, None))

        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self._tensors)))
            f.write(struct.pack("<Q", len(metadata)))

            for key, value, value_type, element_type in metadata:
                self._write_string(f, key)
                self._write_value(f, value, value_type, element_type)

            offset = 0
            for name, array, ggml_type in self._tensors:
                self._write_string(f, name)
                dims = array.shape[::-1]  # ne order: fastest-varying dim first
                f.write(struct.pack("<I", len(dims)))
                for dim in dims:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", int(ggml_type)))
                f.write(struct.pack("<Q", offset))
                offset = self._align(offset + array.nbytes)

            self._pad_to_alignment(f)
            for i, (name, array, _) in enumerate(self._tensors):
                f.write(array.tobytes())
                if i < len(self._tensors) - 1:
                    self._pad_to_alignment(f)

        logger.info(
            "Wrote GGUF file %s (%d tensors, %d metadata entries)",
            path,
            len(self._tensors),
            len(metadata),
        )

    def _align(self, offset: int) -> int:
        """Round ``offset`` up to the next multiple of the alignment."""
        return (offset + self.alignment - 1) // self.alignment * self.alignment

    def _pad_to_alignment(self, f):
        """Pad the file with zero bytes up to the next aligned position."""
        pad = self._align(f.tell()) - f.tell()
        if pad:
            f.write(b"\x00" * pad)

    @staticmethod
    def _infer_value_type(value: Any) -> GGUFValueType:
        """Infer the GGUF value type from a Python value."""
        if isinstance(value, bool):
            return GGUFValueType.BOOL
        if isinstance(value, int):
            if 0 <= value < 2**32:
                return GGUFValueType.UINT32
            if value >= 0:
                return GGUFValueType.UINT64
            return GGUFValueType.INT64
        if isinstance(value, float):
            return GGUFValueType.FLOAT32
        if isinstance(value, str):
            return GGUFValueType.STRING
        if isinstance(value, (list, tuple)):
            return GGUFValueType.ARRAY
        raise ValueError(f"cannot infer GGUF value type for {type(value)!r}")

    def _write_string(self, f, value: str):
        """Write a GGUF string (u64 length + UTF-8 bytes)."""
        data = value.encode("utf-8")
        f.write(struct.pack("<Q", len(data)))
        f.write(data)

    def _write_value(
        self,
        f,
        value: Any,
        value_type: GGUFValueType,
        element_type: Optional[GGUFValueType] = None,
    ):
        """Write a typed metadata value (type tag + payload)."""
        f.write(struct.pack("<I", int(value_type)))
        self._write_payload(f, value, value_type, element_type)

    def _write_payload(
        self,
        f,
        value: Any,
        value_type: GGUFValueType,
        element_type: Optional[GGUFValueType] = None,
    ):
        """Write the payload of a metadata value without the type tag."""
        if value_type == GGUFValueType.STRING:
            self._write_string(f, value)
        elif value_type == GGUFValueType.BOOL:
            f.write(struct.pack("<B", 1 if value else 0))
        elif value_type == GGUFValueType.ARRAY:
            elem_type = element_type
            if elem_type is None:
                if len(value) == 0:
                    raise ValueError(
                        "cannot infer the element type of an empty GGUF array; "
                        "pass element_type explicitly"
                    )
                elem_type = self._infer_value_type(value[0])
                # Promote integer arrays to a width shared by all elements.
                if elem_type in (GGUFValueType.UINT32, GGUFValueType.UINT64):
                    if any(v < 0 for v in value):
                        elem_type = GGUFValueType.INT64
                    elif any(v >= 2**32 for v in value):
                        elem_type = GGUFValueType.UINT64
            if elem_type == GGUFValueType.ARRAY:
                raise ValueError("nested GGUF arrays are not supported")
            f.write(struct.pack("<I", int(elem_type)))
            f.write(struct.pack("<Q", len(value)))
            for elem in value:
                self._write_payload(f, elem, elem_type)
        else:
            fmt = _SCALAR_PACK_FORMATS[value_type]
            f.write(struct.pack(fmt, value))
