"""Lightweight GGUF v3 reader for testing and validation.

Parses the GGUF header, metadata key/value pairs, and the tensor
directory (name/shape/dtype/offset) without loading tensor data into
memory.  Individual tensors can be read on demand for verification.
This module is intended for validating files produced by
``onecomp.export.gguf_writer``, not as a general-purpose GGUF loader.

Classes:
    GGUFTensorInfo: Tensor directory entry (name, shape, type, offset).
    GGUFReader: Reader for GGUF v3 headers, metadata, and tensors.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .gguf_writer import GGUF_MAGIC, GGMLQuantType, GGUFValueType

_SCALAR_UNPACK_FORMATS = {
    GGUFValueType.UINT8: ("<B", 1),
    GGUFValueType.INT8: ("<b", 1),
    GGUFValueType.UINT16: ("<H", 2),
    GGUFValueType.INT16: ("<h", 2),
    GGUFValueType.UINT32: ("<I", 4),
    GGUFValueType.INT32: ("<i", 4),
    GGUFValueType.FLOAT32: ("<f", 4),
    GGUFValueType.UINT64: ("<Q", 8),
    GGUFValueType.INT64: ("<q", 8),
    GGUFValueType.FLOAT64: ("<d", 8),
}

_GGML_TO_NUMPY = {
    GGMLQuantType.F32: np.dtype(np.float32),
    GGMLQuantType.F16: np.dtype(np.float16),
}


@dataclass
class GGUFTensorInfo:
    """Tensor directory entry of a GGUF file.

    Attributes:
        name: GGUF tensor name.
        shape: Tensor shape in numpy (row-major) order.
        ggml_type: GGML data type code.
        offset: Byte offset relative to the start of the data section.
    """

    name: str
    shape: Tuple[int, ...]
    ggml_type: GGMLQuantType
    offset: int

    @property
    def nbytes(self) -> int:
        """Size of the tensor data in bytes."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count * _GGML_TO_NUMPY[self.ggml_type].itemsize


class GGUFReader:
    """Reader for GGUF v3 files.

    Parses the header, all metadata, and the tensor directory on
    construction, and validates tensor offsets against the file size
    and alignment.

    Attributes:
        version: GGUF format version found in the header.
        metadata: Decoded metadata key/value pairs.
        tensors: Tensor directory entries in file order.
        alignment: Data alignment (``general.alignment``, default 32).
        data_start: Absolute file offset of the tensor data section.

    Examples:
        >>> reader = GGUFReader("model-f16.gguf")
        >>> reader.metadata["general.architecture"]
        'llama'
        >>> reader.tensors[0].name
        'token_embd.weight'
    """

    def __init__(self, path: str):
        """
        Args:
            path (str): GGUF file to parse.

        Raises:
            ValueError: If the magic, version, offsets, or alignment
                are invalid.
        """
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[GGUFTensorInfo] = []

        with open(path, "rb") as f:
            magic, version = struct.unpack("<II", f.read(8))
            if magic != GGUF_MAGIC:
                raise ValueError(f"not a GGUF file (magic=0x{magic:08X}): {path}")
            if version != 3:
                raise ValueError(f"unsupported GGUF version {version}: {path}")
            self.version = version

            tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            for _ in range(kv_count):
                key = self._read_string(f)
                self.metadata[key] = self._read_value(f)

            for _ in range(tensor_count):
                name = self._read_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                ggml_type_raw, offset = struct.unpack("<IQ", f.read(12))
                self.tensors.append(
                    GGUFTensorInfo(
                        name=name,
                        shape=tuple(reversed(dims)),  # ne order -> numpy order
                        ggml_type=GGMLQuantType(ggml_type_raw),
                        offset=offset,
                    )
                )

            self.alignment = int(self.metadata.get("general.alignment", 32))
            header_end = f.tell()
            self.data_start = (header_end + self.alignment - 1) // self.alignment * self.alignment

            f.seek(0, 2)
            self._file_size = f.tell()

        self._validate()

    def tensor(self, name: str) -> GGUFTensorInfo:
        """Look up a tensor directory entry by name.

        Args:
            name (str): GGUF tensor name.

        Returns:
            GGUFTensorInfo: The matching directory entry.

        Raises:
            KeyError: If the tensor is not present.
        """
        for info in self.tensors:
            if info.name == name:
                return info
        raise KeyError(f"tensor not found in {self.path}: {name}")

    def read_tensor(self, name: str) -> np.ndarray:
        """Read the data of a single tensor from the file.

        Args:
            name (str): GGUF tensor name.

        Returns:
            np.ndarray: Tensor data with the shape and dtype recorded
            in the tensor directory.
        """
        info = self.tensor(name)
        dtype = _GGML_TO_NUMPY[info.ggml_type]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + info.offset)
            data = f.read(info.nbytes)
        return np.frombuffer(data, dtype=dtype).reshape(info.shape)

    def _validate(self):
        """Check tensor offsets for alignment and file-size consistency."""
        for info in self.tensors:
            if info.offset % self.alignment != 0:
                raise ValueError(
                    f"tensor {info.name} offset {info.offset} is not aligned "
                    f"to {self.alignment} bytes"
                )
            end = self.data_start + info.offset + info.nbytes
            if end > self._file_size:
                raise ValueError(
                    f"tensor {info.name} extends past end of file " f"({end} > {self._file_size})"
                )

    @staticmethod
    def _read_string(f) -> str:
        """Read a GGUF string (u64 length + UTF-8 bytes)."""
        (length,) = struct.unpack("<Q", f.read(8))
        return f.read(length).decode("utf-8")

    def _read_value(self, f) -> Any:
        """Read a typed metadata value (type tag + payload)."""
        (value_type_raw,) = struct.unpack("<I", f.read(4))
        return self._read_payload(f, GGUFValueType(value_type_raw))

    def _read_payload(self, f, value_type: GGUFValueType) -> Any:
        """Read the payload of a metadata value without the type tag."""
        if value_type == GGUFValueType.STRING:
            return self._read_string(f)
        if value_type == GGUFValueType.BOOL:
            return f.read(1) != b"\x00"
        if value_type == GGUFValueType.ARRAY:
            elem_type_raw, count = struct.unpack("<IQ", f.read(12))
            elem_type = GGUFValueType(elem_type_raw)
            return [self._read_payload(f, elem_type) for _ in range(count)]
        fmt, size = _SCALAR_UNPACK_FORMATS[value_type]
        (value,) = struct.unpack(fmt, f.read(size))
        return value
