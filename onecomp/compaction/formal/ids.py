"""Fixed-width, occurrence-addressed identifiers for formal ARC.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .errors import ArchiveCapacityExceeded

_HEX_RE = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class OccurrenceIdConfig:
    """Permanent identifier namespace configuration."""

    max_occurrences: int = 1 << 48
    fingerprint_bits: int = 64

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_occurrences, int)
            or isinstance(self.max_occurrences, bool)
            or not 1 <= self.max_occurrences <= (1 << 63)
        ):
            raise ValueError("max_occurrences must be an integer in [1, 2**63]")
        if (
            not isinstance(self.fingerprint_bits, int)
            or isinstance(self.fingerprint_bits, bool)
            or self.fingerprint_bits % 8
            or not 16 <= self.fingerprint_bits <= 256
        ):
            raise ValueError("fingerprint_bits must be a multiple of 8 between 16 and 256")

    @property
    def index_hex_chars(self) -> int:
        bits = max(1, (self.max_occurrences - 1).bit_length())
        return max(1, (bits + 3) // 4)

    @property
    def fingerprint_hex_chars(self) -> int:
        return self.fingerprint_bits // 4

    @property
    def id_hex_chars(self) -> int:
        return self.index_hex_chars + self.fingerprint_hex_chars

    def to_dict(self) -> dict[str, int]:
        return {
            "max_occurrences": self.max_occurrences,
            "fingerprint_bits": self.fingerprint_bits,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "OccurrenceIdConfig":
        return cls(
            max_occurrences=int(value["max_occurrences"]),
            fingerprint_bits=int(value["fingerprint_bits"]),
        )


class OccurrenceIdFactory:
    """Mint and verify canonical occurrence identifiers."""

    def __init__(self, config: OccurrenceIdConfig | None = None) -> None:
        self.config = config or OccurrenceIdConfig()

    def mint(self, sequence: int, cmd_signature: str, full_content: str) -> str:
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise TypeError("sequence must be an integer")
        if not 0 <= sequence < self.config.max_occurrences:
            raise ArchiveCapacityExceeded(
                "occurrence sequence exceeds the declared archive capacity"
            )
        if not isinstance(cmd_signature, str) or not isinstance(full_content, str):
            raise TypeError("cmd_signature and full_content must be strings")
        digest = hashlib.blake2s(
            cmd_signature.encode("utf-8") + b"\x00" + full_content.encode("utf-8"),
            digest_size=self.config.fingerprint_bits // 8,
            person=b"onec-arc",
        ).hexdigest()
        return f"{sequence:0{self.config.index_hex_chars}x}{digest}"

    def validate(self, obs_id: str) -> bool:
        return (
            isinstance(obs_id, str)
            and len(obs_id) == self.config.id_hex_chars
            and _HEX_RE.fullmatch(obs_id) is not None
        )

    def sequence_from_id(self, obs_id: str) -> int:
        if not self.validate(obs_id):
            raise ValueError(
                f"obs_id must be exactly {self.config.id_hex_chars} lowercase hex characters"
            )
        return int(obs_id[: self.config.index_hex_chars], 16)

    def verify(self, obs_id: str, cmd_signature: str, full_content: str) -> bool:
        if not self.validate(obs_id):
            return False
        return self.mint(self.sequence_from_id(obs_id), cmd_signature, full_content) == obs_id
