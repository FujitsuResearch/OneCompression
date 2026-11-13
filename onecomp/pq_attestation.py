# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Quantova Inc

"""Optional post quantum attestation of compressed model artifacts.

Signs a saved model file, such as a quantized ``model.pt``, with FIPS 204
ML-DSA so a user can confirm the compressed model is authentic and has not
been altered after it was produced, using a signature scheme that stays safe
once large quantum computers exist. This gives a compressed model a tamper
evident provenance record for a model supply chain.

The attestation is self contained. It carries the public key and the
signature, so a verifier needs only the attestation and the model file.
Integrity is always covered by the file digest and authenticity is covered
when the verifier pins the expected public key.

The scheme is provided by PyCryptodome's ``MLDSA``/``mldsa`` modules, through
the optional ``pq`` extra. Install it with ``pip install onecomp[pq]``.

Command line usage::

    python -m onecomp.pq_attestation keygen signer.json
    python -m onecomp.pq_attestation attest signer.json model.pt
    python -m onecomp.pq_attestation export-public-key signer.json trusted.pub
    python -m onecomp.pq_attestation verify model.pt model.pt.attestation.json [trusted.pub]

The attestation file carries its own public key, so ``verify`` with no
trusted key argument only checks that the model matches a signature made
under *some* key, not that it matches the key you expect. An attacker who
can replace both the model and its attestation can sign the replacement
under their own key and still produce a result of ``valid``. Pass a public
key obtained independently of the attestation, such as the file written by
``export-public-key`` and distributed out of band, to pin the expected
signer and catch that case.
"""

import base64
import hashlib
import json
import os
import sys

SCHEME = "ML-DSA-65"
ISSUER = "Quantova"

_MISSING = (
    "Post quantum attestation needs the pq extra. " "Install it with pip install onecomp[pq]."
)


def _public_key_module():
    """Return Crypto.PublicKey.MLDSA or raise if the pq extra is not installed."""
    try:
        from Crypto.PublicKey import MLDSA
    except ImportError as error:
        raise RuntimeError(_MISSING) from error
    return MLDSA


def _signature_module():
    """Return Crypto.Signature.mldsa or raise if the pq extra is not installed."""
    try:
        from Crypto.Signature import mldsa
    except ImportError as error:
        raise RuntimeError(_MISSING) from error
    return mldsa


def generate_keypair():
    """Return a fresh (public_key, secret_key) pair for the signer, as raw bytes."""
    mldsa_keys = _public_key_module()
    key = mldsa_keys.generate(param_set=mldsa_keys.ML_DSA_65)
    return key.pk, key.sk


def build_attestation(payload, public_key, secret_key):
    """Return a self contained attestation over the payload bytes."""
    mldsa_keys = _public_key_module()
    key = mldsa_keys.construct(param_set=mldsa_keys.ML_DSA_65, sk=secret_key)
    signature = _signature_module().new(key, deterministic=True).sign(payload)
    return {
        "scheme": SCHEME,
        "issuer": ISSUER,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_attestation(payload, attestation, expected_public_key=None):
    """Return True when the attestation matches the payload.

    The file digest is always checked. When expected_public_key is given the
    signer is pinned too, so a valid signature made under a different key is
    rejected.
    """
    if attestation.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        return False
    try:
        public_key = base64.b64decode(attestation["public_key"], validate=True)
        signature = base64.b64decode(attestation["signature"], validate=True)
    except (ValueError, KeyError):
        return False
    if expected_public_key is not None and public_key != expected_public_key:
        return False
    try:
        mldsa_keys = _public_key_module()
        key = mldsa_keys.construct(param_set=mldsa_keys.ML_DSA_65, pk=public_key)
        _signature_module().new(key).verify(payload, signature)
        return True
    except (ValueError, TypeError):
        return False


def _owner_only(path, flags):
    """Opener that creates a file with owner only permissions from the start.

    Setting the mode on open, rather than chmod after the fact, avoids a
    window where the file briefly has the process default permissions, which
    is commonly group or world readable under a 022 umask.
    """
    return os.open(path, flags, 0o600)


def save_signer(path, public_key, secret_key):
    """Write a signer key file that load_signer reads back.

    The file holds the secret key and is created with owner only (0600)
    permissions so other local users cannot read it.
    """
    data = {
        "scheme": SCHEME,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "secret_key": base64.b64encode(secret_key).decode("ascii"),
    }
    with open(path, "w", encoding="utf-8", opener=_owner_only) as handle:
        json.dump(data, handle)


def load_signer(path):
    """Read (public_key, secret_key) from a signer key file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    public_key = base64.b64decode(data["public_key"])
    secret_key = base64.b64decode(data["secret_key"])
    return public_key, secret_key


def save_public_key(path, public_key):
    """Write just the public key, as base64 text, for out of band distribution.

    Share this file through a channel independent of the attestation, such as
    a release page or a signed commit, so a verifier can pin the expected
    signer with the verify CLI's trusted key argument.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(base64.b64encode(public_key).decode("ascii") + "\n")


def load_public_key(path):
    """Read a base64 encoded public key written by save_public_key."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    return base64.b64decode(text)


def _main(argv):
    if len(argv) == 2 and argv[0] == "keygen":
        public_key, secret_key = generate_keypair()
        save_signer(argv[1], public_key, secret_key)
        print(f"wrote {SCHEME} signer to {argv[1]}")
        return 0
    if len(argv) == 3 and argv[0] == "attest":
        public_key, secret_key = load_signer(argv[1])
        with open(argv[2], "rb") as handle:
            payload = handle.read()
        attestation = build_attestation(payload, public_key, secret_key)
        out = argv[2] + ".attestation.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(attestation, handle, indent=2)
        print(f"wrote attestation to {out}")
        return 0
    if len(argv) == 3 and argv[0] == "export-public-key":
        public_key, _secret_key = load_signer(argv[1])
        save_public_key(argv[2], public_key)
        print(f"wrote public key to {argv[2]}")
        return 0
    if len(argv) in (3, 4) and argv[0] == "verify":
        with open(argv[1], "rb") as handle:
            payload = handle.read()
        with open(argv[2], "r", encoding="utf-8") as handle:
            attestation = json.load(handle)
        expected_public_key = load_public_key(argv[3]) if len(argv) == 4 else None
        ok = verify_attestation(payload, attestation, expected_public_key=expected_public_key)
        print("valid" if ok else "invalid")
        return 0 if ok else 1
    print(
        "usage: python -m onecomp.pq_attestation "
        "keygen <key> | attest <key> <file> | export-public-key <key> <out> | "
        "verify <file> <attestation> [trusted_key]"
    )
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
