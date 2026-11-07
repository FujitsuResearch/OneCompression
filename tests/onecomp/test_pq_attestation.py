# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Quantova Inc

import base64
import os
import tempfile

import pytest

pytest.importorskip("dilithium_py")

from onecomp import pq_attestation as pqa

MODEL = b"\x80\x02}q\x00(X\x06quantqK\x04X\x05layerq\x01K\x10u."


def test_build_and_verify_roundtrip():
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(MODEL, public_key, secret_key)
    assert att["scheme"] == "ML-DSA-65"
    assert att["issuer"] == "Quantova"
    assert pqa.verify_attestation(MODEL, att) is True
    assert pqa.verify_attestation(MODEL, att, public_key) is True


def test_tampered_model_rejected():
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(MODEL, public_key, secret_key)
    assert pqa.verify_attestation(MODEL + b"\x00", att) is False


def test_wrong_signer_rejected_when_pinned():
    public_key, secret_key = pqa.generate_keypair()
    other_public_key, _ = pqa.generate_keypair()
    att = pqa.build_attestation(MODEL, public_key, secret_key)
    assert pqa.verify_attestation(MODEL, att, other_public_key) is False


def test_forged_signature_rejected():
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(MODEL, public_key, secret_key)
    att["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    assert pqa.verify_attestation(MODEL, att) is False


def test_malformed_field_rejected():
    public_key, secret_key = pqa.generate_keypair()
    att = pqa.build_attestation(MODEL, public_key, secret_key)
    att["public_key"] = "!!!not base64!!!"
    assert pqa.verify_attestation(MODEL, att) is False


def test_signer_file_roundtrip():
    public_key, secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "signer.json")
        pqa.save_signer(path, public_key, secret_key)
        loaded_public, loaded_secret = pqa.load_signer(path)
    assert loaded_public == public_key
    assert loaded_secret == secret_key
