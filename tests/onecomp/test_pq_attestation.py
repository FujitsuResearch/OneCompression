# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Quantova Inc

import base64
import json
import os
import stat
import tempfile

import pytest

pytest.importorskip("Crypto.Signature.mldsa")

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


def test_signer_file_is_owner_only():
    public_key, secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "signer.json")
        pqa.save_signer(path, public_key, secret_key)
        mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_public_key_file_roundtrip():
    public_key, _secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trusted.pub")
        pqa.save_public_key(path, public_key)
        loaded = pqa.load_public_key(path)
    assert loaded == public_key


def test_cli_verify_accepts_trusted_key_matching(capsys):
    public_key, secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        model_path = os.path.join(directory, "model.pt")
        with open(model_path, "wb") as handle:
            handle.write(MODEL)
        attestation = pqa.build_attestation(MODEL, public_key, secret_key)
        attestation_path = model_path + ".attestation.json"
        with open(attestation_path, "w", encoding="utf-8") as handle:
            json.dump(attestation, handle)
        trusted_path = os.path.join(directory, "trusted.pub")
        pqa.save_public_key(trusted_path, public_key)

        code = pqa._main(["verify", model_path, attestation_path, trusted_path])
    assert code == 0
    assert "valid" in capsys.readouterr().out


def test_cli_verify_rejects_untrusted_key(capsys):
    """A model and attestation both replaced by an attacker, and signed under
    the attacker's own key, must fail verification once a trusted key the
    attacker does not control is pinned on the command line."""
    public_key, secret_key = pqa.generate_keypair()
    attacker_public_key, attacker_secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        model_path = os.path.join(directory, "model.pt")
        forged_model = MODEL + b"\x00forged"
        with open(model_path, "wb") as handle:
            handle.write(forged_model)
        forged_attestation = pqa.build_attestation(
            forged_model, attacker_public_key, attacker_secret_key
        )
        attestation_path = model_path + ".attestation.json"
        with open(attestation_path, "w", encoding="utf-8") as handle:
            json.dump(forged_attestation, handle)
        trusted_path = os.path.join(directory, "trusted.pub")
        pqa.save_public_key(trusted_path, public_key)

        code = pqa._main(["verify", model_path, attestation_path, trusted_path])
    assert code == 1
    assert "invalid" in capsys.readouterr().out


def test_cli_export_public_key(capsys):
    public_key, secret_key = pqa.generate_keypair()
    with tempfile.TemporaryDirectory() as directory:
        signer_path = os.path.join(directory, "signer.json")
        pqa.save_signer(signer_path, public_key, secret_key)
        out_path = os.path.join(directory, "trusted.pub")

        code = pqa._main(["export-public-key", signer_path, out_path])
        exported = pqa.load_public_key(out_path)
    assert code == 0
    assert exported == public_key
