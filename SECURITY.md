# Security Policy

## Supported Versions

Security updates are provided for the latest release line of Fujitsu One Compression (OneComp).

| Version | Supported          |
| ------- | ------------------ |
| 1.2.2 >= | :white_check_mark: |
| < 1.2.2 | :x:                |

We recommend always upgrading to the latest release before reporting an issue.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues, discussions, or pull requests.**

Instead, report them privately by email to:

**[contact-onecompression@cs.jp.fujitsu.com](mailto:contact-onecompression@cs.jp.fujitsu.com)**

To help us triage and resolve the issue quickly, please include as much of the following as you can:

- The type of issue (e.g., remote code execution, information disclosure, denial of service).
- The affected version(s) and, if applicable, the affected module or component.
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept or exploit code, if available.
- The potential impact of the issue, including how an attacker might exploit it.

## Our Commitment

- We will acknowledge receipt of your report within **5 business days**.
- We will provide an initial assessment and expected timeline as soon as we have triaged the report.
- We will keep you informed of our progress toward a fix and public disclosure.
- We will credit you for the discovery unless you prefer to remain anonymous.

Please make a good-faith effort to avoid privacy violations, data destruction, and service interruption while investigating. We ask that you give us a reasonable amount of time to address the issue before any public disclosure.

Thank you for helping keep OneComp and its users safe.

## Resolved Security Issues

### CVE-2026-73325 — Unsafe Deserialization

An unsafe deserialization vulnerability (CWE-502) was identified in
`QuantizedModelLoader.load_quantized_model_pt()`.

- **Affected versions:** v1.2.0 and earlier
- **Fixed version:** v1.2.1
- **CVE:** CVE-2026-73325
- **Resolution:** The issue was addressed in v1.2.1. Users should upgrade to the latest supported release.

## Security Acknowledgments

We thank the following researchers for responsibly disclosing security issues in OneComp:

- **Nir Yehoshua, Cipher Security Labs** — unsafe deserialization in `QuantizedModelLoader.load_quantized_model_pt()` (CVE-2026-73325, CWE-502), affecting OneComp versions through v1.2.0 and fixed in v1.2.1.
