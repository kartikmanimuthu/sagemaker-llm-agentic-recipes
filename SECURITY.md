# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of our deployment recipes seriously. If you find a security vulnerability, please do NOT open a public issue. Instead, follow these steps:

1.  **Email**: Send a detailed report to the maintainer (replace with your email if you have one, or use GitHub Private Vulnerability Reporting).
2.  **Details**: Please include:
    *   Description of the vulnerability.
    *   Steps to reproduce.
    *   Potential impact.
3.  **Response**: We will acknowledge your report within 48 hours and provide a timeline for a fix.

### Critical Note on AWS Credentials
**NEVER** commit your `ACCESS_KEY`, `SECRET_KEY`, or `HF_TOKEN` to this repository. This project uses environment variables (`AWS_PROFILE`) and dynamic IAM role retrieval to remain secure.
