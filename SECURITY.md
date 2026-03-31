# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.4.x   | Yes       |
| < 1.4   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Solitaire, please report it
responsibly. **Do not open a public GitHub issue.**

Email: **security@usedicta.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Any potential impact assessment

We will acknowledge your report within 72 hours.

## Scope

Solitaire is a local-first application. All data is stored on the user's
machine in SQLite databases and JSONL files. There is no remote server,
no cloud storage, and no network-accessible API.

The primary attack surface is:
- **Local file access**: rolodex.db, JSONL audit trails, persona configs
- **CLI input handling**: JSON payloads via stdin, command arguments
- **Optional network calls**: GitHub API for update checks (read-only, non-critical)

Solitaire does not handle authentication, user accounts, or network services.
Vulnerabilities related to local privilege escalation, data corruption, or
CLI input injection are in scope.

## Security Practices

- No `eval()`, `exec()`, or `pickle` in production code
- Parameterized SQL queries throughout (no string interpolation)
- Subprocess calls use list format (no shell injection)
- All file I/O uses explicit UTF-8 encoding
- No hardcoded secrets or API keys
- Optional dependencies use lazy imports with graceful fallback
