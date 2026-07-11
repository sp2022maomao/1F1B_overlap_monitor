# Security Policy

## Supported Versions

Only the latest `main` branch is currently maintained.

## Reporting a Vulnerability

Please open a private security advisory on GitHub when available, or contact the
repository owner directly before publishing details.

This package is a profiling and analysis library. Reports are especially useful
when they include:

- affected version or commit
- command used to reproduce the issue
- operating system and Python version
- whether GPU/runtime integrations were enabled
- whether sensitive traces or cluster identifiers are present in logs

## Data Handling

Trace files can contain hostnames, usernames, rank mappings, paths, and workload
details. Before sharing traces publicly, remove private cluster names, IP
addresses, access tokens, and proprietary model or dataset paths.
