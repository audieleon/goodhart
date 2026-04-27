# Security Policy

## Scope

Goodhart parses YAML, JSON, and TOML configuration files and executes Python modules via `--check` and `--example`. It does not make network requests, access credentials, or modify files.

## Reporting Vulnerabilities

For non-sensitive bugs, report via GitHub Issues. For vulnerabilities that could be exploited before a fix is available, email audieleon@users.noreply.github.com instead of opening a public issue.

## Known Considerations

- YAML parsing uses `yaml.safe_load` (not `yaml.load`) to prevent code execution.
- Config file size is limited to 1MB to prevent denial-of-service.
- The `--check` flag imports and executes arbitrary Python modules. Only run `--check` on code you trust.
