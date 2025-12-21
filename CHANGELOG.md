# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-12-21

### Fixed
- **Critical Terminology:** Corrected "Superposed Quantum Network Topologies" to "Superpositional Quantum Network Topologies (SQNT)" throughout documentation and citations.
- **Repository Structure:** Updated README to accurately reflect the flat file structure (removed incorrect `src/` references).
- **Packaging:** Added `pyproject.toml` to support standard `pip install -e .` installation.
- **Documentation:** Consolidated protocols into `docs/` and created `docs/sqnt_definition.md` and `REPRODUCE.md`.
- **Hygiene:** Cleaned up `.gitignore` and removed empty boilerplate directories.
- **CI/CD:** Added GitHub Actions workflow (`.github/workflows/ci.yml`) for automated testing and decisive experiment verification.
- **Testing:** Added CI denominator edge-case protection in `InvariantConstrainedCI`.

### Added
- **Reproducibility:** Added `REPRODUCE.md` with explicit one-command reproduction steps.
- **Results:** Added `results/` directory for artifact storage.
- **Analysis:** Exposed step-size envelope and distance-triad correlation analysis via CLI.
