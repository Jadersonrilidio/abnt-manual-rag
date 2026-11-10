# Copilot instructions

Use [AGENTS.md](../AGENTS.md) for the project map, setup commands, and
available ABNT manual skill. Use [README.md](../README.md) for user-facing
installation and usage guidance.

## Working in this repository

- Keep changes focused and follow the existing Python module boundaries:
  document extraction, embeddings, vector storage, index orchestration, and
  answer generation have separate modules under `src/`.
- Preserve source grounding: generated answers must rely on retrieved manual
  passages, cite the supporting excerpts, and state when the context is
  insufficient. Do not present the assistant as authoritative compliance
  review.
- For questions about the manual, prefer the existing `/abnt-manual` skill and
  its search script over duplicating retrieval logic.
- Keep failures visible. Follow the existing explicit exceptions and logging
  patterns; do not silently substitute empty or success-shaped results.
- Avoid committing environment files, credentials, virtual environments, or
  generated local index/cache files. Check `.gitignore` before adding artifacts.
- Update the README when user setup, commands, or expected behavior changes.
- Run the smallest relevant validation. The repository currently has no test
  suite; for Python edits, use the syntax-check command in `AGENTS.md` and
  report tests that could not be run.
