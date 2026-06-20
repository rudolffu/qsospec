# Changelog

## Unreleased

- Add coverage-aware Lyα/N V fitting with red-side continuum anchoring,
  deterministic absorption masking, reliability flags, schema-v3 archival,
  and dedicated QA rendering.

## 0.1.0

- Extract the array-based `neofit` implementation into the standalone
  `qsospec` package.
- Add a modern `src` package layout and bundled iron/Balmer resources.
- Preserve local, global, batch, Parquet archive, QA, and optional pPXF
  workflows.
- Add canonical `WorkflowResult`, `HostWorkflowResult`, and `FitWarning`
  names with deprecated `NeoFit*` aliases.
