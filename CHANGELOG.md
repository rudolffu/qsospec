# Changelog

## Unreleased

- Add a measurement-first Euclid RGS catalogue workflow with uniform
  one-narrow plus one-broad decompositions for H-alpha, H-beta, Mg II, and
  He I/Pa-gamma, covariance-aware continuous broad fractions, summed-profile
  widths, resumable parts, population exploration, and selected QA.
- Make schema-v5 archive reuse scalable with direct object-key shard loading,
  worker-local stores, deferred manifest reconciliation, authoritative resume,
  timing telemetry, and a read-only benchmark command.
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
