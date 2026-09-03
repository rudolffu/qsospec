# Repository cleanup audit

Audit date: 2026-09-03. Cleanup branch: `codex/cleanup-repository-hygiene`.

## Starting state

- Target branch and commit: `codex/rgs-lsf-halpha-selection` at `96c534335b8a92ea96f5a0068ded5b9a22c82ce4`.
- Working tree: clean before the cleanup branch was created; no submodules.
- Python: 3.12.11 from the configured conda base environment.
- Test collection: 331 tests. Baseline full suite: 331 passed with two numerical warnings in 60.64 s.
- Baseline lint: failed under a user-wide Ruff rule selection not declared by this repository. This change adds a repository-local correctness policy; broader formatting remains a separate mechanical cleanup.
- Baseline build: isolated build could not download build requirements in the network-restricted audit environment. The no-isolation build initially needed write access to the existing egg-info directory and is rechecked after cleanup.
- Sphinx documentation exists and is added to CI.

## Inventory and decisions

- No empty implementation stubs or dummy tests were found.
- `qsospec.io.run_store` is the canonical owner of run schemas, fingerprints, immutable staged payloads, reconciliation, and atomic finalization. `qsospec.workflows.batch` is the canonical owner of scalar resume planning and process execution. Existing invariant suites cover complete, partial, failed, duplicate-key, mismatch, deterministic-order, and schema behavior; these contracts were retained unchanged.
- `build_euclid_dr1_gold_rgs_input.py` implements the fixed identified-gold membership policy. `build_euclid_dr1_qsospec_input.py` is a generic external-survey bridge. They overlap in output shape but not sample policy, so neither is deprecated or merged.
- Host internals are already separated across `workflows/host/io.py`, `preprocess.py`, `templates.py`, `ppxf_host.py`, `results.py`, `serialization.py`, `plots.py`, and `euclid.py`. Moving namespaces again without numerical fixtures would add compatibility risk.
- The top-level API contains specialist re-exports, but documented callers and compatibility tests exist. No export was removed merely to reduce surface area.
- Legacy host vocabulary aliases have explicit deprecation tests; their behavior remains unchanged.
- Workload benchmarks moved to `benchmarks/` and `docs/benchmarks/`; they are not collected by the default test suite.

## Portability and optional dependencies

- The brittle literal-version assertion now compares against installed package metadata.
- pPXF plus local E-MILES tests are marked `integration`, `external_data`, and `slow`; synthetic host tests remain in the host CI job.
- Plotting and executable-documentation tests have explicit markers.
- Sphinx imports the installed package rather than modifying `sys.path`.
- Active Euclid scripts no longer default to `/Users/yuming/...`; optional defaults come from `DUSTMAPS_DATA_DIR` and `EUCLID_COMPOSITE_ROOT`.

## Compatibility-sensitive items deliberately unchanged

- Scientific line definitions, H-alpha thresholds, kinematic ties, signed-line evidence, masks, LSF handling, units, frame conventions, host fractions, and no-extrapolation behavior.
- Run-store schema version 5, object keys, manifests, fingerprints, staging layout, and resume/retry semantics.
- Host model organization and optional-template data boundary.
- Generic luminosity and DESI-to-Euclid transfer ownership were not changed: cross-repository numerical consolidation requires fixed real/synthetic equivalence fixtures and a separately versioned minimum QSOSpec dependency in MLSpecZ.

## Deferred work

- Large Euclid classifier/measurement scripts still contain reusable orchestration. Extraction is deferred until each CLI has row-count, column-order, identifier, NaN-mask, and fingerprint fixtures.
- Specialist top-level exports need a release-version deprecation schedule before removal.
- External template integration remains manual because licensed/local template assets are not provisioned in CI.
