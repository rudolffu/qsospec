Status and roadmap
==================

Supported features
------------------

|project_name| currently provides:

- Local and global NumPy/SciPy fitting APIs.
- Global power-law, Fe II, and continuous Balmer pseudo-continuum fitting.
- Coverage-aware UV, optical, and NIR emission recipes.
- Dedicated Lyα absorption masking and reliability flags.
- Optional pPXF host decomposition with an redshift requirement.
- Galactic dereddening before file and batch workflows.
- Covariance and Monte Carlo uncertainty summaries.
- Resumable schema-v5 Parquet run bundles and QA regeneration.

Near-term priorities
--------------------

1. Broaden real-spectrum regression validation across surveys and redshift.
2. Benchmark production batch throughput and memory use.
3. Clarify warning and reliability conventions for science-catalog filtering.
4. Expand recipe models with synthetic and real-data validation.
5. Stabilize public APIs and archive compatibility before a post-alpha release.

Compatibility
-------------

Changes to scientific defaults, result fields, warning codes, and archive
schemas should include migration notes. Unsupported pre-release schemas
require recreating the run.

Detailed change history is in :doc:`changelog`.
