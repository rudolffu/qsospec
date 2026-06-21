Status and roadmap
==================

Current production scope
------------------------

|project_name| currently provides:

- Local and global NumPy/SciPy fitting APIs.
- Global power-law, Fe II, and continuous Balmer pseudo-continuum fitting.
- Coverage-aware UV, optical, and NIR emission recipes.
- Dedicated Lyα absorption masking and reliability flags.
- Optional pPXF host decomposition with an object-level redshift gate.
- Galactic dereddening before file and batch workflows.
- Covariance and Monte Carlo uncertainty summaries.
- Resumable schema-v3 Parquet run bundles and QA regeneration.

Near-term priorities
--------------------

1. Broaden real-spectrum regression validation across surveys and redshift.
2. Benchmark production batch throughput and memory use.
3. Harden warning/reliability conventions for science-catalog filtering.
4. Expand recipe models only after explicit synthetic and real-data validation.
5. Stabilize public APIs and archive compatibility before a post-alpha release.

Compatibility
-------------

Scientific defaults, result fields, warning codes, and archive schemas should
change deliberately and with migration notes. Existing run archives remain
readable; they are not rewritten in place.

Detailed change history is in :doc:`changelog`.
