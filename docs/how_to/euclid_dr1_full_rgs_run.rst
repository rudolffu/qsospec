Full Euclid DR1 RGS catalogue runs
==================================

This page describes generic fitting of one portable Euclid RGS sample or shard.
Catalogue membership, VI enrichment, and MLSpecZ source priorities are defined
upstream by MLSpecZ and are not reproduced in ``qsospec``.

Portable sample identity
------------------------

Each input row should carry ``qsospec_object_key``.  The DR1 sample uses
``euclid_dr1_rgs:<signed_object_id>``.  The reader also accepts ``object_key``
and ``spectrum_key`` aliases, but new writers should emit the canonical name.
Explicit keys are non-empty, unique, and path-independent.  Inputs without an
explicit key retain the legacy absolute-path plus row-index key, so existing
gold archives remain readable.

The input ``manifest.json`` is checked before a run.  It locks the sample ID and
scope, input count and SHA-256, Arrow schema, ordered object IDs and keys,
wavelength grid, physical flux/wavelength metadata, shard ID, and shard count.
A transferred or modified Parquet that changes any locked value is rejected.

Scientific configuration
------------------------

``qsospec.euclid_rgs`` exposes the fixed first-pass RGS configuration: no host
decomposition, no Balmer pseudo-continuum, Planck/F99 Galactic correction with
``R_V=3.1``, covariance errors, zero Monte Carlo trials, and every automatically
covered standard complex.  The established 500-bin observed grid is accepted;
the 531-bin source grid is sliced only with ``[11:511]``.  Arbitrary grids are
not resampled.

Run one shard
-------------

.. code-block:: bash

   export SAMPLE_ROOT=/path/to/dr1_identified_full_rgs_v1
   export DUSTMAPS_DATA_DIR=/store/public/databases/dustmaps

   python scripts/run_euclid_rgs_catalog.py \
     --mode production \
     --input "$SAMPLE_ROOT/input/full/shards/shard-000-of-032/spectra.parquet" \
     --sample-manifest "$SAMPLE_ROOT/input/full/shards/shard-000-of-032/manifest.json" \
     --run-directory "$SAMPLE_ROOT/runs/full/shard-000-of-032" \
     --dustmaps-data-dir "$DUSTMAPS_DATA_DIR" \
     --workers 44 --parquet-batch-size 256 --task-size 4

The runner is resumable by default, does not retry failures unless
``--retry-failures`` is explicit, does not create QA or legacy products, and
shows an object-level progress bar. Completed, failed, and resumed/skipped rows
all advance the bar. Use ``--no-progress`` for log-only execution.

``--mode smoke`` selects 64 deterministic redshift-spanning rows but still
validates the full input against its manifest. ``--finalize-only`` rebuilds
run tables from an existing run store without fitting. A disk preflight uses
input size and, when supplied, ``--reference-bytes-per-object`` plus configurable
headroom.

The sample-manifest path/hash and validated identity are recorded in the run
invocation and run manifest. A finalized shard can therefore be inspected or
resumed safely after transfer.

Broad/narrow second pass
------------------------

The existing command retains its gold behavior when no sample manifest is
given. Generic per-shard operation requires a sample manifest, an explicit
versioned product name, and a non-gold output location:

.. code-block:: bash

   python scripts/measure_euclid_dr1_broad_narrow_lines.py \
     --mode production \
     --input "$SHARD_DIR/spectra.parquet" \
     --sample-manifest "$SHARD_DIR/manifest.json" \
     --run-directory "$SAMPLE_ROOT/runs/full/shard-000-of-032" \
     --measurement-directory "$SAMPLE_ROOT/measurements/full/shard-000-of-032/dr1_identified_full_rgs_broad_narrow_r480_v1" \
     --product-name dr1_identified_full_rgs_broad_narrow_r480_v1 \
     --workers 8 --chunk-size 128

Generic mode validates exact IDs, stable keys, count, source-run sample
provenance, and membership order. It does not use the synthetic-test-only
``--allow-noncanonical-gold-count`` escape hatch. Existing part files are
resumed when their fingerprints match, and chunk progress includes reused
parts.

Gold compatibility
------------------

``run_euclid_dr1_gold_rgs.py`` and the 8,530-row gold input remain an immutable
special case. ``qsospec.euclid_gold`` delegates only the shared scientific
configuration to ``qsospec.euclid_rgs`` and retains all gold selection, smoke,
and exact snapshot checks.
