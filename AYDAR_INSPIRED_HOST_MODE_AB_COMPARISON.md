# Aydar-inspired host-mode A/B validation status

## Status

The requested canonical 16-object smoke and 96-object AIMS-z local-coadd pilot
were not run. No manifest in the current qsospec checkout or the inspected
nearby astronomy workspace identified the exact historical 16/96 membership
with validated object-specific AIMS-z/DESI LSF provenance.

The checkout contains unrelated or noncanonical runs, including a 56-object
DESI/Euclid analysis and older DESI parquet runs. They were not substituted,
because doing so would violate the fixed-membership and LSF stop gates.

## Completed bounded validation

- Synthetic tests compare physical component closure and global AGN fractions.
- A real installed pPXF/E-MILES smoke test exercises the new separated-component
  API with the local `spectra_emiles_9.0.npz` bundle.
- The complete unit/integration suite passes.

## Required input before the A/B run

Provide or identify one immutable manifest containing:

1. the canonical 96 AIMS-z object keys and the deterministic 16-object subset;
2. local-coadd spectrum locations and reader information;
3. validated object-specific wavelength-dependent LSF provenance;
4. identities, redshifts, and physical flux units; and
5. the baseline run/configuration to compare against.

Once available, run `masked_simple` and `agn_pseudocontinuum_masked` into
separate new run directories. The 16-object stop gates must pass before the
96-object pilot. No historical run should be overwritten.

## Planned comparison fields

The implementation exposes the required inputs for a future per-object A/B
table: host fractions at 4000/5100 Å, global AGN fraction, stellar kinematics,
pPXF and final-continuum fit statistics, clean fraction, coverage, closure,
host-subtracted continuum samples, Hα/Hβ broad widths and fluxes, fallback,
residual-region summaries, and stage runtimes.

