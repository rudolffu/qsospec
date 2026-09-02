# Native-XSL runtime versus exact-preconvolved XSL parity

## Result

Native XSL runtime convolution and exact object/grid/LSF-specific XSL caches are
computationally equivalent in the bounded validation.

The first 16 objects were used as the stop gate. The comparison was then
repeated over the immutable 96-row AIMS-z local-coadd pilot. All 96 final
records and all 51 redshift-eligible host decompositions completed. Across the
96-object comparison, the maximum absolute differences were exactly zero for:

- prepared wavelength, flux, and uncertainty arrays;
- pPXF/final continuum masks and complex statuses;
- fit-time stellar matrix behavior and retained pixels;
- stellar weights and per-template normalization scales;
- stellar host and AGN component models;
- pPXF reduced χ² and global AGN fraction;
- host fractions and host-subtracted native spectrum;
- final continuum and emission-line parameters;
- native-source template-weighted HostSED;
- split host reliability outcomes.

This exact equality is stronger than the configured numerical tolerance and
confirms that the cache is a derived computational product, not an independent
stellar-library result.

## Provenance and rejection checks

Each accepted product was tied to the native XSL SHA-256
`b8bee334ee1705a0d4e7ff0f161990c248d68ac1b8c8eb138b0d8c05cf37d853`,
the explicit flattened 208-template ordering, target object key, redshift,
3600–7000 Å host interval, exact pPXF log grid, rest-frame target LSF vector,
and valid-resolution mask. Unit tests confirm rejection after changing the
object identity, LSF, grid, fit range, source hash, or template order.

Only the 51 objects that passed the existing `z < 1.2` host gate received a
derived cache in the 96-row run. The other records still exercised identical
final fitting and host-skip behavior without creating unused products.

## Timing

For the 51 host fits:

- native-XSL runtime convolution: median 0.0887 s, 95th percentile 0.1464 s;
- native-XSL complete stellar-template preparation: median 0.1805 s;
- exact-cache read: median 0.1165 s;
- exact-cache validation: median 0.0010 s;
- exact-cache fit-time template preparation after loading: median 0.0143 s.

The external 96-row exact-cache validation (cache construction plus 96 fits,
four workers) took 181.8 s. Cache construction cost is intentionally explicit
and outside normal E-MILES use. Exact caches are useful when they are reused;
they are not generated automatically and there is no universal DESI-convolved
XSL file.

## HostSED invariant

Both paths reconstruct the HostSED from the native XSL `source_wave` and
`source_flux` arrays. The exact fit product contains no duplicate native SSP
library. Equality of all 51 reconstructed HostSEDs verifies that no
data-LSF-convolved matrix leaked into intrinsic/Euclid-transfer SED prediction.
