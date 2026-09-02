# E-MILES versus XSL host comparison

## Scope

This is a bounded stellar-library systematic comparison, not a ranking or a
default-selection exercise. The same immutable 96-row AIMS-z local-coadd DESI
pilot, object identities, redshifts, masks, complete object-specific LSFs,
Galactic correction, Aydar-inspired host strategy, and final qsospec settings
were used. Host decomposition was requested for the 51 objects with `z < 1.2`;
the other 45 retained the normal redshift-gated behavior.

## Completion and reliability

| Result among 51 eligible objects | E-MILES native | XSL native |
|---|---:|---:|
| pPXF completed | 51 | 51 |
| Overall host fit reliable | 30 | 30 |
| Host continuum reliable | 30 | 30 |
| Host fraction reliable | 30 | 30 |
| Median pPXF good pixels | 3590 | 3590 |
| Median coarser-template fraction on good pixels | 1.000 | 0.000 |

The E-MILES coarser-template condition therefore no longer removes valid
pixels or forces a host-continuum reliability failure. Its detailed stellar
kinematics and population statuses retain the resolution caveat.

## Host and AGN fractions

| Median quantity | E-MILES native | XSL native |
|---|---:|---:|
| Host fraction at 4000 Å | 0.4333 | 0.4525 |
| Host fraction at 5100 Å | 0.5542 | 0.5461 |
| Global pPXF AGN fraction | 0.1958 | 0.1228 |

For XSL minus E-MILES, the median 5100 Å host-fraction difference is −0.0084;
the 95th percentile absolute difference is 0.1357 and the maximum is 0.1763.
At 4000 Å the corresponding values are 0.0000, 0.1288, and 0.1544. These are
stellar-library systematics and are not expected to vanish.

The global pPXF AGN fraction has a median difference of 0.0000 but a 95th
percentile absolute difference of 0.2995. Its largest difference (0.9974)
identifies a profile-sensitive object for scientific QA rather than a reason to
reinterpret native/preconvolved XSL as independent fits.

## Absorption-region residuals

Median normalized residual RMS on fitted, non-emission-masked pixels:

| Rest-frame region | E-MILES native | XSL native |
|---|---:|---:|
| Ca H+K | 1.0277 | 1.0285 |
| 4000 Å break region | 1.0291 | 1.0288 |
| G band | 1.0399 | 1.0403 |
| Stellar Hβ region | 1.0942 | 1.0900 |
| Mg b | 1.0217 | 1.0219 |
| Na D | 1.0362 | 1.0361 |

The median pPXF reduced-χ² difference is `+5.14e-5` (XSL minus E-MILES),
with a 95th percentile absolute difference of 0.00707. The bounded pilot does
not establish one SPS family as universally superior.

## Runtime and interpretation

Median pPXF fit time was 1.20 s for E-MILES and 1.78 s for native XSL. Median
complete host-workflow time was 8.01 s and 7.79 s, respectively; the latter is
dominated by work outside the isolated pPXF solve and should not be read as a
general speed ranking. Peak memory was not instrumented in this pass.

The optical-to-NIR HostSED differs between E-MILES and XSL because the source
SSP families and grids differ. This comparison intentionally does not require
stellar weights or HostSEDs to agree. Euclid aperture scaling itself was not
rerun; both native-source HostSED products preserve the existing 1.0/1.6/2.2 μm
input support required by that later step.

## Recommendation

Keep `emiles_native` as the public and adopted default. Expose `xsl_native` as
an opt-in systematic check or higher-resolution stellar-template alternative,
with its own wavelength-coverage and SPS-family limitations. Any future default
change should follow a separate scientific decision using larger validated
samples and targeted QA of the largest profile differences.
