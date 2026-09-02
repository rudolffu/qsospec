# Aydar-inspired masked pseudo-continuum host mode

## Scope and repository state

- Repository: `qsospec`
- Starting commit: `6470abdcd3fe6858b24d6d6701306a699d898bb5`
- Implementation branch: `codex/rgs-lsf-halpha-selection`
- Default host strategy remains `masked_simple`.
- New opt-in strategy: `agn_pseudocontinuum_masked`.
- No external spectra, E-MILES arrays, or validation outputs were committed.

## Implemented workflow

The new strategy performs a nonrecursive direct qsospec Hα/Hβ prefit on the
total spectrum, chooses the nearest width from the Aydar et al. grid, runs
pPXF with separated stellar and fixed-kinematic AGN template components,
subtracts only the stellar model, and then runs the standard qsospec
continuum and line fit. One width update is allowed after the final line fit.

The exact width grid is 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800,
3400, 4000, 4800, 5800, 7000, 8400, 10000, and 11800 km/s. Ties select the
lower width deterministically. Hα has precedence over Hβ. The default failure
policy falls back to `masked_simple`.

The pPXF AGN basis contains:

- 31 analytic `F_lambda` power laws with slopes -3.0 through 0.0 in steps of
  0.1 and a 5100-Å pivot;
- bundled BG92 optical Fe II at the selected width;
- optional bundled UV Fe II when configured and covered; and
- the existing qsospec KD13/Storey-Hummer Balmer continuum and high-order
  Balmer series at the selected width.

The new strategy has additive degree -1 and multiplicative degree 0 by
default. Emission-line masks, observed-frame artifact masks, adaptive mask
expansion, noise rescaling, residual clipping, and object-specific spectral
resolution handling remain active. AGN templates are physically broadened
before pPXF and are assigned a separate fixed-kinematic component, so the
fitted stellar LOSVD is not applied to them.

## Provenance and differences from Aydar et al. (2026)

This is an Aydar-inspired bounded update, not an exact reproduction.
`host_exact_replication` and
`host_pseudocontinuum_exact_replication` are therefore false. The power-law
slope grid and discrete broadening grid reproduce the published values. The
Fe II and Balmer arrays are qsospec's bundled BG92 and KD13/Storey-Hummer
implementations because the paper's exact nuisance-template files are not
distributed. Strong emission lines remain masked instead of being represented
as simultaneous pPXF gas components.

Results retain the pPXF version, E-MILES path and SHA-256 hash, AGN template
sources and hashes, normalization conventions, selected width, width grid,
power-law grid, LSF status, polynomial policy, strategy/fallback state, and
coverage policy. External template data remain outside the repository.

## Measurements and model products

The workflow records:

- broad-prefit line, flux/width S/N, continuous width, and selected width;
- initial/final pseudo-continuum width, bounded iteration status, and timings;
- coverage class and support for Ca H+K, both D4000 bands, G band, Hβ
  absorption, Mg b, and Na D;
- global flux-integrated pPXF AGN fraction and its exact wavelength support;
- the >0.8 AGN-fraction warning without turning it into a reliability veto;
- component weights and normalization/template provenance; and
- closure RMS, median, p95, maximum, relative scale, and status.

`fAGN_5100` retains its historical meaning as a flux-density sample, not a
fraction. The weight-based Aydar fraction is stored as unavailable with status
`exact_definition_not_reproduced`.

Aligned model products retain the stellar model, power law, optical and
optional UV Fe II, Balmer continuum, high-order Balmer series, aggregate AGN,
physical component total, pPXF best fit, closure residual, and host-subtracted
flux. These survive schema-v5 run-store round trips and are also available in
the optional full-grid/host diagnostic products.

## Reliability and QA

Actual valid rest-frame coverage is classified as `full_optical`,
`optical_core`, `blue_optical`, or `insufficient`. Blue-only leverage is
returned with a limitation reason; insufficient coverage and unexplained
model closure prevent a reliable host verdict. The historical `z < 1.2`
request gate remains unchanged.

The pPXF diagnostic plot now shows the pPXF best fit, stellar host, aggregate
AGN pseudo-continuum, separate power-law/Fe II/Balmer branches, masks, final
good pixels, standardized residuals, optional final qsospec model, selected
width, global AGN fraction, coverage, closure, LSF status, and reliability.

## Validation performed

- Complete pytest suite: 305 tests passed. Focused host/provenance tests also
  passed after the final template-cache and support-boundary changes.
- Dedicated new-mode tests cover defaults, validation, exact grids, width
  tie-breaking, Hα/Hβ precedence, fallback, template branch closure, template
  support, all coverage classes, separated pPXF components, non-negative
  weights, global AGN fraction, numerical closure, real local pPXF API use,
  and run-store component round trips.
- Existing host-decomposition tests remain passing.
- Sphinx resolves the new page and generated API pages. A fully warning-free
  online build still depends on external intersphinx inventories being
  reachable; the local restricted-network build reported only unreachable
  inventories plus one pre-existing unlisted Euclid runbook warning.

## Known limitations

- No simultaneous broad/narrow pPXF gas-line system is introduced.
- No stellar masses, ages, metallicities, black-hole masses, or production
  stellar-dispersion selection are produced.
- Native BG92 resolution is not supplied by the bundled source and is recorded
  as unknown/assumed negligible relative to the selected broad widths.
- A banded input resolution matrix is not applied to the AGN templates and is
  reported as unsupported for this path.
- The canonical AIMS-z 16/96-object validation membership was not present in
  an identifiable current manifest, so no substitute sample was run.

The new strategy should remain opt-in until the canonical AIMS-z A/B
validation is available and reviewed.
