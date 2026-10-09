Emission-line models
====================

Recipes
-------

Each emission complex is an immutable recipe containing its fit window,
components, line IDs, roles, profile multiplicities, kinematic bounds,
coverage rules, and QA labels. Adaptive Hβ/[O III] and generic recipes use shared variable projection.
Dedicated adapters retain the legacy Hβ/[O III], Mg II, Hα, and Lyα/N V models.

Profiles and roles
------------------

- Broad components represent the summed broad-line profile used for flux,
  centroid, dispersion, numerical FWHM, and EW.
- Narrow components use narrower FWHM and velocity bounds.
- Additional components (legacy ``wing`` keys) describe profile structure.
  Physical outflow classification requires a separate analysis.
- Fixed ratios and shared kinematics are encoded in recipe metadata.

Adaptive [O III] fitting is the default. Added components require convergence,
ΔBIC ≥20 and flux S/N ≥5. Centroid separation and width contrast are diagnostic;
a third component requires matched doublet residual structure and improvement
in both lines. Narrow Hβ is independent of [O III], and a residual linear
continuum is fitted jointly. See :doc:`../how_to/adaptive_oiii` for resolution,
multistart, adequacy and uncertainty definitions.

``HbetaComplexConfig(oiii_profile_mode="legacy")`` uses the tied
Hβ/[O III] model, initialization and rejection rules, including a factor-of-two
width contrast and centroid separation of at least 150 km/s.

Current UV defaults
-------------------

- Lyα: two broad Gaussians; N V: one broad effective 1240.14 Å blend.
- C IV: three broad Gaussians at the unresolved 1549.06 Å blend, with
  velocities from -5000 to +3000 km/s.
- C III]: two broad Gaussians at 1908.73 Å with velocities within
  :math:`\pm2000` km/s.
- No narrow UV components are included by default.

Current optical/NIR defaults
----------------------------

Dedicated recipes fit Mg II, Hβ/[O III], and Hα/[N II]/[S II]. The
optical-blue and Paschen/NIR recipes are component-adaptive after the full
window passes its minimum coverage requirement.

The :doc:`nir_line_coverage` preset (``extended_quasar_v1``) adds compact He I
5877, O I 8449, and [S III] NIR recipes and replaces the umbrella NIR
recipe with four locally covered compact recipes. Local-support recipes
decide coverage from the actual valid pixels of each local window, so a
distant unobserved line cannot disable a covered one. The [Ne III]
label 3868.58 Å is documented as a referencing difference from the
registry's SDSS MaNGA vacuum value 3869.86 Å.

See :doc:`../reference/recipes` for exact windows and components.
